"""LOD1-conditioned autoregressive mesh transformer.

Follows MeshAnything: Artist-Created Mesh Generation with Autoregressive
Transformers (Chen et al., arXiv:2406.10163), https://github.com/buaacyw/MeshAnything

Adaptations for this dataset:
  * The paper conditions on a point cloud, encoded by a frozen pretrained shape
    encoder and projected into the mesh-token space. Here the condition is the
    building's LOD1 mesh, which is already a mesh, so it goes through the *same*
    tokenizer and the *same* embedding table as the target and is prepended as a
    prefix. Prefix rather than cross-attention: it is what the paper does with
    its shape tokens (concatenated in front of the mesh tokens), it needs no
    second attention stack or encoder tower, and conditioning and target live in
    one coordinate frame so a shared vocabulary is meaningful. Cross-attention
    would be the move once the condition stops being a mesh.
  * No VQ-VAE. Tokens are discretized coordinates straight from
    `src.dataset.mesh_dataset`, so there is no codebook to pretrain and the
    detokenizer is exact. Sequence length is the price.

TODO(next): evaluate as alternatives to this autoregressive formulation
  * PartCrafter: Structured 3D Mesh Generation via Compositional Latent
    Diffusion Transformers (Lin et al., 2025), https://arxiv.org/abs/2506.05573
    -- per-part latent token sets could map onto ground/roof/wall surfaces.
  * MeshFlow: Efficient Artistic Mesh Generation via MeshVAE and Flow-based
    Diffusion Transformer (Li et al., 2026), https://arxiv.org/abs/2606.04621
    -- drops next-token prediction for a flow-based DiT; removes the quadratic
    sequence-length wall this model runs into on dense LOD2 roofs.
"""
import logging

import lightning as L
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from src.dataset.mesh_dataset import (
    NUM_BINS,
    amt_detokenize,
    break_token,
    detokenize,
    specials,
    vocab_size,
)
from src.eval.mesh_metrics import chamfer_distance, surface_distances

logger = logging.getLogger(__name__)


def invalid_logits_mask(tgt, vocab, num_bins, tokenization="coord", pad=None):
    """Which next tokens are structurally illegal, given the sequence so far.

    "Masking Invalid Predictions", MeshAnything V2 section 3.2, inherited from
    PolyGen (Nash et al., 2020). A decoder-only model can end a sequence in the
    middle of a face or emit two breaks in a row; nothing in the loss forbids
    it, and the result decodes to a mesh missing its last triangle. Masking the
    logits at sampling time makes those states unreachable instead of merely
    unlikely.

    Rules, all keyed off how far into the current face/strip the sequence is:
      * BOS and PAD are never legal after position 0 -- they are not mesh
        content, and PAD in particular would be read back as padding.
      * On the coordinate path a face is 9 tokens, so EOS is legal only on a
        multiple of 9.
      * Under AMT a vertex is 3 tokens and a strip needs 3 vertices before it
        describes a face, so BREAK and EOS are legal only on a vertex boundary
        with at least 3 vertices standing -- the paper's "at least three
        vertices must be generated before allowing any interruptions", which
        also rules out a break straight after a break.

    Coordinate tokens are never masked, so the mask can never block the whole
    vocabulary and deadlock sampling.

    Args:
        tgt: [B, L] tokens generated so far, including the leading BOS.
        vocab (int): width of the logits, from `vocab_size`.
        num_bins (int): coordinate discretization.
        tokenization (str): ``"coord"`` or ``"amt"``.
        pad (int, optional): PAD id. Defaults to the coordinate layout's.

    Returns:
        Tensor: [B, vocab] bool, True where sampling that token is illegal.
    """
    bos, eos, pad_id = specials(num_bins)
    pad = pad_id if pad is None else pad
    brk = break_token(num_bins)

    device = tgt.device
    b = tgt.shape[0]
    mask = torch.zeros((b, vocab), dtype=torch.bool, device=device)

    # Never re-open the sequence, never emit padding as content.
    for tok in (bos, pad):
        if tok < vocab:
            mask[:, tok] = True

    body = tgt[:, 1:]                      # drop the leading BOS
    is_coord = (body >= 0) & (body < num_bins)

    if body.shape[1] == 0:
        # Nothing emitted yet: no face stands, so only coordinates are legal.
        for tok in (eos, brk):
            if tok < vocab:
                mask[:, tok] = True
        return mask

    if tokenization == "amt":
        # Coordinates since the last break: a break resets the strip, so only
        # the tail after the final break counts toward the 3-vertex rule.
        is_break = body == brk
        # Index of the most recent break per row, -1 when there is none.
        idx = torch.arange(body.shape[1], device=device)[None].expand_as(body)
        last_break = torch.where(is_break, idx, torch.full_like(idx, -1)).max(dim=1).values
        after = (idx > last_break[:, None]) & is_coord
        n = after.sum(dim=1)
        closed = (n % 3 == 0) & (n >= 9)   # whole vertices, and a face standing
        if brk < vocab:
            mask[:, brk] = ~closed
        if eos < vocab:
            mask[:, eos] = ~closed
    else:
        n = is_coord.sum(dim=1)
        # `n >= 9` as well as the boundary: every LOD1 condition has exactly one
        # correct answer and it always has at least one face, so an immediate
        # EOS is never right. Mirrors AMT's three-vertex rule below.
        if eos < vocab:
            mask[:, eos] = ((n % 9) != 0) | (n < 9)

    return mask


class MeshEmbedding(nn.Module):
    """How a mesh sequence becomes vectors, independent of what consumes them.

    Shared by the from-scratch stack and the OPT-backed one so there is a single
    implementation of the part that is easy to get subtly wrong: a code id is
    embedded from the frozen codebook, the specials come from a small free
    table, and every token carries both its slot within a face and which half of
    the sequence it belongs to.

    Args:
        vocab_size (int): coordinate bins, or codebook entries, plus BOS/EOS/PAD.
        d_model (int): width of the vectors handed to the backbone.
        cond_dim (int, optional): width of the continuous LOD1 features under
            the VQ-VAE tokenizer. None keeps the coordinate path, where the
            condition is token ids.
        codebook (Tensor, optional): ``[codebook_size, codebook_dim]`` snapshot.
        tokens_per_face (int, optional): 3 vertices x depth, for the face slot.
    """

    def __init__(self, vocab_size, d_model, cond_dim=None, codebook=None,
                 tokens_per_face=None):
        super().__init__()
        # Under the VQ-VAE tokenizer the condition arrives as continuous
        # per-face features rather than token ids, so it needs a projection
        # instead of a lookup. None keeps the coordinate path exactly as it was.
        self.cond_proj = nn.Linear(cond_dim, d_model) if cond_dim else None

        self.codebook_size = 0 if codebook is None else codebook.shape[0]
        self.tokens_per_face = tokens_per_face
        if codebook is None:
            self.token_embed = nn.Embedding(vocab_size, d_model)
        else:
            # A code id is embedded by looking its vector up in the frozen
            # codebook and projecting it, as in MeshAnything's `embed_with_vae`
            # (whose `embed_tokens` is commented "# not used"). Two codes that
            # are close in codebook space then arrive as nearly the same vector
            # instead of two unrelated rows the model has to relate from data.
            #
            # A buffer, not a live reference to the VQ-VAE: registering that
            # module twice would duplicate stage 1 in the checkpoint. Stage 2
            # freezes the codebook, so a snapshot stays exact -- and it travels
            # with the checkpoint, which a reference would not.
            self.token_embed = None
            self.register_buffer("codebook", codebook.detach().clone())
            self.code_proj = nn.Linear(codebook.shape[1], d_model)
            # BOS/EOS/PAD have no codebook vector, so they keep a free table.
            self.extra_embed = nn.Embedding(vocab_size - self.codebook_size, d_model)
            # Which of the `tokens_per_face` slots inside a face this token
            # fills -- which vertex, which residual stage -- plus one slot for
            # each special. MeshAnything's `OPTFacePositionalEmbedding`, sized
            # `face_per_token + 3` with the specials at 0..2.
            self.face_pos_embed = nn.Embedding(tokens_per_face + 3, d_model)

        # Which half of the sequence a position belongs to. Without it the model
        # has to infer the LOD1/LOD2 boundary from the BOS token alone.
        self.segment_embed = nn.Embedding(2, d_model)

    def embed_cond(self, cond):
        """The LOD1 prefix: a projection of face features, or a token lookup.

        The two halves are embedded separately because under the VQ-VAE
        tokenizer they no longer share a vocabulary: the condition is a
        projected face feature, the target a code id.
        """
        return self.cond_proj(cond) if cond.is_floating_point() else self.embed_tokens(cond)

    def embed_tokens(self, ids):
        """[B, L] ids to [B, L, d], where ``ids[:, 0]`` is the segment's first token.

        On the coordinate path this is a plain lookup. On the code path a code
        is `code_proj(codebook[id])`, the specials come from `extra_embed`, and
        every token additionally carries its slot within the face.
        """
        if self.token_embed is not None:
            return self.token_embed(ids)

        is_code = ids < self.codebook_size
        codes = self.code_proj(self.codebook[ids.clamp(max=self.codebook_size - 1)])
        special = (ids - self.codebook_size).clamp(min=0)
        out = torch.where(is_code[..., None], codes, self.extra_embed(special))

        # The target segment is [BOS, code 0, code 1, ...], so the token at
        # index i is code i-1 and fills slot (i-1) % tokens_per_face. Specials
        # take slots 0..2, mirroring the reference's `% face_per_token + 3`.
        index = torch.arange(ids.shape[1], device=ids.device)
        slot = 3 + (index - 1) % self.tokens_per_face
        slot = torch.where(is_code, slot[None].expand_as(ids), special)
        return out + self.face_pos_embed(slot)


class MeshTransformer(MeshEmbedding):
    """Decoder-only transformer over ``[LOD1 tokens] + [BOS, LOD2 tokens, EOS]``.

    Trained from scratch. `MeshOPTTransformer` is the alternative backbone.

    Args:
        vocab_size (int): coordinate bins plus BOS/EOS/PAD.
        d_model (int): channel width; must be divisible by n_head.
        max_seq_len (int): capacity of the learned positional embedding.
            Positions restart per segment, so it must cover the longer of
            ``len(cond)`` and ``len(tgt) - 1``, not their sum;
            `MeshDataset.max_seq_len` reports what the data needs.
    """

    def __init__(self, vocab_size, d_model=256, n_head=8, num_layers=6,
                 dropout=0.1, max_seq_len=4096, cond_dim=None,
                 codebook=None, tokens_per_face=None):
        super().__init__(vocab_size, d_model, cond_dim, codebook, tokens_per_face)
        self.max_seq_len = max_seq_len
        self.pos_embed = nn.Embedding(max_seq_len, d_model)
        self.drop = nn.Dropout(dropout)

        layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=n_head, dim_feedforward=4 * d_model,
            dropout=dropout, activation="gelu", batch_first=True, norm_first=True,
        )
        self.blocks = nn.TransformerEncoder(layer, num_layers=num_layers)
        self.norm = nn.LayerNorm(d_model)
        self.head = nn.Linear(d_model, vocab_size)

    def forward(self, cond, tgt, cond_pad_mask=None, tgt_pad_mask=None):
        """Next-token logits for the target sequence.

        Args:
            cond: [B, Lc] LOD1 token ids, or [B, Fc, cond_dim] continuous
                per-face features from a VQ-VAE encoder.
            tgt: [B, Lt] target tokens, BOS-led and EOS-terminated.
            cond_pad_mask, tgt_pad_mask: [B, L] bool, True at padding.

        Returns:
            Tensor: [B, Lt - 1, vocab] where position ``i`` predicts ``tgt[:, i+1]``.
        """
        n_cond, n_tgt = cond.shape[1], tgt.shape[1]
        length = n_cond + n_tgt - 1
        if max(n_cond, n_tgt - 1) > self.max_seq_len:
            raise ValueError(
                f"Segment of {max(n_cond, n_tgt - 1)} exceeds "
                f"max_seq_len={self.max_seq_len}. "
                "Raise model.max_seq_len or lower data.max_faces."
            )

        # Positions restart at 0 in the target segment. Counting straight through
        # would make every target position depend on the longest *condition* in
        # the batch -- the same building would land on different position
        # embeddings from batch to batch, and on yet another set at sampling
        # time, where the condition is unpadded. The segment embedding is what
        # tells the two halves apart, so the indices need not.
        device = cond.device
        pos = torch.cat([torch.arange(n_cond, device=device),
                         torch.arange(n_tgt - 1, device=device)])
        segment = (torch.arange(length, device=device) >= n_cond).long()

        x = torch.cat([self.embed_cond(cond), self.embed_tokens(tgt[:, :-1])], dim=1)
        h = self.drop(x + self.pos_embed(pos) + self.segment_embed(segment))

        # Bool, not the float mask from generate_square_subsequent_mask: torch
        # deprecates mixing mask dtypes with a bool src_key_padding_mask.
        causal = torch.ones(length, length, dtype=torch.bool, device=device).triu(1)
        pad_mask = None
        if cond_pad_mask is not None and tgt_pad_mask is not None:
            pad_mask = torch.cat([cond_pad_mask, tgt_pad_mask[:, :-1]], dim=1)

        h = self.blocks(h, mask=causal, src_key_padding_mask=pad_mask, is_causal=True)
        return self.head(self.norm(h[:, n_cond:]))


class MeshOPTTransformer(MeshEmbedding):
    """The same sequence, over an OPT backbone (MeshAnything's `ShapeOPT`).

    The paper adopts OPT-350M for stage 2. This reaches that a different way
    than the reference does, deliberately:

      * The reference subclasses `OPTDecoder` to inject its embeddings, pinned
        to `transformers==4.39.3`. Subclassing library internals across a major
        version is a silent-breakage machine, and this project is on 5.x. Here
        every embedding is built in `MeshEmbedding` and handed to OPT as
        `inputs_embeds`, so no HF internal is touched.
      * Positions therefore come from OPT and count straight through the
        condition into the target, rather than restarting per segment. That is
        what the reference does too, and what its fixed 257-token condition
        makes safe. Ours is variable-length, so the segment embedding is doing
        more work here than it does there.

    Note the reference builds this with `from_config` -- random weights, then
    its own checkpoint. Whether the published model was warm-started from OPT's
    language weights is in the paper text only, not in the released code, which
    is why `pretrained` is a switch rather than an assumption.

    Args:
        vocab_size (int): codebook entries plus BOS/EOS/PAD.
        opt_name (str): HF model id, e.g. ``facebook/opt-350m``.
        pretrained (bool): load OPT's weights, or only its architecture.
        opt_config (OPTConfig, optional): bypass the hub entirely. Tests pass a
            tiny config here; nothing else should need it.
    """

    def __init__(self, vocab_size, opt_name="facebook/opt-350m", pretrained=False,
                 cond_dim=None, codebook=None, tokens_per_face=None,
                 dropout=None, opt_config=None):
        from transformers import AutoConfig, OPTConfig, OPTForCausalLM

        if opt_config is not None:
            opt = OPTForCausalLM(opt_config)
        elif pretrained:
            opt = OPTForCausalLM.from_pretrained(opt_name)
        else:
            opt = OPTForCausalLM(AutoConfig.from_pretrained(opt_name))
        cfg = opt.config
        if dropout is not None:
            cfg.dropout = dropout

        # Embeddings are `word_embed_proj_dim` wide -- OPT projects that up to
        # hidden_size internally, which is why the two differ on 350m (512/1024).
        super().__init__(vocab_size, cfg.word_embed_proj_dim, cond_dim, codebook,
                         tokens_per_face)

        # Drops OPT's 50k language vocabulary, which is dead weight here: input
        # comes from `inputs_embeds`, so the table survives only as the tied
        # output projection, and that only ever needs `vocab_size` rows. Worth
        # ~51M parameters on opt-350m.
        opt.resize_token_embeddings(vocab_size)
        self.opt = opt
        # The combined budget, not a per-segment one: OPT's positions run
        # through both halves. `forward` checks against it.
        self.max_seq_len = cfg.max_position_embeddings

    def forward(self, cond, tgt, cond_pad_mask=None, tgt_pad_mask=None):
        """Next-token logits for the target sequence. Same contract as
        `MeshTransformer.forward`, so nothing downstream can tell them apart."""
        n_cond, n_tgt = cond.shape[1], tgt.shape[1]
        length = n_cond + n_tgt - 1
        if length > self.max_seq_len:
            raise ValueError(
                f"Condition ({n_cond}) plus target ({n_tgt - 1}) is {length}, past "
                f"OPT's max_position_embeddings={self.max_seq_len}. Unlike the "
                "from-scratch backbone this budget is shared, because positions "
                "run through both halves. Lower mesh_data.max_faces."
            )

        x = torch.cat([self.embed_cond(cond), self.embed_tokens(tgt[:, :-1])], dim=1)
        segment = (torch.arange(length, device=x.device) >= n_cond).long()
        x = x + self.segment_embed(segment)

        # 1 at real tokens: HF's convention is the opposite of torch's
        # key_padding_mask, so the masks are inverted rather than passed through.
        if cond_pad_mask is not None and tgt_pad_mask is not None:
            attn = (~torch.cat([cond_pad_mask, tgt_pad_mask[:, :-1]], dim=1)).long()
        else:
            attn = torch.ones(x.shape[:2], dtype=torch.long, device=x.device)

        out = self.opt(inputs_embeds=x, attention_mask=attn)
        return out.logits[:, n_cond:]


class MeshTransformerModule(L.LightningModule):
    """Lightning wrapper: next-token cross-entropy over the LOD2 token sequence."""

    def __init__(self, num_bins=NUM_BINS, d_model=256, n_head=8, num_layers=6,
                 dropout=0.1, max_seq_len=4096, lr=1e-4,
                 lr_scheduler="none", lr_decay_steps=50, lr_decay_rate=0.5,
                 vqvae=None, backbone="scratch", opt_name="facebook/opt-350m",
                 opt_pretrained=True, opt_config=None,
                 tokenization="coord", mask_invalid=False):
        """
        Args:
            num_bins (int): coordinate discretization used by the tokenizer.
                Must match the datamodule's, or the vocabulary is misaligned;
                saved as a hyperparameter so inference restores it.
            lr_scheduler (str): 'none' | 'cosine' | 'step', as in
                `CityJSONDiffusionModule`.
            vqvae (MeshVQVAE, optional): a trained stage-1 tokenizer. Given one,
                the model predicts *codes* rather than coordinates and the LOD1
                condition is encoded (never quantized -- see the design spec).
                None keeps the coordinate tokenizer, which stays the default.
        """
        super().__init__()
        # The tokenizer is a module, not a hyperparameter: pickling it into the
        # checkpoint would store a second copy of stage 1 in every stage-2 file.
        # `opt_config` likewise -- it is a test hook, and a checkpoint that
        # carried one would rebuild the tiny stand-in instead of the real model.
        self.save_hyperparameters(ignore=["vqvae", "opt_config"])

        self.num_bins = num_bins
        self.lr = lr
        self.lr_scheduler = lr_scheduler
        self.lr_decay_steps = lr_decay_steps
        self.lr_decay_rate = lr_decay_rate

        # Adjacent Mesh Tokenization (MeshAnything V2, arXiv:2408.02555). Only
        # meaningful on the coordinate path: under the VQ-VAE the sequence is
        # codes, and V2 drops the VQ-VAE precisely so AMT can operate on
        # coordinates. Rejected rather than ignored, because a config naming
        # both would otherwise silently get neither.
        if tokenization not in ("coord", "amt"):
            raise ValueError(f"Unknown tokenization: {tokenization!r}. "
                             "Expected 'coord' or 'amt'.")
        if tokenization == "amt" and vqvae is not None:
            raise ValueError(
                "tokenization='amt' cannot be combined with a VQ-VAE tokenizer: "
                "AMT rewrites the *coordinate* sequence, while the VQ-VAE replaces "
                "it with codes. MeshAnything V2 drops the VQ-VAE for this reason "
                "(arXiv:2408.02555 section 3.2). Set mesh_model.tokenizer: coord."
            )
        self.tokenization = tokenization
        self.mask_invalid = mask_invalid

        self.vqvae = vqvae
        if vqvae is None:
            self.bos, self.eos, self.pad = specials(num_bins)
            vocab, cond_dim = vocab_size(num_bins, tokenization), None
        else:
            # Frozen: stage 2 must not move the vocabulary underneath the codes
            # it is learning to predict, and a drifting codebook would make the
            # target distribution non-stationary.
            vqvae.eval()
            for p in vqvae.parameters():
                p.requires_grad_(False)
            # One id space of `codebook_size`, as in MeshAnything
            # (`vocab_size = self.tokenizer.codebook_size + 3`). The residual
            # stage is carried by *position* -- it is the fastest-varying index
            # of the [F, 3, depth] layout, and `lookup` reads it off that axis --
            # so a per-stage id offset only tripled this softmax.
            base = vqvae.codebook_size
            self.bos, self.eos, self.pad = base, base + 1, base + 2
            vocab, cond_dim = base + 3, vqvae.head.in_features

        # Shared across residual stages, so one book is the whole vocabulary.
        codebook = None if vqvae is None else vqvae.quantizer.codebooks[0]
        per_face = None if vqvae is None else vqvae.tokens_per_face

        if backbone == "scratch":
            self.network = MeshTransformer(
                vocab_size=vocab, d_model=d_model, n_head=n_head,
                num_layers=num_layers, dropout=dropout, max_seq_len=max_seq_len,
                cond_dim=cond_dim, codebook=codebook, tokens_per_face=per_face,
            )
        elif backbone == "opt":
            # d_model / n_head / num_layers / max_seq_len come from the OPT
            # config instead; leaving them in the signature keeps one set of
            # hyperparameters for both backbones rather than two.
            self.network = MeshOPTTransformer(
                vocab_size=vocab, opt_name=opt_name, pretrained=opt_pretrained,
                cond_dim=cond_dim, codebook=codebook, tokens_per_face=per_face,
                dropout=dropout, opt_config=opt_config,
            )
        else:
            raise ValueError(f"Unknown backbone: {backbone!r}. Expected "
                             "'scratch' or 'opt'.")

    def train(self, mode=True):
        """Keep the frozen tokenizer in eval, whatever Lightning does.

        `nn.Module.train` recurses into submodules, and Lightning calls
        `model.train()` at the start of every training epoch -- which re-enables
        the VQ-VAE's dropout. `_prepare` runs that tokenizer inside
        `training_step`, so the effect is that the *target labels* are resampled
        each epoch (~8% of code ids move at dropout 0.1) while validation, run
        under `eval()`, scores against clean ones. Two different label
        distributions, and a train/val loss gap that is partly an artifact.

        The `requires_grad` guard in `MeshVQVAE.quantize` covers the codebook
        EMA; this covers module mode, which is a different question.
        """
        super().train(mode)
        if self.vqvae is not None:
            self.vqvae.eval()
        return self

    # ------------------------------------------------------------------
    # Batch -> this model's vocabulary
    # ------------------------------------------------------------------

    def _prepare(self, batch):
        """``(cond, tgt, cond_pad_mask, tgt_pad_mask)`` in the active vocabulary.

        `MeshDataset` always emits coordinate tokens -- stage 2 changed nothing
        about it -- so under the VQ-VAE the conversion happens here, on device,
        under no_grad. Returns the batch untouched on the coordinate path.
        """
        if self.vqvae is None:
            return (batch["cond"], batch["tgt"],
                    batch.get("cond_pad_mask"), batch.get("tgt_pad_mask"))

        with torch.no_grad():
            cond_coords, cond_faces = self._faces(batch["cond"], batch.get("cond_pad_mask"))
            cond = self.vqvae.encode(cond_coords, cond_faces)

            # Drop BOS; EOS and PAD both sit above the bin range, so one test
            # separates real coordinates from everything else.
            body = batch["tgt"][:, 1:]
            body_pad = batch.get("tgt_pad_mask")
            body_pad = body_pad[:, 1:] if body_pad is not None else torch.zeros_like(body, dtype=torch.bool)
            coords, faces_pad = self._faces(body, body_pad | (body >= self.num_bins))
            codes = self.vqvae.tokenize(coords, faces_pad)      # [B, F, 3, depth]

            # Flatten to the reference's `b (nf nv q)`: face-major, then vertex,
            # with the residual stage varying fastest.
            per_face = self.vqvae.tokens_per_face
            flat = codes.flatten(start_dim=1)
            valid = (~faces_pad).repeat_interleave(per_face, dim=1)

            # Padding is always a suffix, so the valid codes are a prefix and
            # EOS goes at index 1 + count.
            b, width = flat.shape
            tgt = flat.new_full((b, width + 2), self.pad)
            tgt[:, 0] = self.bos
            tgt[:, 1:width + 1] = torch.where(valid, flat, self.pad)
            n = valid.sum(dim=1)
            tgt[torch.arange(b, device=flat.device), n + 1] = self.eos

            positions = torch.arange(width + 2, device=flat.device)[None]
            tgt_pad = positions > (n + 1)[:, None]

        return cond, tgt, cond_faces, tgt_pad

    def _faces(self, tokens, pad_mask):
        """[B, L] coordinate tokens to ``([B, F, 9] coords, [B, F] pad)``.

        Trimmed to the last real coordinate, rounded up to a whole face. The
        target sequence carries EOS and padding past that point, and reshaping
        over them would append a face slot made entirely of padding to every
        batch -- masked out, so harmless to the loss, but it would let the
        sequence width be set by where EOS landed rather than by the face count.
        """
        real = int((~pad_mask).sum(dim=1).max()) if pad_mask.numel() else 0
        width = 9 * ((real + 8) // 9)
        if width < tokens.shape[1]:
            tokens, pad_mask = tokens[:, :width], pad_mask[:, :width]
        elif width > tokens.shape[1]:
            short = width - tokens.shape[1]
            tokens = F.pad(tokens, (0, short), value=0)
            pad_mask = F.pad(pad_mask, (0, short), value=True)
        # Clamp because PAD/EOS are out of range for the coordinate embedding;
        # those slots are masked out anyway.
        coords = tokens.clamp(0, self.num_bins - 1).reshape(tokens.shape[0], -1, 9)
        # A face survives only if all nine of its coordinates are real.
        return coords, pad_mask.reshape(tokens.shape[0], -1, 9).any(dim=-1)

    def decode_tokens(self, tokens, cond=None):
        """A generated sequence to ``(verts, faces)`` in the unit box.

        The one place that knows which tokenizer produced the sequence, so
        `run_mesh_eval` does not have to.

        Args:
            tokens: [L] ids for one mesh.
            cond: [Fc, d] encoded LOD1 features for *this* mesh, with padding
                already trimmed. Required when the VQ-VAE's decoder was
                fine-tuned with a condition (`conditioned_decoder`), ignored
                otherwise: a plain stage-1 decoder has untrained condition
                weights, so handing it one would corrupt the decode rather than
                sharpen it. `_item_cond` produces it from a `_prepare` batch.

        Raises:
            ValueError: the decoder went through the stage-1b fine-tune and no
                condition was given.
        """
        # Callers hand this CPU tensors (the coordinate path is numpy anyway);
        # the VQ-VAE decode below is a forward pass on this module's device.
        tokens = torch.as_tensor(tokens).reshape(-1).to(self.device)
        if self.vqvae is None:
            inverse = amt_detokenize if self.tokenization == "amt" else detokenize
            return inverse(tokens.cpu().numpy(), self.num_bins)

        depth, size = self.vqvae.depth, self.vqvae.codebook_size
        per_face = self.vqvae.tokens_per_face
        codes = tokens[tokens < size]
        # Whole faces only: a trailing partial face has no vertices to decode.
        codes = codes[: len(codes) - len(codes) % per_face]
        if len(codes) == 0:
            return np.zeros((0, 3)), np.zeros((0, 3), dtype=np.int64)

        if self.vqvae.conditioned_decoder:
            # Loudly, not silently. `cond=None` here still returns a mesh --
            # a scrambled one, because the decoder was fine-tuned with the
            # condition prepended and never learned to work without it. That
            # reads downstream as a broken model or a broken dataset, which is
            # a far more expensive thing to debug than a missing argument.
            if cond is None:
                raise ValueError(
                    "This VQ-VAE went through the stage-1b fine-tune "
                    "(conditioned_decoder=True), so its decoder needs the LOD1 "
                    "condition it was trained with. Pass "
                    "cond=model._item_cond((cond, cond_pad), i) from the same "
                    "`_prepare` batch as these tokens."
                )
            cond = torch.as_tensor(cond).to(self.device)
            cond = cond[None] if cond.dim() == 2 else cond
        else:
            cond = None
        return self.vqvae.detokenize(codes.reshape(-1, 3, depth), cond=cond)

    def _shared_step(self, batch):
        """Returns (loss, logits, targets, cond); targets are PAD where masked out.

        ``cond`` comes back because the noise-resistant decoder needs it at
        decode time and recomputing `_prepare` to get it would re-run the
        tokenizer.
        """
        cond, tgt, cond_pad_mask, tgt_pad_mask = self._prepare(batch)
        logits = self.network(cond, tgt, cond_pad_mask, tgt_pad_mask)

        targets = tgt[:, 1:]
        pad_mask = tgt_pad_mask
        if pad_mask is not None:
            # Mask by position, not by token value: a padded slot must not train
            # the model to emit anything, whatever id happens to sit there.
            targets = targets.masked_fill(pad_mask[:, 1:], self.pad)

        loss = F.cross_entropy(
            logits.reshape(-1, logits.shape[-1]), targets.reshape(-1),
            ignore_index=self.pad,
        )
        return loss, logits, targets, (cond, cond_pad_mask)

    def _token_metrics(self, logits, targets, scale=None):
        """Teacher-forced token statistics. Pure tensor ops -- no host sync.

        Exact-match accuracy alone is a poor read on this model. It scores an
        off-by-one-bin coordinate (0.16 m on a 20 m building, invisible in a
        render) the same as a coordinate at the other end of the grid, and it
        is dominated by the tokens the model can copy straight off the LOD1
        prefix. So this adds the magnitude of the error, in bins and in metres,
        and separates the one token that decides whether the mesh terminates.

        Args:
            logits: [B, L, vocab]; targets: [B, L], PAD at masked positions.
            scale: optional [B, 3] metres-per-unit-box from the batch, which
                turns the bin error into a distance.

        Returns:
            dict: str -> scalar tensor, unprefixed.
        """
        eos = self.eos
        pred = logits.argmax(dim=-1)
        keep = targets != self.pad
        err = (pred - targets).abs().float()

        n_tok = keep.sum().clamp(min=1)
        out = {"token_acc": ((pred == targets) & keep).sum() / n_tok}

        if self.vqvae is not None:
            # A code id is a nominal label: |code_a - code_b| means nothing, so
            # every magnitude statistic below would be noise dressed as a metric.
            # Per-stage accuracy is the honest equivalent -- later residual
            # stages are strictly harder, and that split is what shows it.
            stage = torch.arange(targets.shape[1], device=targets.device) % self.vqvae.depth
            for k in range(self.vqvae.depth):
                sel = keep & (stage == k) & (targets < self.vqvae.codebook_size)
                out[f"code_acc_{k}"] = (((pred == targets) & sel).sum()
                                        / sel.sum().clamp(min=1))
        else:
            is_coord = keep & (targets < self.num_bins)
            n_coord = is_coord.sum().clamp(min=1)
            out["bin_mae"] = (err * is_coord).sum() / n_coord
            out["acc_1bin"] = ((err <= 1) & is_coord).sum() / n_coord

            # The tokenizer emits x, y, z per vertex, so position mod 3 is the
            # axis. Split that way rather than by depth: z is where LOD2 actually
            # differs from its LOD1 condition, while x and y are largely copyable.
            axis = torch.arange(targets.shape[1], device=targets.device) % 3
            for k, name in enumerate("xyz"):
                sel = is_coord & (axis == k)
                out[f"bin_mae_{name}"] = (err * sel).sum() / sel.sum().clamp(min=1)

            if scale is not None:
                # Per-axis bin width in metres; `scale` already carries the margins.
                width = (scale / (self.num_bins - 1))[:, axis]
                out["coord_mae_m"] = (err * width * is_coord).sum() / n_coord

        # One EOS in ~1800 tokens is ~0.06% of the gradient and invisible in
        # any aggregate. If the model never learns it, every generation runs to
        # max_new_tokens instead of closing, and only these two show it.
        at_eos = targets == eos
        not_eos = keep & (targets != eos)
        out["eos_acc"] = ((pred == eos) & at_eos).sum() / at_eos.sum().clamp(min=1)
        out["eos_fp_rate"] = ((pred == eos) & not_eos).sum() / not_eos.sum().clamp(min=1)
        return out

    def _log_token_metrics(self, loss, logits, targets, batch, prefix, on_step):
        self.log(f"{prefix}_loss", loss, on_step=on_step, on_epoch=True, prog_bar=True)
        self.log(f"{prefix}_ppl", torch.exp(loss.detach()),
                 on_step=on_step, on_epoch=True)

        metrics = self._token_metrics(logits.detach(), targets, batch.get("scale"))
        for name, value in metrics.items():
            self.log(f"{prefix}_{name}", value, on_step=on_step, on_epoch=True,
                     prog_bar=name in ("token_acc", "coord_mae_m"))

    def _item_cond(self, condition, i):
        """The [Fc, d] condition for item ``i``, unpadded, or None.

        None on the coordinate path (ids, not features) and whenever the decoder
        was not fine-tuned to expect one.
        """
        cond, cond_pad = condition if condition else (None, None)
        if cond is None or not cond.is_floating_point():
            return None
        if self.vqvae is None or not self.vqvae.conditioned_decoder:
            return None
        return cond[i] if cond_pad is None else cond[i][~cond_pad[i]]

    def _log_tf_chamfer(self, logits, targets, batch, prefix, batch_idx, stride,
                        condition=None):
        """Surface distance between the teacher-forced decode and its target.

        Read this as *the metric size of a typical token error*, not as
        generation quality: teacher forcing feeds the ground-truth prefix, so
        the decoded mesh is the target with isolated swaps and no exposure
        bias. The free-running counterpart lives in `MeshEvalCallback`, and the
        gap between the two is the exposure-bias readout.

        One sample per batch, strided, because it needs a host sync and an
        exact point-to-surface query against both meshes -- ~44 ms at n=1024,
        cheap per call but not at every training step. At `log_every_n_steps`
        50 that is ~12 s per training epoch on this corpus.
        """
        if stride <= 0 or batch_idx % stride or "scale" not in batch:
            return

        i = 0
        pred = logits[i].argmax(dim=-1)
        real = targets[i] != self.pad
        if not bool(real.any()):
            return

        center = batch["center"][i].detach().cpu().numpy()
        scale = batch["scale"][i].detach().cpu().numpy()
        item_cond = self._item_cond(condition, i)
        gen = self.decode_tokens(pred[real].detach(), cond=item_cond)
        ref = self.decode_tokens(targets[i][real].detach(), cond=item_cond)
        if len(gen[1]) == 0 or len(ref[1]) == 0:
            return

        d_ab, d_ba = surface_distances((gen[0] * scale + center, gen[1]),
                                       (ref[0] * scale + center, ref[1]), n=1024)
        value = chamfer_distance(d_ab, d_ba)
        if np.isfinite(value):
            self.log(f"{prefix}_tf_chamfer_m", value, on_epoch=True,
                     batch_size=logits.shape[0])

    def training_step(self, batch, batch_idx):
        loss, logits, targets, condition = self._shared_step(batch)
        self._log_token_metrics(loss, logits, targets, batch, "train", on_step=True)
        # Strided on train only: every step would stall the pipeline on a sync.
        # `_trainer`, not `trainer`: the property raises when detached, which a
        # unit test calling training_step directly always is.
        trainer = getattr(self, "_trainer", None)
        stride = trainer.log_every_n_steps if trainer is not None else 0
        self._log_tf_chamfer(logits, targets, batch, "train", batch_idx, stride,
                             condition)
        return loss

    def _eval_step(self, batch, prefix, batch_idx=0):
        loss, logits, targets, condition = self._shared_step(batch)
        self._log_token_metrics(loss, logits, targets, batch, prefix, on_step=False)
        # Every eval batch: one sample each, and eval is not on the hot path.
        self._log_tf_chamfer(logits, targets, batch, prefix, batch_idx, stride=1,
                             condition=condition)
        return loss

    def validation_step(self, batch, batch_idx):
        return self._eval_step(batch, "val", batch_idx)

    def test_step(self, batch, batch_idx):
        return self._eval_step(batch, "test", batch_idx)

    def configure_optimizers(self):
        # requires_grad filter, not self.parameters(): a frozen VQ-VAE handed to
        # AdamW would still accumulate optimizer state for every codebook entry.
        optimizer = torch.optim.AdamW(
            [p for p in self.parameters() if p.requires_grad], lr=self.lr)

        if self.lr_scheduler == "cosine":
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                optimizer,
                T_max=self.trainer.max_epochs if self.trainer else 100,
                eta_min=1e-6,
            )
        elif self.lr_scheduler == "step":
            scheduler = torch.optim.lr_scheduler.StepLR(
                optimizer, step_size=self.lr_decay_steps, gamma=self.lr_decay_rate)
        elif self.lr_scheduler == "none":
            return optimizer
        else:
            raise ValueError(f"Unknown lr_scheduler: {self.lr_scheduler}")

        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "monitor": "val_loss",
                "interval": "epoch",
                "frequency": 1,
            },
        }

    @torch.no_grad()
    def generate(self, cond, cond_pad_mask=None, max_new_tokens=None, temperature=1.0):
        """Greedy/temperature sampling of an LOD2 token sequence from LOD1 tokens.

        Args:
            max_new_tokens: defaults to what the positional embedding can hold.
                Asking for more raises in `forward` rather than sampling.

        Returns:
            Tensor: [B, L] generated tokens, BOS-led, EOS-terminated where the
            model chose to stop. Decode with
            `src.dataset.mesh_dataset.detokenize`.
        """
        bos, eos = self.bos, self.eos
        if max_new_tokens is None:
            max_new_tokens = self.network.max_seq_len - 1
        tgt = torch.full((cond.shape[0], 1), bos, dtype=torch.long, device=cond.device)
        done = torch.zeros(cond.shape[0], dtype=torch.bool, device=cond.device)

        for _ in range(max_new_tokens):
            # ponytail: no KV cache, so this is O(L^2) per step. Add one when
            # sampling time actually hurts -- correctness first.
            step = torch.cat([tgt, torch.full_like(tgt[:, :1], self.pad)], dim=1)
            logits = self.network(cond, step, cond_pad_mask,
                                  torch.zeros_like(step, dtype=torch.bool))[:, -1]

            # Masking Invalid Predictions (MeshAnything V2 section 3.2, after
            # PolyGen): make structurally impossible tokens unreachable rather
            # than merely unlikely. -inf, not a small penalty, so it survives
            # any temperature. Coordinates are never masked, so this cannot
            # starve the softmax.
            if self.mask_invalid and self.vqvae is None:
                bad = invalid_logits_mask(tgt, logits.shape[-1], self.num_bins,
                                          tokenization=self.tokenization,
                                          pad=self.pad)
                logits = logits.masked_fill(bad, float("-inf"))

            if temperature <= 0:
                nxt = logits.argmax(dim=-1)
            else:
                nxt = torch.multinomial(
                    torch.softmax(logits / temperature, dim=-1), num_samples=1).squeeze(-1)
            nxt = torch.where(done, torch.full_like(nxt, self.pad), nxt)
            tgt = torch.cat([tgt, nxt[:, None]], dim=1)
            done |= nxt == eos
            if bool(done.all()):
                break
        return tgt
