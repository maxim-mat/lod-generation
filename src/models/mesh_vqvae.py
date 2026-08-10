"""VQ-VAE mesh tokenizer: a learned face vocabulary for the mesh transformer.

Follows MeshAnything (Chen et al., arXiv:2406.10163) §4.2, which in turn follows
MeshGPT (Siddiqui et al., 2023): a mesh is fed in as a sequence of triangle
faces, the encoder produces one feature vector per face, residual vector
quantization turns those into codebook indices, and the decoder predicts logits
over each vertex coordinate.

Why this exists: `src.dataset.mesh_dataset` emits 9 discretized coordinates per
face, so a 200-face building is 1800 tokens and every single one of them has to
be right for the mesh to close. Run `mesh-1` converged there with a coherent but
systematically simplified mesh (see the design spec). A learned vocabulary buys
3x compression and, more importantly, a decoder that maps imperfect codes onto
the manifold of faces it was trained on, which raw coordinate arithmetic cannot.

Quantization is **per vertex**, not per face, which is what the reference
implementation does: `face_per_token = num_quantizers * 3` in
`MeshAnything/models/meshanything.py`, with the decoder concatenating three
vertex embeddings per face (`project_down_codebook = Linear(codebook_dim * 3,
n_embd)`). That is 3 * depth = 9 tokens per face, the same count as raw
coordinates -- the codebook buys a *learned vocabulary*, not compression.

An earlier version here quantized one feature per face for 3 tokens, and paid
for it: 3 codes of a 1024-entry book is ~30 bits to carry 9 coordinates of 7
bits each, so the bottleneck was 2x narrower than the data it had to pass.
Reconstruction bottomed out at 0.246 m chamfer against the coordinate
tokenizer's 0.107 m, and run mesh-3 inherited that as a ceiling it could not
generate its way past.

Two deliberate departures from the paper, both recorded in
`docs/superpowers/specs/2026-08-08-mesh-vqvae-design.md`:

  * The encoder is exposed separately from the quantizer (`encode`), because the
    LOD1 condition is encoded but never quantized. The paper never faces this
    question -- its condition is a point cloud through a different, frozen
    encoder -- but ours is the same modality as the target, and quantizing an
    input that is never sampled only discards precision.
  * The codebook starts at 1024 rather than 8192. Walls, ground planes and roof
    slopes are far less varied than Objaverse art assets; `codebook_stats`
    reports usage so the size is calibrated from a run instead of guessed.

Geometry is not reimplemented: `quantize`, `dequantize` and `canonicalize` come
from the coordinate tokenizer, so `detokenize` here merges vertices by exactly
the same rule as `detokenize` there.
"""
import logging

import lightning as L
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from vector_quantize_pytorch import ResidualVQ

from src.dataset.mesh_dataset import NUM_BINS, canonicalize, dequantize

logger = logging.getLogger(__name__)

COORDS_PER_FACE = 9      # 3 vertices x (x, y, z)
VERTS_PER_FACE = 3


def _stack(d_model, n_head, num_layers, dropout):
    """A transformer stack, same shape for the encoder and the decoder.

    The paper uses BERT encoders for both rather than MeshGPT's graph
    convolutions ("we employ transformers with identical structures").
    """
    layer = nn.TransformerEncoderLayer(
        d_model=d_model, nhead=n_head, dim_feedforward=4 * d_model,
        dropout=dropout, activation="gelu", batch_first=True, norm_first=True,
    )
    return nn.TransformerEncoder(layer, num_layers=num_layers)


class MeshVQVAE(nn.Module):
    """Faces to codebook indices and back.

    Args:
        num_bins (int): coordinate discretization; must match the dataset's.
        codebook_size (int): entries per residual stage.
        depth (int): residual quantization stages, i.e. codes per *vertex*. 3 is
            the paper's setting, giving 3 * 3 = 9 tokens per face.
        d_model (int): channel width; must be divisible by n_head.
        max_faces (int): capacity of the per-face positional embedding. Faces are
            canonically ordered, so their position carries information.
        commitment (float): weight on the term pulling the encoder toward the
            codebook (van den Oord et al., 2017). MeshGPT uses 0.1.
        decay (float): EMA decay for the codebook.
        rotation_trick (bool): rotate the gradient from code to encoder output
            instead of the straight-through copy. See `MeshVQVAEConfig`.
    """

    def __init__(self, num_bins=NUM_BINS, codebook_size=1024, depth=3,
                 d_model=256, n_head=8, num_layers=4, dropout=0.1,
                 max_faces=512, commitment=0.1, decay=0.8, rotation_trick=False,
                 codebook_dim=None, conditioned_decoder=False):
        super().__init__()
        # True once the noise-resistant fine-tune has trained the decoder with a
        # condition. Callers must not feed one otherwise: `cond_proj` and
        # `segment_embed` would still be at their initialization, so offering a
        # condition to a plain stage-1 decoder injects noise into its input.
        self.conditioned_decoder = conditioned_decoder
        self.num_bins = num_bins
        self.codebook_size = codebook_size
        self.depth = depth
        self.max_faces = max_faces
        # Codes live in their own width. MeshGPT's default is 192 against a
        # model dim of 512; None keeps them tied, which is a coincidence rather
        # than a design.
        self.codebook_dim = codebook_dim or d_model

        # A face is 9 coordinates; embed each and project the concatenation, so
        # the slot a coordinate sits in (which vertex, which axis) is preserved.
        # Mean-pooling would throw exactly that away.
        self.coord_embed = nn.Embedding(num_bins, d_model)
        self.face_proj = nn.Linear(COORDS_PER_FACE * d_model, d_model)
        self.pos_embed = nn.Embedding(max_faces, d_model)

        # Face feature <-> its three vertex features. `vert_proj` is the split
        # the paper's encoder gets from gathering per-vertex, and `face_proj_down`
        # is its exact counterpart in the reference decoder
        # (`project_down_codebook = Linear(codebook_dim * 3, n_embd)`).
        # ponytail: a linear split, not MeshGPT's cross-face vertex sharing --
        # add that if vertices shared between faces start disagreeing visibly.
        self.vert_proj = nn.Linear(d_model, VERTS_PER_FACE * self.codebook_dim)
        self.face_proj_down = nn.Linear(VERTS_PER_FACE * self.codebook_dim, d_model)

        # Noise-resistant decoder (paper section 4.2): the condition is injected
        # into the *decoder* so it can correct imperfect codes, not just smooth
        # them. Unused until a fine-tune passes `cond`.
        self.cond_proj = nn.Linear(d_model, d_model)
        self.segment_embed = nn.Embedding(2, d_model)

        self.encoder = _stack(d_model, n_head, num_layers, dropout)
        self.decoder = _stack(d_model, n_head, num_layers, dropout)
        # Settings are MeshGPT's `rvq_kwargs` verbatim, which is what
        # MeshAnything inherits (its §5.2 cites the RVQ of SoundStream,
        # Zeghidour et al. arXiv:2107.03312, and its README credits
        # lucidrains/vector-quantize-pytorch). Hand-rolling this is what produced
        # the collapse in run mesh-vqvae-2 -- 28 live entries of 1024 by epoch 1
        # -- because the paper text describes only the 2017 objective and leaves
        # the machinery that makes it trainable to the library:
        #   * `kmeans_init` seeds entries from real encoder outputs. Without it
        #     an entry far from the data is never selected, never updated, and
        #     therefore dead permanently.
        #   * `threshold_ema_dead_code` revives entries whose EMA usage decays
        #     below 2 (random restarts, Jukebox arXiv:2005.00341 §3.1).
        #   * EMA codebook updates (van den Oord 2017 appendix; VQ-VAE-2,
        #     arXiv:1906.00446) make the codebook a buffer rather than a
        #     parameter, so the codebook MSE leaves the objective entirely and
        #     only the commitment term is left to weigh against `recon_ce`.
        self.quantizer = ResidualVQ(
            dim=self.codebook_dim,
            num_quantizers=depth,
            codebook_size=codebook_size,
            kmeans_init=True,
            threshold_ema_dead_code=2,
            decay=decay,
            commitment_weight=commitment,
            rotation_trick=rotation_trick,
            # One book for every residual stage, as in MeshGPT
            # (`shared_codebook = True`) and MeshAnything
            # (`quantize_codebooks[0][indices]`, `vocab_size = codebook_size + 3`).
            #
            # This is what lets stage 2 embed a code id by looking its vector up
            # in the codebook: with per-stage books, id 5 means a different
            # vector at each stage and the lookup would need position
            # arithmetic. Measured cost of sharing at a realistic
            # points-per-entry ratio: ~4% on quantization error (2.450 vs
            # 2.357) and 508/512 live entries against 512/512.
            shared_codebook=True,
            # Lets a fine-tune ask for Gumbel-noised code selection. A no-op
            # until `sample_temp > 0`, and inert in eval whatever it is set to:
            # `gumbel_sample` gates on `training and stochastic and temp > 0`.
            stochastic_sample_codes=True,
        )

        self.head = nn.Linear(d_model, COORDS_PER_FACE * num_bins)

    @property
    def tokens_per_face(self):
        """Code tokens a face costs: 3 vertices x `depth` residual stages."""
        return VERTS_PER_FACE * self.depth

    # ------------------------------------------------------------------
    # Encode / quantize / decode
    # ------------------------------------------------------------------

    def encode(self, coords, pad_mask=None):
        """Continuous per-face features, no quantization.

        This is the LOD1 condition path: the condition is never sampled and
        never decoded, so it needs an embedding, not a vocabulary.

        Args:
            coords: [B, F, 9] int64 discretized coordinates, or [F, 9].
            pad_mask: [B, F] bool, True at padded faces.

        Returns:
            Tensor: [B, F, d_model].
        """
        coords = self._batched(coords)
        if coords.shape[1] > self.max_faces:
            raise ValueError(
                f"{coords.shape[1]} faces exceeds max_faces={self.max_faces}; "
                "raise MeshVQVAEConfig.max_faces or lower mesh_data.max_faces.")

        e = self.coord_embed(coords)                       # [B, F, 9, d]
        x = self.face_proj(e.flatten(start_dim=2))         # [B, F, d]
        pos = torch.arange(x.shape[1], device=x.device)
        x = x + self.pos_embed(pos)[None]
        return self.encoder(x, src_key_padding_mask=pad_mask)

    def vertex_ids(self, coords, pad_mask=None):
        """[B, F, 9] coordinates to ``([B, F, 3] indices, [B] vertex counts)``.

        The token stream carries coordinates, never topology -- `detokenize`
        recovers indices by merging identical grid points, and this applies the
        same merge at encode time so the quantizer can see which faces meet.

        Padded faces point one index past the mesh's real vertices, into a slot
        the quantizer masks out.
        """
        coords = self._batched(coords)
        b, f, _ = coords.shape
        ids = torch.zeros((b, f, VERTS_PER_FACE), dtype=torch.long, device=coords.device)
        counts = torch.zeros(b, dtype=torch.long, device=coords.device)
        for i in range(b):
            verts = coords[i].reshape(-1, 3)
            keep = (None if pad_mask is None
                    else (~pad_mask[i]).repeat_interleave(VERTS_PER_FACE))
            # A padded face's coordinates are arbitrary; letting them define
            # vertices would shift the indices of the real ones.
            uniq, inverse = torch.unique(verts if keep is None else verts[keep],
                                         dim=0, return_inverse=True)
            counts[i] = len(uniq)
            flat = ids[i].reshape(-1)
            if keep is None:
                flat[:] = inverse
            else:
                flat[keep] = inverse
                flat[~keep] = len(uniq)          # the pad vertex
        return ids, counts

    def quantize(self, z, faces, counts=None, pad_mask=None, sample_temp=0.0):
        """Quantize the mesh's *unique* vertices, then gather codes back per face.

        Follows MeshGPT's `MeshAutoencoder.quantize`: the face feature is
        projected into three vertex slots, the slots of one vertex are averaged
        over every face touching it (`scatter_mean`), the unique vertices are
        quantized, and the result is gathered back per face
        (`get_at('b [n] q, b nf nvf -> b (nf nvf) q')`).

        The gather is the point. A vertex shared by k faces emits *the same
        code* k times, so it decodes to one coordinate and `canonicalize`'s
        exact-equality merge cannot crack the mesh -- and the token sequence
        gains a large, learnable redundancy instead of k independent guesses.

        Args:
            z: [B, F, d] face features from `encode`.
            faces: [B, F, 3] vertex indices from `vertex_ids`.
            counts: [B] real vertex count per mesh, also from `vertex_ids`.
            pad_mask: [B, F] bool, True at padded faces.
            sample_temp: Gumbel temperature for code selection. 0 (the default)
                is plain argmin. The paper's noise-resistant fine-tune raises it
                so the decoder learns to survive codes it would not have picked.

        Returns:
            tuple: (z_q [B, F, 3, dc], commit_loss scalar, codes [B, F, 3, depth]).
            Padded faces come back as ``-1`` codes and are excluded from the
            commitment loss. ``z_q`` carries ``z``'s gradient: argmin is not
            differentiable, so without the straight-through pass the encoder
            never trains.
        """
        b, f, _ = z.shape
        dc = self.codebook_dim
        n_vert = int(faces.max()) + 1                  # includes the pad vertex
        v = self.vert_proj(z).reshape(b, f * VERTS_PER_FACE, dc)

        # Average every face's contribution to a vertex into one feature.
        index = faces.reshape(b, -1, 1).expand(-1, -1, dc)
        pooled = v.new_zeros((b, n_vert, dc)).scatter_reduce(
            1, index, v, reduce="mean", include_self=False)

        # Real vertices only: the pad slot, and the tail of a mesh with fewer
        # vertices than the batch's longest, must not train the codebook.
        if counts is None:
            counts = torch.full((b,), n_vert, device=z.device)
        mask = torch.arange(n_vert, device=z.device)[None] < counts.to(z.device)[:, None]

        # The codebook is an EMA buffer, so `self.training` alone decides whether
        # it moves -- and stage 2 holds this network as a frozen submodule whose
        # `eval()` Lightning undoes on every epoch by recursing `model.train()`.
        #
        # Gate on the encoder, not on every parameter: the noise-resistant
        # fine-tune trains the decoder while the vocabulary must stay put, so
        # "some parameter wants a gradient" is no longer the right question.
        # If nothing can move `z`, nothing may move the codebook either.
        frozen = not any(p.requires_grad for p in self.encoder.parameters())
        quantized, codes, losses = self.quantizer(
            pooled, mask=mask, freeze_codebook=frozen, sample_codebook_temp=sample_temp)

        # Gather back into face order; every face touching a vertex gets its code.
        z_q = quantized.gather(1, index).reshape(b, f, VERTS_PER_FACE, dc)
        codes = codes.gather(
            1, faces.reshape(b, -1, 1).expand(-1, -1, self.depth)
        ).reshape(b, f, VERTS_PER_FACE, self.depth)
        if pad_mask is not None:
            codes = codes.masked_fill(pad_mask[:, :, None, None], -1)
        return z_q, losses.sum(), codes

    def lookup(self, codes):
        """Sum the residual stages back into one feature vector per vertex.

        Args:
            codes: ``[..., 3, depth]`` codebook indices.

        Out-of-range indices are clamped rather than raising: a stage-2
        transformer samples from a softmax and can emit anything, and a decoder
        that crashes on bad input cannot be measured. This also folds away the
        ``-1`` that `quantize` returns for padded faces.
        """
        flat = codes.clamp(0, self.codebook_size - 1).reshape(-1, self.depth)
        return self.quantizer.get_output_from_indices(flat).reshape(*codes.shape[:-1], -1)

    def decode(self, z_q, pad_mask=None, cond=None, cond_pad_mask=None):
        """Per-coordinate logits.

        Args:
            z_q: [B, F, 3, d] quantized vertex features.
            pad_mask: [B, F] bool, True at padded faces.
            cond, cond_pad_mask: optional [B, Fc, d] condition features and their
                mask. Prepended to the decoder input and sliced back off, as in
                the reference decoder's
                ``concatenate([point_feature, face_embeds])`` followed by
                ``last_hidden_state[:, cond_length:]``. This is what lets a
                noise-resistant decoder *correct* a bad code instead of only
                smoothing it; without it the fine-tune has no reference to
                correct against.

        Returns:
            Tensor: [B, F, 9, num_bins]. Coordinates, not offsets -- the paper
            trains the decoder with cross-entropy on vertex coordinate logits,
            which keeps the output on the same grid the tokenizer uses.
        """
        # Three vertex features back into one face token, the inverse of the
        # split in `quantize`.
        h = self.face_proj_down(z_q.flatten(start_dim=-2))
        pos = torch.arange(h.shape[1], device=h.device)
        h = h + self.pos_embed(pos)[None] + self.segment_embed.weight[1]

        n_cond = 0
        if cond is not None:
            n_cond = cond.shape[1]
            c = self.cond_proj(cond) + self.segment_embed.weight[0]
            h = torch.cat([c, h], dim=1)
            if pad_mask is not None:
                zeros = (torch.zeros(cond.shape[:2], dtype=torch.bool, device=h.device)
                         if cond_pad_mask is None else cond_pad_mask)
                pad_mask = torch.cat([zeros, pad_mask], dim=1)

        h = self.decoder(h, src_key_padding_mask=pad_mask)[:, n_cond:]
        return self.head(h).unflatten(-1, (COORDS_PER_FACE, self.num_bins))

    # ------------------------------------------------------------------
    # The tokenizer interface, mirroring src.dataset.mesh_dataset
    # ------------------------------------------------------------------

    def tokenize(self, coords, pad_mask=None, sample_temp=0.0):
        """[B, F, 9] discretized coordinates to [B, F, 3, depth] codes."""
        faces, counts = self.vertex_ids(coords, pad_mask)
        return self.quantize(self.encode(coords, pad_mask), faces, counts,
                             pad_mask, sample_temp)[2]

    def detokenize(self, codes, cond=None):
        """[F, 3, depth] codes to ``(verts [V, 3], faces [F, 3])`` in [-0.5, 0.5].

        The inverse of `tokenize` for a single mesh. Mirrors the coordinate
        `detokenize`: empty in, empty out, never an exception, so an aggregator
        can average over a batch where some items decoded to nothing.
        """
        codes = torch.as_tensor(codes)
        if codes.numel() == 0:
            return np.zeros((0, 3)), np.zeros((0, 3), dtype=np.int64)

        logits = self.decode(self.lookup(codes)[None], cond=cond)   # [1, F, 9, bins]
        q = logits.argmax(dim=-1)[0].reshape(-1, 3).cpu().numpy()
        faces = np.arange(len(q), dtype=np.int64).reshape(-1, 3)
        q, faces = canonicalize(q, faces)
        return dequantize(q, self.num_bins), faces

    # ------------------------------------------------------------------

    def codebook_stats(self, codes):
        """Codebook occupancy, pooled and per residual stage.

        The diagnostic for whether `codebook_size` is set anywhere near the
        diversity of the data: a codebook far larger than the geometry warrants
        leaves most entries dead and the survivors undertrained.

        Reported per stage as well as pooled, because the two failures need
        different fixes and the pooled number cannot tell them apart: stage 0
        healthy with the later stages collapsed onto one entry each (the usual
        residual-VQ failure) pools to the same count as every stage being
        equally starved.

        Args:
            codes: ``[..., depth]`` codebook indices, as `quantize` returns.

        Returns:
            dict: ``distinct``/``perplexity`` over all stages pooled, plus
            ``distinct_per_stage``/``perplexity_per_stage`` lists.
        """
        codes = torch.as_tensor(codes)
        if codes.numel() == 0:
            return {"distinct": 0, "perplexity": 0.0,
                    "distinct_per_stage": [], "perplexity_per_stage": []}

        def occupancy(flat):
            # `quantize` returns -1 for padded faces; those are not codes.
            flat = flat[flat >= 0]
            if flat.numel() == 0:
                return 0, 0.0
            counts = torch.bincount(flat, minlength=self.codebook_size).float()
            p = counts / counts.sum()
            nz = p[p > 0]
            return int((counts > 0).sum()), float(torch.exp(-(nz * nz.log()).sum()))

        distinct, perplexity = occupancy(codes.reshape(-1))
        per_stage = [occupancy(codes[..., k].reshape(-1))
                     for k in range(codes.shape[-1])]
        return {"distinct": distinct, "perplexity": perplexity,
                "distinct_per_stage": [d for d, _ in per_stage],
                "perplexity_per_stage": [p for _, p in per_stage]}

    @staticmethod
    def _batched(coords):
        coords = torch.as_tensor(coords)
        return coords[None] if coords.dim() == 2 else coords

    @staticmethod
    def _valid(z, pad_mask):
        """[B, F] float, 1 at real faces. Padding must not train anything."""
        if pad_mask is None:
            return z.new_ones(z.shape[:-1])
        return (~pad_mask).to(z.dtype)


class MeshVQVAEModule(L.LightningModule):
    """Stage-1 training: reconstruct a mesh's own faces through the codebook.

    Loss is cross-entropy on the coordinate logits plus the quantizer's
    commitment term, exactly as the paper trains it end to end.

    There is no weight on the quantizer side here: the codebook is updated by
    EMA, not by gradient, so the only term that reaches this objective is the
    commitment loss, already scaled by `commitment` inside `ResidualVQ`. Run
    mesh-vqvae-1 instead added a raw codebook MSE in `d_model` space -- order 20
    against a cross-entropy of order 3 -- which carried ~85% of the objective and
    of `val_loss`, and is why that run's best checkpoint was epoch 0.
    Monitor `val_recon_ce`, never `val_loss`.
    """

    def __init__(self, num_bins=NUM_BINS, codebook_size=1024, depth=3,
                 d_model=256, n_head=8, num_layers=4, dropout=0.1,
                 max_faces=512, commitment=0.1, decay=0.8, rotation_trick=False,
                 lr=1e-4, lr_scheduler="none", lr_decay_steps=50, lr_decay_rate=0.5,
                 noise_resistant=False, noise_temp=1.0, codebook_dim=None):
        """
        Args:
            noise_resistant (bool): run the paper's section-4.2 fine-tune instead
                of plain stage-1 training -- decoder only, condition injected,
                codes drawn with Gumbel noise. Load stage-1 weights first
                (`mesh_vqvae.init_from`); starting this from scratch trains a
                decoder against a random codebook.
            noise_temp (float): Gumbel temperature for that fine-tune. Higher
                means the decoder sees codes further from the ones the encoder
                would have chosen.
        """
        super().__init__()
        self.save_hyperparameters()
        self.lr = lr
        self.lr_scheduler = lr_scheduler
        self.lr_decay_steps = lr_decay_steps
        self.lr_decay_rate = lr_decay_rate
        self.noise_resistant = noise_resistant
        self.noise_temp = noise_temp
        # Fixes the Gumbel draw behind `noisy_recon_ce`, so that metric measures
        # the decoder rather than the sample. Not the run seed: this must stay
        # the same across epochs, and across a resume.
        self.noise_seed = 20260810
        self.network = MeshVQVAE(
            num_bins=num_bins, codebook_size=codebook_size, depth=depth,
            d_model=d_model, n_head=n_head, num_layers=num_layers,
            dropout=dropout, max_faces=max_faces, commitment=commitment,
            decay=decay, rotation_trick=rotation_trick, codebook_dim=codebook_dim,
            # Carried in the hyperparameters, so a checkpoint knows whether its
            # decoder expects a condition and stage 2 does not have to be told.
            conditioned_decoder=noise_resistant,
        )
        if noise_resistant:
            # Everything the codes depend on. `pos_embed` is shared with the
            # decoder, so it is frozen too rather than given its own table --
            # the decoder was trained with these positions already, and letting
            # them move would shift the encoder's output and with it the
            # vocabulary stage 2 is being trained to predict.
            for part in (self.network.coord_embed, self.network.face_proj,
                         self.network.pos_embed, self.network.vert_proj,
                         self.network.encoder):
                part.requires_grad_(False)

    def _shared_step(self, batch):
        """Returns (loss, metrics dict)."""
        coords, pad = batch["coords"], batch.get("pad_mask")

        cond = cond_pad = None
        temp = 0.0
        if self.noise_resistant:
            if "cond" not in batch:
                raise ValueError(
                    "noise_resistant=True needs a 'cond' mesh in the batch "
                    "(MeshVQVAEDataModule(condition=True)); without one the "
                    "fine-tune would train an unconditioned decoder.")
            cond_pad = batch.get("cond_pad_mask")
            cond = self.network.encode(batch["cond"], cond_pad)
            temp = self.noise_temp

        faces, counts = self.network.vertex_ids(coords, pad)
        z = self.network.encode(coords, pad)
        z_q, vq_loss, codes = self.network.quantize(z, faces, counts, pad, sample_temp=temp)
        logits = self.network.decode(z_q, pad, cond=cond, cond_pad_mask=cond_pad)

        # Mask by position: a padded face must not train the decoder toward
        # whatever coordinates happen to sit in its slot.
        valid = self.network._valid(z, pad).bool()
        recon = F.cross_entropy(logits[valid].reshape(-1, self.network.num_bins),
                                coords[valid].reshape(-1))

        err = (logits[valid].argmax(-1) - coords[valid]).abs().float()
        stats = self.network.codebook_stats(codes[valid])
        # Pooled `codes_used`/`codebook_ppl` keep their old meaning so the curves
        # stay comparable to mesh-vqvae-1/2; the per-stage ones are the diagnostic.
        per_stage = {f"codes_used_s{k}": torch.tensor(float(d))
                     for k, d in enumerate(stats["distinct_per_stage"])}
        # The fine-tune trains the decoder only, so the commitment term is a
        # constant with no gradient. Leaving it in the objective would repeat
        # mesh-vqvae-1's mistake of monitoring a loss whose movement is not the
        # model improving.
        loss = recon if self.noise_resistant else recon + vq_loss
        return loss, {
            "recon_ce": recon.detach(), "vq_loss": vq_loss.detach(),
            "bin_mae": err.mean(), "acc_1bin": (err <= 1).float().mean(),
            "codes_used": torch.tensor(float(stats["distinct"])),
            "codebook_ppl": torch.tensor(stats["perplexity"]),
            **per_stage,
        }

    def _log(self, metrics, prefix, on_step):
        # Guarded so the step methods stay callable from a unit test, where
        # there is no trainer to route self.log through.
        if self._trainer is None:
            return
        for name, value in metrics.items():
            self.log(f"{prefix}_{name}", value, on_step=on_step, on_epoch=True,
                     prog_bar=name in ("recon_ce", "bin_mae"))

    def training_step(self, batch, batch_idx):
        loss, metrics = self._shared_step(batch)
        self._log({"loss": loss.detach(), **metrics}, "train", on_step=True)
        return loss

    @torch.no_grad()
    def _noisy_recon_ce(self, batch):
        """Reconstruction cross-entropy from *noised* codes.

        `gumbel_sample` gates on the codebook module's training flag, so an
        ordinary validation pass sees clean argmin codes however high
        `noise_temp` is set -- which makes `val_recon_ce` a measurement of the
        one thing this fine-tune is not optimizing. Running the quantizer in
        train mode for one extra pass is what un-gates the noise.

        Safe because the EMA is gated separately, on whether anything can move
        the encoder: `quantize` computes `freeze_codebook` from
        `encoder.requires_grad`, which the fine-tune has already turned off.
        """
        quantizer = self.network.quantizer
        was_training = quantizer.training
        rng = torch.random.get_rng_state()
        try:
            quantizer.train()
            # A fixed draw, so epoch-to-epoch movement in this metric is the
            # decoder improving rather than the sample changing under it.
            torch.manual_seed(self.noise_seed)
            return self._shared_step(batch)[1]["recon_ce"]
        finally:
            quantizer.train(was_training)
            torch.random.set_rng_state(rng)

    def _eval_metrics(self, batch):
        """``(loss, metrics)`` for a validation or test batch."""
        loss, metrics = self._shared_step(batch)
        if self.noise_resistant:
            # What `mesh-vqvae-nr.yaml` monitors. See `_noisy_recon_ce`.
            metrics["noisy_recon_ce"] = self._noisy_recon_ce(batch)
        return loss, metrics

    def validation_step(self, batch, batch_idx):
        loss, metrics = self._eval_metrics(batch)
        self._log({"loss": loss.detach(), **metrics}, "val", on_step=False)
        return loss

    def test_step(self, batch, batch_idx):
        loss, metrics = self._eval_metrics(batch)
        self._log({"loss": loss.detach(), **metrics}, "test", on_step=False)
        return loss

    def configure_optimizers(self):
        # requires_grad filter, not self.parameters(): the noise-resistant
        # fine-tune freezes the encoder side, and AdamW would still carry moment
        # buffers for every frozen codebook-facing weight.
        opt = torch.optim.AdamW(
            [p for p in self.parameters() if p.requires_grad], lr=self.lr)
        if self.lr_scheduler == "cosine":
            sched = torch.optim.lr_scheduler.CosineAnnealingLR(
                opt, T_max=self.trainer.max_epochs)
        elif self.lr_scheduler == "step":
            sched = torch.optim.lr_scheduler.StepLR(
                opt, step_size=self.lr_decay_steps, gamma=self.lr_decay_rate)
        else:
            return opt
        return {"optimizer": opt, "lr_scheduler": sched}
