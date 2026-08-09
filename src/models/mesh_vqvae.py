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
        depth (int): residual quantization stages, i.e. codes per face. 3 is the
            paper's setting and gives 9 -> 3 token compression.
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
                 max_faces=512, commitment=0.1, decay=0.8, rotation_trick=False):
        super().__init__()
        self.num_bins = num_bins
        self.codebook_size = codebook_size
        self.depth = depth
        self.max_faces = max_faces

        # A face is 9 coordinates; embed each and project the concatenation, so
        # the slot a coordinate sits in (which vertex, which axis) is preserved.
        # Mean-pooling would throw exactly that away.
        self.coord_embed = nn.Embedding(num_bins, d_model)
        self.face_proj = nn.Linear(COORDS_PER_FACE * d_model, d_model)
        self.pos_embed = nn.Embedding(max_faces, d_model)

        self.encoder = _stack(d_model, n_head, num_layers, dropout)
        self.decoder = _stack(d_model, n_head, num_layers, dropout)
        # One codebook per residual stage: stage i quantizes what stages
        # 0..i-1 left behind, so they must not share entries.
        #
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
            dim=d_model,
            num_quantizers=depth,
            codebook_size=codebook_size,
            kmeans_init=True,
            threshold_ema_dead_code=2,
            decay=decay,
            commitment_weight=commitment,
            rotation_trick=rotation_trick,
        )

        self.head = nn.Linear(d_model, COORDS_PER_FACE * num_bins)

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

    def quantize(self, z, pad_mask=None):
        """Residual vector quantization with a straight-through estimator.

        Returns:
            tuple: (z_q [B, F, d], commit_loss scalar, codes [B, F, depth]).
            Padded faces come back as ``-1`` codes and zeroed features, and are
            excluded from the commitment loss. ``z_q`` carries ``z``'s gradient:
            argmin is not differentiable, so without the straight-through pass
            the encoder never trains.
        """
        # The codebook is an EMA buffer, so `self.training` alone decides whether
        # it moves -- and stage 2 holds this network as a frozen submodule whose
        # `eval()` Lightning undoes on every epoch by recursing `model.train()`.
        # Freezing on "no parameter wants a gradient" keeps the stage-2
        # vocabulary fixed without depending on the mode flag surviving.
        frozen = not any(p.requires_grad for p in self.parameters())
        z_q, codes, losses = self.quantizer(
            z, mask=None if pad_mask is None else ~pad_mask, freeze_codebook=frozen)
        return z_q, losses.sum(), codes

    def lookup(self, codes):
        """Sum the residual stages back into one feature vector per face.

        Out-of-range indices are clamped rather than raising: a stage-2
        transformer samples from a softmax and can emit anything, and a decoder
        that crashes on bad input cannot be measured. This also folds away the
        ``-1`` that `quantize` returns for padded faces.
        """
        return self.quantizer.get_output_from_indices(
            codes.clamp(0, self.codebook_size - 1))

    def decode(self, z_q, pad_mask=None):
        """Per-coordinate logits.

        Returns:
            Tensor: [B, F, 9, num_bins]. Coordinates, not offsets -- the paper
            trains the decoder with cross-entropy on vertex coordinate logits,
            which keeps the output on the same grid the tokenizer uses.
        """
        pos = torch.arange(z_q.shape[1], device=z_q.device)
        h = self.decoder(z_q + self.pos_embed(pos)[None], src_key_padding_mask=pad_mask)
        return self.head(h).unflatten(-1, (COORDS_PER_FACE, self.num_bins))

    # ------------------------------------------------------------------
    # The tokenizer interface, mirroring src.dataset.mesh_dataset
    # ------------------------------------------------------------------

    def tokenize(self, coords, pad_mask=None):
        """[B, F, 9] discretized coordinates to [B, F, depth] codes."""
        return self.quantize(self.encode(coords, pad_mask), pad_mask)[2]

    def detokenize(self, codes):
        """[F, depth] codes to ``(verts [V, 3], faces [F, 3])`` in [-0.5, 0.5].

        The inverse of `tokenize` for a single mesh. Mirrors the coordinate
        `detokenize`: empty in, empty out, never an exception, so an aggregator
        can average over a batch where some items decoded to nothing.
        """
        codes = torch.as_tensor(codes)
        if codes.numel() == 0:
            return np.zeros((0, 3)), np.zeros((0, 3), dtype=np.int64)

        logits = self.decode(self.lookup(codes)[None])          # [1, F, 9, bins]
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
                 lr=1e-4, lr_scheduler="none", lr_decay_steps=50, lr_decay_rate=0.5):
        super().__init__()
        self.save_hyperparameters()
        self.lr = lr
        self.lr_scheduler = lr_scheduler
        self.lr_decay_steps = lr_decay_steps
        self.lr_decay_rate = lr_decay_rate
        self.network = MeshVQVAE(
            num_bins=num_bins, codebook_size=codebook_size, depth=depth,
            d_model=d_model, n_head=n_head, num_layers=num_layers,
            dropout=dropout, max_faces=max_faces, commitment=commitment,
            decay=decay, rotation_trick=rotation_trick,
        )

    def _shared_step(self, batch):
        """Returns (loss, metrics dict)."""
        coords, pad = batch["coords"], batch.get("pad_mask")
        z = self.network.encode(coords, pad)
        z_q, vq_loss, codes = self.network.quantize(z, pad)
        logits = self.network.decode(z_q, pad)

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
        return recon + vq_loss, {
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

    def validation_step(self, batch, batch_idx):
        loss, metrics = self._shared_step(batch)
        self._log({"loss": loss.detach(), **metrics}, "val", on_step=False)
        return loss

    def test_step(self, batch, batch_idx):
        loss, metrics = self._shared_step(batch)
        self._log({"loss": loss.detach(), **metrics}, "test", on_step=False)
        return loss

    def configure_optimizers(self):
        opt = torch.optim.AdamW(self.parameters(), lr=self.lr)
        if self.lr_scheduler == "cosine":
            sched = torch.optim.lr_scheduler.CosineAnnealingLR(
                opt, T_max=self.trainer.max_epochs)
        elif self.lr_scheduler == "step":
            sched = torch.optim.lr_scheduler.StepLR(
                opt, step_size=self.lr_decay_steps, gamma=self.lr_decay_rate)
        else:
            return opt
        return {"optimizer": opt, "lr_scheduler": sched}
