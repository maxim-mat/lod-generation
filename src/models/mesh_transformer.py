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
import torch
import torch.nn as nn
import torch.nn.functional as F

from src.dataset.mesh_dataset import NUM_BINS, specials, vocab_size

logger = logging.getLogger(__name__)


class MeshTransformer(nn.Module):
    """Decoder-only transformer over ``[LOD1 tokens] + [BOS, LOD2 tokens, EOS]``.

    Args:
        vocab_size (int): coordinate bins plus BOS/EOS/PAD.
        d_model (int): channel width; must be divisible by n_head.
        max_seq_len (int): capacity of the learned positional embedding. Must
            cover ``len(cond) + len(tgt) - 1``; `MeshDataset.max_seq_len` reports
            what the data needs.
    """

    def __init__(self, vocab_size, d_model=256, n_head=8, num_layers=6,
                 dropout=0.1, max_seq_len=4096):
        super().__init__()
        self.max_seq_len = max_seq_len
        self.token_embed = nn.Embedding(vocab_size, d_model)
        self.pos_embed = nn.Embedding(max_seq_len, d_model)
        # Which half of the prefix a position belongs to. Without it the model
        # has to infer the LOD1/LOD2 boundary from the BOS token alone.
        self.segment_embed = nn.Embedding(2, d_model)
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
            cond: [B, Lc] LOD1 tokens.
            tgt: [B, Lt] target tokens, BOS-led and EOS-terminated.
            cond_pad_mask, tgt_pad_mask: [B, L] bool, True at padding.

        Returns:
            Tensor: [B, Lt - 1, vocab] where position ``i`` predicts ``tgt[:, i+1]``.
        """
        n_cond, n_tgt = cond.shape[1], tgt.shape[1]
        x = torch.cat([cond, tgt[:, :-1]], dim=1)
        length = x.shape[1]
        if length > self.max_seq_len:
            raise ValueError(
                f"Sequence of {length} exceeds max_seq_len={self.max_seq_len}. "
                "Raise model.max_seq_len or lower data.max_faces."
            )

        pos = torch.arange(length, device=x.device)
        segment = (pos >= n_cond).long()
        h = self.drop(self.token_embed(x) + self.pos_embed(pos) + self.segment_embed(segment))

        # Bool, not the float mask from generate_square_subsequent_mask: torch
        # deprecates mixing mask dtypes with a bool src_key_padding_mask.
        causal = torch.ones(length, length, dtype=torch.bool, device=x.device).triu(1)
        pad_mask = None
        if cond_pad_mask is not None and tgt_pad_mask is not None:
            pad_mask = torch.cat([cond_pad_mask, tgt_pad_mask[:, :-1]], dim=1)

        h = self.blocks(h, mask=causal, src_key_padding_mask=pad_mask, is_causal=True)
        return self.head(self.norm(h[:, n_cond:]))


class MeshTransformerModule(L.LightningModule):
    """Lightning wrapper: next-token cross-entropy over the LOD2 token sequence."""

    def __init__(self, num_bins=NUM_BINS, d_model=256, n_head=8, num_layers=6,
                 dropout=0.1, max_seq_len=4096, lr=1e-4,
                 lr_scheduler="none", lr_decay_steps=50, lr_decay_rate=0.5):
        """
        Args:
            num_bins (int): coordinate discretization used by the tokenizer.
                Must match the datamodule's, or the vocabulary is misaligned;
                saved as a hyperparameter so inference restores it.
            lr_scheduler (str): 'none' | 'cosine' | 'step', as in
                `CityJSONDiffusionModule`.
        """
        super().__init__()
        self.save_hyperparameters()

        self.num_bins = num_bins
        self.lr = lr
        self.lr_scheduler = lr_scheduler
        self.lr_decay_steps = lr_decay_steps
        self.lr_decay_rate = lr_decay_rate
        _, _, self.pad = specials(num_bins)

        self.network = MeshTransformer(
            vocab_size=vocab_size(num_bins), d_model=d_model, n_head=n_head,
            num_layers=num_layers, dropout=dropout, max_seq_len=max_seq_len,
        )

    def _shared_step(self, batch):
        """Returns (loss, logits, targets); targets are PAD where masked out."""
        logits = self.network(batch["cond"], batch["tgt"],
                              batch.get("cond_pad_mask"), batch.get("tgt_pad_mask"))

        targets = batch["tgt"][:, 1:]
        pad_mask = batch.get("tgt_pad_mask")
        if pad_mask is not None:
            # Mask by position, not by token value: a padded slot must not train
            # the model to emit anything, whatever id happens to sit there.
            targets = targets.masked_fill(pad_mask[:, 1:], self.pad)

        loss = F.cross_entropy(
            logits.reshape(-1, logits.shape[-1]), targets.reshape(-1),
            ignore_index=self.pad,
        )
        return loss, logits, targets

    def training_step(self, batch, batch_idx):
        loss, _, _ = self._shared_step(batch)
        self.log("train_loss", loss, on_step=True, on_epoch=True, prog_bar=True)
        return loss

    def _eval_step(self, batch, prefix):
        loss, logits, targets = self._shared_step(batch)

        # Teacher-forced token accuracy over the real (non-padded) positions.
        # Not a geometry metric -- it says nothing about whether the decoded
        # mesh closes -- but it is the cheap per-epoch signal.
        keep = targets != self.pad
        correct = (logits.argmax(dim=-1) == targets) & keep
        acc = correct.sum() / keep.sum().clamp(min=1)

        self.log(f"{prefix}_loss", loss, on_epoch=True, prog_bar=True)
        self.log(f"{prefix}_token_acc", acc, on_epoch=True, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        return self._eval_step(batch, "val")

    def test_step(self, batch, batch_idx):
        return self._eval_step(batch, "test")

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.parameters(), lr=self.lr)

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
    def generate(self, cond, cond_pad_mask=None, max_new_tokens=2048, temperature=1.0):
        """Greedy/temperature sampling of an LOD2 token sequence from LOD1 tokens.

        Returns:
            Tensor: [B, L] generated tokens, BOS-led, EOS-terminated where the
            model chose to stop. Decode with
            `src.dataset.mesh_dataset.detokenize`.
        """
        bos, eos, _ = specials(self.num_bins)
        tgt = torch.full((cond.shape[0], 1), bos, dtype=torch.long, device=cond.device)
        done = torch.zeros(cond.shape[0], dtype=torch.bool, device=cond.device)

        for _ in range(max_new_tokens):
            # ponytail: no KV cache, so this is O(L^2) per step. Add one when
            # sampling time actually hurts -- correctness first.
            step = torch.cat([tgt, torch.full_like(tgt[:, :1], self.pad)], dim=1)
            logits = self.network(cond, step, cond_pad_mask,
                                  torch.zeros_like(step, dtype=torch.bool))[:, -1]
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
