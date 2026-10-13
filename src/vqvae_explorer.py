#!/usr/bin/env python3
"""Streamlit browser for stage-1 VQ-VAE reconstructions on the test split.

Exploratory: pick a test building, pick one or more checkpoints under
`outputs/`, see the ground-truth mesh next to what each checkpoint's codebook
round-trips it into. Throwaway visualization -- exempt from TDD and
reproducibility requirements (project instructions).

Usage:
    streamlit run src/vqvae_explorer.py
"""
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:            # `streamlit run` puts src/ on the path, not the root
    sys.path.insert(0, str(REPO))

import numpy as np                                                     # noqa: E402
import streamlit as st                                                 # noqa: E402
import torch                                                           # noqa: E402

from src.dataset.mesh_datamodule import MeshDataModule                 # noqa: E402
from src.dataset.mesh_dataset import detokenize                        # noqa: E402
from src.eval.mesh_metrics import chamfer_distance, surface_distances  # noqa: E402
from src.models.mesh_vqvae import MeshVQVAEModule                      # noqa: E402
from src.utils.initialization import load_config                       # noqa: E402
from src.visualize_mesh import mesh_figure, side_by_side               # noqa: E402

DEFAULT_CONFIG = REPO / "configs" / "mesh-vqvae.yaml"

st.set_page_config(page_title="VQ-VAE reconstructions", layout="wide")


@st.cache_resource(show_spinner="Scanning corpus and rebuilding the split...")
def test_split(config_path, overrides):
    """The test split `mesh-vqvae.yaml` trains against, from its own config.

    Cached on the config path and overrides -- the split is seeded, so the same
    config always produces the same held-out buildings as the training run did.
    Overriding the corpus therefore also changes *which* buildings are held out,
    which is why the sidebar defaults to leaving it alone.
    """
    cfg = load_config(Path(config_path), list(overrides))
    dm = MeshDataModule(
        dataset_dir=cfg.mesh_data.dataset_dir,
        lod_in=cfg.mesh_data.lod_in,
        lod_out=cfg.mesh_data.lod_out,
        num_bins=cfg.mesh_data.num_bins,
        margin_lo=list(cfg.mesh_data.margin_lo),
        margin_hi=list(cfg.mesh_data.margin_hi),
        max_faces=cfg.mesh_data.max_faces,
        max_files=cfg.mesh_data.max_files,
        train_val_test_split=tuple(cfg.training.train_val_test_split),
        num_workers=0,
        persistent_workers=False,
        seed=cfg.seed,
    )
    dm.setup()
    return cfg, dm.test_dataset


@st.cache_resource(show_spinner="Loading checkpoint...")
def load_model(ckpt_path):
    # eval(): the codebook is an EMA buffer that moves on any training-mode
    # forward pass, so browsing would quietly edit the checkpoint's vocabulary.
    model = MeshVQVAEModule.load_from_checkpoint(ckpt_path, map_location="cpu")
    return model.eval()


@torch.no_grad()
def reconstruct(model, coords):
    """``(predicted bins [F, 9], metrics dict)`` for one mesh's coordinates."""
    net = model.network
    coords = coords[None]
    faces, counts = net.vertex_ids(coords)
    z_q, _, codes = net.quantize(net.encode(coords), faces, counts)
    pred = net.decode(z_q).argmax(-1)[0]
    err = (pred - coords[0]).abs().float()
    stats = net.codebook_stats(codes)
    return pred, {
        "bin_mae": err.mean().item(),
        "acc_1bin": (err <= 1).float().mean().item(),
        "exact": (err == 0).float().mean().item(),
        "codes_used": stats["distinct"],
        "codes_per_stage": stats["distinct_per_stage"],
    }


def chamfer_m(gen, gt, scale):
    """Chamfer between two normalized meshes, put back into metres.

    ``scale`` is per axis (the config's z margin makes it anisotropic), so the
    un-normalization is a broadcast multiply, not one number.
    """
    gen = (gen[0] * scale, gen[1])
    gt = (gt[0] * scale, gt[1])
    return chamfer_distance(*surface_distances(gen, gt))


st.title("VQ-VAE test-split reconstruction")

with st.sidebar:
    config_path = st.text_input("config", str(DEFAULT_CONFIG))
    # The config's corpus (`data/The Hague/mini`, LOD1_synth) is generated, not
    # committed, so a fresh clone has to be pointed at whatever it does have.
    with st.expander("corpus override"):
        over_dir = st.text_input("mesh_data.dataset_dir", "")
        over_in = st.text_input("mesh_data.lod_in", "")
        over_files = st.number_input("mesh_data.max_files (0 = all)", 0, value=0)
    overrides = tuple(f"mesh_data.{k}={v}" for k, v in
                      (("dataset_dir", over_dir), ("lod_in", over_in),
                       ("max_files", over_files or None)) if v)

    found = sorted(str(p) for p in (REPO / "outputs").rglob("checkpoints/*.ckpt"))
    if not found:
        st.error("No .ckpt under outputs/")
        st.stop()
    picked_ckpts = st.multiselect(
        "checkpoints", found, default=found[:1],
        format_func=lambda p: "/".join(Path(p).parts[-3:]))
    lod = st.radio("mesh", ["LOD2 (target)", "LOD1 (condition)"])

if overrides:
    # The split is seeded off the corpus's id list, so an override does not just
    # change which files are read -- it hands back a different held-out set than
    # the training run and the notebooks saw. Silent when it happened is how you
    # end up hunting for a building that was never in this split.
    st.warning("Corpus overridden: " + ", ".join(overrides) +
               " — this is NOT the split the checkpoints were trained against.")

try:
    cfg, split = test_split(config_path, overrides)
except (FileNotFoundError, ValueError) as exc:
    st.error(f"{exc}\n\nSet a corpus override in the sidebar.")
    st.stop()
if not len(split):
    st.error("Test split is empty — check train_val_test_split and the dataset dir.")
    st.stop()

ids = [split.dataset.ids[i] for i in split.indices]
name = st.selectbox("test building", range(len(ids)),
                    format_func=lambda i: f"{i}  ·  {ids[i]}")
item = split[name]

# Both views strip the specials `MeshDataset` adds; the condition carries none.
coords = (item["tgt"][1:-1] if lod.startswith("LOD2") else item["cond"]).reshape(-1, 9)
scale = item["scale"].numpy()

gt = detokenize(coords.numpy(), cfg.mesh_data.num_bins)
figs = [mesh_figure(*gt, color="#444444")]
titles = [f"ground truth · {len(gt[1])} faces"]

for ckpt in picked_ckpts:
    model = load_model(ckpt)
    label = "/".join(Path(ckpt).parts[-3:-1] + (Path(ckpt).stem,))
    if len(coords) > model.network.max_faces:
        st.warning(f"{label}: mesh has {len(coords)} faces, checkpoint's "
                   f"max_faces={model.network.max_faces} — skipped.")
        continue

    pred, metrics = reconstruct(model, coords)
    rec = detokenize(pred.numpy(), model.network.num_bins)
    figs.append(mesh_figure(*rec, color="#2a78d6"))
    titles.append(f"{Path(ckpt).stem} · {len(rec[1])} faces")

    cols = st.columns(5)
    cols[0].metric("chamfer (m)", f"{chamfer_m(rec, gt, scale):.4f}")
    cols[1].metric("bin MAE", f"{metrics['bin_mae']:.3f}")
    cols[2].metric("within 1 bin", f"{metrics['acc_1bin']:.1%}")
    cols[3].metric("exact bin", f"{metrics['exact']:.1%}")
    cols[4].metric("codes used", metrics["codes_used"],
                   help=f"per stage: {metrics['codes_per_stage']}")
    st.caption(f"↑ {label}")

if len(figs) > 1:
    st.plotly_chart(side_by_side([figs], titles, height=620), width="stretch")
else:
    st.plotly_chart(figs[0], width="stretch")

st.caption(f"{len(split)} test buildings · {cfg.mesh_data.dataset_dir} · "
           f"{cfg.mesh_data.num_bins} bins · scale {np.round(scale, 2).tolist()} m")
