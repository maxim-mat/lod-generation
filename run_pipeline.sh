#!/usr/bin/env bash
#
# Runs the whole mesh pipeline end to end, chaining each stage's checkpoint into
# the next:
#
#   stage 1   configs/mesh-vqvae.yaml      learn the vocabulary
#   stage 1b  configs/mesh-vqvae-nr.yaml   noise-resistant decoder fine-tune
#   stage 2   configs/mesh3-train.yaml     the transformer over codes
#
# Usage:
#   chmod +x run_pipeline.sh
#   nohup ./run_pipeline.sh > pipeline.out 2>&1 &
#
# Knobs (environment variables):
#   DATA_DIR    dataset for every stage, so the three cannot disagree about the
#               split. Default: whatever the configs say.
#   RUN_NR      1 (default) runs stage 1b; 0 sends stage 1 straight to stage 2.
#   SMOKE       1 = one epoch on a handful of files. Run this first: it catches
#               a config or path error in minutes instead of at hour six.
#   STAGE1_CKPT skip stage 1 and start from this checkpoint.
#
set -euo pipefail
cd "$(dirname "$0")"

RUN_NR="${RUN_NR:-1}"
SMOKE="${SMOKE:-0}"
STAGE1_CKPT="${STAGE1_CKPT:-}"

LOG_DIR="logs/pipeline-$(date +%Y%m%d-%H%M%S)"
mkdir -p "$LOG_DIR"

# Overrides applied to every stage. Keeping the dataset identical across stages
# is not cosmetic: the split is seeded from it, and a building in the
# transformer's test set must never have trained the tokenizer.
COMMON=()
if [[ -n "${DATA_DIR:-}" ]]; then
  COMMON+=("mesh_data.dataset_dir=$DATA_DIR")
fi
if [[ "$SMOKE" == "1" ]]; then
  COMMON+=("training.max_epochs=1" "mesh_data.max_files=20" "mesh_eval.enabled=false")
  LOG_DIR="$LOG_DIR-smoke"; mkdir -p "$LOG_DIR"
fi
# `set -u` treats "${COMMON[@]}" on an empty array as unbound on bash < 4.4, so
# expansions use this guarded form. It has to stay an array expansion, not a
# string: the default dataset path contains a space ("data/The Hague/mini").

log() { echo "[pipeline $(date +%H:%M:%S)] $*"; }

# "<run dir>\t<checkpoint mode>" as the training code will actually resolve
# them -- read through the project's own loader so an override here cannot
# drift from it. The mode matters: stage 2 monitors val_token_acc (max) while
# the stage-1 runs monitor val_recon_ce (min), so "best" is not always smallest.
run_info() {
  python - "$@" <<'PY'
import sys
from pathlib import Path
from src.utils.initialization import load_config
cfg = load_config(Path(sys.argv[1]), list(sys.argv[2:]))
d = Path(cfg.logging.save_dir, cfg.logging.experiment_name, cfg.logging.run_name)
print(f"{d.as_posix()}\t{cfg.training.checkpoint.mode}")
PY
}

# Best checkpoint in a run directory. The monitored metric is embedded in the
# filename ({epoch:02d}-{metric:.4f}.ckpt), so the winner is the smallest or
# largest trailing number depending on the config's checkpoint mode. last.ckpt
# is excluded -- Lightning always writes it, and it is the final epoch rather
# than the best one.
best_ckpt() {
  local dir="$1/checkpoints" mode="${2:-min}" order picked
  if [[ "$mode" == "max" ]]; then order="-gr"; else order="-g"; fi
  picked=$(find "$dir" -maxdepth 1 -name '*.ckpt' ! -name 'last.ckpt' 2>/dev/null \
    | sed -E 's|.*-([0-9]+\.[0-9]+)\.ckpt$|\1 &|' \
    | grep -E '^[0-9]' | sort "$order" | head -1 | cut -d' ' -f2-)
  if [[ -z "$picked" ]]; then
    # No metric-named checkpoint (early stop before the first save, or a
    # renamed template). last.ckpt is the honest fallback, but say so loudly.
    picked="$dir/last.ckpt"
    log "WARNING: no metric-named checkpoint in $dir, falling back to last.ckpt"
    [[ -f "$picked" ]] || { log "FATAL: no checkpoint at all in $dir"; exit 1; }
  fi
  echo "$picked"
}

stage() {
  local name="$1" config="$2"; shift 2
  log "=== $name : $config ${*:-}"
  # pipefail is on, so a training failure here aborts the pipeline rather than
  # feeding a missing checkpoint to the next stage.
  python main.py --train --config "$config" \
    ${COMMON[@]+"${COMMON[@]}"} "$@" 2>&1 | tee "$LOG_DIR/$name.log"
  log "=== $name done"
}

log "logs -> $LOG_DIR   (RUN_NR=$RUN_NR SMOKE=$SMOKE)"
if [[ ${#COMMON[@]} -gt 0 ]]; then log "common overrides: ${COMMON[*]}"; fi

# ---------------------------------------------------------------- stage 1
if [[ -n "$STAGE1_CKPT" ]]; then
  VQVAE_CKPT="$STAGE1_CKPT"
  log "skipping stage 1, using $VQVAE_CKPT"
else
  stage stage1-vqvae configs/mesh-vqvae.yaml
  info=$(run_info configs/mesh-vqvae.yaml ${COMMON[@]+"${COMMON[@]}"})
  VQVAE_CKPT=$(best_ckpt "${info%%$'	'*}" "${info##*$'	'}")
  log "stage 1 best checkpoint: $VQVAE_CKPT"
fi

# ---------------------------------------------------------------- stage 1b
if [[ "$RUN_NR" == "1" ]]; then
  stage stage1b-noise-resistant configs/mesh-vqvae-nr.yaml \
    "mesh_vqvae.init_from=$VQVAE_CKPT"
  # Leaves the encoder and codebook untouched, so this is a drop-in for the
  # stage-1 checkpoint -- only its decoder differs.
  info=$(run_info configs/mesh-vqvae-nr.yaml ${COMMON[@]+"${COMMON[@]}"})
  VQVAE_CKPT=$(best_ckpt "${info%%$'	'*}" "${info##*$'	'}")
  log "stage 1b best checkpoint: $VQVAE_CKPT"
fi

# ---------------------------------------------------------------- stage 2
stage stage2-transformer configs/mesh3-train.yaml \
  "mesh_model.vqvae_ckpt=$VQVAE_CKPT"

log "pipeline complete. tokenizer: $VQVAE_CKPT"
info=$(run_info configs/mesh3-train.yaml ${COMMON[@]+"${COMMON[@]}"})
log "stage 2 run dir: ${info%%$'	'*}  (best = ${info##*$'	'} of val_token_acc)"
