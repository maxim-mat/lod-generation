#!/usr/bin/env bash
#
# Runs the mesh pipeline end to end, chaining each stage's checkpoint into the
# next:
#
#   stage 1   learn the vocabulary            (configs/mesh-vqvae.yaml)
#   stage 1b  noise-resistant decoder         (configs/mesh-vqvae-nr.yaml)
#   stage 2   the transformer over codes      (configs/mesh4-train.yaml)
#
# The configs are required, with no defaults, so that a --smoke run is the real
# command with one flag added and therefore verifies the configs you are about
# to commit hours to. A default would let you smoke-test one set and launch
# another.
#
# Usage:
#   ./run_pipeline.sh --smoke -1 configs/mesh-vqvae.yaml -n configs/mesh-vqvae-nr.yaml -2 configs/mesh4-train.yaml
#   nohup ./run_pipeline.sh -1 configs/mesh-vqvae.yaml -n configs/mesh-vqvae-nr.yaml -2 configs/mesh4-train.yaml > pipeline.out 2>&1 &
#
set -euo pipefail
cd "$(dirname "$0")"

STAGE1_CFG="${STAGE1_CFG:-}"
NR_CFG="${NR_CFG:-}"
STAGE2_CFG="${STAGE2_CFG:-}"
RUN_NR="${RUN_NR:-1}"
SMOKE="${SMOKE:-0}"
STAGE1_CKPT="${STAGE1_CKPT:-}"
DATA_DIR="${DATA_DIR:-}"

usage() {
  cat <<'USAGE'
Usage: ./run_pipeline.sh -1 STAGE1 -n STAGE1B -2 STAGE2 [options]

  -1, --stage1 FILE     stage 1 config   (required unless --stage1-ckpt)
  -n, --nr FILE         stage 1b config  (required unless --skip-nr)
  -2, --stage2 FILE     stage 2 config   (required)

      --skip-nr         skip the noise-resistant fine-tune; stage 1's
                        checkpoint goes straight to stage 2
      --stage1-ckpt F   skip stage 1 and start from this checkpoint
      --data DIR        dataset for every stage, so the three cannot disagree
                        about the seeded split
      --smoke           one epoch on a handful of files, into a scratch
                        directory that is deleted on exit, with no logger
                        attached. Run the real command with --smoke added: it
                        catches a config or path error in minutes rather than
                        at hour six, and verifies the configs you are actually
                        about to launch.
  -h, --help

Examples:
  ./run_pipeline.sh --smoke -1 configs/mesh-vqvae.yaml -n configs/mesh-vqvae-nr.yaml -2 configs/mesh4-train.yaml
  nohup ./run_pipeline.sh -1 configs/mesh-vqvae.yaml -n configs/mesh-vqvae-nr.yaml -2 configs/mesh4-train.yaml > pipeline.out 2>&1 &

Every option also has an environment-variable form (STAGE1_CFG, NR_CFG,
STAGE2_CFG, RUN_NR, STAGE1_CKPT, DATA_DIR, SMOKE); the flags win.
USAGE
}

die() { echo "run_pipeline: $*" >&2; exit 2; }
need() { [[ $# -ge 2 ]] || die "$1 requires a value"; }

while [[ $# -gt 0 ]]; do
  case "$1" in
    -1|--stage1)     need "$@"; STAGE1_CFG="$2";  shift 2 ;;
    -n|--nr)         need "$@"; NR_CFG="$2";      shift 2 ;;
    -2|--stage2)     need "$@"; STAGE2_CFG="$2";  shift 2 ;;
    --stage1-ckpt)   need "$@"; STAGE1_CKPT="$2"; shift 2 ;;
    --data)          need "$@"; DATA_DIR="$2";    shift 2 ;;
    --skip-nr)       RUN_NR=0; shift ;;
    --smoke)         SMOKE=1;  shift ;;
    -h|--help)       usage; exit 0 ;;
    *)               usage >&2; die "unknown option: $1" ;;
  esac
done

# Require, then check, every config that will actually be used -- now, rather
# than after stage 1 has run for six hours. --stage1-ckpt makes the stage-1
# config moot and --skip-nr makes the stage-1b one moot, so neither is demanded
# when it would go unread.
require_cfg() {
  local flag="$1" path="$2" what="$3"
  [[ -n "$path" ]] || die "$what config is required: pass $flag FILE (see --help)"
  [[ -f "$path" ]] || die "no such config: $path"
}
if [[ -z "$STAGE1_CKPT" ]]; then require_cfg "-1" "$STAGE1_CFG" "stage 1"; fi
if [[ "$RUN_NR" == "1" ]]; then require_cfg "-n" "$NR_CFG" "stage 1b"; fi
require_cfg "-2" "$STAGE2_CFG" "stage 2"

LOG_DIR="logs/pipeline-$(date +%Y%m%d-%H%M%S)"

# Overrides applied to every stage. Keeping the dataset identical across stages
# is not cosmetic: the split is seeded from it, and a building in the
# transformer's test set must never have trained the tokenizer.
COMMON=()
if [[ -n "$DATA_DIR" ]]; then
  COMMON+=("mesh_data.dataset_dir=$DATA_DIR")
fi
if [[ "$SMOKE" == "1" ]]; then
  # Everything the run would persist goes to a scratch directory that is
  # removed on exit, success or failure: a smoke run must not drop a 1-epoch
  # checkpoint into the directory the real run will use, and must not create a
  # wandb run under the same name. `logging.loggers=[]` makes create_loggers
  # return nothing, which the Trainer takes as logger=False.
  SMOKE_OUT=$(mktemp -d)
  trap 'rm -rf "$SMOKE_OUT"' EXIT
  COMMON+=("training.max_epochs=1" "mesh_data.max_files=20" "mesh_eval.enabled=false"
           "logging.save_dir=$SMOKE_OUT" "logging.loggers=[]")
  LOG_DIR="$LOG_DIR-smoke"
fi
mkdir -p "$LOG_DIR"
# `set -u` treats "${COMMON[@]}" on an empty array as unbound on bash < 4.4, so
# expansions use the guarded ${COMMON[@]+"${COMMON[@]}"} form. It has to stay an
# array expansion: the dataset path contains a space ("data/The Hague/mini").

# Diagnostics go to stderr, always. `best_ckpt` and `pick` return their value on
# stdout and are called inside $( ), so a log line on stdout is silently
# captured into the checkpoint path -- which then reached OmegaConf as
# `mesh_vqvae.init_from=[pipeline 17:28:03] WARNING: ...` and died in the YAML
# parser. Keeping stdout for values only makes that unrepresentable.
log() { echo "[pipeline $(date +%H:%M:%S)] $*" >&2; }

# "<run dir>\t<mode>\t<monitor>" as the training code will actually resolve
# them -- read through the project's own loader so an override here cannot drift
# from it. The mode matters: stage 2 monitors val_token_acc (max) while the
# stage-1 runs monitor a cross-entropy (min), so "best" is not always smallest.
run_info() {
  python - "$@" <<'PY'
import sys
from pathlib import Path
from src.utils.initialization import load_config
cfg = load_config(Path(sys.argv[1]), list(sys.argv[2:]))
d = Path(cfg.logging.save_dir, cfg.logging.experiment_name, cfg.logging.run_name)
ck = cfg.training.checkpoint
print(f"{d.as_posix()}\t{ck.mode}\t{ck.monitor}")
PY
}

# Best checkpoint in a run directory. The monitored metric is embedded in the
# filename ({epoch:02d}-{metric:.4f}.ckpt), so the winner is the smallest or
# largest trailing number depending on the config's checkpoint mode. last.ckpt
# is excluded -- Lightning always writes it, and it is the final epoch rather
# than the best one.
best_ckpt() {
  local dir="$1/checkpoints" mode="${2:-min}" monitor="${3:-the monitored metric}" order picked
  if [[ "$mode" == "max" ]]; then order="-gr"; else order="-g"; fi
  picked=$(find "$dir" -maxdepth 1 -name '*.ckpt' ! -name 'last.ckpt' 2>/dev/null \
    | sed -E 's|.*-([0-9]+\.[0-9]+)\.ckpt$|\1 &|' \
    | grep -E '^[0-9]' | sort "$order" | head -1 | cut -d' ' -f2-)
  if [[ -z "$picked" ]]; then
    # No metric-named checkpoint. Usually means validation never produced the
    # monitored metric, so ModelCheckpoint saved nothing but last.ckpt -- worth
    # looking into rather than shrugging at, so list what is actually there.
    picked="$dir/last.ckpt"
    log "WARNING: no metric-named checkpoint in $dir"
    log "WARNING: found: $(find "$dir" -maxdepth 1 -name '*.ckpt' -printf '%f ' 2>/dev/null || echo '(nothing)')"
    log "WARNING: check the stage log -- did validation run and log $monitor?"
    log "WARNING: falling back to last.ckpt"
    [[ -f "$picked" ]] || { log "FATAL: no checkpoint at all in $dir"; exit 1; }
  fi
  echo "$picked"
}

# A finished stage to its best checkpoint, logging how it was chosen. Logs go to
# stderr so the checkpoint path is the only thing on stdout for the caller.
pick() {
  local label="$1" config="$2" info dir mode monitor ckpt
  info=$(run_info "$config" ${COMMON[@]+"${COMMON[@]}"})
  dir=${info%%$'\t'*}; info=${info#*$'\t'}
  mode=${info%%$'\t'*}; monitor=${info#*$'\t'}
  ckpt=$(best_ckpt "$dir" "$mode" "$monitor")
  log "$label best checkpoint ($mode of $monitor): $ckpt"
  echo "$ckpt"
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

log "logs -> $LOG_DIR"
if [[ "$SMOKE" == "1" ]]; then
  log "SMOKE: 1 epoch, 20 files, no logger; outputs -> $SMOKE_OUT (deleted on exit)"
fi
log "stage 1 : ${STAGE1_CKPT:-$STAGE1_CFG}"
if [[ "$RUN_NR" == "1" ]]; then log "stage 1b: $NR_CFG"; else log "stage 1b: skipped"; fi
log "stage 2 : $STAGE2_CFG"
if [[ ${#COMMON[@]} -gt 0 ]]; then log "common overrides: ${COMMON[*]}"; fi

# ---------------------------------------------------------------- stage 1
if [[ -n "$STAGE1_CKPT" ]]; then
  VQVAE_CKPT="$STAGE1_CKPT"
  log "skipping stage 1, using $VQVAE_CKPT"
else
  stage stage1-vqvae "$STAGE1_CFG"
  VQVAE_CKPT=$(pick "stage 1" "$STAGE1_CFG")
fi

# ---------------------------------------------------------------- stage 1b
if [[ "$RUN_NR" == "1" ]]; then
  stage stage1b-noise-resistant "$NR_CFG" "mesh_vqvae.init_from=$VQVAE_CKPT"
  # Leaves the encoder and codebook untouched, so this is a drop-in for the
  # stage-1 checkpoint -- only its decoder differs.
  VQVAE_CKPT=$(pick "stage 1b" "$NR_CFG")
fi

# ---------------------------------------------------------------- stage 2
stage stage2-transformer "$STAGE2_CFG" "mesh_model.vqvae_ckpt=$VQVAE_CKPT"

log "pipeline complete. tokenizer: $VQVAE_CKPT"
pick "stage 2" "$STAGE2_CFG" > /dev/null
