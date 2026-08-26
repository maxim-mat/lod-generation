#!/usr/bin/env bash
#
# Runs a list of single-stage training configs one after another, each into its
# own wandb run (the run name comes from each config's logging.run_name, so they
# cannot collide).
#
# Not run_pipeline.sh: that chains three *dependent* stages, feeding one
# checkpoint into the next. These four are independent arms of one comparison,
# so the useful behaviour is the opposite -- a failure in arm 2 must not stop
# arms 3 and 4 from running overnight. Hence --keep-going, on by default.
#
# Usage:
#   ./run_experiments.sh --smoke                 # verify every config, minutes
#   nohup ./run_experiments.sh > experiments.out 2>&1 &
#   ./run_experiments.sh configs/mesh-v3-amt.yaml   # just one
#
set -uo pipefail
cd "$(dirname "$0")"

# The default queue: four arms varying only tokenization and backbone.
# Cheapest first, so a config-level mistake surfaces before the long runs.
DEFAULT_CONFIGS=(
  configs/mesh-v3-coord.yaml
  configs/mesh-v3-amt.yaml
  configs/mesh-v3-opt-coord.yaml
  configs/mesh-v3-opt-amt.yaml
)

# The mesh-diffusion grid, in stages. Stage 1 decides denoiser and loss;
# stage 2 decides the objective on stage 1's winner; stage 3 adds the two
# sampling-time levers. Run one stage, read it, then edit the next stage's
# configs to match the winner -- they ship set to arm a1's axes, not to a
# placeholder, so an unedited stage-2 launch is a valid run rather than a
# crash. It just answers a slightly different question.
DIFF_STAGE1=(
  configs/mesh-diff-a1-unet-mse.yaml
  configs/mesh-diff-a2-tf-mse.yaml
  configs/mesh-diff-a3-tf-hungarian.yaml
)
# Cheapest first inside each stage, so a config-level mistake surfaces before
# the long runs. Within stage 2 that means the continuous arms before the
# 1153-channel one-hot arm.
DIFF_STAGE2=(
  configs/mesh-diff-b1-x0.yaml
  configs/mesh-diff-b2-flow.yaml
  configs/mesh-diff-b3-quant-mse.yaml
  configs/mesh-diff-b4-quant-ce.yaml
  configs/mesh-diff-b6-d3pm-uniform.yaml
  configs/mesh-diff-b7-d3pm-gauss.yaml
  configs/mesh-diff-b5-onehot-ce.yaml
)
# c3 is conditional on stage 2's winner being b4 or b5 -- see its config header.
DIFF_STAGE3=(
  configs/mesh-diff-c1-scaffold.yaml
  configs/mesh-diff-c2-guidance.yaml
  configs/mesh-diff-c3-clamp-soft.yaml
)
# The revisit pass. The grid is greedy -- stage 1 ranks denoiser and loss under
# ONE objective and stage 2 assumes that ranking transfers to six others -- and
# this is the cheap check on that assumption, not a full factorial: re-run the
# stage-1 alternatives under stage 2's winner. Two runs against the ~8 a full
# cross would add. r2 is conditional: a `ce` winner has no unordered form
# (plan D4), so the order axis cannot reach b4-b7 at all. See its config header.
DIFF_REVISIT=(
  configs/mesh-diff-r1-denoiser.yaml
  configs/mesh-diff-r2-loss.yaml
)
DIFF_BASE="configs/mesh-diff-base.yaml"

SMOKE="${SMOKE:-0}"
KEEP_GOING="${KEEP_GOING:-1}"
DATA_DIR="${DATA_DIR:-}"
DIFF_STAGE=""
CONFIGS=()

usage() {
  cat <<'USAGE'
Usage: ./run_experiments.sh [options] [CONFIG ...]

  CONFIG ...        configs to run, in order. Defaults to the four mesh-v3 arms.

      --smoke       1 epoch on 20 files, no logger, outputs to a scratch dir
                    that is deleted on exit. Run the real command with --smoke
                    added: it catches a bad path or key in minutes rather than
                    at hour six, and it verifies the configs you are about to
                    commit to. No wandb run is created.
      --data DIR    override mesh_data.dataset_dir for every config, so the
                    arms cannot silently disagree about the dataset.
      --stop-early  abort the queue on the first failure (default: carry on,
                    because these arms are independent).
      --diff-stage N
                    run stage N of the mesh-diffusion grid instead of the
                    default AR queue. N is 1, 2, 3, or `revisit`.
                    `revisit` re-runs stage 1's alternatives under stage 2's
                    winner: the cheap check that the greedy search did not pick
                    a denoiser or a loss that only won under stage 1's own
                    objective. Each arm config carries only the axes it sets,
                    so it is merged over configs/mesh-diff-base.yaml first --
                    that base is what makes every arm share a dataset, split,
                    effective batch and seed with the others and with the
                    mesh-v3 AR arms.
  -h, --help

Every option has an environment-variable form (SMOKE, DATA_DIR, KEEP_GOING).
USAGE
}

die() { echo "run_experiments: $*" >&2; exit 2; }
need() { [[ $# -ge 2 ]] || die "$1 requires a value"; }
log() { echo "[experiments $(date +%H:%M:%S)] $*" >&2; }

while [[ $# -gt 0 ]]; do
  case "$1" in
    --smoke)      SMOKE=1; shift ;;
    --stop-early) KEEP_GOING=0; shift ;;
    --data)       need "$@"; DATA_DIR="$2"; shift 2 ;;
    --diff-stage) need "$@"; DIFF_STAGE="$2"; shift 2 ;;
    -h|--help)    usage; exit 0 ;;
    -*)           usage >&2; die "unknown option: $1" ;;
    *)            CONFIGS+=("$1"); shift ;;
  esac
done

if [[ -n "$DIFF_STAGE" ]]; then
  [[ ${#CONFIGS[@]} -eq 0 ]] || die "--diff-stage selects its own queue; do not also name configs"
  case "$DIFF_STAGE" in
    1) CONFIGS=("${DIFF_STAGE1[@]}") ;;
    2) CONFIGS=("${DIFF_STAGE2[@]}") ;;
    3) CONFIGS=("${DIFF_STAGE3[@]}") ;;
    revisit|r) CONFIGS=("${DIFF_REVISIT[@]}") ;;
    *) die "--diff-stage must be 1, 2, 3 or revisit (got '$DIFF_STAGE')" ;;
  esac
  [[ -f "$DIFF_BASE" ]] || die "no such config: $DIFF_BASE"
fi
[[ ${#CONFIGS[@]} -gt 0 ]] || CONFIGS=("${DEFAULT_CONFIGS[@]}")

# Check every config before running any of them. Finding a typo in arm 4 after
# arms 1-3 have run for a day is the failure this prevents.
for cfg in "${CONFIGS[@]}"; do
  [[ -f "$cfg" ]] || die "no such config: $cfg"
done

COMMON=()
[[ -n "$DATA_DIR" ]] && COMMON+=("mesh_data.dataset_dir=$DATA_DIR")

LOG_DIR="logs/experiments-$(date +%Y%m%d-%H%M%S)"
if [[ "$SMOKE" == "1" ]]; then
  # Everything the run would persist goes to a scratch directory removed on
  # exit: a smoke run must not drop a 1-epoch checkpoint where the real run
  # will write, and must not create a wandb run under the same name.
  # `logging.loggers=[]` makes create_loggers return nothing, which the Trainer
  # reads as logger=False.
  SMOKE_OUT=$(mktemp -d)
  trap 'rm -rf "$SMOKE_OUT"' EXIT
  COMMON+=("training.max_epochs=1" "mesh_data.max_files=20" "mesh_eval.enabled=false"
           "logging.save_dir=$SMOKE_OUT" "logging.loggers=[]")
  # The diffusion branch's test-end eval is NOT epoch-gated -- it runs once,
  # unconditionally, and at the shipped n_test=64 x eval_steps=50 that is 3200
  # reverse-diffusion forward passes, which is eight minutes per arm and turns
  # a thirteen-arm smoke into an afternoon. Shrink it: a smoke asks whether the
  # sampler runs at all, not how well it scores. Harmless for the AR configs,
  # which have no mesh_diffusion block of their own to read.
  COMMON+=("mesh_diffusion.n_test=4" "mesh_diffusion.n_val=4"
           "mesh_diffusion.eval_steps=5" "mesh_diffusion.eval_batch_size=4"
           "mesh_diffusion.save_samples=1")
  # And no dataloader workers. On Windows each one is a fresh process importing
  # torch, respawned for fit/validate/test; at 20 files that spawn cost was
  # measured at 6 of the 8 minutes an arm took, against 30s with workers off.
  # Real runs keep the configured count -- this is a smoke-only trade.
  COMMON+=("mesh_data.num_workers=0")
  LOG_DIR="$LOG_DIR-smoke"
fi
mkdir -p "$LOG_DIR"

log "queue (${#CONFIGS[@]}): ${CONFIGS[*]}"
log "logs -> $LOG_DIR"
[[ ${#COMMON[@]} -gt 0 ]] && log "common overrides: ${COMMON[*]}"
[[ "$SMOKE" == "1" ]] && log "SMOKE: 1 epoch, 20 files, no logger; outputs -> $SMOKE_OUT (deleted on exit)"

declare -a NAMES=() RESULTS=() SECS=()
failed=0

for cfg in "${CONFIGS[@]}"; do
  name=$(basename "$cfg" .yaml)
  run_cfg="$cfg"
  if [[ -n "$DIFF_STAGE" ]]; then
    # Arm files carry only the axes they set. `load_config` takes one --config,
    # so the base is merged in here rather than by inventing a CLI flag for it.
    run_cfg="$LOG_DIR/$name.merged.yaml"
    if ! python - "$DIFF_BASE" "$cfg" > "$run_cfg" <<'PY'
import sys
from omegaconf import OmegaConf
print(OmegaConf.to_yaml(OmegaConf.merge(
    OmegaConf.load(sys.argv[1]), OmegaConf.load(sys.argv[2]))))
PY
    then
      die "failed to merge $cfg over $DIFF_BASE"
    fi
  fi
  log "=== start $name ($run_cfg)"
  start=$SECONDS
  # `set -e` is deliberately off: a non-zero exit here is recorded and the queue
  # continues. pipefail is on, so the exit status is python's, not tee's.
  if python main.py --train --config "$run_cfg" \
       ${COMMON[@]+"${COMMON[@]}"} 2>&1 | tee "$LOG_DIR/$name.log"; then
    status=ok
  else
    status="FAILED(${PIPESTATUS[0]})"
    failed=$((failed + 1))
    log "!!! $name failed -- see $LOG_DIR/$name.log"
    if [[ "$KEEP_GOING" != "1" ]]; then
      NAMES+=("$name"); RESULTS+=("$status"); SECS+=($((SECONDS - start)))
      log "--stop-early: aborting the queue"
      break
    fi
  fi
  NAMES+=("$name"); RESULTS+=("$status"); SECS+=($((SECONDS - start)))
  log "=== done $name : $status ($((SECONDS - start))s)"
done

echo
log "summary"
for i in "${!NAMES[@]}"; do
  printf '  %-24s %-14s %6ss\n' "${NAMES[$i]}" "${RESULTS[$i]}" "${SECS[$i]}" >&2
done
log "$((${#NAMES[@]} - failed))/${#NAMES[@]} succeeded; logs in $LOG_DIR"
exit $(( failed > 0 ? 1 : 0 ))
