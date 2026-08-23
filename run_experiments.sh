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

SMOKE="${SMOKE:-0}"
KEEP_GOING="${KEEP_GOING:-1}"
DATA_DIR="${DATA_DIR:-}"
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
    -h|--help)    usage; exit 0 ;;
    -*)           usage >&2; die "unknown option: $1" ;;
    *)            CONFIGS+=("$1"); shift ;;
  esac
done
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
  log "=== start $name ($cfg)"
  start=$SECONDS
  # `set -e` is deliberately off: a non-zero exit here is recorded and the queue
  # continues. pipefail is on, so the exit status is python's, not tee's.
  if python main.py --train --config "$cfg" \
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
