#!/usr/bin/env bash
# Negative-mining / similarity experiments (new entities, train 2019, test
# 2019-2022, candidate mode, relation_extra_negatives=8), three arms:
#   extranneg         random negatives + dot product (paper loss; seed 52313 exists)
#   extranneg_cos     random negatives + cosine InfoNCE
#   extranneg_hardcos hard-mined negatives + cosine InfoNCE
# Same staging/queue pattern as run_candidate.sh. Evaluation uses
# --unmapped-adjustment mean; each run also gets a collapse measurement.
# OUT_ROOT (default outputs) holds staging and finished runs; when it is
# elsewhere (e.g. /DATA, to keep checkpoints off a full /home) each finished
# run is symlinked back to outputs/<run>.
#   PYTHON=... CYCLE_BERT_PATH=... [OUT_ROOT=...] bash scripts/run_mining.sh ARM:SEED [ARM:SEED ...]
set -u
cd "$(dirname "$0")/.."
SP=scripts
export PYTHON="${PYTHON:-python}"
export CYCLE_BERT_PATH="${CYCLE_BERT_PATH:-}"
export PYTHONPATH="src${PYTHONPATH:+:$PYTHONPATH}"
mkdir -p outputs/logs
OUT_ROOT="${OUT_ROOT:-outputs}"
mkdir -p "$OUT_ROOT"
STATUS=outputs/logs/gpu_queue.status; MARK=outputs/logs/.last_launch
COMMON=(--set train.num_epochs=12 --set train.early_stopping.enabled=true
        --set train.early_stopping.patience=3 --set data.debug_n=1024 --set data.valid_n=256
        --set fusion.mode=candidate --set graph.relation_extra_negatives=8)

run() {  # run ARM SEED
  local ARM="$1" SEED="$2" EXTRA=()
  case "$ARM" in
    extranneg)         ;;
    extranneg_cos)     EXTRA=(--set graph.similarity=cosine) ;;
    extranneg_hardcos) EXTRA=(--set graph.similarity=cosine --set graph.negative_mining=hard) ;;
    *) echo "unknown arm $ARM" >&2; return 1 ;;
  esac
  local NAME="paper_candidate_${ARM}_seed${SEED}"
  local STAGE="$OUT_ROOT/_staging_$NAME" FINAL="outputs/new_2019_$NAME"
  local REAL="$OUT_ROOT/new_2019_$NAME"
  [ -e "$FINAL" ] && { echo "$FINAL exists, skipping" >> "$STATUS"; return 0; }
  bash "$SP/gpu_queue.sh" "TRAIN_$NAME" 4096 "$STATUS" "$MARK" -- \
    bash scripts/train.sh configs/paper.yaml new 2019 "$SEED" "${COMMON[@]}" "${EXTRA[@]}" \
      --set run.output_dir="$STAGE" > "outputs/logs/train_new_2019_$NAME.log" 2>&1 || return 1
  grep -q '"type": "done"' "$STAGE/new_2019_paper_seed$SEED/metrics.jsonl" || { echo "$NAME train failed" >> "$STATUS"; return 1; }
  mv "$STAGE/new_2019_paper_seed$SEED" "$REAL" && rmdir "$STAGE"
  [ "$REAL" -ef "$FINAL" ] || ln -s "$(realpath "$REAL")" "$FINAL"
  bash "$SP/gpu_queue.sh" "EVAL_$NAME" 14336 "$STATUS" "$MARK" -- \
    bash scripts/evaluate.sh "$FINAL" 2019,2020,2021,2022 --unmapped-adjustment mean \
      > "outputs/logs/eval_new_2019_$NAME.log" 2>&1
  "$PYTHON" "$SP/check_embedding_collapse.py" --run-dir "$FINAL" > "$FINAL/collapse.json" 2>/dev/null
}

for spec in "$@"; do run "${spec%%:*}" "${spec##*:}"; done
echo "$(date +%T) RUN_MINING DONE: $*" >> "$STATUS"
