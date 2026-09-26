#!/usr/bin/env bash
# Candidate-mode experiments (new entities, train 2019, test 2019-2022):
# text-only control, paper-strict and paper+extranneg. Each run trains into a
# staging dir (train.py names run dirs after the config file), is renamed,
# then evaluated. Every job waits for enough free GPU memory (shared GPU).
#   PYTHON=... CYCLE_BERT_PATH=... bash scripts/run_candidate.sh
set -u
cd "$(dirname "$0")/.."
SP=scripts
export PYTHON="${PYTHON:-python}"
export CYCLE_BERT_PATH="${CYCLE_BERT_PATH:-}"   # local BERT snapshot if the HF cache is offline
mkdir -p outputs/logs
STATUS=outputs/logs/gpu_queue.status; MARK=outputs/logs/.last_launch
COMMON=(--set train.num_epochs=12 --set train.early_stopping.enabled=true
        --set train.early_stopping.patience=3 --set data.debug_n=1024 --set data.valid_n=256)

run() {  # run NAME extra-overrides...
  local NAME="$1"; shift
  local STAGE="outputs/_staging_$NAME" FINAL="outputs/new_2019_${NAME}_seed52313"
  bash "$SP/gpu_queue.sh" "TRAIN_$NAME" 4096 "$STATUS" "$MARK" -- \
    bash scripts/train.sh configs/paper.yaml new 2019 52313 "${COMMON[@]}" \
      --set run.output_dir="$STAGE" "$@" > "outputs/logs/train_new_2019_$NAME.log" 2>&1 || return 1
  grep -q '"type": "done"' "$STAGE/new_2019_paper_seed52313/metrics.jsonl" || { echo "$NAME train failed" >> "$STATUS"; return 1; }
  mv "$STAGE/new_2019_paper_seed52313" "$FINAL" && rmdir "$STAGE"
  bash "$SP/gpu_queue.sh" "EVAL_$NAME" 14336 "$STATUS" "$MARK" -- \
    bash scripts/evaluate.sh "$FINAL" 2019,2020,2021,2022 > "outputs/logs/eval_new_2019_$NAME.log" 2>&1
}

run textonly                  --set graph.enabled=false --set fusion.enabled=false
run paper_candidate           --set fusion.mode=candidate
run paper_candidate_extranneg --set fusion.mode=candidate --set graph.relation_extra_negatives=8
echo "$(date +%T) ALL_CANDIDATE_RUNS DONE" >> "$STATUS"
