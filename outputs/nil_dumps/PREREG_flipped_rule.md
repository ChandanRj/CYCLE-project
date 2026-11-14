# Pre-registration: flipped max rule (written before running)

- Rule: predict NIL if top-1 score >= threshold; otherwise link to top-1.
- Data: KB 2019, run outputs/new_2019_default_seed52313, dumps default_kb2019_y2020_{validation,test}.npz.
- Calibration: on 2020 validation only. Objective nil_f1 (cycle.nil.metrics).
  Candidate thresholds: unique validation top-1 scores, subsampled to 512 quantiles
  (same scheme as cycle.nil_eval.calibrate_topk), plus +inf (never NIL).
  Ties: keep the largest threshold (fewest NIL predictions).
- Evaluation: apply that single threshold to 2020 test, once. Report nil_precision,
  nil_recall, nil_f1, unnormalized_acc (+ inkb_acc, link_rate).
- No tuning on test; no change of rule, objective, candidates or tie-break after results.
- Status: the flip direction was chosen after seeing AUC < 0.5 on these dumps -> post hoc.
