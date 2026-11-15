# NIL flipped-rule check, second seed (52314)

Copy of `outputs/nil_dumps_seed52314/NIL_SUMMARY_SEED2.md` (on /DATA), kept here so it can be committed.

Run: `outputs/new_2019_default_seed52314`, a link to `/DATA/kmpooja/cycle_outputs/new_2019_default_seed52314`. Config `default.yaml`, new entities, train year 2019, **seed 52314**. Date: 2026-11-14.

**Launch:** the same command as the seed-52313 run in `overnight/chain_default.sh`, with only the seed changed:
`bash scripts/train.sh configs/default.yaml new 2019 52314 --set run.output_dir=/DATA/kmpooja/cycle_outputs`

The chain script is `/DATA/kmpooja/cycle_outputs/seed52314/chain_seed52314.sh`, and its step log is `seed52314/queue_status.txt`. The steps were:
1. Smoke test.
2. Train.
3. `evaluate.sh` on 2019–2022.
4. Eight runs of `dump_topk.py` (KB 2019, k = 64, validation and test for each year).
5. `scripts/flipped_rule_check.py --year Y --dump-dir outputs/nil_dumps_seed52314`.

The chain stops at the first failing step.

## Caveats

- **Post hoc rule:** NIL if top-1 score ≥ threshold. The direction was chosen after seeing the seed-52313 AUC. This seed is the first check of it on a different model.
- **Frozen procedure, unchanged:** `outputs/nil_dumps/PREREG_flipped_rule.md` (sha256 `7cba61ae…`) and `scripts/flipped_rule_check.py` (sha256 `b8254992…`) have the same hashes before and after the run (`sha256sum -c` OK). Each threshold was fitted on that year's validation dump only and applied once to that year's test dump. Nothing was tuned on test.
- **Bi-encoder only:** `scripts/dump_topk.py` is bi-encoder only and **not candidate-fused**.
- **2019 is not scored for NIL:** its dumps have no gold NIL, so `nil_f1` is undefined for every threshold. The procedure needs NIL gold, so it was run on 2020–2022 only.

## Training and recall

- **Exit codes:** smoke 0, train 0, evaluate 0, all 8 dumps 0, all 3 flipped checks 0.
- **Training:** 12 epochs, best epoch **9** (valid in-batch acc 0.92310). Seed 52313 had best epoch 10 at 0.92725.

`evaluate.sh` (candidate-fused, `unmapped=zero`) gave:

| Test year | R@1 seed 52314 | R@1 seed 52313 | R@64 seed 52314 |
|---|---|---|---|
| 2019 | 0.4166 | 0.4172 | 0.8847 |
| 2020 | 0.3381 | 0.3778 | 0.8573 |
| 2021 | 0.3466 | 0.3831 | 0.8616 |
| 2022 | 0.3401 | 0.3810 | 0.8597 |

On 2020+ the seed-52314 R@1 is 0.04 lower than seed 52313 under `unmapped=zero`. The `--unmapped-adjustment mean` evaluation was not run for this seed.

## Flipped rule, seed 52314 (2020–2022)

| Year | Split | Threshold | NIL predicted | nil_precision | nil_recall | nil_f1 | unnormalized_acc |
|---|---|---|---|---|---|---|---|
| 2020 | validation (fit) | 186.4774 | 1,423 / 43,327 | 0.2178 | 0.2518 | 0.2336 | 0.3931 |
| 2020 | **test** | 186.4774 | 1,738 / 48,215 | **0.1945** | **0.2331** | **0.2120** | **0.3887** |
| 2021 | validation (fit) | 186.5114 | 1,261 / 43,327 | 0.1356 | 0.1389 | 0.1372 | 0.3875 |
| 2021 | **test** | 186.5114 | 1,615 / 48,215 | **0.1474** | **0.1641** | **0.1553** | **0.3873** |
| 2022 | validation (fit) | 186.1253 | 1,509 / 43,327 | 0.2015 | 0.2470 | 0.2219 | 0.3882 |
| 2022 | **test** | 186.1253 | 1,903 / 48,215 | **0.1844** | **0.2421** | **0.2094** | **0.3854** |

## Baselines (same seed-52314 dumps)

| Year | Split | Always link: unnormalized_acc | All NIL: P / R / F1 | All NIL: unnormalized_acc |
|---|---|---|---|---|
| 2020 | validation | 0.3986 | 0.0284 / 1.0000 / 0.0553 | 0.0284 |
| 2020 | test | 0.3946 | 0.0301 / 1.0000 / 0.0584 | 0.0301 |
| 2021 | validation | 0.3956 | 0.0284 / 1.0000 / 0.0553 | 0.0284 |
| 2021 | test | 0.3945 | 0.0301 / 1.0000 / 0.0584 | 0.0301 |
| 2022 | validation | 0.3942 | 0.0284 / 1.0000 / 0.0553 | 0.0284 |
| 2022 | test | 0.3922 | 0.0301 / 1.0000 / 0.0584 | 0.0301 |

Always link has NIL precision, recall and F1 of 0 in every year.

## Seed comparison: test nil_f1 / unnormalized_acc

| Year | Seed 52313 | Seed 52314 | All-NIL F1 |
|---|---|---|---|
| 2020 | 0.2330 / 0.3903 | 0.2120 / 0.3887 | 0.0584 |
| 2021 | 0.1668 / 0.3914 | 0.1553 / 0.3873 | 0.0584 |
| 2022 | 0.2321 / 0.3901 | 0.2094 / 0.3854 | 0.0584 |

## Reading

- **The effect replicates on a second seed.** Test NIL F1 is 0.155–0.212, against 0.058 for all NIL, in all three years. The year pattern is the same as seed 52313, with 2021 weakest. F1 is 0.01–0.02 lower than seed 52313 in each year.
- **The thresholds are stable.** They range over 186.1–186.5 for seed 52314 and 187.9–189.6 for seed 52313, so the scale of the raw dot product shifts slightly with the seed.
- **It still costs accuracy.** unnormalized_acc is 0.006–0.007 below always-link in every year, which is slightly worse than seed 52313 (0.003–0.006).
- **Limits:** two seeds is too few for a confidence interval (`cycle.stats.summarize_seeds` notes that 3–5 seeds are needed). The direction of the rule is still post hoc.

## Resources

- **GPU free:** 32,223 MiB before and 32,223 MiB after (32,742 MiB at every queue check). No other GPU processes were seen, and nothing was killed.
- **Disk:** `/DATA` had 2.8 TB free before and after; this run added 950 MB of checkpoints and 272 MB of dumps there. `/home` is at 99% with about 22 GB free.
- **One rewrite on `/home`:** `scripts/smoke_test.sh` hard-codes `outputs/smoke` on `/home` and deletes it first, so it replaced the existing 1.9 GB smoke output in place with no overall growth. Everything else is on `/DATA`, linked from `outputs/`.
