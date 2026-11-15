# Temporal NIL with the flipped max rule: two seeds

Model: `default.yaml`, new entities, train year 2019. Seeds **52313** (`outputs/new_2019_default_seed52313`) and **52314** (`outputs/new_2019_default_seed52314`). Date: 2026-11-14.

## What was run

- **KB:** the 2019 entity pool, 10,373 entities. A gold entity missing from the 2019 pool is labelled NIL. Gold NIL is 1,450 of 48,215 test mentions (3.0%) in each of 2020–2022.
- **Scores:** bi-encoder top-64 dumps from `scripts/dump_topk.py`. These are bi-encoder only and **not candidate-fused**.
- **Rule (post hoc):** NIL if top-1 score ≥ threshold.
- **Procedure:** frozen in `outputs/nil_dumps/PREREG_flipped_rule.md` (sha256 `7cba61ae…`) and run with `scripts/flipped_rule_check.py`. The threshold is fitted on that year's validation dump only (objective `nil_f1`) and applied once to that year's test dump.
- **Reproducibility:** `scripts/flipped_rule_check.py --year 2020` reproduces `outputs/nil_dumps/reference/result_y2020_flipped_ref.json` byte for byte. Neither the procedure nor the script changed between the two seeds (hashes checked).

## Results (test)

| Year | Seed 52313 NIL F1 | Seed 52314 NIL F1 | Mean NIL F1 | All-NIL F1 | Seed 52313 unnormalized_acc (always link) | Seed 52314 unnormalized_acc (always link) |
|---|---|---|---|---|---|---|
| 2020 | 0.2330 | 0.2120 | **0.2225** | 0.058 | 0.3903 (0.3967) | 0.3887 (0.3946) |
| 2021 | 0.1668 | 0.1553 | **0.1611** | 0.058 | 0.3914 (0.3947) | 0.3873 (0.3945) |
| 2022 | 0.2321 | 0.2094 | **0.2208** | 0.058 | 0.3901 (0.3946) | 0.3854 (0.3922) |

- **Thresholds fitted on validation:**

  | Seed | 2020 | 2021 | 2022 |
  |---|---|---|---|
  | 52313 | 187.89 | 189.61 | 188.44 |
  | 52314 | 186.48 | 186.51 | 186.13 |

- **All-NIL F1** (= 2p/(1+p), p = 1,450/48,215 = 0.0301) is 0.0584, the same for every year and seed. Always link has NIL F1 0.
- **Mean** is over two seeds only. There is no confidence interval: two seeds are too few (`cycle.stats.summarize_seeds` notes that 3–5 are needed).
- **Per-seed details** (precision, recall, validation rows, baselines): `outputs/nil_dumps/NIL_SUMMARY.md` for seed 52313 and `outputs/nil_dumps/NIL_SUMMARY_SEED2.md` for seed 52314.

## Reading

- **The rule detects some NIL in both seeds.** Mean test NIL F1 of 0.16–0.22 is about 3–4 times the all-NIL baseline in every year. 2021 is weakest in both seeds.
- **It costs accuracy.** In every year and seed, unnormalized_acc is 0.003–0.007 below always-link. At about 3% NIL, the rule gives up more correct in-KB links than it gains in correct NIL predictions.

## Caveats

1. **The direction is post hoc.** The flipped direction (high top-1 means NIL) was chosen after seeing seed 52313's inverted AUC (0.247 on 2020 validation). Seed 52313's 2020 numbers were therefore not an independent test. Its 2021 and 2022 numbers and all of seed 52314 used the frozen procedure, but the rule itself came from looking at seed 52313's data.
2. **One setting only.** Both seeds use KB year 2019, the same data split, the same config and the same train year. The result says nothing yet about other KB years, the continual entity set, other configs, or candidate-fused scores.
3. **NIL may be partly a labelling question.** The gold NIL entities are genuinely absent from the 2019 KB (for example `2019_Mutua_Madrid_Open`). But the model's top-1 for them is often the *previous instance* of the same recurring entity (`2018_Mutua_Madrid_Open`, `2018–19_A-League` for `2019–20_A-League`). For seed 52313, the top-1 for 2020 NIL mentions is one of the 2019-specific entities in 59% of cases on validation. Among those, 49% share the first non-year name token, against 15–25% in the comparison groups (see `NIL_SUMMARY.md`). Whether such a link should count as wrong, NIL, or a separate "predecessor" class is a task-definition choice. The flipped rule probably works because these predecessor matches score very high, so it may partly be detecting "recurring-event successor" rather than "new entity" in general. The name-overlap check is crude and was run only for seed 52313.
4. **Scores are bi-encoder only, not candidate-fused**, so they don't match `evaluate.py` recall. The thresholds are raw dot products whose scale shifts with the seed (about 186 against 188–190).
5. **Seed 52314 has lower R@1 on 2020+** (0.34 against 0.38 under `unmapped=zero`). The `mean`-adjusted evaluation was not run for seed 52314.
