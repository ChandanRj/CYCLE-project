# NIL evaluation: default run, KB year 2019

Run: `outputs/new_2019_default_seed52313` (default.yaml, new entities, train year 2019, seed 52313, `best.pt` = epoch 10). Code: commit `506e7e4` on branch `nil-and-stats`. Date: 2026-11-14.

## Caveat: bi-encoder only

`scripts/dump_topk.py` is **bi-encoder only and not candidate-fused**. It scores mentions with the dot product of the mention and entity encodings. It does **not** apply the candidate-side graph fusion that `evaluate.py` applies to this candidate-mode run. These numbers are therefore **not** numerically comparable to the `evaluate.py` recall in `REPORT_overnight.md`. For example, 2019 test in-KB accuracy is 0.4185 here, against R@1 0.4172 from `evaluate.py`.

## Setup

- **KB:** the 2019 test entity pool (10,373 entities; `--kb-split test`, the default). Gold entities are matched to the KB by QID. A gold entity that is missing from the 2019 pool is labelled NIL.
- **Rule:** `max`, where NIL score = −(top-1 score). A mention is predicted NIL when its NIL score is above the threshold.
- **Threshold:** chosen on the same year's **validation** dump to maximise `unnormalized_acc`, then applied unchanged to that year's test dump. k = 64.

## Gold NIL share

| Year | Validation mentions | Validation gold NIL | Test mentions | Test gold NIL |
|---|---|---|---|---|
| 2019 | 43,327 | 0 (0.0%) | 48,215 | 0 (0.0%) |
| 2020 | 43,327 | 1,231 (2.8%) | 48,215 | 1,450 (3.0%) |
| 2021 | 43,327 | 1,231 (2.8%) | 48,215 | 1,450 (3.0%) |
| 2022 | 43,327 | 1,231 (2.8%) | 48,215 | 1,450 (3.0%) |

Where the NIL comes from, checked on the raw data: every year's pool is the same 10,000 core entities plus 373 entities specific to that year. The 2019-specific 373 are exactly the 373 entities in the 2019 training set. In 2020, 2021 and 2022, those 373 are replaced by 373 entities specific to that year. Against the 2019 KB, temporal NIL is therefore exactly the mentions of the later year's 373 entities. These are the same 373 entities reported as "unmapped" in `REPORT_overnight.md`.

## Threshold chosen on validation

| Year | Threshold (NIL score) | NIL predictions on validation | Validation unnormalized_acc | Validation inkb_acc |
|---|---|---|---|---|
| 2019 | −124.1071 | 0 | 0.4214 | 0.4214 |
| 2020 | −120.2227 | 0 | 0.4003 | 0.4120 |
| 2021 | −135.6413 | 0 | 0.3983 | 0.4099 |
| 2022 | −145.2953 | 0 | 0.3965 | 0.4081 |

In every year, the chosen threshold equals the largest NIL score on validation, so the model never predicts NIL. Calibration chose **always link**: no threshold under the `max` rule raised validation `unnormalized_acc` above always-link.

## Test results (rule = max, objective = unnormalized_acc)

### 2019 (gold NIL = 0)
| metric | NIL-aware (test) | always link (test) |
|---|---|---|
| unnormalized_acc | 0.4185 | 0.4185 |
| inkb_acc | 0.4185 | 0.4185 |
| nil_precision | 0.0000 | 0.0000 |
| nil_recall | nan | nan |
| nil_f1 | nan | nan |
| link_rate | 1.0000 | 1.0000 |

### 2020 (gold NIL = 1,450)
| metric | NIL-aware (test) | always link (test) |
|---|---|---|
| unnormalized_acc | 0.3967 | 0.3967 |
| inkb_acc | 0.4090 | 0.4090 |
| nil_precision | 0.0000 | 0.0000 |
| nil_recall | 0.0000 | 0.0000 |
| nil_f1 | 0.0000 | 0.0000 |
| link_rate | 1.0000 | 1.0000 |

### 2021 (gold NIL = 1,450)
| metric | NIL-aware (test) | always link (test) |
|---|---|---|
| unnormalized_acc | 0.3947 | 0.3947 |
| inkb_acc | 0.4069 | 0.4069 |
| nil_precision | 0.0000 | 0.0000 |
| nil_recall | 0.0000 | 0.0000 |
| nil_f1 | 0.0000 | 0.0000 |
| link_rate | 1.0000 | 1.0000 |

### 2022 (gold NIL = 1,450)
| metric | NIL-aware (test) | always link (test) |
|---|---|---|
| unnormalized_acc | 0.3946 | 0.3946 |
| inkb_acc | 0.4068 | 0.4068 |
| nil_precision | 0.0000 | 0.0000 |
| nil_recall | 0.0000 | 0.0000 |
| nil_f1 | 0.0000 | 0.0000 |
| link_rate | 1.0000 | 1.0000 |

## Reading

- **No NIL detection under this protocol.** With `rule=max` and `objective=unnormalized_acc`, the model never predicts NIL, so NIL-aware and always-link are identical in every year. The diagnosis below shows the reason: the `max` score points the wrong way (AUC < 0.5). NIL mentions get *higher* top-1 scores than in-KB mentions.
- **The 2019 row is trivial.** It has no gold NIL, so NIL recall and F1 are undefined (nan) and NIL precision is 0 by convention.
- **Thresholds are raw, unnormalised dot products** (top-1 scores of roughly 120–145). Their scale shifts from year to year, so a threshold cannot be carried from one year to another.
- One seed, one checkpoint. No uncertainty estimate.

## Diagnosis: why NIL predicted on validation is 0 (CPU only, existing dumps)

Script: `/tmp/claude-1004/-home-kmpooja-projects-cycle-el-CYCLE-Cross-Year-Contrastive-Learning-in-Entity-Linking/d90dea0b-eb85-4115-81f7-3a5fa15efe81/scratchpad/nil_diag.py`, using `cycle.nil.nil_score` and `cycle.nil_eval.load_topk`.

### 1. Recount at the reported threshold

| Year | Threshold (from result JSON) | Recount: nil_score > threshold | Reported | Threshold = largest validation NIL score |
|---|---|---|---|---|
| 2019 | −124.10708618164062 | 0 | 0 | yes |
| 2020 | −120.22271728515625 | 0 | 0 | yes |
| 2021 | −135.64126586914062 | 0 | 0 | yes |
| 2022 | −145.29525756835938 | 0 | 0 | yes |

The report is correct. In each year the threshold is the largest observed NIL score, so `>` is never true. This is the tie-break choosing the lowest threshold that already gives always-link accuracy.

### 2. ROC AUC of the NIL score (NIL gold = positive class; 0.5 = no signal)

| Year | Split | Gold NIL | AUC, `max` rule | AUC, `margin` rule | Mean top-1 score, NIL / in-KB |
|---|---|---|---|---|---|
| 2020 | validation | 1,231 / 43,327 | **0.2470** | 0.5414 | 181.93 / 174.43 |
| 2021 | validation | 1,231 / 43,327 | **0.3685** | 0.5710 | 178.46 / 174.42 |
| 2022 | validation | 1,231 / 43,327 | **0.2433** | 0.5528 | 181.88 / 174.40 |
| 2020 | test | 1,450 / 48,215 | 0.2352 | 0.5115 | 182.19 / 174.50 |
| 2021 | test | 1,450 / 48,215 | 0.3481 | 0.5626 | 179.15 / 174.52 |
| 2022 | test | 1,450 / 48,215 | 0.2379 | 0.5514 | 182.04 / 174.48 |

- **The `max` score is inverted.** NIL mentions get *higher* top-1 scores than in-KB mentions, so "low top-1 means NIL" flags the wrong mentions first. Calibration therefore correctly falls back to always-link. Read inverted, −NIL score would give AUC of about 0.63–0.76. That has not been tested as a rule, and it would be a post-hoc choice.
- **The `margin` score is close to chance** (0.51–0.57).

**A likely cause, partly checked:** the top-1 prediction for NIL mentions often lands on one of the 373 entities specific to 2019. Those are the 2019 training entities, and they are the ones replaced in later pools.

| Year (validation) | Share of top-1 in the 2019-specific 373: NIL mentions | in-KB mentions |
|---|---|---|
| 2020 | 0.591 | 0.063 |
| 2021 | 0.413 | 0.064 |
| 2022 | 0.578 | 0.062 |

Examples from 2020 validation, gold → top-1 in the 2019 KB: `2019–20_A-League` → `2018–19_A-League` (192.3); `2019_UCI_Road_World_Championships_–_Men's_road_race` → `2018_UCI_Road_World_Championships_–_Women's_time_trial` (176.2). Two explanations fit, and this check cannot separate them:
- Later "new" entities have near-duplicate predecessors in the 2019-specific set, such as the previous season of the same series.
- The model scores its own training entities high.

### 3. Other rules and objectives, 2020–2022

The threshold is fitted on validation and applied to test. 2019 is skipped because it has no gold NIL. The always-link baseline on test has unnormalized_acc 0.3967 / 0.3947 / 0.3946 for 2020 / 2021 / 2022. Predicting NIL for every mention would give NIL F1 = 0.0584 on test (precision 0.0301, recall 1).

| Year | Rule | Objective | Threshold | NIL predicted on validation (of 43,327) | Validation P / R / F1 | Validation unnormalized_acc | Test P / R / F1 | Test unnormalized_acc | Test link_rate |
|---|---|---|---|---|---|---|---|---|---|
| 2020 | max | unnormalized_acc | −120.2227 | 0 | 0 / 0 / 0 | 0.4003 | 0 / 0 / 0 | 0.3967 | 1.0000 |
| 2020 | max | nil_f1 | −202.5656 | 43,326 | 0.0284 / 0.9992 / 0.0552 | 0.0284 | 0.0301 / 1.0000 / 0.0584 | 0.0301 | 0.0000 |
| 2020 | margin | unnormalized_acc | −0.0001 | 0 | 0 / 0 / 0 | 0.4003 | 0 / 0 / 0 | 0.3967 | 0.9999 |
| 2020 | margin | nil_f1 | −0.8632 | 17,249 | 0.0331 / 0.4639 / 0.0618 | 0.3309 | 0.0316 / 0.4207 / 0.0588 | 0.3259 | 0.5996 |
| 2021 | max | unnormalized_acc | −135.6413 | 0 | 0 / 0 / 0 | 0.3983 | 0 / 0 / 0 | 0.3947 | 1.0000 |
| 2021 | max | nil_f1 | −203.4482 | 43,326 | 0.0284 / 0.9992 / 0.0552 | 0.0284 | 0.0301 / 1.0000 / 0.0584 | 0.0301 | 0.0000 |
| 2021 | margin | unnormalized_acc | −0.0000 | 0 | 0 / 0 / 0 | 0.3983 | 0 / 0 / 0 | 0.3947 | 1.0000 |
| 2021 | margin | nil_f1 | −0.6570 | 13,994 | 0.0361 / 0.4102 / 0.0663 | 0.3468 | 0.0356 / 0.3821 / 0.0651 | 0.3442 | 0.6768 |
| 2022 | max | unnormalized_acc | −145.2953 | 0 | 0 / 0 / 0 | 0.3965 | 0 / 0 / 0 | 0.3946 | 1.0000 |
| 2022 | max | nil_f1 | −201.9480 | 43,326 | 0.0284 / 1.0000 / 0.0553 | 0.0284 | 0.0301 / 1.0000 / 0.0584 | 0.0301 | 0.0000 |
| 2022 | margin | unnormalized_acc | −0.0001 | 0 | 0 / 0 / 0 | 0.3965 | 0 / 0 / 0 | 0.3946 | 1.0000 |
| 2022 | margin | nil_f1 | −0.4993 | 11,261 | 0.0346 / 0.3168 / 0.0624 | 0.3573 | 0.0331 / 0.2855 / 0.0593 | 0.3542 | 0.7403 |

- **`max` + `nil_f1` is degenerate.** It predicts NIL for every mention but one on validation, and for every mention on test. Its F1 (0.0584) is just the predict-all-NIL baseline.
- **`margin` + `nil_f1`** reaches test NIL F1 of 0.059–0.065. That is barely above predicting all NIL (0.058), and unnormalized_acc falls from about 0.395 to 0.33–0.35.
- **`margin` + `unnormalized_acc`** again chooses always-link. In 2020 it flags a handful of test mentions with near-zero margin (link_rate 0.9999).
- **Conclusion:** none of the tested rule and objective pairs separates NIL from in-KB mentions usefully on these bi-encoder dumps.

## Flipped max rule: POST HOC, pre-registered single test (2020 only)

> **Post hoc.** The direction of this rule was chosen *after* seeing the inverted AUC (0.247 on 2020 validation) on these same dumps. Rule, objective, threshold candidates and tie-break were written down before the run: `outputs/nil_dumps/PREREG_flipped_rule.md`, 2026-11-14T13:21:38+05:30, sha256 `7cba61ae…5e240d81`. They were not changed afterwards, and the test dump was scored once. This is one year, one seed and one checkpoint, so treat it as a hypothesis to confirm on other runs, not as a result.

- **Rule:** NIL if top-1 score ≥ threshold, otherwise link to top-1.
- **Threshold:** fitted on 2020 validation only, maximising `nil_f1`. Candidates are 512 quantiles of the unique validation top-1 scores plus +inf, the same scheme as `calibrate_topk`. Ties keep the largest threshold. The chosen threshold is **187.8945**, from 513 candidates.
- **Script:** `/tmp/claude-1004/-home-kmpooja-projects-cycle-el-CYCLE-Cross-Year-Contrastive-Learning-in-Entity-Linking/d90dea0b-eb85-4115-81f7-3a5fa15efe81/scratchpad/flipped.py`. Result: `outputs/nil_dumps/result_y2020_flipped_nil_f1.json`.

| 2020 | NIL predicted | Gold NIL | nil_precision | nil_recall | nil_f1 | unnormalized_acc | inkb_acc | link_rate |
|---|---|---|---|---|---|---|---|---|
| validation (fit) | 1,594 / 43,327 | 1,231 | 0.2171 | 0.2811 | 0.2450 | 0.3945 | 0.3978 | 0.9632 |
| **test (applied once)** | 1,880 / 48,215 | 1,450 | **0.2064** | **0.2676** | **0.2330** | **0.3903** | 0.3941 | 0.9610 |

The 2020 test references are:

| Reference | nil_f1 | unnormalized_acc |
|---|---|---|
| Always link | 0.0000 | 0.3967 |
| All NIL | 0.0584 | 0.0301 |
| `margin` + `nil_f1` | 0.0588 | 0.3259 |

The flipped rule is the first one with NIL F1 well above the all-NIL baseline: 0.233 against 0.058. It costs 0.006 of unnormalized_acc against always-link. Validation-to-test transfer is stable (F1 0.245 → 0.233).

### Confirmation on 2021 and 2022: same frozen procedure, unchanged

- **Script:** `scripts/flipped_rule_check.py --year {Y}`. It is the 2020 scratch logic copied unchanged; the only change is that the year is now an argument, which a `diff` of the two scripts confirms.
- **Regression check:** `--year 2020` reproduces `result_y2020_flipped_nil_f1.json` byte for byte (`cmp`).
- **Reference file:** `outputs/nil_dumps/reference/result_y2020_flipped_ref.json` (sha256 `b3a813cc…54ad998a`) is the byte-identical reference output for `scripts/flipped_rule_check.py --year 2020`.
- **Pre-registration file:** unchanged (sha256 `7cba61ae…5e240d81`, rechecked before the runs).
- **Protocol:** each threshold was fitted on that year's validation dump only and applied once to that year's test dump.
- **Results:** `outputs/nil_dumps/result_y{2021,2022}_flipped_nil_f1.json`. Logs: `outputs/nil_dumps/logs/flipped_y{2020,2021,2022}.log`.

| Year | Split | Threshold | NIL predicted | Gold NIL | nil_precision | nil_recall | nil_f1 | unnormalized_acc | inkb_acc | link_rate |
|---|---|---|---|---|---|---|---|---|---|---|
| 2020 | validation (fit) | 187.8945 | 1,594 / 43,327 | 1,231 | 0.2171 | 0.2811 | 0.2450 | 0.3945 | 0.3978 | 0.9632 |
| 2020 | **test** | 187.8945 | 1,880 / 48,215 | 1,450 | **0.2064** | **0.2676** | **0.2330** | **0.3903** | 0.3941 | 0.9610 |
| 2021 | validation (fit) | 189.6107 | 843 / 43,327 | 1,231 | 0.1981 | 0.1357 | 0.1610 | 0.3948 | 0.4024 | 0.9805 |
| 2021 | **test** | 189.6107 | 1,032 / 48,215 | 1,450 | **0.2006** | **0.1428** | **0.1668** | **0.3914** | 0.3991 | 0.9786 |
| 2022 | validation (fit) | 188.4422 | 1,344 / 43,327 | 1,231 | 0.2210 | 0.2413 | 0.2307 | 0.3922 | 0.3966 | 0.9690 |
| 2022 | **test** | 188.4422 | 1,617 / 48,215 | 1,450 | **0.2202** | **0.2455** | **0.2321** | **0.3901** | 0.3946 | 0.9665 |

| Year | Test, always link: unnormalized_acc | Test, all NIL: nil_f1 | Test, `margin` + `nil_f1`: nil_f1 |
|---|---|---|---|
| 2020 | 0.3967 | 0.0584 | 0.0588 |
| 2021 | 0.3947 | 0.0584 | 0.0651 |
| 2022 | 0.3946 | 0.0584 | 0.0593 |

- **The effect holds out of sample.** In both held-out years, test NIL F1 (0.167 and 0.232) is well above all-NIL (0.058) and above the `margin` rule. Validation-to-test transfer is stable: 0.161 → 0.167 in 2021 and 0.231 → 0.232 in 2022.
- **2021 is weaker.** Recall falls to 0.14, which matches its weaker inverted AUC (0.369 against 0.24–0.25 for 2020 and 2022) and its lower predecessor-name overlap in the near-duplicate table.
- **Cost:** unnormalized_acc is 0.003–0.006 below always-link in every year. The rule finds NIL mentions, but not well enough to pay for the in-KB links it gives up under this metric.
- **Caveats:** the direction is still post hoc. All three years use the same seed, checkpoint and KB year (2019), so this is not independent confirmation across models. The fitted thresholds are close to each other (187.9–189.6), but the `max` scores are raw dot products and their scale is specific to this checkpoint.

## Near-duplicate hypothesis: name overlap of the top-1 prediction

**Question:** for NIL mentions whose top-1 is one of the 2019-specific entities (the 373 replaced in later pools), how often does the top-1 entity's name share its first token with the gold entity's name?

- Names are Wikipedia titles split on `_` and lowercased.
- **Comparison groups (added by me):** NIL mentions whose top-1 is a core entity, and in-KB mentions with a wrong top-1.
- **Extra variant (added by me, not requested):** the first token is very often a year (`2019_…`, `2019–20_…`), which differs from a predecessor season by construction. I therefore also report the first *non-year* token, where a year token matches `^\d{4}([–-]\d{2,4})?$`. I defined this variant in the same script, before seeing its output.
- **Script:** `/tmp/claude-1004/-home-kmpooja-projects-cycle-el-CYCLE-Cross-Year-Contrastive-Learning-in-Entity-Linking/d90dea0b-eb85-4115-81f7-3a5fa15efe81/scratchpad/neardup.py`.

| Year | Split | NIL, top-1 2019-specific: n / first-token / first non-year token | NIL, top-1 core: n / first / non-year | in-KB wrong: n / first / non-year |
|---|---|---|---|---|
| 2020 | validation | 728 / **0.070** / **0.486** | 503 / 0.082 / 0.233 | 24,751 / 0.082 / 0.151 |
| 2020 | test | 852 / **0.070** / **0.491** | 598 / 0.052 / 0.253 | 27,640 / 0.086 / 0.156 |
| 2021 | validation | 509 / 0.010 / 0.275 | 722 / 0.018 / 0.141 | 24,841 / 0.082 / 0.149 |
| 2021 | test | 659 / 0.018 / 0.363 | 791 / 0.023 / 0.143 | 27,734 / 0.084 / 0.154 |
| 2022 | validation | 711 / 0.010 / 0.446 | 520 / 0.031 / 0.188 | 24,915 / 0.080 / 0.151 |
| 2022 | test | 784 / 0.017 / 0.366 | 666 / 0.035 / 0.276 | 27,741 / 0.083 / 0.154 |

- **The first-token test as specified does not support the hypothesis.** The match rate is 0.01–0.07, no higher than the comparison groups (about 0.08).
- **With the leading year removed, it does.** 28–49% of these NIL mentions share the first non-year token, against 14–28% when top-1 is a core entity and about 15% for wrong in-KB links. Examples from 2020 test: `2019_Mutua_Madrid_Open` → `2018_Mutua_Madrid_Open`, `2019–20_LigaPro` → `2018–19_LigaPro`, `The_Flash_(season_6)` → `The_Flash_(season_5)`.
- **Caveat:** the first-token match is a crude proxy, and it also fires on loose matches such as `2019_WNBA_season` → `2019_NFL_season`. This supports, but does not prove, that many temporal-NIL errors are confusions with the previous instance of a recurring entity. The flipped rule plausibly works because these predecessor matches score very high.

## Failures and notes

- **Failures:** none. All 8 dumps and all 4 `eval_nil.py` runs exited with 0. The dump loop's final exit code of 1 came from its last `[ $rc -ne 0 ]` test returning false, not from a failed command.
- **pytest:** it was not installed. It was installed into `/home/kmpooja/envs/cycle-env` (pytest 9.2.0, with user approval) before the tests ran: `27 passed in 0.84s`.
- **Patch file:** `nil_full_patch.diff` is in the repo root (825 lines, 31,945 bytes). It is untracked and not committed.
- **Environment:** same as the overnight chain: `CYCLE_BERT_PATH=<bert_uncased_L-8_H-512_A-8 snapshot>` and `/home/kmpooja/envs/cycle-env/bin/python`, with `PYTHONPATH=src`.
- **Resources:** about 31.5 GiB of GPU memory was free and no other GPU process was running. The dumps use 272 MB on `/home`, which is 99% full with about 23 GB free.
- **Logs:** `outputs/nil_dumps/logs/dump_y{Y}_{split}.log` and `outputs/nil_dumps/logs/eval_nil_y{Y}.log`. The JSON results are in `outputs/nil_dumps/result_y{Y}_max.json`.
- **Diagnosis runs:** CPU only (`CUDA_VISIBLE_DEVICES=""`). All 9 extra `eval_nil.py` runs exited with 0. Their logs are `outputs/nil_dumps/logs/eval_nil_y{Y}_{rule}_{objective}.log` and their results `outputs/nil_dumps/result_y{Y}_{rule}_{objective}.json`. Nothing was committed or pushed.
- **Flipped-rule test and name-overlap check:** CPU only, and both ran without errors. The flipped rule is not part of `cycle.nil` or `scripts/eval_nil.py`; it was implemented in a scratch script. No code in the repo was changed. Nothing was committed or pushed.
