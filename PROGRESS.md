# CYCLE Reproduction — Progress Notes

Last updated: 2026-09-29. Branch: `cycle-implementation`.

Goal: reproduce CYCLE (Cross-Year Contrastive Learning in Entity Linking, CIKM 2024) on GCL-TempEL and compare recall@k with paper Table 2 (continual: 0.286–0.301, new: 0.184–0.222).

---

## 1. Where things stand

- All model code is implemented and committed. Smoke tests pass for both `baseline` and `paper` configs.
- The debug-scale experiments for train year 2019 are finished (`debug_n=1024`, seed 52313).
- The candidate-mode drop from 2019 to 2020+ is diagnosed and fixed at evaluation time (finding 5, commit `c92103c`).
- The graph-embedding collapse is diagnosed, and the cosine / hard-negative-mining arms are run over 3 seeds (finding 6, commit `e0329a7`).
- Learned (Kendall) loss weighting is implemented and run over 3 seeds (finding 7, commit `a6ef8e6`). It gives no R@1 gain. **Finding 7 shows that checkpoint selection, not the method, drives R@1 differences between runs. This is the most important methodological result of the mining/weighting work and qualifies every R@1 comparison between runs in this file.**
- **Next step: the writeup.** Experiments are finished; no more runs.

## 2. Commits on this branch

| Commit | What |
|---|---|
| `e066de6` | Model files (BiEncoder, ScoreFusion, PaperGraphModule, HeCo) for paper Eq. 1–15; `graph.relation_extra_negatives` extension; `--set` override flag; `config.apply_overrides`; `evaluate.py` infers the fusion bias from the checkpoint; `.gitignore` `models/` rule anchored to the repo root |
| `c9efe09` | `scripts/diag_negatives.py` (Eq. 11 zero-negatives diagnostic), `scripts/summary_chart.py` (chart; now writes `results/summary_chart.png`) |
| `6e248bd` | Candidate-mode fusion experiments: `scripts/run_candidate.sh`, `scripts/gpu_queue.sh`, results in `results/new_2019_candidate/` |
| `000cb57` | Explicit `data.valid_n: null` and `fusion.mode: gold_mention` in both configs; `train.py` rejects unknown `--set` keys (previously `model.relation_extra_negatives` was silently accepted; the correct key is `graph.relation_extra_negatives`). `evaluate.py` already rejected them. Adds this file. |
| `c92103c` | `evaluate.py --unmapped-adjustment {zero,mean}` (default `zero`, the original behaviour); `scripts/check_unmapped_bias.py` (finding 5) |
| `c0361de` | Candidate-mode results after the fix: `mean` rows in `results/new_2019_candidate/*/eval_results.jsonl`, `unmapped_bias_check.jsonl`; this file updated |
| `9468a62` | Summary chart regenerated with the `--unmapped-adjustment mean` results |
| `e0329a7` | `graph.similarity {dot,cosine}` and `graph.negative_mining {random,hard}` (defaults `dot`/`random`, i.e. the paper loss); `scripts/check_embedding_collapse.py`; `scripts/run_mining.sh` (finding 6) |
| `b08ca3a` | Finding 6 documented; mining/similarity run outputs and collapse metrics in `results/new_2019_mining/` |
| `a6ef8e6` | `loss.weighting {fixed,learned}` + `loss.weighting_lr` (default `fixed`); `src/cycle/models/loss_weighting.py`; `extranneg_cos_learned` arm in `scripts/run_mining.sh` |
| (this commit) | Finding 7 documented; learned-weighting run outputs and `checkpoint_selection.jsonl` in `results/new_2019_mining/` |

## 3. Code map

| File | Notes |
|---|---|
| `src/cycle/models/biencoder.py` | Two BERT towers. The attribute must be named `bert_model` because `optim.py` matches `bert_model.encoder.layer` for differential LR. `CYCLE_BERT_PATH` env var overrides the model path. |
| `src/cycle/models/fusion.py` | `ScoreFusion`: additive tanh-gated fusion. Must keep the name `.linear` because `evaluate.py` reassigns it. `learned` projection is zero-init; `random_frozen` is random and frozen. `qid` or `sequential` alignment. |
| `src/cycle/models/graph_paper.py` | `PaperGraphModule`: GAT attention (Eq. 6/7), shared MLP (Eq. 8/9), L_f (Eq. 13), L_r (Eq. 14). Call `resample_neighbors()` once per epoch. |
| `src/cycle/models/loss_weighting.py` | `UncertaintyWeighting` (Kendall et al. 2018), used when `loss.weighting=learned`. log σ_i = s0_i + δ_i with δ_i learned from 0, so effective weights start at exactly a/b/c. Its parameter group is excluded from gradient clipping. |
| `src/cycle/models/heco.py` | HeCo, used only by `baseline.yaml`. Output is cached in `data/processed/cache/heco_*.pt`. |

### Design decisions (settled, don't re-derive)
- L_f denominator is restricted to P_i^r ∪ N_i^r (Eq. 13). The L_r denominator covers all n nodes (Eq. 14).
- L_r positives are the kNN feature-graph neighbours. L_f positives are the newly added cross-year relation edges.
- InfoNCE is sum-then-log: `-(logsumexp(pos) - logsumexp(denom))` per anchor.
- Zero-positive guard: `anchor.sum() * 0`, which keeps the grad_fn.
- OOM fixes: `_aggregate_and_project` is chunked (`agg_chunk=1024`) with gradient checkpointing and split-weight attention. `_infonce_r` is chunked row-wise. `_infonce_f` gathers `target[pos_idx]` directly. Training now peaks at about 2.5 GiB.

## 4. Key findings (for the writeup)

1. **The Eq. 11 cross-year negatives are always 0 on GCL-TempEL.** This comes from the data, not a bug. Each year has 10373 nodes: 10000 shared entities plus 373 year-specific ones. The 5007 edges among the shared entities are identical in every year, and every added or removed edge touches a year-specific entity. As a result, L_f = 0 exactly, with zero gradient. From 2020→21 onward there are no positives either. `scripts/diag_negatives.py` reproduces this. Deviation: `graph.relation_extra_negatives=K` adds sampled negatives to L_f.
2. **The main results are bi-encoder-only at test time.** With `fusion.mode=gold_mention` (the default), `evaluate.py` never applies the graph. The graph only affects BERT indirectly, through the fused training loss.
3. **Candidate-mode fusion** (with the finding-5 fix, `--unmapped-adjustment mean`; qualified by finding 7, since the R@1 ordering below follows the runs' chosen epochs, 2 < 4 < 7 < 8–9): the paper-strict version is *worse* than the text-only control (R@1 ≈ 0.22 vs ≈ 0.25 in every test year). With extra negatives it beats the control (≈ 0.28) and roughly matches the gold_mention runs (0.280–0.282 vs 0.279–0.286), but does not clearly exceed them. So test-time graph fusion adds little on top of what the graph already contributes through training.
4. **Continual loss_e reaches about 0.01, which means memorization.** Report this plainly.
5. **Entities absent from the train-year graph break candidate-mode evaluation.** The paper does not say how to score entities that appear after the training year. In test 2020–22, 373/10373 pool entities have no 2019-graph node, and `enhance_pool` gave them a zero adjustment, which never occurs in training. They then took the top-1 slot far above their 3.6% fair share: 35.7% of mentions for paper-strict (whose adjustment lowers mapped scores by 5.3 on average) and 11.5% with extra negatives, against 3.3–3.4% with no adjustment at all. Giving them the mean adjustment of mapped entities (`--unmapped-adjustment mean`) brings the rate to 3.2–3.7% (matching the no-adjustment control) and removes the 2019→2020+ drop. Test 2019 (all entities mapped) is bit-identical. `scripts/check_unmapped_bias.py` reproduces this; output in `results/new_2019_candidate/unmapped_bias_check.jsonl`.
6. **The paper's dot-product graph embeddings collapse, and neither cosine similarity nor hard-negative mining raises R@1.** Setting: candidate mode, extra negatives=8, train 2019, 3 seeds per arm (tables in §5). Four points:
   1. *Collapse is present at initialization.* X is a row-normalized bag-of-words over 2929 dims, so the input term of the first layer is tiny next to its bias (|W1 x| ≈ 0.014 vs |b1| ≈ 0.083). Every node therefore maps to nearly the same vector: pairwise cosine 0.99985 (z_r) and 0.99994 (z_f), with 99.98% of the energy in the top singular direction.
   2. *The dot-product loss cannot escape it, and "cheats" through vector length.* After training, the dot arm is still at pairwise cosine ≥ 0.9998 in every seed. Entities differ only in norm (about 4–5, up from 0.57 at init). L_f drops from chance (2.02) to about 0.5–1.0 because positives are made longer than negatives (|z_r| 4.97 vs 4.30 for seed 52313), not because they point elsewhere: both have cosine ≈ 1.000 with the anchor. The bias still dominates (|b1| 0.67–0.78 vs |W1 x| ≈ 0.055). In candidate mode this makes the graph adjustment close to a constant shift, so it barely changes rankings.
   3. *Cosine similarity resolves the collapse and stabilizes seeds, but does not raise mean R@1.* Pairwise cosine drops to 0.44–0.55 (z_r) and 0.17–0.26 (z_f), top-direction share to 0.77–0.84, and L_f separates by direction (positives cos 0.96–0.98, negatives −0.61 to −0.73). Seed-to-seed R@1 range shrinks from 6.6 points (dot) to 1.3 points, but the mean (0.277) is within noise of dot's best seeds (0.281, 0.277).
   4. *Hard-negative mining (on top of cosine) has a measurable effect on the loss and none on R@1.* L_f stays at 1.5–1.85 instead of falling to about 0.9–1.0 with random negatives, since the negatives are now the hardest ones (negative cosine ≈ 0 instead of −0.6 to −0.7). R@1 per seed is 0.299 / 0.275 / 0.224, the same pattern and spread as dot. A likely reason: L_f only touches 12 of the 10373 anchor nodes on the 2019 graph (16 positive pairs), so how its negatives are picked barely reaches the rest of the model.

   **Checkpoint-selection noise, not the choice of negatives or similarity, dominates the seed-to-seed spread** (this is the issue flagged earlier as option 3). The best epoch is picked by in-batch accuracy on `valid_n=256` mentions, where one mention is 0.39 points and, near the best epoch, consecutive epochs differ by 1–4 mentions. That signal does not track test R@1: seed 3 early-stopped at epoch 2 in both the dot and hard arms (validation 0.883 and 0.871, the highest or near-highest of those runs) and landed at the lowest test R@1 (0.215, 0.224). With cosine + random, seed 3 picked epoch 5 and reached 0.271. The seed-3 drop (5–7 points) is far larger than any difference between arm means (1–2 points). **With 3 seeds under this validation setup, 1–2 point differences between arms cannot be resolved.** Resolving them would take a larger validation set or a fixed epoch budget first, then more seeds.
   The between-arm R@1 comparisons in this finding are further qualified by finding 7; the collapse metrics are not.
   `scripts/check_embedding_collapse.py` reproduces the collapse metrics (`--init` for the untrained module); outputs are in `results/new_2019_mining/`.

7. **Learned loss weighting gives no R@1 gain. Checkpoint selection, not the method, drives R@1 across the whole mining/weighting batch. This is the most important methodological finding of that investigation.**
   *Setup.* This replaces the fixed Eq. 15 weights with Kendall et al. (2018) uncertainty weights, L = Σ_i L_i/(2σ_i²) + log σ_i (`loss.weighting=learned`, `weighting_lr=0.01`). The effective weights start at exactly a = b = c = 1. Arm: random + cosine, extra negatives=8, candidate mode, 3 seeds, compared with the existing random + cosine seeds as the fixed-weight arm. Regression runs (run by hand, not committed as tests) were bit-exact: the fixed path reproduces the committed `extranneg_cos_seed52313` log, and `weighting_lr=0` reproduces the fixed run, logs and all 276 checkpoint tensors included.

   *The learned weights follow loss size, not noise.* Numbers are from epochs ≥ 2, over 3 seeds:

   | Term | Trajectory (start 1.0) | End weight | Mean loss | 1/(2·loss) | Step-to-step CV |
   |---|---|---|---|---|---|
   | L_e | dips to 0.65, then rises every epoch | 0.67–0.74 | 0.50 | 0.9–1.0 | 0.59 (noisiest) |
   | L_f | down, flat from epoch 1 | 0.52–0.54 | 0.94 | 0.53 (w·2L = 0.97–0.99) | 0.12 |
   | L_r | down every epoch | 0.09–0.12 | 7.45 | 0.067 | 0.01 (least noisy) |

   The weight order e > f > r matches the loss-size order and is the reverse of the noise order. L_f sits at its magnitude equilibrium; it is not weighted down further for its noise or for having only 12 anchors. L_e and L_r are still moving toward 1/(2L) when the learning rate decays to 0. This is expected: the objective's stationary point is σ_i² = E[L_i], whatever the variance. So the mechanism reacts to raw loss size and does not compensate for L_f's sparsity.

   *R@1 (mean over test 2019–22):* learned 0.299 / 0.279 / 0.225 (mean 0.268 ± 0.038, range 7.4 points) vs fixed 0.277 / 0.284 / 0.271 (mean 0.277 ± 0.006, range 1.3 points). No improvement: per seed +2.1, −0.5, −4.6 points, and seed 3 again early-stopped at epoch 2.

   *Checkpoint selection dominates.* Over all 12 runs of the batch (4 arms × 3 seeds, table in §5), **the chosen best epoch correlates with test R@1 at Pearson r = 0.97** (r² = 0.94; Spearman 0.91). R@1 rises by about 1.0 point per epoch of the chosen checkpoint. For comparison:
   - the arm accounts for 6% of the R@1 variance, the seed for 75%;
   - the selection criterion itself (in-batch accuracy on `valid_n=256`) correlates only 0.41 with test R@1.
   Every epoch-2 run scores 0.215–0.225, whichever arm it belongs to, and both epoch-9 runs score 0.299. Seed 3 early-stops at epoch 2 in 3 of 4 arms, so the seed acts mainly through where the noisy validation curve happens to peak. Tie handling adds to this: `acc > best_acc` keeps the earliest of tied epochs, and ties are common at 1/256 resolution; this changed the pick in 2 of the 12 runs (dot seed 52313: 0.883 at epochs 7–9, epoch 7 kept; hard + cosine seed 52313: 0.887 at epochs 9 and 11, epoch 9 kept). This is a correlation over 12 runs, not a controlled test. The direct test would be to evaluate every run at a fixed epoch budget, which has not been done.

   **What this qualifies, retroactively.** Under this validation setup (early stopping, patience 3, on 256 validation mentions), a 1–2 point R@1 difference between two runs cannot be attributed to the method that differs between them. That applies to every R@1 comparison between runs in this file:
   - finding 6: dot vs cosine vs hard mining;
   - finding 7: fixed vs learned weights;
   - §5: extra negatives vs v2 (+0.3 points, epochs 9 vs 8);
   - finding 3: paper-strict candidate 0.219 (epoch 2) < text-only control 0.246 (epoch 4) < candidate + extra negatives 0.280 (epoch 7) ≈ gold_mention 0.279–0.286 (epochs 8–9). This ordering follows the chosen epochs exactly;
   - v1 vs v2 (0.195 at epoch 1 with `valid_n=64`, vs 0.28 at epoch 8), which was already attributed to validation.

   Not affected, because they are not R@1 comparisons between runs:
   - findings 1 and 2: a data property and a code path;
   - finding 4: training loss;
   - finding 5: two scoring rules on the same checkpoint;
   - finding 6's collapse metrics: embedding geometry, consistent across all seeds.

   Fixing this needs a larger validation set, a fixed epoch budget (or evaluating a fixed last epoch), then more seeds. Data: `results/new_2019_mining/checkpoint_selection.jsonl` (arm, seed, chosen epoch, per-epoch validation accuracy, R@1) and the per-run folders.
## 5. Results (train year 2019, new entities unless noted, debug_n=1024, seed 52313, R@1)

Each test year has 48215 mentions and 10373 entities.

| Run (in `outputs/`) | Notes | Test 2019 | Test 2020–22 |
|---|---|---|---|
| `new_2019_paper_seed52313` | valid_n=64, best ep 1 | ~0.195–0.200 | ~0.195–0.200 |
| `new_2019_paper_seed52313_v2` | valid_n=256, best ep 8 | ~0.279–0.283 | ~0.279–0.283 |
| `new_2019_paper_extranneg_seed52313` | extra negatives=8, best ep 9 | ~0.283–0.286 | ~0.283–0.286 |
| `continual_2019_paper_extranneg_seed52313` | continual, best ep 5 | ~0.384–0.388 | ~0.384–0.388 |
| `new_2019_textonly_seed52313` | graph and fusion off (control), best ep 4 | 0.246 | 0.247–0.250 |
| `new_2019_paper_candidate_seed52313` | fusion.mode=candidate, best ep 2; unmapped=zero (original) | 0.219 | 0.153–0.155 |
| ″ | same checkpoint, `--unmapped-adjustment mean` | 0.219 | 0.221–0.222 |
| `new_2019_paper_candidate_extranneg_seed52313` | candidate + extra negatives=8, best ep 7; unmapped=zero (original) | 0.280 | 0.259–0.261 |
| ″ | same checkpoint, `--unmapped-adjustment mean` | 0.280 | 0.280–0.282 |
| `new_2015_paper_seed52313` | early debug run, best ep 0 | 0.093 (2015) | ~0.093 (2016–18) |

### Similarity / negative-mining arms (finding 6)

Candidate mode, extra negatives=8, `--unmapped-adjustment mean`, valid_n=256, early stopping (patience 3, max 12 epochs). R@1 is averaged over test years 2019–2022 (the four years are within 0.6 points of each other in every run). Runs are in `outputs/new_2019_paper_candidate_extranneg{,_cos,_hardcos}_seed{52313,2,3}`; copies are in `results/new_2019_mining/`.

| Arm | Seed 52313 | Seed 2 | Seed 3 | Mean | SD | Range |
|---|---|---|---|---|---|---|
| random + dot (paper loss) | 0.281 (ep 7) | 0.277 (ep 8) | 0.215 (ep 2) | 0.258 | 0.037 | 0.066 |
| random + cosine | 0.277 (ep 7) | 0.284 (ep 8) | 0.271 (ep 5) | 0.277 | 0.006 | 0.013 |
| hard + cosine | 0.299 (ep 9) | 0.275 (ep 7) | 0.224 (ep 2) | 0.266 | 0.038 | 0.074 |

Collapse metrics (`collapse.json` per run; ranges over the 3 seeds; "init" is the untrained module, which is the same for all arms):

| | Init | random + dot | random + cosine | hard + cosine |
|---|---|---|---|---|
| z_r pairwise cosine | 0.99985 | 0.9998–0.9999 | 0.44–0.55 | 0.38–0.44 |
| z_f pairwise cosine | 0.99994 | 0.9999 | 0.17–0.26 | 0.14–0.18 |
| z_r top-1 singular energy | 0.9998 | 0.9998–0.9999 | 0.77–0.84 | 0.71–0.75 |
| z_r mean norm | 0.57 | 4.19–5.30 | 0.15–0.20 | 0.12–0.17 |
| \|b1\| vs mean \|W1 x\| | 0.083 vs 0.014 | 0.67–0.78 vs 0.05–0.06 | 0.25 vs 0.09–0.10 | 0.23–0.24 vs 0.08–0.10 |
| L_f positive: \|z_r\| / cos to anchor | 0.57 / 1.000 | 4.83–5.92 / 0.999–1.000 | 0.15–0.16 / 0.96–0.98 | 0.08–0.11 / 0.84–0.95 |
| L_f negative: \|z_r\| / cos to anchor | 0.57 / 1.000 | 4.14–5.26 / 1.000 | 0.15–0.20 / −0.61 to −0.73 | 0.12–0.17 / −0.02 to 0.05 |
| L_f per-epoch mean after epoch 0 | — | 0.3–1.3 | 0.9–1.1 | 1.5–1.85 |

### Pooled checkpoint selection (finding 7)

Same setup as above. The learned-weights arm uses `loss.weighting=learned`, `weighting_lr=0.01`. Rows are sorted by chosen epoch.

| Arm | Seed | Chosen epoch | Epochs run | Best valid acc | R@1 (mean 2019–22) |
|---|---|---|---|---|---|
| random + dot | 3 | 2 | 6 | 0.883 | 0.215 |
| hard + cosine | 3 | 2 | 6 | 0.871 | 0.224 |
| random + cosine, learned weights | 3 | 2 | 6 | 0.867 | 0.225 |
| random + cosine | 3 | 5 | 9 | 0.867 | 0.271 |
| hard + cosine | 2 | 7 | 11 | 0.871 | 0.275 |
| random + cosine | 52313 | 7 | 11 | 0.887 | 0.277 |
| random + dot | 52313 | 7 | 11 | 0.883 | 0.281 |
| random + dot | 2 | 8 | 12 | 0.871 | 0.277 |
| random + cosine, learned weights | 2 | 8 | 12 | 0.879 | 0.279 |
| random + cosine | 2 | 8 | 12 | 0.875 | 0.284 |
| random + cosine, learned weights | 52313 | 9 | 12 | 0.887 | 0.299 |
| hard + cosine | 52313 | 9 | 12 | 0.887 | 0.299 |

Pearson r(chosen epoch, R@1) = **0.97** (r² = 0.94), n = 12. Arm share of the variance is 6%, seed share 75%. r(best validation accuracy, R@1) = 0.41.

The extra-negatives gain over v2 is +0.3 points from a single seed, so it is not significant. The candidate-run outputs (config, metrics, eval, log) are also stored in `results/new_2019_candidate/`. The chart is at `results/summary_chart.png` (text-only control, paper-strict and paper + extra negatives in candidate mode with `--unmapped-adjustment mean`, and continual + extra negatives; regenerate with `python scripts/summary_chart.py`, which needs matplotlib).

## 6. Open questions / possible next work
- ~~Candidate-mode R@1 drops from 2019 to 2020+~~ — resolved, see finding 5.
- Multiple seeds would show whether the small differences are real. For the similarity/mining arms (finding 6), 3 seeds are not enough under the current checkpoint selection; fix the validation setup first (larger `valid_n` or a fixed epoch budget). **Finding 7 makes this the top open item:** re-evaluating all 12 batch runs (and the §5 single-seed runs) at a fixed epoch would test the checkpoint-selection explanation directly.
- Full-scale runs (no `debug_n`) and other train years have not been run. Note that train years 2021 and 2022 give L_f no signal.

## 7. Environment & how to run

- Python: `/home/kmpooja/envs/cycle-env/bin/python` (not a standard conda path).
- GPU: RTX 5000 Ada (32 GiB). **It is shared with other students' jobs, so never stop them.** Evaluation needs more than 10 GiB free. `scripts/gpu_queue.sh` waits for free memory before starting a job.
- BERT: `google/bert_uncased_L-8_H-512_A-8`, converted to safetensors.

```bash
SNAP="/DATA/kmpooja/hf_cache/hub/models--google--bert_uncased_L-8_H-512_A-8/snapshots/53b17ea0d90728ac770dfd13fcb3abdc75e4f4bf"
export CYCLE_BERT_PATH="$SNAP" PYTHON=/home/kmpooja/envs/cycle-env/bin/python
bash scripts/smoke_test.sh                 # quick sanity check
bash scripts/run_candidate.sh              # reproduces the candidate-mode runs
# overrides: --set graph.relation_extra_negatives=8 --set fusion.mode=candidate --set data.debug_n=1024
bash scripts/evaluate.sh <run_dir> 2019,2020,2021,2022 --unmapped-adjustment mean   # candidate mode, finding-5 fix
OUT_ROOT=/DATA/kmpooja/cycle_outputs bash scripts/run_mining.sh extranneg_cos:2 extranneg_hardcos:3   # finding 6 arms; OUT_ROOT keeps checkpoints off /home (it filled up once)
$PYTHON scripts/check_embedding_collapse.py --run-dir <run_dir> [--init]
OUT_ROOT=/DATA/kmpooja/cycle_outputs bash scripts/run_mining.sh extranneg_cos_learned:52313   # finding 7 (learned loss weights)
$PYTHON scripts/check_unmapped_bias.py --run-dir <run_dir> --test-years 2020 --unmapped-adjustment zero
```

Graph data is in `data/processed/graphs/<year>.npz` (n=10373, feat_dim=2929, K=8).
