# CYCLE Reproduction — Progress Notes

Last updated: 2026-09-29. Branch: `cycle-implementation`.

Goal: reproduce CYCLE (Cross-Year Contrastive Learning in Entity Linking, CIKM 2024) on GCL-TempEL and compare recall@k with paper Table 2 (continual: 0.286–0.301, new: 0.184–0.222).

---

## 1. Where things stand

- All model code is implemented and committed. Smoke tests pass for both `baseline` and `paper` configs.
- The debug-scale experiments for train year 2019 are finished (`debug_n=1024`, seed 52313).
- The candidate-mode drop from 2019 to 2020+ is diagnosed and fixed at evaluation time (finding 5, commit `c92103c`).
- **Next step: the writeup.** Experiments are finished; no more runs.

## 2. Commits on this branch

| Commit | What |
|---|---|
| `e066de6` | Model files (BiEncoder, ScoreFusion, PaperGraphModule, HeCo) for paper Eq. 1–15; `graph.relation_extra_negatives` extension; `--set` override flag; `config.apply_overrides`; `evaluate.py` infers the fusion bias from the checkpoint; `.gitignore` `models/` rule anchored to the repo root |
| `c9efe09` | `scripts/diag_negatives.py` (Eq. 11 zero-negatives diagnostic), `scripts/summary_chart.py` (chart; now writes `results/summary_chart.png`) |
| `6e248bd` | Candidate-mode fusion experiments: `scripts/run_candidate.sh`, `scripts/gpu_queue.sh`, results in `results/new_2019_candidate/` |

| `000cb57` | Explicit `data.valid_n: null` and `fusion.mode: gold_mention` in both configs; `train.py` rejects unknown `--set` keys (previously `model.relation_extra_negatives` was silently accepted; the correct key is `graph.relation_extra_negatives`). `evaluate.py` already rejected them. Adds this file. |
| `c92103c` | `evaluate.py --unmapped-adjustment {zero,mean}` (default `zero`, the original behaviour); `scripts/check_unmapped_bias.py` (finding 5) |
| (this commit) | Candidate-mode results after the fix: `mean` rows in `results/new_2019_candidate/*/eval_results.jsonl`, `unmapped_bias_check.jsonl`; this file updated |

## 3. Code map

| File | Notes |
|---|---|
| `src/cycle/models/biencoder.py` | Two BERT towers. The attribute must be named `bert_model` because `optim.py` matches `bert_model.encoder.layer` for differential LR. `CYCLE_BERT_PATH` env var overrides the model path. |
| `src/cycle/models/fusion.py` | `ScoreFusion`: additive tanh-gated fusion. Must keep the name `.linear` because `evaluate.py` reassigns it. `learned` projection is zero-init; `random_frozen` is random and frozen. `qid` or `sequential` alignment. |
| `src/cycle/models/graph_paper.py` | `PaperGraphModule`: GAT attention (Eq. 6/7), shared MLP (Eq. 8/9), L_f (Eq. 13), L_r (Eq. 14). Call `resample_neighbors()` once per epoch. |
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
3. **Candidate-mode fusion** (with the finding-5 fix, `--unmapped-adjustment mean`): the paper-strict version is *worse* than the text-only control (R@1 ≈ 0.22 vs ≈ 0.25 in every test year). With extra negatives it beats the control (≈ 0.28) and roughly matches the gold_mention runs (0.280–0.282 vs 0.279–0.286), but does not clearly exceed them. So test-time graph fusion adds little on top of what the graph already contributes through training.
4. **Continual loss_e reaches about 0.01, which means memorization.** Report this plainly.
5. **Entities absent from the train-year graph break candidate-mode evaluation.** The paper does not say how to score entities that appear after the training year. In test 2020–22, 373/10373 pool entities have no 2019-graph node, and `enhance_pool` gave them a zero adjustment, which never occurs in training. They then took the top-1 slot far above their 3.6% fair share: 35.7% of mentions for paper-strict (whose adjustment lowers mapped scores by 5.3 on average) and 11.5% with extra negatives, against 3.3–3.4% with no adjustment at all. Giving them the mean adjustment of mapped entities (`--unmapped-adjustment mean`) brings the rate to 3.2–3.7% (matching the no-adjustment control) and removes the 2019→2020+ drop. Test 2019 (all entities mapped) is bit-identical. `scripts/check_unmapped_bias.py` reproduces this; output in `results/new_2019_candidate/unmapped_bias_check.jsonl`.

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

The extra-negatives gain over v2 is +0.3 points from a single seed, so it is not significant. The candidate-run outputs (config, metrics, eval, log) are also stored in `results/new_2019_candidate/`. The chart is at `results/summary_chart.png` (text-only control, paper-strict and paper + extra negatives in candidate mode with `--unmapped-adjustment mean`, and continual + extra negatives; regenerate with `python scripts/summary_chart.py`, which needs matplotlib).

## 6. Open questions / possible next work
- ~~Candidate-mode R@1 drops from 2019 to 2020+~~ — resolved, see finding 5.
- Multiple seeds would show whether the small differences are real.
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
$PYTHON scripts/check_unmapped_bias.py --run-dir <run_dir> --test-years 2020 --unmapped-adjustment zero
```

Graph data is in `data/processed/graphs/<year>.npz` (n=10373, feat_dim=2929, K=8).
