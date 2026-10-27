# Known issues

Defects found while re-packaging the code. Anything that would change a reported
number is **documented rather than fixed** — v2 is meant to reproduce the paper's
results, not to silently improve on them. Each item states the v2 status.

## Fixed during the refactor

| # | issue | where | fix |
| --- | --- | --- | --- |
| 1 | `argparse.parse_args()` executed at import time (three modules), aborting any host process with unknown flags | `models/cw_gnn.py`, `models/encoders.py`, `models/cw_hgt.py` | config singleton |
| 2 | `HGTLayer(dropout=args.dropout)` froze the default at import | `models/encoders.py` | `dropout=None` + resolve in the constructor |
| 3 | confusion-matrix indices swapped, corrupting specificity | retired `main.py` | `training/metrics.py` uses `[[TN, FP], [FN, TP]]` |
| 4 | `cal_conUs(num_nodes, is_test=True)` — argument does not exist | retired `CW_ays.py` | `analysis/baselines.py` passes a seed |
| 5 | block adjacency sliced with CLI defaults instead of the actual node counts | `models/cw_gnn.py` callers | `training/cv.py` copies `hg.num_nodes(...)` into the config first |
| 6 | saving walks/metrics crashed when the output directory was missing | `utils` | `ensure_dir` before every write |
| 7 | `if ~os.path.exists(...)` (always true; `~True == -2`) | retired `main.py` | path removed from v2 |
| 8 | case study reshaped scores to a hard-coded `8498` rows | `analysis/case_study.py` | shape taken from `np.nonzero` |
| 9 | `main.py` unpacked 4 values from a 3-value `forward`, and omitted the required `n_layers`/`n_headers` | retired `main.py` | single arity, explicit kwargs |

## Documented, not fixed

| # | issue | impact |
| --- | --- | --- |
| 1 | **`m_s` folds are never shuffled.** `mi_stress` edges are shuffled, but the fold indices are rebuilt from a fresh `arange`, so the folds are contiguous blocks. | Fold composition differs from a random split; per-fold variance is understated. |
| 2 | **`RandomWalk.random_walks` raises `ValueError`.** `walks[0].numpy()` is 1-D and cannot be concatenated with an `(n, 1)` array. | The `RW` baseline is unusable; `N2VRW` works. |
| 3 | **Meta-path attention capture is disabled.** The `attn_edge_score`/`msg_edge_score` accumulation in `HGTLayer.forward` is commented out, so it always returns `([], {})` and `utils.attn_arr` cannot work. | `MetaRelationAttentionAnalysis.py` was retired; the attention path is dead. |
| 4 | **`HGTLayer` appends to `self.n_layers_att` on every forward without clearing.** | Unbounded memory/autograd retention across epochs when the capture lines are re-enabled. |
| 5 | **`lnc_stress_graphs` ignores `--kNN`** — `k=11` is hard-coded — and its `split_train_test` neither seeds nor shuffles, unlike its siblings. | Not directly comparable to the other two tasks. |
| 6 | **`custom_knn_graph_blas` ignores its `dist`/`algorithm` arguments.** It always uses Euclidean distance, and because the diagonal is zero, `topk(largest=False)` includes the node itself and half of the selected neighbours are the *farthest* ones. | Similarity edges are noisier than the docstring suggests. |
| 7 | **`l_m` / `m_l` edges are copied into both the train and test subgraphs.** | Information leaks across folds for the two-type aux relations. |
| 8 | **Causal-walk transition probabilities are normalized twice** (`c_utils.compute_transition_probs` already normalizes). | Scale factor only; argmax selection is unaffected. |
| 9 | **`causal_walk` indexes the neighbour list with an `argmax` over a dict.** `c_utils.cb_backdoor` fills that dict inside a `prange`, so insertion order is not guaranteed. | Potential node/probability misalignment; a Python fallback would need an ordered mapping. |
| 10 | **`CW_HGT.CW_GNN` cannot be instantiated** (`self.mlp_predictor` is never created) and its `forward` has three separate failures (single-value return read as a pair, attributes set only by an uncalled method, implicit `None`). | Kept in `models/cw_hgt.py` as reference only; do not wire it in. |
| 11 | **`lnc_stress` splits use `k=11` while `Config.kNN` defaults to 15.** | Changing `--knn` has no effect on that task. |
| 12 | Unused code kept for fidelity: `MLP_Predictor`, `Hadmard`, `Concatenate`, `GIN`, `dglHGT`. | None; they are not exported. |

## Environment

`dataset/adj` and `dataset/sim` are **not** in the repository, and no code path
creates them. Until they exist (see [`data.md`](data.md)), every builder fails in
`process()` with `FileNotFoundError`. This is inherited behaviour, now at least
reported by the CLI before training starts.
