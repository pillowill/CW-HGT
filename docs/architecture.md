# Architecture

## Module graph

```
cli.py ──▶ config.Config (single mutable instance)
   │            ▲
   │            └── read by models/data/walk modules as `config.get_args()`
   ├──▶ data/lnc_mi_stress.py · lnc_stress.py · mi_stress.py      (DGLDataset)
   │        └── builds the heterogeneous graph + dense block adjacency
   ├──▶ training/cv.py ──▶ models/cw_gnn.py ──▶ walk/causal.py ──▶ c_utils (Cython)
   │                     └── models/encoders.py (GCN / HGT)
   └──▶ analysis/{case_study,baselines}.py
```

`config`, `constants` and `utils` have **no** third-party imports beyond NumPy, so
they can be imported in a bare environment; everything else pulls in DGL/PyTorch
on demand.

## Graph construction

Each builder wraps the per-species tables into a `dgl.DGLHeteroGraph`:

| task (`--DataClass`) | node types | association edges | similarity edges |
| --- | --- | --- | --- |
| `lnc_mi_stress_graphs` | lnc, mi, stress | `l_s`, `m_s`, `l_m` (+ reverses) | `l_l`, `m_m`, `s_s` (kNN) |
| `lnc_stress_graphs` | lnc, stress | `l_s` (+ reverse) | `l_l`, `s_s` |
| `mi_stress_graphs` | mi, stress | `m_s` (+ reverse) | `m_m`, `s_s` |

`get_adj()` additionally returns the dense block adjacency the model consumes:

```
[[S_lnc_lnc, A_lnc_mi,  A_lnc_stress ],
 [A_lnc_mi.T, S_mi_mi,  A_mi_stress.T],
 [A_lnc_stress.T, A_mi_stress, S_stress_stress]]
```

so that a single `N×N` matrix covers intra-type similarity and inter-type links.

## Model

`CW_GNN` (in `models/cw_gnn.py`) is the model the results come from:

1. **Causal walk.** `MultiCausalWalk` samples `NumMultiWalk` walks per node. The
   transition probability to a neighbour is proportional to a backdoor-adjusted
   causal weight computed in `c_utils.cb_backdoor`, then re-weighted by the
   return/in-out parameters `p`, `q` (`c_utils.compute_transition_probs`).
2. **Walk encoder.** Each walk becomes an adjacency block; node features are
   aggregated over the resulting meta-paths.
3. **HGT stack.** `nLayers` `HGTLayer`s (`models/encoders.py`) run typed attention
   over the nine meta-relations, with `nHeaders` heads per layer.
4. **Prediction.** `HeteroDotProductPredictor` scores edges of the target type.

`forward` returns `(pos_score, neg_score, causal_walks_dict)` for
`is_hetero=True` and appends a trailing `None` for the homogeneous path. The
third element is `None` unless the walk dictionary was requested.

`models/cw_hgt.py` is an older, HGT-only variant kept for reference. It is **not**
wired into the pipeline and does not currently instantiate — see
[`known_issues.md`](known_issues.md).

## Training loop

`training/cv.py::run_cross_validation` implements the consolidated `new_main.py`
semantics:

```
for fold in range(kFold):
    seed  = cfg.seed - fold                       # 3047 - i
    train on perturbed features, perturbed graph  # Ranger + BCE/margin loss
    evaluate on unperturbed features, test edges  # perturbation is train-time only
      → calc_metrics(scores, labels)              # 7 numbers per fold
```

Two configuration details are load-bearing:

* node counts are copied into `cfg` from `hg.num_nodes(...)` *before* the model is
  constructed — the model slices the block adjacency with them;
* the evaluation pass deliberately reuses the training fold's adjacency while
  swapping in the test edge sets, matching the reference implementation.

## Configuration

`config.Config` is a dataclass whose field names mirror the original argparse
destinations (`DataClass`, `HidDim`, `WalkLength`, ...). A single instance is
shared process-wide; `config.configure(**values)` mutates it. This keeps the
migrated module bodies (`args.walk_type`, `args.num_lnc`, ...) untouched while
removing the import-time `parse_args()` that made the code unimportable.

`validate()` fails fast on combinations the original crashed on late, e.g.
`nHeaders=0` (division by zero in `HGTLayer`) or `nLayers=0` (no layer built).
