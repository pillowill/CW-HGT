# Migration from the original scripts

The original checkout was a flat pile of entry scripts, each with its own copy of
the training loop and its own globals. This document maps every original file to
its place in v2, and lists every intentional behaviour change.

## File mapping

| original | v2 | notes |
| --- | --- | --- |
| `new_main.py` | `cw_hgt/training/cv.py` + `cw_hgt/cli.py` | the canonical k-fold loop (see below) |
| `main.py` | *retired* | superseded by `training/cv.py`; its `train`/`test` split-on-disk flow is replaced by `--save-checkpoints` |
| `draw_auc.py` | *retired* | third copy of the same loop; its AUC/PR curves can be computed from `CVResult.folds` |
| `set_params.py` | `cw_hgt/config.py` + `cw_hgt/cli.py` | argparse → dataclass; no import-time parsing |
| `utils.py` | `cw_hgt/utils/{rng,io,attention}.py` | dead helpers dropped (`rand_cat_fast`, `histcnd`, `cb_backdoor`, …) |
| `CausalWalk.py` | `cw_hgt/walk/causal.py` | bodies verbatim |
| `CW_GNN.py` | `cw_hgt/models/cw_gnn.py` | bodies verbatim |
| `GNNs.py` | `cw_hgt/models/encoders.py` | bodies verbatim, one default-argument fix |
| `CW_HGT.py` | `cw_hgt/models/cw_hgt.py` | kept as reference; not importable into the pipeline |
| `ranger.py` | `cw_hgt/optim/ranger.py` | vendored, attribution added |
| `bulit_graphs/lnc_mi_stress_graphs.py` | `cw_hgt/data/lnc_mi_stress.py` | bodies verbatim |
| `bulit_graphs/lnc_stress_graphs.py` | `cw_hgt/data/lnc_stress.py` | bodies verbatim |
| `bulit_graphs/mi_stress_graphs.py` | `cw_hgt/data/mi_stress.py` | bodies verbatim |
| `bulit_graphs/ab_graphs/*.py` | `cw_hgt/data/ablation/*.py` | now reachable via `--variant ablation` |
| `data_process.py` | `cw_hgt/preprocess/plant_data.py` | bodies verbatim |
| `CGRepresentation.py` | `cw_hgt/preprocess/cgr.py` | unchanged |
| `CaseStudy.py` | `cw_hgt/analysis/case_study.py` | script → function, no hard-coded row count |
| `CW_ays.py` | `cw_hgt/analysis/baselines.py` | script → function |
| `MetaRelationAttentionAnalysis.py` | *retired* | depends on attention capture that is commented out upstream (see known issues) |
| `models.py` | *dropped* | unrelated OGB/GAT tutorial code, imported by nothing |
| `hgttest.py` | *dropped* | scratch snippet |
| `train_val_test.py` | *dropped* | unused standalone splitter, duplicated inside the builders |
| `c_utils.pyx/.c/.pyd`, `setup.py` | repository root | unchanged; `setup.py` now only declares the extension |

## What "verbatim" means here

For the model, walker, encoder and data modules the *function and method bodies are
byte-identical* to the originals. Only these lines were touched:

* imports (`from bulit_graphs import …` → `from cw_hgt.data import …`, `ranger` →
  `cw_hgt.optim.ranger`, `set_params` → `cw_hgt.config`);
* `device = torch.device('cuda')` → `device = config.get_device()`;
* `args = set_params.get_args()` → `args = config.get_args()` (same attribute
  names, so no body needs to change);
* the PyCharm `@Time/@Author/@File` template headers, replaced by real module
  docstrings;
* one default argument in `HGTLayer.__init__` (see below).

This is what allows v2 to reproduce `new_main.py` numbers with the same seeds.

## Intentional behaviour changes

1. **`argparse` no longer runs at import.** `CW_GNN`, `CW_HGT` and `GNNs` used to
   call `parse_args()` on the host `sys.argv`; importing the package from a host
   program with unknown flags used to abort with `SystemExit(2)`.
2. **`HGTLayer(dropout=...)` default resolved inside the constructor.** It was
   bound to `args.dropout` at import time, so a config value set later was ignored.
3. **Metrics use `[[TN, FP], [FN, TP]]`.** `main.py` had the indices swapped
   (`new_main.py` was already correct); v2 follows `new_main.py`.
4. **`cal_conUs(num_nodes, seed)` is called with an explicit seed.** The retired
   `CW_ays.py` called `cal_conUs(..., is_test=True)`, which raised `TypeError`.
5. **Node counts come from the graph.** `CW_GNN` slices the block adjacency with
   `args.num_lnc/num_mi/num_stress`; v2 overwrites them from `hg.num_nodes()`
   before construction, so a species with different sizes cannot be mis-sliced.
6. **Outputs go to `--output-dir`** (`runs/` by default) instead of the Chinese
   directory names `参数设置/`, `定量分析/`, `案例分析/`.
7. **Directories are created on demand** (`utils.ensure_dir`); the original raised
   `FileNotFoundError` when saving walks.
8. **Device defaults to `auto`** (CUDA when available, else CPU) instead of
   hard-wired CUDA, so the CLI can at least run its checks on a CPU-only machine.
9. **`seed_everything` also seeds Python and NumPy**, in addition to PyTorch.

## Reproducing a `new_main.py` run

```bash
# equivalent of:  python new_main.py            (lnc_mi_stress_graphs, l_s, CW)
cw-hgt train --task lnc_mi_stress_graphs --pre-edge-type l_s --walk-length 5 \
             --layers 5 --headers 2 --epochs 200 --folds 10 --seed 3047
```

`new_main.py` iterated a grid hard-coded to a single point
(`walk_length_list=[5]`, `HGT_layers_list=[5]`, `headers_list=[2]`); pass comma
separated values to scan the grid instead.
