# CW-HGT

Predicting **lncRNA / miRNA – abiotic stress associations** in plants with a
causal-walk sampler and a heterogeneous graph transformer.

The model builds a heterogeneous graph over three node types (`lnc`, `mi`,
`stress`), samples causal walks that use a backdoor-adjusted transition
probability, and fuses the resulting meta-paths with an HGT encoder plus a
dot-product link predictor. Evaluation is stratified 10-fold cross-validation on
the held-out association edges.

```
                 ┌───────────────────────┐
 lncRNA/miRNA ──▶│ heterogeneous graph   │──▶ causal walk (backdoor-adjusted)
 /stress tables  │ l_s · m_s · l_m · knn │            │
                 └───────────────────────┘            ▼
                                              HGT encoder (CW-GNN)
                                                        │
                                              dot-product predictor ──▶ P(lnc–stress)
```

---

## What's new in v2

This branch is the same model, re-packaged as an importable, installable project:

| Before | v2 |
| --- | --- |
| `main.py`, `new_main.py`, `draw_auc.py` each carried a copy of the training loop | one loop: `cw_hgt.training.run_cross_validation` |
| `set_params.py` ran `argparse.parse_args()` **at import**, from three modules, so the package could not be imported | one mutable config (`cw_hgt.config.Config`), parsed only by the CLI |
| hyper-parameters reached the model through module globals | `Config` defaults + `--flag` + `--config file.yaml` |
| `bulit_graphs/` (typo) with a hidden duplicate `ab_graphs/` copy | `cw_hgt/data/`, ablation variants explicit and selectable with `--variant` |
| Chinese output folders (`参数设置/`, `定量分析/`, `案例分析/`) hard-coded in scripts | `--output-dir` (default `runs/`) |
| no packaging, no tests, no docs | `pyproject.toml`, console script, pytest suite, `docs/` |

The numerical path is unchanged: model, data and walker bodies were moved
verbatim (only imports and module headers were rewritten), so the same seeds give
the same results as `new_main.py`. See [`docs/migration.md`](docs/migration.md)
for the file-by-file mapping and [`docs/known_issues.md`](docs/known_issues.md)
for defects inherited from the original code.

## Install

```bash
conda env create -f environment.yml     # or: pip install -r requirements.txt
conda activate cwhgt
pip install -e .                        # also installs the `cw-hgt` console script
```

The causal-walk sampler needs the `c_utils` Cython extension. A prebuilt
Windows / CPython 3.11 binary is committed, so the common case needs no
compiler; otherwise build it in place:

```bash
python setup.py build_ext --inplace      # requires Cython + a C toolchain
```

Without the extra package you can still run everything from the repository root:

```bash
python -m cw_hgt info
```

## Data

The repository ships the raw per-species tables under `dataset/`
(eight species: *Arabidopsis thaliana*, *Gossypium hirsutum*, *Hordeum vulgare*,
*Medicago truncatula*, *Oryza sativa*, *Populus trichocarpa*, *Triticum aestivum*,
*Zea mays*).

The graph builders additionally expect **preprocessed** artefacts, which are not
committed:

```
dataset/adj/lnc2stress_map.csv     dataset/sim/lnc/lnc_gcrna_arr.npz
dataset/adj/mi2lnc_map.csv         dataset/sim/mi/mi_gcrna_arr.npz
dataset/adj/mi2stress_map.csv      dataset/sim/stress/stress_vec_array.npz
```

Produce them with `cw_hgt.preprocess` (needs `fasttext`, `requests` and the
external `RNA` package) or drop your own files there. See
[`docs/data.md`](docs/data.md) for the exact schema — this is the one manual step
that stands between a clone and a training run.

## Quick start

```bash
# 10-fold CV on the default task, 200 epochs, evaluated on the held-out l_s edges
cw-hgt train --task lnc_mi_stress_graphs --pre-edge-type l_s --epochs 200

# hyper-parameter sweep over walk length / depth / heads
cw-hgt train --walk-length 5,10,15 --layers 2,5 --headers 2,4

# ablation variant, all settings from a file
cw-hgt train --config configs/ablation.yaml --output-dir runs/ablation

# what would run?
cw-hgt info --config configs/default.yaml
```

`cw-hgt train` writes `runs/<task>/<edge>/metrics.json` with per-fold and
aggregated AUC / AUPR / precision / recall / accuracy / F1 / specificity.

### Python API

```python
from cw_hgt import config
from cw_hgt.data import get_dataset
from cw_hgt.training import run_cross_validation

config.configure(DataClass="lnc_mi_stress_graphs", WalkLength=5, nLayers=5,
                 nHeaders=2, epochs=200, kFold=10)

dataset = get_dataset("lnc_mi_stress_graphs")   # reads ./dataset/{adj,sim}
hg = dataset[0]
result = run_cross_validation(dataset, hg)
print(result.summary())        # {'auc': {'mean': ..., 'std': ...}, ...}
```

Downstream analyses:

```python
from cw_hgt.analysis.case_study import run_case_study
from cw_hgt.analysis.baselines import evaluate_walk_baselines

run_case_study(dataset, hg, out_csv="runs/case_study.csv")
evaluate_walk_baselines(dataset, hg, walk_type="N2VRW", walk_length=50)
```

## Repository layout

```
cw_hgt/
├── config.py            Config dataclass, shared active instance, device resolution
├── constants.py         task registry (node/edge types per --DataClass)
├── cli.py               cw-hgt train | info
├── data/                heterogeneous graph builders (one module per task)
│   └── ablation/        ablation variants (--variant ablation)
├── walk/causal.py       causal walk + RW / N2VRW baselines (uses c_utils)
├── models/
│   ├── cw_gnn.py        the model: CW-GNN
│   ├── encoders.py      GCN / GIN / HGT layers
│   └── cw_hgt.py        legacy HGT-only variant (reference only)
├── training/            losses, metrics, cross-validation loop
├── analysis/            case study, walk baselines
├── preprocess/          offline data preparation (optional extras)
└── optim/ranger.py      vendored Ranger optimiser
c_utils.pyx / .c / .pyd  Cython sampler kernel
configs/                  example run configurations
docs/                     architecture, data schema, migration, known issues
tests/                    pytest suite (no dataset required)
```

## Reproducibility

* fold *i* is seeded with `seed - i` (default `3047`), for both PyTorch and the
  conditioning vector `Con_Us` — the same convention as the original scripts;
* `cw_hgt.utils.seed_everything` additionally seeds Python and NumPy, so a run is
  reproducible from its first call rather than from the first fold;
* the node counts passed to the model are always re-read from the built graph
  (the original read them from CLI defaults, which could silently mis-slice the
  block adjacency when species differ in size);
* `docs/environment.lock.yml` records the full conda export used for the results.

## Known limitations

Inherited defects that were deliberately **not** fixed, because fixing them would
change published numbers, are listed in [`docs/known_issues.md`](docs/known_issues.md)
— including the unshuffled `m_s` folds, the dead `RandomWalk` baseline and the
disabled meta-path attention capture.

## Citation

If you use this code, please cite it via [`CITATION.cff`](CITATION.cff) (and the
accompanying paper).

## License

MIT — see [`LICENSE`](LICENSE). `cw_hgt/optim/ranger.py` is vendored under
Apache-2.0 by its original authors.
