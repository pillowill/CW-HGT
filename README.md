# CW-HGT

The model for lncRNA-abiotic stress association predictions using causal walk and heterogeneous graph transformer.

## Repository structure

```
code/     Source code of the model
dataset/  lncRNA / miRNA / stress data for eight plant species
```

`dataset/` covers *Arabidopsis thaliana*, *Gossypium hirsutum*, *Hordeum vulgare*,
*Medicago truncatula*, *Oryza sativa*, *Populus trichocarpa*, *Triticum aestivum*
and *Zea mays* (per-species `*_lncRNA.fa`, `*_lncRNA_stress.csv`, `*_miRNA.fa`,
`*_miRNA_stress.csv` and `mi_lnc.txt`).

## Code layout

| Path | Role |
| --- | --- |
| `code/main.py`, `code/new_main.py` | Training / evaluation entry points; CLI arguments are defined in `code/set_params.py` |
| `code/CausalWalk.py` | Causal walk sampling over the lncRNA–miRNA–stress graph |
| `code/CW_GNN.py`, `code/CW_HGT.py`, `code/GNNs.py`, `code/models.py` | Graph encoder and heterogeneous graph transformer modules |
| `code/bulit_graphs/` | Heterogeneous graph construction; `code/bulit_graphs/ab_graphs/` holds the ablation variants |
| `code/c_utils.pyx`, `code/c_utils.c`, `code/setup.py` | Cython helper used by the walk sampler |
| `code/CaseStudy.py`, `code/draw_auc.py`, `code/MetaRelationAttentionAnalysis.py`, `code/CGRepresentation.py`, `code/CW_ays.py` | Analysis and plotting scripts |
| `code/environment.yaml`, `code/requirements.txt` | Environment specification |

## Usage

From the repository root:

```bash
python code/main.py --DataClass lnc_mi_stress_graphs --pre_edge_type l_s --walk_type CW
```

The graph builders read derived inputs from `./dataset/adj` and `./dataset/sim`
relative to the working directory, so those files must be present there (or the
paths in `code/bulit_graphs/*.py` adjusted) before training. The walk sampler
uses the compiled `c_utils` extension; prebuilt Windows binaries for Python 3.9
and 3.11 are included, otherwise rebuild with:

```bash
cd code && python setup.py build_ext --inplace
```
