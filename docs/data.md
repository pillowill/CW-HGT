# Data

## What is in the repository

`dataset/<species>/` holds the raw tables for eight species:

| file | content |
| --- | --- |
| `<species>_lncRNA.fa` | lncRNA sequences (FASTA) |
| `<species>_miRNA.fa` | miRNA sequences (FASTA) |
| `<species>_lncRNA_stress.csv` | curated lncRNA–stress associations |
| `<species>_miRNA_stress.csv` | curated miRNA–stress associations |
| `mi_lnc.txt` | miRNA–lncRNA interactions |

## What the graph builders need

The builders read **preprocessed** artefacts from `dataset/adj` and `dataset/sim`,
resolved against the working directory (or `--data-dir`). These are *not*
committed because they are derived.

| path | shape / format | producer |
| --- | --- | --- |
| `adj/lnc2stress_map.csv` | integer pairs `lnc_id,stress_id` | `preprocess.plant_data` |
| `adj/mi2lnc_map.csv` | integer pairs `mi_id,lnc_id` | " |
| `adj/mi2stress_map.csv` | integer pairs `mi_id,stress_id` | " |
| `sim/lnc/lnc_gcrna_arr.npz` | `arr_0`: `(n_lnc, d)` float features | " |
| `sim/mi/mi_gcrna_arr.npz` | `arr_0`: `(n_mi, d)` float features | " |
| `sim/stress/stress_vec_array.npz` | `arr_0`: `(n_stress, d)` float features | " |

The feature arrays are the CGR (chaos-game-representation) encodings of the
sequences, reduced to a fixed-length vector; the stress vectors come from a
fastText embedding of the stress vocabulary.

Node counts follow the files, not the CLI: `Config.num_lnc / num_mi / num_stress`
are overwritten from the built graph before the model slices its block adjacency.

## Regenerating the derived artefacts

```bash
pip install -e ".[preprocess]"      # fasttext + requests
# plus the external `RNA` package used for secondary structure
python -c "from cw_hgt.preprocess.plant_data import Plant_data_process"
```

`preprocess/plant_data.py` expects the raw tables above plus
`fastText_PreTrained/cc.en.300.bin` (download separately, ~7 GB). It writes the
`adj/` and `sim/` trees. Paths inside it are the ones from the original script;
adjust them for your checkout.

## Tasks

| `--DataClass` | predicted edge | node types |
| --- | --- | --- |
| `lnc_mi_stress_graphs` | `l_s` or `m_s` | lnc, mi, stress |
| `lnc_stress_graphs` | `l_s` | lnc, stress |
| `mi_stress_graphs` | `m_s` | mi, stress |

The ablation variants (`cw_hgt/data/ablation/`) differ in three ways: features are
standardised with `StandardScaler`, the negative-sample graph is symmetrised and
padded to fixed node blocks, and off-diagonal adjacency blocks are zero-filled.
They are separate modules rather than a flag so the default path stays numerically
identical to the reported results.
