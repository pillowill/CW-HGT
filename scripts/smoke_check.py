"""Synthetic-data wiring check for the training pipeline.

Run this when you want to verify that ``cw_hgt`` is installed and wired correctly
*without* the real ``dataset/adj`` and ``dataset/sim`` artefacts:

    python scripts/smoke_check.py

It generates a tiny heterogeneous graph (20 lncRNA / 15 miRNA / 8 stress nodes),
runs a 2-fold, 2-epoch cross-validation on CPU and prints the metric summary.

Note: the block adjacency is computed with a NumPy replica of ``MYdataset.get_adj``
because that method goes through ``dgl.heterograph.adj()``, which requires DGL's
optional ``dgl_sparse`` C++ add-on. If your DGL build has it, the replica and the
real method produce the same matrix (same 3x3 block order, same 0/1 values).
"""

from __future__ import annotations

import os
import sys

import numpy as np
import pandas as pd

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
WORKDIR = os.path.join(REPO_ROOT, ".smoke-work")
sys.path.insert(0, REPO_ROOT)

N_LNC, N_MI, N_STRESS, DIM = 20, 15, 8, 5


def build_fixture(root: str) -> None:
    """Write a minimal dataset/adj + dataset/sim tree under ``root``."""
    rng = np.random.RandomState(0)
    adj_dir = os.path.join(root, "dataset", "adj")
    os.makedirs(adj_dir, exist_ok=True)
    links = {
        "lnc2stress_map.csv": [(rng.randint(0, N_STRESS), rng.randint(0, N_LNC)) for _ in range(10)],
        "mi2stress_map.csv": [(rng.randint(0, N_STRESS), rng.randint(0, N_MI)) for _ in range(8)],
        "mi2lnc_map.csv": [(rng.randint(0, N_MI), rng.randint(0, N_LNC)) for _ in range(6)],
    }
    for name, rows in links.items():
        pd.DataFrame(rows).to_csv(os.path.join(adj_dir, name), index=False, header=False)

    for sub in ("lnc", "mi", "stress"):
        os.makedirs(os.path.join(root, "dataset", "sim", sub), exist_ok=True)
    np.savez(os.path.join(root, "dataset/sim/lnc/lnc_gcrna_arr.npz"),
             rng.rand(N_LNC, DIM).astype("float32"))
    np.savez(os.path.join(root, "dataset/sim/mi/mi_gcrna_arr.npz"),
             rng.rand(N_MI, DIM).astype("float32"))
    np.savez(os.path.join(root, "dataset/sim/stress/stress_vec_array.npz"),
             rng.rand(N_STRESS, DIM).astype("float32"))


def dense_block_adj(g):
    """NumPy replica of ``MYdataset.get_adj`` (same block order as ``g.etypes``)."""
    counts = {t: g.num_nodes(t) for t in g.ntypes}
    blocks = []
    for etype in g.etypes:
        src_type, _, dst_type = g.to_canonical_etype(etype)
        src, dst = g.edges(etype=etype)
        block = np.zeros((counts[src_type], counts[dst_type]))
        block[src.numpy(), dst.numpy()] = 1.0
        blocks.append(block)
    return np.concatenate([
        np.concatenate(blocks[0:3], axis=1),
        np.concatenate(blocks[3:6], axis=1),
        np.concatenate(blocks[6:9], axis=1),
    ], axis=0)


def main() -> int:
    os.makedirs(WORKDIR, exist_ok=True)
    build_fixture(WORKDIR)
    os.chdir(WORKDIR)  # the builders resolve ./dataset relative to the cwd

    from cw_hgt import config
    from cw_hgt.data import get_dataset
    from cw_hgt.training import run_cross_validation

    config.configure(
        DataClass="lnc_mi_stress_graphs", pre_edge_type="l_s", walk_type="None",
        WalkLength=3, kNN=2, kFold=2, epochs=2, HidDim=8, OutDim=8, nLayers=2,
        nHeaders=2, isHetero=True, NumMultiWalk=1, device="cpu",
        output_dir=os.path.join(WORKDIR, "runs"), name="smoke", seed=3047,
    )
    cfg = config.get_config()

    dataset = get_dataset(cfg.DataClass)
    hg = dataset[0]
    print(f"[smoke] graph: {hg.num_nodes()=} {hg.num_edges()=}")
    dataset.get_adj = dense_block_adj

    result = run_cross_validation(dataset, hg, cfg, verbose=False)
    print(f"[smoke] metrics: {result.metrics.shape}")
    print("[smoke] " + "  ".join(f"{k}={v['mean']:.3f}" for k, v in result.summary().items()))
    print("[smoke] OK")
    return 0


if __name__ == "__main__":
    sys.exit(main())
