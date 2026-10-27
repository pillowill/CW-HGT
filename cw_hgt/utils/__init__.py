"""Small shared helpers: seeding, causal-walk conditioning, JSON output."""

from __future__ import annotations

from cw_hgt.utils.attention import META_PATH_ORDER, attn_arr
from cw_hgt.utils.io import (
    NumpyEncoder,
    ensure_dir,
    read_json,
    save_causal_walks,
    write_json,
)
from cw_hgt.utils.rng import (
    cal_conUs,
    get_perturbation,
    seed_everything,
    set_random_seed,
)

__all__ = [
    # rng
    "set_random_seed",
    "seed_everything",
    "cal_conUs",
    "get_perturbation",
    # io
    "NumpyEncoder",
    "ensure_dir",
    "save_causal_walks",
    "write_json",
    "read_json",
    # attention
    "attn_arr",
    "META_PATH_ORDER",
]
