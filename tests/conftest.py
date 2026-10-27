"""Shared fixtures.

The suite is designed to run *without* the dataset: everything that needs DGL or
the preprocessed graphs is guarded with ``pytest.importorskip`` at module level,
so a bare ``pytest`` run still exercises the config, metric and RNG layers.

``tmp_dir`` replaces pytest's ``tmp_path`` because the latter needs write access to
``%TEMP%``, which is not always granted (e.g. inside a restricted sandbox).
"""

from __future__ import annotations

import os
import shutil
import sys
from pathlib import Path

import pytest

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO_ROOT not in sys.path:  # allows running `pytest` from any directory
    sys.path.insert(0, REPO_ROOT)

_TMP_ROOT = Path(REPO_ROOT) / ".pytest-tmp"


@pytest.fixture(autouse=True)
def _fresh_config():
    """Restore the default configuration after every test."""
    from cw_hgt import config

    config.reset_config()
    yield
    config.reset_config()


@pytest.fixture
def tmp_dir(request) -> Path:
    """A writable scratch directory inside the repository."""
    safe = "".join(ch if ch.isalnum() or ch in "-_." else "_" for ch in request.node.name)
    path = _TMP_ROOT / safe
    shutil.rmtree(path, ignore_errors=True)
    path.mkdir(parents=True, exist_ok=True)
    yield path
    shutil.rmtree(path, ignore_errors=True)
