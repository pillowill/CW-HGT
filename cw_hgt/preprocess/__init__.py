"""Offline preprocessing of the plant lncRNA / miRNA / stress tables.

Requires the optional extras (``fasttext``, ``requests``, and the external
``RNA`` package). Nothing in this subpackage is imported by the training path,
which only consumes the artefacts it produces (``dataset/adj``, ``dataset/sim``).
"""

from __future__ import annotations

__all__: list = []
