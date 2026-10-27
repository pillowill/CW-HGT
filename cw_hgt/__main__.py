"""Allow ``python -m cw_hgt``."""

from __future__ import annotations

import sys

from cw_hgt.cli import main

if __name__ == "__main__":
    sys.exit(main())
