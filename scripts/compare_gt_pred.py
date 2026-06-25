#!/usr/bin/env python
"""Thin shim → ``cvlib compare``. See ``cvlib compare --help``.

Kept for backwards compatibility; the implementation lives in
``cv_lib.cli._compare`` and is shared with the unified ``cvlib`` CLI.

  python scripts/compare_gt_pred.py image.jpg --model best.pt --names car person
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from cv_lib.cli import _compare
from cv_lib.cli._common import configure_console, load_env, setup_logging


def main() -> None:
    configure_console()
    load_env()
    parser = argparse.ArgumentParser(description=_compare.HELP)
    _compare.add_arguments(parser)
    args = parser.parse_args()
    setup_logging(args.verbose)
    _compare.run(args)


if __name__ == "__main__":
    main()
