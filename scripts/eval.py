#!/usr/bin/env python
"""Thin shim → ``cvlib eval``. See ``cvlib eval --help``.

Kept for backwards compatibility; the implementation lives in
``cv_lib.cli._eval`` and is shared with the unified ``cvlib`` CLI.

  python scripts/eval.py --model runs/train/best.pt --data dataset/data.yaml
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from cv_lib.cli import _eval
from cv_lib.cli._common import configure_console, load_env, setup_logging


def main() -> None:
    configure_console()
    load_env()
    parser = argparse.ArgumentParser(description=_eval.HELP)
    _eval.add_arguments(parser)
    args = parser.parse_args()
    setup_logging(args.verbose)
    _eval.run(args)


if __name__ == "__main__":
    main()
