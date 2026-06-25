#!/usr/bin/env python
"""Thin shim → ``cvlib compare-runs``. See ``cvlib compare-runs --help``.

Kept for backwards compatibility; the implementation lives in
``cv_lib.cli._compare_runs`` and is shared with the unified ``cvlib`` CLI.

  python scripts/compare_runs.py runs/train/exp1 runs/train/exp2
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from cv_lib.cli import _compare_runs
from cv_lib.cli._common import configure_console, load_env, setup_logging


def main() -> None:
    configure_console()
    load_env()
    parser = argparse.ArgumentParser(
        description=_compare_runs.HELP,
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=_compare_runs.EPILOG,
    )
    _compare_runs.add_arguments(parser)
    args = parser.parse_args()
    setup_logging(args.verbose)
    _compare_runs.run(args)


if __name__ == "__main__":
    main()
