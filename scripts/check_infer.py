#!/usr/bin/env python
"""Thin shim → ``cvlib bench``. See ``cvlib bench --help``.

Kept for backwards compatibility; the implementation lives in
``cv_lib.cli._bench`` and is shared with the unified ``cvlib`` CLI.

  python scripts/check_infer.py --model best.pt --imgsz 320 640 1280
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from cv_lib.cli import _bench
from cv_lib.cli._common import configure_console, load_env, setup_logging


def main() -> None:
    configure_console()
    load_env()
    parser = argparse.ArgumentParser(
        description=_bench.HELP,
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=_bench.EPILOG,
    )
    _bench.add_arguments(parser)
    args = parser.parse_args()
    setup_logging(args.verbose)
    _bench.run(args)


if __name__ == "__main__":
    main()
