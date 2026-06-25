#!/usr/bin/env python
"""Thin shim → ``cvlib infer``. See ``cvlib infer --help``.

Kept for backwards compatibility; the implementation lives in
``cv_lib.cli._infer`` and is shared with the unified ``cvlib`` CLI.

  python scripts/batch_infer.py --model best.pt --images dataset/images/val --save-labels
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from cv_lib.cli import _infer
from cv_lib.cli._common import configure_console, load_env, setup_logging


def main() -> None:
    configure_console()
    load_env()
    parser = argparse.ArgumentParser(description=_infer.HELP)
    _infer.add_arguments(parser)
    args = parser.parse_args()
    setup_logging(args.verbose)
    _infer.run(args)


if __name__ == "__main__":
    main()
