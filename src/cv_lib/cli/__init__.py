"""cvlib — unified command-line interface for cv_lib.

Run ``cvlib --help`` for the list of subcommands. Each subcommand mirrors a
script under ``scripts/`` and shares the same implementation, so behaviour is
identical whether invoked as ``cvlib eval ...`` or ``python scripts/eval.py ...``.
"""

from __future__ import annotations

import argparse

from cv_lib import __version__
from cv_lib.cli import (
    _augment,
    _autolabel,
    _bench,
    _compare,
    _compare_runs,
    _convert,
    _crops,
    _cvat_query,
    _dedup,
    _distribution,
    _dvc_init,
    _eval,
    _export,
    _fix,
    _infer,
    _inspect,
    _merge,
    _mine,
    _qa,
    _remap,
    _split,
    _threshold,
    _train,
)
from cv_lib.cli._common import configure_console, load_env, setup_logging

# Ordered registry: command name → implementation module.
# Each module exposes HELP, add_arguments(parser), run(args), and optional EPILOG.
COMMANDS = {
    "inspect": _inspect,
    "convert": _convert,
    "cvat-query": _cvat_query,
    "compare": _compare,
    "infer": _infer,
    "eval": _eval,
    "export": _export,
    "bench": _bench,
    "compare-runs": _compare_runs,
    "dvc-init": _dvc_init,
    "split": _split,
    "distribution": _distribution,
    "augment": _augment,
    "remap": _remap,
    "qa": _qa,
    "dedup": _dedup,
    "crops": _crops,
    "autolabel": _autolabel,
    "mine": _mine,
    "threshold": _threshold,
    "train": _train,
    "merge": _merge,
    "fix": _fix,
}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="cvlib",
        description="CV utility toolkit for iterating on object detection models.",
    )
    parser.add_argument(
        "--version", "-V", action="version", version=f"cvlib {__version__}"
    )

    sub = parser.add_subparsers(dest="command", metavar="<command>")
    for name, module in COMMANDS.items():
        epilog = getattr(module, "EPILOG", None)
        subparser = sub.add_parser(
            name,
            help=module.HELP,
            description=module.HELP,
            epilog=epilog,
            formatter_class=argparse.RawDescriptionHelpFormatter,
        )
        module.add_arguments(subparser)
        subparser.set_defaults(_run=module.run)

    return parser


def main(argv: list[str] | None = None) -> int:
    configure_console()
    load_env()
    parser = build_parser()
    args = parser.parse_args(argv)

    run = getattr(args, "_run", None)
    if run is None:
        parser.print_help()
        return 1

    setup_logging(getattr(args, "verbose", False))
    result = run(args)
    return result if isinstance(result, int) else 0


if __name__ == "__main__":
    raise SystemExit(main())
