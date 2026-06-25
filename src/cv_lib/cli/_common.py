"""Shared helpers for cvlib CLI subcommands."""

from __future__ import annotations

import os
import sys
from pathlib import Path


def configure_console() -> None:
    """Force UTF-8 on stdout/stderr.

    CLI output and help text use box-drawing and math characters (─, █, ×)
    that crash on legacy Windows code pages (cp1251/cp1252). Reconfiguring the
    streams to UTF-8 makes the CLI usable in those consoles.
    """
    for stream in (sys.stdout, sys.stderr):
        reconfigure = getattr(stream, "reconfigure", None)
        if reconfigure is not None:
            try:
                reconfigure(encoding="utf-8")
            except (ValueError, OSError):
                pass


def add_verbose(parser) -> None:
    """Register the shared ``--verbose`` flag on a (sub)parser."""
    parser.add_argument(
        "--verbose", action="store_true", help="Enable debug-level logging."
    )


def setup_logging(verbose: bool = False) -> None:
    """Configure loguru to log to stderr at INFO (or DEBUG when verbose)."""
    from loguru import logger

    logger.remove()
    logger.add(
        sys.stderr,
        level="DEBUG" if verbose else "INFO",
        format="{level}: {message}",
    )


def load_env() -> None:
    """Load variables from a .env file if python-dotenv is available.

    Searches from the current working directory upward (dotenv's default),
    so it works both when running from a checkout and from an installed package.
    """
    try:
        from dotenv import load_dotenv
    except ImportError:
        return
    load_dotenv()


def resolve_names(names: list[str] | None, data_yaml: str | None) -> list[str] | None:
    """Resolve class names from an explicit list or a YOLO data.yaml.

    Returns None if neither source is provided.
    """
    if data_yaml:
        from cv_lib.data import class_names_from_yaml

        return class_names_from_yaml(data_yaml)
    return names


def apply_data_root(path: str | Path) -> Path:
    """Prefix a relative path with DATA_ROOT when that env var is set."""
    p = Path(path)
    data_root = os.environ.get("DATA_ROOT", "")
    if not p.is_absolute() and data_root:
        return Path(data_root) / p
    return p
