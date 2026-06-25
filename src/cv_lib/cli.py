"""Minimal cvlib entry point — full subcommands live in P0."""

import argparse

from cv_lib import __version__


def main() -> None:
    parser = argparse.ArgumentParser(
        prog="cvlib",
        description="CV utility library for object detection model iteration.",
    )
    parser.add_argument(
        "--version", "-V",
        action="version",
        version=f"cvlib {__version__}",
    )
    parser.parse_args()


if __name__ == "__main__":
    main()
