"""Enable ``python -m cv_lib.cli`` as an alias for the ``cvlib`` entry point."""

from cv_lib.cli import main

if __name__ == "__main__":
    raise SystemExit(main())
