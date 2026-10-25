"""Module entry point; importing this module does not run the CLI."""

from spire_learn.cli import main


if __name__ == "__main__":
    raise SystemExit(main())
