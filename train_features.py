"""Compatibility wrapper for feature-sequence training CLI."""

from cityclassifiers.cli.train_features import *  # noqa: F403
from cityclassifiers.cli.train_features import main


if __name__ == "__main__":
    main()
