"""Loads .env into the process environment for every script entrypoint."""

from __future__ import annotations

import os
from pathlib import Path

from dotenv import find_dotenv, load_dotenv


def load_env() -> None:
    """Loads a .env file into os.environ (searched for from the current
    working directory upward, via python-dotenv's find_dotenv — matches
    running scripts from the repo root, the documented usage throughout this
    project), then derives HF_HOME / HF_DATASETS_CACHE from DATA_DIR when
    it's set, so DATA_DIR actually controls where HuggingFace caches
    downloaded models/datasets instead of being an inert placeholder.

    Variables already present in the environment always win: load_dotenv
    doesn't override them, and the HF cache vars are only set if not already
    set (os.environ.setdefault), so an operator's own HF_HOME/DATA_DIR
    choices are respected either way.
    """
    load_dotenv(find_dotenv(usecwd=True))

    data_dir = os.environ.get("DATA_DIR")
    if data_dir:
        os.environ.setdefault("HF_HOME", str(Path(data_dir) / "huggingface"))
        os.environ.setdefault(
            "HF_DATASETS_CACHE", str(Path(data_dir) / "huggingface" / "datasets")
        )
