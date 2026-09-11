"""Tests for load_env: .env loading and DATA_DIR -> HF cache derivation.

Every test that calls the real load_env() pre-registers the env vars it
touches via monkeypatch.setenv/delenv *before* calling it, so pytest's
monkeypatch fixture — which restores exactly the pre-test state (present or
absent) for any key it's been told about, regardless of what later code did
to that key — cleans up after load_dotenv's direct os.environ mutation, which
monkeypatch wouldn't otherwise know about.
"""

import os

from diffusion_moe.utils.env import load_env


def test_load_env_loads_variables_from_dotenv_file(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    (tmp_path / ".env").write_text("MY_TEST_VAR=hello\n")
    monkeypatch.delenv("MY_TEST_VAR", raising=False)

    load_env()

    assert os.environ.get("MY_TEST_VAR") == "hello"


def test_load_env_does_not_override_existing_env_vars(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    (tmp_path / ".env").write_text("MY_TEST_VAR=from_dotenv\n")
    monkeypatch.setenv("MY_TEST_VAR", "from_shell")

    load_env()

    assert os.environ["MY_TEST_VAR"] == "from_shell"


def test_load_env_handles_missing_dotenv_file_gracefully(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)  # no .env written here
    load_env()  # should not raise


def test_data_dir_derives_hf_cache_locations(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    monkeypatch.delenv("HF_HOME", raising=False)
    monkeypatch.delenv("HF_DATASETS_CACHE", raising=False)
    monkeypatch.setenv("DATA_DIR", str(tmp_path / "data"))

    load_env()

    assert os.environ["HF_HOME"] == str(tmp_path / "data" / "huggingface")
    assert os.environ["HF_DATASETS_CACHE"] == str(tmp_path / "data" / "huggingface" / "datasets")


def test_data_dir_does_not_override_explicit_hf_home(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("DATA_DIR", str(tmp_path / "data"))
    monkeypatch.setenv("HF_HOME", "/explicit/hf/home")

    load_env()

    assert os.environ["HF_HOME"] == "/explicit/hf/home"


def test_no_data_dir_leaves_hf_cache_vars_untouched(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    monkeypatch.delenv("DATA_DIR", raising=False)
    monkeypatch.delenv("HF_HOME", raising=False)
    monkeypatch.delenv("HF_DATASETS_CACHE", raising=False)

    load_env()

    assert "HF_HOME" not in os.environ
    assert "HF_DATASETS_CACHE" not in os.environ
