"""Tests for the TOML config file loading (nuclearcutter/utils/config.py)."""

import tomllib
from pathlib import Path

import pytest

from nuclearcutter.utils.config import AppConfig, default_config_path, load_config


def test_defaults_when_no_file(tmp_path):
    """load_config with a nonexistent path returns all defaults."""
    cfg = load_config(tmp_path / "nope.toml")
    assert cfg.model_backend == "mlx-vlm"
    assert cfg.base_url == "http://localhost:1234/v1"
    assert cfg.sweep_interval == 2.0
    assert cfg.nudity == "blur"
    assert cfg.foul_language == "mute"


def test_loads_known_keys(tmp_path):
    p = tmp_path / "config.toml"
    p.write_text(
        "model_backend = \"standalone\"\n"
        "base_url = \"http://localhost:11434/v1\"\n"
        "sweep_interval = 2.0\n"
        "nudity = \"none\"\n"
        "mute_scope = \"utterance\"\n"
    )
    cfg = load_config(p)
    assert cfg.model_backend == "standalone"
    assert cfg.base_url == "http://localhost:11434/v1"
    assert cfg.sweep_interval == 2.0
    assert cfg.nudity == "none"
    assert cfg.mute_scope == "utterance"


def test_ignores_unknown_keys(tmp_path):
    """Unknown keys (e.g. newer configs) must not crash old versions."""
    p = tmp_path / "config.toml"
    p.write_text("model_backend = \"mlx-vlm\"\nfuture_setting = 42\n")
    cfg = load_config(p)
    assert cfg.model_backend == "mlx-vlm"


def test_invalid_enum_raises(tmp_path):
    p = tmp_path / "config.toml"
    p.write_text("nudity = \"zoom\"\n")
    with pytest.raises(ValueError, match="nudity"):
        load_config(p)


def test_invalid_backend_raises(tmp_path):
    p = tmp_path / "config.toml"
    p.write_text("model_backend = \"cloud\"\n")
    with pytest.raises(ValueError, match="model_backend"):
        load_config(p)


def test_default_config_path_is_repo_root():
    """The default config path should be <repo>/config.toml, i.e. not inside
    the nuclearcutter package."""
    p = default_config_path()
    assert p.name == "config.toml"
    # one parent up from nuclearcutter/utils
    assert p.parent.name == "NuclearCutter"


def test_repo_config_toml_is_valid_toml():
    """The checked-in config.toml template must be parseable."""
    p = default_config_path()
    if not p.exists():
        pytest.skip("no checked-in config.toml")
    with open(p, "rb") as f:
        tomllib.load(f)
