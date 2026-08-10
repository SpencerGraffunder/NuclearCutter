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
    assert cfg.nudity_visual == "blur"
    assert cfg.nudity_audio == "none"
    assert cfg.gore_visual == "blur"
    assert cfg.violence_visual == "blur"
    assert cfg.foul_language_visual == "none"
    assert cfg.foul_language_audio == "mute_phrase"
    assert cfg.blur_strength == 1.0
    assert cfg.mute_padding == 0.5
    assert cfg.nudity_level == "med"
    assert cfg.gore_level == "med"
    assert cfg.violence_level == "med"
    assert cfg.foul_language_level == "med"


def test_loads_known_keys(tmp_path):
    p = tmp_path / "config.toml"
    p.write_text(
        "model_backend = \"standalone\"\n"
        "base_url = \"http://localhost:11434/v1\"\n"
        "sweep_interval = 2.0\n"
        "nudity_visual = \"black\"\n"
        "foul_language_audio = \"mute_word\"\n"
        "nudity_prompt = \"custom nudity def\"\n"
        "nudity_level = \"high\"\n"
        "gore_level = \"exhigh\"\n"
    )
    cfg = load_config(p)
    assert cfg.model_backend == "standalone"
    assert cfg.base_url == "http://localhost:11434/v1"
    assert cfg.sweep_interval == 2.0
    assert cfg.nudity_visual == "black"
    assert cfg.foul_language_audio == "mute_word"
    assert cfg.nudity_prompt == "custom nudity def"
    assert cfg.nudity_level == "high"
    assert cfg.gore_level == "exhigh"


def test_ignores_unknown_keys(tmp_path):
    """Unknown keys (e.g. newer configs) must not crash old versions."""
    p = tmp_path / "config.toml"
    p.write_text("model_backend = \"mlx-vlm\"\nfuture_setting = 42\n")
    cfg = load_config(p)
    assert cfg.model_backend == "mlx-vlm"


def test_invalid_enum_raises(tmp_path):
    p = tmp_path / "config.toml"
    p.write_text("nudity_visual = \"zoom\"\n")
    with pytest.raises(ValueError, match="nudity_visual"):
        load_config(p)


def test_invalid_audio_enum_raises(tmp_path):
    p = tmp_path / "config.toml"
    p.write_text("foul_language_audio = \"fade\"\n")
    with pytest.raises(ValueError, match="foul_language_audio"):
        load_config(p)


def test_invalid_level_raises(tmp_path):
    p = tmp_path / "config.toml"
    p.write_text("nudity_level = \"extreme\"\n")
    with pytest.raises(ValueError, match="nudity_level"):
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
