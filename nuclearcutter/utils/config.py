"""
User configuration for NuclearCutter.

A single TOML file holds all the scan/render settings that used to be
command-line flags, so the user can set everything once and then run
`nuclearcutter scan MOVIE.mkv` / `nuclearcutter render MOVIE.mkv` with just
the movie path. The file lives at `<repo>/config.toml` (next to the project),
is plain text that any editor (TextEdit/Notepad) can open, and every key is
optional with a sane default. CLI flags still exist and override the config.

Example config.toml:

    # --- Scan / inference ---
    model_backend = "mlx-vlm"          # "mlx-vlm" (auto-start, default) | "standalone"
    model_path = "/Users/you/.lmstudio/models/lmstudio-community/Qwen3.5-9B-MLX-4bit"
    base_url = "http://localhost:1234/v1"
    vlm_model = "Qwen3.5-9B-MLX-4bit"
    text_model = "Qwen3.5-9B-MLX-4bit"
    whisper_model = "mlx-community/whisper-small-mlx"
    sweep_interval = 2.0
    vision_timeout = 8000
    timestamps_dir = ""

    # --- Render preferences ---
    nudity = "blur"
    nudity_blur_mute_audio = false
    intimate_scenes = "blur"
    intimate_scenes_blur_mute_audio = false
    gore_violence = "blur"
    gore_violence_blur_mute_audio = false
    foul_language = "mute"
    mute_scope = "word"
    font = ""
"""

from __future__ import annotations

import dataclasses
import os
import tomllib
from dataclasses import dataclass, field
from pathlib import Path

# Where the config file lives by default: next to the repo (Documents/NuclearCutter/config.toml).
DEFAULT_CONFIG_NAME = "config.toml"


def default_config_path() -> Path:
    """Return <repo>/config.toml (the directory containing this package's parent)."""
    # nuclearcutter/ is a package inside the repo; config sits one level up.
    here = Path(__file__).resolve().parent  # .../nuclearcutter/utils
    repo_root = here.parents[1]  # .../nuclearcutter
    return repo_root / DEFAULT_CONFIG_NAME


@dataclass
class AppConfig:
    # inference / scan
    model_backend: str = "mlx-vlm"
    model_path: str = "/Users/spencer/.lmstudio/models/lmstudio-community/Qwen3.5-9B-MLX-4bit"
    base_url: str = "http://localhost:1234/v1"
    vlm_model: str = "Qwen3.5-9B-MLX-4bit"
    text_model: str = "Qwen3.5-9B-MLX-4bit"
    whisper_model: str = "mlx-community/whisper-small-mlx"
    sweep_interval: float = 2.0
    vision_timeout: int = 8000
    timestamps_dir: str = ""

    # render preferences
    nudity: str = "blur"
    nudity_blur_mute_audio: bool = False
    intimate_scenes: str = "blur"
    intimate_scenes_blur_mute_audio: bool = False
    gore_violence: str = "blur"
    gore_violence_blur_mute_audio: bool = False
    blur_strength: float = 1.0
    foul_language: str = "mute"
    mute_scope: str = "word"
    mute_padding: float = 0.5
    font: str = ""

    def __post_init__(self):
        # Validate enums early with clear messages.
        for f in ("nudity", "intimate_scenes", "gore_violence"):
            if getattr(self, f) not in ("blur", "none"):
                raise ValueError(f"config: {f} must be 'blur' or 'none', got {getattr(self, f)!r}")
        if self.foul_language not in ("mute", "none"):
            raise ValueError(f"config: foul_language must be 'mute' or 'none', got {self.foul_language!r}")
        if self.mute_scope not in ("word", "utterance"):
            raise ValueError(f"config: mute_scope must be 'word' or 'utterance', got {self.mute_scope!r}")
        if self.model_backend not in ("mlx-vlm", "standalone"):
            raise ValueError(f"config: model_backend must be 'mlx-vlm' or 'standalone', got {self.model_backend!r}")


def load_config(path: str | Path | None = None) -> AppConfig:
    """Load config from `path` (default: <repo>/config.toml).

    Missing file → all defaults. Unknown keys → ignored (forward-compatible).
    """
    path = Path(path) if path else default_config_path()
    if not path.exists():
        return AppConfig()

    with open(path, "rb") as f:
        data = tomllib.load(f)

    # TOML is flat; filter to fields we know.
    known = {f.name for f in dataclasses.fields(AppConfig)}
    kwargs = {k: v for k, v in data.items() if k in known and v is not None}
    return AppConfig(**kwargs)
