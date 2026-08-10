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

    # --- Render preferences — per-category visual + audio corrections ---
    nudity_visual = "blur"
    nudity_audio = "none"
    gore_visual = "blur"
    gore_audio = "none"
    violence_visual = "blur"
    violence_audio = "none"
    foul_language_visual = "none"
    foul_language_audio = "mute_phrase"
    blur_strength = 1.0
    mute_padding = 0.5
    font = ""

    # --- Per-category severity threshold ("low"/"med"/"high"/"exhigh") ---
    nudity_level = "med"
    gore_level = "med"
    violence_level = "med"
    foul_language_level = "med"

    # --- Optional custom prompts (empty = built-in fixed level scale) ---
    nudity_prompt = ""
    gore_prompt = ""
    violence_prompt = ""
    foul_language_prompt = ""
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

    # render preferences — per-category visual + audio corrections
    nudity_visual: str = "blur"       # "none" | "blur" | "black"
    nudity_audio: str = "none"        # "none" | "mute_word" | "mute_phrase" | "replace_word" | "replace_phrase"
    gore_visual: str = "blur"
    gore_audio: str = "none"
    violence_visual: str = "blur"
    violence_audio: str = "none"
    foul_language_visual: str = "none"
    foul_language_audio: str = "mute_phrase"
    blur_strength: float = 1.0
    mute_padding: float = 0.5
    font: str = ""

    # Per-category severity THRESHOLD — correct content at/above this level.
    # The level scale itself is FIXED (built into the scan, so scans are
    # shareable); this is just each user's personal cutoff.
    # "low" | "med" | "high" | "exhigh"
    nudity_level: str = "med"
    gore_level: str = "med"
    violence_level: str = "med"
    foul_language_level: str = "med"

    # OPTIONAL custom prompts — replace the built-in fixed level scale for a
    # category entirely. Empty (default) = use the built-in standardized scale.
    # Only set these if you really want fully custom definitions.
    nudity_prompt: str = ""
    gore_prompt: str = ""
    violence_prompt: str = ""
    foul_language_prompt: str = ""

    def __post_init__(self):
        # Validate enums early with clear messages.
        for f in ("nudity_visual", "gore_visual", "violence_visual", "foul_language_visual"):
            if getattr(self, f) not in ("none", "blur", "black"):
                raise ValueError(f"config: {f} must be 'none', 'blur' or 'black', got {getattr(self, f)!r}")
        # Visual categories can only mute the whole scene (no per-word/per-phrase
        # sound recognition in a visual scene). Foul language has word-level
        # timestamps, so it keeps the full word/phrase/replace set.
        for f in ("nudity_audio", "gore_audio", "violence_audio"):
            if getattr(self, f) not in ("none", "mute_scene"):
                raise ValueError(
                    f"config: {f} must be 'none' or 'mute_scene' (visual scenes "
                    f"can't do word/phrase-level audio), got {getattr(self, f)!r}"
                )
        if self.foul_language_audio not in ("none", "mute_word", "mute_phrase",
                                            "replace_word", "replace_phrase"):
            raise ValueError(
                f"config: foul_language_audio must be 'none'/'mute_word'/'mute_phrase'/"
                f"'replace_word'/'replace_phrase', got {self.foul_language_audio!r}"
            )
        for f in ("nudity_level", "gore_level", "violence_level", "foul_language_level"):
            if getattr(self, f) not in ("low", "med", "high", "exhigh"):
                raise ValueError(
                    f"config: {f} must be 'low'/'med'/'high'/'exhigh', got {getattr(self, f)!r}"
                )
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
