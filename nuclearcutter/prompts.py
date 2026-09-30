"""
Shared prompt templates — the single source of truth for what the scanner
asks the models.

The web GUI and the scanner both load this same file (`prompts.json` next to
this module), so the prompts you view/test in the GUI are exactly the prompts
the scan runs — there is no second copy to drift out of sync.

Templates use `{name}` placeholders. Formatting is "safe": unknown braces are
left as literal text, so prompt files may contain raw JSON braces (e.g. the
example response in the foul-language prompt) without escaping them.
"""

from __future__ import annotations

import json
from pathlib import Path

PROMPTS_FILE = Path(__file__).parent / "prompts.json"

# The keys every prompt file must carry. Missing keys fall back to the
# built-in templates baked into this module (kept in sync with prompts.json).
REQUIRED_KEYS = (
    "system_prompt",
    "sweep_prompt",
    "confirm_prompt",
    "foul_language_context_prompt",
    "summary_prompt",
    "muted_caption_prompt",
)

_DEFAULT_TEMPLATES = None  # lazy-loaded from prompts.json (its own source of truth)


def prompts_path() -> Path:
    return PROMPTS_FILE


def _read_file() -> dict:
    """Read prompts.json, returning {} if missing or unreadable."""
    try:
        return json.loads(PROMPTS_FILE.read_text())
    except (OSError, ValueError):
        return {}


def load_prompts() -> dict:
    """Return the merged prompt templates: file contents over built-in defaults.

    The built-in defaults are the templates compiled into this distribution
    (a snapshot of prompts.json), so a deleted/corrupt file still yields a
    working prompt set — and once the file is restored it wins again.
    """
    global _DEFAULT_TEMPLATES
    if _DEFAULT_TEMPLATES is None:
        file_data = _read_file()
        _DEFAULT_TEMPLATES = {k: file_data[k] for k in REQUIRED_KEYS if file_data.get(k)}
    out = dict(_DEFAULT_TEMPLATES)
    file_data = _read_file()
    for k in REQUIRED_KEYS:
        if file_data.get(k):
            out[k] = file_data[k]
    if not any(out.values()):
        print(f"warning: no prompt templates found in {PROMPTS_FILE} and no built-in "
              "defaults available — prompts will be empty", file=__import__("sys").stderr)
    return out


def save_prompts(templates: dict, path: Path | None = None) -> Path:
    """Write prompt templates back to prompts.json (merging over existing)."""
    path = Path(path) if path else PROMPTS_FILE
    data = _read_file()
    for k in REQUIRED_KEYS:
        if templates.get(k):
            data[k] = templates[k]
    path.write_text(json.dumps(data, indent=2) + "\n")
    return path


def get_prompt(name: str, **kwargs) -> str:
    """Load one template and substitute its {placeholders}.

    Substitution is a plain token replace of the exact `{key}` tokens, so
    templates may contain raw JSON braces (e.g. the example response in the
    foul-language prompt) without any escaping — only the tokens you pass are
    touched.
    """
    template = load_prompts().get(name, "")
    for key, val in kwargs.items():
        template = template.replace("{" + key + "}", str(val))
    return template
