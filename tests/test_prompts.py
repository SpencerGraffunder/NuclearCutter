"""Tests for the shared prompt file (nuclearcutter/prompts.py) — the single
source of truth the web GUI and the scanner both read from."""

import json

import pytest

from nuclearcutter import prompts as P


def test_load_has_all_required_keys():
    t = P.load_prompts()
    for key in ("system_prompt", "sweep_prompt", "confirm_prompt",
                "foul_language_context_prompt"):
        assert t.get(key), f"missing prompt template: {key}"


def test_get_prompt_substitutes_placeholders():
    p = P.get_prompt("sweep_prompt", n=4, definitions="- nudity: x")
    assert "{n}" not in p
    assert "{definitions}" not in p
    assert "4 sampled frames" in p
    assert "- nudity: x" in p


def test_get_prompt_preserves_raw_json_braces():
    """The foul-language template contains example JSON with single braces;
    formatting must not mangle them."""
    p = P.get_prompt("foul_language_context_prompt", definition="d", text="hi", matches="[]")
    assert '"confirmed_words"' in p
    assert "{definition}" not in p and "{text}" not in p and "{matches}" not in p


def test_unknown_template_returns_empty():
    assert P.get_prompt("does_not_exist", n=1) == ""


def test_save_prompts_roundtrip(tmp_path):
    target = tmp_path / "prompts.json"
    t = P.load_prompts()
    t["sweep_prompt"] = "custom sweep with {n}"
    P.save_prompts(t, path=target)

    # Saving to a custom path must NOT clobber the real prompts file.
    real = P.load_prompts()
    assert real["sweep_prompt"] != "custom sweep with {n}"

    data = json.loads(target.read_text())
    assert data["sweep_prompt"] == "custom sweep with {n}"
    # Other keys preserved in the written file.
    assert data["confirm_prompt"]


def test_missing_file_falls_back_to_defaults(monkeypatch, tmp_path):
    """A deleted prompts.json still yields a working prompt set."""
    monkeypatch.setattr(P, "PROMPTS_FILE", tmp_path / "nope.json")
    t = P.load_prompts()
    assert t["sweep_prompt"]
