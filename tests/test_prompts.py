"""Tests for the shared prompt file (nuclearcutter/prompts.py) — the single
source of truth the web GUI and the scanner both read from.

These matter more than usual because the templates live in JSON: nothing in
Python fails if a placeholder is renamed, the model simply receives a literal
`{kinds}` in the middle of an instruction and answers the wrong question.
"""

import json

import pytest

from nuclearcutter import prompts as P


def test_load_has_all_required_keys():
    t = P.load_prompts()
    for key in P.REQUIRED_KEYS:
        assert t.get(key), f"missing prompt template: {key}"


def test_required_keys_cover_both_passes_of_each_medium():
    """A reminder of what the set means, so nobody re-adds `confirm_prompt`.

    The v1 vocabulary asked the model "how offensive is this?" and the answer was
    a level. The v2 set asks "what is in the frame?" (sweep -> trigger, identify
    -> tags) and "what is said?" (word line, dialogue group), which is why the
    old `confirm_prompt` / `foul_language_context_prompt` keys are gone.
    """
    assert set(P.REQUIRED_KEYS) == {
        "system_prompt",            # shared preamble, keeps replies terse
        "sweep_prompt",             # cheap pass: WHERE to look (triggers)
        "identify_prompt",          # careful pass: WHAT it is (tag ids)
        "language_line_prompt",     # one wordlist hit, judged in context
        "dialogue_group_prompt",    # a run of dialogue: slurs/talk the list cannot see
        "summary_prompt",           # optional: nicer blur/black card text
        "muted_caption_prompt",     # optional: caption while audio is muted
    }


def test_summary_prompt_substitutes_placeholders():
    p = P.get_prompt("summary_prompt", action="blurred", context_before=3,
                     description_frames=4, context_after=3,
                     start="10.0", end="22.0", description="seed",
                     dialogue="Danger ahead.")
    for key in ("action", "context_before", "description_frames", "context_after",
                "start", "end", "description", "dialogue"):
        assert "{" + key + "}" not in p, f"unsubstituted placeholder: {key}"
    assert "blurred" in p and "10.0" in p and "22.0" in p
    assert "Danger ahead." in p and "seed" in p


def test_muted_caption_prompt_substitutes_placeholders():
    p = P.get_prompt("muted_caption_prompt", dialogue="You fool.",
                     kinds="Strong profanity")
    assert "{dialogue}" not in p and "{kinds}" not in p
    assert "You fool." in p and "Strong profanity" in p


def test_muted_caption_prompt_is_told_not_to_quote():
    """The caption sits on screen in front of whoever is watching. A model that
    echoes the slur becomes the thing the user paid us to remove."""
    p = P.load_prompts()["muted_caption_prompt"].lower()
    assert "without using any profanity" in p or "no profanity" in p
    assert "without repeating what was said" in p or "do not quote" in p


def _placeholders(template: str) -> set[str]:
    import re

    return set(re.findall(r"\{([a-z_0-9]+)\}", template))


@pytest.mark.parametrize("key,kwargs", [
    ("sweep_prompt", dict(n=4, sweep_triggers="- skin: bare skin")),
    ("identify_prompt", dict(n=6, dialogue="(none)", taxonomy="- nudity.x: y")),
    ("language_line_prompt", dict(categories="- lang.a: z", text="hi there",
                                  matches='["hi"]')),
    ("dialogue_group_prompt", dict(categories="- lang.a: z", lines="[0.0] hi")),
])
def test_every_template_fully_substitutes(key, kwargs):
    """A left-over `{token}` is the quiet failure mode of prompts-in-JSON: the
    model reads it as prose and invents an answer."""
    p = P.get_prompt(key, **kwargs)
    leftover = _placeholders(p)
    assert not leftover, f"{key} left {sorted(leftover)} unsubstituted"
    for value in kwargs.values():
        if isinstance(value, str) and value:
            assert value in p, f"{key} dropped its {value!r} argument"


def test_identify_prompt_offers_the_real_vocabulary():
    """The classify pass may only answer with ids from the taxonomy, so the built
    prompt has to contain them. A prompt that lost its vocabulary block makes
    every returned tag unknown, and every detection 'unclear'."""
    from nuclearcutter.detection.visual_scan import build_identify_prompt
    from nuclearcutter.taxonomy import TAGS, tags_in_family, Family

    built = build_identify_prompt(6, "(no dialogue)")
    # Every visual tag must be name-able by the model, not just a sample: a tag
    # the prompt never mentions is a tag the scan can never find.
    missing = [t.id for t in TAGS.values() if t.visual and t.id not in built]
    assert not missing, f"not offered to the model: {missing[:6]}"
    # Audio-only tags do not belong in a picture prompt.
    for t in tags_in_family(Family.LANGUAGE):
        if not t.visual:
            assert t.id not in built, f"{t.id} is spoken, not visible"
    assert "{taxonomy}" not in built and "{dialogue}" not in built


def test_sweep_prompt_asks_for_triggers_not_tags():
    """The sweep decides WHERE to look. If it starts emitting tag ids it has
    begun classifying from 4 thumbnails, and the identify pass loses its job."""
    p = P.get_prompt("sweep_prompt", n=4, sweep_triggers="- skin: bare skin")
    assert "trigger" in p.lower()
    assert "4 sampled frames" in p or "4 frame" in p


def test_get_prompt_preserves_raw_json_braces():
    """Templates contain example JSON with single braces; formatting must not
    mangle them (this is why get_prompt replaces tokens instead of .format())."""
    p = P.get_prompt("language_line_prompt", categories="C", text="hi",
                     matches='["hi"]')
    assert '"matches"' in p
    assert "{" in p and "}" in p, "the example JSON response must survive"
    assert "{text}" not in p and "{matches}" not in p and "{categories}" not in p


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
    assert data["identify_prompt"]


def test_missing_file_falls_back_to_defaults(monkeypatch, tmp_path):
    """A deleted prompts.json still yields a working prompt set."""
    monkeypatch.setattr(P, "PROMPTS_FILE", tmp_path / "nope.json")
    t = P.load_prompts()
    assert t["sweep_prompt"]


def test_prompts_file_declares_its_taxonomy_version():
    """A prompts.json from the v1 build describes levels, not tags. Loading it
    silently would give the model the old question with the new answer format."""
    data = json.loads(P.prompts_path().read_text())
    assert data.get("schema") == 2, "prompts.json must state which taxonomy it speaks"


def test_no_template_mentions_the_dead_vocabulary():
    """'low/med/high/exhigh' in a prompt means the model was asked to rate
    offensiveness again — the exact thing this release removed."""
    for key in P.REQUIRED_KEYS:
        text = P.load_prompts()[key].lower()
        for dead in ("exhigh", "severity level", "foul_language_level"):
            assert dead not in text, f"{key} still says {dead!r}"
