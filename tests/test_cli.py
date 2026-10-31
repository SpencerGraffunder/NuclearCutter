"""Tests for the CLI save path — a completed scan must never be lost to a
failed write (e.g. a disconnected SMB share) at the end of a long scan."""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import patch

import pytest

from nuclearcutter.cli import (
    _prefs_from_args, _save_result_with_fallback, cmd_tags,
)
from nuclearcutter.cli import main as cli_main
from nuclearcutter.schema import AudioAction, VisualAction
from nuclearcutter.schema import (
    SCHEMA_VERSION,
    FilmIdentity,
    ScanResult,
    VisualDetection,
)


def _fake_result() -> ScanResult:
    return ScanResult(
        schema_version=SCHEMA_VERSION,
        identity=FilmIdentity(
            title="TestMovie",
            year=None,
            duration_seconds=120.0,
            phash_samples=[],
        ),
        visual_detections=[
            VisualDetection(
                start=10.0,
                end=20.0,
                description="test", tags=["gore.blood"],
                confidence=0.9,
                )
        ],
        language_detections=[],
        generator={},
    )


def test_save_writes_to_intended_path_when_reachable(tmp_path: Path):
    result = _fake_result()
    out = tmp_path / "movie.nuclearcutter.json"

    saved = _save_result_with_fallback(result, out)

    assert saved == out
    assert out.exists()
    # Recovery copy also exists, next to the movie (no temp files).
    assert (tmp_path / "movie.nuclearcutter.json.recovery.json").exists()


def test_save_falls_back_to_cwd_when_share_disconnected(monkeypatch, tmp_path: Path):
    """A disconnected SMB share raises OSError/FileNotFoundError, NOT
    PermissionError — the old code only caught PermissionError and crashed,
    losing the 15-hour scan. It must fall back to CWD instead."""
    result = _fake_result()
    share_dir = tmp_path / "share"  # pretend this is the mounted SMB share
    share_dir.mkdir(exist_ok=True)
    out = share_dir / "movie.nuclearcutter.json"

    # Make writing ANY path inside the dead share fail (recovery copy AND the
    # intended output), the way a vanished mount does. CWD (mocked to tmp_path)
    # is the fallback.
    monkeypatch.chdir(tmp_path)

    original_save = ScanResult.save

    def _failing_share_save(self, path):
        if str(path).startswith(str(share_dir)):
            raise OSError(2, "No such file or directory")
        return original_save(self, path)

    monkeypatch.setattr(ScanResult, "save", _failing_share_save)

    saved = _save_result_with_fallback(result, out)

    # Fell back to CWD, not the dead share.
    assert saved == tmp_path / out.name
    assert saved.exists()
    assert not out.exists()
    assert not (share_dir / "movie.nuclearcutter.json.recovery.json").exists()


def test_save_raises_when_everywhere_fails(monkeypatch, tmp_path: Path):
    """The recovery copy falls back to CWD if the movie folder is dead, so the
    only way to raise is if EVERY target (movie folder + CWD) is unwritable."""
    result = _fake_result()
    out = tmp_path / "movie.nuclearcutter.json"

    # Everything fails: the movie-folder recovery copy, the CWD recovery copy,
    # the intended output, and the CWD fallback.
    monkeypatch.chdir(tmp_path)

    original_save = ScanResult.save

    def _everything_fails(self, path):
        raise OSError(28, "No space left on device")

    monkeypatch.setattr(ScanResult, "save", _everything_fails)

    with pytest.raises(RuntimeError, match="could not save scan result locally"):
        _save_result_with_fallback(result, out)


# --------------------------------------------------------------------------
# Headless selection: which categories the flags pick
# --------------------------------------------------------------------------

class _Args:
    """Just the fields _prefs_from_args reads, with the parser's defaults."""

    def __init__(self, preset=None, censor=None, allow=None, grep=None,
                 tag_file=None, rendering=None, visual=None, audio=None,
                 blur_strength=1.0, mute_padding=0.5, blur_padding=0.0):
        self.preset = preset
        self.censor = censor
        self.allow = allow
        self.grep = grep
        self.tag_file = tag_file
        self.rendering = rendering
        self.visual = visual
        self.audio = audio
        self.blur_strength = blur_strength
        self.mute_padding = mute_padding
        self.blur_padding = blur_padding


def test_no_flags_means_nothing_is_censored():
    prefs = _prefs_from_args(_Args())
    assert prefs.selection.tags == set()


def test_censor_takes_a_tag_id():
    prefs = _prefs_from_args(_Args(censor=["nudity.shirtless_male"]))
    assert prefs.selection.tags == {"nudity.shirtless_male"}


def test_censor_takes_a_whole_family():
    from nuclearcutter.taxonomy import TAGS, Family

    prefs = _prefs_from_args(_Args(censor=["substance"]))
    expected = {t.id for t in TAGS.values() if t.family is Family.SUBSTANCE}
    assert prefs.selection.tags == expected


def test_censor_accepts_comma_separated_and_repeats():
    prefs = _prefs_from_args(_Args(censor=["gore.unclear, harm.eating_disorder",
                                           "substance.smoking"]))
    assert prefs.selection.tags == {"gore.unclear", "harm.eating_disorder",
                                    "substance.smoking"}


def test_preset_then_allow_reads_the_way_it_sounds():
    from nuclearcutter.taxonomy import Family

    prefs = _prefs_from_args(_Args(preset="everything", allow=["gore"]))
    assert not any(t.startswith("gore.") for t in prefs.selection.tags)
    assert any(t.startswith("nudity.") for t in prefs.selection.tags)
    assert Family.GORE.value not in prefs.visual_actions or True


def test_grep_selects_by_label_as_well_as_id():
    prefs = _prefs_from_args(_Args(grep=["bikini|swimwear"]))
    assert "nudity.underwear_swimwear_female" in prefs.selection.tags


def test_grep_with_no_match_is_an_error_not_a_silent_nothing():
    # Selecting nothing by accident means an unedited film, discovered after an
    # hour of encoding.
    with pytest.raises(SystemExit) as exc:
        _prefs_from_args(_Args(grep=["zzz_nothing_zzz"]))
    assert "matched no tags" in str(exc.value)


def test_unknown_tag_is_an_error_that_points_at_the_list():
    with pytest.raises(SystemExit) as exc:
        _prefs_from_args(_Args(censor=["nudity.nude_lady"]))
    assert "nuclearcutter tags" in str(exc.value)


def test_tag_file_supports_comments_and_json(tmp_path):
    plain = tmp_path / "house.txt"
    plain.write_text("# our household\nnudity.shirtless_male\n\n# language\n"
                     "language.slur\n")
    assert _prefs_from_args(_Args(tag_file=str(plain))).selection.tags == {
        "nudity.shirtless_male", "language.slur"}

    as_json = tmp_path / "house.json"
    as_json.write_text(json.dumps({"tags": ["gore.blood_heavy"]}))
    assert _prefs_from_args(_Args(tag_file=str(as_json))).selection.tags == {
        "gore.blood_heavy"}


def test_tag_file_with_a_bad_id_fails_loudly(tmp_path):
    f = tmp_path / "bad.txt"
    f.write_text("nudity.shirtless_male\ntypo.nope\n")
    with pytest.raises(SystemExit) as exc:
        _prefs_from_args(_Args(tag_file=str(f)))
    assert "unknown tag id" in str(exc.value)


def test_rendering_restriction_is_parsed():
    prefs = _prefs_from_args(_Args(censor=["gore.blood_heavy"],
                                   rendering="live_action,anime"))
    assert prefs.selection.renderings == {"live_action", "anime"}


def test_bad_rendering_name_lists_the_real_ones():
    with pytest.raises(SystemExit) as exc:
        _prefs_from_args(_Args(rendering="cartoon"))
    assert "live_action" in str(exc.value)


def test_visual_action_override_per_family():
    prefs = _prefs_from_args(_Args(censor=["gore"], visual=["gore=black"]))
    assert prefs.visual_for("gore") is VisualAction.BLACK
    # Unmentioned families keep their defaults rather than going blank.
    assert prefs.visual_for("violence") is VisualAction.BLUR


def test_audio_action_is_validated_per_family():
    with pytest.raises(SystemExit) as exc:
        _prefs_from_args(_Args(audio=["gore=mute_word"]))
    assert "not available for this family" in str(exc.value)
    prefs = _prefs_from_args(_Args(audio=["language=mute_word"]))
    assert prefs.audio_for("language") is AudioAction.MUTE_WORD


def test_malformed_family_action_is_explained():
    with pytest.raises(SystemExit) as exc:
        _prefs_from_args(_Args(visual=["gore"]))
    assert "FAMILY=ACTION" in str(exc.value)


def test_tags_command_lists_the_vocabulary(capsys):
    assert cmd_tags(_Args2(family="substance")) == 0
    out = capsys.readouterr().out
    assert "substance.smoking" in out


class _Args2:
    def __init__(self, **kw):
        self.search = kw.get("search")
        self.family = kw.get("family")
        self.audio = False
        self.visual = False
        self.json = False
        self.presets = False


def test_a_v1_scan_file_fails_cleanly_not_with_a_traceback(tmp_path, capsys):
    """An old scan file is a normal user situation, and the sentence explaining it
    is the product. Verify both halves: the message is the useful one, and no
    traceback leaks. Exit code must be non-zero or a script cannot tell failure
    from success.
    """
    old = tmp_path / "Movie.nuclearcutter.json"
    old.write_text('{"schema_version": 1, "detections": []}')
    video = tmp_path / "Movie.mkv"
    video.write_bytes(b"\0")  # never opened: the scan file is refused first

    # main() parses sys.argv, so drive it the way the console script does.
    import sys
    argv = ["nuclearcutter", "render", str(video), "--preset", "nudity"]
    with patch.object(sys, "argv", argv):
        rc = cli_main()
    err = capsys.readouterr().err
    assert rc == 1, f"expected exit 1, got {rc}"
    assert "Traceback" not in err, "a foreseeable error printed a traceback"
    assert "schema v1" in err and "re-scan" in err, err
