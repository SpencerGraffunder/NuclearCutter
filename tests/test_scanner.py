"""The scan orchestrator: order, resume, and what happens when a pass fails.

The individual passes have their own tests; these cover the *sequence* — that
language runs off the transcript before the visual work, that a resume does not
re-spend money or skip un-swept footage, and that a dead model produces an error
instead of a scan file that looks clean.

Everything model-side and ffmpeg-side is faked; the only thing under test is the
scheduling and the files written to disk.
"""

from __future__ import annotations

import json
import threading
from pathlib import Path

import pytest

from nuclearcutter.detection.transcribe import Utterance, Word
from nuclearcutter.detection.visual_scan import SweepWindow
from nuclearcutter.schema import (
    SCHEMA_VERSION, FilmIdentity, LanguageDetection, ScanResult, VisualDetection,
)
from nuclearcutter.scan import scanner
from nuclearcutter.scan.scanner import ScanStopped, scan
from nuclearcutter.utils.llm_client import LLMConfig
from nuclearcutter.utils.scan_status import ScanStatus


class FakeClient:
    def __init__(self, visual=None, text=None):
        self.visual = visual or {"tags": [], "triggers": []}
        self.text = text or {}
        self.visual_calls = 0
        self.text_calls = 0

    def test_connection(self):
        return True

    def vision_query_json(self, prompt, frame_paths, max_pixels=None):
        self.visual_calls += 1
        if isinstance(self.visual, Exception):
            raise self.visual
        return dict(self.visual)

    def text_query_json(self, prompt):
        self.text_calls += 1
        if isinstance(self.text, Exception):
            raise self.text
        return dict(self.text)


@pytest.fixture
def video(tmp_path) -> Path:
    v = tmp_path / "Movie.mkv"
    v.write_text("not a real video")
    return v


@pytest.fixture
def no_media(monkeypatch):
    """Stub out fingerprint, whisper and both visual passes."""
    calls: dict = {"sweep": [], "identify": [], "phases": []}

    monkeypatch.setattr(scanner, "load_cached_fingerprint", lambda _p: (100.0, []))
    monkeypatch.setattr(scanner, "compute_fingerprint", lambda _p: (100.0, []))
    monkeypatch.setattr(scanner, "cache_fingerprint", lambda *a: None)

    transcript = [Utterance("oh damn it", 10.0, 11.0,
                            [Word("oh", 10.0, 10.4), Word("damn", 10.4, 10.7),
                             Word("it", 10.7, 11.0)])]
    monkeypatch.setattr(scanner, "read_transcript_cache", lambda *a, **k: list(transcript))
    monkeypatch.setattr(scanner, "transcribe", lambda *a, **k: list(transcript))
    monkeypatch.setattr(scanner, "transcribe_killable", lambda *a, **k: list(transcript))
    monkeypatch.setattr(scanner, "write_transcript_cache", lambda *a, **k: None)
    monkeypatch.setattr(scanner, "find_subtitle_file", lambda _p: None)
    monkeypatch.setattr(scanner, "detect_language",
                        lambda *a, **k: [LanguageDetection(
                            start=10.4, end=10.7, utterance_start=10.0,
                            utterance_end=11.0, text="damn",
                            tags=["language.profanity_mild"], llm_confirmed=True)])

    def fake_sweep(self, video_path, sample_interval=2.0, on_flagged_window=None,
                   on_progress=None, resume_from=0, existing_windows=None,
                   stop_event=None):
        calls["sweep"].append({"resume_from": resume_from,
                               "existing": list(existing_windows or []),
                               "interval": sample_interval})
        windows = list(existing_windows or [])
        if resume_from == 0:
            windows.append(SweepWindow(20.0, 22.0, ["skin"], "live_action",
                                       {"adult_female": 0, "adult_male": 1,
                                        "minors": "no"}, 0.7))
        if on_progress:
            on_progress(50, 50)
        from nuclearcutter.detection.visual_scan import CandidateRange, SweepStats

        stats = SweepStats()
        stats.batches_queried = 5
        stats.frames_total = 50
        stats.frames_done = 50
        ranges = [CandidateRange(20.0, 22.0, ["skin"], "live_action",
                                 {"adult_female": 0, "adult_male": 1, "minors": "no"},
                                 0.7)]
        calls["phases"].append("sweep")
        return ranges, stats

    def fake_identify(self, video_path, candidate, dialogue_text="",
                      invented_tags=None):
        calls["identify"].append({"start": candidate.start,
                                  "dialogue": dialogue_text,
                                  "triggers": list(candidate.triggers)})
        calls["phases"].append("identify")
        return (VisualDetection(start=candidate.start, end=candidate.end,
                                description="a shirtless man", confidence=0.8,
                                tags=["nudity.shirtless_male"],
                                triggers=list(candidate.triggers)), True)

    monkeypatch.setattr(scanner.VisualScanDetector, "sweep", fake_sweep, raising=True)
    monkeypatch.setattr(scanner.VisualScanDetector, "identify", fake_identify,
                        raising=True)
    return calls


def _run(video, tmp_path, client=None, llm_override=None, **kw):
    status = tmp_path / "Movie.nuclearcutter.status.json"
    partial = tmp_path / "Movie.nuclearcutter.json"
    result = scan(video, llm_config=llm_override or LLMConfig(), client_factory=lambda cfg: client or FakeClient(),
                  status_path=status, partial_result_path=partial, **kw)
    return result, status, partial


# --------------------------------------------------------------------------
# Order and outputs
# --------------------------------------------------------------------------

def test_phases_run_in_the_documented_order(video, tmp_path, no_media):
    calls = no_media
    _run(video, tmp_path)
    assert calls["phases"] == ["sweep", "identify"], \
        "the sweep decides WHERE to look; classify can only run after it"


def test_language_is_tagged_from_the_vocabulary(video, tmp_path, no_media):
    result, _s, _p = _run(video, tmp_path)
    assert result.language_detections[0].tags == ["language.profanity_mild"]


def test_result_carries_tags_not_levels(video, tmp_path, no_media):
    result, _s, _p = _run(video, tmp_path)
    det = result.visual_detections[0]
    assert det.tags == ["nudity.shirtless_male"]
    assert not hasattr(det, "level")
    assert result.schema_version == SCHEMA_VERSION


def test_status_file_records_every_phase(video, tmp_path, no_media):
    _run(video, tmp_path)
    # The final write says done; the intermediate ones are checked elsewhere.
    st = ScanStatus.load(tmp_path / "Movie.nuclearcutter.status.json")
    assert st.phase == "done"
    assert st.visual_detections[0]["tags"] == ["nudity.shirtless_male"]
    assert st.language_detections[0]["tags"] == ["language.profanity_mild"]


def test_partial_result_is_written_as_it_goes(video, tmp_path, no_media):
    _run(video, tmp_path)
    partial = tmp_path / "Movie.nuclearcutter.json"
    assert partial.exists()
    data = json.loads(partial.read_text())
    assert data["schema_version"] == 2
    assert data["visual_detections"][0]["tags"] == ["nudity.shirtless_male"]
    assert data["generator"]["taxonomy_version"] == 2


def test_dialogue_in_the_range_is_reaches_classify(video, tmp_path, no_media, monkeypatch):
    """The classify pass sees words: a line of dialogue is often the only clue
    that a blood-streaked face is a cartoon rather than a wound."""
    def utt(text, start, end):
        n = len(text.split())
        step = (end - start) / max(n, 1)
        return Utterance(text, start, end,
                         [Word(w, start + i * step, start + (i + 1) * step)
                          for i, w in enumerate(text.split())])

    monkeypatch.setattr(scanner, "read_transcript_cache", lambda *a, **k: [
        utt("look at all that blood", 20.5, 21.5),
        utt("nowhere near this scene", 90.0, 91.0),
    ])
    calls = no_media
    _run(video, tmp_path)
    assert calls["identify"][0]["dialogue"] == "look at all that blood", \
        "the line that overlaps the range must be passed to the classify pass"


def test_classify_is_skipped_for_ranges_already_classified(video, tmp_path, no_media,
                                                           monkeypatch):
    """A resumed scan must not re-spend a classify call on a range it already
    named — that is the money the resume exists to save."""
    status_path = tmp_path / "Movie.nuclearcutter.status.json"
    prior = ScanStatus(video="Movie.mkv", phase="identify", frames_done=50,
                       frames_total=50)
    prior.visual_candidates.append({"triggers": ["skin"], "start": 20.0, "end": 22.0,
                                    "confidence": 0.7})
    prior.visual_detections.append({"tags": ["nudity.shirtless_male"], "start": 20.0,
                                    "end": 22.0, "description": "kept",
                                    "confidence": 0.8, "rendering": "live_action",
                                    "minors": "no"})
    prior.language_detections.append({"tags": ["language.profanity_mild"], "text": "damn",
                                      "start": 1.0, "end": 1.2, "utterance_start": 0.5,
                                      "utterance_end": 2.0})
    prior.write(status_path)

    result, _s, _p = _run(video, tmp_path)
    assert no_media["identify"] == [], "the range was already classified"
    kept = [d for d in result.visual_detections if d.description == "kept"]
    assert kept, "the prior detection must survive the resume"
    assert result.language_detections[0].tags == ["language.profanity_mild"]


def test_resume_reconverts_frame_counts_when_the_interval_changed(video, tmp_path,
                                                                  no_media):
    """Progress is stored in SAMPLES. Resuming at a different sweep interval must
    preserve the swept POSITION in seconds, not the sample count — otherwise
    halving the interval resumes twice as deep and skips un-swept footage."""
    status_path = tmp_path / "Movie.nuclearcutter.status.json"
    st = ScanStatus(video="Movie.mkv", phase="sweep", frames_done=25, frames_total=50)
    st.sweep_interval = 4.0  # 100 s swept
    st.write(status_path)

    _run(video, tmp_path, sweep_interval=2.0)
    assert no_media["sweep"][0]["resume_from"] == 50  # 100 s / 2 s, not 25


def test_resume_refuses_a_status_file_from_another_film(tmp_path, no_media, monkeypatch):
    """Two films can share a stem. Resuming across them would put one film's
    detections on the other's timeline."""
    other = tmp_path / "Other.mkv"
    other.write_text("x")
    status_path = tmp_path / "Other.nuclearcutter.status.json"
    st = ScanStatus(video="Different.mkv", phase="sweep", frames_done=25, frames_total=50)
    st.write(status_path)

    scan(other, llm_config=LLMConfig(),
                  client_factory=lambda cfg: FakeClient(),
                  status_path=status_path,
                  partial_result_path=tmp_path / "Other.nuclearcutter.json")
    assert no_media["sweep"][0]["resume_from"] == 0


def test_a_v1_status_file_is_not_resumable(tmp_path, no_media):
    """Old status files hold category/level records with no tags. Mixing the two
    vocabularies in one scan file would be unrecoverable."""
    status_path = tmp_path / "Movie.nuclearcutter.status.json"
    status_path.write_text(json.dumps({
        "schema": 1, "video": "Movie.mkv", "phase": "visual_confirm",
        "frames_done": 50, "frames_total": 50,
        "visual_candidates": [{"category": "NUDITY", "start": 3.0, "end": 5.0,
                               "confidence": 0.9, "level": "high"}],
        "visual_detections": [], "language_detections": [],
    }))
    scan(tmp_path / "Movie.mkv", llm_config=LLMConfig(),
                  client_factory=lambda cfg: FakeClient(), status_path=status_path,
                  partial_result_path=tmp_path / "Movie.nuclearcutter.json")
    assert no_media["sweep"][0]["resume_from"] == 0
    after = ScanStatus.load(status_path)
    assert all("category" not in c and "triggers" in c for c in after.visual_candidates)


# --------------------------------------------------------------------------
# Failure modes: a scan must never look clean when it could not look
# --------------------------------------------------------------------------

def test_every_sweep_batch_failing_raises_instead_of_a_clean_scan(video, tmp_path,
                                                                  monkeypatch):
    """With the sweep unable to read anything, the ranges are all 'doubt' and the
    classify pass would churn through the whole film. Fail now, say why."""
    monkeypatch.setattr(scanner, "load_cached_fingerprint", lambda _p: (100.0, []))
    monkeypatch.setattr(scanner, "compute_fingerprint", lambda _p: (100.0, []))
    monkeypatch.setattr(scanner, "cache_fingerprint", lambda *a: None)
    monkeypatch.setattr(scanner, "read_transcript_cache", lambda *a, **k: [])
    monkeypatch.setattr(scanner, "find_subtitle_file", lambda _p: None)
    monkeypatch.setattr(scanner, "detect_language", lambda *a, **k: [])

    def all_fail(self, video_path, **kw):
        from nuclearcutter.detection.visual_scan import SweepStats

        stats = SweepStats()
        stats.batches_queried = 7
        stats.batches_failed = 7
        return [], stats

    monkeypatch.setattr(scanner.VisualScanDetector, "sweep", all_fail)
    client = FakeClient()
    with pytest.raises(RuntimeError) as exc:
        scan(video, llm_config=LLMConfig(), client_factory=lambda cfg: client,
             status_path=tmp_path / "s.json",
             partial_result_path=tmp_path / "p.json")
    assert "not answering" in str(exc.value).lower() or "failed" in str(exc.value).lower()


def test_every_classify_failing_raises(video, tmp_path, no_media, monkeypatch):
    def always_broken(self, video_path, candidate, dialogue_text="",
                      invented_tags=None):
        from nuclearcutter.detection.visual_scan import _unclassified

        return _unclassified(candidate, reason="classify query failed"), False

    monkeypatch.setattr(scanner.VisualScanDetector, "identify", always_broken)
    with pytest.raises(RuntimeError) as exc:
        _run(video, tmp_path)
    assert "classify" in str(exc.value).lower()


def test_partial_progress_survives_a_stop(video, tmp_path, no_media):
    ev = threading.Event()
    ev.set()
    with pytest.raises(ScanStopped):
        _run(video, tmp_path, stop_event=ev)
    status = tmp_path / "Movie.nuclearcutter.status.json"
    assert status.exists(), "a stopped scan must leave a resumable status file"


def test_stop_midway_keeps_found_detections(video, tmp_path, no_media, monkeypatch):
    """Stopping during classify must still write the detections found so far —
    otherwise Stop costs you the whole run."""
    ev = threading.Event()

    def sweeping(self, video_path, sample_interval=2.0, on_flagged_window=None,
                 on_progress=None, resume_from=0, existing_windows=None,
                 stop_event=None):
        from nuclearcutter.detection.visual_scan import CandidateRange, SweepStats

        stats = SweepStats()
        stats.batches_queried = 3
        # Two candidates, and Stop is pressed before classify starts.
        ev.set()
        ranges = [CandidateRange(20.0, 22.0, ["skin"], "live_action",
                                 {"adult_female": 0, "adult_male": 1, "minors": "no"},
                                 0.7),
                  CandidateRange(40.0, 42.0, ["blood"], "live_action",
                                 {"adult_female": 0, "adult_male": 1, "minors": "no"},
                                 0.7)]
        stats.frames_total = stats.frames_done = 50
        return ranges, stats

    monkeypatch.setattr(scanner.VisualScanDetector, "sweep", sweeping)
    with pytest.raises(ScanStopped):
        _run(video, tmp_path, stop_event=ev)
    partial = json.loads((tmp_path / "Movie.nuclearcutter.json").read_text())
    assert partial["language_detections"], "language found before the stop must persist"


def test_status_write_failure_does_not_kill_a_scan(video, tmp_path, no_media,
                                                   monkeypatch):
    """The status file lives next to the movie, often on a network share. If the
    share drops mid-scan, losing the status file is bad; losing the scan is worse."""
    monkeypatch.setattr(ScanStatus, "write",
                        lambda self, path: (_ for _ in ()).throw(OSError("share gone")))
    result, _s, _p = _run(video, tmp_path)
    assert result.visual_detections[0].tags == ["nudity.shirtless_male"]


def test_partial_result_write_failure_is_survivable(video, tmp_path, no_media,
                                                    monkeypatch):
    """Same reasoning for the partial scan file, which is written after every
    detection and is the first thing to hit a dead share."""
    real_write = scanner._atomic_write_json

    def exploding(path, data):
        raise OSError("smb disconnected")

    monkeypatch.setattr(scanner, "_atomic_write_json", exploding)
    result, _s, _p = _run(video, tmp_path)  # must not raise
    assert result.visual_detections


# --------------------------------------------------------------------------
# Provenance in the file
# --------------------------------------------------------------------------

def test_generator_records_the_model_and_taxonomy(video, tmp_path, no_media):
    result, _s, _p = _run(video, tmp_path, client=FakeClient(),
                          llm_override=LLMConfig(vlm_model="gemma-test",
                                                 text_model="gemma-test"))
    assert result.generator["taxonomy_version"] == 2
    assert result.generator.get("vlm_model")


def test_resume_state_is_loaded_from_the_matching_video_only(video, tmp_path, no_media):
    """The happy path of the guard above: same name, same status file -> resume."""
    st = ScanStatus(video="Movie.mkv", phase="sweep", frames_done=10, frames_total=50)
    st.sweep_interval = 2.0
    path = tmp_path / "Movie.nuclearcutter.status.json"
    st.write(path)
    scan(video, llm_config=LLMConfig(), client_factory=lambda c: FakeClient(),
         status_path=path, partial_result_path=tmp_path / "Movie.nuclearcutter.json",
         sweep_interval=2.0)
    assert no_media["sweep"][0]["resume_from"] == 10
