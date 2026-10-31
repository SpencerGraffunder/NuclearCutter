"""The two visual passes: the trigger sweep and the tag classify call.

Nothing here talks to a model — a fake client answers with canned JSON — but the
video is real (generated with ffmpeg) when ffmpeg is present, because the sweep's
job is mapping *time* to *what the model saw*, and that is where the bugs live.

The invariants under test are the ones the redesign depends on:

  * an unreadable batch becomes a `doubt` window, never a clean sample
  * merging ranges UNIONs triggers (v1 kept the first window's category and
    silently dropped every other family in the same merged span)
  * the classify pass can never return a detection with no tags
  * a model that invents tag ids loses the id, not the finding
"""

from __future__ import annotations

import shutil
import subprocess
from pathlib import Path

import pytest

from nuclearcutter.detection.visual_scan import (
    CandidateRange, SWEEP_FRAMES_PER_CALL, SweepWindow, VisualScanDetector,
    _clean_triggers,
    _flagged_timestamps, _genders_hint, _identify_frame_count, _merge_subjects,
    _minors_of, _rendering_of, _subjects_of, build_identify_prompt,
    build_sweep_prompt, merge_windows,
)
from nuclearcutter.taxonomy import Family, Minors, Rendering, TAGS, tags_in_family

HAS_FFMPEG = shutil.which("ffmpeg") is not None and shutil.which("ffprobe") is not None


class FakeClient:
    """Answers vision queries from a queue, recording what it was asked."""

    def __init__(self, responses):
        # `responses` is either a queue (popped per call, an Exception instance
        # raising) or a callable of (prompt, frame_paths) -> dict, for sweeps
        # whose batch count depends on the video.
        self.responses = responses if callable(responses) else list(responses)
        self.prompts: list[str] = []
        self.frames: list[list] = []
        self.calls = 0

    def vision_query_json(self, prompt, frame_paths, max_pixels=None):
        self.calls += 1
        self.prompts.append(prompt)
        self.frames.append(list(frame_paths))
        if callable(self.responses):
            reply = self.responses(prompt, frame_paths)
        else:
            if not self.responses:
                raise AssertionError("FakeClient ran out of responses")
            reply = self.responses.pop(0)
        if isinstance(reply, Exception):
            raise reply
        return reply


def _sweep_reply(triggers, frames=None, **kw) -> dict:
    out = {"triggers": triggers, "frames": frames if frames is not None else [],
           "subjects": {"adult_female": 0, "adult_male": 1, "minors": "no"},
           "rendering": "live_action", "confidence": 0.8}
    out.update(kw)
    return out


def _identify_reply(tags, **kw) -> dict:
    out = {"tags": tags, "description": "something on screen", "confidence": 0.7,
           "rendering": "live_action", "minors": "no", "uncertain_tags": []}
    out.update(kw)
    return out


@pytest.fixture
def tiny_video(tmp_path):
    """A 4-second test pattern. Real file, real ffmpeg, real timestamps."""
    if not HAS_FFMPEG:
        pytest.skip("ffmpeg not installed")
    out = tmp_path / "tiny.mp4"
    subprocess.run(
        ["ffmpeg", "-y", "-f", "lavfi", "-i", "testsrc=duration=4:size=128x72:rate=10",
         "-pix_fmt", "yuv420p", str(out)],
        check=True, capture_output=True)
    return out


# --------------------------------------------------------------------------
# Trigger-list normalisation
# --------------------------------------------------------------------------

def test_clean_triggers_keeps_known_names():
    assert _clean_triggers(["skin", "restraint"]) == ["skin", "restraint"]


def test_clean_triggers_maps_unknown_to_doubt():
    # An invented trigger must not read as "nothing here".
    assert _clean_triggers(["shiny_things"]) == ["doubt"]


def test_clean_triggers_accepts_a_bare_string():
    assert _clean_triggers("blood") == ["blood"]


def test_clean_triggers_none_means_nothing_flagged():
    assert _clean_triggers(None) == []
    assert _clean_triggers([]) == []


def test_clean_triggers_dedupes_and_cleans_case():
    assert _clean_triggers(["Skin", "skin", " SKIN "]) == ["skin"]


def test_every_trigger_name_is_in_the_prompt():
    from nuclearcutter.taxonomy import SWEEP_TRIGGERS

    block = build_sweep_prompt(4)
    for name, _desc in SWEEP_TRIGGERS:
        assert name in block, name


# --------------------------------------------------------------------------
# Frame localisation
# --------------------------------------------------------------------------

def test_flagged_timestamps_uses_zero_based_indices():
    ts = [0.0, 2.0, 4.0, 6.0]
    assert _flagged_timestamps(ts, [0, 2]) == [0.0, 4.0]


def test_flagged_timestamps_drops_out_of_range_and_junk():
    ts = [0.0, 2.0]
    # An index past the end is dropped rather than clamped — the caller falls
    # back to covering the whole batch, which is more coverage, never less.
    assert _flagged_timestamps(ts, [5, -1, "two", True, 1.5]) == []


def test_flagged_timestamps_empty_means_whole_batch_not_nothing():
    assert _flagged_timestamps([0.0, 2.0], None) == []
    assert _flagged_timestamps([0.0, 2.0], "nonsense") == []


def test_flagged_timestamps_dedupes_repeats():
    assert _flagged_timestamps([0.0, 2.0, 4.0], [1, 1, 1]) == [2.0]


# --------------------------------------------------------------------------
# Subjects / rendering qualifiers
# --------------------------------------------------------------------------

def test_subjects_of_garbage_is_all_zero_and_no_minors():
    assert _subjects_of(None) == {"adult_female": 0, "adult_male": 0,
                                  "minors": "no"}
    assert _subjects_of({"adult_male": "lots", "minors": "toddler"}) == {
        "adult_female": 0, "adult_male": 0, "minors": "no"}


def test_merge_subjects_takes_the_worst_answer_on_children():
    a = {"adult_female": 2, "adult_male": 0, "minors": "uncertain"}
    b = {"adult_female": 0, "adult_male": 1, "minors": "yes"}
    assert _merge_subjects(a, b)["minors"] == "yes"
    assert _merge_subjects(b, a)["minors"] == "yes"
    assert _merge_subjects(a, {"adult_female": 0, "adult_male": 0,
                               "minors": "no"})["minors"] == "uncertain"


def test_minors_of_prefers_the_classify_answer():
    assert _minors_of("yes", {"minors": "no"}) == "yes"
    # No answer from classify -> keep the sweep's, which saw more footage.
    assert _minors_of(None, {"minors": "uncertain"}) == "uncertain"
    assert _minors_of("maybe", {"minors": "yes"}) == "yes"


def test_rendering_of_unknown_falls_back_to_live_action():
    assert _rendering_of("anime") == "anime"
    assert _rendering_of("puppet_show") == Rendering.LIVE_ACTION.value
    assert _rendering_of(None) == Rendering.LIVE_ACTION.value


def test_genders_hint_suppresses_the_wrong_gender_from_the_prompt():
    assert _genders_hint({"adult_male": 2, "adult_female": 0}) == ("male",)
    assert _genders_hint({"adult_male": 1, "adult_female": 1}) is None
    assert _genders_hint(None) is None


# --------------------------------------------------------------------------
# Merging: the bug that motivated multi-tag
# --------------------------------------------------------------------------

def _win(start, end, triggers, **kw):
    return SweepWindow(start, end, list(triggers),
                       kw.get("rendering", "live_action"),
                       kw.get("subjects", {"adult_female": 0, "adult_male": 1,
                                           "minors": "no"}),
                       kw.get("confidence", 0.5))


def test_merge_unions_triggers_instead_of_keeping_the_first():
    # A shot that goes blood then undress must keep BOTH, so the viewer who
    # filters nudity and not blood still gets this range.
    wins = [_win(10.0, 11.0, ["blood"]), _win(11.5, 12.0, ["skin"])]
    out = merge_windows(wins, duration=60.0, padding=0.0, merge_gap=2.0)
    assert len(out) == 1
    assert set(out[0].triggers) == {"blood", "skin"}


def test_merge_keeps_distant_windows_apart():
    wins = [_win(10.0, 11.0, ["blood"]), _win(40.0, 41.0, ["skin"])]
    out = merge_windows(wins, duration=60.0, padding=0.0, merge_gap=2.0)
    assert len(out) == 2


def test_merge_pads_but_never_past_duration():
    wins = [_win(0.0, 1.0, ["skin"])]
    out = merge_windows(wins, duration=10.0, padding=2.0, merge_gap=1.0)
    assert out[0].start == 0.0
    assert out[0].end == pytest.approx(3.0)

    tail = merge_windows([_win(9.5, 10.0, ["skin"])], duration=10.0, padding=2.0,
                         merge_gap=1.0)
    assert tail[0].end == 10.0


def test_merge_takes_peak_confidence_and_worst_minors():
    wins = [_win(0.0, 1.0, ["skin"], confidence=0.3),
            _win(1.2, 2.0, ["blood"], confidence=0.9,
                 subjects={"adult_female": 0, "adult_male": 1, "minors": "yes"})]
    out = merge_windows(wins, duration=30.0, padding=0.0, merge_gap=2.0)
    assert out[0].confidence == 0.9
    assert out[0].subjects["minors"] == "yes"


def test_merge_empty_input_is_empty_output():
    assert merge_windows([], duration=10.0, padding=1.0, merge_gap=1.0) == []


def test_padding_shrinks_with_sweep_interval():
    # Padding must reach toward the neighbouring unflagged samples without
    # swallowing the next verified-clean one: a 5s interval needs more padding
    # than a 1s interval, but never a fixed 2 seconds at every setting.
    from nuclearcutter.detection.visual_scan import (
        _frame_padding_for_interval, _padding_for_interval,
    )

    tight = _padding_for_interval(1.0)
    loose = _padding_for_interval(5.0)
    assert 0 < tight < loose
    assert _frame_padding_for_interval(5.0) <= _padding_for_interval(5.0)


# --------------------------------------------------------------------------
# Prompts
# --------------------------------------------------------------------------

def test_sweep_prompt_mentions_frame_count_and_no_censoring_language():
    block = build_sweep_prompt(7)
    assert "7" in block
    for banned in ("exhigh", "severity", "censor level"):
        assert banned not in block


def test_identify_prompt_lists_real_tag_ids():
    block = build_identify_prompt(6, "some dialogue")
    for tag in tags_in_family(Family.NUDITY)[:6]:
        assert tag.id in block
    assert "unclear" in block


def test_identify_prompt_can_narrow_to_one_gender():
    block = build_identify_prompt(4, "", genders=("male",))
    female = [t.id for t in tags_in_family(Family.NUDITY) if t.gender == "female"]
    assert female and not any(f in block for f in female)


def test_identify_prompt_includes_dialogue_when_given():
    assert "the dialogue is" in build_identify_prompt(4, "I will kill you").lower() or \
        "kill you" in build_identify_prompt(4, "kill you")


# --------------------------------------------------------------------------
# The sweep, against a real file
# --------------------------------------------------------------------------

def test_sweep_returns_a_candidate_where_the_model_flagged(tiny_video):
    # Only the very first batch (0-1.5 s) says "skin", and points at frame 1 of
    # it — so the candidate must land near the START of the film, not anywhere.
    state = {"n": 0}

    def reply(prompt, frames):
        state["n"] += 1
        if state["n"] == 1:
            return _sweep_reply(["skin"], frames=[1])
        return _sweep_reply([])

    client = FakeClient(reply)
    det = VisualScanDetector(client, sweep_resolution="240p")
    ranges, stats = det.sweep(tiny_video, sample_interval=0.5)
    assert stats.batches_failed == 0
    assert ranges, "a flagged batch must produce a candidate range"
    assert any("skin" in r.triggers for r in ranges)
    assert ranges[0].start < 1.5 and ranges[0].end > 0.4


def test_sweep_reports_progress_in_absolute_counts(tiny_video):
    client = FakeClient(lambda p, f: _sweep_reply([]))
    det = VisualScanDetector(client)
    seen: list[tuple[int, int]] = []
    ranges, stats = det.sweep(tiny_video, sample_interval=1.0,
                              on_progress=lambda d, t: seen.append((d, t)))
    assert seen, "progress must fire"
    assert seen[-1][0] == seen[-1][1]  # ends at 100%
    assert all(d <= t for d, t in seen)


def test_resume_skips_already_swept_samples(tiny_video):
    # Interval chosen from the batch size so the 4 s video always spans >= 4
    # full batches; a half-resume must then skip roughly half the calls.
    interval = 3.5 / (SWEEP_FRAMES_PER_CALL * 4 + 1)
    client = FakeClient(lambda p, f: _sweep_reply([]))
    det = VisualScanDetector(client)
    _, full = det.sweep(tiny_video, sample_interval=interval)
    assert full.frames_total > 0
    assert client.calls >= 4  # sanity: the test needs several batches to mean anything

    client2 = FakeClient(lambda p, f: _sweep_reply([]))
    det2 = VisualScanDetector(client2)
    _, resumed = det2.sweep(tiny_video, sample_interval=interval,
                            resume_from=full.frames_total // 2)
    # Progress is reported in ABSOLUTE counts so the GUI bar does not jump back.
    assert resumed.frames_done == full.frames_total
    # ...and roughly half the model calls were skipped: the skipped samples are
    # the dominant cost of a scan, which is why resume exists.
    assert client2.calls < client.calls


def test_existing_windows_survive_a_resume(tiny_video):
    # The prior run found something at 1s; this run starts past it. Merging must
    # still see the old window or the scan file loses a segment.
    prior = [_win(1.0, 1.5, ["skin"])]
    client = FakeClient(lambda p, f: _sweep_reply([]))
    det = VisualScanDetector(client)
    ranges, _ = det.sweep(tiny_video, sample_interval=0.5, resume_from=6,
                          existing_windows=prior)
    assert any(r.start <= 1.5 and "skin" in r.triggers for r in ranges)


def test_failed_batch_becomes_doubt_not_clean(tiny_video):
    # THE safety property of the whole pipeline: "could not look" is never
    # recorded as "nothing there".
    state = {"n": 0}

    def reply(prompt, frames):
        state["n"] += 1
        # Fail the whole retry budget of the FIRST batch (MAX_VLM_RETRIES), then
        # recover — that is what "a stretch of the film went unread" looks like.
        if state["n"] <= 3:
            return RuntimeError("server down")  # raised by FakeClient
        return _sweep_reply([])

    client = FakeClient(reply)
    det = VisualScanDetector(client)
    ranges, stats = det.sweep(tiny_video, sample_interval=0.5)
    assert stats.batches_failed == 1
    assert any("doubt" in r.triggers for r in ranges), \
        "a batch that failed every retry must still be classified"


def test_stop_event_leaves_progress_short_of_complete(tiny_video):
    import threading

    ev = threading.Event()
    ev.set()
    client = FakeClient(lambda p, f: _sweep_reply([]))
    det = VisualScanDetector(client)
    seen: list[tuple[int, int]] = []
    ranges, stats = det.sweep(tiny_video, sample_interval=0.5, stop_event=ev,
                              on_progress=lambda d, t: seen.append((d, t)))
    assert client.calls == 0
    assert not seen or seen[-1][0] < seen[-1][1]


# --------------------------------------------------------------------------
# The classify pass
# --------------------------------------------------------------------------

@pytest.fixture
def no_ffmpeg_needed(monkeypatch, tmp_path):
    """Give identify fake frame paths so it never shells out to ffmpeg."""
    paths = [tmp_path / f"f{i}.jpg" for i in range(12)]
    for p in paths:
        p.write_bytes(b"\xff\xd8\xff")
    monkeypatch.setattr("nuclearcutter.detection.visual_scan.extract_frames_in_range",
                        lambda *a, **k: list(paths))
    return paths


def _candidate(**kw):
    return CandidateRange(start=kw.get("start", 10.0), end=kw.get("end", 14.0),
                          triggers=kw.get("triggers", ["skin"]),
                          rendering=kw.get("rendering", "live_action"),
                          subjects=kw.get("subjects", {"adult_female": 0,
                                                       "adult_male": 1,
                                                       "minors": "no"}),
                          confidence=kw.get("confidence", 0.6))


def test_identify_returns_tags_and_description(tiny_video, no_ffmpeg_needed):
    client = FakeClient([_identify_reply(["nudity.shirtless_male"])])
    det = VisualScanDetector(client)
    out, ok = det.identify(tiny_video, _candidate())
    assert ok is True
    assert out.tags == ["nudity.shirtless_male"]
    assert out.description == "something on screen"
    assert out.start == 10.0 and out.end == 14.0


def test_identify_never_returns_an_empty_tag_list(tiny_video, no_ffmpeg_needed):
    # The model says "nothing here". The sweep said otherwise, and this tool's
    # failure mode is showing something, not hiding something.
    client = FakeClient([_identify_reply([])])
    det = VisualScanDetector(client)
    out, ok = det.identify(tiny_video, _candidate())
    assert ok is True  # the look succeeded; the answer was just empty
    assert out.tags == ["safety.unclassified"]


def test_identify_folds_invented_ids_into_the_family_residual(tiny_video,
                                                              no_ffmpeg_needed):
    invented: list[str] = []
    client = FakeClient([_identify_reply(["nudity.topless_lady_half"])])
    det = VisualScanDetector(client)
    out, _ok = det.identify(tiny_video, _candidate(), invented_tags=invented)
    assert out.tags == ["nudity.unclear"]
    assert invented == ["nudity.topless_lady_half"]


def test_identify_query_failure_is_reported_as_not_classified(tiny_video,
                                                              no_ffmpeg_needed):
    client = FakeClient([RuntimeError("boom")] * 10)
    det = VisualScanDetector(client)
    out, ok = det.identify(tiny_video, _candidate())
    assert ok is False
    assert out.tags == ["safety.unclassified"]


def test_identify_keeps_real_tags_alongside_an_invented_one(tiny_video,
                                                            no_ffmpeg_needed):
    client = FakeClient([_identify_reply(["gore.wound_bone", "gore.guts_yummy"])])
    det = VisualScanDetector(client)
    out, _ok = det.identify(tiny_video, _candidate())
    assert "gore.wound_bone" in out.tags
    assert "gore.unclear" in out.tags


def test_identify_enforces_the_child_protection_floor(tiny_video, no_ffmpeg_needed):
    client = FakeClient([_identify_reply(["nudity.full_nude_female"], minors="yes")])
    det = VisualScanDetector(client)
    out, _ok = det.identify(tiny_video, _candidate())
    assert "safety.exposure_with_minor" in out.tags


def test_identify_records_uncertain_tags_only_when_real(tiny_video, no_ffmpeg_needed):
    client = FakeClient([_identify_reply(["violence.fight"],
                                         uncertain_tags=["violence.fight",
                                                         "gore.blood_heavy"])])
    det = VisualScanDetector(client)
    out, _ok = det.identify(tiny_video, _candidate())
    # 'uncertain' about a tag you did not assert is meaningless, so it is dropped.
    assert out.uncertain_tags == ["violence.fight"]


def test_identify_uses_more_frames_for_longer_ranges(tiny_video, no_ffmpeg_needed):
    # Six frames across a 40 s merged range is one look every 6.7 s — the exact
    # way v1 missed the brief frame that actually shows the thing.
    assert _identify_frame_count(2.0) < _identify_frame_count(30.0)
    assert _identify_frame_count(30.0) <= 12
    assert _identify_frame_count(0.5) >= 4  # never fewer than a handful


def test_unclassified_detection_keeps_the_sweep_triggers(tiny_video,
                                                         no_ffmpeg_needed):
    client = FakeClient([RuntimeError("boom")] * 10)
    det = VisualScanDetector(client)
    out, _ok = det.identify(tiny_video, _candidate(triggers=["blood", "doubt"]))
    assert out.triggers == ["blood", "doubt"]
    assert out.confidence == pytest.approx(0.6)


def test_identify_sends_dialogue_text_into_the_prompt(tiny_video, no_ffmpeg_needed):
    client = FakeClient([_identify_reply(["language.unclear"])])
    det = VisualScanDetector(client)
    det.identify(tiny_video, _candidate(), dialogue_text="I will stab you")
    assert "I will stab you" in client.prompts[0]


def test_identify_drops_frames_that_do_not_exist(monkeypatch, tiny_video):
    # Frame extraction returning nothing must fail the look, not invent a
    # detection out of zero images.
    monkeypatch.setattr("nuclearcutter.detection.visual_scan.extract_frames_in_range",
                        lambda *a, **k: [])
    client = FakeClient([_identify_reply(["nudity.shirtless_male"])])
    det = VisualScanDetector(client)
    out, ok = det.identify(tiny_video, _candidate())
    assert ok is False
    assert client.calls == 0
    assert out.tags == ["safety.unclassified"]


# --------------------------------------------------------------------------
# Sweep query concurrency (the sweep_concurrency knob)
# --------------------------------------------------------------------------

class _ProbeClient:
    """FakeClient that measures overlap and can shape per-call latency.

    `delays[i]` is the sleep of call i (index by call order); `fail_calls`
    makes those call numbers raise so a batch burns retries from the pool.
    """

    def __init__(self, delay=0.05, delays=None, fail_calls=(), flag_all=False):
        import threading
        import time
        self._lock = threading.Lock()
        self._time = time
        self._delay = delay
        self._delays = delays or {}
        self._fail_calls = set(fail_calls)
        self._reply = (_sweep_reply(["skin"], frames=[0]) if flag_all
                       else _sweep_reply([]))
        self.active = 0
        self.peak = 0
        self.calls = 0

    def vision_query_json(self, prompt, frame_paths, max_pixels=None):
        with self._lock:
            self.calls += 1
            n = self.calls
            self.active += 1
            self.peak = max(self.peak, self.active)
        self._time.sleep(self._delays.get(n - 1, self._delay))
        with self._lock:
            self.active -= 1
        if n in self._fail_calls:
            raise RuntimeError("server down")
        return dict(self._reply, triggers=list(self._reply["triggers"]),
                    frames=list(self._reply["frames"]))


def _two_frames_per_call(monkeypatch):
    from nuclearcutter.detection import visual_scan as _vs
    monkeypatch.setattr(_vs, "SWEEP_FRAMES_PER_CALL", 2)


def test_detector_clamps_sweep_concurrency():
    client = FakeClient(lambda p, f: _sweep_reply([]))
    assert VisualScanDetector(client).sweep_concurrency == \
        __import__("nuclearcutter.detection.visual_scan",
                   fromlist=["x"]).DEFAULT_SWEEP_CONCURRENCY
    assert VisualScanDetector(client, sweep_concurrency=0).sweep_concurrency == 1
    assert VisualScanDetector(client, sweep_concurrency=9).sweep_concurrency == 4


def test_sweep_overlaps_queries_when_concurrency_allows(monkeypatch, tiny_video):
    _two_frames_per_call(monkeypatch)  # 8 samples -> 4 batches
    probe = _ProbeClient(delay=0.05)
    det = VisualScanDetector(probe, sweep_concurrency=2)
    _, stats = det.sweep(tiny_video, sample_interval=0.5)
    assert stats.batches_queried == 4
    assert probe.calls == 4
    assert probe.peak == 2, "two batches must be able to run simultaneously"


def test_sweep_concurrency_one_stays_serial(monkeypatch, tiny_video):
    _two_frames_per_call(monkeypatch)
    probe = _ProbeClient(delay=0.05)
    det = VisualScanDetector(probe, sweep_concurrency=1)
    det.sweep(tiny_video, sample_interval=0.5)
    assert probe.peak == 1, "concurrency=1 must never overlap two queries"


def test_results_merge_in_chronological_order_under_overlap(monkeypatch, tiny_video):
    """A slow early batch may not let later batches overtake its RESULT:
    window merging and callbacks are chronological by contract."""
    _two_frames_per_call(monkeypatch)
    # every batch flags its first frame; batch 1's call is slow so a
    # completion-ordered implementation would emit the later batches first
    probe = _ProbeClient(delays={0: 0.4}, flag_all=True)
    flagged: list[float] = []
    det = VisualScanDetector(probe, sweep_concurrency=2)
    ranges, stats = det.sweep(tiny_video, sample_interval=0.5,
                              on_flagged_window=lambda s, e, t, c: flagged.append(s))
    assert stats.batches_failed == 0 and probe.peak == 2 and flagged, \
        "the order assertion is only meaningful with flags to order"
    assert flagged == sorted(flagged), "flagged callbacks must fire in time order"

    # and the merged ranges must equal the serial run's ranges exactly
    probe2 = _ProbeClient(delay=0.0, flag_all=True)
    serial_ranges, _ = VisualScanDetector(probe2, sweep_concurrency=1).sweep(
        tiny_video, sample_interval=0.5)
    assert [(r.start, r.end) for r in ranges] == \
        [(r.start, r.end) for r in serial_ranges]


def test_failed_batch_under_overlap_still_becomes_doubt(monkeypatch, tiny_video):
    _two_frames_per_call(monkeypatch)
    # The first five calls raise. With two workers, pigeonhole says exactly one
    # batch owns >=3 of those five and exhausts MAX_VLM_RETRIES(3); the other
    # owns <=2 and recovers on its third attempt (call 6+) — one `doubt`
    # window, deterministically, whatever the interleaving.
    probe = _ProbeClient(delay=0.02, fail_calls={1, 2, 3, 4, 5})
    det = VisualScanDetector(probe, sweep_concurrency=2)
    ranges, stats = det.sweep(tiny_video, sample_interval=0.5)
    assert stats.batches_failed == 1
    doubts = [r for r in ranges if "doubt" in r.triggers]
    assert len(doubts) == 1


def test_stop_event_mid_overlap_joins_cleanly(monkeypatch, tiny_video):
    import threading
    _two_frames_per_call(monkeypatch)
    probe = _ProbeClient(delay=0.08)
    ev = threading.Event()
    done: list[tuple[int, int]] = []

    def _progress(d, total):
        done.append((d, total))
        if len(done) >= 2:
            ev.set()

    det = VisualScanDetector(probe, sweep_concurrency=2)
    ranges, stats = det.sweep(tiny_video, sample_interval=0.5, stop_event=ev,
                              on_progress=_progress)
    assert stats.frames_done < stats.frames_total, "a stopped sweep is partial"
    assert not [t for t in threading.enumerate()
                if t.name.startswith("sweep_query")], "worker threads must join"
