from nuclearcutter.render.renderer import _language_mute_ranges_within, plan_timeline
from nuclearcutter.schema import (
    Action, Category, FilmIdentity, LanguageDetection, Preferences, ScanResult, VisualDetection,
)


def make_identity(duration=100.0):
    return FilmIdentity(title="T", year=2024, duration_seconds=duration, phash_samples=[])


def test_plan_timeline_full_coverage_no_gaps():
    identity = make_identity(100.0)
    vd1 = VisualDetection(category=Category.NUDITY, start=10.0, end=15.0, description="a", confidence=0.9)
    vd2 = VisualDetection(category=Category.INTIMATE_SCENES, start=50.0, end=60.0, description="b", confidence=0.9)
    scan = ScanResult(schema_version=1, identity=identity, visual_detections=[vd1, vd2], language_detections=[])
    prefs = Preferences(nudity_action=Action.BLUR, intimate_scenes_action=Action.BLUR)

    timeline = plan_timeline(scan, prefs, 100.0)

    # Full coverage, no gaps or overlaps
    assert timeline[0].start == 0.0
    assert timeline[-1].end == 100.0
    for a, b in zip(timeline, timeline[1:]):
        assert a.end == b.start

    total = sum(s.end - s.start for s in timeline)
    assert total == 100.0


def test_plan_timeline_overlapping_detections_no_double_count():
    identity = make_identity(100.0)
    vd1 = VisualDetection(category=Category.NUDITY, start=10.0, end=20.0, description="a", confidence=0.9)
    vd2 = VisualDetection(category=Category.INTIMATE_SCENES, start=15.0, end=25.0, description="b", confidence=0.9)
    scan = ScanResult(schema_version=1, identity=identity, visual_detections=[vd1, vd2], language_detections=[])
    prefs = Preferences(nudity_action=Action.BLUR, intimate_scenes_action=Action.BLUR)

    timeline = plan_timeline(scan, prefs, 100.0)
    total = sum(s.end - s.start for s in timeline)
    assert total == 100.0
    # No segment should start before the previous one ends
    for a, b in zip(timeline, timeline[1:]):
        assert b.start >= a.end


def test_plan_timeline_none_action_creates_no_segment():
    identity = make_identity(100.0)
    vd = VisualDetection(category=Category.NUDITY, start=10.0, end=20.0, description="a", confidence=0.9)
    scan = ScanResult(schema_version=1, identity=identity, visual_detections=[vd], language_detections=[])
    prefs = Preferences(nudity_action=Action.NONE)

    timeline = plan_timeline(scan, prefs, 100.0)
    assert len(timeline) == 1
    assert timeline[0].action is None
    assert timeline[0].start == 0.0
    assert timeline[0].end == 100.0


def test_blur_mute_audio_flag_propagates():
    identity = make_identity(50.0)
    vd = VisualDetection(category=Category.NUDITY, start=10.0, end=20.0, description="a", confidence=0.9)
    scan = ScanResult(schema_version=1, identity=identity, visual_detections=[vd], language_detections=[])

    prefs_muted = Preferences(nudity_action=Action.BLUR, nudity_blur_mute_audio=True)
    timeline = plan_timeline(scan, prefs_muted, 50.0)
    blur_seg = next(s for s in timeline if s.action == Action.BLUR)
    assert blur_seg.audio_muted is True

    prefs_unmuted = Preferences(nudity_action=Action.BLUR, nudity_blur_mute_audio=False)
    timeline2 = plan_timeline(scan, prefs_unmuted, 50.0)
    blur_seg2 = next(s for s in timeline2 if s.action == Action.BLUR)
    assert blur_seg2.audio_muted is False


def test_blur_audio_mutes_only_when_configured():
    identity = make_identity(50.0)
    vd = VisualDetection(category=Category.NUDITY, start=10.0, end=20.0, description="a", confidence=0.9)
    scan = ScanResult(schema_version=1, identity=identity, visual_detections=[vd], language_detections=[])
    prefs = Preferences(nudity_action=Action.BLUR, nudity_blur_mute_audio=True)

    timeline = plan_timeline(scan, prefs, 50.0)
    blur_seg = next(s for s in timeline if s.action == Action.BLUR)
    assert blur_seg.audio_muted is True


def test_language_mute_ranges_convert_to_segment_relative_time():
    identity = make_identity(100.0)
    ld = LanguageDetection(
        start=70.0, end=70.3, utterance_start=69.0, utterance_end=71.0,
        word="damn", transcript_source="whisper", llm_confirmed=True,
    )
    scan = ScanResult(schema_version=1, identity=identity, visual_detections=[], language_detections=[ld])
    prefs = Preferences(foul_language_action=Action.MUTE, foul_language_mute_scope="word")

    ranges = _language_mute_ranges_within(scan, prefs, seg_start=60.0, seg_end=100.0)
    assert ranges == [(10.0, 10.299999999999997)] or (
        abs(ranges[0][0] - 10.0) < 1e-6 and abs(ranges[0][1] - 10.3) < 1e-6
    )


def test_language_mute_respects_utterance_scope():
    identity = make_identity(100.0)
    ld = LanguageDetection(
        start=70.0, end=70.3, utterance_start=69.0, utterance_end=71.0,
        word="damn", transcript_source="whisper", llm_confirmed=True,
    )
    scan = ScanResult(schema_version=1, identity=identity, visual_detections=[], language_detections=[ld])
    prefs = Preferences(foul_language_action=Action.MUTE, foul_language_mute_scope="utterance")

    ranges = _language_mute_ranges_within(scan, prefs, seg_start=60.0, seg_end=100.0)
    assert abs(ranges[0][0] - 9.0) < 1e-6
    assert abs(ranges[0][1] - 11.0) < 1e-6


def test_unconfirmed_language_detection_ignored():
    identity = make_identity(100.0)
    ld = LanguageDetection(
        start=70.0, end=70.3, utterance_start=69.0, utterance_end=71.0,
        word="damn", transcript_source="whisper", llm_confirmed=False,  # rejected by LLM check
    )
    scan = ScanResult(schema_version=1, identity=identity, visual_detections=[], language_detections=[ld])
    prefs = Preferences(foul_language_action=Action.MUTE)

    ranges = _language_mute_ranges_within(scan, prefs, seg_start=60.0, seg_end=100.0)
    assert ranges == []


def test_foul_language_action_none_produces_no_mutes():
    identity = make_identity(100.0)
    ld = LanguageDetection(
        start=70.0, end=70.3, utterance_start=69.0, utterance_end=71.0,
        word="damn", transcript_source="whisper", llm_confirmed=True,
    )
    scan = ScanResult(schema_version=1, identity=identity, visual_detections=[], language_detections=[ld])
    prefs = Preferences(foul_language_action=Action.NONE)

    ranges = _language_mute_ranges_within(scan, prefs, seg_start=60.0, seg_end=100.0)
    assert ranges == []
