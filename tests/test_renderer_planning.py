from nuclearcutter.render.renderer import (
    TimelineSegment, _absolute_ranges, _language_mute_ranges_within, _make_caption_png,
    _muted_words_within, plan_timeline,
)
from nuclearcutter.schema import (
    AudioAction, Category, FilmIdentity, LanguageDetection, Preferences, ScanResult, SeverityLevel,
    VisualAction, VisualDetection,
)


def make_identity(duration=100.0):
    return FilmIdentity(title="T", year=2024, duration_seconds=duration, phash_samples=[])


def make_seg(start, end, category=None):
    return TimelineSegment(start=start, end=end, visual=VisualAction.NONE,
                           audio=AudioAction.NONE, category=category)


def test_plan_timeline_full_coverage_no_gaps():
    identity = make_identity(100.0)
    vd1 = VisualDetection(category=Category.NUDITY, start=10.0, end=15.0, description="a", confidence=0.9)
    vd2 = VisualDetection(category=Category.GORE, start=50.0, end=60.0, description="b", confidence=0.9)
    scan = ScanResult(schema_version=1, identity=identity, visual_detections=[vd1, vd2], language_detections=[])
    prefs = Preferences(nudity_visual=VisualAction.BLUR, gore_visual=VisualAction.BLUR)

    timeline = plan_timeline(scan, prefs, 100.0)

    # Full coverage, no gaps or overlaps
    assert timeline[0].start == 0.0
    assert timeline[-1].end == 100.0
    for a, b in zip(timeline, timeline[1:]):
        assert a.end == b.start

    total = sum(s.end - s.start for s in timeline)
    assert total == 100.0

    # The nudity detection is blurred, gore is blurred
    nud = next(s for s in timeline if s.category == Category.NUDITY)
    assert nud.visual == VisualAction.BLUR
    assert nud.audio == AudioAction.NONE
    gore = next(s for s in timeline if s.category == Category.GORE)
    assert gore.visual == VisualAction.BLUR


def test_plan_timeline_overlapping_detections_no_double_count():
    identity = make_identity(100.0)
    vd1 = VisualDetection(category=Category.NUDITY, start=10.0, end=20.0, description="a", confidence=0.9)
    vd2 = VisualDetection(category=Category.GORE, start=15.0, end=25.0, description="b", confidence=0.9)
    scan = ScanResult(schema_version=1, identity=identity, visual_detections=[vd1, vd2], language_detections=[])
    prefs = Preferences(nudity_visual=VisualAction.BLUR, gore_visual=VisualAction.BLUR)

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
    prefs = Preferences(nudity_visual=VisualAction.NONE)

    timeline = plan_timeline(scan, prefs, 100.0)
    assert len(timeline) == 1
    assert timeline[0].visual == VisualAction.NONE
    assert timeline[0].start == 0.0
    assert timeline[0].end == 100.0


def test_plan_timeline_black_visual_action():
    identity = make_identity(50.0)
    vd = VisualDetection(category=Category.VIOLENCE, start=10.0, end=20.0, description="fight", confidence=0.9)
    scan = ScanResult(schema_version=1, identity=identity, visual_detections=[vd], language_detections=[])
    prefs = Preferences(violence_visual=VisualAction.BLACK)

    timeline = plan_timeline(scan, prefs, 50.0)
    seg = next(s for s in timeline if s.category == Category.VIOLENCE)
    assert seg.visual == VisualAction.BLACK
    assert seg.description == "fight"


def test_plan_timeline_visual_and_audio_propagate():
    identity = make_identity(50.0)
    vd = VisualDetection(category=Category.NUDITY, start=10.0, end=20.0, description="a", confidence=0.9)
    scan = ScanResult(schema_version=1, identity=identity, visual_detections=[vd], language_detections=[])

    prefs_muted = Preferences(nudity_visual=VisualAction.BLUR, nudity_audio=AudioAction.MUTE_SCENE)
    timeline = plan_timeline(scan, prefs_muted, 50.0)
    blur_seg = next(s for s in timeline if s.category == Category.NUDITY)
    assert blur_seg.visual == VisualAction.BLUR
    assert blur_seg.audio == AudioAction.MUTE_SCENE

    prefs_black = Preferences(nudity_visual=VisualAction.BLACK, nudity_audio=AudioAction.NONE)
    timeline2 = plan_timeline(scan, prefs_black, 50.0)
    blur_seg2 = next(s for s in timeline2 if s.category == Category.NUDITY)
    assert blur_seg2.visual == VisualAction.BLACK
    assert blur_seg2.audio == AudioAction.NONE


def test_language_mute_ranges_convert_to_segment_relative_time():
    identity = make_identity(100.0)
    ld = LanguageDetection(
        start=70.0, end=70.3, utterance_start=69.0, utterance_end=71.0,
        word="damn", transcript_source="whisper", llm_confirmed=True,
    )
    scan = ScanResult(schema_version=1, identity=identity, visual_detections=[], language_detections=[ld])
    # Default padding is 0.5s each side: (70-0.5, 70.3+0.5) -> (69.5, 70.8)
    prefs = Preferences(foul_language_audio=AudioAction.MUTE_WORD)

    ranges = _language_mute_ranges_within(scan, prefs, make_seg(60.0, 100.0))
    assert abs(ranges[0][0] - 9.5) < 1e-6  # (69.5 - 60)
    assert abs(ranges[0][1] - 10.8) < 1e-6  # (70.8 - 60)


def test_language_mute_respects_phrase_scope():
    identity = make_identity(100.0)
    ld = LanguageDetection(
        start=70.0, end=70.3, utterance_start=69.0, utterance_end=71.0,
        word="damn", transcript_source="whisper", llm_confirmed=True,
    )
    scan = ScanResult(schema_version=1, identity=identity, visual_detections=[], language_detections=[ld])
    prefs = Preferences(foul_language_audio=AudioAction.MUTE_PHRASE)

    ranges = _language_mute_ranges_within(scan, prefs, make_seg(60.0, 100.0))
    # Utterance (69, 71) padded 0.5 each side -> (68.5, 71.5)
    assert abs(ranges[0][0] - 8.5) < 1e-6
    assert abs(ranges[0][1] - 11.5) < 1e-6


def test_language_mute_padding_clamped_to_segment():
    """Padding must not push a mute outside this segment's bounds."""
    identity = make_identity(100.0)
    ld = LanguageDetection(
        start=2.0, end=2.3, utterance_start=1.0, utterance_end=3.0,
        word="damn", transcript_source="whisper", llm_confirmed=True,
    )
    scan = ScanResult(schema_version=1, identity=identity, visual_detections=[], language_detections=[ld])
    prefs = Preferences(foul_language_audio=AudioAction.MUTE_WORD)

    ranges = _language_mute_ranges_within(scan, prefs, make_seg(0.0, 5.0))
    # (2-0.5, 2.3+0.5) -> (1.5, 2.8), both inside the segment
    assert abs(ranges[0][0] - 1.5) < 1e-6
    assert abs(ranges[0][1] - 2.8) < 1e-6


def test_unconfirmed_language_detection_ignored():
    identity = make_identity(100.0)
    ld = LanguageDetection(
        start=70.0, end=70.3, utterance_start=69.0, utterance_end=71.0,
        word="damn", transcript_source="whisper", llm_confirmed=False,  # rejected by LLM check
    )
    scan = ScanResult(schema_version=1, identity=identity, visual_detections=[], language_detections=[ld])
    prefs = Preferences(foul_language_audio=AudioAction.MUTE_WORD)

    ranges = _language_mute_ranges_within(scan, prefs, make_seg(60.0, 100.0))
    assert ranges == []


def test_language_mute_zero_padding_disabled():
    """Padding can be set to 0 to mute only the exact word window."""
    identity = make_identity(100.0)
    ld = LanguageDetection(
        start=70.0, end=70.3, utterance_start=69.0, utterance_end=71.0,
        word="damn", transcript_source="whisper", llm_confirmed=True,
    )
    scan = ScanResult(schema_version=1, identity=identity, visual_detections=[], language_detections=[ld])
    prefs = Preferences(foul_language_audio=AudioAction.MUTE_WORD, mute_padding=0.0)

    ranges = _language_mute_ranges_within(scan, prefs, make_seg(60.0, 100.0))
    assert abs(ranges[0][0] - 10.0) < 1e-6
    assert abs(ranges[0][1] - 10.3) < 1e-6


def test_foul_language_audio_none_produces_no_mutes():
    identity = make_identity(100.0)
    ld = LanguageDetection(
        start=70.0, end=70.3, utterance_start=69.0, utterance_end=71.0,
        word="damn", transcript_source="whisper", llm_confirmed=True,
    )
    scan = ScanResult(schema_version=1, identity=identity, visual_detections=[], language_detections=[ld])
    prefs = Preferences(foul_language_audio=AudioAction.NONE)

    ranges = _language_mute_ranges_within(scan, prefs, make_seg(60.0, 100.0))
    assert ranges == []


def test_visual_category_audio_does_not_add_language_style_mutes():
    """Language mute ranges come only from foul_language_audio. A visual
    category's own audio action is handled separately (whole-segment silence),
    so it must not inject language-style word windows here."""
    identity = make_identity(50.0)
    ld = LanguageDetection(
        start=10.0, end=10.3, utterance_start=9.0, utterance_end=11.0,
        word="damn", transcript_source="whisper", llm_confirmed=True,
    )
    scan = ScanResult(schema_version=1, identity=identity, visual_detections=[], language_detections=[ld])
    prefs = Preferences(nudity_audio=AudioAction.MUTE_SCENE, foul_language_audio=AudioAction.NONE)

    ranges = _language_mute_ranges_within(scan, prefs, make_seg(0.0, 50.0, Category.NUDITY))
    assert ranges == []


def test_foul_language_visual_drives_segments_when_configured():
    """If foul_language_visual is set to a real action, confirmed language
    detections create visual segments over their utterance window."""
    identity = make_identity(100.0)
    ld = LanguageDetection(
        start=40.0, end=40.3, utterance_start=39.0, utterance_end=42.0,
        word="damn", transcript_source="whisper", llm_confirmed=True,
    )
    scan = ScanResult(schema_version=1, identity=identity, visual_detections=[], language_detections=[ld])

    # Default: foul_language_visual = NONE -> no segment from language.
    timeline = plan_timeline(scan, Preferences(), 100.0)
    assert len(timeline) == 1
    assert timeline[0].visual == VisualAction.NONE

    # Configured: blur the video while the speaker swears.
    prefs = Preferences(foul_language_visual=VisualAction.BLUR)
    timeline2 = plan_timeline(scan, prefs, 100.0)
    lang_seg = next(s for s in timeline2 if s.category == Category.FOUL_LANGUAGE)
    assert lang_seg.visual == VisualAction.BLUR
    assert lang_seg.start == 39.0
    assert lang_seg.end == 42.0
    assert "damn" in lang_seg.description

    # Unconfirmed language detections never drive segments.
    ld2 = LanguageDetection(
        start=70.0, end=70.3, utterance_start=69.0, utterance_end=71.0,
        word="damn", transcript_source="whisper", llm_confirmed=False,
    )
    scan2 = ScanResult(schema_version=1, identity=identity, visual_detections=[],
                       language_detections=[ld, ld2])
    timeline3 = plan_timeline(scan2, prefs, 100.0)
    cats = [s.category for s in timeline3 if s.category is not None]
    assert cats.count(Category.FOUL_LANGUAGE) == 1  # only the confirmed one


def test_plan_timeline_respects_severity_threshold():
    """low = least censorship (only the worst content), exhigh = most.
    A detection is corrected if its level rank <= the threshold rank."""
    identity = make_identity(100.0)
    vd_low = VisualDetection(category=Category.NUDITY, start=10.0, end=20.0, description="a",
                             confidence=0.9, level=SeverityLevel.LOW)  # the WORST nudity
    vd_high = VisualDetection(category=Category.GORE, start=50.0, end=60.0, description="b",
                              confidence=0.9, level=SeverityLevel.HIGH)  # mild-ish gore
    scan = ScanResult(schema_version=1, identity=identity, visual_detections=[vd_low, vd_high],
                      language_detections=[])

    # Default threshold med: LOW (worst) nudity is kept, HIGH (mild) gore is skipped.
    prefs = Preferences(nudity_visual=VisualAction.BLUR, gore_visual=VisualAction.BLUR)
    timeline = plan_timeline(scan, prefs, 100.0)
    cats = [s.category for s in timeline if s.category is not None]
    assert cats == [Category.NUDITY]  # HIGH gore filtered out at med threshold

    # exhigh threshold (most censorship) corrects everything.
    prefs_strict = Preferences(nudity_visual=VisualAction.BLUR, gore_visual=VisualAction.BLUR,
                               nudity_level=SeverityLevel.EXHIGH, gore_level=SeverityLevel.EXHIGH)
    timeline2 = plan_timeline(scan, prefs_strict, 100.0)
    cats2 = [s.category for s in timeline2 if s.category is not None]
    assert Category.NUDITY in cats2 and Category.GORE in cats2

    # low threshold (least censorship) keeps only the WORST content.
    prefs_low = Preferences(nudity_visual=VisualAction.BLUR, nudity_level=SeverityLevel.LOW,
                            gore_visual=VisualAction.BLUR, gore_level=SeverityLevel.LOW)
    timeline3 = plan_timeline(scan, prefs_low, 100.0)
    cats3 = [s.category for s in timeline3 if s.category is not None]
    assert cats3 == [Category.NUDITY]  # only the low (worst) detection


def test_language_mute_respects_severity_threshold():
    """low = least censorship. Only words at/above the threshold (rank <=
    threshold rank) get muted."""
    identity = make_identity(100.0)
    ld_low = LanguageDetection(
        start=10.0, end=10.2, utterance_start=9.0, utterance_end=11.0,
        word="fuck", transcript_source="whisper", llm_confirmed=True, level=SeverityLevel.LOW,
    )
    ld_high = LanguageDetection(
        start=20.0, end=20.2, utterance_start=19.0, utterance_end=21.0,
        word="crap", transcript_source="whisper", llm_confirmed=True, level=SeverityLevel.HIGH,
    )
    scan = ScanResult(schema_version=1, identity=identity, visual_detections=[],
                      language_detections=[ld_low, ld_high])

    # Threshold med: LOW "fuck" (worst) is muted, HIGH "crap" (mild) is not.
    prefs = Preferences(foul_language_audio=AudioAction.MUTE_WORD, foul_language_level=SeverityLevel.MED)
    ranges = _language_mute_ranges_within(scan, prefs, make_seg(0.0, 100.0))
    assert len(ranges) == 1  # only the low (worst) one
    # word 10.0-10.2 padded 0.5 -> (9.5, 10.7)
    assert abs(ranges[0][0] - 9.5) < 1e-6
    assert abs(ranges[0][1] - 10.7) < 1e-6


# ---------------------------------------------------------------------------
# Muted-audio caption plumbing (small bottom caption on still-visible video)
# ---------------------------------------------------------------------------

def test_absolute_ranges_shifts_relative_to_segment_start():
    assert _absolute_ranges([(9.5, 10.8)], 60.0) == [(69.5, 70.8)]


def test_muted_words_within_matches_phrase_scope():
    identity = make_identity(100.0)
    ld = LanguageDetection(
        start=70.0, end=70.3, utterance_start=69.0, utterance_end=71.0,
        word="damn", transcript_source="whisper", llm_confirmed=True,
    )
    scan = ScanResult(schema_version=1, identity=identity, visual_detections=[],
                      language_detections=[ld])
    prefs = Preferences(foul_language_audio=AudioAction.MUTE_PHRASE)

    # The utterance (69,71) overlaps this segment -> the word is listed.
    words = _muted_words_within(scan, prefs, make_seg(60.0, 100.0))
    assert words == ["damn"]
    # A segment that doesn't overlap the utterance gets nothing.
    assert _muted_words_within(scan, prefs, make_seg(0.0, 60.0)) == []


def test_muted_words_within_respects_unconfirmed():
    identity = make_identity(100.0)
    ld = LanguageDetection(
        start=70.0, end=70.3, utterance_start=69.0, utterance_end=71.0,
        word="damn", transcript_source="whisper", llm_confirmed=False,
    )
    scan = ScanResult(schema_version=1, identity=identity, visual_detections=[],
                      language_detections=[ld])
    prefs = Preferences(foul_language_audio=AudioAction.MUTE_PHRASE)
    assert _muted_words_within(scan, prefs, make_seg(60.0, 100.0)) == []


def test_make_caption_png_writes_small_image(tmp_path):
    out = tmp_path / "caption.png"
    _make_caption_png("A character swears.", 640, 360, None, out)
    assert out.exists()
    assert out.stat().st_size > 0
    from PIL import Image

    img = Image.open(out)
    assert img.size == (640, 360)
    assert img.mode == "RGBA"
