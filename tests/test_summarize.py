"""Tests for the render-time summary pass (nuclearcutter/render/summarize.py):

the separate 'summary model' that rewrites the blur/black on-screen text from
frames + transcript, and writes the small captions for muted-audio segments.
"""

from pathlib import Path
from unittest.mock import MagicMock, patch

from nuclearcutter.render.summarize import (
    SegmentSummarizer, SummaryConfig, _clean_summary_text, _extract_n_frames,
)
from nuclearcutter.utils.llm_client import LLMConfig


def _summarizer(client, tmp_path, **kw):
    video = tmp_path / "movie.mkv"
    video.write_bytes(b"x")
    transcript = tmp_path / "movie.nuclearcutter.transcript.json"
    transcript.write_text("{}")
    defaults = dict(client=client, video_path=video, transcript_path=transcript,
                    frames=12, max_context=30000)
    defaults.update(kw)
    return SegmentSummarizer(SummaryConfig(**defaults))


def _client(**cfg_kw):
    cfg = LLMConfig(
        base_url="http://x/v1", vlm_model="vlm", text_model="text",
        summary_model="summary", summary_frames=12, summary_max_context=30000,
    )
    for k, v in cfg_kw.items():
        setattr(cfg, k, v)
    client = MagicMock()
    client.config = cfg
    client.model_context_length.return_value = None  # server reports nothing by default
    return cfg, client


class TestCleanSummaryText:
    def test_strips_markdown_fence(self):
        assert _clean_summary_text("```text\nHello world\n```") == "Hello world"

    def test_strips_surrounding_quotes(self):
        assert _clean_summary_text('"A character falls."') == "A character falls."

    def test_collapses_newlines(self):
        assert _clean_summary_text("line one\nline two\n") == "line one line two"

    def test_caps_length(self):
        assert len(_clean_summary_text("x" * 10000)) <= 600

    def test_empty(self):
        assert _clean_summary_text("   ") == ""


class TestExtractNFrames:
    @patch("nuclearcutter.render.summarize.extract_frame_at")
    def test_samples_n_frames_evenly(self, mock_extract):
        mock_extract.side_effect = lambda v, ts, scale_height=None: Path(f"/tmp/f_{ts:.2f}.png")
        frames = _extract_n_frames(Path("/tmp/m.mp4"), 10.0, 22.0, 12, 360)
        assert len(frames) == 12
        assert mock_extract.call_count == 12
        timestamps = [c.args[1] for c in mock_extract.call_args_list]
        assert timestamps[0] >= 10.0 and timestamps[-1] <= 22.0
        assert timestamps[0] < timestamps[-1]
        # all sampled at the requested scale
        assert all(c.kwargs.get("scale_height") == 360 for c in mock_extract.call_args_list)

    @patch("nuclearcutter.render.summarize.extract_frame_at")
    def test_zero_frames(self, mock_extract):
        assert _extract_n_frames(Path("/tmp/m.mp4"), 0.0, 10.0, 0) == []
        mock_extract.assert_not_called()


class TestTranscriptSlicing:
    def _utterances(self):
        from nuclearcutter.detection.transcribe import Utterance

        return [
            Utterance(text="First line.", start=0.0, end=3.0, words=[]),
            Utterance(text="Second line.", start=3.0, end=6.0, words=[]),
        ]

    def test_transcript_text_between_overlaps(self, tmp_path):
        cfg, client = _client()
        s = _summarizer(client, tmp_path)
        with patch("nuclearcutter.detection.transcribe.read_transcript_cache",
                   return_value=self._utterances()):
            assert s.transcript_text_between(2.5, 5.0) == "First line. Second line."
            assert s.transcript_text_between(10.0, 20.0) == ""


class TestDescribeSegment:
    @staticmethod
    def _seg(start=10.0, end=22.0, visual="blur"):
        from nuclearcutter.render.renderer import TimelineSegment
        from nuclearcutter.schema import AudioAction, VisualAction

        return TimelineSegment(start=start, end=end,
                               visual=VisualAction(visual), audio=AudioAction.NONE,
                               description="old desc")

    @patch("nuclearcutter.render.summarize._extract_n_frames",
           return_value=[Path("/tmp/f1.png"), Path("/tmp/f2.png")])
    def test_improves_description_with_frames(self, mock_frames, tmp_path):
        cfg, client = _client()
        client.summary_query.return_value = ("The crew react to an alarm.", True)
        s = _summarizer(client, tmp_path)
        text = s.describe_segment(self._seg())
        assert text == "The crew react to an alarm."
        # The vision call received the frame paths.
        assert client.summary_query.call_args[0][1]

    def test_uses_transcript_and_seed_description(self, tmp_path):
        from nuclearcutter.detection.transcribe import Utterance

        cfg, client = _client()
        client.summary_query.return_value = ("Better text.", True)
        s = _summarizer(client, tmp_path)
        with patch("nuclearcutter.render.summarize._extract_n_frames",
                   return_value=[Path("/tmp/f1.png")]), \
             patch("nuclearcutter.detection.transcribe.read_transcript_cache",
                   return_value=[Utterance(text="Danger ahead.", start=11.0, end=13.0, words=[])]):
            s.describe_segment(self._seg())
        prompt = client.summary_query.call_args[0][0]
        assert "Danger ahead." in prompt  # transcript slice included
        assert "old desc" in prompt  # existing description used as a seed
        assert "22.0" in prompt  # segment end time passed

    @patch("nuclearcutter.render.summarize._extract_n_frames", return_value=[])
    def test_failure_returns_empty_and_warns(self, mock_frames, tmp_path):
        cfg, client = _client()
        client.summary_query.side_effect = RuntimeError("server down")
        s = _summarizer(client, tmp_path)
        assert s.describe_segment(self._seg()) == ""
        assert any("summary pass failed" in w for w in s.warnings)

    @patch("nuclearcutter.render.summarize._extract_n_frames", return_value=[])
    def test_empty_response_returns_empty(self, mock_frames, tmp_path):
        cfg, client = _client()
        client.summary_query.return_value = ("   ", False)
        s = _summarizer(client, tmp_path)
        assert s.describe_segment(self._seg()) == ""
        assert any("empty description" in w for w in s.warnings)


class TestCaption:
    def _utt(self, text, start, end):
        from nuclearcutter.detection.transcribe import Utterance

        return Utterance(text=text, start=start, end=end, words=[])

    def test_caption_uses_muted_dialogue(self, tmp_path):
        cfg, client = _client()
        client.summary_query.return_value = ("A character swears.", False)
        s = _summarizer(client, tmp_path)
        with patch("nuclearcutter.detection.transcribe.read_transcript_cache",
                   return_value=[self._utt("You damn fool.", 70.0, 72.0)]):
            cap = s.caption_for_segment([(70.0, 72.0)], ["damn"])
        assert cap == "A character swears."
        prompt = client.summary_query.call_args[0][0]
        assert "damn" in prompt and "You damn fool." in prompt

    def test_generic_caption_when_no_dialogue(self, tmp_path):
        cfg, client = _client()
        s = _summarizer(client, tmp_path)
        with patch("nuclearcutter.detection.transcribe.read_transcript_cache", return_value=[]):
            cap = s.caption_for_segment([(70.0, 72.0)], ["damn"])
        assert cap == "Audio muted (foul language)"
        client.summary_query.assert_not_called()

    def test_caption_falls_back_on_model_error(self, tmp_path):
        cfg, client = _client()
        client.summary_query.side_effect = RuntimeError("down")
        s = _summarizer(client, tmp_path)
        with patch("nuclearcutter.detection.transcribe.read_transcript_cache",
                   return_value=[self._utt("hi", 0.0, 1.0)]):
            cap = s.caption_for_segment([(0.0, 1.0)], [])
        assert cap == "Audio muted (foul language)"


class TestContextAwareness:
    def test_effective_context_uses_reported(self, tmp_path):
        cfg, client = _client()
        client.model_context_length.return_value = 8192
        s = _summarizer(client, tmp_path)
        assert s.effective_context() == 8192

    def test_effective_context_falls_back(self, tmp_path):
        cfg, client = _client()
        client.model_context_length.return_value = None
        s = _summarizer(client, tmp_path, max_context=30000)
        assert s.effective_context() == 30000

    def test_context_warning_when_too_small(self, tmp_path):
        cfg, client = _client()
        client.model_context_length.return_value = 4096
        s = _summarizer(client, tmp_path)
        warn = s.context_warning("x" * 4000, 12)
        assert warn is not None
        assert "4096" in warn

    def test_no_warning_when_fits(self, tmp_path):
        cfg, client = _client()
        client.model_context_length.return_value = 32768
        s = _summarizer(client, tmp_path)
        assert s.context_warning("x" * 4000, 12) is None

    def test_preflight_warning_uses_configured_frames(self, tmp_path):
        cfg, client = _client()
        client.model_context_length.return_value = 2048
        s = _summarizer(client, tmp_path, frames=12)
        warn = s.preflight_context_warning()
        assert warn is not None
        assert "2048" in warn


class TestRunSummaryPass:
    """The scan-time helper that rewrites every visual detection's description."""

    @staticmethod
    def _det(start, end, desc):
        from nuclearcutter.schema import Category, SeverityLevel, VisualDetection

        return VisualDetection(category=Category.NUDITY, start=start, end=end,
                               description=desc, confidence=0.9, level=SeverityLevel.MED)

    def test_enriches_descriptions_in_place(self, tmp_path):
        from nuclearcutter.render.summarize import run_summary_pass
        from nuclearcutter.utils.llm_client import LLMConfig

        video = tmp_path / "movie.mkv"
        video.write_bytes(b"x")
        dets = [self._det(10.0, 22.0, "old1"), self._det(50.0, 60.0, "old2")]
        llm_cfg = LLMConfig(base_url="http://x/v1", vlm_model="v", text_model="t")
        progress = []

        with patch("nuclearcutter.render.summarize.SegmentSummarizer") as MockS:
            inst = MockS.return_value
            inst.preflight_context_warning.return_value = None
            inst.describe_range.side_effect = lambda s, e, d: f"NEW[{s:.0f}-{e:.0f}]"
            inst.warnings = []
            warnings = run_summary_pass(
                llm_cfg, video, dets, summary_model="sum", summary_frames=12,
                on_progress=lambda i, t: progress.append((i, t)),
            )

        assert dets[0].description == "NEW[10-22]"
        assert dets[1].description == "NEW[50-60]"
        # the original description is used as the seed
        assert inst.describe_range.call_args_list[0].args[2] == "old1"
        assert inst.describe_range.call_args_list[1].args[2] == "old2"
        assert progress == [(1, 2), (2, 2)]
        assert warnings == []

    def test_empty_result_keeps_original(self, tmp_path):
        from nuclearcutter.render.summarize import run_summary_pass
        from nuclearcutter.utils.llm_client import LLMConfig

        video = tmp_path / "movie.mkv"
        video.write_bytes(b"x")
        dets = [self._det(10.0, 22.0, "keep me")]
        llm_cfg = LLMConfig(base_url="http://x/v1", vlm_model="v", text_model="t")

        with patch("nuclearcutter.render.summarize.SegmentSummarizer") as MockS:
            inst = MockS.return_value
            inst.preflight_context_warning.return_value = None
            inst.describe_range.return_value = ""  # model returned nothing
            inst.warnings = ["empty description"]
            run_summary_pass(llm_cfg, video, dets, summary_model="sum")

        assert dets[0].description == "keep me"

    def test_preflight_warning_surfaced(self, tmp_path):
        from nuclearcutter.render.summarize import run_summary_pass
        from nuclearcutter.utils.llm_client import LLMConfig

        video = tmp_path / "movie.mkv"
        video.write_bytes(b"x")
        dets = [self._det(10.0, 22.0, "x")]
        llm_cfg = LLMConfig(base_url="http://x/v1", vlm_model="v", text_model="t")

        with patch("nuclearcutter.render.summarize.SegmentSummarizer") as MockS:
            inst = MockS.return_value
            inst.preflight_context_warning.return_value = "context too small"
            inst.describe_range.return_value = "new"
            inst.warnings = []
            warnings = run_summary_pass(llm_cfg, video, dets, summary_model="sum")

        assert any("context too small" in w for w in warnings)
        assert dets[0].description == "new"
