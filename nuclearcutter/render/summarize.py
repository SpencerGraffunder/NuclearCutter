"""
Render-time summary pass (docs/SPEC.md section 2): the optional 'summary model'
writes the on-screen text that replaces blurred/blacked footage, and the small
bottom captions that explain muted foul-language audio while the video stays
visible.

The scan-time confirm pass writes a short description per visual detection
using 6 frames. At render time we give the user a chance to point a SEPARATE,
usually larger model at each blurred/blacked segment — with up to `frames`
sampled frames AND the actual dialogue transcript for that window — so the
on-screen text is far more accurate (e.g. it stops mistaking a spaceship
interior for a hospital, which 6 under-powered frames + no dialogue do easily).
Because the summary model runs once per segment rather than every frame, a big
model is affordable.

Design notes:

  * Frames-first, text-only fallback. The vision request (frames + dialogue) is
    attempted first; if the model/backend rejects or fails it — e.g. the user
    picked a text-only model as the summary model — the same prompt is retried
    text-only using just the dialogue, so a description/caption still appears.
  * Context-length aware. The summary prompt must fit the model's LOADED
    context window. We try to read the server-reported context length for the
    selected summary model (see LLMClient.model_context_length); when the
    server reports none we fall back to `max_context` (default 30000) and warn
    when the estimated prompt (frames + dialogue + template) exceeds the bound.
  * Never breaks the render. Any failure degrades to the existing description
    (or a generic caption) and a warning in the terminal.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from nuclearcutter.prompts import get_prompt
from nuclearcutter.utils.ffmpeg import extract_frame_at
from nuclearcutter.utils.llm_client import LLMClient

# Rough vision-token cost per 360p frame used by the context-size estimate.
# Deliberately conservative (real image tokens vary by model/family).
_SUMMARY_VISION_TOKENS_PER_FRAME = 300

# Hard cap on a produced description/caption, so a runaway response can't grow
# a full-frame overlay out of control.
_MAX_TEXT_LENGTH = 600


@dataclass
class SummaryConfig:
    client: LLMClient
    video_path: Path
    transcript_path: Path | None = None
    frames: int = 12  # frames sampled across each segment (0 = text-only)
    scale_height: int = 360  # downscale frames to this height before vision
    max_context: int = 30000  # fallback LOADED context when server reports none


def _extract_n_frames(
    video_path: Path, start: float, end: float, n: int, scale_height: int | None = None,
) -> list[Path]:
    """Sample exactly `n` frames evenly across [start, end) into temp files.

    Returns the frame paths (caller deletes them when done). A short segment
    may yield fewer than `n` frames (timestamps clamp to the source).
    """
    if n <= 0:
        return []
    duration = max(0.0, end - start)
    if duration <= 0:
        return []
    out: list[Path] = []
    for i in range(n):
        ts = start + (i + 0.5) * (duration / n)
        ts = max(start, min(ts, end - 0.01))
        try:
            out.append(extract_frame_at(video_path, ts, scale_height=scale_height))
        except Exception:
            continue  # one bad frame shouldn't kill the summary pass
    return out


def _clean_summary_text(raw: str) -> str:
    """Strip the cruft models wrap around plain-text answers: markdown fences
    (and their language tags), surrounding quotes, and excess whitespace.
    Capped at _MAX_TEXT_LENGTH."""
    text = (raw or "").strip()
    if "```" in text:
        parts = [p.strip() for p in text.split("```") if p.strip()]
        if parts:
            text = max(parts, key=len)
            # Drop a fence language tag like ```text or ```json on the first line.
            lines = text.split("\n")
            if len(lines) > 1 and lines[0].strip().lower() in ("text", "json", "markdown", "md"):
                text = "\n".join(lines[1:])
            text = text.strip()
    if len(text) >= 2 and text[0] == text[-1] and text[0] in ('"', "'", "\u201c", "\u2018"):
        text = text[1:-1].strip()
    text = " ".join(text.split())  # collapse newlines -> single line
    if len(text) > _MAX_TEXT_LENGTH:
        text = text[:_MAX_TEXT_LENGTH - 3].rstrip() + "..."
    return text


class SegmentSummarizer:
    """Generates per-segment on-screen text with the summary model.

    Constructed once per render (see renderer._render_track_segments), shared
    by all worker threads. It lazily loads the transcript cache and memoizes
    the summary model's reported context length. All failures degrade to the
    caller's existing text plus an entry in `warnings`.
    """

    def __init__(self, config: SummaryConfig):
        self.config = config
        self.warnings: list[str] = []
        self._utterances = None  # lazy: None = not loaded
        self._context_length: int | None = None  # None = not resolved yet

    # ------------------------------------------------------------------
    # Transcript helpers
    # ------------------------------------------------------------------

    def _load_transcript(self):
        if self._utterances is None:
            self._utterances = []
            if self.config.transcript_path and self.config.transcript_path.exists():
                try:
                    from nuclearcutter.detection.transcribe import read_transcript_cache

                    uts = read_transcript_cache(self.config.transcript_path, self.config.video_path)
                    self._utterances = uts or []
                except Exception as exc:
                    self.warnings.append(f"could not read transcript cache: {exc}")
        return self._utterances

    def transcript_text_between(self, start: float, end: float) -> str:
        """Concatenate transcript lines that overlap [start, end)."""
        parts = []
        for u in self._load_transcript():
            if u.end > start and u.start < end:
                text = (u.text or "").strip()
                if text:
                    parts.append(text)
        return " ".join(parts)

    # ------------------------------------------------------------------
    # Context-length awareness
    # ------------------------------------------------------------------

    def _summary_model_name(self, vision: bool) -> str:
        cfg = self.config.client.config
        if cfg.summary_model:
            return cfg.summary_model
        return cfg.vlm_model if vision else cfg.text_model

    def effective_context(self) -> int:
        """The loaded context window we should assume for the summary model.

        Uses the server-reported value when available (see
        LLMClient.model_context_length); otherwise falls back to max_context.
        """
        if self._context_length is None:
            reported = self.config.client.model_context_length(self._summary_model_name(vision=True))
            self._context_length = reported if isinstance(reported, int) and reported > 0 else None
        return self._context_length or self.config.max_context

    @staticmethod
    def _estimate_tokens(text: str, frames: int) -> int:
        """Rough prompt-size estimate (text tokens + vision tokens)."""
        text_tokens = int(len(text) / 4)  # ~4 chars/token for English
        vision_tokens = frames * _SUMMARY_VISION_TOKENS_PER_FRAME
        return text_tokens + vision_tokens

    def context_warning(self, prompt: str, frames: int) -> str | None:
        """Return a warning when the summary prompt likely exceeds the loaded
        context window of the summary model, else None."""
        est = self._estimate_tokens(prompt, frames)
        limit = self.effective_context()
        if est > limit:
            return (
                f"summary prompt may exceed the summary model's {limit}-token "
                f"loaded context (est. ~{est} tokens) — the on-screen text could "
                f"be truncated. Use a larger context or fewer summary frames."
            )
        return None

    def preflight_context_warning(self) -> str | None:
        """Estimate the summary prompt size for the configured frame count and
        warn if it exceeds the model's loaded context. Called before rendering
        so the GUI can surface it before any segment is encoded."""
        model = self._summary_model_name(vision=True)
        reported = self.config.client.model_context_length(model)
        self._context_length = reported if isinstance(reported, int) and reported > 0 else None
        limit = self.effective_context()
        est = self._estimate_tokens("", self.config.frames)
        if est > limit:
            src = f"reported" if reported else "assumed"
            return (
                f"summary model {model!r}: estimated summary prompt ~{est} tokens "
                f"vs {limit} {src} context — text may be truncated. Load the model "
                f"with a larger context or reduce Summary frames."
            )
        return None

    # ------------------------------------------------------------------
    # Description generation (blurred/blacked segments)
    # ------------------------------------------------------------------

    # How much un-blurred footage to sample around the flagged part (seconds
    # before/after) and how many total CONTEXT frames to extract there — the
    # model sees these so it knows the situation, but is told to describe ONLY
    # the flagged/description frames.
    _CONTEXT_SECONDS = 8.0
    _CONTEXT_FRAMES = 6

    def describe_range(self, start: float, end: float, description: str = "",
                       action: str = "blurred",
                       context_seconds: float | None = None,
                       context_frames: int | None = None) -> str:
        """Produce an improved on-screen description for a blur/black range.

        The model gets TWO kinds of frames: DESCRIPTION frames sampled across
        the flagged/blurred range (the part it must describe), and CONTEXT
        frames sampled from the surrounding un-blurred footage (before + after)
        plus the transcript — so it knows who/what/why but describes ONLY the
        hidden part (e.g. it can say "close-up of the wound being stapled"
        instead of re-describing the character it saw in context). Returns the
        new text, or "" on failure (the caller keeps the original description).
        """
        try:
            start, end = float(start), float(end)
            ctx_s = float(context_seconds) if context_seconds is not None else self._CONTEXT_SECONDS
            ctx_frames = int(context_frames) if context_frames is not None else self._CONTEXT_FRAMES

            # DESCRIPTION frames: the hidden/blurred part itself.
            desc_frames = _extract_n_frames(
                self.config.video_path, start, end, self.config.frames, self.config.scale_height
            )
            # CONTEXT frames: the surrounding un-blurred footage (before + after).
            before_n = max(0, ctx_frames // 2)
            after_n = max(0, ctx_frames - before_n)
            ctx_before = _extract_n_frames(
                self.config.video_path, max(0.0, start - ctx_s), start,
                before_n, self.config.scale_height,
            )
            ctx_after = _extract_n_frames(
                self.config.video_path, end, end + ctx_s,
                after_n, self.config.scale_height,
            )
            # Chronological order: context-before, description, context-after.
            frame_paths = ctx_before + desc_frames + ctx_after
            dialogue = self.transcript_text_between(max(0.0, start - ctx_s), end + ctx_s)

            try:
                prompt = get_prompt(
                    "summary_prompt",
                    action=action,
                    context_before=len(ctx_before),
                    description_frames=len(desc_frames),
                    context_after=len(ctx_after),
                    start=f"{start:.1f}",
                    end=f"{end:.1f}",
                    description=(description or "(none)"),
                    dialogue=(dialogue or "(none)"),
                )
                warn = self.context_warning(prompt, len(frame_paths))
                if warn and warn not in self.warnings:
                    self.warnings.append(warn)
                raw, _used_vision = self.config.client.summary_query(prompt, frame_paths)
            finally:
                for p in frame_paths:
                    try:
                        p.unlink(missing_ok=True)
                    except OSError:
                        pass
            text = _clean_summary_text(raw)
            if text:
                return text
            self.warnings.append("summary model returned an empty description")
        except Exception as exc:
            self.warnings.append(
                f"summary pass failed for {start:.0f}s\u2013{end:.0f}s: {exc}"
            )
        return ""

    def describe_segment(self, seg) -> str:
        """Convenience wrapper around describe_range for a render TimelineSegment."""
        action = "blurred" if str(getattr(seg.visual, "value", "")).lower() == "blur" else "hidden"
        return self.describe_range(seg.start, seg.end, seg.description or "", action=action)

    # ------------------------------------------------------------------
    # Muted-audio caption generation (video still visible)
    # ------------------------------------------------------------------

    def caption_for_segment(
        self, mute_ranges_abs: list[tuple[float, float]], words: list[str],
    ) -> str:
        """Produce a SHORT bottom caption for an audio-only mute segment.

        `mute_ranges_abs` are the muted windows in ABSOLUTE movie time;
        `words` are the confirmed foul words within the segment. Uses the
        summary/text model to paraphrase the muted dialogue; falls back to a
        generic caption when the model isn't reachable or there's no dialogue.
        """
        parts = []
        for r0, r1 in mute_ranges_abs:
            slice_text = self.transcript_text_between(r0, r1)
            if slice_text:
                parts.append(slice_text)
        dialogue = " ".join(parts)
        words_txt = ", ".join(sorted({w for w in words if w})) or "(none)"

        generic = "Audio muted (foul language)"
        if not dialogue.strip():
            return generic  # no transcript to paraphrase — nothing for a model to add

        try:
            prompt = get_prompt(
                "muted_caption_prompt", dialogue=dialogue, words=words_txt,
            )
            raw, _used_vision = self.config.client.summary_query(prompt, [])
            caption = _clean_summary_text(raw)
            return caption or generic
        except Exception as exc:
            self.warnings.append(f"muted caption failed: {exc}")
            return generic


def run_summary_pass(
    llm_config,
    video_path: Path,
    detections,
    summary_model: str,
    summary_frames: int = 12,
    summary_max_context: int = 30000,
    on_progress=None,
    client_factory=None,
) -> list[str]:
    """Run the summary model over a list of detections at SCAN time.

    Enriches each detection's `description` in place with the summary model —
    using sampled frames + the transcript slice for that window + the existing
    description as a seed — so the timeline and the final render both carry the
    improved on-screen text without re-running at render time. Each item in
    `detections` needs `.start`, `.end` and `.description` attributes (e.g. a
    `VisualDetection`).

    Returns the collected warnings (context-size warnings, per-range failures);
    any failure degrades to keeping the original description. `on_progress(i,
    total)`, if given, is called after each detection. `client_factory(cfg)`, if
    given, builds the LLMClient (so the caller's usage/request-log hooks attach
    to the summary requests too).
    """
    import copy

    cfg = copy.copy(llm_config)
    cfg.summary_model = summary_model
    cfg.summary_frames = summary_frames
    cfg.summary_max_context = summary_max_context
    client = client_factory(cfg) if client_factory is not None else LLMClient(cfg)
    transcript_path = video_path.with_suffix(".nuclearcutter.transcript.json")
    summarizer = SegmentSummarizer(SummaryConfig(
        client=client,
        video_path=video_path,
        transcript_path=transcript_path if transcript_path.exists() else None,
        frames=summary_frames,
        max_context=summary_max_context,
    ))
    warnings = []
    pf = summarizer.preflight_context_warning()
    if pf:
        warnings.append(pf)
    total = len(detections)
    for i, d in enumerate(detections):
        new_desc = summarizer.describe_range(d.start, d.end, d.description or "")
        if new_desc:
            d.description = new_desc
        if on_progress:
            on_progress(i + 1, total)
    warnings.extend(summarizer.warnings)
    return warnings
