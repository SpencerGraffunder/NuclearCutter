"""
Visual detection: a cheap full-film *trigger* sweep, then ONE classify pass per
candidate range.

Two passes, and they are not redundant, because they answer different questions
at different resolutions:

  sweep    "is there anything here worth looking at?"     12 coarse triggers,
          cheap pixels, ~900 calls for a feature film. A boolean-ish answer
          that fails safe: an unparseable answer counts as NOT verified clean.

  identify "what exactly is in it?"                        the full tag
          vocabulary, the dialogue, better pixels, a few dozen calls.

Why not one pass over the whole film: the vocabulary has ~120 categories whose
differences are things a model can only judge from real pixels — nipple versus
covered breast, real versus stage blood, child versus short adult. Cost here is
pixels, roughly linear, measured at ~16 s per megapixel of images per call on a
two-GPU 3080 box (4x410k px ~27s, 4x102k px ~7s; see SWEEP_QUALITY_PIXELS).
Answering 120 categories at 410k px per frame across a 2 h film at 2 s sampling
is 900 calls x ~27 s ~ 6.8 h, versus ~1.8 h for a 102k px trigger sweep plus a
few dozen classify calls. One pass is not cheaper; it is ~4x more expensive and
still answers the fine questions off thumbnails. The measured sweep/classify
split lives in scripts/probe_vlm_cost.py.

History that still binds the design: the pipeline once used NudeNet (a local
CNN) as the cheap gate in front of the VLM. It scored zero on every frame of a
real nude scene that a VLM flagged at 0.95, so the scene was never seen. A
cheap detector may ADD work and may NEVER remove it. Both passes here obey
that: the sweep only decides where to look, and anything the classify pass
cannot explain becomes `safety.unclassified` — visible and censorable — rather
than nothing.
"""

from __future__ import annotations

import logging
import shutil
import sys
import tempfile
from collections import Counter, OrderedDict
from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import dataclass, field
from pathlib import Path

from nuclearcutter.prompts import get_prompt
from nuclearcutter.taxonomy import (
    Family, Minors, Rendering, TAGS, enforce_minor_protection, normalize_tags,
    sweep_trigger_block, taxonomy_prompt,
)
from nuclearcutter.schema import VisualDetection
from nuclearcutter.utils.ffmpeg import (
    extract_frame_at, extract_frames_chunk, extract_frames_in_range, probe_duration,
)
from nuclearcutter.utils.llm_client import LLMClient

# ---------------------------------------------------------------------------
# Tuning
# ---------------------------------------------------------------------------

# Seconds between sampled frames in the full-film sweep. Smaller catches shorter
# scenes but costs calls: total sweep cost is samples x pixels, so this is the
# main cost dial along with sweep_resolution.
DEFAULT_SWEEP_INTERVAL_SECONDS = 2.0
DEFAULT_SWEEP_INTERVAL = DEFAULT_SWEEP_INTERVAL_SECONDS

# Sampled frames per batched VLM call. Was 4 when image pixel cost dominated
# the sweep; with sweep_resolution capping pixels and thinking off, what
# dominates instead is a fixed ~8s cost per fresh request on the model server
# (the conversation cache cannot carry over across batches because every batch
# has new pictures) plus ~0.5-1s per image. So MORE frames per call is the
# speed lever: measured live with never-seen 240p frames, 4/call = 2.8-3.6
# s/frame, 8/call = 1.5, 16/call = 0.9. Recall-checked against a full-film
# sidecar ground truth (10-min clip, 302 samples): 16/call covers 75/80 (94%)
# GT detections at 375s vs 8/call 72/80 (90%) at 523s -- faster AND better
# recall; 4/call only reaches 96% and costs 3.7x.
SWEEP_FRAMES_PER_CALL = 16

# How many batches' frames one chunked ffmpeg pass extracts. Frames are grabbed
# in chunks (seek once, decode forward) instead of one seek+decode PER FRAME:
# per-frame seeking re-pays keyframe search + codec startup for every sample
# (~1-4s each on 1080p HEVC), which dominated full-film scan time. The chunk
# pass costs ~0.2-0.3s/frame AND the next chunk is extracted on a background
# thread while the VLM chews the current batch.
SWEEP_CHUNK_BATCHES = 8

# How many sweep VLM calls to keep in flight at once. The idea was to hide the
# fixed per-request cost (~8s of prompt + vision preprocessing against ~1s of
# decode) by giving the server a second prompt to read beside a running
# decode. MEASURED AS A LOSS on servers that queue instead of truly decoding
# conversations together (llama.cpp-style slot batching): two in flight cost
# +18-50% wall per call and multiply p90 latency. The knob stays for servers
# that do batch conversations; the default is serial. Results are merged in
# chronological order either way. Clamped to 1..4.
DEFAULT_SWEEP_CONCURRENCY = 1

# Frame scale-down presets selectable in the web GUI. The chosen height also
# bounds the VLM request's vision_max_pixels so images are never sent larger.
SCALE_HEIGHTS = {"360p": 360, "480p": 480, "720p": 720, "1080p": 1080}
DEFAULT_SCALE = "480p"


def scale_height_for(scale: str | None) -> int:
    return SCALE_HEIGHTS.get((scale or DEFAULT_SCALE).strip().lower(),
                             SCALE_HEIGHTS[DEFAULT_SCALE])


def vision_max_pixels_for_scale(scale: str | None) -> int:
    h = scale_height_for(scale)
    return int(h * (h * 16 / 9))


# Per-image pixel caps for the SWEEP pass, keyed by the GUI's "sweep_resolution".
# The sweep is the hot path (2000+ vision calls per movie vs dozens of classify
# calls) and the server's VISION PREPROCESSING cost — not model prefill —
# dominates each call: measured on the LLPC/Strata box, 4 images at 410k px take
# ~27 s round-trip vs ~7 s at 4x102k px, and the server's own text-prefill log
# counts only ~9 s of that.
#
# Since v2 the sweep asks a strictly easier question than it used to (12 coarse
# triggers instead of three full severity scales), and it no longer writes a
# description, so it can look at fewer pixels than it used to need. "160p" is
# new for that reason. Classify always uses the full configured scale: the tag
# vocabulary asks questions that a 240p thumbnail cannot answer, and using the
# sweep's cap for both passes (as v1 did) is why the old "scale" setting quietly
# degraded classification.
SWEEP_QUALITY_PIXELS = {
    "full": None,
    "360p": 230_400,
    "240p": 102_400,
    "160p": 57_600,
}

IDENTIFY_FRAMES_MIN = 4
IDENTIFY_FRAMES_MAX = 12
# One frame per this many seconds of segment, clamped to the range above. The v1
# fixed 6 frames were spread across the PADDED span, so a 40 s candidate got one
# frame per ~6.7 s and several of the six landed on guaranteed-clean padding —
# the classify pass could literally miss the frames the sweep flagged.
IDENTIFY_SECONDS_PER_FRAME = 2.5

MAX_VLM_RETRIES = 3

# Padding and merge distance are DERIVED from the sweep interval, not hardcoded:
# frames are sampled every `interval`, and any sampled frame NOT in a flagged
# batch is a verified-clean frame we must never blur past. So pad by exactly one
# interval (reaching up to, not past, the next clean sample) and merge two
# flagged batches within one interval of each other. This keeps the blur window
# tight and never extends over clean footage.
def _padding_for_interval(interval: float) -> float:
    return interval


def _merge_gap_for_interval(interval: float) -> float:
    return interval


def _frame_padding_for_interval(interval: float) -> float:
    """Padding for FRAME-level flagged windows (half an interval each side), so
    a single flagged sample blurs ~one interval instead of the whole 4-frame
    batch. Batch-level fallback keeps `_padding_for_interval` (full interval)."""
    return interval * 0.5


# ---------------------------------------------------------------------------
# Prompt construction (vocabulary-driven, never hand-copied)
# ---------------------------------------------------------------------------

def build_sweep_prompt(n: int) -> str:
    """The trigger prompt. Its 12-item list comes from taxonomy.SWEEP_TRIGGERS."""
    return get_prompt("sweep_prompt", n=n, sweep_triggers=sweep_trigger_block())


def build_identify_prompt(n: int, dialogue_text: str,
                          genders: tuple[str, ...] | None = None) -> str:
    """The classify prompt, carrying the WHOLE visual vocabulary.

    Deliberately not narrowed to the sweep's guess at a family. v1 handed the
    confirm pass only the one category the sweep had picked, and the sweep's
    merge step kept the FIRST flagged window's category and never updated it —
    so a shot that went blood-then-undress was judged as whichever came first,
    and the other family was invisible to the rest of the pipeline. Narrowing
    also saves almost nothing: this prompt is ~4k tokens and runs a few dozen
    times, while the sweep's ~0.7k-token prompt runs a thousand.

    `genders` narrows only the BODY-STATE tags, which is a different thing: when
    every subject is male, showing nineteen female body tags invites the model
    to invent a woman. Family residuals and shared tags stay available.
    """
    return get_prompt("identify_prompt", n=n, dialogue=dialogue_text or "(no dialogue)",
                      taxonomy=taxonomy_prompt(genders=genders))


# ---------------------------------------------------------------------------
# Values passed between the two passes
# ---------------------------------------------------------------------------

@dataclass
class SweepWindow:
    """One flagged sample (or batch) straight out of the sweep."""
    start: float
    end: float
    triggers: list[str]
    rendering: str
    subjects: dict
    confidence: float


@dataclass
class CandidateRange:
    """A merged, padded range to classify.

    Carries the UNION of every trigger that produced it. v1 carried a single
    category and lost co-occurring content; a range that was blood in one second
    and undress in the next must be classified as both.
    """
    start: float
    end: float
    triggers: list[str] = field(default_factory=list)
    rendering: str = Rendering.LIVE_ACTION.value
    subjects: dict = field(default_factory=dict)
    confidence: float = 0.5


@dataclass
class SweepStats:
    """Sweep bookkeeping: what was checked, what could not be looked at."""
    frames_total: int = 0
    frames_done: int = 0
    batches_queried: int = 0
    batches_failed: int = 0
    candidates: int = 0


# ---------------------------------------------------------------------------
# Parsing the sweep's answer
# ---------------------------------------------------------------------------

def _clean_triggers(raw) -> list[str]:
    """Normalise the model's trigger list. Anything unrecognised becomes "doubt",
    which still sends the range to classify — an invented trigger name must not
    silently mean 'nothing here'."""
    from nuclearcutter.taxonomy import SWEEP_TRIGGERS
    known = {name for name, _ in SWEEP_TRIGGERS}
    if raw is None:
        return []
    if isinstance(raw, str):
        raw = [raw]
    out: list[str] = []
    for item in raw:
        if not isinstance(item, str):
            continue
        name = item.strip().lower().replace(" ", "_").replace("-", "_")
        if name in known and name not in out:
            out.append(name)
        elif name and name not in out:
            out.append("doubt")
    return out


def _subjects_of(raw) -> dict:
    out = {"adult_female": 0, "adult_male": 0, "minors": Minors.NO.value}
    if not isinstance(raw, dict):
        return out
    for key in ("adult_female", "adult_male"):
        try:
            out[key] = max(0, int(raw.get(key) or 0))
        except (TypeError, ValueError):
            out[key] = 0
    m = str(raw.get("minors") or "").strip().lower()
    out["minors"] = m if m in {x.value for x in Minors} else Minors.NO.value
    return out


def _rendering_of(raw) -> str:
    val = str(raw or "").strip().lower()
    return val if val in {r.value for r in Rendering} else Rendering.LIVE_ACTION.value


def _merge_subjects(a: dict, b: dict) -> dict:
    order = {Minors.NO.value: 0, Minors.UNCERTAIN.value: 1, Minors.YES.value: 2}
    return {
        "adult_female": max(a.get("adult_female", 0), b.get("adult_female", 0)),
        "adult_male": max(a.get("adult_male", 0), b.get("adult_male", 0)),
        # Any window that saw a child decides for the whole merged range.
        "minors": (a.get("minors") if order.get(a.get("minors"), 0)
                   >= order.get(b.get("minors"), 0) else b.get("minors")),
    }


def _arange(start: float, stop: float, step: float) -> list[float]:
    out, t = [], start
    while t < stop:
        out.append(t)
        t += step
    return out


def _flagged_timestamps(batch_ts: list[float], flagged_frames) -> list[float]:
    """Map the model's frame indices to sample timestamps.

    Returns [] when the model did not localise (caller keeps the whole batch
    window — never LESS coverage than before). Non-integer, out-of-range or
    repeated indices are dropped so a malformed answer cannot produce bogus
    windows.
    """
    if not isinstance(flagged_frames, (list, tuple)):
        return []
    out, seen = [], set()
    for idx in flagged_frames:
        if isinstance(idx, bool) or not isinstance(idx, (int, float)):
            continue
        i = int(idx)
        if i != idx or not (0 <= i < len(batch_ts)):
            continue
        ts = batch_ts[i]
        if ts not in seen:
            seen.add(ts)
            out.append(ts)
    return out


# ---------------------------------------------------------------------------
# The detector
# ---------------------------------------------------------------------------

class VisualScanDetector:
    """Full-film trigger sweep + one classify pass per candidate range."""

    def __init__(self, client: LLMClient, scale: str | None = None,
                 sweep_resolution: str | None = None,
                 identify_scale: str | None = None,
                 sweep_concurrency: int | None = None):
        self.client = client
        self.scale_height = scale_height_for(scale)
        self.sweep_max_pixels = SWEEP_QUALITY_PIXELS.get(
            (sweep_resolution or "full").strip().lower())
        # Classify gets its own resolution (defaults to the run's `scale`) so a
        # fast thumbnail sweep no longer drags classification down with it.
        self.identify_max_pixels = vision_max_pixels_for_scale(identify_scale or scale)
        if sweep_concurrency is None:
            sweep_concurrency = DEFAULT_SWEEP_CONCURRENCY
        self.sweep_concurrency = max(1, min(4, int(sweep_concurrency)))

    # -- pass 1: sweep ------------------------------------------------------
    def sweep(self, video_path: Path,
              sample_interval: float = DEFAULT_SWEEP_INTERVAL_SECONDS,
              on_flagged_window=None, on_progress=None,
              resume_from: int = 0, existing_windows: list | None = None,
              stop_event=None) -> tuple[list[CandidateRange], SweepStats]:
        """Sample the whole film; return merged, padded ranges worth classifying.

        `on_flagged_window(start, end, triggers, confidence)` fires per raw
        flagged sample/batch (live timeline markers). `on_progress(done, total)`
        fires after every batch, in ABSOLUTE frame counts so resumes line up.

        Resume: `resume_from` = samples already swept in a prior run (skipped —
        this is the dominant cost of a scan), `existing_windows` = raw windows
        already found, so merging still sees the whole film.

        `stop_event` is checked between batches; a stopped sweep reports partial
        progress and never 100%.
        """
        duration = probe_duration(video_path)
        windows: list[SweepWindow] = list(existing_windows or [])
        stats = SweepStats()
        tmp_dir = Path(tempfile.mkdtemp(prefix="nccut_sweep_"))
        stopped = False
        frame_granularity = False

        try:
            timestamps = _arange(0.0, duration - 0.5, sample_interval)
            stats.frames_total = len(timestamps)
            stats.frames_done = min(max(resume_from, 0), len(timestamps))
            batch_iter = self._iter_prefetched_batches(
                video_path, timestamps, max(resume_from, 0), tmp_dir, stop_event)
            try:
                # Slide up to `sweep_concurrency` VLM calls at a time: a batch
                # costs ~8s of fixed server work (prompt + vision preprocess)
                # against ~1s of decode, so serial calls idle the server's
                # ability to read a second prompt beside a running decode.
                # Batches are SUBMITTED and CONSUMED in chronological order —
                # window merging and the flagged callbacks depend on it — only
                # the execution overlaps. concurrency=1 keeps one worker and
                # behaves exactly like the old serial loop.
                _skip = Future()            # completed stand-in for a batch
                _skip.set_result(None)      # whose frames all failed extraction
                window: "OrderedDict[int, tuple]" = OrderedDict()
                seq = 0
                exhausted = False
                with ThreadPoolExecutor(max_workers=self.sweep_concurrency,
                                        thread_name_prefix="sweep_query") as query_pool:

                    def _fill():
                        nonlocal seq, exhausted
                        while not exhausted and len(window) < self.sweep_concurrency:
                            nxt = next(batch_iter, None)
                            if nxt is None:
                                exhausted = True
                                break
                            idx, bts, fps_ = nxt
                            if not fps_:
                                window[seq] = (idx, bts, _skip, False)
                            else:
                                stats.batches_queried += 1
                                window[seq] = (idx, bts,
                                               query_pool.submit(
                                                   self._query_and_clean, fps_), True)
                            seq += 1

                    while True:
                        if stop_event is not None and stop_event.is_set():
                            # Nothing new is submitted; the in-flight queries
                            # still land (their threads and temp frames must
                            # finish before tmp_dir goes) and progress stays
                            # partial, like a stopped serial sweep.
                            for _, _, fut, ran in window.values():
                                if ran:
                                    try:
                                        fut.result()
                                    except Exception:  # noqa: BLE001
                                        pass
                            stopped = True
                            break

                        _fill()
                        if not window:
                            break

                        i, batch_ts, fut, ran = window.popitem(last=False)[1]
                        if ran:
                            result = fut.result()
                            if result is None:
                                # NOT verified clean — count it. This is the
                                # difference between "nothing found" and
                                # "couldn't look", and the caller decides how
                                # loud to be.
                                stats.batches_failed += 1
                                logging.warning(
                                    "sweep: batch at %.1f-%.1fs FAILED after %d attempts "
                                    "(%d so far) — these frames are NOT verified clean.",
                                    batch_ts[0], batch_ts[-1], MAX_VLM_RETRIES,
                                    stats.batches_failed)
                                # A batch we could not read is not evidence of
                                # cleanliness; hand it to classify anyway. v1
                                # only warned, which meant a dead stretch of
                                # the server quietly produced a clean-looking
                                # scan.
                                windows.append(SweepWindow(
                                    start=batch_ts[0], end=batch_ts[-1],
                                    triggers=["doubt"],
                                    rendering=Rendering.LIVE_ACTION.value,
                                    subjects=_subjects_of(None), confidence=0.0))
                                continue

                            triggers = _clean_triggers(result.get("triggers"))
                            if triggers:
                                confidence = _as_float(result.get("confidence"), 0.5)
                                rendering = _rendering_of(result.get("rendering"))
                                subjects = _subjects_of(result.get("subjects"))
                                flagged_ts = _flagged_timestamps(
                                    batch_ts, result.get("frames"))
                                if flagged_ts:
                                    frame_granularity = True
                                    for ts in flagged_ts:
                                        w = SweepWindow(ts, ts, list(triggers), rendering,
                                                        subjects, confidence)
                                        windows.append(w)
                                        if on_flagged_window:
                                            on_flagged_window(ts, ts, list(triggers),
                                                              confidence)
                                else:
                                    w = SweepWindow(batch_ts[0], batch_ts[-1],
                                                    list(triggers), rendering, subjects,
                                                    confidence)
                                    windows.append(w)
                                    if on_flagged_window:
                                        on_flagged_window(batch_ts[0], batch_ts[-1],
                                                          list(triggers), confidence)

                        done = i + len(batch_ts)
                        stats.frames_done = done
                        if on_progress:
                            on_progress(done, len(timestamps))
                        if done % 100 == 0 or done >= len(timestamps):
                            print(f"[sweep] {done} / {len(timestamps)} frames",
                                  file=sys.stderr, flush=True)
            finally:
                # Close the generator so its prefetch thread finishes (and any
                # in-flight ffmpeg exits) before tmp_dir is removed below.
                batch_iter.close()
        finally:
            shutil.rmtree(tmp_dir, ignore_errors=True)

        if on_progress and not stopped:
            on_progress(len(timestamps), len(timestamps))

        if not stopped and stats.batches_failed and stats.batches_failed >= max(
                1, stats.batches_queried // 10):
            print(f"[sweep] WARNING: {stats.batches_failed}/{stats.batches_queried} "
                  f"batches failed their VLM query. They were sent to the classify "
                  f"pass as 'doubt' rather than trusted as clean, so the scan may "
                  f"contain extra segments that turn out to be nothing.",
                  file=sys.stderr, flush=True)

        merge_gap = _merge_gap_for_interval(sample_interval)
        padding = (_frame_padding_for_interval(sample_interval) if frame_granularity
                   else _padding_for_interval(sample_interval))
        ranges = merge_windows(windows, duration, padding=padding, merge_gap=merge_gap)
        stats.candidates = len(ranges)
        return ranges, stats

    def _iter_prefetched_batches(self, video_path, timestamps, start, tmp_dir, stop_event):
        """Yield (absolute_index, batch_ts, frame_paths) with extraction overlapping the VLM query.

        One ffmpeg decode pass per chunk of SWEEP_CHUNK_BATCHES batches (seek
        once, decode forward), with a one-thread prefetcher extracting the NEXT
        chunk while the caller's VLM query chews the current one (the generator
        is suspended at `yield` during the query, so the prefetch runs exactly
        during model time).

        The generator never stops itself — the CALLER checks stop_event and
        breaks, so a stopped sweep still reports partial progress. Closing the
        generator joins the prefetcher, letting an in-flight ffmpeg finish
        before the caller removes tmp_dir. Frames the chunk pass did not produce
        (ffmpeg fell short, corrupt range, past EOF) fall back to per-frame
        extraction.
        """
        chunk_frames = SWEEP_CHUNK_BATCHES * SWEEP_FRAMES_PER_CALL

        def _extract_chunk(ci: int) -> dict:
            chunk_ts = timestamps[ci:ci + chunk_frames]
            try:
                return extract_frames_chunk(
                    video_path, chunk_ts, tmp_dir,
                    scale_height=self.scale_height, prefix=f"c{ci:06d}_%05d.jpg")
            except Exception as exc:  # noqa: BLE001
                logging.warning("sweep: chunk extraction at %.1fs failed, falling back "
                                "to per-frame: %s", chunk_ts[0], exc)
                return {}

        with ThreadPoolExecutor(max_workers=1, thread_name_prefix="sweep_prefetch") as pool:
            fut = (None if start >= len(timestamps)
                   else pool.submit(_extract_chunk, start))
            ci = start
            while fut is not None:
                got = fut.result()
                ci_next = ci + chunk_frames
                fut = (pool.submit(_extract_chunk, ci_next)
                       if ci_next < len(timestamps)
                       and not (stop_event is not None and stop_event.is_set())
                       else None)
                chunk_ts = timestamps[ci:ci_next]
                for bi in range(0, len(chunk_ts), SWEEP_FRAMES_PER_CALL):
                    batch_ts = chunk_ts[bi:bi + SWEEP_FRAMES_PER_CALL]
                    frame_paths = []
                    for ts in batch_ts:
                        try:
                            p = got.get(ts)
                            if p is None or not p.exists() or p.stat().st_size == 0:
                                p = extract_frame_at(video_path, ts,
                                                     scale_height=self.scale_height)
                            frame_paths.append(p)
                        except Exception as exc:  # noqa: BLE001
                            logging.warning("sweep: frame extraction failed at %.1fs: %s",
                                            ts, exc)
                    yield ci + bi, batch_ts, frame_paths
                ci = ci_next

    def _query_sweep_batch(self, frame_paths: list[Path]) -> dict | None:
        """Ask the VLM whether a batch contains anything worth a closer look."""
        prompt = build_sweep_prompt(len(frame_paths))
        for attempt in range(1, MAX_VLM_RETRIES + 1):
            try:
                if self.sweep_max_pixels:
                    return self.client.vision_query_json(prompt, frame_paths,
                                                         max_pixels=self.sweep_max_pixels)
                return self.client.vision_query_json(prompt, frame_paths)
            except Exception as exc:  # noqa: BLE001
                logging.warning("sweep VLM query failed (attempt %d/%d): %s",
                                attempt, MAX_VLM_RETRIES, exc)
                if attempt == MAX_VLM_RETRIES:
                    return None
        return None

    def _query_and_clean(self, frame_paths: list[Path]) -> dict | None:
        """_query_sweep_batch plus temp-frame cleanup — the worker the query
        window runs. Safe to call from any thread (LLMClient opens its own
        connection per request)."""
        try:
            return self._query_sweep_batch(frame_paths)
        finally:
            for p in frame_paths:
                p.unlink(missing_ok=True)

    # -- pass 2: identify ---------------------------------------------------
    def identify(self, video_path: Path, candidate: CandidateRange,
                 dialogue_text: str = "",
                 invented_tags: list[str] | None = None
                 ) -> tuple[VisualDetection, bool]:
        """Classify one merged range into tag ids + the on-screen description.

        Returns `(detection, classified_ok)`. `classified_ok` is False only when
        the look itself failed (no frames, or every query errored), so the caller
        can tell "the server is dead" apart from "the model looked and had
        nothing to say" and fail loudly on the first instead of printing a
        film-length wall of unclassified markers for the second.

        Never returns a detection with an empty tag list: a range the sweep
        flagged that this pass cannot explain becomes `safety.unclassified`,
        which the chooser shows and the render honours. v1's equivalent was an
        invisible `level=low` placeholder, which censored silently and could not
        be reviewed as a fact.
        """
        n = _identify_frame_count(candidate.end - candidate.start)
        tmp_dir = Path(tempfile.mkdtemp(prefix="nccut_identify_"))
        try:
            fps = n / max(candidate.end - candidate.start, 0.1)
            frame_paths = extract_frames_in_range(
                video_path, candidate.start, candidate.end, fps, tmp_dir,
                scale_height=self.scale_height)
            frame_paths = frame_paths[:n]
            if not frame_paths:
                return _unclassified(candidate, reason="frame extraction failed"), False

            genders = _genders_hint(candidate.subjects)
            prompt = build_identify_prompt(len(frame_paths), dialogue_text, genders)

            result = None
            for attempt in range(1, MAX_VLM_RETRIES + 1):
                try:
                    result = self.client.vision_query_json(prompt, frame_paths,
                                                           max_pixels=self.identify_max_pixels)
                    break
                except Exception as exc:  # noqa: BLE001
                    logging.warning("identify query failed for [%.1f, %.1f] "
                                    "(attempt %d/%d): %s", candidate.start,
                                    candidate.end, attempt, MAX_VLM_RETRIES, exc)
            if result is None:
                return _unclassified(candidate, reason="classify query failed"), False

            raw_tags = result.get("tags")
            if isinstance(raw_tags, str):
                raw_tags = [raw_tags]
            unknown: list[str] = []
            tags = normalize_tags(raw_tags, unknown)
            if unknown:
                if invented_tags is not None:
                    invented_tags.extend(unknown)
                logging.warning("identify: model returned tag ids outside the vocabulary "
                                "at [%.1f, %.1f]: %s", candidate.start, candidate.end,
                                unknown)
            uncertain = [t for t in normalize_tags(result.get("uncertain_tags") or [], None)
                         if t in tags]

            if not tags:
                # The model says there is nothing here. It is not the only
                # witness: the sweep already said otherwise, and this tool's
                # failure mode is showing something, not hiding something.
                return _unclassified(candidate, reason="classify returned no tags",
                                     description=str(result.get("description") or ""),
                                     confidence=_as_float(result.get("confidence"),
                                                          candidate.confidence)), True

            rendering = _rendering_of(result.get("rendering"))
            minors = _minors_of(result.get("minors"), candidate.subjects)
            det = VisualDetection(
                start=candidate.start, end=candidate.end,
                description=str(result.get("description") or ""),
                confidence=_as_float(result.get("confidence"), candidate.confidence),
                tags=tags, rendering=rendering, minors=minors,
                uncertain_tags=uncertain, triggers=list(candidate.triggers))
            enforce_minor_protection(det)
            return det, True
        finally:
            shutil.rmtree(tmp_dir, ignore_errors=True)


# ---------------------------------------------------------------------------
# Helpers shared by the passes (module-level so they are unit-testable)
# ---------------------------------------------------------------------------

def _as_float(val, default: float) -> float:
    try:
        f = float(val)
    except (TypeError, ValueError):
        return default
    return max(0.0, min(1.0, f))


def _minors_of(raw, subjects: dict) -> str:
    """Reconcile the classify answer with whatever the sweep already saw.

    The sweep saw the whole batch at low resolution; classify sees the segment.
    Taking the WORST of the two means a sweep that spotted a child is never
    overruled by a classify pass that did not look for one.
    """
    val = str(raw or "").strip().lower()
    own = val if val in {m.value for m in Minors} else Minors.NO.value
    saw = str((subjects or {}).get("minors") or Minors.NO.value)
    order = {Minors.NO.value: 0, Minors.UNCERTAIN.value: 1, Minors.YES.value: 2}
    return own if order.get(own, 0) >= order.get(saw, 0) else saw


def _genders_hint(subjects: dict | None) -> tuple[str, ...] | None:
    """Body-state tags worth showing, from the sweep's coarse subject count.

    None (show everything) whenever there is any doubt, including a minor
    present: pruning is an accuracy nicety, and pruning a category the scene
    needed is an un-censored scene.
    """
    s = subjects or {}
    if s.get("minors") in (Minors.UNCERTAIN.value, Minors.YES.value):
        return None
    female = int(s.get("adult_female") or 0)
    male = int(s.get("adult_male") or 0)
    if female and not male:
        return ("female",)
    if male and not female:
        return ("male",)
    return None


def _unclassified(candidate: CandidateRange, reason: str, description: str = "",
                  confidence: float | None = None) -> VisualDetection:
    """Keep a flagged range visible and censorable when classify could not name it."""
    det = VisualDetection(
        start=candidate.start, end=candidate.end,
        description=description,
        confidence=candidate.confidence if confidence is None else confidence,
        tags=["safety.unclassified"],
        rendering=candidate.rendering,
        minors=str((candidate.subjects or {}).get("minors") or Minors.NO.value),
        uncertain_tags=["safety.unclassified"],
        triggers=list(candidate.triggers))
    det.description = description or ""
    logging.info("identify: [%.1f, %.1f] kept as safety.unclassified (%s)",
                 candidate.start, candidate.end, reason)
    enforce_minor_protection(det)
    return det


def _identify_frame_count(duration: float) -> int:
    return max(IDENTIFY_FRAMES_MIN,
               min(IDENTIFY_FRAMES_MAX, int(round(duration / IDENTIFY_SECONDS_PER_FRAME))))


def merge_windows(windows: list[SweepWindow], duration: float, padding: float,
                  merge_gap: float) -> list[CandidateRange]:
    """Merge adjacent flagged windows into padded candidate ranges.

    `padding` is derived from the sweep interval so it reaches up to, but never
    past, the next verified-clean sample. `merge_gap` is how close two windows
    must be to merge. Merging UNIONs triggers and takes the WORST answer for
    subjects — v1 kept the first window's category and silently dropped every
    other family that appeared in the same merged range.
    """
    if not windows:
        return []
    ordered = sorted(windows, key=lambda w: w.start)

    ranges: list[CandidateRange] = []
    start = ordered[0].start
    end = ordered[0].end
    triggers = list(ordered[0].triggers)
    renderings = Counter([ordered[0].rendering])
    subjects = dict(ordered[0].subjects)
    peak_confidence = ordered[0].confidence

    def _flush():
        ranges.append(CandidateRange(
            start=max(0.0, start - padding),
            end=min(duration, end + padding),
            triggers=list(triggers),
            # Most common answer wins; a range is nearly always one rendering.
            rendering=renderings.most_common(1)[0][0],
            subjects=dict(subjects),
            confidence=peak_confidence))

    for w in ordered[1:]:
        if w.start - end <= merge_gap:
            end = max(end, w.end)
            peak_confidence = max(peak_confidence, w.confidence)
            renderings[w.rendering] += 1
            subjects = _merge_subjects(subjects, w.subjects)
            for t in w.triggers:
                if t not in triggers:
                    triggers.append(t)
        else:
            _flush()
            start, end = w.start, w.end
            triggers = list(w.triggers)
            renderings = Counter([w.rendering])
            subjects = dict(w.subjects)
            peak_confidence = w.confidence
    _flush()
    return ranges
