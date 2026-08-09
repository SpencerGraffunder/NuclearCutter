# NuclearCutter — Project Specification

This document is the source of truth for what NuclearCutter is, why it's built the way
it is, and what decisions have already been made (and why) so future contributors
— human or AI — don't have to re-derive them from scratch.

## 1. Purpose

NuclearCutter is a self-hosted, open-source tool for homelab users to automatically
detect and censor specific content categories in their own movie/TV files, and
produce a permanently modified copy of the file (`Movie_cleaned.mkv` next to
`Movie.mkv`). It is explicitly **not** a live-playback filter (unlike VidAngel,
ClearPlay, Skipit, etc.) — the whole point is a file you can drop into Plex/Jellyfin
and it's just... clean, with no client-side plugin or real-time dependency.

Target user: technically capable homelab operator, comfortable with CLI tools,
running local AI inference (Ollama / LM Studio) on Apple Silicon hardware. Runtime
of "hours to days per movie" is explicitly acceptable — this is a batch job that
runs once per file, not a live system. Output video/audio quality outside of
flagged segments must be unaffected (stream-copy where possible, no unnecessary
re-encoding of untouched material).

## 2. Two-pass architecture

**Pass 1 — Scan.** Analyze the movie, produce a JSON file recording every
detected instance of every content category, with timestamps and (for
nudity/intimate categories) an AI-generated scene description. This pass does
**not** know or care what the user wants done about any of it — it is a neutral
record of "what is in this movie and when."

**Pass 2 — Render.** Read the scan JSON + a user preferences file (what action to
take per category), and produce the final `_cleaned` output file via ffmpeg.

This separation is deliberate: it's what makes the scan JSON shareable in the repo
(see §6) — two users with different censorship preferences can use the exact same
scan data.

## 3. Content categories and actions

Three categories, each independently configurable to one of the available actions:

| Category | Available actions |
|---|---|
| `nudity`, `immodesty` | `blur` |
| `intimate_scenes` (sex scenes, not necessarily nudity — e.g. implied/clothed) | `blur` |
| `gore_violence` (graphic gore/blood/wounds and graphic violence, e.g. brutal fighting, murder, torture) | `blur` |
| `foul_language` | `mute` |

Actions:

- **`blur`** — apply an intense box blur to the video for the flagged range.
  Whether audio is also muted during a blur is a **separate configurable
  sub-option per category** (`blur_mute_audio: true/false`), not implied by blur
  itself. On top of the blur display a short, clean, VLM-generated text summary of what happens during
  that segment (see §4 for how this is generated and timed). This approach is
  chosen because an intense blur obscures the flagged content while preserving
  scene flow and avoiding the abrupt disruption of a skip card. Runtime is
  preserved, which matters for sync with subtitles, chapter markers, etc.
- **`mute`** — silence the audio only, video untouched. Used for foul language.
  Default granularity is **the offending word only** (tightest mute window
  possible), but this is configurable to mute the whole sentence/utterance
  instead.

## 4. Detection pipeline

### 4.1 Nudity / intimate scenes / gore (unified VLM sweep)

Visual detection is a single **full-film VLM sweep** — there is no separate
cheap classifier anymore. A two-stage NudeNet-then-VLM design was tried and
removed: NudeNet (a fast local ONNX classifier) scored **zero** on every frame
of a full nude scene in The Martian that a vision model flagged at 0.95
confidence, which meant the VLM never got a chance to confirm the scene. A
blind fast pass can't be the gatekeeper if it can silently drop real content.

- **Sweep:** Sample frames across the whole film at a configurable interval
  (default 5s — catches any scene that lasts ≥ ~5s). Frames are sent to the
  vision model in small batches (4 per call), and the model is asked one
  single-verdict question per batch: *"does this batch contain ANY flagged
  content?"* covering nudity, intimate scenes, and gore/violence in one pass.
  This single-verdict format (rather than per-frame index JSON) is what proved
  reliable with local reasoning VLMs.
- **Merge + pad:** Flagged batch windows are merged into ranges with generous
  before/after padding (`SWEEP_PADDING_SECONDS`, default 10s) so a scene is
  never clipped. The sweep verdict is authoritative: a range that was flagged
  is never dropped, even if a later pass can't enrich it — the goal is to
  never show bad content, at the cost of occasionally blurring more than the
  minimum.
- **Confirm + describe:** Each merged range is sent to the vision model with
  the per-category confirm prompt to (a) produce the human-readable scene
  description used for `blur` cards and (b) refine the category. The nudity
  confirm prompt deliberately treats underwear/swimwear/lingerie/suggestive
  clothing as flagged content.
- The VLM description must weave together **both** what is visually happening
  and what is being said/plot-relevant during the scene (dialogue content from
  the audio pipeline, see §4.2) — not just a visual description. E.g. not just
  "two characters embrace in bed" but incorporating relevant plot-carrying
  dialogue that happens during the scene, since the whole point of the `blur`
  card is that the viewer doesn't miss story content.
- Model access is via an **OpenAI-compatible chat completions API**
  (`/v1/chat/completions` with image content blocks). Two backends, one API:
  - **`mlx-vlm` (default):** NuclearCutter spawns its own
    `python -m mlx_vlm.server` on `localhost:1234`, pre-loads the configured
    MLX model (a Qwen MLX 4-bit model, defaulting to the same filesystem path
    the project previously used through LM Studio), and serves it in-process.
    Speed knobs: images are downscaled (~480px) before upload — the single
    biggest latency lever (~6x) — plus 4-bit KV-cache quantization and
    continuous batching (already on by default in mlx-vlm). The server is shut
    down when the scan finishes.
  - **`standalone`:** talk to an already-running OpenAI-compatible server (LM
    Studio, Ollama, a manual mlx-vlm server) via `base_url`. No process is
    started or stopped.
  - Important mlx-vlm detail: the server caches models keyed by the exact
    `model` string from each request, and it pre-loads the model under its
    **full filesystem path**. Requests must therefore send the full
    `model_path` as the model id (not the short name), or the server tries to
    fetch the short name as a Hugging Face repo. Both VLM and text checks go
    through the same loaded model.

### 4.2 Foul language

- **Whisper transcription** (via `mlx-whisper` for Apple Silicon acceleration)
  with word-level timestamps, run always.
- **Subtitle cross-check:** if an embedded or sidecar subtitle file (SRT/ASS/etc)
  is present, parse it and cross-check against the Whisper transcript for the
  same time ranges — improves accuracy and catches cases Whisper mishears.
- **Wordlist match:** both the Whisper transcript and subtitle text (when
  present) are checked against a configurable profanity wordlist.
- **LLM context check (always run, not optional):** flagged words/lines are also
  passed to an LLM (same OpenAI-compatible endpoint) for a contextual check —
  catches things a static wordlist misses and reduces false positives (e.g. a
  non-profane homophone, a word used in a non-profane sense). Wordlist match
  alone is never sufficient on its own; the LLM pass always runs on top of it.
- Output: word-level (or utterance-level) timestamps for each flagged instance.
  Muting pads each flagged window by `foul_language_mute_padding` seconds (default
  0.5) on both sides so the word's onset/offset audio doesn't leak through —
  whisper word timestamps can be tight, and an exact window lets the first/last
  phoneme escape the mute.

## 5. Fingerprinting (for matching shared timestamp files to a local file)

Movie files a user has locally may be a different encode/rip/source than the one
a shared timestamp JSON was generated against (different container, bitrate,
subtitle burn-in, cropping, etc), so filename and file size are not reliable
matches.

**Chosen approach:** perceptual hash (pHash) of sampled frames at fixed points
through the film (e.g. by percentage of runtime, so it's resilient to different
container/frame-rate metadata). This is robust across re-encodes because it's
based on actual visual content, not file bytes. Combined with total duration as a
first cheap filter before bothering with frame hashing.

**Verification flow when a candidate JSON match is found:**
1. Compare duration (cheap, fast rejection of obviously-different releases/cuts).
2. Compare perceptual hashes at several sampled points.
3. If both pass some confidence threshold, optionally spot-check a small number
   of flagged timestamp ranges with the VLM to confirm the content actually
   matches at those specific points (catches cases where e.g. a director's cut
   has different scene content despite similar overall runtime/hashes).
4. Only if verification passes do we skip the full rescan and go straight to
   using the existing JSON's detections with the user's chosen actions.

## 6. Shared timestamp repository

Scan JSON files are meant to be shareable and PR-able in the project repo, keyed
by film identity (fingerprint + duration + title/year metadata), **not** by
filename. Critically:

- The JSON stores **raw detections only** — categories, timestamps, VLM
  descriptions, confidence data. It does **not** store which action the
  original generator chose (blur vs skip vs mute) — that lives in a separate,
  local, non-shared user-preferences file. This is what makes one scan JSON
  useful to every user regardless of their personal filtering choices.
- Format: plain JSON (not EDL/M3U/etc) — chosen specifically for readability and
  ease of reviewing/editing in a PR diff.

## 7. Interface

CLI only for the initial version. Two primary commands: `scan` and `render`
(names TBD in implementation), operating on the two-pass model in §2. A review
UI for inspecting/editing flagged scenes before rendering is a valid future
addition but explicitly out of scope for v1.

All persistent settings (backend, models, sweep interval, render preferences)
live in a single **`config.toml`** in the project root — plain text, editable
in any editor, every key optional with a sane default. The only required
argument on the command line is the movie path; per-run CLI flags override the
config. This replaced the earlier "every model must be named on the command
line every time" approach, which was error-prone for a tool meant to be run
daily on a whole library. Concrete examples (see README for the full set):

```bash
# Scan a film (slow; hours on a long feature). Everything is read from config.toml.
nuclearcutter scan "/Volumes/Media/Movies/The.Martian.2015.1080p.mkv"

# Render a censored copy from the scan JSON (fast; no models needed).
nuclearcutter render "/Volumes/Media/Movies/The.Martian.2015.1080p.mkv"
```

Key configuration keys: `model_backend` (`mlx-vlm` auto-start or `standalone`),
`model_path` (filesystem path to the MLX model), `base_url`, `vlm_model`,
`text_model`, `whisper_model`, `sweep_interval`, `vision_timeout`,
`timestamps_dir`, plus per-category render actions
(`nudity`/`intimate_scenes`/`gore_violence`/`foul_language`), the
`nudity_blur_mute_audio`/`intimate_scenes_blur_mute_audio`/
`gore_violence_blur_mute_audio` toggles, and `foul_language_mute_padding`
(extra seconds muted around each flagged word).

### 7.1 Live dashboard

`scan` and `render` render an inline TUI dashboard while they run (film
timeline with detection markers, current-position cursor, ETA, and CPU/RAM).
It's enabled by default on a TTY and disabled by `--no-tui` or when output is
piped. Both commands also stream a live status JSON (`ScanStatus`, see
`nuclearcutter/utils/scan_status.py`); the same dashboard can be attached to a
scan running elsewhere via `nuclearcutter tui --status FILE` (or `--log FILE`
to tail an older scan's log that predates status files).

## 8. Non-goals / explicit exclusions

- Not a live-playback filter. No Plex/Jellyfin plugin, no client-side
  integration. Operates only on files, before they ever reach a media server.
- Not scoped to build a full review/edit web UI in v1.
