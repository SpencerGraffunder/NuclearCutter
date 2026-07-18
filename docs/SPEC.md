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
| `nudity` | `blur`, `skip` |
| `intimate_scenes` (sex scenes, not necessarily nudity — e.g. implied/clothed) | `blur`, `skip` |
| `foul_language` | `mute` |

Actions:

- **`blur`** — apply an intense box blur to the video for the flagged range.
  Whether audio is also muted during a blur is a **separate configurable
  sub-option per category** (`blur_mute_audio: true/false`), not implied by blur
  itself.
- **`skip`** — replace the video for the flagged range with a black screen
  displaying a short, clean, VLM-generated text summary of what happens during
  that segment (see §4 for how this is generated and timed). Audio is muted
  during a skip. This is NOT a hard cut — runtime is preserved, which matters for
  sync with subtitles, chapter markers, etc. It's a "read this instead of
  watching this" substitution, not an edit that shortens the film.
- **`mute`** — silence the audio only, video untouched. Used for foul language.
  Default granularity is **the offending word only** (tightest mute window
  possible), but this is configurable to mute the whole sentence/utterance
  instead.

## 4. Detection pipeline

### 4.1 Nudity / intimate scenes (hybrid two-stage)

- **Stage A (cheap classifier):** Sample video frames at a fixed interval (not
  every single frame — configurable, default something like every 0.5–1s) and
  run a lightweight local NSFW/nudity image classifier (e.g. NudeNet or similar
  ONNX model) to flag *candidate* time ranges cheaply. This stage exists purely
  to avoid burning expensive VLM calls on the ~99% of a movie with nothing to
  flag.
- **Stage B (VLM confirmation + description):** For every candidate range Stage A
  flags, send sampled frames from that range to a vision-language model to (a)
  confirm or reject the classifier's flag, (b) classify it as `nudity` vs
  `intimate_scenes`, and (c) generate the human-readable scene description text
  used later for `skip` cards.
- The VLM description must weave together **both** what is visually happening
  and what is being said/plot-relevant during the scene (dialogue content from
  the audio pipeline, see §4.2) — not just a visual description. E.g. not just
  "two characters embrace in bed" but incorporating relevant plot-carrying
  dialogue that happens during the scene, since the whole point of the `skip`
  card is that the viewer doesn't miss story content.
- Model access is via an **OpenAI-compatible chat completions API**
  (`/v1/chat/completions` with image content blocks), so the user can point
  NuclearCutter at a local Ollama or LM Studio server by base URL. No hard dependency
  on a specific inference backend. See README for suggested models.

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

### 4.3 Skip-card timing

The black-screen text card duration for `skip` actions scales with how long the
description takes to read, roughly 200–250 words/minute reading pace. A short
visual-only beat ("character walks past camera, briefly nude") might render for
~2 seconds; a scene with meaningful dialogue might need ~10 seconds. This is a
formula based on word count of the generated description, not a fixed duration.

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

## 8. Non-goals / explicit exclusions

- Not a live-playback filter. No Plex/Jellyfin plugin, no client-side
  integration. Operates only on files, before they ever reach a media server.
- Not trying to preserve original runtime for `skip` (that's the point of the
  text-card mechanic — better than a hard cut which risks losing plot content or
  breaking sync with subtitle files).
- Not scoped to build a full review/edit web UI in v1.
