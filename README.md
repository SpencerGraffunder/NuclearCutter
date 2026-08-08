# NuclearCutter

Self-hosted, open-source content censoring for your local movie collection.
Detects nudity, intimate scenes, and foul language and produces a permanently
modified copy of the file — no live-playback plugin, no dependency on Plex or
any particular player. Runs on your own hardware, targeting Apple Silicon.

Not a live filter like VidAngel/ClearPlay/Skipit — those apply filters at
playback time. NuclearCutter edits the file itself, once, and you keep the result.

**Status:** early / actively developed. See `docs/SPEC.md` for the full design
rationale behind every decision below — read that first if you're contributing.

## What it does

Two passes:

1. **`nuclearcutter scan MOVIE.mkv`** — analyzes the whole file (video + audio),
   writes a JSON file recording every detected instance of nudity, intimate
   scenes, and foul language, with timestamps and (for visual detections) an
   AI-written description of the scene. This pass takes no action on the file
   itself — it's a neutral record of what's in the movie. This is the slow
   part; realistically hours, and can run for a day or more on a long film
   depending on your hardware. That's expected and fine.

2. **`nuclearcutter render MOVIE.mkv`** — reads the scan JSON plus your personal
   preferences (what to do about each category) and produces
   `MOVIE_cleaned.mkv` in the same folder. This is much faster than the scan.

You can re-render the same scan with different preferences without rescanning
— scan data and your censorship choices are stored separately on purpose (see
`docs/SPEC.md` §2, §6).

## Actions available per category

| Category | Actions | Notes |
|---|---|---|
| Nudity | `blur`, `none` | blur = intense box blur over the video; audio mute during blur is a separate toggle |
| Intimate scenes | `blur`, `none` | same as above; distinct category from nudity (e.g. a clothed sex scene) |
| Gore / violence | `blur`, `none` | graphic gore (blood, wounds, mutilation) and graphic violence (brutal fighting, murder, torture) |
| Foul language | `mute`, `none` | mutes audio only; defaults to muting just the flagged word, configurable to the whole sentence/utterance |

**`blur`** applies an intense box blur to the flagged video range and overlays a short, clean, AI-generated summary so you keep story context. This keeps the scene intact and less disruptive than replacing the footage entirely, while still obscuring the flagged content. Audio can be muted separately for each blur category using `--nudity-blur-mute-audio`, `--intimate-scenes-blur-mute-audio`, and `--gore-violence-blur-mute-audio`.

## Setup

### Requirements

- macOS on Apple Silicon (M-series) recommended — `mlx-whisper` is
  MLX-accelerated and Apple-Silicon-specific. The rest of the pipeline is
  plain Python/ffmpeg and should run elsewhere, but isn't the primary target.
- Python 3.10+
- `ffmpeg` and `ffprobe` on your PATH
- A local OpenAI-compatible inference server — **Ollama** or **LM Studio**,
  running both a vision-capable model and a text model (see below). NuclearCutter
  talks to it over the standard `/v1/chat/completions` API, so any
  OpenAI-compatible local server works, not just these two.

### Install

```bash
git clone <this-repo>
cd nuclearcutter
pip install -e .
```

Activate the local virtual environment before running the CLI, or invoke the
installed binary directly:

```bash
source .venv/bin/activate
nuclearcutter scan "/path/to/Movie.mkv"
```

Or:

```bash
./.venv/bin/nuclearcutter scan "/path/to/Movie.mkv"
```

### Suggested models

NuclearCutter needs two local models available through your OpenAI-compatible
server. For an 8GB VRAM Apple Silicon setup, use a single recommended vision+
text model pair:

- `qwen3.5:4b-mlx` — the one recommended model for this workflow.
  - vision-language capable for scene confirmation and descriptions
  - 64K context is sufficient for the scan and render prompts used here
  - the smallest practical MLX model for stable local performance on 8GB

Pull it in Ollama:

```bash
ollama pull qwen3.5:4b-mlx
ollama serve
```

Specify the model used during the scan step with `--vlm-model` and
`--text-model`. For example:

```bash
nuclearcutter scan "/path/to/Movie.mkv" \
  --base-url http://localhost:1234/v1 \
  --vlm-model qwen3.5:4b-mlx \
  --text-model qwen3.5:4b-mlx \
  --whisper-model mlx-community/whisper-small-mlx
```

NuclearCutter has **no built-in model defaults** — you must always pass
`--vlm-model`, `--text-model` (and `--whisper-model` for the scan) explicitly,
so it only ever uses the models you name. Point it at your server with
`--base-url` (e.g. LM Studio's `http://localhost:1234/v1` or Ollama's
`http://localhost:11434/v1`).

### Full-film VLM sweep (the only visual detector)

Visual detection is a single **full-film VLM sweep** — there is no separate
NudeNet classifier anymore. NudeNet was removed because it can miss a real
nude scene entirely (it scored zero on every frame of a full nude scene in
The Martian that a vision model immediately flagged at 0.95 confidence).

The sweep samples frames across the whole film, sends them to the vision
model in small batches, and asks one question per batch: *"does this batch
contain ANY flagged content?"* — covering nudity, intimate scenes, **and**
gore/violence in a single pass. Flagged batch windows are merged into ranges
with generous before/after padding, so a scene is never clipped and nothing
is silently dropped. Each range is then confirmed and described (the nudity
confirm prompt deliberately treats underwear/swimwear/lingerie/suggestive
clothing as flagged content).

`--sweep-interval` controls sampling density (seconds between samples;
default 5). Smaller catches shorter scenes but makes more VLM calls; larger
is faster but can miss brief flashes:

```bash
nuclearcutter scan ... --sweep-interval 5   # default — densest, most thorough
nuclearcutter scan ... --sweep-interval 10  # faster, still catches ~10s+ scenes
```

Because this is one VLM sweep covering all three visual categories, it
replaces what used to be NudeNet + a VLM confirm pass + two separate opt-in
sweeps — and it never depends on a fast-but-blind classifier deciding what's
worth looking at.

### Rendering preserves the source codec

The renderer keeps the source codec family — an x265/HEVC source is re-encoded
with `libx265` (not forced to H.264), an H.264 source uses `libx264`, AV1 stays
AV1, etc. This avoids the large output-size jump you'd get from transcoding a
compact x265 file into H.264. (Blur/mute segments are still re-encoded; the
point is the *codec* is preserved, not that output is bit-identical.)

## Usage

The two commands, at a glance (models are required — see Examples below):

```bash
# Scan a movie (slow — hours to a day+ depending on length/hardware)
nuclearcutter scan "/path/to/Movie.mkv" \
  --base-url http://localhost:1234/v1 \
  --vlm-model qwen3.5 \
  --text-model qwen3.5 \
  --whisper-model mlx-community/whisper-small-mlx

# Render with your preferred actions per category
nuclearcutter render "/path/to/Movie.mkv" \
  --scan "/path/to/Movie.nuclearcutter.json" \
  --nudity blur \
  --nudity-blur-mute-audio \
  --intimate-scenes blur \
  --intimate-scenes-blur-mute-audio \
  --foul-language mute \
  --mute-scope word
```

This produces `/path/to/Movie_cleaned.mkv`.

## Examples

All examples below are complete, runnable commands. Swap in your own video
path, model names, and base URL. **There are no built-in model defaults** —
every scan must name `--vlm-model`, `--text-model`, and `--whisper-model`
explicitly.

### Example 1 — Full scan + render (LM Studio, Apple Silicon)

The most common case: a local movie, LM Studio serving on the default port,
using a single `qwen3.5` model for both vision and text.

```bash
# 1. Scan (slow — run it and walk away)
nuclearcutter scan "/Volumes/Media/Movies/The.Martian.2015.1080p.mkv" \
  --base-url http://localhost:1234/v1 \
  --vlm-model qwen/qwen3.5-9b \
  --text-model qwen/qwen3.5-9b \
  --whisper-model mlx-community/whisper-small-mlx

# 2. Render the censored copy (fast)
nuclearcutter render "/Volumes/Media/Movies/The.Martian.2015.1080p.mkv" \
  --scan "/Volumes/Media/Movies/The.Martian.2015.1080p.nuclearcutter.json" \
  --nudity blur \
  --nudity-blur-mute-audio \
  --intimate-scenes blur \
  --intimate-scenes-blur-mute-audio \
  --foul-language mute \
  --mute-scope word
```

Result: `/Volumes/Media/Movies/The.Martian.2015.1080p_cleaned.mkv` (original
left untouched).

### Example 2 — Ollama instead of LM Studio

Same workflow, different server. Ollama's default port is `11434`, and its
model names use the `name:tag` form.

```bash
# Make sure the model is pulled first
ollama pull qwen3.5:7b
ollama serve

nuclearcutter scan "/Users/you/Movies/The.Martian.2015.1080p.mkv" \
  --base-url http://localhost:11434/v1 \
  --vlm-model qwen3.5:7b \
  --text-model qwen3.5:7b \
  --whisper-model mlx-community/whisper-large-v3-turbo
```

### Example 3 — Re-render an old scan with different preferences (no rescan)

You already scanned once; now you want a different censorship policy without
re-analyzing the film. Just render again from the same JSON.

```bash
# This time: don't mute audio during blur, and mute whole utterances, not just words
nuclearcutter render "/Volumes/Media/Movies/The.Martian.2015.1080p.mkv" \
  --scan "/Volumes/Media/Movies/The.Martian.2015.1080p.nuclearcutter.json" \
  --output "/Volumes/Media/Movies/The.Martian.2015.1080p_lite_cut.mkv" \
  --nudity blur \
  --intimate-scenes blur \
  --foul-language mute \
  --mute-scope utterance
```

### Example 4 — Render from a saved preferences file

Define your policy once in JSON, reuse it for every movie.

```bash
nuclearcutter render "/Volumes/Media/Movies/The.Martian.2015.1080p.mkv" \
  --scan "/Volumes/Media/Movies/The.Martian.2015.1080p.nuclearcutter.json" \
  --prefs ~/.config/nuclearcutter/prefs.json
```

To create that prefs file, see "Saving preferences" below.

### Example 5 — Skip the scan using a shared timestamp file

If someone has already scanned the same film, reuse their work instead of
re-analyzing for hours. NuclearCutter fingerprints your file and, if it
matches a scan in `timestamps/`, spot-checks a few frames with the VLM and
reuses the result.

```bash
nuclearcutter scan "/Volumes/Media/Movies/The.Martian.2015.1080p.mkv" \
  --base-url http://localhost:1234/v1 \
  --vlm-model qwen/qwen3.5-9b \
  --text-model qwen/qwen3.5-9b \
  --whisper-model mlx-community/whisper-small-mlx \
  --timestamps-dir ~/Code/nuclearcutter/timestamps
```

Note: `--timestamps-dir` still needs the models for the VLM spot-check.

### Example 6 — Write the scan JSON to a specific location

By default the scan lands next to your movie as `Movie.nuclearcutter.json`.
On a read-only or shared drive you can send it somewhere writable instead.

```bash
nuclearcutter scan "/Volumes/Media/Movies/The.Martian.2015.1080p.mkv" \
  --output "$HOME/scans/The.Martian.2015.1080p.json" \
  --base-url http://localhost:1234/v1 \
  --vlm-model qwen/qwen3.5-9b \
  --text-model qwen/qwen3.5-9b \
  --whisper-model mlx-community/whisper-small-mlx
```

### Using the shared timestamps repo

If someone has already scanned the same film (even a different rip/encode —
see fingerprinting below), you can skip the expensive scan entirely:

```bash
nuclearcutter scan "/Volumes/Media/Movies/The.Martian.2015.1080p.mkv" \
  --base-url http://localhost:1234/v1 \
  --vlm-model qwen/qwen3.5-9b \
  --text-model qwen/qwen3.5-9b \
  --whisper-model mlx-community/whisper-small-mlx \
  --timestamps-dir ./timestamps
```

This checks `./timestamps/*.json` for a fingerprint match against your local
file, spot-checks a few flagged scenes with the VLM to make sure it's really
the same content (not just a similar-length file), and if confirmed, copies
that data over instead of doing a full rescan.

Scans you produce are meant to be shared back — copy your `.nuclearcutter.json`
output into `timestamps/` and open a PR. **Scan files never contain your
personal action preferences** (blur vs mute) — only what's in the
film and when. That's what makes one scan file useful to everyone regardless
of what they each want censored.

### Saving preferences

Instead of passing `--nudity`/`--intimate-scenes`/`--foul-language` flags
every time, you can save a preferences file and reuse it:

```python
from nuclearcutter.schema import Preferences, Action
prefs = Preferences(
    nudity_action=Action.BLUR,
    intimate_scenes_action=Action.BLUR,
    foul_language_action=Action.MUTE,
    foul_language_mute_scope="utterance",
)
prefs.save(Path("my_prefs.json"))
```

```bash
nuclearcutter render "/Volumes/Media/Movies/The.Martian.2015.1080p.mkv" \
  --scan "/Volumes/Media/Movies/The.Martian.2015.1080p.nuclearcutter.json" \
  --prefs ~/.config/nuclearcutter/prefs.json
```

Render needs no models — it just applies the actions from your scan JSON and
preferences via ffmpeg.

## How fingerprinting works

Movie files often come from different rips/encodes of the same underlying
film, so filenames and file sizes aren't reliable ways to match a shared scan
file to your local copy. NuclearCutter instead computes a perceptual hash (pHash)
of frames sampled at fixed **percentages** of total runtime (not fixed
timestamps), plus overall duration. This is resilient to different
containers, bitrates, and frame rates, as long as it's fundamentally the same
cut of the film. See `docs/SPEC.md` §5 for the full matching/verification
flow, including the VLM spot-check step that runs before a match is trusted.

## Known limitations

- Untouched segments are currently re-encoded during render rather than
  stream-copied, to keep the segment-concat step reliable across arbitrary
  cut points. This costs some render time but doesn't affect final quality
  (re-encode uses a high-quality CRF). A stream-copy fast path for untouched
  segments (splitting at keyframe boundaries instead of arbitrary detection
  timestamps) is a reasonable future optimization — contributions welcome.
- The default profanity wordlist (`nuclearcutter/detection/data/profanity_wordlist.txt`)
  is a starting point, not exhaustive. It's intentionally broad/loose since
  every match is re-checked in context by an LLM before being flagged — see
  `docs/SPEC.md` §4.2.
- Multi-audio-track / multi-subtitle-track files: current implementation
  operates on the first audio track and looks for a single sidecar subtitle.
  Multi-track handling is a good area for contribution.
- No review/edit UI yet for inspecting flagged scenes before rendering — CLI
  only for now, per `docs/SPEC.md` §7. You can hand-edit the scan JSON
  directly if you want to correct or remove a detection before rendering.

## Contributing

Read `docs/SPEC.md` first — it's the design doc that captures not just what
was built but *why*, including several explicit decisions (e.g. why intense blur
is used instead of a skip card, why scan data and preferences are kept separate,
why fingerprinting uses percentage-based pHash sampling) that you'll want to
understand before changing behavior.
