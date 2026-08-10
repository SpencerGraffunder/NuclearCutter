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

Every category has **two independent corrections**: a *visual* action (what to
do to the video) and an *audio* action (what to do to the audio). They default
per category and can be overridden per-run with CLI flags or in `config.toml`.

**Visual actions:**

| Visual action | Effect |
|---|---|
| `none` | leave the video untouched |
| `blur` | intense box blur over the flagged range + a short, clean AI summary overlaid |
| `black` | replace the flagged range entirely with a black screen + the clean summary |

**Audio actions:**

For **visual categories** (nudity/gore/violence) there's no per-sound
recognition, so the only options are:

| Audio action | Effect |
|---|---|
| `none` | leave the audio untouched |
| `mute_scene` | silence the whole flagged scene |

For **foul language** (which has word-level timestamps):

| Audio action | Effect |
|---|---|
| `none` | leave the audio untouched |
| `mute_word` | silence just the flagged word |
| `mute_phrase` | silence the whole utterance/phrase |
| `replace_word` | (upcoming) AI voice replacement of the word — falls back to mute for now |
| `replace_phrase` | (upcoming) AI voice replacement of the phrase — falls back to mute for now |

**Categories and defaults:**

| Category | Default visual | Default audio | Default level | Notes |
|---|---|---|---|---|
| Nudity | `blur` | `none` | `med` | bare/partly covered private parts, underwear/swimwear/lingerie in an intimate context, sex scenes. (Formerly two categories — nudity + intimate scenes — now merged into one.) |
| Gore | `blur` | `none` | `med` | visible blood, open wounds, surgery showing blood/incisions, corpses with wounds, mutilation |
| Violence | `blur` | `none` | `med` | characters deliberately hurting each other (fighting, punching, attacks, murder, torture) even if no blood is shown |
| Foul language | `none` | `mute_phrase` | `med` | profanity; audio-only by default |

### Severity levels (shareable scans)

Every detection is classified into a **fixed severity level** — `low` / `med` /
`high` / `exhigh` — by the model during the scan. This level is recorded in the
scan JSON, so **scan files are shareable**: the scan says *how bad* a scene is,
and each person decides their own cutoff.

The level is the amount of **censorship**:
- `low` = **low censorship** — only the worst content is corrected
- `med` = medium
- `high` = high
- `exhigh` = **max censorship** — basically everything gets corrected

Your per-category `*_level` setting is that threshold: content at or above it
gets corrected, below is left alone. E.g. `nudity_level = "low"` blurs only the
most extreme nudity, while `nudity_level = "exhigh"` blurs essentially anything
that isn't fully modest (it makes the film play like a documentary — which is
fine, since every blur/black segment shows a clean text summary of the scene).

The level *scale itself* is standardized (built in, not per-user), so the same
scan JSON produces consistent results for everyone. The **DETECTION LEVELS**
section of `config.toml` shows the **full prompt** used for each level of each
category, so you can see exactly what will be caught at each setting. If you
really want a fully custom definition for a category, set its `*_prompt` in
`config.toml` — that replaces the built-in scale entirely.

**`blur`** keeps the scene intact and less disruptive than replacing the footage
entirely, while still obscuring the flagged content. **`black`** shows nothing
but the clean summary text. Audio can be set independently per category, so
e.g. nudity can be `visual=blur` + `audio=none`, or violence can be
`visual=black` + `audio=mute_scene`.

Blur intensity is tunable with `--blur-strength N` (or `blur_strength` in
`config.toml`): `1.0` is the standard intense blur, `2.0` is twice as extreme
(bigger radius + more passes), `0.5` is lighter.

Foul-language muting pads each flagged word by `--mute-padding` seconds (default 0.5) on both sides, so the word's onset/offset audio doesn't leak through — whisper's word timestamps can be tight.

## Setup

### Requirements

- macOS on Apple Silicon (M-series) recommended — `mlx-whisper` and `mlx-vlm`
  are MLX-accelerated and Apple-Silicon-specific. The rest of the pipeline is
  plain Python/ffmpeg and should run elsewhere, but isn't the primary target.
- Python 3.10+
- `ffmpeg` and `ffprobe` on your PATH
- An inference backend. The default backend is **mlx-vlm**, which NuclearCutter
  starts for you automatically (a local MLX vision model served over an
  OpenAI-compatible `/v1` API on port 1234). You can also point it at any
  OpenAI-compatible local server you run yourself (`standalone` backend — LM
  Studio, Ollama, or a manually started mlx-vlm server) — see "Configuration"
  below.

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

### Configuration (config.toml)

Settings live in a single `config.toml` file in the project root
(`Documents/NuclearCutter/config.toml` on this machine) — plain text, editable
in any editor (TextEdit / Notepad / VS Code). A fully commented template is
created for you; every key is optional and has a sane default. After you set it
once, the only thing you pass on the command line is the movie path:

```bash
nuclearcutter scan  "/path/to/Movie.mkv"
nuclearcutter render "/path/to/Movie.mkv"
```

The default config file looks like this (all commented-out = defaults):

```toml
# ---- Inference backend -------------------------------------------------
# model_backend = "mlx-vlm"   # "mlx-vlm" (auto-start its own server) | "standalone"
# model_path = "/Users/<you>/.lmstudio/models/lmstudio-community/Qwen3.5-9B-MLX-4bit"
# base_url = "http://localhost:1234/v1"

# ---- Models ------------------------------------------------------------
# vlm_model = "Qwen3.5-9B-MLX-4bit"
# text_model = "Qwen3.5-9B-MLX-4bit"
# whisper_model = "mlx-community/whisper-small-mlx"

# ---- Scan tuning -------------------------------------------------------
# sweep_interval = 5.0
# vision_timeout = 8000
# timestamps_dir = ""

# ---- Per-category severity threshold (correct at/above this) ------------
# nudity_level = "med"      # "low" | "med" | "high" | "exhigh"
# gore_level = "med"
# violence_level = "med"
# foul_language_level = "med"
# (The level SCALE itself is fixed/standardized so scans are shareable — see
#  the DETECTION LEVELS section of config.toml. Only these thresholds vary.)

# ---- Optional custom prompts (empty = built-in fixed level scale) -------
# nudity_prompt = ""
# gore_prompt = ""
# violence_prompt = ""
# foul_language_prompt = ""

# ---- Render corrections (per category: visual + audio) -----------------
# nudity_visual = "blur"            # "none" | "blur" | "black"
# nudity_audio = "none"             # "none" | "mute_scene"
# gore_visual = "blur"
# gore_audio = "none"
# violence_visual = "blur"
# violence_audio = "none"
# foul_language_visual = "none"
# foul_language_audio = "mute_phrase"
# blur_strength = 1.0
# mute_padding = 0.5
# font = ""
```

Use a different config file with `nuclearcutter --config /path/to/config.toml ...`.
Every setting can still be overridden per-run with a CLI flag — see `--help`.

### The mlx-vlm backend (default)

The default backend is **mlx-vlm**, a fast MLX vision-language model server that
runs entirely on your Mac's GPU. NuclearCutter:

1. checks it's installed (`pip install mlx-vlm`; note there is no Homebrew
   formula for mlx-vlm, so it's pip-installed into your venv),
2. starts its own server (`python -m mlx_vlm.server --port 1234 --model <path>`,
   with 4-bit KV-cache quantization) pointed at your `model_path`,
3. waits for it to come up, runs the scan against it, and shuts it down when
   the scan finishes.

The default `model_path` is the same Qwen MLX 4-bit model that the project was
previously using through LM Studio:
`/Users/<you>/.lmstudio/models/lmstudio-community/Qwen3.5-9B-MLX-4bit`. If
your model lives elsewhere, set `model_path` in `config.toml`. Images sent to
the model are downscaled (~480px) before upload, which is the single biggest
speed lever in the whole pipeline.

If you'd rather run your own server (LM Studio, Ollama, or a manual
`python -m mlx_vlm.server ...`), set `model_backend = "standalone"` in
`config.toml` and point `base_url` at it. NuclearCutter will use it without
starting or stopping anything.

### Full-film VLM sweep (the only visual detector)

Visual detection is a single **full-film VLM sweep** — there is no separate
NudeNet classifier anymore. NudeNet was removed because it can miss a real
nude scene entirely (it scored zero on every frame of a full nude scene in
The Martian that a vision model immediately flagged at 0.95 confidence).

The sweep samples frames across the whole film, sends them to the vision
model in small batches, and asks one question per batch: *"does this batch
contain ANY flagged content?"* — covering nudity, gore, and violence in a
single pass. Flagged batch windows are merged into ranges with generous
before/after padding, so a scene is never clipped and nothing is silently
dropped. Each range is then confirmed, described, **and classified into a
severity level** (low/med/high/exhigh) using the fixed, standardized scale.
The level is stored in the scan JSON, which is what makes scans shareable.

`--sweep-interval` controls sampling density (seconds between samples;
default 2). Smaller catches shorter scenes but makes more VLM calls; larger
is faster but can miss brief flashes:

```bash
nuclearcutter scan ... --sweep-interval 2   # default — densest, most thorough
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

The two commands, at a glance (settings come from `config.toml` — see above):

```bash
# Scan a movie (slow — hours to a day+ depending on length/hardware)
nuclearcutter scan "/path/to/Movie.mkv"

# Render with your preferred actions per category
nuclearcutter render "/path/to/Movie.mkv"
```

This produces `/path/to/Movie_cleaned.mkv`.

## Live dashboard (integrated into scan/render)

Both `scan` and `render` show a **live TUI dashboard** while they run — no
separate command needed. It displays a film timeline with markers for detected
visual/language cut locations, a current-position cursor, a time estimate, and
live CPU/RAM (GPU/temps when passwordless `sudo powermetrics` is available;
otherwise shown as n/a). The dashboard appears automatically when you run the
command in a terminal; pass `--no-tui` to get plain text output instead (also
auto-disabled when stdout is piped/redirected).

The dashboard also streams a live status JSON to a temp file (phase, position,
and every detection as it's found). You can watch a scan started elsewhere with:

```bash
nuclearcutter tui --status /path/to/Movie.nuclearcutter.status.json
# or attach to an older scan's log (no status file):
nuclearcutter tui --log /path/to/scan.log
```

With no `--status`/`--log`, `nuclearcutter tui` auto-detects the most recent
status file or scan log.

## Examples

All examples below are complete, runnable commands — with `config.toml` set up
once, none of them need model flags. CLI flags shown override the config for
that single run.

### Example 1 — Full scan + render (mlx-vlm backend, defaults)

The most common case: a local movie, config.toml set, default mlx-vlm backend
auto-starting on port 1234.

```bash
# 1. Scan (slow — run it and walk away)
nuclearcutter scan "/Volumes/Media/Movies/The.Martian.2015.1080p.mkv"

# 2. Render the censored copy (fast)
nuclearcutter render "/Volumes/Media/Movies/The.Martian.2015.1080p.mkv"
```

Result: `/Volumes/Media/Movies/The.Martian.2015.1080p_cleaned.mkv` (original
left untouched).

### Example 2 — Standalone backend (LM Studio / Ollama)

Run your own server instead of letting NuclearCutter start one. Set
`model_backend = "standalone"` in `config.toml` (and `base_url` to your
server), then start it:

```bash
# LM Studio serving on port 1234 (default) — just open LM Studio and load the model
# OR Ollama (different port + model names):
ollama pull qwen3.5:7b
ollama serve
```

```bash
nuclearcutter scan "/Users/you/Movies/The.Martian.2015.1080p.mkv" \
  --base-url http://localhost:11434/v1 \
  --vlm-model qwen3.5:7b \
  --text-model qwen3.5:7b \
  --whisper-model mlx-community/whisper-large-v3-turbo
```

(Here the flags override `config.toml` just for this run — with
`model_backend = "standalone"` set in the config, the `--base-url`/model flags
are the only ones needed each time.)

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
  --timestamps-dir ~/Code/nuclearcutter/timestamps
```

(Or set `timestamps_dir` in `config.toml` and just pass the movie path.)

### Example 6 — Write the scan JSON to a specific location

By default the scan lands next to your movie as `Movie.nuclearcutter.json`.
On a read-only or shared drive you can send it somewhere writable instead.

```bash
nuclearcutter scan "/Volumes/Media/Movies/The.Martian.2015.1080p.mkv" \
  --output "$HOME/scans/The.Martian.2015.1080p.json"
```

### Using the shared timestamps repo

If someone has already scanned the same film (even a different rip/encode —
see fingerprinting below), you can skip the expensive scan entirely:

```bash
nuclearcutter scan "/Volumes/Media/Movies/The.Martian.2015.1080p.mkv" \
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

Instead of passing visual/audio flags every time, you can save a preferences
file and reuse it:

```python
from pathlib import Path
from nuclearcutter.schema import AudioAction, Preferences, VisualAction
prefs = Preferences(
    nudity_visual=VisualAction.BLUR,
    nudity_audio=AudioAction.NONE,
    gore_visual=VisualAction.BLUR,
    gore_audio=AudioAction.NONE,
    violence_visual=VisualAction.BLUR,
    violence_audio=AudioAction.NONE,
    foul_language_visual=VisualAction.NONE,
    foul_language_audio=AudioAction.MUTE_PHRASE,
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
