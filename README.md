# NuclearCutter

Self-hosted, open-source content censoring for your local movie collection.
Detects nudity, gore, violence, and foul language and produces a permanently
modified copy of the file — no live-playback plugin, no dependency on Plex or
any particular player. Runs on your own hardware, targeting Apple Silicon.

Not a live filter like VidAngel/ClearPlay/Skipit — those apply filters at
playback time. NuclearCutter edits the file itself, once, and you keep the result.

**Status:** early / actively developed. See `docs/SPEC.md` for the full design
rationale behind every decision below — read that first if you're contributing.

## Quick start — launch the web GUI

Requirements: macOS on Apple Silicon (recommended), Python 3.10+, and
`ffmpeg`/`ffprobe` on your PATH. Everything else is handled for you — the
first run creates a local virtual environment and installs dependencies
automatically (no `pip install`, no `source activate`).

```bash
git clone https://github.com/SpencerGraffunder/NuclearCutter.git
cd NuclearCutter

python3 nuclearcutter.py            # starts the web GUI server
```

Then open a browser:

- **On this machine:** http://localhost:8000
- **From any other device on your network:** `http://<this-machine-ip>:8000`
  (find the IP with `ipconfig getifaddr en0` — it's printed in the server
  banner too)

The server binds `0.0.0.0` and has **no login** — anyone on your network who
can reach the port can use it. Everything is controlled from the browser: pick
your movie, choose a model backend, start the scan, watch it progress, then
render the cleaned copy. That's the whole workflow.

`python3 nuclearcutter.py` creates `.venv/` and installs dependencies on first
run, then starts the server. You can also run `./nuclearcutter.py` (it's
executable). To stop the server, press Ctrl-C in that terminal.

Headless `scan`/`render` commands still exist for scripting — see
[Usage](#usage) below.

## What it does

Two passes:

1. **Scan** — analyzes the whole file (video + audio), writes a JSON file
   recording every detected instance of nudity, gore, violence, and foul
   language, with timestamps and (for visual detections) an AI-written
   description of the scene. This pass takes no action on the file itself —
   it's a neutral record of what's in the movie. This is the slow part;
   realistically hours, and can run for a day or more on a long film depending
   on your hardware. That's expected and fine.

2. **Render** — reads the scan JSON plus your preferences (what to do about
   each category) and produces `<movie>_cleaned.<ext>` in the same folder.
   This is much faster than the scan.

You can re-render the same scan with different preferences without rescanning
— scan data and your censorship choices are stored separately on purpose (see
`docs/SPEC.md` §2, §6). In the web GUI, the **Status** section shows a
timeline of every detection that the *currently selected* render settings
would catch, so you can tune the levels and immediately see what would be
missed or over-corrected before you hit Render.

## Actions available per category

Every category has **two independent corrections**: a *visual* action (what to
do to the video) and an *audio* action (what to do to the audio). They are set
per category in the web GUI's **Render** section (the corrections matrix).

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

Your per-category **level** setting in the Render section is that threshold:
detections at or below it get corrected; milder ones are left alone. E.g.
`nudity_level = "low"` blurs only the most extreme nudity, while
`nudity_level = "exhigh"` blurs essentially anything that isn't fully modest
(it makes the film play like a documentary — which is fine, since every
blur/black segment shows a clean text summary of the scene).

The level *scale itself* is standardized (built in, not per-user), so the same
scan JSON produces consistent results for everyone. The **full definitions of
each level** (i.e. exactly what gets caught at each setting) are part of the
sweep/confirm prompts in `nuclearcutter/prompts.json`, which the web GUI's
**VLM Prompts** panel lets you view and edit.

**`blur`** keeps the scene intact and less disruptive than replacing the
footage entirely, while still obscuring the flagged content. **`black`** shows
nothing but the clean summary text. Audio can be set independently per
category, so e.g. nudity can be `visual=blur` + `audio=none`, or violence can
be `visual=black` + `audio=mute_scene`.

Blur intensity is tunable with the **Blur amount** field in the GUI:
`1.0` is the standard intense blur, `2.0` is twice as extreme (bigger radius +
more passes), `0.5` is lighter. **Mute padding** (default 0.5s) pads each
flagged word on both sides so word onset/offset audio doesn't leak through —
whisper's word timestamps can be tight. **Blur padding** (default 0s) extends
each blur/black segment by extra seconds on both sides.

## Setup

### Requirements

- macOS on Apple Silicon (M-series) recommended — `mlx-whisper` and `mlx-vlm`
  are MLX-accelerated and Apple-Silicon-specific. The rest of the pipeline is
  plain Python/ffmpeg and should run elsewhere, but isn't the primary target.
- Python 3.10+
- `ffmpeg` and `ffprobe` on your PATH
- An inference backend. The default is **mlx-vlm**, which NuclearCutter starts
  for you automatically (a local MLX vision model served over an
  OpenAI-compatible `/v1` API on port 1234). You can also launch a local
  **llama.cpp** `llama-server`, or point at any **already-running**
  OpenAI-compatible server (LM Studio, Ollama, a manually started mlx-vlm
  server) — see below.

### Install

No install step needed — `python3 nuclearcutter.py` sets up a local virtual
environment (`.venv/`) and installs the package + dependencies automatically
on first run. After that, every invocation is just:

```bash
python3 nuclearcutter.py            # web GUI
python3 nuclearcutter.py scan MOVIE.mkv      # headless scan
python3 nuclearcutter.py render MOVIE.mkv    # headless render
```

(Equivalently: `./nuclearcutter.py ...`, or activate the venv once and use
`nuclearcutter ...` — `source .venv/bin/activate`.)

### Configuration

Settings are saved automatically on the server to `settings.json` (next to
the repo) whenever you change them, and reloaded when the server starts — so
your movie location, model, and render preferences survive restarts. The path
is shown in the GUI header and in the server banner. The Scan section covers:

- **Source file location** — the movie to analyze.
- **Model backend** — one of:
  - *launch local mlx-vlm* — NuclearCutter spawns `mlx_vlm.server` on port 1234
    with the **local model path** you give it (default: the LM Studio MLX
    folder, e.g. `~/.lmstudio/models/lmstudio-community/Qwen3.5-9B-MLX-4bit`).
  - *launch local llama.cpp* — spawns `llama-server` (from Homebrew's
    `llama.cpp`) with a `.gguf` file, optionally a `--mmproj` vision
    projector for image input.
  - *use existing model server* — type the server's IP / base URL and hit
    **Scan for models**; the available model ids are fetched from its
    `/v1/models` and shown in the VLM model dropdown.
- **Whisper model** — for transcription (default
  `mlx-community/whisper-small-mlx`).
- **Scale frames before VLM** — `360p`/`480p`/`720p`/`1080p`; frames are
  downscaled before being sent to the vision model. Lower is much faster, and
  480p is the recommended default for scene-level detection.
- **Scan interval** — seconds between sampled frames (default 2).
- **Start / Stop / Clear progress** — stopping saves progress to a status file
  next to the movie, so hitting Start again resumes from where it stopped.
  Clear progress wipes the saved state to force a fresh scan.
- **Test / benchmark VLM** — runs the real sweep + confirm prompts against the
  selected model on 12 frames from the movie and reports speed and accuracy
  (a **CANCEL BENCH** button stops it mid-run).

### The mlx-vlm backend (default)

The default backend is **mlx-vlm**, a fast MLX vision-language model server
that runs entirely on your Mac's GPU. It's installed automatically into the
venv by `nuclearcutter.py` on first run (there's no Homebrew formula for
mlx-vlm, so it's pip-installed). When you start a scan or benchmark,
NuclearCutter:

1. checks it's installed,
2. starts its own server (`python -m mlx_vlm.server --port 1234 --model <path>`,
   with 4-bit KV-cache quantization) pointed at your model path,
3. waits for it to come up, runs the scan against it, and keeps it running
   until you stop the NuclearCutter server.

The default model path is the same Qwen MLX 4-bit model that the project was
previously using through LM Studio:
`/Users/<you>/.lmstudio/models/lmstudio-community/Qwen3.5-9B-MLX-4bit`. If
your model lives elsewhere, change the **local model path** in the GUI. Images
sent to the model are downscaled to the selected scale before upload, which is
the single biggest speed lever in the whole pipeline.

If you'd rather run your own server (LM Studio, Ollama, or a manual
`python -m mlx_vlm.server ...`), select **use existing model server** and point
it at your base URL — NuclearCutter will use it without starting or stopping
anything.

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

The **scan interval** controls sampling density (seconds between samples;
default 2). Smaller catches shorter scenes but makes more VLM calls; larger
is faster but can miss brief flashes. Because this is one VLM sweep covering
all three visual categories, it replaces what used to be NudeNet + a VLM
confirm pass + two separate opt-in sweeps — and it never depends on a
fast-but-blind classifier deciding what's worth looking at.

### Rendering preserves the source codec

The renderer keeps the source codec family — an x265/HEVC source is re-encoded
with `libx265` (not forced to H.264), an H.264 source uses `libx264`, AV1 stays
AV1, etc. This avoids the large output-size jump you'd get from transcoding a
compact x265 file into H.264. (Blur/mute segments are still re-encoded; the
point is the *codec* is preserved, not that output is bit-identical.)

## Usage

### Web GUI (recommended)

```bash
python3 nuclearcutter.py
# open http://localhost:8000 (or http://<this-machine-ip>:8000 from any device)
```

Workflow in the browser:

1. **Scan section** — set the source file, backend + model, scale, interval,
   then **Start scan**. The Status section below shows the timeline, progress
   bars for each step (transcribe / scan / verify / render), ETA, frame
   counter, and model speed stats. Stop saves progress; Start resumes.
2. **Render section** — once the scan is done, pick the per-category levels
   and corrections, the output file name (default `<movie>_cleaned`), blur
   amount, mute/blur padding, then **Start render**. The timeline marks update
   live as you change levels so you can see exactly what will be corrected.
   (A stopped render can't be resumed — the half-done output is discarded.)
3. The cleaned file appears next to the original.

### Headless CLI

The same operations are available from the terminal for scripting:

```bash
# Scan a movie (slow — hours to a day+ depending on length/hardware)
python3 nuclearcutter.py scan "/path/to/Movie.mkv" [--scale 480p] [--sweep-interval 2]

# Render with your preferred actions per category
python3 nuclearcutter.py render "/path/to/Movie.mkv" [--nudity-level high] [--blur-strength 1.5]
```

This produces `/path/to/Movie_cleaned.mkv`. The default backend is mlx-vlm
with the default model path; for a standalone server pass
`--backend standalone --base-url ... --vlm-model ... --text-model ...`. See
`python3 nuclearcutter.py scan --help` / `render --help` for every flag.

### Status section (the dashboard)

The web GUI's **Status** section is the live dashboard, replacing the old
terminal TUI:

- **System stats** — RAM and CPU used by NuclearCutter itself (not system
  totals), GPU active residency, and CPU temperature. GPU/temps are shown when
  passwordless `sudo powermetrics` is available; otherwise they read n/a and
  the hint tells you the one-line sudoers rule to enable them.
- **Timeline** — a bar for the whole movie with a color-coded mark per
  detection *that the currently selected render levels/actions would catch*.
  Changing the Render section levels re-filters the marks immediately, so you
  can see whether a looser setting would miss things or a stricter one would
  correct too much.
- **Step progress bars** — one each for transcription, scan, verify, and
  render.
- **Model status** — pp speed (t/s), generation speed (t/s), tokens per
  prompt, and speed per frame, measured from the live requests.
- **ETA and frames counter**.

## Why a scan takes hours

A scan samples a frame every 2 seconds across the whole film and sends each
batch of 4 frames to the vision model for review. That's a lot of model calls —
a ~2-hour film is roughly **900 model calls** (3600 sampled frames ÷ 4), and
each one takes several seconds on a MacBook's GPU. Whisper transcription and
the per-scene confirm pass add more on top. So hours are normal (roughly
proportional to film length × your GPU speed). To trade thoroughness for speed,
raise the scan interval in the GUI: `5` halves the model calls (but can miss
scenes shorter than ~5s); `10` is faster still. Lower intervals (the 2s
default) catch short flashes at the cost of more calls.

## Examples

### Example 1 — Full scan + render in the GUI

Open the GUI, set the source file to your movie, leave the default backend
(mlx-vlm, auto-started) and scale (480p), hit **Start scan**. When the scan
finishes, set your render preferences (or keep defaults: blur nudity/gore/
violence, mute foul-language phrases) and hit **Start render**. Result:
`Movie_cleaned.mkv` next to the original, which is left untouched.

### Example 2 — Standalone backend (LM Studio / Ollama)

Start your own server first:

```bash
# LM Studio serving on port 1234 (default) — open LM Studio and load the model
# OR Ollama (different port + model names):
ollama pull qwen3.5:7b
ollama serve
```

In the GUI: select **use existing model server**, type the base URL
(e.g. `http://localhost:11434/v1`), hit **Scan for models**, and pick the VLM
model from the dropdown. Headless equivalent:

```bash
python3 nuclearcutter.py scan "/Users/you/Movies/Movie.mkv" \
  --backend standalone \
  --base-url http://localhost:11434/v1 \
  --vlm-model qwen3.5:7b \
  --text-model qwen3.5:7b \
  --whisper-model mlx-community/whisper-large-v3-turbo
```

### Example 3 — Re-render an old scan with different preferences (no rescan)

You already scanned once; now you want a different censorship policy without
re-analyzing the film. In the GUI, the render reads the same scan JSON
(`Movie.nuclearcutter.json`) — just change the levels/corrections and render
again. Headless equivalent:

```bash
python3 nuclearcutter.py render "/path/to/Movie.mkv" \
  --scan "/path/to/Movie.nuclearcutter.json" \
  --output "/path/to/Movie_lite_cut.mkv" \
  --nudity-level low \
  --foul-language-audio mute_word
```

### Example 4 — Benchmark the model before committing to a scan

In the GUI, click **Test / benchmark VLM** (optionally after setting the
backend and scale). It builds a 12-frame collection from the movie — including
frames from known flagged windows if a scan already exists — and runs the real
sweep + confirm prompts, reporting per-batch time, tokens, pp/gen speed, and
whether each batch was flagged correctly.

## How fingerprinting works

Movie files often come from different rips/encodes of the same underlying
film, so filenames and file sizes aren't reliable ways to match a shared scan
file to your local copy. NuclearCutter instead computes a perceptual hash (pHash)
of frames sampled at fixed **percentages** of total runtime (not fixed
timestamps), plus overall duration. This is resilient to different
containers, bitrates, and frame rates, as long as it's fundamentally the same
cut of the film. See `docs/SPEC.md` §5 for the full matching/verification
flow.

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
- No review/edit UI yet for inspecting flagged scenes before rendering — the
  web GUI is the interface; you can hand-edit the scan JSON directly if you
  want to correct or remove a detection before rendering.
- The GUI binds 0.0.0.0 with no login. On a trusted home network that's the
  point (any device can open it); if you'd rather restrict it, run
  `python3 nuclearcutter.py serve --host 127.0.0.1` to limit it to this machine.

## Contributing

Read `docs/SPEC.md` first — it's the design doc that captures not just what
was built but *why*, including several explicit decisions (e.g. why intense blur
is used instead of a skip card, why scan data and preferences are kept separate,
why fingerprinting uses percentage-based pHash sampling) that you'll want to
understand before changing behavior.
