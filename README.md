# NuclearCutter

Self-hosted, open-source content censoring for your local movie collection.
Detects nudity, gore, violence, and foul language and produces a permanently
modified copy of the file — no live-playback plugin, no dependency on Plex or
any particular player. NuclearCutter is a **client** that does all the ffmpeg
work locally and talks to your own, already-running model servers over the
network: an OpenAI-compatible LLM/VLM server for visual + text understanding,
and a whisper.cpp server for transcription. It never launches a model server
of its own.

Not a live filter like VidAngel/ClearPlay/Skipit — those apply filters at
playback time. NuclearCutter edits the file itself, once, and you keep the result.

**Status:** early / actively developed. See `docs/SPEC.md` for the full design
rationale behind every decision below — read that first if you're contributing.

## Quick start — launch the web GUI

Requirements: Python 3.10+, `ffmpeg`/`ffprobe` on your PATH, and two model
servers you already run yourself (see [Setup](#setup)): an
OpenAI-compatible LLM/VLM server (e.g. llama.cpp `llama-server`) and a
whisper.cpp server. NuclearCutter connects to them over the network and does
not start or stop them. Everything else is handled for you — the first run
creates a local virtual environment and installs dependencies automatically
(no `pip install`, no `source activate`).

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
your movie, point it at your model + whisper servers, start the scan, watch it
progress, then render the cleaned copy. That's the whole workflow.

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

- Python 3.10+ (any OS the ffmpeg toolchain and your model servers run on)
- `ffmpeg` and `ffprobe` on your PATH
- Two model servers you run yourself (NuclearCutter only ever *talks* to
  them — it never launches or stops one):
  - an **OpenAI-compatible LLM/VLM server** serving a vision-capable model
    over a `/v1` API (e.g. llama.cpp `llama-server` with a multimodal GGUF +
    `--mmproj`, LM Studio, Ollama, …). This does the visual sweep, the scene
    descriptions, and the text-level foul-language re-checks.
  - a **whisper.cpp server** for transcription (foul language). The GUI can
    hot-swap which whisper model it has loaded.

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
your movie location, server addresses, models, and render preferences survive
restarts. The path is shown in the GUI header and in the server banner. The
Scan section covers:

- **Source file location** — the movie to analyze.
- **Model server IP / URL** — the OpenAI-compatible `/v1` server for the
  LLM/VLM. Hit **Scan for models** to fetch its model ids from `/v1/models`
  into the dropdowns.
- **VLM model / Text model** — the vision model for the sweep + descriptions,
  and (optionally different) the text model for the foul-language re-checks.
- **Whisper server IP** + **Whisper model folder** — the whisper.cpp server
  for transcription and the folder that holds its `.bin` models.
- **Whisper model** — a dropdown of the `.bin` files in that folder. Hit
  **Load** to hot-swap the server's loaded model to the selected one (this
  interrupts the shared whisper server for a few seconds while it loads).
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

### Remote model servers (no local spawning)

NuclearCutter is a pure client: it never starts a model server. Point the GUI
(or the CLI flags) at servers that are already running and it will use them.
The header badge shows, live, whether the LLM/VLM server and the whisper
server are reachable. If a server is down, the scan refuses to start with a
clear message rather than hanging. Images sent to the vision model are
downscaled to the selected scale before upload — the single biggest speed
lever in the whole pipeline.

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

1. **Scan section** — set the source file, point at your LLM/VLM + whisper
   servers and pick the models, set scale + interval, then **Start scan**. The
   Status section below shows the timeline, progress
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
# Requires the LLM/VLM server running (default http://127.0.0.1:8080/v1) and
# the whisper.cpp server (default http://127.0.0.1:8081).
python3 nuclearcutter.py scan "/path/to/Movie.mkv" \
  --base-url http://192.168.4.164:8080/v1 \
  --vlm-model "unsloth/Qwen3.8-27B-GGUF" \
  [--scale 480p] [--sweep-interval 2] \
  [--whisper-base-url http://127.0.0.1:8081] [--whisper-model ggml-base.en] \
  [--whisper-models-dir /path/to/whisper/models]

# Render with your preferred actions per category
python3 nuclearcutter.py render "/path/to/Movie.mkv" [--nudity-level high] [--blur-strength 1.5]
```

This produces `/path/to/Movie_cleaned.mkv`. There is no `--backend` flag any
more — NuclearCutter always uses the remote servers you point it at; see
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
each one takes several seconds on a typical GPU. Whisper transcription and
the per-scene confirm pass add more on top. So hours are normal (roughly
proportional to film length × your model server's speed). To trade
thoroughness for speed, raise the scan interval in the GUI: `5` halves the
model calls (but can miss scenes shorter than ~5s); `10` is faster still.
Lower intervals (the 2s default) catch short flashes at the cost of more calls.

## Examples

### Example 1 — Full scan + render in the GUI

Open the GUI, set the source file to your movie, point the **Model server**
and **Whisper server** rows at your running servers (hit **Scan for models** /
**Scan** to fill the dropdowns), keep scale at 480p, and hit **Start scan**.
When the scan finishes, set your render preferences (or keep defaults: blur
nudity/gore/violence, mute foul-language phrases) and hit **Start render**.
Result: `Movie_cleaned.mkv` next to the original, which is left untouched.

### Example 2 — Point it at llama.cpp (LLM/VLM) + whisper.cpp

Start your own servers first (or use ones that are already running):

```bash
# llama.cpp serving a vision-capable model on port 8080
./llama-server --host 0.0.0.0 --port 8080 -m Qwen3-VL-8B-Instruct-Q4_K_M.gguf \
  --mmproj mmproj-model-f16.gguf --jinja

# whisper.cpp server on port 8081
./server -m models/ggml-base.en.bin
```

In the GUI: type the model server URL (e.g. `http://192.168.4.164:8080/v1`),
hit **Scan for models**, pick the VLM model. Type the whisper server IP
(e.g. `http://127.0.0.1:8081`), set its **model folder**, hit **Scan**, and
pick a whisper model — hit **Load** to hot-swap the server to that model.
Headless equivalent:

```bash
python3 nuclearcutter.py scan "/path/to/Movie.mkv" \
  --base-url http://192.168.4.164:8080/v1 \
  --vlm-model "Qwen3-VL-8B-Instruct-Q4_K_M" \
  --whisper-base-url http://127.0.0.1:8081 \
  --whisper-models-dir /path/to/whisper/models \
  --whisper-model ggml-base.en
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
model and scale). It builds a 12-frame collection from the movie — including
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
