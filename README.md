# NuclearCutter

Self-hosted, open-source content censoring for your local movie collection.
Describes what is in every scene of a movie — nudity, violence, blood, sexual
content, language, drug use — and produces a permanently
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

For the **language** family (which has word-level timestamps):

| Audio action | Effect |
|---|---|
| `none` | leave the audio untouched |
| `mute_word` | silence just the flagged word |
| `mute_phrase` | silence the whole utterance/phrase |
| `replace_word` | (upcoming) AI voice replacement of the word — falls back to mute for now |
| `replace_phrase` | (upcoming) AI voice replacement of the phrase — falls back to mute for now |

**Families and defaults:**

Detection describes **what is in the frame**. Rendering decides what to do about
it. Those are separate, and the table below is the *default* half of the second
step — every one of these is a dropdown in the GUI and a flag in the CLI.

| Family | Tags | Default picture | Default audio | What it covers |
|---|---|---|---|---|
| Nudity & modesty | 35 | `blur` | `none` | How much of a body, or what clothing, is visible |
| Sexual situations | 11 | `blur` | `none` | Kissing, undressing, sex — with or without nudity |
| Violence | 26 | `blur` | `none` | One person hurting, or threatening to hurt, another |
| Blood & injury | 21 | `blur` | `none` | Wounds, blood, damaged bodies |
| Self-harm & suicide | 5 | `blur` | `none` | Someone hurting themselves, and how it is depicted |
| Language | 16 | `none` | `mute_phrase` | Swearing, slurs, insults, explicit talk |
| Drugs, drinking, smoking | 4 | `blur` | `none` | Smoking, heavy drinking, drug use |
| Scan was unsure | 3 | `blur` | `none` | Flagged but could not be named — see below |

### What gets censored is a set of tags, not a level

The scan records facts — `nudity.shirtless_male`, `violence.weapon_display`,
`gore.blood_moderate` — and **nothing else about them**. There is no severity
score, no threshold, and no "how bad is this" anywhere in the pipeline. What gets
censored is simply which tag ids you have ticked:

```python
det.censorable(your_selection)   # any overlap → that moment is censored
```

So the workflow is: scan once, then decide. Tick "shirtless men" off and the
beach scenes come back; tick it on later and re-render in minutes with no rescan.
Presets (`Nothing`, `Nudity`, `Sexual`, `Language`, `Graphic`, `Everything`) are
one-tap starting points that only ever *add* to what you had picked — and there is
deliberately no "recommended" one, because deciding what is offensive is the entire
decision this tool hands back to you.

**This is also why scan files are shareable.** A detection is a fact, so two people
who disagree completely about what deserves censoring use the same file. Under the
old severity levels they could not: the scan encoded somebody else's threshold, and
disagreeing with it meant rescanning the film — hours of GPU time — to hold a
different opinion.

Two things are enforced rather than offered, because a safety setting that can be
missed is a safety setting that will be missed:

- **A child in the scene is always censored.** `safety.exposure_with_minor` is the
  one non-negotiable tag; the GUI shows it greyed out and labelled *always on*
  rather than hiding it.
- **"Unsure" is a tag, never silence.** A moment the scan flagged but cannot name
  lands on `safety.unclassified` or a family's `unclear`, and is censorable by
  default. A dropped detection is an uncensored scene, which is the one failure
  mode this tool cannot afford. (Self-harm and substance detection deliberately
  forgo an `unclear` tag — guessing those from a blurry frame deletes plotlines;
  [docs/TAXONOMY.md](docs/TAXONOMY.md) §3.3 explains the trade.)

**`blur`** keeps the scene intact and less disruptive than replacing the footage
entirely, while still obscuring it. **`black`** shows nothing but the clean summary
text. Picture and sound are set independently per family, so nudity can be
`blur`+`none` while violence is `black`+`mute_scene`. When one moment matches two
families asking for different actions, the **stronger** ask wins.

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
    descriptions, and the tagging of spoken language.
  - a **whisper.cpp server** for transcription (the language family). NuclearCutter
    loads the model you pick in the GUI on demand — the server hot-swaps to
    it when a transcription starts.

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
  and (optionally different) the text model used to tag spoken language.
- **Whisper server** — derived from the same server address (port +1); the
  whisper.cpp server for transcription. Its `.bin` model folder is
  auto-detected from the server process — no setting needed.
- **Whisper model** — a dropdown of the `.bin` models the whisper server can
  use (the model folder is auto-detected from the server process). The model
  is loaded on demand: when a scan transcribes, the server hot-swaps to the
  selected model first (a few-second interruption of the shared whisper
  server while it loads). The entry marked `(loaded)` is the one active now.
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

### How a scan works (trigger sweep, then one classifier pass)

Visual detection is two passes over the film, split because they answer different
questions at very different cost. There is no NudeNet-style pre-filter: it scored
**zero** on every frame of a full nude scene that a vision model immediately
flagged at 0.95 confidence, so the VLM never got shown those scenes at all. A blind
fast pass cannot be the gatekeeper if it can silently drop real content.

1. **Sweep** samples frames across the whole film and asks the vision model one
   cheap question per batch: *which of these categories, if any, appear in these
   frames?* — a short list of trigger names, not the 121-tag vocabulary. Coarse
   presence is what low-resolution batches are good at, and keeping the vocabulary
   out of this prompt is what keeps the call cheap.
2. Flagged windows are merged with generous before/after padding so a scene is
   never clipped, and merging takes the **union** of what was seen.
3. **Classify** then re-watches each merged range at higher resolution and returns
   the specific tag ids, a scene description, a confidence, how the content is
   presented (live-action, animation, anime, artwork, archival, fantasy or stage)
   and an age estimate for the people present. Those presentation styles are the
   **Filter** checkboxes in the GUI, all ticked by default — untick `artwork` to
   stop a museum scene being blurred, and nothing else.

**The sweep's verdict is final — classify can add detail but can never un-flag a
range.** That asymmetry is why the two passes stay separate: a cheap pass that can
only *include* and an expensive pass that can only *enrich* means nothing gets
dropped on a bad model day. It is also why `harm` and `substance` tags appear only
from the classify pass — no blurry thumbnail can support those claims.

The **scan interval** controls sampling density (seconds between samples; default
2). Smaller catches shorter scenes but makes more VLM calls; larger is faster but
can miss brief flashes. Expect a full feature to take hours: this is the slow pass,
and rendering afterwards is fast and needs no models.

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
which categories to censor and how) and hit **Start render**.
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
(e.g. `http://127.0.0.1:8081`), hit **Scan** — pick a whisper model and it is
loaded on demand when the transcription starts (the server hot-swaps to it
first).
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
(`Movie.nuclearcutter.json`) — just change which tags are selected and render
again; no rescan is involved. Headless equivalent — the selection is a set of tag
ids or families, there is no level to tune:

```bash
# What can even be chosen, if you don't know the ids:
python3 nuclearcutter.py tags --search swimwear
python3 nuclearcutter.py tags --family gore --json

# Censor a starting preset, then subtract what you don't want:
python3 nuclearcutter.py render "/path/to/Movie.mkv" \
  --scan "/path/to/Movie.nuclearcutter.json" \
  --output "/path/to/Movie_lite_cut.mkv" \
  --preset sexual \
  --allow nudity.shirtless_male \
  --audio language=mute_word
```

`--censor` / `--allow` take a tag id *or* a whole family name, so
`--censor gore` means every gore tag. The flags apply in a fixed order —
**preset → `--censor` → `--allow` → `--grep` → `--tag-file`** — which is what makes
`--preset sexual --allow nudity.shirtless_male` read exactly as it sounds: start
broad, then subtract. A later flag overrules an earlier one, so put what you want
to *keep* before anything that re-adds it. `--grep REGEX` matches tag ids and
labels (`--grep 'blood|wound'` is the cheap way to ask for "all the blood"), and
`--tag-file` reads a household profile — one id per line, `#` comments allowed.

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
