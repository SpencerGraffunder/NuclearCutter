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

**Pass 1 — Scan.** Analyze the movie and produce a JSON file recording every
detected moment: its time range, the **tag ids** describing what is in the frame,
and (for moments being censored) an AI-generated scene description. This pass does
**not** know or care what the user wants done about any of it — it is a neutral
record of "what is in this movie and when". "Neutral" is load-bearing, and it is
why the scan records facts rather than a verdict: see [TAXONOMY.md](TAXONOMY.md) §1.

**Pass 2 — Render.** Read the scan JSON + the viewer's `Selection` (which tag ids
to censor, and what each family's tags should do), and produce the final `_cleaned`
output file via ffmpeg.

This separation is deliberate: it's what makes the scan JSON shareable in the repo
(see §6) — two users with different censorship preferences can use the exact same
scan data.

## 3. What is detected, and what gets done about it

Detection answers **one factual question** — *what is in this frame?* — and
records the answer as a set of tag ids. Deciding what deserves censorship is a
separate, local, revisitable act performed against that record. The full
vocabulary and the reasoning behind it live in **[TAXONOMY.md](TAXONOMY.md)**;
this section is the interface summary.

**8 families, 68 tags** (taxonomy v3 merged v2's 121; retired ids still resolve
through `LEGACY_TAGS`) (`nuclearcutter/taxonomy.py` is the single source of
truth: prompts, scan files, the render plan, the GUI and the CLI's presets are all
generated from it):

| Family | Tags | Default | Covers |
|---|---|---|---|
| `nudity` | 35 | blur picture | How much of a body, or what clothing, is visible |
| `intimacy` | 11 | blur picture | Kissing, undressing, sex — with or without nudity |
| `violence` | 26 | blur picture | One person hurting, or threatening to hurt, another |
| `gore` | 21 | blur picture | Wounds, blood and damaged bodies |
| `harm` | 5 | blur picture | Self-harm and suicide, and how it is depicted |
| `language` | 16 | mute phrase | Swearing, slurs, insults, explicit talk |
| `substance` | 4 | blur picture | Smoking, heavy drinking, drug use |
| `safety` | 3 | blur picture | The scan flagged the moment but could not name it |

**There is no severity scale anymore, anywhere.** v0.1.0 classified each
detection as `low`/`med`/`high`/`exhigh` on a per-category scale of *how much
censorship it needs*, and gated the render on a threshold comparison. That
question is unanswerable — "bad" is a property of the frame *plus the viewer* —
and it froze a guess into the scan file, so people who disagreed could not share
one. `SeverityLevel` is deleted; planning is set membership:

```python
det.censorable(selection.tags)   # any overlap → the segment is censored
```

**A `Selection` is a set of tag ids and nothing else.** No rank, no threshold, no
comparison in the planner. Because it is a set it is shareable the way a threshold
never was, and it is *revisitable*: change the set and re-render in minutes, with
no rescan. This is what makes §6 worth having.

**Visual actions** (per family, all overridable):

| Action | Effect |
|---|---|
| `none` | leave untouched |
| `blur` | intense box blur over the range + a short clean summary overlaid |
| `black` | replace the range with a black screen + the summary |

Only these three exist (`schema.VisualAction`). A "freeze on the last clean frame"
action was considered and not built: it needs a reliable keyframe outside the
censored range, and getting that wrong freezes on a frame *of* the scene being
hidden — a failure too quiet to notice.

**Audio actions:** visual families only recognise a *time range*, not a sound, so
their options are `none` / `mute_scene`. `language` has word-level timestamps and
keeps the fine-grained set: `none` / `mute_word` / `mute_phrase` /
`replace_word` / `replace_phrase` (phrase replacement is the default, because a
slur's meaning survives cutting one syllable; `replace_*` currently fall back to
the equivalent mute with a warning).

`blur` preserves scene flow and avoids the jolt of a skip card; `black` shows
nothing but the summary. When one range matches families asking for different
actions, the **strongest** ask wins (a range that is both gore→blur and
violence→black goes black) — leaving one of the viewer's asks unfulfilled is worse
than over-censoring. Audio-only language detections never create a segment
boundary; their windows are layered inside whichever segment contains them.

Runtime is always preserved (subtitles and chapter markers stay in sync). Blur
strength: `blur_strength` (1.0 standard, 2.0 twice as extreme).

**Three things are enforced rather than chosen**, because leaving them optional
would put the burden of a safety decision on the person least able to audit it:

1. `safety.exposure_with_minor` — the taxonomy's single `always_censor` tag,
   applied whenever a detection reports a minor present. The GUI shows it disabled
   and labelled *always on*: a rule the user cannot see looks like a bug and gets
   worked around. `assert_vocabulary_sane()` asserts exactly one such tag exists —
   a second would mean the tool started making moral decisions again.
2. Uncertainty is a tag, never silence. Unknown model output folds into the
   family's `unclear` residual (or `safety.unclassified`), so a misspelt or
   invented tag cannot silently delete a finding. (`harm` and `substance` opt out
   of having residuals — see TAXONOMY §3.3 for why inferring those from a blurry
   frame does more harm than the gap it closes.)
3. v1 scan files are rejected with a non-zero exit, never coerced. There is no
   honest way to turn "how bad was this" back into "what was in the frame", and a
   lossy auto-conversion would produce a scan file that looks authoritative while
   inventing tags.

## 4. Detection pipeline

### 4.1 Visual: trigger sweep, then one classifier pass

Two passes, split because they answer different questions at different cost — and
because a single pass cannot explain the difference between what a batch of
thumbnails says and what real pixels of one range show.

**Sweep (cheap, whole film).** Frames are sampled across the film and sent in
small batches. The model is asked one cheap question per batch with a **short list
of 12 trigger names** (`taxonomy.SWEEP_TRIGGERS`) rather than the 68-tag
vocabulary: *which of these, if any, appear in these frames?*

```
skin · body_parts · intimate · blood · injury · fighting
hostile · restraint · danger · substance · offensive_speech · doubt
```

Note the triggers are deliberately **not** the family names. They are written for
recall at thumbnail resolution — `skin` means "skin that street clothes would
cover", not a `nudity.*` verdict, and `danger` sweeps up a child in peril that no
family owns. A one-to-one mapping to families would tempt someone into routing a
trigger straight to a tag, which is exactly the shortcut that drops content.

Coarse presence questions are what a low-resolution batch is good at, and keeping
the vocabulary out of this prompt is what keeps the call cheap.

The trigger verdict is **authoritative — the sweep can flag, and classify can never
un-flag.** The confirm-style pass has no veto power: it can only enrich or keep the
sweep's verdict. That asymmetry is why the passes cannot be merged into one
"smarter" call; merging two classifications duplicates work without adding safety.

**Merge + pad.** Flagged windows merge into ranges with generous padding so a scene
is never clipped. When windows merge, tags take the **union** — not the worst
level, since levels no longer exist. Losing a tag in a merge is an uncensored
scene.

**Classify (expensive, per range).** Each merged range is sent at higher resolution
with the **full vocabulary** and returns tag ids, a `description`, `confidence`,
a `rendering` qualifier and a `minors` estimate (both chosen on the CLI, not in
the GUI). Family routing is deliberately
not used: the full vocabulary is always offered, because a nudity range routinely
contains violence too and routing would drop it. Unknown ids fold to the family
residual (§3). `minors` takes the **worst** of sweep and classify, so a sweep that
spotted a child is never overruled by a pass that was not looking for one.

**Descriptions.** A `description` must weave together what is visually happening
*and* what is being said during the range — the whole point of a blur card is that
the viewer does not lose story content. Not "two characters embrace" but the same
sentence carrying the plot-relevant dialogue over the embrace.

Descriptions are polished **lazily at render time, only for ranges actually being
censored** (`polish_descriptions`), not in a mandatory whole-film summary pass. On a
two-hour film where the viewer censors little, a scan-time summary pass spent model
time narrating scenes nobody intended to hide.

**Model access** is an OpenAI-compatible chat completions API
(`/v1/chat/completions` with image content blocks). Two backends, one API:

- **`mlx-vlm` (default):** NuclearCutter spawns its own `python -m
  mlx_vlm.server` on `localhost:1234`, pre-loads the configured model, and shuts it
  down when the scan finishes. Speed knobs: images downscaled (~480px) before
  upload — the single biggest latency lever (~6x) — plus 4-bit KV-cache
  quantization and continuous batching.
- **`standalone`:** an already-running server (LM Studio, Ollama, a manual
  mlx-vlm instance) via `base_url`. Nothing is started or stopped.

mlx-vlm caches models keyed by the exact `model` string, pre-loading under the
**full filesystem path**, so requests must send `model_path` as the model id — not
the short name — or the server fetches the short name as a Hugging Face repo. The
same loaded model serves VLM and text calls.

A blind fast pass is never the gatekeeper. A NudeNet-then-VLM design was tried and
removed: NudeNet scored **zero** on every frame of a full nude scene that the VLM
flagged at 0.95 confidence, so the VLM never saw those scenes at all. Any future
pre-filter must be recall-first or it silently drops real content.

### 4.2 Language: wordlist for precision, model for meaning

- **Whisper transcription** with word-level timestamps, run always (via
  `mlx-whisper` on Apple Silicon).
- **Subtitle cross-check:** an embedded or sidecar subtitle file (SRT/ASS/etc) is
  parsed and cross-checked against the transcript for the same ranges — subtitles
  are exact text and correct Whisper's mishearings.
- **Wordlist pass (~190 entries, `detection/data/profanity_wordlist.txt`)** gives
  *precision*: word-level timestamps, and a deterministic second opinion the model
  cannot talk you out of. A wordlist alone is wrong often enough to matter ("assess"
  containing a slur, a clinical use of a crude anatomical term).
- **A dialogue sweep over the transcript** tags the language family descriptively:
  which kind of utterance it is (`slur_racial`, `profanity_severe`,
  `sexual_talk_explicit`, `degradation_insult`, ...). A model alone is expensive and
  unreliable at timestamps; a wordlist alone cannot tell a slur from a medical
  term, so the two are merged and each does what the other cannot.
- **Merging unions the tags.** A wordlist hit plus a model hit is one detection with
  both tags, never two detections and never one winning.
- **What a language tag describes is the utterance, not the speaker's intent.**
  `hostile_tone` and `degradation_insult` are judged from words alone, so they will
  be wrong about irony, and about a villain being cruel *to* someone on purpose.
  Per-detection correction exists for exactly this.
- Language is **not** gendered, and never will be: "a slur" is the same words
  whoever says them, and the target's identity is a guess a model gets wrong.

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

- The JSON stores **raw detections only** — tag ids, timestamps, VLM
  descriptions, confidence data. It does **not** store which action the original
  generator chose (blur vs skip vs mute), nor *which of the tags the generator
  cared about* — that lives in a separate, local, non-shared `Selection`. This is
  what makes one scan JSON useful to every user regardless of their personal
  filtering choices.
- Because a detection is a **fact** ("a shirtless man, a gun, blood") rather than
  a verdict ("severity: high"), two people who disagree completely about what
  deserves censoring can still use the same file. Under v1's severity levels they
  could not: the scan encoded somebody else's threshold, and disagreeing with it
  meant rescanning the film — hours of GPU time — to hold a different opinion.
- Tags are **additive across versions**: adding a tag to the vocabulary does not
  invalidate facts already recorded, so a scan ages rather than rots. A renamed or
  removed tag bumps `TAXONOMY_VERSION`; a format change bumps `SCHEMA_VERSION`, and
  a file on an older schema is refused loudly (never silently reinterpreted).
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
`timestamps_dir`, `blur_strength`, `mute_padding` (extra seconds muted around each
flagged word) and `font`.

The censoring intent lives in **five** keys, replacing v1's per-category
`*_level` thresholds and per-category `*_visual`/`*_audio` pairs:

| Key | Shape | Meaning |
|---|---|---|
| `selection` | list of tag ids | which facts get censored — the whole decision |
| `family_actions` | `{family: {visual, audio}}` | what to do when a family's tag is selected |
| `rendering` | list from `live_action`/`animated`/`anime`/`artwork`/`archival`/`fantasy`/`stage` | which presentation styles to filter (all on = widest) |
| `minors` | list of `yes`/`uncertain` | which age estimates count as "a child is present" for the floor |
| `polish_descriptions` | bool | spend model time at render to improve blur-card text |

The removed keys had no replacement because the concept went away: there is no
per-category threshold to set, and no per-category custom prompt — a prompt that
redefines what a tag *means* would silently invalidate every scan file written
under the built-in definition (see [TAXONOMY.md](TAXONOMY.md) §1).

`settings.json` holds defaults for new scans; a scan file's own `Preferences` (and
the GUI's edits to them) win over it, so a scan can be re-rendered later with
different intent without touching the shared defaults.

### 7.1 Live dashboard

`scan` and `render` render an inline TUI dashboard while they run (film
timeline with detection markers, current-position cursor, ETA, and CPU/RAM).
It's enabled by default on a TTY and disabled by `--no-tui` or when output is
piped. Both commands also stream a live status JSON (`ScanStatus`, see
`nuclearcutter/utils/scan_status.py`); the same dashboard can be attached to a
scan running elsewhere via `nuclearcutter tui --status FILE` (or `--log FILE`
to tail an older scan's log that predates status files).

### 7.2 Progress bars are a contract, not decoration

The web STATUS panel draws one bar per pipeline stage, in the order the stages
actually run:

| Row | Wire key | Job | Meaning of the fill |
|---|---|---|---|
| TRANSCRIBE | `transcribe` | scan | whisper pass over the audio |
| LANGUAGE | `language` | scan | dialogue passages checked for words to mute |
| SCAN | `scan` | scan | VLM sweep samples examined (where to look) |
| VERIFY | `identify` | scan | candidate ranges classified (what is there) |
| SUMMARIZE | `summary` | render | censored segments whose caption is being rewritten |
| RENDER | `render` | render | segments cut so far |

Two rules, both enforced by tests in `tests/test_server.py`:

1. **Every row reads a key the server actually puts in `job.steps`, and every key
   the server reports has a row.** VERIFY used to read `steps.verify`, a key no
   server code has ever set (the pass is `identify` everywhere else), so the row got
   `undefined` and drew as the indeterminate sliding block — moving constantly,
   showing no position, for the whole of a scan in which an exact
   `(ranges_done, ranges_total)` was available unread. The row label is VERIFY
   because that is what the ✕ clears; the wire key is `identify` because that is what
   the pass is called. A row for `language` was simply missing, so that whole stage
   ran with nothing on screen moving.
2. **A missing key may never render as progress.** `setBar` distinguishes three
   states: a number (real position), an explicit `null` (running, position genuinely
   unknowable — the sliding block, and only reachable when asked for), and
   `undefined` (the key does not exist, which is a bug, drawn as red hatching with a
   `?` and one `console.warn`). Silently animating a bar is worse than a flat bar,
   because it looks alive.

Since no stage reports `None` any more, a scan or render should always show where it
is. Stages report `(done, total)` pairs; `summary` counts only the segments polish
will rewrite, so it reaches 100% on a mostly-uncensored film instead of crawling to
1/50.

## 8. Non-goals / explicit exclusions

- Not a live-playback filter. No Plex/Jellyfin plugin, no client-side
  integration. Operates only on files, before they ever reach a media server.
- Not scoped to build a full review/edit web UI in v1.
