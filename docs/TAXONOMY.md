# The tag vocabulary — what is in a scene, not how bad it is

**Taxonomy version 2 · schema version 2 · 121 tags in 8 families**

This document explains the vocabulary in
[`nuclearcutter/taxonomy.py`](../nuclearcutter/taxonomy.py). That file is the
single source of truth: the model's prompts, the scan file, the render plan, the
GUI's checkboxes and the CLI's `--preset` flags are all generated from it. Adding
a tag there is the only edit required to make it appear everywhere.

---

## 1. The one design decision everything else follows from

Version 0.1.0 asked the model *"how bad is this, on a low/med/high/exhigh scale
of how much censorship it needs?"* and threw away what it actually saw.

That question is unanswerable, for a reason worth stating plainly: **"bad" is not
a property of the frame.** It is a property of the frame *plus the person
watching*. So the model was asked to guess the viewer's values, and the answer
got frozen into the scan file. Consequences observed in real scans:

- A woman dancing with a fan came back as `violence / low` — the top tier of a
  scale whose own definition reads *"brutal acts, murder, torture"*. The model
  had no word for "dancing with a fan", so it reached for the least-worst label
  available and the scale's prose gave it no better one.
- A scan file became an *opinion*, so two people who disagree about swimwear
  cannot share it — one of them has to rescan the film, at hours of GPU time.
- Every threshold was a coin flip against a paragraph of pre-written prose
  (`DEFAULT_LEVEL_SCALES`, now deleted), written by whoever set up the server and
  invisible to the person using it.

Version 2 asks **one factual question** — *what is in this frame?* — and records
the answer as a set of tag ids. Deciding what deserves censorship is a separate,
local, revisitable act: a `Selection` is a set of tag ids and nothing more.

```
v1:  frames → model → "nudity, level=high"        → threshold gates the render
                                              ↑ one opinion, two places

v2:  frames → model → ["nudity.shirtless_male",
                        "intimacy.kissing"]        → your checkboxes pick which
                              ↑ a fact                of those facts to censor
```

The scan file stops being an opinion, which is what makes it worth sharing
([SPEC §6](SPEC.md)).

---

## 2. Families

| Family | Tags | Default action | What it covers |
|---|---|---|---|
| `nudity` | 35 | blur picture | How much of a body, or what kind of clothing, is visible |
| `intimacy` | 11 | blur picture | Kissing, undressing and sex — with or without nudity |
| `violence` | 26 | blur picture | One person hurting, or threatening to hurt, another |
| `gore` | 21 | blur picture | Wounds, blood and bodies that are damaged |
| `harm` | 5 | blur picture | Someone hurting themselves, and how it is done |
| `language` | 16 | mute phrase | Swearing, slurs, insults and explicit talk |
| `substance` | 4 | blur picture | Smoking, heavy drinking and drug use |
| `safety` | 3 | blur picture | The scan noticed something but could not name it |

Defaults are **starting points, not policies** — every family's action is a
dropdown in the GUI and a `--action` flag in the CLI. `language` mutes a phrase
rather than a word because a slur's meaning survives cutting one syllable out of
it; `word`-level muting is still available for mild swearing.

`harm` is visual because the *depiction* is the thing being censored, and it is
kept out of every preset (see §6).

---

## 3. Four rules the vocabulary holds to

### 3.1 Tags are facts, never verdicts

A tag's `hint` — the text the model actually reads — says when the thing **is**
visible. It never says the thing is offensive. Words like *flag*,
*inappropriate*, *sensitive* and *err on the side of* are banned from hints on
purpose: they are what turned the fan dance into `violence/low`.

Compare, for the same tag:

```diff
- # v1 prose, inside a level definition
- "any immodesty: bare shoulders, legs, towels, pajamas, kissing"
+ "nudity.shirtless_male — A male character's bare chest/torso is visible
+  through any opening, not just a bare chest."
```

The v1 line smuggles a judgement ("immodesty") into a detection instruction, and
a judgement about a *woman's* shoulders in a scale applied to everyone. The v2
line is checkable against a frame.

`assert_vocabulary_sane()` enforces this at import time — a hint containing a
banned word raises rather than ships, so the rule cannot rot back in.

### 3.2 Gender only where a body state is gendered

Gender lives in the **tag id**, never in a parallel column, so the chooser is one
checkbox per fact: `nudity.breasts_full_female`, `nudity.shirtless_male`.

Three deliberate limits:

- **Interactions are not gendered.** `intimacy.kissing`, not
  `kissing_male_female`. Getting the pair right needs *both* subjects identified;
  one flipped guess silently un-censors the scene, and the failure is invisible.
  A gender-neutral tag can only ever be over-inclusive.
- **Ambiguity tags both.** If the model cannot tell whether a figure is male, it
  is asked to emit both gendered tags. Over-censoring is recoverable by unticking
  a box; under-censoring is not recoverable at all.
- **Gore and violence are never gendered.** "A person is bleeding" does not get
  less bloody depending on who they are.

`Rendering` (live-action / animated / anime / artwork) is a *modifier* stored per
detection, not a tag dimension. Cross-producting it into ids would quadruple the
vocabulary for information the viewer wants as four checkboxes.

### 3.3 Uncertainty degrades to a tag, never to nothing

Every family has an `<family>.unclear` residual **except two**, plus two safety
leftovers:

| Tag | Meaning |
|---|---|
| `<family>.unclear` | Something in this family is on screen; it cannot be pinned to a specific tag |
| `safety.unclassified` | The scan flagged the moment but could not name it at all |
| `safety.exposure_age_uncertain` | Exposure where the subject's age could not be determined |

A model tag outside the vocabulary is **folded into its family's residual** — not
dropped. A dropped tag is an uncensored scene, the one failure mode this tool
cannot have. The same folding happens when a scan file from a *different build*
is loaded: unknown ids are preserved and treated as censorable.

**The bend: `harm` and `substance` have no residual.** This is the one place the
rule gives way, and both bend for the same reason — their content is identified
by **context**, not by appearance, so a blurry frame cannot support the claim.
Is that an eating-disorder plot or an athletic character? Is that medicine or a
narcotic? A `harm.unclear` would fire on dark clothing and a raised arm, and
because it is censorable by default it would silently delete plotlines the viewer
sat down to watch. A filter that eats the story gets switched off, which is worse
for the people who need it than the gap it leaves. Contrast `gore.unclear`: blood
*is* a visual property, so "looks like blood" is an honest observation. Nothing
is dropped — an unnamed-but-flagged moment still lands on `safety.unclassified`.
Only the specific *claim* about harm is withheld.

`uncertain` is an ordinary checkbox, on by default in every preset that censors
anything. That is the honest position: "we don't know what this is" is a reason to
*show the user the moment*, not a reason to hide it from them or censor it silently.

### 3.4 Exactly one thing is not the user's choice

`safety.exposure_with_minor` is the taxonomy's single `always_censor` tag.
`assert_vocabulary_sane()` asserts the count is exactly 1 — a second one would
mean the tool started making moral decisions again.

**What triggers it: the `minors` flag, not the presence of a nudity tag.**

- `minors == yes` → floor applies, unconditionally. The model's confident claim
  that a child is in frame is the whole case; whether it also named an exposure
  tag is a detail we should not bet a child's protection on.
- `minors == uncertain` → floor does *not* apply, and `exposure_age_uncertain`
  is offered instead, and only when something is actually exposed. Estimating age
  from thumbnails says "possible child" for crowds, doorways and toys; forcing it
  would censor a large share of ordinary scenes and the tool would get switched
  off — which is worse for children than the gap it closes.

Keying the floor on *tags* instead would have been a bug with a nice UI: tag
lists get **replaced** when a viewer corrects the model, so deleting the exposure
tags would delete the floor with them. `schema.replace_tags()` is the only
supported whole-list rewrite, it re-adds the floor while `minors == yes`, and
clearing `minors` is the honest way to say the age estimate was wrong.

The tag is shown in the chooser, disabled, marked *always on*. A rule the user
cannot see looks like a bug and gets worked around; a rule they can see gets
trusted.

---

## 4. Selections: the viewer's side

A `Selection` is a set of tag ids. Nothing else. There is no rank, no threshold,
and no comparison anywhere in the render planner — planning is membership:

```python
det.censorable(selection.tags)     # any overlap → this segment is censored
```

Because it is a set, it is shareable in a way a threshold never was: two people
can send each other "these 48 categories" and mean exactly the same thing.

### Presets

Presets are **starting points**, one tap to get close and then edit. From
`taxonomy.PRESET_NAMES`, generated by `selection_named()`:

| Preset | Tags | Adds |
|---|---|---|
| `none` | 0 | nothing — the default, the tool does not decide for you |
| `nudity` | 37 | the `nudity` family |
| `sexual` | 48 | + `intimacy` |
| `language` | 22 | `language` + `substance` (orthogonal to the above) |
| `graphic` | 49 | `violence` + `gore` (**not** `harm`, see below) |
| `everything` | 120 | every choosable tag (121 minus the child floor, which is not a choice) |

They nest (`nudity` ⊂ `sexual` ⊂ `everything`), so a preset button never *removes*
what an earlier one picked. The GUI builds its row of buttons from the server's
`preset_order`, and the CLI's `--preset` choices come from the same tuple — three
hand-maintained lists drift, and the failure is a button that 400s or a flag
nobody can guess.

Deliberately absent:

- **No "recommended" preset.** Choosing what is offensive is the entire decision
  this tool hands back to the viewer.
- **No "adult" preset.** `safety` holds only the floor (not a choice) plus two
  leftovers, so "all adult families" and "everything" resolved to the identical
  tag set. Two buttons that censor identically teach the viewer to distrust both.
- **`harm` in no preset.** Self-harm and suicide depiction is the category where
  a filter can plausibly do harm (a viewer who needs the depiction of an
  addiction, an eating disorder, or suicide to understand it is a viewer who
  benefits from it), and it is small enough to choose deliberately. Tick it; do
  not inherit it.

---

## 5. Rendering qualifiers

Stored per detection, applied as checkboxes, **all on by default** — 7 styles:

| Style | What it catches |
|---|---|
| `live_action` | Ordinary filmed footage |
| `animated` | Animation, CGI, VFX creature |
| `anime` | Anime-style specifically |
| `artwork` | Painting, drawing, statue, stained glass |
| `archival` | Newsreel and documentary file footage |
| `fantasy` | Dream, flashback or imagined sequence, or non-real fluid |
| `stage` | Filmed stage or theatrical performance |

Defaults are all-on because a wrong guess must never silently un-censor. A
detection tagged `artwork` that the viewer forgot to include would otherwise
vanish from the timeline; instead, un-ticking `artwork` is a deliberate narrowing
("I don't need the museum scene blurred").

---

## 6. What the scan cannot tell you

Stated here because the tool should not imply confidence it does not have. The
sweep sees ~480px thumbnails at 1 frame per 4 seconds; classify sees 6–8 frames
of the range.

**Reliable.** Presence/absence of a clear object or act: a gun, a bare chest, a
punch, blood, a body count. Coarse facts at low resolution are what VLMs are good
at.

**Unreliable — do not trust it to be complete:**

- **Brief exposure.** A half-second flash between sampled frames is missed and no
  tag says so. This is why `<family>.unclear` and dialogue-informed densification
  exist: if the audio says "get her out of there", the sweep's silence is suspect.
- **Gender of a distant or ambiguous figure**, hence the tag-both rule in §3.2.
- **Age.** `minors` is a guess. It biases toward `uncertain` for this reason, and
  `uncertain` is a checkbox rather than a floor.
- **Simulated versus real.** `gore` and `violence` cannot tell a staged fight from
  a real one, and `harm` cannot tell a depiction from instruction. Both are
  conservative by construction, which means they over-report.
- **Whether a person is a minor *in fiction*.** An adult actor playing a teenager
  will be tagged by how they look. The renderer cannot know, and should not guess
  in the other direction.

The honest summary: **a scan is a floor, not a ceiling.** Content absent from the
scan file may still be in the film. When in doubt about a specific film, watch the
`*.unclear` and `safety.unclassified` segments first — they are the moments the
scan is telling you it was unsure about, and that is the only place the tool can
be candid.

---

## 7. Migration from v1

v1 scan files are **rejected with a non-zero exit**, never coerced. A v1 `level`
is a judgement about how much censorship a scene needed; there is no honest way
to turn it back into "what is in the frame", and a lossy auto-conversion would
produce a scan file that looks authoritative while quietly inventing tags.

```
$ nuclearcutter render --scan old.scan.json
scan file is schema v1; this build reads v2. Re-run `nuclearcutter scan`.
```

Re-scan each film once. The upgrade is one scan per film and then the vocabulary
can grow without another rescan, because adding a tag does not invalidate facts
already recorded — it only makes new distinctions possible in future scans.

Detections converted by hand from v1 (if you script it yourself) should carry
`confidence=0.4` and land on `<family>.unclear`, so the GUI presents them as the
guesses they are rather than as observations.

---

## 8. Adding a tag

1. Add it to `_TAGS` in `taxonomy.py` with `_t()`: id, label, family, `hint`.
   The `hint` describes **when it is visible**, in the model's language, with no
   evaluative vocabulary (enforced by `assert_vocabulary_sane()`).
2. Decide whether it is gendered, `visual`, `audio`, and whether the family's
   default action suits it.
3. Run `pytest tests/test_taxonomy.py`. The vocabulary-sanity tests check id
   uniqueness, that every family with a `_RESIDUAL` entry has that tag, that
   exactly one tag is `always_censor`, that presets nest, and that every trigger
   name the sweep prompt offers exists in the vocabulary.
4. Bump `TAXONOMY_VERSION` if a tag was renamed or removed (adding is safe and
   does not require it), and `SCHEMA_VERSION` only if the *file format* changed.

The prompt text, GUI chooser, CLI presets and `/api/taxonomy` all pick it up
automatically. Nothing is hand-maintained in a second place — that is the point.
