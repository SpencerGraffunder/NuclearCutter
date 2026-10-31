"""
Language detection: what is being SAID that a viewer may want filtered.

v1 did one thing: match a profanity wordlist against the transcript, ask an LLM
whether each match was really profane, then grade the word on a four-step
severity scale whose widest rung ("exhigh") was defined as "ANYTHING rude, mean,
disrespectful, or unkind — including 'omg', 'I hate you', 'shut up', 'idiot'".
That design had two failures, and they pulled in opposite directions:

  * A slur and the word "damn" both lived in the same family, so a household
    that cares enormously about slurs and nothing about "fuck" could not say so.
  * Everything non-lexical was invisible, because the wordlist was the only
    gate: a sexual remark, a crude joke, a cruelty aimed at someone, a reference
    to doing drugs, or a slur the wordlist does not carry could never be found
    at all — no LLM ever saw a line without a wordlist hit.

v2 runs two passes and records tag ids:

  A  word pass   wordlist hits, then `language_line_prompt` judges each hit in
     context and assigns per-word categories (and rejects homophones).
  B  dialogue pass   every line, grouped, judged for the things a wordlist
     cannot find: slurs it does not carry, sexual talk, crude jokes, deliberate
     cruelty, and drug/drinking/smoking references.

Both write the same `LanguageDetection` shape, and overlapping hits are merged
so one line is one timeline marker with a union of tags.
"""

from __future__ import annotations

import logging
import re
from pathlib import Path

from nuclearcutter.detection.transcribe import Utterance, Word
from nuclearcutter.prompts import get_prompt
from nuclearcutter.schema import LanguageDetection
from nuclearcutter.taxonomy import (
    DIALOGUE_GROUP_IDS, Family, dialogue_group_prompt, normalize_tags, taxonomy_prompt,
)
from nuclearcutter.utils.llm_client import LLMClient

DEFAULT_WORDLIST_PATH = Path(__file__).parent / "data" / "profanity_wordlist.txt"

# Lines per dialogue-group call. Text calls are an order of magnitude cheaper
# than vision calls, but a 2 h musical has ~1500 lines, so the group size still
# matters: 12 lines is ~130 calls for the whole film.
DIALOGUE_GROUP_SIZE = 12

# How far apart two language detections must be to stay separate markers.
_MERGE_WINDOW = 0.75


def load_wordlist(path: Path = DEFAULT_WORDLIST_PATH) -> set[str]:
    """Read the trigger list. Missing file -> empty set (the LLM pass still runs).

    Normalised and de-duplicated on read because the file is meant to be edited
    by hand: an entry typed as ` FUCK ` or repeated in two sections must not turn
    into two detections, and a trailing space would stop it matching anything.
    """
    try:
        raw = path.read_text(encoding="utf-8", errors="replace").splitlines()
    except OSError:
        logging.warning("wordlist %s not readable — language pass relies on the "
                        "LLM alone", path)
        return set()
    out: set[str] = set()
    for line in raw:
        word = line.strip().lower()
        if not word or word.startswith("#"):
            continue
        # A line with a trailing comment ("hell  # mild") is a hand-edit someone
        # will make; keep the word and drop the note.
        word = word.split("  #", 1)[0].split("\t#", 1)[0].strip()
        if word and not word.startswith("#"):
            out.add(word)
    return out


def wordlist_matches(text: str, wordlist: set[str]) -> list[str]:
    """Distinct trigger words present in `text`, in first-seen order.

    Word-boundary matching, because substring matching on this list is a joke
    nobody wants: "class" contains "ass", "classic" contains "ass", "ahem"
    contains "hem"-ish nonsense, and "Scunthorpe" famously contains a slur. The
    boundary allows spaces AND punctuation on either side so "damn!" and
    "(shit)" still match, but not "asinine".

    Multi-word entries ("pain in the ass") work too — the boundary check is the
    same, and the phrase form is what lets the list cover fixed insults a
    single-word list would miss.

    Returns each word ONCE per line. Two occurrences of the same word in one line
    are still both muted: `_word_timings` finds every occurrence's timestamp, so
    repeating the entry here would only duplicate the same detection.
    """
    if not wordlist or not text:
        return []
    lowered = text.lower()
    hits: list[str] = []
    for word in wordlist:
        if not word:
            continue
        start = 0
        found = False
        while True:
            idx = lowered.find(word, start)
            if idx < 0:
                break
            before_ok = idx == 0 or not lowered[idx - 1].isalnum()
            after = idx + len(word)
            after_ok = after >= len(lowered) or not lowered[after].isalnum()
            if before_ok and after_ok:
                found = True
                break
            start = idx + 1
        if found:
            hits.append(word)
    return hits


def language_categories_prompt(exclude_ids: set[str] | None = None) -> str:
    """The category text for the word pass: all language + substance."""
    return taxonomy_prompt([Family.LANGUAGE, Family.SUBSTANCE],
                           exclude_ids=exclude_ids or set())


def detect_language(
    utterances: list[Utterance],
    client: LLMClient,
    wordlist: set[str] | None = None,
    subtitle_utterances: list[Utterance] | None = None,
    run_dialogue_sweep: bool = True,
    on_progress=None,
) -> list[LanguageDetection]:
    """Full language pass over a transcript. Returns tagged detections.

    `on_progress(done, total)` reports the dialogue-sweep progress (the word
    pass is fast and does not report).
    """
    wordlist = wordlist if wordlist is not None else load_wordlist()
    subtitle_utterances = subtitle_utterances or []

    detections = _word_pass(utterances, client, wordlist, subtitle_utterances)
    if run_dialogue_sweep:
        detections += _dialogue_pass(utterances, client, subtitle_utterances,
                                     on_progress=on_progress)
    merged = merge_language_detections(detections)
    logging.info("language: %d detection(s) (%d before merging)",
                 len(merged), len(detections))
    return merged


# ---------------------------------------------------------------------------
# Pass A: wordlist hit -> contextual check -> categories
# ---------------------------------------------------------------------------

def _word_pass(utterances, client, wordlist, subtitle_utterances) -> list[LanguageDetection]:
    categories = language_categories_prompt()
    out: list[LanguageDetection] = []

    for utt in utterances:
        hits = wordlist_matches(utt.text, wordlist)
        if not hits:
            continue
        try:
            result = client.text_query_json(get_prompt(
                "language_line_prompt", categories=categories,
                text=utt.text, matches=hits))
            llm_failed = False
        except Exception as exc:  # noqa: BLE001
            # Fall back to trusting the wordlist rather than silently dropping a
            # possible detection — but say which categories we are claiming, and
            # do not let a dead server turn into the strongest category.
            logging.warning("language line check failed (%s); trusting the wordlist "
                            "for %d word(s)", exc, len(hits))
            result = {"matches": {w: [] for w in hits},
                      "reasoning": f"llm_check_failed: {exc}"}
            llm_failed = True

        match_map = result.get("matches") or {}
        if not isinstance(match_map, dict):
            match_map = {}
        line_tags = normalize_tags(result.get("line_tags") or [])
        source = _cross_check_source(utt, subtitle_utterances)

        for word in hits:
            entry = match_map.get(word)
            if entry is None:
                entry = match_map.get(word.lower())
            if isinstance(entry, str):
                entry = [entry]
            if llm_failed:
                # The wordlist hit on its own: the honest claim is "this word is
                # here", not a category the LLM never confirmed. Unknown ids and
                # empties fold into language.unclear, so it stays censorable by
                # anyone who turns the residual on without being labelled a slur.
                tags = ["language.unclear"]
                confirmed = True
            elif entry is None:
                # The model was asked about this word and did not answer for it:
                # it rejected it. Not a detection.
                continue
            else:
                tags = normalize_tags(entry)
                confirmed = bool(tags)
            if confirmed and not tags:
                tags = ["language.unclear"]
            if not confirmed:
                continue

            timings = _word_timings(utt, word)
            if not timings:
                # No word-level timings (a subtitle-only transcript, or a
                # transcript server that omits them): mute the sentence rather
                # than nothing.
                timings = [(utt.start, utt.end)]
            for (start, end) in timings:
                out.append(LanguageDetection(
                    start=start, end=end,
                    utterance_start=utt.start, utterance_end=utt.end,
                    text=word, tags=tags, transcript_source=source,
                    llm_confirmed=not llm_failed,
                    llm_reasoning=result.get("reasoning")))

        if line_tags:
            out.append(LanguageDetection(
                start=utt.start, end=utt.end,
                utterance_start=utt.start, utterance_end=utt.end,
                text=utt.text[:80], tags=line_tags, transcript_source=source,
                llm_confirmed=True, llm_reasoning=result.get("reasoning")))
    return out


# ---------------------------------------------------------------------------
# Pass B: the things a wordlist cannot find
# ---------------------------------------------------------------------------

def _dialogue_pass(utterances, client, subtitle_utterances, on_progress=None):
    categories = dialogue_group_prompt()
    out: list[LanguageDetection] = []
    groups = [utterances[i:i + DIALOGUE_GROUP_SIZE]
              for i in range(0, len(utterances), DIALOGUE_GROUP_SIZE)]
    total = len(groups)

    for gi, group in enumerate(groups):
        lines = "\n".join(f"{n}: {u.text}" for n, u in enumerate(group))
        try:
            result = client.text_query_json(get_prompt("dialogue_group_prompt",
                                                       categories=categories, lines=lines))
        except Exception as exc:  # noqa: BLE001
            # A failed group is lost recall, so say so — but it is a cheap text
            # call and the sweep keeps going; aborting a whole scan over one
            # timeout would be worse.
            logging.warning("dialogue group %d/%d failed (%s) — its lines were not "
                            "checked for non-wordlist content", gi + 1, total, exc)
            if on_progress:
                on_progress(gi + 1, total)
            continue

        for item in result.get("lines") or []:
            if not isinstance(item, dict):
                continue
            try:
                idx = int(item.get("index"))
            except (TypeError, ValueError):
                continue
            if not (0 <= idx < len(group)):
                continue
            tags = normalize_tags(item.get("tags") or [])
            # Anything the model invented that is not in the group's own
            # vocabulary is dropped here rather than trusted — this pass is
            # allowed to report only what it was asked about.
            tags = [t for t in tags if t in DIALOGUE_GROUP_IDS]
            if not tags:
                continue
            utt = group[idx]
            out.append(LanguageDetection(
                start=utt.start, end=utt.end,
                utterance_start=utt.start, utterance_end=utt.end,
                text=str(item.get("text") or utt.text)[:80],
                tags=tags,
                transcript_source=_cross_check_source(utt, subtitle_utterances),
                llm_confirmed=True,
                llm_reasoning="dialogue sweep"))
        if on_progress:
            on_progress(gi + 1, total)
    return out


# ---------------------------------------------------------------------------
# Merge + timing helpers
# ---------------------------------------------------------------------------

def merge_language_detections(dets: list[LanguageDetection]) -> list[LanguageDetection]:
    """Collapse detections that describe the same moment into one marker.

    Pass A fires on a word, pass B on the line containing it; showing both is
    two checkboxes for one thing the user has to decide once. Union of tags, and
    the tightest start/end and the containing utterance are kept, because the
    renderer mutes either the word or the utterance depending on the action.
    """
    if not dets:
        return []
    ordered = sorted(dets, key=lambda d: (d.start, d.end))
    out: list[LanguageDetection] = []
    for d in ordered:
        prev = out[-1] if out else None
        if prev is not None and abs(prev.start - d.start) <= _MERGE_WINDOW:
            for t in d.tags:
                if t not in prev.tags:
                    prev.tags.append(t)
            prev.start = min(prev.start, d.start)
            prev.end = max(prev.end, d.end)
            prev.utterance_start = min(prev.utterance_start, d.utterance_start)
            prev.utterance_end = max(prev.utterance_end, d.utterance_end)
            prev.llm_confirmed = prev.llm_confirmed or d.llm_confirmed
            if "whisper+subtitle" in (prev.transcript_source, d.transcript_source):
                prev.transcript_source = "whisper+subtitle"
            continue
        out.append(d)
    return out


def _word_timings(utt: Utterance, target_word: str) -> list[tuple[float, float]]:
    """Every occurrence of `target_word` in the utterance, as (start, end).

    All of them, not just the first: "damn it, goddamn" has to mute both, and
    v1's single-timing lookup muted one and left the other audible.
    """
    target_clean = re.sub(r"[^a-z' ]", "", target_word.lower()).strip()
    if not target_clean:
        return []
    if " " in target_clean:
        # Phrase entry: locate it in the plain text, then interpolate the timing
        # from the surrounding word timestamps.
        joined = " ".join(re.sub(r"[^a-z' ]", "", w.text.lower()).strip()
                          for w in utt.words)
        pos = joined.find(target_clean)
        if pos < 0 or not utt.words:
            return []
        chars = 0
        span = [None, None]
        for w in utt.words:
            wlen = len(re.sub(r"[^a-z' ]", "", w.text.lower()).strip())
            if span[0] is None and chars + wlen > pos:
                span[0] = w.start
            if span[0] is not None and chars + wlen >= pos + len(target_clean):
                span[1] = w.end
                break
            chars += wlen + 1
        if span[0] is not None and span[1] is not None:
            return [(span[0], span[1])]
        return []
    return [(w.start, w.end) for w in utt.words
            if re.sub(r"[^a-z' ]", "", w.text.lower()).strip() == target_clean]


def _cross_check_source(utt: Utterance, subtitle_utterances: list[Utterance]) -> str:
    for sub in subtitle_utterances:
        if sub.start < utt.end and sub.end > utt.start:
            return "whisper+subtitle"
    return "whisper"
