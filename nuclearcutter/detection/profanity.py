"""
Foul-language flagging: wordlist match against the transcript, followed by
an always-on LLM context check (docs/SPEC.md section 4.2). The wordlist
alone is never treated as sufficient — it's a cheap first-pass filter, and
every match it produces is then checked in context by the LLM, both to
catch things the wordlist misses (via a broader sweep prompt over each
utterance) and to reject false positives (e.g. non-profane homophones or
non-profane usage of a flagged word).
"""

from __future__ import annotations

import re
from pathlib import Path

from nuclearcutter.detection.transcribe import Utterance, Word
from nuclearcutter.schema import LanguageDetection, SeverityLevel
from nuclearcutter.utils.llm_client import LLMClient

DEFAULT_WORDLIST_PATH = Path(__file__).parent / "data" / "profanity_wordlist.txt"

# Fixed severity scale for foul language. low = WORST (severe slurs only),
# exhigh = anything rude/mean/disrespectful (basically everything). This is the
# standardized, shareable definition — the LLM context check assigns each
# confirmed word to one of these levels.
DEFAULT_FOUL_LANGUAGE_SCALE = (
    "- low: ONLY the strongest words — severe slurs and the most offensive expletives. "
    "Ignore mild or common profanity.\n"
    "- med: profanity and crude words such as fuck, shit, bitch, asshole, bastard, "
    "damn, hell, dick, pussy, cunt, and their variants. Ignore mild words (crap, "
    "darn) and non-profane uses (e.g. \"hell\" as a place).\n"
    "- high: ANY crude or offensive word, including mild ones like crap, darn, sucks, "
    "and euphemisms for profanity, in addition to common profanity and slurs. "
    "Be aggressive about flagging.\n"
    "- exhigh: ANYTHING rude, mean, disrespectful, or unkind — including 'omg', "
    "'I hate you', 'shut up', 'idiot', sarcasm, mockery, or any harsh or dismissive "
    "tone — in addition to all profanity and crude words. Basically any line that "
    "isn't purely polite should be flagged; err on the side of flagging."
)

CONTEXT_CHECK_PROMPT = """You are helping a content-filtering tool review a line of movie \
dialogue for foul language.

Here is the FULL description of each level for foul language:

Note: despite the name, "low" is the STRICTEST definition — it only matches the \
single worst/most offensive words (severe slurs). "exhigh" is the LOOSEST \
definition — it matches almost anything rude/mean/disrespectful. Going from \
low → med → high → exhigh, each definition gets broader and easier to satisfy. \
It's expected and correct for a word to match multiple levels at once — \
evaluate each one independently and mark it true if the word fits, regardless \
of what you answered for the others.

The level definitions (in order, low → exhigh):
{definition}

Line: "{text}"

Wordlist matches found in this line (may be false positives — e.g. homophones or non-profane \
usage): {matches}

For EACH wordlist match, decide if it is genuinely foul language as used in this \
specific line, or a false positive. Also flag any OTHER foul words in the line that \
the wordlist missed.

Respond with ONLY a JSON object with this exact structure:
{{
  "confirmed_words": ["word1", "word2"],
  "matches": {{
    "word1": {{"matches_low": false, "matches_med": true, "matches_high": true, "matches_exhigh": true}},
    "word2": {{"matches_low": false, "matches_med": false, "matches_high": false, "matches_exhigh": true}}
  }},
  "reasoning": "one short sentence explaining any false-positive rejections or additions"
}}

confirmed_words should be the exact word(s) as they appear in the line, lowercase, that \
should be flagged as foul language. Empty list if none. matches maps each confirmed word \
to four independent booleans: does the word fit that level's definition above? Mark each \
independently — a word can match multiple levels."""


def load_wordlist(path: Path = DEFAULT_WORDLIST_PATH) -> set[str]:
    if not path.exists():
        return set()
    return {
        line.strip().lower()
        for line in path.read_text().splitlines()
        if line.strip() and not line.startswith("#")
    }


def wordlist_matches(text: str, wordlist: set[str]) -> list[str]:
    """Return the wordlist words that appear in the given text (whole-word match)."""
    tokens = re.findall(r"[a-zA-Z']+", text.lower())
    return [t for t in tokens if t in wordlist]


def detect_foul_language(
    utterances: list[Utterance],
    client: LLMClient,
    wordlist: set[str] = None,
    subtitle_utterances: list[Utterance] = None,
    foul_language_prompt: str | None = None,
) -> list[LanguageDetection]:
    """Run wordlist + LLM context check over all utterances. Returns confirmed detections."""
    wordlist = wordlist if wordlist is not None else load_wordlist()
    subtitle_utterances = subtitle_utterances or []
    definition = foul_language_prompt or DEFAULT_FOUL_LANGUAGE_SCALE

    detections: list[LanguageDetection] = []

    for utt in utterances:
        matches = wordlist_matches(utt.text, wordlist)
        if not matches:
            continue

        try:
            result = client.text_query_json(
                CONTEXT_CHECK_PROMPT.format(definition=definition, text=utt.text, matches=matches)
            )
            llm_failed = False
        except Exception:
            # If the LLM call fails, fall back to trusting the wordlist match
            # rather than silently dropping a possible detection.
            result = {"confirmed_words": matches, "reasoning": "llm_check_failed_fallback_to_wordlist"}
            llm_failed = True

        confirmed = [w.lower() for w in result.get("confirmed_words", [])]
        if not confirmed:
            continue

        # Per-word severity from the LLM's four independent match booleans.
        matches_map = result.get("matches") or {}
        source = _cross_check_source(utt, subtitle_utterances)

        for confirmed_word in confirmed:
            word_obj = _find_word_timing(utt, confirmed_word)
            if word_obj:
                start, end = word_obj.start, word_obj.end
            else:
                # No word-level timing (e.g. detection came purely from subtitle
                # text without a matching whisper word) — fall back to the
                # whole utterance span for this word's mute window.
                start, end = utt.start, utt.end

            if llm_failed:
                # Confirm call failed entirely — grade worst (LOW) so the word
                # is never missed regardless of the user's threshold.
                level = SeverityLevel.LOW
            else:
                # First (narrowest/strictest) true match wins; if the word was
                # confirmed but has no positive match data, default to exhigh
                # (weakest defensible claim — only max-filtering users catch it).
                level = _select_word_level(matches_map.get(confirmed_word))

            detections.append(LanguageDetection(
                start=start,
                end=end,
                utterance_start=utt.start,
                utterance_end=utt.end,
                word=confirmed_word,
                transcript_source=source,
                llm_confirmed=True,
                llm_reasoning=result.get("reasoning"),
                level=level,
            ))

    return detections


def _select_word_level(matches: dict | None) -> SeverityLevel:
    """Pick a confirmed word's level from its four independent match booleans.
    First (narrowest/strictest) true match wins. If the word was confirmed but
    has no positive match data, fall back to exhigh (the weakest defensible
    claim — only corrected by max-filtering users)."""
    if matches:
        for lv, key in [
            (SeverityLevel.LOW, "matches_low"),
            (SeverityLevel.MED, "matches_med"),
            (SeverityLevel.HIGH, "matches_high"),
            (SeverityLevel.EXHIGH, "matches_exhigh"),
        ]:
            if matches.get(key):
                return lv
    return SeverityLevel.EXHIGH


def _find_word_timing(utt: Utterance, target_word: str) -> Word | None:
    target_clean = re.sub(r"[^a-z']", "", target_word.lower())
    for w in utt.words:
        if re.sub(r"[^a-z']", "", w.text.lower()) == target_clean:
            return w
    return None


def _cross_check_source(utt: Utterance, subtitle_utterances: list[Utterance]) -> str:
    for sub in subtitle_utterances:
        if sub.start < utt.end and sub.end > utt.start:
            return "whisper+subtitle"
    return "whisper"
