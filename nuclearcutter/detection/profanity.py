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
from nuclearcutter.schema import LanguageDetection
from nuclearcutter.utils.llm_client import LLMClient

DEFAULT_WORDLIST_PATH = Path(__file__).parent / "data" / "profanity_wordlist.txt"

CONTEXT_CHECK_PROMPT = """You are helping a content-filtering tool review a line of movie \
dialogue for profanity/foul language.

Line: "{text}"

Wordlist matches found in this line (may be false positives — e.g. homophones or non-profane \
usage): {matches}

For EACH wordlist match, decide if it is genuinely profane/foul language as used in this \
specific line, or a false positive. Also flag any OTHER profane words in the line that \
the wordlist missed.

Respond with ONLY a JSON object with this exact structure:
{{
  "confirmed_words": ["word1", "word2"],
  "reasoning": "one short sentence explaining any false-positive rejections or additions"
}}

confirmed_words should be the exact word(s) as they appear in the line, lowercase, that \
should be flagged as foul language. Empty list if none."""


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
) -> list[LanguageDetection]:
    """Run wordlist + LLM context check over all utterances. Returns confirmed detections."""
    wordlist = wordlist if wordlist is not None else load_wordlist()
    subtitle_utterances = subtitle_utterances or []

    detections: list[LanguageDetection] = []

    for utt in utterances:
        matches = wordlist_matches(utt.text, wordlist)
        if not matches:
            continue

        try:
            result = client.text_query_json(
                CONTEXT_CHECK_PROMPT.format(text=utt.text, matches=matches)
            )
        except Exception:
            # If the LLM call fails, fall back to trusting the wordlist match
            # rather than silently dropping a possible detection.
            result = {"confirmed_words": matches, "reasoning": "llm_check_failed_fallback_to_wordlist"}

        confirmed = [w.lower() for w in result.get("confirmed_words", [])]
        if not confirmed:
            continue

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

            detections.append(LanguageDetection(
                start=start,
                end=end,
                utterance_start=utt.start,
                utterance_end=utt.end,
                word=confirmed_word,
                transcript_source=source,
                llm_confirmed=True,
                llm_reasoning=result.get("reasoning"),
            ))

    return detections


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
