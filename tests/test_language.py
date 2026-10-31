"""The language pass: wordlist hit -> contextual check -> descriptive tags.

v1 ran the same transcript through an LLM and asked it to pick a severity level
from a hand-written foul-language scale, which filed a racial slur and "damn" in
the same bucket whenever the scale's prose said so. The pass under test instead
asks *which things are in this line*, so a viewer can filter slurs and keep mild
swearing — the split that motivated the redesign.

Two sub-passes:
  word pass     wordlist hit, then one LLM call per matching line for context
  dialogue sweep  every line, in groups, for what a wordlist cannot see (slurs
                  spelled around, threats, sexual remarks, insults)
"""

from __future__ import annotations

import pytest

from nuclearcutter.detection.language import (
    detect_language, load_wordlist, merge_language_detections, wordlist_matches,
)
from nuclearcutter.detection.transcribe import Utterance, Word
from nuclearcutter.taxonomy import Family, tags_in_family


class FakeClient:
    """Answers text queries from a queue or callable, recording the prompts."""

    def __init__(self, responses):
        if isinstance(responses, Exception):
            responses = _AlwaysRaises(responses)
        self.responses = responses if callable(responses) else list(responses)
        self.prompts: list[str] = []
        self.calls = 0

    def _next(self, prompt):
        self.calls += 1
        self.prompts.append(prompt)
        if callable(self.responses):
            return self.responses(prompt)
        if not self.responses:
            raise AssertionError("FakeClient ran out of responses")
        reply = self.responses.pop(0)
        if isinstance(reply, Exception):
            raise reply
        return reply

    def text_query_json(self, prompt):
        return self._next(prompt)

    def vision_query_json(self, prompt, frame_paths, max_pixels=None):
        raise AssertionError("language pass must not send frames")


class _AlwaysRaises:
    """Callable that raises the same exception for every call (a dead server)."""

    def __init__(self, exc: Exception):
        self.exc = exc

    def __call__(self, prompt):
        raise self.exc


def _utt(text, start=0.0, end=2.0, words=None) -> Utterance:
    if words is None:
        # Word timings spread evenly over the utterance, like whisper's output.
        parts = text.split()
        step = (end - start) / max(len(parts), 1)
        words = [Word(w, start + i * step, start + (i + 1) * step)
                 for i, w in enumerate(parts)]
    return Utterance(text=text, start=start, end=end, words=words)


# --------------------------------------------------------------------------
# Wordlist
# --------------------------------------------------------------------------

def test_wordlist_loads_and_is_normalised():
    words = load_wordlist()
    # The file is the tool's first line of defence; 22 entries could not cover
    # even the common words, let alone the slurs the redesign exists to split out.
    assert len(words) > 100, len(words)
    assert all(w == w.lower().strip() for w in words)
    assert "fuck" in words and "damn" in words
    assert any(" " in w for w in words), "phrase entries (pain in the ass) should work"


def test_wordlist_read_is_forgiving_of_hand_edits(tmp_path):
    f = tmp_path / "wl.txt"
    f.write_text("# comment\n  FUCK \ndamn\ndamn\n\nshit  # mild\n")
    assert load_wordlist(f) == {"fuck", "damn", "shit"}


def test_missing_wordlist_file_is_not_fatal(tmp_path):
    # The dialogue sweep can still work without any list at all.
    assert load_wordlist(tmp_path / "nope.txt") == set()


def test_wordlist_matches_whole_words_only():
    words = {"damn", "shit"}
    assert wordlist_matches("oh damn it", words) == ["damn"]
    # The classic embedding false positive: "class" contains "ass".
    assert wordlist_matches("what a classic class", words) == []
    assert wordlist_matches("SHIT happens", words) == ["shit"]


def test_wordlist_matches_punctuation_and_is_distinct_per_line():
    words = {"damn"}
    # Three occurrences, ONE entry: the LLM is asked about the word once, and
    # every occurrence still gets its own mute window (see _word_timings).
    assert wordlist_matches("damn! Damn... damn", words) == ["damn"]
    assert wordlist_matches("no hits here", words) == []


def test_wordlist_supports_phrase_entries():
    words = {"pain in the ass"}
    assert wordlist_matches("you are a pain in the ass to me", words) == \
        ["pain in the ass"]
    assert wordlist_matches("the pain in the assumption", words) == []


def test_wordlist_does_not_match_substrings_inside_words():
    assert wordlist_matches("he was hammered by the news", {"damn"}) == []
    assert wordlist_matches("this is assinine", {"ass"}) == []


# --------------------------------------------------------------------------
# Word pass: context decides the tag
# --------------------------------------------------------------------------

def test_word_hit_becomes_a_tagged_detection():
    client = FakeClient([{"matches": {"damn": ["language.profanity"]},
                          "line_tags": [], "reasoning": "mild"}])
    dets = detect_language([_utt("oh damn it", 10.0, 12.0)], client,
                           wordlist={"damn"}, run_dialogue_sweep=False)
    assert len(dets) == 1
    assert dets[0].tags == ["language.profanity"]
    assert dets[0].text == "damn"
    assert dets[0].llm_confirmed is True


def test_word_timing_is_the_word_not_the_whole_line():
    client = FakeClient([{"matches": {"damn": ["language.profanity"]}}])
    dets = detect_language([_utt("oh damn it", 10.0, 14.0)], client,
                           wordlist={"damn"}, run_dialogue_sweep=False)
    d = dets[0]
    # Muting the whole utterance because one word was said is the complaint
    # people have about tools like this; word timing is the answer.
    assert d.end - d.start < (d.utterance_end - d.utterance_start)
    assert d.utterance_start == 10.0 and d.utterance_end == 14.0


def test_model_rejecting_a_word_drops_it():
    # "Hell" in "go to hell" vs "what the hell" is exactly why the wordlist alone
    # is not allowed to decide. No entry for the word == the model said no.
    client = FakeClient([{"matches": {}, "line_tags": []}])
    dets = detect_language([_utt("what the hell")], client, wordlist={"hell"},
                           run_dialogue_sweep=False)
    assert dets == []


def test_one_line_can_carry_two_different_tags():
    client = FakeClient([{"matches": {"damn": ["language.profanity"],
                                      "bastard": ["language.cruel_tone"]}}])
    # Far apart in time, so the merge step keeps them as two decisions.
    dets = detect_language([_utt("you damn idiot over there you bastard",
                                 0.0, 12.0)], client,
                           wordlist={"damn", "bastard"}, run_dialogue_sweep=False)
    by_text = {d.text: d.tags for d in dets}
    assert by_text["damn"] == ["language.profanity"]
    assert by_text["bastard"] == ["language.cruel_tone"]


def test_a_slur_is_tagged_as_a_slur_not_as_profanity():
    # The whole point: these two must be separable in the chooser.
    client = FakeClient([{"matches": {"nword": ["language.slur"]}},
                         {"matches": {"damn": ["language.profanity"]}}])
    # Different times on purpose: two words in the same second are one moment,
    # and the merge step would union them into a single marker.
    dets = detect_language([_utt("nword", 1.0, 2.0), _utt("damn", 30.0, 31.0)], client,
                           wordlist={"nword", "damn"}, run_dialogue_sweep=False)
    tags = {d.text: d.tags for d in dets}
    assert tags["nword"] == ["language.slur"]
    assert tags["damn"] == ["language.profanity"]


def test_line_level_tags_attach_to_the_lines_that_earned_them():
    # The word pass only reads line_tags for lines the wordlist hit; a line with
    # no listed word is the dialogue sweep's job (tested below). Here: a hit line
    # whose LINE is also offensive keeps both.
    client = FakeClient([{"matches": {"damn": ["language.profanity"]},
                          "line_tags": ["language.sexual_talk"]}])
    dets = detect_language(
        [_utt("damn, tell me what you want me to do to you", 5.0, 11.0)], client,
        wordlist={"damn"}, run_dialogue_sweep=False)
    tags = {t for d in dets for t in d.tags}
    assert "language.profanity" in tags
    assert "language.sexual_talk" in tags


def test_llm_failure_trusts_the_wordlist_but_not_a_category():
    # A dead server must not turn into the strongest accusation, and must not
    # silently delete a real hit either.
    client = FakeClient(RuntimeError("server down"))
    dets = detect_language([_utt("oh damn it")], client, wordlist={"damn"},
                           run_dialogue_sweep=False)
    assert len(dets) == 1
    assert dets[0].tags == ["language.unclear"]
    assert dets[0].llm_confirmed is False
    assert "llm_check_failed" in (dets[0].llm_reasoning or "")


def test_garbage_reply_still_leaves_a_censorable_detection():
    client = FakeClient([{"matches": {"damn": "language.profanity"}}])  # str not list
    dets = detect_language([_utt("oh damn it")], client, wordlist={"damn"},
                           run_dialogue_sweep=False)
    assert dets and dets[0].tags == ["language.profanity"]

    client2 = FakeClient([{"matches": "not a dict"}])
    dets2 = detect_language([_utt("oh damn it")], client2, wordlist={"damn"},
                            run_dialogue_sweep=False)
    assert dets2 == [] or all(d.tags for d in dets2)  # never a tagless detection


def test_invented_category_from_the_model_folds_to_unclear():
    client = FakeClient([{"matches": {"damn": ["language.swearing_light"]}}])
    dets = detect_language([_utt("oh damn it")], client, wordlist={"damn"},
                           run_dialogue_sweep=False)
    assert dets[0].tags == ["language.unclear"]


# --------------------------------------------------------------------------
# Cross-checking the subtitle file
# --------------------------------------------------------------------------

def test_subtitle_cross_check_is_recorded_in_the_source():
    subs = [_utt("oh damn it", 10.0, 12.0)]
    client = FakeClient([{"matches": {"damn": ["language.profanity"]}}])
    dets = detect_language([_utt("oh damn it", 10.0, 12.0)], client,
                           wordlist={"damn"}, subtitle_utterances=subs,
                           run_dialogue_sweep=False)
    assert "+" in dets[0].transcript_source  # whisper + subtitle agreed


def test_whisper_only_when_no_subtitle_matches():
    client = FakeClient([{"matches": {"damn": ["language.profanity"]}}])
    dets = detect_language([_utt("oh damn it", 10.0, 12.0)], client,
                           wordlist={"damn"}, subtitle_utterances=[],
                           run_dialogue_sweep=False)
    assert dets[0].transcript_source == "whisper"


# --------------------------------------------------------------------------
# Dialogue sweep
# --------------------------------------------------------------------------

def test_dialogue_sweep_catches_what_a_wordlist_cannot():
    # No listed word, but a threat. Group replies carry per-line indices so one
    # call covers a group of lines.
    def reply(prompt):
        return {"lines": [{"index": 0, "tags": ["language.cruel_tone"],
                           "text": "you are worthless to everyone"}]}

    client = FakeClient(reply)
    dets = detect_language([_utt("you are worthless to everyone", 3.0, 5.0)], client,
                           wordlist=set(), run_dialogue_sweep=True)
    assert dets
    assert "language.cruel_tone" in dets[0].tags
    # `transcript_source` records where the WORDS came from (whisper / subtitle),
    # not which pass found them; the pass is in llm_reasoning.
    assert dets[0].transcript_source == "whisper"
    assert dets[0].llm_reasoning == "dialogue sweep"


def test_dialogue_sweep_can_be_skipped():
    client = FakeClient([])
    dets = detect_language([_utt("nothing here")], client, wordlist=set(),
                           run_dialogue_sweep=False)
    assert dets == []
    assert client.calls == 0


def test_dialogue_sweep_only_reports_tags_it_was_asked_about():
    # A model that invents a visual tag in the language pass must not smuggle it
    # in — this pass has no business classifying pictures.
    def reply(prompt):
        return {"lines": [{"index": 0, "tags": ["nudity.full_nude_female",
                                                "language.cruel_tone"]}]}

    client = FakeClient(reply)
    dets = detect_language([_utt("you are worthless")], client, wordlist=set(),
                           run_dialogue_sweep=True)
    assert dets
    assert all(not t.startswith("nudity.") for t in dets[0].tags)
    assert "language.cruel_tone" in dets[0].tags


def test_dialogue_sweep_reports_progress():
    utts = [_utt(f"line {i}", float(i), float(i) + 0.5) for i in range(30)]
    client = FakeClient(lambda p: {"lines": []})
    seen: list[tuple[int, int]] = []
    detect_language(utts, client, wordlist=set(), run_dialogue_sweep=True,
                    on_progress=lambda d, t: seen.append((d, t)))
    assert seen, "the dialogue sweep is the slow language pass; it must report"
    assert seen[-1][0] >= seen[-1][1] or seen[-1][0] == len(utts)


def test_dialogue_sweep_failure_does_not_lose_the_word_pass():
    client = FakeClient(RuntimeError("server down"))
    dets = detect_language([_utt("oh damn it")], client, wordlist={"damn"},
                           run_dialogue_sweep=True)
    assert any(d.text == "damn" for d in dets)


def test_dialogue_line_index_out_of_range_is_ignored():
    def reply(prompt):
        return {"lines": [{"index": 99, "tags": ["language.profanity"]}]}

    client = FakeClient(reply)
    dets = detect_language([_utt("a sentence", 1.0, 2.0)], client, wordlist=set(),
                           run_dialogue_sweep=True)
    assert dets == []


# --------------------------------------------------------------------------
# Merging
# --------------------------------------------------------------------------

def test_every_occurrence_of_a_word_gets_its_own_window():
    client = FakeClient([{"matches": {"damn": ["language.profanity"]}}])
    dets = detect_language([_utt("damn it and also damn again later", 0.0, 12.0)],
                           client, wordlist={"damn"}, run_dialogue_sweep=False)
    # v1 timed only the first occurrence, so the second one stayed audible.
    assert len(dets) == 2
    assert dets[0].start < dets[1].start


def test_merge_language_detections_unions_tags_of_the_same_window():
    from nuclearcutter.schema import LanguageDetection

    a = LanguageDetection(start=5.0, end=5.4, utterance_start=4.0, utterance_end=7.0,
                          text="damn", tags=["language.profanity"])
    b = LanguageDetection(start=5.0, end=5.4, utterance_start=4.0, utterance_end=7.0,
                          text="damn", tags=["language.slur"])
    out = merge_language_detections([a, b])
    assert len(out) == 1
    assert set(out[0].tags) == {"language.profanity", "language.slur"}


def test_merge_keeps_distant_detections_apart():
    from nuclearcutter.schema import LanguageDetection

    a = LanguageDetection(start=5.0, end=5.4, utterance_start=4.0, utterance_end=7.0,
                          text="damn", tags=["language.profanity"])
    b = LanguageDetection(start=40.0, end=40.4, utterance_start=39.0,
                          utterance_end=42.0, text="damn",
                          tags=["language.profanity"])
    assert len(merge_language_detections([a, b])) == 2


def test_merge_does_not_union_different_words_into_one_text():
    from nuclearcutter.schema import LanguageDetection

    a = LanguageDetection(start=5.0, end=5.4, utterance_start=4.0, utterance_end=9.0,
                          text="damn", tags=["language.profanity"])
    b = LanguageDetection(start=8.0, end=8.4, utterance_start=4.0, utterance_end=9.0,
                          text="bastard", tags=["language.cruel_tone"])
    out = merge_language_detections([a, b])
    # Whether they merge or not, both tags and both words must survive: losing a
    # word means losing the mute window for it.
    all_tags = {t for d in out for t in d.tags}
    assert {"language.profanity", "language.cruel_tone"} <= all_tags


# --------------------------------------------------------------------------
# Vocabulary shape
# --------------------------------------------------------------------------

def test_language_tags_are_all_audio_and_none_are_visual():
    for tag in tags_in_family(Family.LANGUAGE):
        assert tag.audio and not tag.visual, tag.id


def test_substance_family_is_visual_not_audio():
    # Smoking/drugs are something you SEE; the mute belongs to language.
    for tag in tags_in_family(Family.SUBSTANCE):
        assert tag.visual, tag.id


def test_prompt_lists_every_language_tag():
    from nuclearcutter.detection.language import language_categories_prompt

    block = language_categories_prompt()
    for tag in tags_in_family(Family.LANGUAGE) + tags_in_family(Family.SUBSTANCE):
        assert tag.id in block, tag.id


def test_prompt_can_exclude_tags_already_covered_by_the_wordlist():
    from nuclearcutter.detection.language import language_categories_prompt

    block = language_categories_prompt(exclude_ids={"language.profanity"})
    # Checked line by line, not as a substring: `language.profanity` is a prefix of
    # `language.profanity_severe`, and since the merge the two sit next to each other,
    # so a substring assert fails for a reason that has nothing to do with the
    # exclusion actually working.
    offered = {ln.split(":", 1)[0].strip().lstrip("- ")
               for ln in block.splitlines() if ln.strip().startswith("- language.")}
    assert "language.profanity" not in offered
    assert "language.profanity_severe" in offered, "the wordlist cannot catch the worst tier"
    assert "language.slur" in offered
