from nuclearcutter.render.skip_card import (
    MAX_CARD_SECONDS, MIN_CARD_SECONDS, _wrap_for_display, card_duration_for_text,
)


def test_short_description_gets_minimum_duration():
    duration = card_duration_for_text("Brief nudity.")
    assert duration == MIN_CARD_SECONDS


def test_empty_description_gets_minimum_duration():
    assert card_duration_for_text("") == MIN_CARD_SECONDS


def test_long_description_scales_with_word_count():
    short = card_duration_for_text("A character walks by.")
    long = card_duration_for_text(" ".join(["word"] * 30))
    assert long > short


def test_duration_capped_at_maximum():
    very_long = " ".join(["word"] * 500)
    assert card_duration_for_text(very_long) == MAX_CARD_SECONDS


def test_duration_never_below_minimum():
    assert card_duration_for_text("Hi.") >= MIN_CARD_SECONDS


def test_wrap_for_display_breaks_long_lines():
    text = " ".join(["word"] * 20)
    wrapped = _wrap_for_display(text, max_chars_per_line=20)
    for line in wrapped.split("\n"):
        assert len(line) <= 25  # allow a little slack for the last word pushing over
