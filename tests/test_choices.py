"""Tests for dialogue-option detection.

Every case here was a real screen or a real failure. The impostors matter more than the
happy path: a false positive means the player presses "continue" onto an answer and the
game makes an unrecoverable choice for them.
"""

from __future__ import annotations

import pytest

from adapters.game_hsr.choices import Option, find_options
from adapters.game_hsr.perception import TextBox


def box(text, x, y, w, h, confidence=0.9):
    return TextBox(text=text, confidence=confidence, bbox=[x, y, w, h])


def stacked(texts, *, x=430, y=300, gap=70, w=300, h=24):
    return [box(text, x, y + i * gap, w, h) for i, text in enumerate(texts)]


def test_three_options_are_found_in_order():
    boxes = stacked(["Fine, let's go.", "Not interested.", "What's in it for me?"])
    options = find_options(boxes)
    assert [option.text for option in options] == [
        "Fine, let's go.", "Not interested.", "What's in it for me?"
    ]


def test_options_come_back_top_to_bottom_even_when_ocr_is_shuffled():
    boxes = stacked(["Second", "Third", "First"])
    boxes = [boxes[2], boxes[0], boxes[1]]
    options = find_options(boxes)
    assert [option.bbox[1] for option in options] == sorted(o.bbox[1] for o in options)


def test_two_options_are_enough():
    assert len(find_options(stacked(["Yes.", "No."]))) == 2


def test_single_line_is_not_a_choice():
    assert find_options([box("Fine, let's go.", 430, 300, 300, 24)]) == []


def test_name_plate_above_a_subtitle_is_not_a_choice():
    # The live false positive: "Kafka" at h=33 sits 78 px above a wrapped subtitle line
    # at h=20. Two boxes, aligned left, gap in range -- only the height gives it away.
    boxes = [
        box("Kafka", 595, 506, 89, 33),
        box("destination?", 575, 584, 130, 20),
    ]
    assert find_options(boxes) == []


def test_wrapped_subtitle_lines_are_not_a_choice():
    # One sentence, two lines, ~28 px apart.
    boxes = [box("Cleaning up other people's mess isn't", 232, 590, 700, 26),
             box("in my job description...", 232, 618, 300, 26)]
    assert find_options(boxes) == []


def test_text_inside_the_subtitle_band_is_never_an_option():
    boxes = [box("Meet up with Silver Wolf and leave the", 60, 560, 380, 24),
             box("Reception Center", 60, 630, 220, 24)]
    assert find_options(boxes) == []


def test_quest_tracker_lines_outside_the_region_are_ignored():
    boxes = [box("Chaos In the Deep", 50, 240, 220, 24),
             box("Go to mission target", 60, 320, 200, 24)]
    assert find_options(boxes) == []


def test_options_must_share_a_left_edge():
    boxes = [box("First answer", 430, 300, 260, 24),
             box("Second answer", 760, 370, 260, 24)]
    assert find_options(boxes) == []


def test_gap_that_is_too_wide_is_not_a_stack():
    boxes = stacked(["First answer", "Second answer"], gap=200)
    assert find_options(boxes) == []


def test_crop_offset_is_applied_to_reported_coordinates():
    cropped = [box("First answer", 60, 40, 260, 24), box("Second answer", 60, 110, 260, 24)]
    options = find_options(cropped, crop_offset=(400, 260), region=None)
    assert options[0].bbox[:2] == [460, 300]


def test_option_centre_is_the_tap_point():
    option = Option(text="Fine, let's go.", bbox=[430, 300, 300, 24], confidence=0.9)
    assert option.centre == (580, 312)
