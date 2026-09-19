"""Tests for sankhya.triage_issue against the shipped int8 weights."""
import json
import os
import sys

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from sankhya.triage_issue import DEFAULT_WEIGHTS, triage

pytestmark = pytest.mark.skipif(
    not os.path.exists(DEFAULT_WEIGHTS),
    reason=f"shipped int8 weights not found: {DEFAULT_WEIGHTS}",
)


def test_reproducing_positive_report():
    # the model correctly returns 125000 for this text; claiming a
    # different correct value should reproduce as a mismatch.
    result = triage(
        text="sava lakh ka phone",
        lang="hi_latn",
        expect_value=999999,
    )
    assert result["reproduces"] is True
    assert result["gold_file"] == "gold.jsonl"
    assert result["gold_line"] is not None
    line = json.loads(result["gold_line"])
    assert line["text"] == "sava lakh ka phone"
    assert line["lang"] == "hi_latn"
    assert len(line["spans"]) == 1
    assert line["spans"][0]["value"] == 999999
    assert line["spans"][0]["start"] == 0
    assert result["text"][line["spans"][0]["start"]:line["spans"][0]["end"]] == "sava lakh"


def test_non_reproducing_report():
    # the model already returns the value the reporter says is correct, so
    # this should not reproduce and should emit no gold line.
    result = triage(
        text="sava lakh ka phone",
        lang="hi_latn",
        expect_value=125000,
    )
    assert result["reproduces"] is False
    assert result["gold_line"] is None


def test_negative_report_reproduces():
    # the model currently returns a span for this text, but the reporter
    # says it's a false positive -- should reproduce, with an empty-spans
    # gold line.
    result = triage(
        text="5 crore log dekh rahe honge",
        lang="hi_latn",
        negative=True,
    )
    assert result["reproduces"] is True
    assert result["gold_line"] is not None
    line = json.loads(result["gold_line"])
    assert line["spans"] == []


def test_negative_report_does_not_reproduce():
    # the model already returns no spans, so a false-positive report
    # against this text does not reproduce.
    result = triage(
        text="aaj mausam accha hai",
        lang="hi_latn",
        negative=True,
    )
    assert result["reproduces"] is False
    assert result["gold_line"] is None


def test_expect_range():
    result = triage(
        text="2-3 lakh mein aa jayega",
        lang="hi_latn",
        expect_range=(200000.0, 300000.0),
    )
    # whatever the model currently returns, the function should not raise
    # and should produce a valid gold line only when it reproduces.
    if result["reproduces"]:
        line = json.loads(result["gold_line"])
        assert line["spans"][0]["range"] == [200000.0, 300000.0]
    else:
        assert result["gold_line"] is None
