"""Offline tests for GSM8K answer extraction (no backend / datasets needed)."""

from dlm_explorer.eval.tasks import Gsm8kTask


def test_extract_gold():
    ans = "Janet sells 16 - 3 - 4 = 9 eggs.\nShe makes 9 * 2 = 18.\n#### 18"
    assert Gsm8kTask._extract_gold(ans) == "18"
    assert Gsm8kTask._extract_gold("no marker here") == ""
    assert Gsm8kTask._extract_gold("#### 1,234") == "1234"


def test_extract_pred_handles_marker_and_runon():
    # "#### N" format
    assert Gsm8kTask._extract_pred(" reasoning ... #### 42") == "42"
    # bare final number
    assert Gsm8kTask._extract_pred("so the answer is 7") == "7"
    # run-on into a hallucinated next question is cut off
    assert Gsm8kTask._extract_pred(" the answer is 5\n\nQuestion: ... 999") == "5"
    # commas stripped
    assert Gsm8kTask._extract_pred("total = 1,000") == "1000"
    # nothing numeric
    assert Gsm8kTask._extract_pred("no number") == ""
