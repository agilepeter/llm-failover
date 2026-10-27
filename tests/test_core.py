"""Tests for LLMFailover's handling of answers that are no use and of payment failures.

Found 2026-10-27 in the production chain this library twins: an EMPTY answer counted as success
(only None did not), so a provider that returned "" ended the chain while others were untried;
gpt-oss on Groq does this, as providers.groq() warns. And a 402 "prepayment credits are depleted"
from Gemini was retried as a rate limit on every call, because its body says RESOURCE_EXHAUSTED,
the same words as its 429.

Pinned here: an empty answer, or one accept() rejects, fails over to the next provider; a 402 is
never retried and the provider is skipped for the session; a 429 is still retried; with no usable
answer anywhere generate() still raises RuntimeError and fires on_blackout once.
"""
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import llm_failover_chain.core as core  # noqa: E402
from llm_failover_chain import LLMFailover  # noqa: E402

GEMINI_402 = ("402 RESOURCE_EXHAUSTED. {'error': {'code': 402, 'message': 'Your prepayment credits "
              "are depleted.', 'status': 'RESOURCE_EXHAUSTED'}}")
GEMINI_429 = "429 RESOURCE_EXHAUSTED. {'error': {'code': 429, 'message': 'Resource has been exhausted.'}}"


@pytest.fixture(autouse=True)
def no_sleep(monkeypatch):
    monkeypatch.setattr(core.time, "sleep", lambda s: None)


def scripted(calls, name, *answers):
    answers = list(answers)

    def fn(prompt, max_tokens):
        calls.append(name)
        a = answers.pop(0)
        if isinstance(a, Exception):
            raise a
        return a
    return (name, fn)


def test_an_empty_answer_fails_over_to_the_next_provider():
    calls = []
    f = LLMFailover([scripted(calls, "Groq", ""), scripted(calls, "Gemini", "answer")])
    assert f.generate("p") == "answer" and calls == ["Groq", "Gemini"]


def test_a_blank_or_none_answer_fails_over_too():
    calls = []
    f = LLMFailover([scripted(calls, "A", "  \n"), scripted(calls, "B", None), scripted(calls, "C", "answer")])
    assert f.generate("p") == "answer" and calls == ["A", "B", "C"]


def test_an_answer_accept_rejects_fails_over():
    calls = []
    f = LLMFailover([scripted(calls, "A", "prose"), scripted(calls, "B", '["json"]')])
    assert f.generate("p", accept=lambda t: t.startswith("[")) == '["json"]' and calls == ["A", "B"]


def test_without_accept_any_non_empty_answer_is_returned_as_before():
    calls = []
    f = LLMFailover([scripted(calls, "A", "prose"), scripted(calls, "B", "unused")])
    assert f.generate("p") == "prose" and calls == ["A"]


def test_a_rejected_or_empty_answer_does_not_count_as_a_successful_call():
    calls = []
    f = LLMFailover([scripted(calls, "A", ""), scripted(calls, "B", "answer")])
    f.generate("p")
    assert f.call_counts == {"B": 1}


def test_a_402_is_not_retried_and_the_provider_is_skipped_for_the_session():
    calls = []
    f = LLMFailover([scripted(calls, "Gemini", Exception(GEMINI_402)), scripted(calls, "B", "one", "two")])
    assert f.generate("p") == "one"
    assert calls == ["Gemini", "B"], "a payment failure must not be retried like a rate limit"
    assert "Gemini" in f.exhausted
    assert f.generate("p") == "two" and calls == ["Gemini", "B", "B"]


def test_a_429_is_still_retried():
    calls = []
    f = LLMFailover([scripted(calls, "Gemini", Exception(GEMINI_429), "answer")])
    assert f.generate("p") == "answer" and calls == ["Gemini", "Gemini"]


def test_with_no_usable_answer_anywhere_it_raises_and_fires_the_blackout_once():
    fired = []
    calls = []
    f = LLMFailover([scripted(calls, "A", "", ""), scripted(calls, "B", "nope", "nope")],
                    on_blackout=lambda chain, exhausted: fired.append(chain))
    for _ in range(2):
        with pytest.raises(RuntimeError, match="All providers failed"):
            f.generate("p", accept=lambda t: False)
    assert fired == ["default"]


@pytest.mark.parametrize("err, expected", [
    (GEMINI_402, True),
    ("402 Client Error: Payment Required for url: https://router.huggingface.co/v1/chat/completions", True),
    (GEMINI_429, False),
    ("timeout after 120s at line 1402", False),
])
def test_payment_failures_are_told_apart_from_rate_limits(err, expected):
    assert core._is_payment_failure(Exception(err)) is expected


def test_the_package_no_longer_takes_the_llm_failover_import_name():
    # A different PyPI project installs as llm_failover; sharing it meant the two could not be
    # installed together (0.6.0 renamed the import to llm_failover_chain).
    assert not (ROOT / "llm_failover").exists()
    assert core.logger.name == "llm_failover_chain"
