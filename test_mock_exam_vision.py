"""Unit tests for process_mock_exam_vision.py (the model-assisted re-read for Mock exam questions
the deterministic scan already flagged needs_review -- see that module's own docstring).

Run with: python test_mock_exam_vision.py
Same plain pass/fail harness as test_mock_exam_parser.py / test_pyq_review_gate.py.

Every model response and every question fixture here is hand-built fake text, never real exam
content -- these tests inject a fake generate_fn, no real network/model call is ever made.
"""
import base64
import os
import re
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import process_mock_exam_vision as pmv  # noqa: E402

_passed = []
_failed = []


def check(label, condition, detail=""):
    if condition:
        _passed.append(label)
        print(f"  PASS: {label}")
    else:
        _failed.append(label)
        print(f"  FAIL: {label}  {detail}")


HERE = os.path.dirname(os.path.abspath(__file__))
_FAKE_IMAGE_B64 = base64.standard_b64encode(b"not-a-real-png-just-test-bytes").decode("ascii")


def _flagged_q(**overrides):
    q = {
        "question": "garbled stem text",
        "option_a": "2", "option_b": "1", "option_c": "-", "option_d": "-",
        "correct_answer": "", "question_type": "mcq", "has_diagram": False,
        "year": None, "source_tag": None, "source_page": 7,
        "needs_review": True, "review_reason": "leftover fragmented math glyphs detected",
        "subject": "Physics", "source": "mock", "mock_name": "Fake Mock",
        "mock_position": 1, "printed_number": 1,
        "crop_image_b64": _FAKE_IMAGE_B64,
        "_text_layer_hint": "A ball of mass two kg falls with options W m 2 K 4 W m 2 K 4 units given below",
    }
    q.update(overrides)
    return q


def _clean_q(printed_number=2):
    return {
        "question": "a clean question", "option_a": "a1", "option_b": "b1",
        "option_c": "c1", "option_d": "d1", "needs_review": False, "review_reason": None,
        "subject": "Physics", "mock_position": printed_number, "printed_number": printed_number,
        "source_page": 7,
    }


def _good_reply(stem="A ball of mass two kg falls with units given below",
                 a="W m 2 K 4", b="W m 2 K 4 extra", c="W m 2 K 4 third", d="W m 2 K 4 fourth"):
    return f"Q: {stem}\nA) {a}\nB) {b}\nC) {c}\nD) {d}\n"


def _usage(input_tokens=500, output_tokens=150):
    return {"input_tokens": input_tokens, "output_tokens": output_tokens}


# ---------------------------------------------------------------- cost rate mirrors server.py

def test_rate_mirrors_server():
    print("\n=== Cost rate mirrors server.py's GEMINI_FLASH_LITE_RATE ===")
    src = open(os.path.join(HERE, "server.py"), encoding="utf-8").read()
    m = re.search(r'GEMINI_FLASH_LITE_RATE\s*=\s*\{"input":\s*([\d.]+),\s*"output":\s*([\d.]+)\}', src)
    check("server rate found", bool(m))
    if m:
        check("rates equal", (float(m.group(1)), float(m.group(2))) == (pmv.GEMINI_RATE_PER_M["input"], pmv.GEMINI_RATE_PER_M["output"]))


# ---------------------------------------------------------------- _parse_model_reply

def test_parse_model_reply_basic():
    print("\n# _parse_model_reply: fixed Q:/A)/B)/C)/D) shape")
    stem, options = pmv._parse_model_reply(_good_reply())
    check("stem parsed", stem == "A ball of mass two kg falls with units given below", stem)
    check("4 options parsed", len(options) == 4, options)
    check("option A content", options.get("A") == "W m 2 K 4", options)
    check("option D content", options.get("D") == "W m 2 K 4 fourth", options)


def test_parse_model_reply_malformed_returns_fewer_options():
    print("\n# _parse_model_reply: a malformed reply (missing D) never raises, just returns fewer options")
    stem, options = pmv._parse_model_reply("Q: stem here\nA) one\nB) two\nC) three\n")
    check("stem still parsed", stem == "stem here", stem)
    check("only 3 options found, no crash", len(options) == 3, options)


# ---------------------------------------------------------------- _gate

def test_gate_wrong_option_count():
    print("\n# _gate: wrong option count fails")
    reason = pmv._gate("stem", {"A": "a", "B": "b", "C": "c"}, "hint text")
    check("flags wrong count", reason is not None and "3" in reason, reason)


def test_gate_empty_option():
    print("\n# _gate: an empty option fails")
    reason = pmv._gate("stem", {"A": "a", "B": "", "C": "c", "D": "d"}, "hint text")
    check("flags empty option B", reason is not None and "B" in reason, reason)


def test_gate_duplicate_options():
    print("\n# _gate: two identical options fail")
    reason = pmv._gate("stem", {"A": "same text", "B": "other", "C": "same text", "D": "different"}, "hint text")
    check("flags duplicate A/C", reason is not None and "A/C" in reason, reason)


def test_gate_word_mismatch():
    print("\n# _gate: wording that shares almost no words with the PDF text layer fails")
    hint = "A capacitor of capacitance five microfarad is connected across a battery"
    reason = pmv._gate("completely unrelated sentence about photosynthesis",
                        {"A": "chlorophyll", "B": "stomata", "C": "xylem", "D": "phloem"}, hint)
    check("flags word mismatch", reason is not None and "overlap" in reason, reason)


def test_gate_accepts_matching_content_despite_latex_vs_raw_notation():
    print("\n# _gate: LaTeX-wrapped model text vs. the PDF's own raw glyph hint still matches on words")
    # Realistic hint: each of the 4 options prints its OWN "W m 2 K 4" unit glyphs in the real PDF,
    # so the raw text layer repeats those tokens once per option, same as the model's reply does --
    # not once for the whole question (an earlier version of this fixture got this wrong and the
    # max-denominator ratio correctly penalized it; see _word_overlap_ratio's own docstring for why
    # the ratio is strict on purpose).
    hint = ("A ball of mass two kg falls with options W m 2 K 4 W m 2 K 4 W m 2 K 4 W m 2 K 4 "
            "units given below")
    stem = "A ball of mass two kg falls with units given below"
    options = {
        "A": "$W\\,m^{2}\\,K^{4}$", "B": "$W\\,m^{-2}\\,K^{4}$",
        "C": "$W\\,m^{2}\\,K^{-4}$", "D": "$W\\,m^{-2}\\,K^{-4}$",
    }
    reason = pmv._gate(stem, options, hint)
    check("gate passes (None)", reason is None, reason)


def test_gate_rejects_lost_delta_markers():
    print("\n# _gate: option wording matches on words, but the model dropped most delta (charge) markers")
    # Real failure mode found 2026-11-12 on the sample PDF's own Chemistry Q30 (a polarisation
    # diagram): the PDF's raw text layer carries a literal delta glyph, the model's own LaTeX
    # reply spells it "\delta" instead -- both count as the same marker here (see _DELTA_RE),
    # matched as a case-insensitive substring so a plain "delta" in these fixtures is enough.
    hint = "A ball with delta one delta two delta three delta four distinct words alpha beta gamma epsilon"
    stem = "A ball with distinct words"
    options = {
        "A": "alpha delta one", "B": "beta two",  # B's own delta dropped
        "C": "gamma delta three", "D": "epsilon four",  # D's own delta dropped too
    }
    reason = pmv._gate(stem, options, hint)
    check("flags dropped delta markers", reason is not None and "delta" in reason, reason)


def test_gate_tolerates_one_missing_delta():
    print("\n# _gate: losing just ONE delta marker (out of several) is within tolerance")
    hint = "A ball with delta one delta two delta three distinct words alpha beta gamma extra four"
    stem = "A ball with distinct words"
    options = {"A": "alpha delta one", "B": "beta delta two", "C": "gamma three", "D": "extra four"}
    reason = pmv._gate(stem, options, hint)
    check("gate still passes (within -1 tolerance)", reason is None, reason)


def test_apply_vision_rejects_overlong_reply():
    print("\n# apply_vision_to_flagged_questions: a reply that leaks reasoning instead of the fixed format is rejected")
    q = _flagged_q()
    result = {"questions": [q], "completeness": {}}

    def rambling(image_bytes, prompt):
        return "Q: stem\nA) one\nB) two\nC) three\nD) four\n" + ("wait, let me reconsider this. " * 100), _usage()

    out = pmv.apply_vision_to_flagged_questions(result, generate_fn=rambling)
    check("still flagged (reply too long)", out["questions"][0]["needs_review"] is True)
    check("gate_failed counted", out["cost"]["gate_failed"] == 1, out["cost"])


def test_gate_rejects_unpaired_single_option_mismatch():
    print("\n# _gate: distinct, non-empty, word-matching options with no duplicates pass even if not real NEET content")
    hint = "alpha bravo charlie delta echo foxtrot golf hotel"
    reason = pmv._gate("alpha bravo stem", {"A": "charlie", "B": "delta", "C": "echo", "D": "foxtrot"}, hint)
    check("gate passes when every word traces back to the hint", reason is None, reason)


# ---------------------------------------------------------------- apply_vision_to_flagged_questions

def test_apply_vision_fixes_flagged_question():
    print("\n# apply_vision_to_flagged_questions: a good model reply fixes a flagged question")
    q = _flagged_q()
    result = {"questions": [q], "completeness": {"Physics": {"red_flag_count": 1}}}
    calls = []

    def fake_generate(image_bytes, prompt):
        calls.append(prompt)
        return _good_reply(), _usage()

    out = pmv.apply_vision_to_flagged_questions(result, generate_fn=fake_generate)
    check("exactly one model call made", len(calls) == 1, len(calls))
    check("question no longer flagged", out["questions"][0]["needs_review"] is False)
    check("marked model_read", out["questions"][0].get("model_read") is True)
    check("review_reason cleared", out["questions"][0]["review_reason"] is None)
    check("option_a replaced", out["questions"][0]["option_a"] == "W m 2 K 4", out["questions"][0]["option_a"])
    check("printed_number untouched", out["questions"][0]["printed_number"] == 1)
    check("internal hint field popped", "_text_layer_hint" not in out["questions"][0])
    check("cost.usd > 0", out["cost"]["usd"] > 0, out["cost"])
    check("cost.model_read == 1", out["cost"]["model_read"] == 1, out["cost"])
    check("completeness red_flag_count recomputed to 0", out["completeness"]["Physics"]["red_flag_count"] == 0)


def test_apply_vision_never_calls_model_for_clean_questions():
    print("\n# apply_vision_to_flagged_questions: a clean (not flagged) question never gets a model call")
    flagged = _flagged_q(printed_number=1)
    clean = _clean_q(printed_number=2)
    result = {"questions": [clean, flagged], "completeness": {}}
    calls = []

    def fake_generate(image_bytes, prompt):
        calls.append(1)
        return _good_reply(), _usage()

    pmv.apply_vision_to_flagged_questions(result, generate_fn=fake_generate)
    check("exactly one call (only the flagged question)", len(calls) == 1, len(calls))


def test_apply_vision_retries_once_then_gives_up():
    print("\n# apply_vision_to_flagged_questions: a gate failure retries once, then leaves the question flagged")
    q = _flagged_q()
    original_option_a = q["option_a"]
    result = {"questions": [q], "completeness": {}}
    calls = []

    def always_bad(image_bytes, prompt):
        calls.append(1)
        return "Q: stem\nA) same\nB) same\nC) x\nD) y\n", _usage()  # duplicate A/B -- always fails the gate

    out = pmv.apply_vision_to_flagged_questions(result, generate_fn=always_bad)
    check("tried exactly twice (1 try + 1 retry)", len(calls) == 2, len(calls))
    check("question still flagged", out["questions"][0]["needs_review"] is True)
    check("original deterministic fields untouched", out["questions"][0]["option_a"] == original_option_a)
    check("not marked model_read", not out["questions"][0].get("model_read"))
    check("cost.gate_failed == 1", out["cost"]["gate_failed"] == 1, out["cost"])
    check("cost.retried == 1", out["cost"]["retried"] == 1, out["cost"])
    check("cost.model_read == 0", out["cost"]["model_read"] == 0, out["cost"])


def test_apply_vision_retry_succeeds_on_second_call():
    print("\n# apply_vision_to_flagged_questions: first call fails the gate, retry succeeds")
    q = _flagged_q()
    result = {"questions": [q], "completeness": {}}
    responses = [
        ("Q: stem\nA) same\nB) same\nC) x\nD) y\n", _usage()),  # fails (duplicate)
        (_good_reply(), _usage()),  # succeeds
    ]

    def fake_generate(image_bytes, prompt):
        return responses.pop(0)

    out = pmv.apply_vision_to_flagged_questions(result, generate_fn=fake_generate)
    check("question fixed on retry", out["questions"][0]["needs_review"] is False)
    check("cost.retried == 1", out["cost"]["retried"] == 1, out["cost"])
    check("cost.model_read == 1", out["cost"]["model_read"] == 1, out["cost"])


def test_apply_vision_model_exception_treated_as_gate_failure():
    print("\n# apply_vision_to_flagged_questions: a raised exception never crashes the scan")
    q = _flagged_q()
    result = {"questions": [q], "completeness": {}}

    def always_raises(image_bytes, prompt):
        raise RuntimeError("simulated network failure")

    out = pmv.apply_vision_to_flagged_questions(result, generate_fn=always_raises)
    check("no crash, question stays flagged", out["questions"][0]["needs_review"] is True)
    check("cost.usd == 0 (nothing billed for a raised call)", out["cost"]["usd"] == 0, out["cost"])
    check("cost.gate_failed == 1", out["cost"]["gate_failed"] == 1, out["cost"])


def test_apply_vision_respects_cap():
    print("\n# apply_vision_to_flagged_questions: at most max_calls flagged questions get a model call")
    qs = [_flagged_q(printed_number=i, mock_position=i) for i in range(1, 4)]
    result = {"questions": qs, "completeness": {}}
    calls = []

    def fake_generate(image_bytes, prompt):
        calls.append(1)
        return _good_reply(), _usage()

    out = pmv.apply_vision_to_flagged_questions(result, max_calls=2, generate_fn=fake_generate)
    check("only 2 of 3 flagged questions attempted", len(calls) == 2, len(calls))
    check("cost.skipped_over_cap == 1", out["cost"]["skipped_over_cap"] == 1, out["cost"])
    check("3rd question left flagged, untouched", out["questions"][2]["needs_review"] is True)


def test_apply_vision_wrong_option_count_reply_leaves_question_flagged():
    print("\n# apply_vision_to_flagged_questions: a reply with the wrong option count never gets applied")
    q = _flagged_q()
    result = {"questions": [q], "completeness": {}}

    def wrong_count(image_bytes, prompt):
        return "Q: stem\nA) one\nB) two\nC) three\n", _usage()  # only 3 options

    out = pmv.apply_vision_to_flagged_questions(result, generate_fn=wrong_count)
    check("still flagged", out["questions"][0]["needs_review"] is True)
    check("gate_failed counted", out["cost"]["gate_failed"] == 1, out["cost"])


def test_apply_vision_no_flagged_questions_is_a_no_op():
    print("\n# apply_vision_to_flagged_questions: nothing flagged -> zero model calls, cost 0")
    result = {"questions": [_clean_q(1), _clean_q(2)], "completeness": {}}
    calls = []

    def fake_generate(image_bytes, prompt):
        calls.append(1)
        return _good_reply(), _usage()

    out = pmv.apply_vision_to_flagged_questions(result, generate_fn=fake_generate)
    check("zero model calls", len(calls) == 0, len(calls))
    check("cost.usd == 0", out["cost"]["usd"] == 0, out["cost"])
    check("cost.attempted == 0", out["cost"]["attempted"] == 0, out["cost"])


# ---------------------------------------------------------------- server.py wiring

def test_server_endpoint_calls_vision_pass():
    print("\n# /admin/scan-pdf: layout=mock_exam calls apply_vision_to_flagged_questions when enabled")
    import base64 as b64mod
    from unittest.mock import patch
    import server
    from fastapi.testclient import TestClient

    server.ADMIN_PASSWORD = "test-admin-key"
    client = TestClient(server.app)
    fake_scan_result = {"questions": [], "pages_scanned": 0, "flagged_pages": [], "completeness": {}, "mock_name": "Mock Test 1"}

    with patch.object(server, "scan_mock_exam_pdf", return_value=fake_scan_result), \
         patch.object(server, "MOCK_VISION_FLAGGED_ENABLED", True), \
         patch.object(server, "apply_vision_to_flagged_questions", return_value=fake_scan_result) as mocked_vision:
        r = client.post(
            "/admin/scan-pdf",
            json={"data": b64mod.b64encode(b"fake pdf bytes").decode(), "layout": "mock_exam", "mock_name": "Mock Test 1"},
            headers={"X-Admin-Key": "test-admin-key"},
        )
        check("200 OK", r.status_code == 200, (r.status_code, r.text))
        check("vision pass was called", mocked_vision.called)

    with patch.object(server, "scan_mock_exam_pdf", return_value=fake_scan_result) as mocked_scan, \
         patch.object(server, "MOCK_VISION_FLAGGED_ENABLED", False), \
         patch.object(server, "apply_vision_to_flagged_questions") as mocked_vision2:
        r2 = client.post(
            "/admin/scan-pdf",
            json={"data": b64mod.b64encode(b"fake pdf bytes").decode(), "layout": "mock_exam", "mock_name": "Mock Test 1"},
            headers={"X-Admin-Key": "test-admin-key"},
        )
        check("200 OK with flag off", r2.status_code == 200, (r2.status_code, r2.text))
        check("scan still ran", mocked_scan.called)
        check("vision pass skipped when MOCK_VISION_FLAGGED_ENABLED is False", not mocked_vision2.called)


def main():
    test_rate_mirrors_server()
    test_parse_model_reply_basic()
    test_parse_model_reply_malformed_returns_fewer_options()
    test_gate_wrong_option_count()
    test_gate_empty_option()
    test_gate_duplicate_options()
    test_gate_word_mismatch()
    test_gate_accepts_matching_content_despite_latex_vs_raw_notation()
    test_gate_rejects_lost_delta_markers()
    test_gate_tolerates_one_missing_delta()
    test_apply_vision_rejects_overlong_reply()
    test_gate_rejects_unpaired_single_option_mismatch()
    test_apply_vision_fixes_flagged_question()
    test_apply_vision_never_calls_model_for_clean_questions()
    test_apply_vision_retries_once_then_gives_up()
    test_apply_vision_retry_succeeds_on_second_call()
    test_apply_vision_model_exception_treated_as_gate_failure()
    test_apply_vision_respects_cap()
    test_apply_vision_wrong_option_count_reply_leaves_question_flagged()
    test_apply_vision_no_flagged_questions_is_a_no_op()
    test_server_endpoint_calls_vision_pass()

    print(f"\n{len(_passed)} passed, {len(_failed)} failed")
    if _failed:
        sys.exit(1)


if __name__ == "__main__":
    main()
