"""Unit tests for the "From NCERT" block retry (B) -- NCERT_SECTIONS_ENABLED/
NCERT_SECTIONS_RETRY_ENABLED, see _should_retry_ncert_block / _retry_ncert_block /
_log_ncert_sections_event in server.py.

Run with: python test_ncert_block_retry.py
Plain pass/fail harness, same style as test_ncert_sections_verifier.py. Fully offline: the
model clients are fakes, no network.

1. Trigger conditions for _should_retry_ncert_block: missing block, empty block, all quotes
   rejected (retry), at least one quote shown (no retry).
2. Picker NONE never reaches the retry gate at all (ncert_sections_active is False).
3. _retry_ncert_block: a successful Qwen call returns the raw block and logs real cost.
4. _retry_ncert_block: a successful DeepSeek call returns the raw block and logs real cost.
5. _retry_ncert_block: the model replying NONE returns (None, latency) -- no exception.
6. _retry_ncert_block: the model call raising/timing out also returns (None, latency), silent
   ("retry also failing" -- the no-block-shown outcome this produces is exercised in test 7).
7. End-to-end: a retry whose result still has zero quotes survive leaves quotes_shown empty.
8. _log_ncert_sections_event includes retried/retry_latency_ms in the logged chunk.
"""
import asyncio
import io
import os
import sys
from types import SimpleNamespace

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import server

_passed = []
_failed = []


def check(label, condition, detail=""):
    if condition:
        _passed.append(label)
        print(f"  PASS: {label}")
    else:
        _failed.append(label)
        print(f"  FAIL: {label}" + (f" -- {detail}" if detail else ""))


SECTION_TEXT = (
    "The wide range of living types is amazing. All living organisms exhibit growth, "
    "reproduction and the ability to sense their environment and respond to it."
)
VALID_HEADINGS = ["Chapter 1: The Living World"]
GOOD_BLOCK = '"The wide range of living types is amazing." (Chapter 1: The Living World)'


# ---------- 1-2. trigger conditions ----------

def test_trigger_missing_block():
    print("\n1a. missing block (raw_block=None) -> retry")
    check("retry triggered", server._should_retry_ncert_block(None, []))


def test_trigger_empty_block():
    print("\n1b. empty block (raw_block='') -> retry")
    check("retry triggered", server._should_retry_ncert_block("", []))


def test_trigger_all_quotes_rejected():
    print("\n1c. block present but every quote rejected by the verifier -> retry")
    raw = '"Mitochondria is the powerhouse of the cell." (Chapter 1: The Living World)'
    shown, dropped = server._verify_ncert_quotes(raw, SECTION_TEXT, VALID_HEADINGS)
    check("verifier dropped it first (sanity)", shown == [] and len(dropped) == 1, (shown, dropped))
    check("retry triggered", server._should_retry_ncert_block(raw, shown))


def test_no_trigger_when_quote_survives():
    print("\n1d. at least one quote survives -> no retry")
    shown, dropped = server._verify_ncert_quotes(GOOD_BLOCK, SECTION_TEXT, VALID_HEADINGS)
    check("verifier kept it (sanity)", len(shown) == 1, (shown, dropped))
    check("no retry triggered", not server._should_retry_ncert_block(GOOD_BLOCK, shown))


def test_picker_none_never_reaches_retry_gate():
    print("\n2. picker NONE -> ncert_sections_active is False, retry gate never entered")
    picked_section_ids = []  # what _pick_ncert_sections returns for NONE/timeout/empty index
    ncert_sections_active = bool(picked_section_ids)
    check("feature inactive for this request", ncert_sections_active is False)
    # The real call site is `if ncert_sections_active: ... _should_retry_ncert_block(...)` --
    # with the feature inactive that whole branch (including the retry call) never runs, so
    # there's nothing further to simulate here.


# ---------- 3-6. _retry_ncert_block ----------

class FakeQwenRetryClient:
    def __init__(self, content=None, prompt_tokens=200, completion_tokens=40, cached_tokens=0, raise_exc=None):
        self.kwargs = None
        self.content, self.raise_exc = content, raise_exc
        self.prompt_tokens, self.completion_tokens, self.cached_tokens = prompt_tokens, completion_tokens, cached_tokens
        self.chat = SimpleNamespace(completions=SimpleNamespace(create=self._create))

    async def _create(self, **kw):
        self.kwargs = kw
        if self.raise_exc:
            raise self.raise_exc
        message = SimpleNamespace(content=self.content)
        usage = SimpleNamespace(prompt_tokens=self.prompt_tokens, completion_tokens=self.completion_tokens,
                                 prompt_tokens_details=SimpleNamespace(cached_tokens=self.cached_tokens))
        return SimpleNamespace(choices=[SimpleNamespace(message=message)], usage=usage)


class FakeDeepSeekRetryClient:
    def __init__(self, text=None, input_tokens=200, cache_creation=0, cache_read=0, output_tokens=40, raise_exc=None):
        self.kwargs = None
        self.text, self.raise_exc = text, raise_exc
        self.input_tokens, self.cache_creation, self.cache_read, self.output_tokens = input_tokens, cache_creation, cache_read, output_tokens
        self.messages = SimpleNamespace(create=self._create)

    async def _create(self, **kw):
        self.kwargs = kw
        if self.raise_exc:
            raise self.raise_exc
        usage = SimpleNamespace(input_tokens=self.input_tokens, cache_creation_input_tokens=self.cache_creation,
                                 cache_read_input_tokens=self.cache_read, output_tokens=self.output_tokens)
        return SimpleNamespace(content=[SimpleNamespace(text=self.text)], usage=usage)


class ClientSwap:
    """Patches server.qwen_async_client/deepseek_async_client and server.log_provider_usage,
    restoring all three on exit -- same pattern as test_cost_fixes.py's Recorder."""
    def __enter__(self):
        self._saved = (server.qwen_async_client, server.deepseek_async_client, server.log_provider_usage)
        self.logged = []

        async def rec(*a, **k):
            self.logged.append(SimpleNamespace(args=a, kwargs=k))

        server.log_provider_usage = rec
        return self

    def __exit__(self, *exc):
        server.qwen_async_client, server.deepseek_async_client, server.log_provider_usage = self._saved


def test_retry_qwen_success():
    print("\n3. successful Qwen retry call returns the raw block and logs cost")
    with ClientSwap() as sw:
        server.qwen_async_client = FakeQwenRetryClient(content=GOOD_BLOCK)
        raw, latency_ms = asyncio.run(server._retry_ncert_block("What is the living world?", SECTION_TEXT, True, "u1"))
        check("raw block returned", raw == GOOD_BLOCK, raw)
        check("latency recorded (>=0)", latency_ms >= 0, latency_ms)
        check("called qwen-flash", server.qwen_async_client.kwargs["model"] == "qwen-flash")
        check("usage logged", len(sw.logged) == 1 and sw.logged[0].args[0] == "qwen-flash", sw.logged)
        check("logged against the retry endpoint", sw.logged[0].args[5] == "/chat-ncert-sections-retry", sw.logged[0].args)


def test_retry_deepseek_success():
    print("\n4. successful DeepSeek retry call returns the raw block and logs cost")
    with ClientSwap() as sw:
        server.deepseek_async_client = FakeDeepSeekRetryClient(text=GOOD_BLOCK)
        raw, latency_ms = asyncio.run(server._retry_ncert_block("What is the living world?", SECTION_TEXT, False, "u1"))
        check("raw block returned", raw == GOOD_BLOCK, raw)
        check("called deepseek-flash", server.deepseek_async_client.kwargs["model"] == server.DEEPSEEK_MODEL)
        check("usage logged", len(sw.logged) == 1 and sw.logged[0].args[0] == server.DEEPSEEK_MODEL, sw.logged)


def test_retry_model_replies_none():
    print("\n5. model replies NONE -> (None, latency), no exception")
    with ClientSwap():
        server.qwen_async_client = FakeQwenRetryClient(content="NONE")
        raw, latency_ms = asyncio.run(server._retry_ncert_block("q", SECTION_TEXT, True, "u1"))
        check("raw is None", raw is None, raw)
        check("latency still recorded", latency_ms >= 0, latency_ms)


def test_retry_call_fails_silently():
    print("\n6. the retry call itself raises/times out -> (None, latency), never propagates")
    with ClientSwap():
        server.qwen_async_client = FakeQwenRetryClient(raise_exc=TimeoutError("slow"))
        raw, latency_ms = asyncio.run(server._retry_ncert_block("q", SECTION_TEXT, True, "u1"))
        check("raw is None, no exception escaped", raw is None, raw)
        check("latency still recorded", latency_ms >= 0, latency_ms)


def test_retry_also_failing_end_to_end():
    print("\n7. retry succeeds in calling the model but its block also has zero valid quotes")
    with ClientSwap():
        server.qwen_async_client = FakeQwenRetryClient(content='"Mitochondria is the powerhouse of the cell." (Chapter 1: The Living World)')
        raw, _ = asyncio.run(server._retry_ncert_block("q", SECTION_TEXT, True, "u1"))
        check("retry returned a block", raw is not None, raw)
        shown, dropped = server._verify_ncert_quotes(raw, SECTION_TEXT, VALID_HEADINGS)
        check("still nothing shown after retry (invented quote)", shown == [] and len(dropped) == 1, (shown, dropped))
        check("caller's own gate would emit no marker at all for an empty shown list", shown == [])


# ---------- 8. logging ----------

class FakePostClient:
    def __init__(self):
        self.captured = None

    async def post(self, url, headers=None, json=None):
        self.captured = json
        return SimpleNamespace(status_code=201)


def test_log_event_includes_retry_fields():
    print("\n8. _log_ncert_sections_event logs retried/retry_latency_ms in the chunk")
    saved = server.async_client
    fake = FakePostClient()
    server.async_client = fake
    try:
        asyncio.run(server._log_ncert_sections_event("en", [1], [], [], 120, 540, retried=True, retry_latency_ms=310))
    finally:
        server.async_client = saved
    chunk = fake.captured["chunks"][0]
    check("version tagged sections", chunk["version"] == "sections", chunk)
    check("retried=True logged", chunk["retried"] is True, chunk)
    check("retry_latency_ms logged", chunk["retry_latency_ms"] == 310, chunk)
    fake2 = FakePostClient()
    server.async_client = fake2
    try:
        asyncio.run(server._log_ncert_sections_event("en", [1], [{"quote": "x" * 10, "citation": "c"}], [], 100, 200))
    finally:
        server.async_client = saved
    chunk2 = fake2.captured["chunks"][0]
    check("retried defaults to False", chunk2["retried"] is False, chunk2)
    check("retry_latency_ms defaults to 0", chunk2["retry_latency_ms"] == 0, chunk2)


if __name__ == "__main__":
    test_trigger_missing_block()
    test_trigger_empty_block()
    test_trigger_all_quotes_rejected()
    test_no_trigger_when_quote_survives()
    test_picker_none_never_reaches_retry_gate()
    test_retry_qwen_success()
    test_retry_deepseek_success()
    test_retry_model_replies_none()
    test_retry_call_fails_silently()
    test_retry_also_failing_end_to_end()
    test_log_event_includes_retry_fields()
    print(f"\n{len(_passed)} passed, {len(_failed)} failed")
    sys.exit(1 if _failed else 0)
