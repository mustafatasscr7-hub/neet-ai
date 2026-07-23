from fastapi import FastAPI, Header, Depends, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field
from typing import Optional, List
from contextlib import asynccontextmanager
import asyncio
import anthropic
import requests
import httpx
import openai
import base64
from dotenv import load_dotenv
import os
import sys
# Windows' default console codepage (cp1252) can't encode plenty of real content this app
# handles -- Greek unit prefixes like μF, Hindi/Devanagari answers, etc. -- and an unhandled
# UnicodeEncodeError from a bare print() crashes the request that triggered it. Only matters
# locally (Linux containers default to UTF-8 already), but a crash here takes a streaming
# response down mid-flight with no clean error to the client, so fix it at the source.
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")
load_dotenv()

# Reused across every request instead of opening a fresh connection per call. httpx's
# connection pool keeps the underlying TCP/TLS connection to Supabase warm between calls --
# measured ~2x faster per call than requests' one-off connections in the /chat latency
# investigation (0.67s cold vs 0.31s pooled, same real endpoint, 6-call average). Created once
# at startup via the lifespan handler below, not per-request, which would defeat the purpose.
async_client: httpx.AsyncClient = None

@asynccontextmanager
async def lifespan(app: FastAPI):
    global async_client
    async_client = httpx.AsyncClient()
    yield
    await async_client.aclose()

app = FastAPI(lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
    expose_headers=["*"],
)

ANTHROPIC_KEY = os.getenv("ANTHROPIC_KEY")
OPENAI_KEY = os.getenv("OPENAI_KEY")
SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_KEY = os.getenv("SUPABASE_KEY")
# Service-role key bypasses RLS for admin writes — falls back to the anon key if not set,
# but admin updates may fail under RLS until a real service_role key is added.
SUPABASE_SERVICE_KEY = os.getenv("SUPABASE_SERVICE_KEY", SUPABASE_KEY)
ADMIN_PASSWORD = os.getenv("ADMIN_PASSWORD", "neetai-admin-2027")

openai_client = openai.OpenAI(api_key=OPENAI_KEY)
# Reused across every request, same reasoning as the httpx.AsyncClient fix above -- creating
# a fresh anthropic.Anthropic() per call was measured adding real overhead (client construction
# ~0.2-0.3s, then establishing that fresh connection to Anthropic's API added another 1-2s+ on
# top, sometimes far more), on top of whatever Anthropic's own response time actually is.
anthropic_client = anthropic.Anthropic(api_key=ANTHROPIC_KEY)
# Async client used only for the complexity-classification call below -- that call needs to
# genuinely run concurrently with the NCERT/student-context fetch (asyncio.gather), and the
# sync client would block the event loop for its duration if awaited naively.
anthropic_async_client = anthropic.AsyncAnthropic(api_key=ANTHROPIC_KEY)
import requests as http_requests
from process_pyq_vision import scan_pdf_bytes

SYSTEM_PROMPT = """You are NEET-AI — an expert tutor for Indian medical entrance exam preparation.

You will be given relevant NCERT content to answer the student's question.

For EVERY answer follow this exact format:

NEET Importance: ⭐⭐⭐⭐⭐ (5/5)

📚 Chapter: [NCERT Class X, Chapter X — Chapter Name]

📝 Answer:
[Give the answer in clear points]

🔑 Key Points:
- [Point 1]
- [Point 2]
- [Point 3]

🧠 Easy Way to Remember:
[One simple memory trick]

For numerical problems use this format:
Given:
- [list all given values]

Formula:
[write the formula]

Solution:
[step by step calculation]

Answer: [final answer with units]

Rules:
1. Answer ONLY from the NCERT content provided to you
2. Always show importance stars AT THE TOP
3. Use bullet points — never big paragraphs
4. Answer length should match question complexity
5. If question is outside NCERT say: This is outside the NCERT NEET syllabus.
6. Rate importance HONESTLY based on NEET exam frequency:
   5 stars = Asked almost every year
   4 stars = Asked frequently
   3 stars = Asked sometimes
   2 stars = Rarely asked
   1 star = Almost never asked
7. For ALL math formulas and equations use KaTeX format:
   - Inline math: $formula$ — example: $\\frac{1}{2}mv^2$
   - Display math: $$formula$$ — example: $$E = mc^2$$
   - Always write: $\\frac{1}{2}mv^2$ NOT ½mv²
   - Always write: $v^2 = u^2 + 2as$ NOT v² = u² + 2as
   - Always write: $F = ma$ for all formulas
   - Subscripts: $H_2O$ NOT H₂O
   - Superscripts: $x^2$ NOT x²"""

class ImageAttachment(BaseModel):
    data: str
    media_type: str = "image/jpeg"

class Message(BaseModel):
    text: str
    answer_style: str = "detailed"
    student_name: str = ""
    history: list = []
    images: List[ImageAttachment] = []
    pdf: str = None
    language: str = "en"
    user_id: str = ""
    personalize: bool = True
    skip_cache: bool = False

class PhoneOtpRequest(BaseModel):
    phone: str

class SolveRequest(BaseModel):
    question: str
    option_a: str = ""
    option_b: str = ""
    option_c: str = ""
    option_d: str = ""
    correct_answer: str = ""
    language: str = "en"
    user_id: str = ""

class MergeGuestUsageRequest(BaseModel):
    user_id: str

class PersonalisedTestSelection(BaseModel):
    subject: str
    chapters: list = []

class PersonalisedTestRequest(BaseModel):
    selections: List[PersonalisedTestSelection]
    count: int = 10

class PersonalisedCatalogStartRequest(BaseModel):
    subject: str
    test_number: int

class AdminPyqUpdate(BaseModel):
    chapter: Optional[str] = None
    correct_answer: Optional[str] = None
    is_active: Optional[bool] = None

class AdminPyqBulkUpdate(BaseModel):
    ids: List[str]
    chapter: Optional[str] = None
    is_active: Optional[bool] = None

class SetUserPlanRequest(BaseModel):
    user_id: str
    plan: str  # "free" or "pro"

class ScanPdfRequest(BaseModel):
    subject: str
    data: str  # base64-encoded PDF bytes

class DiagramUploadRequest(BaseModel):
    filename: str
    data: str  # base64-encoded image bytes
    media_type: str = "image/png"

class PyqQuestionCreate(BaseModel):
    subject: str
    chapter: Optional[str] = None
    question: str
    option_a: str
    option_b: str
    option_c: str
    option_d: str
    correct_answer: str = ""
    question_type: str = "mcq"
    year: Optional[int] = None
    source_tag: Optional[str] = None
    source_pdf_filename: Optional[str] = None
    class_: Optional[int] = Field(None, alias="class")
    has_diagram: bool = False
    diagram_url: Optional[str] = None
    option_a_diagram_url: Optional[str] = None
    option_b_diagram_url: Optional[str] = None
    option_c_diagram_url: Optional[str] = None
    option_d_diagram_url: Optional[str] = None

class PyqBulkCreate(BaseModel):
    questions: List[PyqQuestionCreate]

import time

ADMIN_MAX_ATTEMPTS = 3
ADMIN_COOLDOWN_SECONDS = 300  # 5 minutes
_admin_login_attempts = {}  # ip -> {"count": int, "blocked_until": float}

def _client_ip(request: Request) -> str:
    forwarded = request.headers.get("x-forwarded-for")
    if forwarded:
        return forwarded.split(",")[0].strip()
    return request.client.host if request.client else "unknown"

def verify_admin(request: Request, x_admin_key: str = Header(None)):
    ip = _client_ip(request)
    now = time.time()
    record = _admin_login_attempts.get(ip, {"count": 0, "blocked_until": 0})

    if record["blocked_until"] > now:
        remaining = int(record["blocked_until"] - now)
        raise HTTPException(status_code=429, detail=f"Too many failed attempts. Try again in {remaining} seconds.")

    if not x_admin_key or x_admin_key != ADMIN_PASSWORD:
        record["count"] += 1
        if record["count"] >= ADMIN_MAX_ATTEMPTS:
            record["blocked_until"] = now + ADMIN_COOLDOWN_SECONDS
            record["count"] = 0
        _admin_login_attempts[ip] = record
        raise HTTPException(status_code=401, detail="Unauthorized")

    _admin_login_attempts.pop(ip, None)

# ---------- Rate limiting for paid-API endpoints (per IP, per route) ----------
_rate_limit_buckets = {}  # "ip:path" -> [timestamps]

def rate_limiter(max_requests: int = 15, window_seconds: int = 60):
    def dependency(request: Request):
        ip = _client_ip(request)
        key = f"{ip}:{request.url.path}"
        now = time.time()
        timestamps = [t for t in _rate_limit_buckets.get(key, []) if now - t < window_seconds]
        if len(timestamps) >= max_requests:
            raise HTTPException(status_code=429, detail="Too many requests — please slow down and try again shortly.")
        timestamps.append(now)
        _rate_limit_buckets[key] = timestamps
    return dependency

# ---------- Rate limiting for phone OTP sends, per phone number ----------
# Separate from rate_limiter() above (which is per-IP): a phone number can be
# targeted from many IPs, and one IP can try many numbers, so both axes need
# their own cap. IP is covered by rate_limiter(3, 600) on the route itself.
_otp_phone_buckets = {}  # phone -> [timestamps]
OTP_MAX_PER_PHONE = 3
OTP_PHONE_WINDOW_SECONDS = 600  # 10 minutes

def _check_phone_otp_limit(phone: str):
    now = time.time()
    timestamps = [t for t in _otp_phone_buckets.get(phone, []) if now - t < OTP_PHONE_WINDOW_SECONDS]
    if len(timestamps) >= OTP_MAX_PER_PHONE:
        raise HTTPException(status_code=429, detail="Too many OTP requests for this number — please wait a few minutes and try again.")
    timestamps.append(now)
    _otp_phone_buckets[phone] = timestamps

ADMIN_HEADERS = {
    "apikey": SUPABASE_SERVICE_KEY,
    "Authorization": f"Bearer {SUPABASE_SERVICE_KEY}"
}

# ---------- Daily AI usage budget (per-user, resets at IST midnight) ----------
from datetime import datetime, timezone, timedelta

IST = timezone(timedelta(hours=5, minutes=30))  # fixed offset: India has no DST
DAILY_TOKEN_BUDGET_FREE = 20000  # ~8 doubts/day blended, ~4 if all heavy Sonnet numericals --
                                  # lowered from 37000 after real heavy usage showed the blended
                                  # average understates worst-case cost per user

def _ist_today() -> str:
    return datetime.now(timezone.utc).astimezone(IST).date().isoformat()

async def get_user_plan(user_id: str) -> str:
    if not user_id:
        return "free"
    try:
        resp = await async_client.get(
            f"{SUPABASE_URL}/rest/v1/user_plan", headers=ADMIN_HEADERS,
            params={"user_id": f"eq.{user_id}", "select": "plan", "limit": 1}
        )
        rows = resp.json()
        return rows[0]["plan"] if rows else "free"
    except Exception:
        return "free"

DAILY_TOKEN_BUDGET_GUEST = 5000  # ~2 doubts/day -- deliberately tight vs. the logged-in free
                                  # tier (15/day): the goal is to force a login, not to be a
                                  # usable tier on its own

async def enforce_daily_budget(user_id: str, ip: str = ""):
    if user_id:
        if await get_user_plan(user_id) != "free":
            return  # paid = unlimited for now
        resp = await async_client.get(
            f"{SUPABASE_URL}/rest/v1/usage_log", headers=ADMIN_HEADERS,
            params={"user_id": f"eq.{user_id}", "usage_date": f"eq.{_ist_today()}",
                    "select": "tokens_used", "limit": 1}
        )
        rows = resp.json()
        used = rows[0]["tokens_used"] if rows else 0
        if used >= DAILY_TOKEN_BUDGET_FREE:
            raise HTTPException(status_code=402, detail="Daily limit reached")
        return
    # Guest (no account): tracked by IP instead, in a separate table -- an IP is a much weaker
    # identity than a user_id (shared behind NAT/campus wifi, changes on mobile networks), so
    # this is approximate, not airtight. Good enough to nudge toward logging in, which is the
    # actual goal here, not perfect anti-abuse.
    if not ip:
        return
    resp = await async_client.get(
        f"{SUPABASE_URL}/rest/v1/guest_usage_log", headers=ADMIN_HEADERS,
        params={"ip": f"eq.{ip}", "usage_date": f"eq.{_ist_today()}",
                "select": "tokens_used", "limit": 1}
    )
    rows = resp.json()
    used = rows[0]["tokens_used"] if rows else 0
    if used >= DAILY_TOKEN_BUDGET_GUEST:
        raise HTTPException(status_code=402, detail="Guest limit reached — log in to continue")

async def log_token_usage(user_id: str, tokens: int, ip: str = ""):
    # check-then-log (enforce_daily_budget then this) is not atomic -- a handful of concurrent
    # requests from the same user could push them slightly over budget before the next request
    # gets blocked. Accepted: bounded blast radius, cents of cost, no real money on the free
    # tier yet. Revisit with row-locking/reserve-then-refund if paid gating starts protecting
    # real revenue.
    if tokens <= 0:
        return
    try:
        if user_id:
            await async_client.post(
                f"{SUPABASE_URL}/rest/v1/rpc/increment_daily_usage", headers=ADMIN_HEADERS,
                json={"p_user_id": user_id, "p_date": _ist_today(), "p_tokens": tokens}
            )
        elif ip:
            await async_client.post(
                f"{SUPABASE_URL}/rest/v1/rpc/increment_guest_usage", headers=ADMIN_HEADERS,
                json={"p_ip": ip, "p_date": _ist_today(), "p_tokens": tokens}
            )
    except Exception:
        pass  # never let logging failure break a response the student already received

import hashlib

async def get_embedding(text: str):
    question_hash = hashlib.sha256(text.encode()).hexdigest()
    # Service-role key: embedding_cache has RLS with no anon INSERT policy, so writes
    # via the anon key were silently rejected (401) — reads worked, writes never did.
    headers = {
        "apikey": SUPABASE_SERVICE_KEY,
        "Authorization": f"Bearer {SUPABASE_SERVICE_KEY}"
    }

    cache_resp = await async_client.get(
        f"{SUPABASE_URL}/rest/v1/embedding_cache?question_hash=eq.{question_hash}&select=embedding",
        headers=headers
    )
    cached = cache_resp.json()

    if cached:
        return cached[0]["embedding"]

    # Not converted to AsyncOpenAI: openai_client is a single module-level instance already
    # reused across requests, so it already gets connection-pooling benefits.
    response = openai_client.embeddings.create(
        model="text-embedding-3-small",
        input=text
    )
    embedding = response.data[0].embedding

    await async_client.post(
        f"{SUPABASE_URL}/rest/v1/embedding_cache",
        headers={**headers, "Content-Type": "application/json"},
        json={"question_hash": question_hash, "embedding": embedding}
    )

    return embedding

async def search_ncert(query: str, limit: int = 3):
    embedding = await get_embedding(query)
    headers = {
        "apikey": SUPABASE_KEY,
        "Authorization": f"Bearer {SUPABASE_KEY}",
        "Content-Type": "application/json"
    }
    response = await async_client.post(
        f"{SUPABASE_URL}/rest/v1/rpc/match_ncert",
        headers=headers,
        json={
            "query_embedding": embedding,
            "match_threshold": 0.5,
            "match_count": limit
        }
    )
    if response.status_code == 200:
        return response.json()
    return []

async def search_pyq(query: str, limit: int = 5):
    embedding = await get_embedding(query)
    headers = {
        "apikey": SUPABASE_KEY,
        "Authorization": f"Bearer {SUPABASE_KEY}",
        "Content-Type": "application/json"
    }
    response = await async_client.post(
        f"{SUPABASE_URL}/rest/v1/rpc/match_pyq",
        headers=headers,
        json={
            "query_embedding": embedding,
            "match_threshold": 0.4,
            "match_count": limit
        }
    )
    if response.status_code == 200:
        return response.json()
    return []

async def get_student_context(user_id: str) -> str:
    if not user_id:
        return ""
    # Uses the service-role key deliberately: these tables are RLS-scoped to the
    # authenticated owner, and this request is made server-side on the student's
    # behalf (already filtered to their own user_id below), not through their session.
    headers = {"apikey": SUPABASE_SERVICE_KEY, "Authorization": f"Bearer {SUPABASE_SERVICE_KEY}"}

    async def fetch_mock_average():
        try:
            resp = await async_client.get(
                f"{SUPABASE_URL}/rest/v1/mock_results",
                headers=headers,
                params={
                    "user_id": f"eq.{user_id}",
                    "select": "score,correct,wrong,subject_biology_score,subject_physics_score,subject_chemistry_score",
                    "order": "created_at.desc",
                    "limit": 5
                }
            )
            results = resp.json()
            if isinstance(results, list) and results:
                avg_score = sum(r.get("score", 0) for r in results) / len(results)
                return f"Recent mock test average score: {avg_score:.0f}/720 over the last {len(results)} test(s)."
        except Exception:
            pass
        return None

    async def fetch_weak_chapters():
        try:
            resp = await async_client.get(
                f"{SUPABASE_URL}/rest/v1/saved_questions",
                headers=headers,
                params={
                    "user_id": f"eq.{user_id}",
                    "select": "subject,chapter",
                    "order": "saved_at.desc",
                    "limit": 15
                }
            )
            mistakes = resp.json()
            if isinstance(mistakes, list) and mistakes:
                chapter_counts = {}
                for m in mistakes:
                    ch = (m.get("chapter") or "").strip()
                    if ch:
                        chapter_counts[ch] = chapter_counts.get(ch, 0) + 1
                if chapter_counts:
                    weak = sorted(chapter_counts.items(), key=lambda x: -x[1])[:3]
                    weak_str = ", ".join(f"{ch} ({count} missed questions)" for ch, count in weak)
                    return f"Chapters this student struggles with most: {weak_str}."
        except Exception:
            pass
        return None

    # These two queries don't depend on each other -- run concurrently instead of stacking
    # their latency sequentially, same reasoning as the outer NCERT/student-context overlap.
    mock_part, weak_part = await asyncio.gather(fetch_mock_average(), fetch_weak_chapters())
    parts = [p for p in (mock_part, weak_part) if p]

    if not parts:
        return ""

    return (
        "\n\nSTUDENT CONTEXT (use this to naturally tailor depth and examples to this "
        "specific student — e.g. spend more care on their weak chapters, don't over-explain "
        "things they're already strong in. Don't explicitly say 'according to your data' or "
        "similar — just let it shape the answer naturally):\n" + "\n".join(parts)
    )

async def _empty_str():
    return ""

async def _return_true():
    return True

CLASSIFIER_FALLBACK_KEYWORDS = ["explain", "compare", "solve", "mechanism", "difference", "derive", "describe", "elaborate", "distinguish", "why", "how does", "what happens", "process of", "steps", "diagram"]

async def classify_complexity(text: str) -> bool:
    """True routes to Sonnet, False to Haiku. Runs inside the same asyncio.gather as the
    NCERT/student-context fetch (see stream_response), so this adds ~zero wall-clock time in
    the common case rather than stacking an extra round-trip in front of the real answer.
    Replaces the old keyword/length heuristic as the primary signal -- real chat history
    showed length alone was a bad proxy: any multiple-choice-formatted question exceeds 12
    words purely from listing 4 options, regardless of whether the underlying problem is a
    one-step calculation or a genuine multi-concept trap. Falls back to that old heuristic
    only if this classification call itself errors, rather than defaulting every failure onto
    the more expensive model."""
    try:
        message = await anthropic_async_client.messages.create(
            model="claude-haiku-4-5",
            max_tokens=5,
            system=(
                "You are a routing classifier for NEET exam doubts. Decide SIMPLE or COMPLEX:\n"
                "SIMPLE = a single direct formula or fact with no unusual edge case, even if "
                "the question text itself is long (e.g. padded with multiple-choice options).\n"
                "COMPLEX = needs multi-step derivation, combines multiple concepts, OR is a "
                "known trap/exception a student could easily get wrong with a naive approach "
                "(e.g. sign conventions, specific-distance optics behavior, reaction exceptions).\n"
                "Respond with ONLY one word: simple or complex."
            ),
            messages=[{"role": "user", "content": text[:1500]}]
        )
        answer = message.content[0].text.strip().lower()
        return "complex" in answer
    except Exception:
        return any(kw in text.lower() for kw in CLASSIFIER_FALLBACK_KEYWORDS) or len(text.split()) > 12

async def stream_response(text: str, history: list = [], images: list = [], pdf: str = None, answer_style: str = "detailed", student_name: str = "", language: str = "en", user_id: str = "", personalize: bool = True, skip_cache: bool = False, ip: str = ""):
    images = (images or [])[:3]
    import hashlib
    # Personalized answers are specific to this student and must never be served from —
    # or written to — the shared answer cache, which is keyed only on question text.
    # skip_cache is for explicit retries: the student has already seen the cached
    # answer and wants a genuinely different generation, so it must bypass cache too.
    use_shared_cache = not (personalize and user_id) and not skip_cache
    answer_hash = hashlib.sha256(f"{language}:{text.strip().lower()}".encode()).hexdigest()
    # Service-role key: answer_cache has RLS with no anon INSERT policy, so writes via
    # the anon key were silently rejected (401) — reads worked, writes never did.
    headers = {
        "apikey": SUPABASE_SERVICE_KEY,
        "Authorization": f"Bearer {SUPABASE_SERVICE_KEY}"
    }
    if not images and not pdf and use_shared_cache:
        cache_resp = await async_client.get(
            f"{SUPABASE_URL}/rest/v1/answer_cache?question_hash=eq.{answer_hash}&select=answer",
            headers=headers
        )
        cached = cache_resp.json()
        if cached:
            yield cached[0]["answer"]
            return
    # Student context, NCERT search, and model-complexity classification are all independent
    # of each other -- run them concurrently instead of stacking their latency sequentially.
    # Image/PDF attachments always get Sonnet: the classifier only sees text, and interpreting
    # an attached diagram/handwritten problem warrants the stronger model regardless of how
    # little text comes with it, so skip the classification call entirely in that case.
    student_context_coro = get_student_context(user_id) if (personalize and user_id) else _empty_str()
    complexity_coro = _return_true() if (images or pdf) else classify_complexity(text)
    results, student_context, is_complex = await asyncio.gather(search_ncert(text), student_context_coro, complexity_coro)

    if results:
        context = "\n\n".join([
            f"[{r.get('subject', '')} - Class {r.get('class', '')} - {r.get('chapter_name', '')}]\n{r.get('content', '')}"
            for r in results
        ])
        user_message = f"NCERT Content:\n{context}\n\nStudent Question: {text}"
    else:
        user_message = f"Student Question: {text}"

    client = anthropic_client

    messages = []
    for msg in history:
        role = "user" if msg["role"] == "user" else "assistant"
        messages.append({"role": role, "content": msg["text"]})

        print(f"Images received: {len(images)}, PDF received: {bool(pdf)}")
    if images:
        content = [
            {
                "type": "image",
                "source": {
                    "type": "base64",
                    "media_type": img.media_type or "image/jpeg",
                    "data": img.data
                }
            }
            for img in images
        ]
        content.append({"type": "text", "text": user_message})
        messages.append({"role": "user", "content": content})
    elif pdf:
        messages.append({
            "role": "user",
            "content": [
                {
                    "type": "document",
                    "source": {
                        "type": "base64",
                        "media_type": "application/pdf",
                        "data": pdf
                    }
                },
                {
                    "type": "text",
                    "text": user_message
                }
            ]
        })
    else:
        messages.append({"role": "user", "content": user_message})
    try:
        selected_model = "claude-sonnet-4-5" if is_complex else "claude-haiku-4-5"
        import sys
        print(f"MODEL SELECTED: {selected_model}", flush=True)
        sys.stdout.flush()
        name_context = f"\n\nThe student name is {student_name}. Use their name naturally and occasionally in responses to make it personal." if student_name else ""
        style_context = "\n\nIMPORTANT: The student has selected CONCISE mode. Give a very short answer — maximum 3 sentences only. No bullet points, no key points section, no memory tricks. Just the core answer." if answer_style == "concise" else ""
        lang_context = "\n\nIMPORTANT: Respond ONLY in Hindi (Devanagari script). Every word — headings, key points, explanations, memory tricks — must be in Hindi. Do not mix in English words or Hinglish, even for common scientific terms (e.g. write \"गुणसूत्र\" not \"chromosome\"). The ONLY exceptions are: LaTeX/KaTeX math notation, chemical formulas/symbols (e.g. $H_2O$), units (e.g. m/s, kg), and proper nouns like NEET or NCERT — keep those exactly as-is, do not translate or romanize them." if language == "hi" else ""
        with client.messages.stream(
            model=selected_model,
            max_tokens=1024,
            system=SYSTEM_PROMPT + name_context + style_context + lang_context + student_context,
            messages=messages
        ) as stream:
            full_answer = ""
            try:
                for text_chunk in stream.text_stream:
                    full_answer += text_chunk
                    yield text_chunk
            finally:
                # Reliably logs on normal completion. This is now a native async generator (not
                # a sync generator run in Starlette's threadpool wrapper), so an early client
                # disconnect propagates via GeneratorExit at the next yield point -- more
                # responsive than the old setup, though not re-verified with a live abort test
                # after this refactor. Anthropic still bills for tokens generated before any
                # abort either way, so a partial under-count on disconnect remains an accepted
                # edge case, same reasoning as the check-then-log race in log_token_usage() above.
                try:
                    usage = stream.get_final_message().usage
                    await log_token_usage(user_id, usage.input_tokens + usage.output_tokens, ip)
                except Exception:
                    pass
            if not images and not pdf and use_shared_cache:
                await async_client.post(
                    f"{SUPABASE_URL}/rest/v1/answer_cache",
                    headers={**headers, "Content-Type": "application/json"},
                    json={"question_hash": answer_hash, "answer": full_answer}
                )
    except Exception as e:
        print(f"STREAMING ERROR: {e}")
        yield f"Error: {str(e)}"

async def stream_solve_response(question: str, option_a: str, option_b: str, option_c: str, option_d: str, correct_answer: str, language: str = "en", user_id: str = "", ip: str = ""):
    client = anthropic_client
    lang_instruction = "\n5. Respond ONLY in Hindi (Devanagari script) — every word in Hindi, no English words or Hinglish mixing. The ONLY exceptions are LaTeX/KaTeX math notation, chemical formulas/symbols, and units, which stay exactly as-is." if language == "hi" else ""
    try:
        with client.messages.stream(
            model="claude-sonnet-4-5",
            max_tokens=1024,
            system=f"""You are a NEET exam expert. Solve the given NEET question step by step.

Format your response exactly like this:

✅ Answer: [correct option] — [option text]

📝 Solution:
- [step 1]
- [step 2]
- [step 3]

🔑 Key Concept:
[one line memory tip]

Rules:
1. No big headings
2. Use bullet points only
3. Keep it short and clear
4. For ALL math formulas use KaTeX format:
   - Inline: $formula$ example: $\\frac{{1}}{{2}}mv^2$
   - Display: $$formula$$ example: $$E = mc^2$$
   - Always write $H_2O$ not H₂O
   - Always write $v^2$ not v²{lang_instruction}""",
            messages=[
                {"role": "user", "content": f"Solve this NEET question:\n\nQuestion: {question}\n\nA) {option_a}\nB) {option_b}\nC) {option_c}\nD) {option_d}\n\nCorrect Answer: {correct_answer}"}
            ]
        ) as stream:
            try:
                for text_chunk in stream.text_stream:
                    yield text_chunk
            finally:
                # Same accepted early-disconnect caveat as stream_response() above.
                try:
                    usage = stream.get_final_message().usage
                    await log_token_usage(user_id, usage.input_tokens + usage.output_tokens, ip)
                except Exception:
                    pass
    except Exception as e:
        yield f"Error: {str(e)}"

@app.post("/solve")
async def solve_question(req: SolveRequest, request: Request, _: None = Depends(rate_limiter(15, 60))):
    ip = _client_ip(request)
    await enforce_daily_budget(req.user_id, ip)
    return StreamingResponse(
        stream_solve_response(req.question, req.option_a, req.option_b, req.option_c, req.option_d, req.correct_answer, req.language, req.user_id, ip),
        media_type="text/plain"
    )



@app.post("/auth/send-otp")
async def send_otp(req: PhoneOtpRequest, _: None = Depends(rate_limiter(3, 600))):
    phone = req.phone.strip()
    if not phone.startswith("+") or not phone[1:].isdigit() or len(phone) < 8:
        raise HTTPException(status_code=400, detail="Enter a valid phone number in international format, e.g. +919876543210.")

    _check_phone_otp_limit(phone)

    try:
        response = http_requests.post(
            f"{SUPABASE_URL}/auth/v1/otp",
            headers={"apikey": SUPABASE_KEY, "Content-Type": "application/json"},
            json={"phone": phone}
        )
    except Exception:
        raise HTTPException(status_code=502, detail="Could not reach the auth service. Please try again.")

    if response.status_code >= 400:
        msg = "Could not send OTP. Please check the number and try again."
        try:
            msg = response.json().get("msg", msg)
        except Exception:
            pass
        raise HTTPException(status_code=400, detail=msg)

    return {"success": True}

@app.post("/chat")
async def chat(message: Message, request: Request, _: None = Depends(rate_limiter(15, 60))):
    ip = _client_ip(request)
    await enforce_daily_budget(message.user_id, ip)
    return StreamingResponse(
       stream_response(message.text, message.history, message.images, message.pdf, message.answer_style, message.student_name, message.language, message.user_id, message.personalize, message.skip_cache, ip),
        media_type="text/plain"
    )

@app.post("/title")
async def generate_title(message: Message, request: Request, _: None = Depends(rate_limiter(15, 60))):
    ip = _client_ip(request)
    await enforce_daily_budget(message.user_id, ip)
    client = anthropic_client
    title_lang = "entirely in Hindi (Devanagari script) — every word in Hindi, no English words mixed in" if message.language == "hi" else "in English"
    response = client.messages.create(
     model="claude-haiku-4-5",
        max_tokens=15,
        system=f"Generate a short 3-5 word title {title_lang} for this NEET question. Return ONLY the title. No punctuation. No extra words.",
        messages=[{"role": "user", "content": message.text}]
    )
    await log_token_usage(message.user_id, response.usage.input_tokens + response.usage.output_tokens, ip)
    return {"title": response.content[0].text}

# Guests have no Supabase session, so they can't use RLS to read their own usage row the way a
# logged-in user does (auth.uid() is null for anonymous requests) -- this endpoint is the only
# way a guest's remaining count can be shown, computed from the server's own view of their IP.
@app.get("/guest-usage-status")
async def guest_usage_status(request: Request):
    ip = _client_ip(request)
    rows = http_requests.get(
        f"{SUPABASE_URL}/rest/v1/guest_usage_log", headers=ADMIN_HEADERS,
        params={"ip": f"eq.{ip}", "usage_date": f"eq.{_ist_today()}", "select": "tokens_used", "limit": 1}
    ).json()
    used = rows[0]["tokens_used"] if rows else 0
    return {"tokens_used": used, "budget": DAILY_TOKEN_BUDGET_GUEST}

# Called once a session exists (see chat.html on load): folds today's guest usage from this
# browser's IP into the now-known user's usage_log, so logging in right after exhausting the
# guest budget doesn't grant a second, separate allowance on top of the real free-tier one.
# Zeroes the guest row after merging so a repeat call (page refresh, multiple tabs) doesn't
# double-count -- safe to call on every page load, not just the first one after login.
@app.post("/merge-guest-usage")
async def merge_guest_usage(req: MergeGuestUsageRequest, request: Request):
    if not req.user_id:
        return {"merged": 0}
    ip = _client_ip(request)
    today = _ist_today()
    try:
        rows = http_requests.get(
            f"{SUPABASE_URL}/rest/v1/guest_usage_log", headers=ADMIN_HEADERS,
            params={"ip": f"eq.{ip}", "usage_date": f"eq.{today}", "select": "tokens_used", "limit": 1}
        ).json()
        tokens = rows[0]["tokens_used"] if rows else 0
        if tokens > 0:
            await log_token_usage(req.user_id, tokens)
            http_requests.patch(
                f"{SUPABASE_URL}/rest/v1/guest_usage_log", headers=ADMIN_HEADERS,
                params={"ip": f"eq.{ip}", "usage_date": f"eq.{today}"}, json={"tokens_used": 0}
            )
        return {"merged": tokens}
    except Exception as e:
        return {"merged": 0, "error": str(e)}

@app.post("/pyq")
async def get_pyq(message: Message, _: None = Depends(rate_limiter(15, 60))):
    results = await search_pyq(message.text)
    return {"pyqs": results}
@app.get("/mock-test-questions")
async def get_mock_test_questions():
    try:
        import random
        headers = {
            "apikey": SUPABASE_KEY,
            "Authorization": f"Bearer {SUPABASE_KEY}"
        }
        bio = http_requests.get(f"{SUPABASE_URL}/rest/v1/pyq?subject=eq.Biology&has_diagram=eq.false&select=*&limit=200", headers=headers).json()
        phy = http_requests.get(f"{SUPABASE_URL}/rest/v1/pyq?subject=eq.Physics&has_diagram=eq.false&select=*&limit=200", headers=headers).json()
        che = http_requests.get(f"{SUPABASE_URL}/rest/v1/pyq?subject=eq.Chemistry&has_diagram=eq.false&select=*&limit=200", headers=headers).json()
        bio_q = random.sample(bio, min(90, len(bio)))
        phy_q = random.sample(phy, min(45, len(phy)))
        che_q = random.sample(che, min(45, len(che)))
        questions = bio_q + phy_q + che_q
        return {"questions": questions, "total": len(questions)}
    except Exception as e:
        return {"error": str(e)}

@app.get("/pyq-chapters")
async def get_pyq_chapters(subject: str):
    if subject not in ("Biology", "Physics", "Chemistry"):
        return {"error": "Invalid subject"}
    try:
        headers = {
            "apikey": SUPABASE_KEY,
            "Authorization": f"Bearer {SUPABASE_KEY}"
        }
        response = http_requests.get(
            f"{SUPABASE_URL}/rest/v1/pyq",
            headers=headers,
            params={
                "subject": f"eq.{subject}",
                "is_active": "eq.true",
                "chapter": "not.is.null",
                "select": "chapter",
                "limit": 2000
            }
        )
        rows = response.json()
        counts = {}
        for r in rows:
            ch = r.get("chapter")
            if ch and ch.strip():
                ch = ch.strip()
                counts[ch] = counts.get(ch, 0) + 1
        chapters = [{"name": ch, "count": counts[ch]} for ch in sorted(counts.keys())]
        return {"chapters": chapters}
    except Exception as e:
        return {"error": str(e)}

@app.post("/personalised-test-questions")
async def get_personalised_test_questions(req: PersonalisedTestRequest):
    if not req.selections:
        return {"error": "No subjects selected"}
    if any(sel.subject not in ("Biology", "Physics", "Chemistry") for sel in req.selections):
        return {"error": "Invalid subject"}
    try:
        import random
        count = max(1, min(int(req.count), 200))
        headers = {
            "apikey": SUPABASE_KEY,
            "Authorization": f"Bearer {SUPABASE_KEY}"
        }
        pool = []
        seen_subjects = set()
        for sel in req.selections:
            if sel.subject in seen_subjects:
                continue
            seen_subjects.add(sel.subject)
            params = {
                "subject": f"eq.{sel.subject}",
                "is_active": "eq.true",
                "select": "*",
                "limit": 1000
            }
            chapters = [c.strip() for c in sel.chapters if c and c.strip()]
            if chapters:
                params["chapter"] = "in.(" + ",".join(chapters) + ")"
            response = http_requests.get(
                f"{SUPABASE_URL}/rest/v1/pyq",
                headers=headers,
                params=params
            )
            pool.extend(response.json())
        available = len(pool)
        selected = random.sample(pool, min(count, available)) if available else []
        return {"questions": selected, "requested": count, "available": available}
    except Exception as e:
        return {"error": str(e)}

@app.get("/personalised-catalog")
async def get_personalised_catalog(subject: str):
    if subject not in ("Biology", "Physics", "Chemistry"):
        return {"error": "Invalid subject"}
    try:
        headers = {
            "apikey": SUPABASE_KEY,
            "Authorization": f"Bearer {SUPABASE_KEY}"
        }
        response = http_requests.get(
            f"{SUPABASE_URL}/rest/v1/personalised_test_sets",
            headers=headers,
            params={
                "subject": f"eq.{subject}",
                "select": "test_number,title,description,question_count",
                "order": "test_number.asc"
            }
        )
        return {"tests": response.json()}
    except Exception as e:
        return {"error": str(e)}

@app.post("/personalised-catalog-start")
async def start_personalised_catalog_test(req: PersonalisedCatalogStartRequest):
    if req.subject not in ("Biology", "Physics", "Chemistry"):
        return {"error": "Invalid subject"}
    try:
        headers = {
            "apikey": SUPABASE_KEY,
            "Authorization": f"Bearer {SUPABASE_KEY}"
        }
        set_response = http_requests.get(
            f"{SUPABASE_URL}/rest/v1/personalised_test_sets",
            headers=headers,
            params={
                "subject": f"eq.{req.subject}",
                "test_number": f"eq.{req.test_number}",
                "select": "question_ids,title",
                "limit": 1
            }
        )
        rows = set_response.json()
        if not rows:
            return {"error": "Test not found"}
        question_ids = rows[0]["question_ids"]
        id_list = ",".join(str(i) for i in question_ids)
        questions_response = http_requests.get(
            f"{SUPABASE_URL}/rest/v1/pyq",
            headers=headers,
            params={
                "id": f"in.({id_list})",
                "is_active": "eq.true",
                "select": "*"
            }
        )
        questions = questions_response.json()
        return {"questions": questions, "title": rows[0]["title"]}
    except Exception as e:
        return {"error": str(e)}

@app.get("/health")
def health():
    return {"status": "ok"}

# ---------- Admin: PYQ data management (admin-dashboard.html only, not linked from any student page) ----------

ADMIN_SORT_COLUMNS = {"id", "subject", "chapter", "question", "correct_answer", "is_active", "year", "created_at"}

def _admin_count(params):
    resp = http_requests.get(
        f"{SUPABASE_URL}/rest/v1/pyq",
        headers={**ADMIN_HEADERS, "Prefer": "count=exact"},
        params={**params, "select": "id", "limit": 1}
    )
    content_range = resp.headers.get("content-range", "")
    tail = content_range.split("/")[-1] if "/" in content_range else ""
    return int(tail) if tail.isdigit() else 0

@app.post("/admin/verify")
async def admin_verify(_: None = Depends(verify_admin)):
    return {"ok": True}

# Manual stand-in for what a Razorpay success webhook will do automatically later: flip
# `plan` to "pro" on payment, back to "free" on cancellation/expiry. For now, set by hand.
@app.post("/admin/set-user-plan")
async def admin_set_user_plan(req: SetUserPlanRequest, _: None = Depends(verify_admin)):
    if req.plan not in ("free", "pro"):
        return {"error": "plan must be 'free' or 'pro'"}
    try:
        resp = http_requests.post(
            f"{SUPABASE_URL}/rest/v1/user_plan",
            headers={**ADMIN_HEADERS, "Content-Type": "application/json", "Prefer": "resolution=merge-duplicates"},
            json={"user_id": req.user_id, "plan": req.plan, "updated_at": datetime.now(timezone.utc).isoformat()}
        )
        if resp.status_code >= 400:
            return {"error": resp.text}
        return {"success": True}
    except Exception as e:
        return {"error": str(e)}

@app.get("/admin/pyq-stats")
async def admin_pyq_stats(_: None = Depends(verify_admin)):
    try:
        total_active = _admin_count({"is_active": "eq.true"})
        empty_answer = _admin_count({"is_active": "eq.true", "or": "(correct_answer.is.null,correct_answer.eq.)"})
        empty_chapter = _admin_count({"is_active": "eq.true", "or": "(chapter.is.null,chapter.eq.)"})
        return {
            "total_active": total_active,
            "empty_correct_answer": empty_answer,
            "empty_chapter": empty_chapter
        }
    except Exception as e:
        return {"error": str(e)}

@app.get("/admin/pyq-chapters")
async def admin_pyq_chapters(subject: str, _: None = Depends(verify_admin)):
    if subject not in ("Biology", "Physics", "Chemistry"):
        return {"error": "Invalid subject"}
    try:
        response = http_requests.get(
            f"{SUPABASE_URL}/rest/v1/pyq",
            headers=ADMIN_HEADERS,
            params={
                "subject": f"eq.{subject}",
                "chapter": "not.is.null",
                "select": "chapter",
                "limit": 5000
            }
        )
        rows = response.json()
        chapters = sorted(set(r["chapter"].strip() for r in rows if r.get("chapter") and r["chapter"].strip()))
        return {"chapters": chapters}
    except Exception as e:
        return {"error": str(e)}

@app.get("/admin/pyq-search")
async def admin_pyq_search(
    subject: str = None,
    chapter: str = None,
    search: str = None,
    is_active: str = None,
    with_uploaded_diagram: str = None,
    page: int = 1,
    sort_by: str = "id",
    sort_dir: str = "asc",
    full: bool = False,
    _: None = Depends(verify_admin)
):
    try:
        page = max(1, page)
        page_size = 50
        offset = (page - 1) * page_size
        sort_col = sort_by if sort_by in ADMIN_SORT_COLUMNS else "id"
        sort_direction = "desc" if sort_dir == "desc" else "asc"

        # full=true is used by the question-preview tool, which needs everything a student would
        # actually see (options, diagrams, source tag) -- the plain-table admin dashboard doesn't
        # use any of that, so its requests stay on the smaller default select.
        select_fields = (
            "id,subject,chapter,question,option_a,option_b,option_c,option_d,correct_answer,"
            "is_active,year,source_tag,class,has_diagram,diagram_url,option_a_diagram_url,"
            "option_b_diagram_url,option_c_diagram_url,option_d_diagram_url,created_at"
            if full else
            "id,subject,chapter,question,correct_answer,is_active,year"
        )
        params = {
            "select": select_fields,
            "order": f"{sort_col}.{sort_direction}",
            "limit": page_size,
            "offset": offset
        }
        if subject in ("Biology", "Physics", "Chemistry"):
            params["subject"] = f"eq.{subject}"
        if chapter:
            params["chapter"] = f"eq.{chapter}"
        if is_active in ("true", "false"):
            params["is_active"] = f"eq.{is_active}"
        if with_uploaded_diagram == "true":
            # An actually-uploaded image, not just the AI's has_diagram guess from extraction --
            # that flag just means "this looked like it needed one," independent of whether
            # anyone's uploaded the image yet. neq. (not equal to empty string) rather than
            # not.is.null: some rows have '' instead of NULL for a never-uploaded slot, and
            # NULL <> '' is not TRUE in SQL's 3-valued logic, so neq. alone excludes both.
            params["or"] = (
                "(diagram_url.neq.,option_a_diagram_url.neq.,"
                "option_b_diagram_url.neq.,option_c_diagram_url.neq.,"
                "option_d_diagram_url.neq.)"
            )
        if search:
            params["question"] = f"ilike.*{search}*"

        response = http_requests.get(
            f"{SUPABASE_URL}/rest/v1/pyq",
            headers={**ADMIN_HEADERS, "Prefer": "count=exact"},
            params=params
        )
        rows = response.json()
        content_range = response.headers.get("content-range", "")
        tail = content_range.split("/")[-1] if "/" in content_range else ""
        total = int(tail) if tail.isdigit() else len(rows)
        return {"rows": rows, "total": total, "page": page, "page_size": page_size}
    except Exception as e:
        return {"error": str(e)}

@app.patch("/admin/pyq-update/{pyq_id}")
async def admin_pyq_update(pyq_id: str, body: AdminPyqUpdate, _: None = Depends(verify_admin)):
    update_fields = {}
    if body.chapter is not None:
        update_fields["chapter"] = body.chapter
    if body.correct_answer is not None:
        update_fields["correct_answer"] = body.correct_answer
    if body.is_active is not None:
        update_fields["is_active"] = body.is_active
    if not update_fields:
        return {"error": "No fields to update"}
    try:
        response = http_requests.patch(
            f"{SUPABASE_URL}/rest/v1/pyq",
            headers={**ADMIN_HEADERS, "Content-Type": "application/json", "Prefer": "return=representation"},
            params={"id": f"eq.{pyq_id}", "select": "id,subject,chapter,question,correct_answer,is_active,year"},
            json=update_fields
        )
        if response.status_code >= 400:
            return {"error": response.text}
        updated = response.json()
        if not updated:
            return {"error": "Row not found"}
        return {"updated": updated[0]}
    except Exception as e:
        return {"error": str(e)}

@app.patch("/admin/pyq-bulk-update")
async def admin_pyq_bulk_update(body: AdminPyqBulkUpdate, _: None = Depends(verify_admin)):
    if not body.ids:
        return {"error": "No ids provided"}
    update_fields = {}
    if body.chapter is not None:
        update_fields["chapter"] = body.chapter
    if body.is_active is not None:
        update_fields["is_active"] = body.is_active
    if not update_fields:
        return {"error": "No fields to update"}
    try:
        id_list = ",".join(str(i) for i in body.ids)
        response = http_requests.patch(
            f"{SUPABASE_URL}/rest/v1/pyq",
            headers={**ADMIN_HEADERS, "Content-Type": "application/json", "Prefer": "return=representation"},
            params={"id": f"in.({id_list})", "select": "id"},
            json=update_fields
        )
        if response.status_code >= 400:
            return {"error": response.text}
        return {"updated_count": len(response.json())}
    except Exception as e:
        return {"error": str(e)}

# Real, permanent delete -- not a soft is_active=false toggle. saved_questions and
# personalised_test_sets both reference pyq.id directly, so a deleted row can leave a dangling
# reference there (saved_questions keeps its own copy of the text so it still displays fine;
# a personalised test would just quietly serve one fewer question than originally seeded).
# Accepted for a single, deliberate, admin-initiated cleanup -- re-extracting and re-uploading a
# corrected version afterward is the intended workflow, not editing in place.
@app.delete("/admin/pyq-delete/{pyq_id}")
async def admin_pyq_delete(pyq_id: str, _: None = Depends(verify_admin)):
    try:
        response = http_requests.delete(
            f"{SUPABASE_URL}/rest/v1/pyq",
            headers={**ADMIN_HEADERS, "Prefer": "return=representation"},
            params={"id": f"eq.{pyq_id}"}
        )
        if response.status_code >= 400:
            return {"error": response.text}
        deleted = response.json()
        if not deleted:
            return {"error": "Row not found"}
        return {"success": True}
    except Exception as e:
        return {"error": str(e)}

# ---------- Admin: PDF scan -> review -> save pipeline (admin-pdf-review.html) ----------

@app.post("/admin/scan-pdf")
def admin_scan_pdf(req: ScanPdfRequest, _: None = Depends(verify_admin), __: None = Depends(rate_limiter(10, 300))):
    # Deliberately sync (not async def): FastAPI runs sync path functions in a thread pool,
    # so this multi-second-to-multi-minute call doesn't block the event loop for other requests.
    if req.subject not in ("Biology", "Physics", "Chemistry"):
        return {"error": "Invalid subject"}
    try:
        pdf_bytes = base64.b64decode(req.data)
    except Exception:
        return {"error": "Could not decode PDF data"}
    try:
        return scan_pdf_bytes(pdf_bytes, req.subject)
    except Exception as e:
        return {"error": str(e)}

@app.get("/admin/pyq-classifier-data")
async def admin_pyq_classifier_data(subject: str, _: None = Depends(verify_admin)):
    if subject not in ("Biology", "Physics", "Chemistry"):
        return {"error": "Invalid subject"}
    try:
        response = http_requests.get(
            f"{SUPABASE_URL}/rest/v1/pyq",
            headers=ADMIN_HEADERS,
            params={
                "subject": f"eq.{subject}",
                "is_active": "eq.true",
                "select": "question,chapter,class",
                "limit": 3000
            }
        )
        # Includes rows with no chapter yet too, so the frontend's exact-duplicate
        # check can catch dupes against still-untagged rows. buildClassifier() itself
        # skips rows with no chapter since they're useless as labeled training examples.
        return {"rows": response.json()}
    except Exception as e:
        return {"error": str(e)}

@app.post("/admin/pyq-diagram-upload")
async def admin_pyq_diagram_upload(body: DiagramUploadRequest, _: None = Depends(verify_admin)):
    try:
        file_bytes = base64.b64decode(body.data)
    except Exception:
        return {"error": "Could not decode image data"}
    import uuid
    ext = body.filename.rsplit(".", 1)[-1] if "." in body.filename else "png"
    path = f"{uuid.uuid4().hex}.{ext}"
    try:
        response = http_requests.post(
            f"{SUPABASE_URL}/storage/v1/object/Q-Daigrams-BIO/{path}",
            headers={
                "apikey": SUPABASE_SERVICE_KEY,
                "Authorization": f"Bearer {SUPABASE_SERVICE_KEY}",
                "Content-Type": body.media_type
            },
            data=file_bytes
        )
        if response.status_code >= 400:
            return {"error": response.text}
        return {"url": f"{SUPABASE_URL}/storage/v1/object/public/Q-Daigrams-BIO/{path}"}
    except Exception as e:
        return {"error": str(e)}

@app.post("/admin/pyq-bulk-create")
async def admin_pyq_bulk_create(body: PyqBulkCreate, _: None = Depends(verify_admin)):
    if not body.questions:
        return {"error": "No questions provided"}
    if any(q.subject not in ("Biology", "Physics", "Chemistry") for q in body.questions):
        return {"error": "Invalid subject"}
    payload = []
    for q in body.questions:
        item = {
            "subject": q.subject,
            "chapter": q.chapter,
            "question": q.question,
            "option_a": q.option_a,
            "option_b": q.option_b,
            "option_c": q.option_c,
            "option_d": q.option_d,
            "correct_answer": q.correct_answer,
            "question_type": q.question_type,
            "year": q.year,
            "source_tag": q.source_tag,
            "class": q.class_,
            "has_diagram": q.has_diagram,
            "diagram_url": q.diagram_url,
            "option_a_diagram_url": q.option_a_diagram_url,
            "option_b_diagram_url": q.option_b_diagram_url,
            "option_c_diagram_url": q.option_c_diagram_url,
            "option_d_diagram_url": q.option_d_diagram_url,
            "is_active": True
        }
        # Only included when present, not unconditionally like the other fields above -- this
        # column needs a manual `alter table` before it exists, and PostgREST rejects an insert
        # that even mentions an unknown column (with null or otherwise), so omitting the key
        # entirely keeps every save working before that migration is run, not just this feature.
        if q.source_pdf_filename:
            item["source_pdf_filename"] = q.source_pdf_filename
        payload.append(item)
    try:
        response = http_requests.post(
            f"{SUPABASE_URL}/rest/v1/pyq",
            headers={**ADMIN_HEADERS, "Content-Type": "application/json", "Prefer": "return=representation"},
            json=payload
        )
        if response.status_code >= 400:
            return {"error": response.text}
        return {"created": response.json()}
    except Exception as e:
        return {"error": str(e)}

@app.get("/admin/pdf-upload-history")
def admin_pdf_upload_history(_: None = Depends(verify_admin)):
    # Grouped here in Python rather than a Postgres view/RPC function -- PostgREST has no GROUP
    # BY, and at this scale (a handful of PDFs a day) pulling the raw rows and aggregating here
    # avoids asking for a second manual SQL step beyond the one new column.
    try:
        response = http_requests.get(
            f"{SUPABASE_URL}/rest/v1/pyq",
            headers=ADMIN_HEADERS,
            params={
                "select": "source_pdf_filename,subject,created_at",
                # neq. (not equal to empty string) rather than not.is.null -- same reasoning as
                # the with_uploaded_diagram filter above: NULL <> '' isn't TRUE in SQL, so this
                # one condition excludes both a never-set column and an empty-string one.
                "source_pdf_filename": "neq.",
                "order": "created_at.desc",
                "limit": 5000
            }
        )
        if response.status_code >= 400:
            return {"error": response.text}
        rows = response.json()
        by_filename = {}
        for r in rows:
            name = r.get("source_pdf_filename")
            if not name:
                continue
            entry = by_filename.setdefault(name, {
                "filename": name, "subjects": set(), "question_count": 0,
                "first_uploaded": r["created_at"], "last_uploaded": r["created_at"]
            })
            entry["subjects"].add(r.get("subject"))
            entry["question_count"] += 1
            entry["first_uploaded"] = min(entry["first_uploaded"], r["created_at"])
            entry["last_uploaded"] = max(entry["last_uploaded"], r["created_at"])
        result = sorted(
            ({**e, "subjects": sorted(s for s in e["subjects"] if s)} for e in by_filename.values()),
            key=lambda e: e["last_uploaded"], reverse=True
        )
        return {"pdfs": result}
    except Exception as e:
        return {"error": str(e)}

import re

# Common exam-phrasing scaffolding that shows up in unrelated MCQs alike ("which of the
# following is correct", "given below are two statements") -- left in, a plain word-overlap
# ratio flags totally different questions as 50%+ similar on structure alone. Roman numerals
# included since "Statement I / Statement II" list markers are the same kind of noise.
DUP_CHECK_STOPWORDS = {
    "a", "an", "the", "is", "are", "was", "were", "be", "been", "being",
    "of", "to", "in", "on", "at", "by", "for", "with", "from", "and", "or",
    "not", "no", "this", "that", "these", "those", "it", "its", "as",
    "which", "one", "following", "correct", "correctly", "incorrect",
    "statement", "statements", "given", "below", "above", "true", "false",
    "select", "choose", "regarding", "about", "consider", "identify",
    "list", "lists", "match", "matching", "column", "columns", "answer", "answers",
    "option", "options", "only", "most", "appropriate", "light",
    "ii", "iii", "iv", "vi", "vii", "viii"
}

def _tokenize_for_dup_check(text):
    words = re.sub(r"[^a-z0-9\s]", " ", (text or "").lower()).split()
    return set(w for w in words if len(w) > 1 and w not in DUP_CHECK_STOPWORDS)

@app.get("/admin/pyq-duplicates")
def admin_pyq_duplicates(threshold: float = 0.5, _: None = Depends(verify_admin)):
    # Deliberately sync (not async def), same reasoning as /admin/scan-pdf: this is a
    # multi-second CPU-bound sweep (pairwise comparison within each subject), and FastAPI runs
    # sync path functions in a thread pool so it doesn't block the event loop for other requests.
    # Scoped per-subject (never compares across Biology/Physics/Chemistry) but NOT per-chapter --
    # chapter tagging can drift or be missing entirely, and a real duplicate should still be
    # caught even if the two copies ended up tagged to slightly different chapters.
    try:
        # PostgREST caps a single response at its server-side max-rows setting (1000 here)
        # regardless of the `limit` we ask for, so a table with 2000+ active rows silently came
        # back truncated until this loop was added -- page through with offset until a page
        # comes back short of the page size, which means it was the last one.
        all_rows = []
        page_size = 1000
        offset = 0
        while True:
            response = http_requests.get(
                f"{SUPABASE_URL}/rest/v1/pyq",
                headers=ADMIN_HEADERS,
                params={
                    "select": "id,subject,chapter,question,option_a,option_b,option_c,option_d,"
                              "correct_answer,is_active,year,source_tag,class,has_diagram,"
                              "diagram_url,option_a_diagram_url,option_b_diagram_url,"
                              "option_c_diagram_url,option_d_diagram_url,created_at",
                    "is_active": "eq.true",
                    "order": "id",
                    "limit": page_size,
                    "offset": offset
                }
            )
            if response.status_code >= 400:
                return {"error": response.text}
            page = response.json()
            all_rows.extend(page)
            if len(page) < page_size:
                break
            offset += page_size

        # Pairs an admin has already looked at and confirmed aren't real duplicates shouldn't
        # keep coming back on every future scan. Table may not exist yet (pre-migration) -- treat
        # that the same as "nothing dismissed" rather than failing the whole scan over it.
        dismissed = set()
        try:
            dismissed_resp = http_requests.get(
                f"{SUPABASE_URL}/rest/v1/pyq_dismissed_duplicates",
                headers=ADMIN_HEADERS,
                params={"select": "id_a,id_b", "limit": 10000}
            )
            if dismissed_resp.status_code < 400:
                dismissed = set((d["id_a"], d["id_b"]) for d in dismissed_resp.json())
        except Exception:
            pass

        by_subject = {}
        for r in all_rows:
            by_subject.setdefault(r["subject"], []).append(r)

        rows_by_id = {}
        pairs = []
        for subject_rows in by_subject.values():
            # Question text alone isn't enough for "Match List I/II" or "Identify the incorrect
            # pair" style stems -- the stem is nearly content-free, and the actual distinguishing
            # material (which items, which pairing) lives entirely in the four options.
            tokenized = [(r, _tokenize_for_dup_check(" ".join(filter(None, [
                r.get("question"), r.get("option_a"), r.get("option_b"),
                r.get("option_c"), r.get("option_d")
            ])))) for r in subject_rows]
            n = len(tokenized)
            for i in range(n):
                row_a, tokens_a = tokenized[i]
                if not tokens_a:
                    continue
                for j in range(i + 1, n):
                    row_b, tokens_b = tokenized[j]
                    if not tokens_b:
                        continue
                    intersection = len(tokens_a & tokens_b)
                    if intersection == 0:
                        continue
                    union = len(tokens_a | tokens_b)
                    overlap = intersection / union if union else 0
                    if overlap >= threshold:
                        pair_key = tuple(sorted([row_a["id"], row_b["id"]]))
                        if pair_key in dismissed:
                            continue
                        rows_by_id[row_a["id"]] = row_a
                        rows_by_id[row_b["id"]] = row_b
                        pairs.append({"a": row_a["id"], "b": row_b["id"], "overlap": round(overlap, 3), "exact": overlap >= 0.999})

        pairs.sort(key=lambda p: p["overlap"], reverse=True)
        return {"rows": rows_by_id, "pairs": pairs, "rows_scanned": len(all_rows), "threshold": threshold}
    except Exception as e:
        return {"error": str(e)}

class DismissDuplicateRequest(BaseModel):
    id_a: str
    id_b: str

@app.post("/admin/pyq-dismiss-duplicate")
def admin_pyq_dismiss_duplicate(req: DismissDuplicateRequest, _: None = Depends(verify_admin)):
    try:
        # Canonical (smaller id first) ordering -- the scan doesn't guarantee which order a pair
        # comes back in, so without this the same pair could get dismissed twice under (A,B) and
        # (B,A) and still show up again.
        ordered = sorted([req.id_a, req.id_b])
        response = http_requests.post(
            f"{SUPABASE_URL}/rest/v1/pyq_dismissed_duplicates",
            headers={**ADMIN_HEADERS, "Content-Type": "application/json", "Prefer": "resolution=merge-duplicates"},
            json={"id_a": ordered[0], "id_b": ordered[1]}
        )
        if response.status_code >= 400:
            return {"error": response.text}
        return {"success": True}
    except Exception as e:
        return {"error": str(e)}

if __name__ == "__main__":
    import uvicorn
    import os
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port)