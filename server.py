from fastapi import FastAPI, Header, Depends, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field
from typing import Optional, List
import anthropic
import requests
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

app = FastAPI()

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
DAILY_TOKEN_BUDGET_FREE = 37000  # ~15 doubts/day at a ~60/40 Haiku/Sonnet blended mix

def _ist_today() -> str:
    return datetime.now(timezone.utc).astimezone(IST).date().isoformat()

def get_user_plan(user_id: str) -> str:
    if not user_id:
        return "free"
    try:
        rows = http_requests.get(
            f"{SUPABASE_URL}/rest/v1/user_plan", headers=ADMIN_HEADERS,
            params={"user_id": f"eq.{user_id}", "select": "plan", "limit": 1}
        ).json()
        return rows[0]["plan"] if rows else "free"
    except Exception:
        return "free"

DAILY_TOKEN_BUDGET_GUEST = 5000  # ~2 doubts/day -- deliberately tight vs. the logged-in free
                                  # tier (15/day): the goal is to force a login, not to be a
                                  # usable tier on its own

def enforce_daily_budget(user_id: str, ip: str = ""):
    if user_id:
        if get_user_plan(user_id) != "free":
            return  # paid = unlimited for now
        rows = http_requests.get(
            f"{SUPABASE_URL}/rest/v1/usage_log", headers=ADMIN_HEADERS,
            params={"user_id": f"eq.{user_id}", "usage_date": f"eq.{_ist_today()}",
                    "select": "tokens_used", "limit": 1}
        ).json()
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
    rows = http_requests.get(
        f"{SUPABASE_URL}/rest/v1/guest_usage_log", headers=ADMIN_HEADERS,
        params={"ip": f"eq.{ip}", "usage_date": f"eq.{_ist_today()}",
                "select": "tokens_used", "limit": 1}
    ).json()
    used = rows[0]["tokens_used"] if rows else 0
    if used >= DAILY_TOKEN_BUDGET_GUEST:
        raise HTTPException(status_code=402, detail="Guest limit reached — log in to continue")

def log_token_usage(user_id: str, tokens: int, ip: str = ""):
    # check-then-log (enforce_daily_budget then this) is not atomic -- a handful of concurrent
    # requests from the same user could push them slightly over budget before the next request
    # gets blocked. Accepted: bounded blast radius, cents of cost, no real money on the free
    # tier yet. Revisit with row-locking/reserve-then-refund if paid gating starts protecting
    # real revenue.
    if tokens <= 0:
        return
    try:
        if user_id:
            http_requests.post(
                f"{SUPABASE_URL}/rest/v1/rpc/increment_daily_usage", headers=ADMIN_HEADERS,
                json={"p_user_id": user_id, "p_date": _ist_today(), "p_tokens": tokens}
            )
        elif ip:
            http_requests.post(
                f"{SUPABASE_URL}/rest/v1/rpc/increment_guest_usage", headers=ADMIN_HEADERS,
                json={"p_ip": ip, "p_date": _ist_today(), "p_tokens": tokens}
            )
    except Exception:
        pass  # never let logging failure break a response the student already received

import hashlib

def get_embedding(text: str):
    question_hash = hashlib.sha256(text.encode()).hexdigest()
    # Service-role key: embedding_cache has RLS with no anon INSERT policy, so writes
    # via the anon key were silently rejected (401) — reads worked, writes never did.
    headers = {
        "apikey": SUPABASE_SERVICE_KEY,
        "Authorization": f"Bearer {SUPABASE_SERVICE_KEY}"
    }

    cached = http_requests.get(
        f"{SUPABASE_URL}/rest/v1/embedding_cache?question_hash=eq.{question_hash}&select=embedding",
        headers=headers
    ).json()

    if cached:
        return cached[0]["embedding"]

    response = openai_client.embeddings.create(
        model="text-embedding-3-small",
        input=text
    )
    embedding = response.data[0].embedding

    http_requests.post(
        f"{SUPABASE_URL}/rest/v1/embedding_cache",
        headers={**headers, "Content-Type": "application/json"},
        json={"question_hash": question_hash, "embedding": embedding}
    )

    return embedding

def search_ncert(query: str, limit: int = 3):
    embedding = get_embedding(query)
    headers = {
        "apikey": SUPABASE_KEY,
        "Authorization": f"Bearer {SUPABASE_KEY}",
        "Content-Type": "application/json"
    }
    response = requests.post(
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

def search_pyq(query: str, limit: int = 5):
    embedding = get_embedding(query)
    headers = {
        "apikey": SUPABASE_KEY,
        "Authorization": f"Bearer {SUPABASE_KEY}",
        "Content-Type": "application/json"
    }
    response = requests.post(
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

def get_student_context(user_id: str) -> str:
    if not user_id:
        return ""
    # Uses the service-role key deliberately: these tables are RLS-scoped to the
    # authenticated owner, and this request is made server-side on the student's
    # behalf (already filtered to their own user_id below), not through their session.
    headers = {"apikey": SUPABASE_SERVICE_KEY, "Authorization": f"Bearer {SUPABASE_SERVICE_KEY}"}
    parts = []

    try:
        results = http_requests.get(
            f"{SUPABASE_URL}/rest/v1/mock_results",
            headers=headers,
            params={
                "user_id": f"eq.{user_id}",
                "select": "score,correct,wrong,subject_biology_score,subject_physics_score,subject_chemistry_score",
                "order": "created_at.desc",
                "limit": 5
            }
        ).json()
        if isinstance(results, list) and results:
            avg_score = sum(r.get("score", 0) for r in results) / len(results)
            parts.append(f"Recent mock test average score: {avg_score:.0f}/720 over the last {len(results)} test(s).")
    except Exception:
        pass

    try:
        mistakes = http_requests.get(
            f"{SUPABASE_URL}/rest/v1/saved_questions",
            headers=headers,
            params={
                "user_id": f"eq.{user_id}",
                "select": "subject,chapter",
                "order": "saved_at.desc",
                "limit": 15
            }
        ).json()
        if isinstance(mistakes, list) and mistakes:
            chapter_counts = {}
            for m in mistakes:
                ch = (m.get("chapter") or "").strip()
                if ch:
                    chapter_counts[ch] = chapter_counts.get(ch, 0) + 1
            if chapter_counts:
                weak = sorted(chapter_counts.items(), key=lambda x: -x[1])[:3]
                weak_str = ", ".join(f"{ch} ({count} missed questions)" for ch, count in weak)
                parts.append(f"Chapters this student struggles with most: {weak_str}.")
    except Exception:
        pass

    if not parts:
        return ""

    return (
        "\n\nSTUDENT CONTEXT (use this to naturally tailor depth and examples to this "
        "specific student — e.g. spend more care on their weak chapters, don't over-explain "
        "things they're already strong in. Don't explicitly say 'according to your data' or "
        "similar — just let it shape the answer naturally):\n" + "\n".join(parts)
    )

def stream_response(text: str, history: list = [], images: list = [], pdf: str = None, answer_style: str = "detailed", student_name: str = "", language: str = "en", user_id: str = "", personalize: bool = True, skip_cache: bool = False, ip: str = ""):
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
        cached = http_requests.get(
            f"{SUPABASE_URL}/rest/v1/answer_cache?question_hash=eq.{answer_hash}&select=answer",
            headers=headers
        ).json()
        if cached:
            yield cached[0]["answer"]
            return
    results = search_ncert(text)

    if results:
        context = "\n\n".join([
            f"[{r.get('subject', '')} - Class {r.get('class', '')} - {r.get('chapter_name', '')}]\n{r.get('content', '')}"
            for r in results
        ])
        user_message = f"NCERT Content:\n{context}\n\nStudent Question: {text}"
    else:
        user_message = f"Student Question: {text}"

    client = anthropic.Anthropic(api_key=ANTHROPIC_KEY)

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
        complex_keywords = ["explain", "compare", "solve", "mechanism", "difference", "derive", "describe", "elaborate", "distinguish", "why", "how does", "what happens", "process of", "steps", "diagram"]
        is_complex = any(kw in text.lower() for kw in complex_keywords) or len(text.split()) > 12
        selected_model = "claude-sonnet-4-5" if is_complex else "claude-haiku-4-5"
        import sys
        print(f"MODEL SELECTED: {selected_model}", flush=True)
        sys.stdout.flush()
        name_context = f"\n\nThe student name is {student_name}. Use their name naturally and occasionally in responses to make it personal." if student_name else ""
        style_context = "\n\nIMPORTANT: The student has selected CONCISE mode. Give a very short answer — maximum 3 sentences only. No bullet points, no key points section, no memory tricks. Just the core answer." if answer_style == "concise" else ""
        lang_context = "\n\nIMPORTANT: Respond ONLY in Hindi (Devanagari script). Every word — headings, key points, explanations, memory tricks — must be in Hindi. Do not mix in English words or Hinglish, even for common scientific terms (e.g. write \"गुणसूत्र\" not \"chromosome\"). The ONLY exceptions are: LaTeX/KaTeX math notation, chemical formulas/symbols (e.g. $H_2O$), units (e.g. m/s, kg), and proper nouns like NEET or NCERT — keep those exactly as-is, do not translate or romanize them." if language == "hi" else ""
        student_context = get_student_context(user_id) if (personalize and user_id) else ""
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
                # Reliably logs on normal completion (verified live). Does NOT reliably fire on
                # early client disconnect: this is a sync generator, and Starlette runs sync
                # StreamingResponse iterators inside a thread-pool wrapper, so disconnect-driven
                # GeneratorExit can't interrupt an in-flight blocking call the way it would for a
                # native async generator -- confirmed empirically (aborting a stream mid-way did
                # not trigger this finally block). Anthropic still bills for tokens generated
                # before the abort, so aborted/partial streams are a known, accepted under-count.
                # Not worth an async-generator rewrite of this path for a cosmetic edge case with
                # no real money on the free tier yet -- same reasoning as the check-then-log race
                # in log_token_usage() above.
                try:
                    usage = stream.get_final_message().usage
                    log_token_usage(user_id, usage.input_tokens + usage.output_tokens, ip)
                except Exception:
                    pass
            if not images and not pdf and use_shared_cache:
                http_requests.post(
                    f"{SUPABASE_URL}/rest/v1/answer_cache",
                    headers={**headers, "Content-Type": "application/json"},
                    json={"question_hash": answer_hash, "answer": full_answer}
                )
    except Exception as e:
        print(f"STREAMING ERROR: {e}")
        yield f"Error: {str(e)}"

@app.post("/solve")
async def solve_question(req: SolveRequest, request: Request, _: None = Depends(rate_limiter(15, 60))):
    ip = _client_ip(request)
    enforce_daily_budget(req.user_id, ip)
    client = anthropic.Anthropic(api_key=ANTHROPIC_KEY)
    lang_instruction = "\n5. Respond ONLY in Hindi (Devanagari script) — every word in Hindi, no English words or Hinglish mixing. The ONLY exceptions are LaTeX/KaTeX math notation, chemical formulas/symbols, and units, which stay exactly as-is." if req.language == "hi" else ""
    message = client.messages.create(
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
            {"role": "user", "content": f"Solve this NEET question:\n\nQuestion: {req.question}\n\nA) {req.option_a}\nB) {req.option_b}\nC) {req.option_c}\nD) {req.option_d}\n\nCorrect Answer: {req.correct_answer}"}
        ]
    )
    log_token_usage(req.user_id, message.usage.input_tokens + message.usage.output_tokens, ip)
    return {"solution": message.content[0].text}



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
    enforce_daily_budget(message.user_id, ip)
    return StreamingResponse(
       stream_response(message.text, message.history, message.images, message.pdf, message.answer_style, message.student_name, message.language, message.user_id, message.personalize, message.skip_cache, ip),
        media_type="text/plain"
    )

@app.post("/title")
async def generate_title(message: Message, request: Request, _: None = Depends(rate_limiter(15, 60))):
    ip = _client_ip(request)
    enforce_daily_budget(message.user_id, ip)
    client = anthropic.Anthropic(api_key=ANTHROPIC_KEY)
    title_lang = "entirely in Hindi (Devanagari script) — every word in Hindi, no English words mixed in" if message.language == "hi" else "in English"
    response = client.messages.create(
     model="claude-haiku-4-5",
        max_tokens=15,
        system=f"Generate a short 3-5 word title {title_lang} for this NEET question. Return ONLY the title. No punctuation. No extra words.",
        messages=[{"role": "user", "content": message.text}]
    )
    log_token_usage(message.user_id, response.usage.input_tokens + response.usage.output_tokens, ip)
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
            log_token_usage(req.user_id, tokens)
            http_requests.patch(
                f"{SUPABASE_URL}/rest/v1/guest_usage_log", headers=ADMIN_HEADERS,
                params={"ip": f"eq.{ip}", "usage_date": f"eq.{today}"}, json={"tokens_used": 0}
            )
        return {"merged": tokens}
    except Exception as e:
        return {"merged": 0, "error": str(e)}

@app.post("/pyq")
async def get_pyq(message: Message, _: None = Depends(rate_limiter(15, 60))):
    results = search_pyq(message.text)
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

ADMIN_SORT_COLUMNS = {"id", "subject", "chapter", "question", "correct_answer", "is_active", "year"}

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
    page: int = 1,
    sort_by: str = "id",
    sort_dir: str = "asc",
    _: None = Depends(verify_admin)
):
    try:
        page = max(1, page)
        page_size = 50
        offset = (page - 1) * page_size
        sort_col = sort_by if sort_by in ADMIN_SORT_COLUMNS else "id"
        sort_direction = "desc" if sort_dir == "desc" else "asc"

        params = {
            "select": "id,subject,chapter,question,correct_answer,is_active,year",
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
    payload = [{
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
        "class": q.class_,
        "has_diagram": q.has_diagram,
        "diagram_url": q.diagram_url,
        "option_a_diagram_url": q.option_a_diagram_url,
        "option_b_diagram_url": q.option_b_diagram_url,
        "option_c_diagram_url": q.option_c_diagram_url,
        "option_d_diagram_url": q.option_d_diagram_url,
        "is_active": True
    } for q in body.questions]
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

if __name__ == "__main__":
    import uvicorn
    import os
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port)