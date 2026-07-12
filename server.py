from fastapi import FastAPI, Header, Depends, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from typing import Optional, List
import anthropic
import requests
import openai
from dotenv import load_dotenv
import os
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

class SolveRequest(BaseModel):
    question: str
    option_a: str = ""
    option_b: str = ""
    option_c: str = ""
    option_d: str = ""
    correct_answer: str = ""
    language: str = "en"

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

ADMIN_HEADERS = {
    "apikey": SUPABASE_SERVICE_KEY,
    "Authorization": f"Bearer {SUPABASE_SERVICE_KEY}"
}

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

def stream_response(text: str, history: list = [], images: list = [], pdf: str = None, answer_style: str = "detailed", student_name: str = "", language: str = "en", user_id: str = "", personalize: bool = True, skip_cache: bool = False):
    images = (images or [])[:3]
    print(f"stream_response called - text: {text[:50]}, images: {len(images)}")
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
            for text_chunk in stream.text_stream:
                full_answer += text_chunk
                yield text_chunk
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
async def solve_question(req: SolveRequest, _: None = Depends(rate_limiter(15, 60))):
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
    return {"solution": message.content[0].text}



@app.post("/chat")
async def chat(message: Message, _: None = Depends(rate_limiter(15, 60))):
    return StreamingResponse(
       stream_response(message.text, message.history, message.images, message.pdf, message.answer_style, message.student_name, message.language, message.user_id, message.personalize, message.skip_cache),
        media_type="text/plain"
    )

@app.post("/title")
async def generate_title(message: Message, _: None = Depends(rate_limiter(15, 60))):
    client = anthropic.Anthropic(api_key=ANTHROPIC_KEY)
    title_lang = "entirely in Hindi (Devanagari script) — every word in Hindi, no English words mixed in" if message.language == "hi" else "in English"
    response = client.messages.create(
     model="claude-haiku-4-5",
        max_tokens=15,
        system=f"Generate a short 3-5 word title {title_lang} for this NEET question. Return ONLY the title. No punctuation. No extra words.",
        messages=[{"role": "user", "content": message.text}]
    )
    return {"title": response.content[0].text}

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

if __name__ == "__main__":
    import uvicorn
    import os
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port)