from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
import anthropic
import requests
import openai

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
    expose_headers=["*"],
)

ANTHROPIC_KEY = "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"
OPENAI_KEY = "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"
SUPABASE_URL = "https://hvhnfttrfouajlyhvunq.supabase.co"
SUPABASE_KEY = "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJpc3MiOiJzdXBhYmFzZSIsInJlZiI6Imh2aG5mdHRyZm91YWpseWh2dW5xIiwicm9sZSI6ImFub24iLCJpYXQiOjE3Nzg5NDI1NTIsImV4cCI6MjA5NDUxODU1Mn0.8RaSu5yiTpQCEUGKrFO2y6oiRXSqdBehIz933j9Z7WA"

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

class Message(BaseModel):
    text: str
    history: list = []
    image: str = None
    image_type: str = None
    pdf: str = None

class SolveRequest(BaseModel):
    question: str
    option_a: str = ""
    option_b: str = ""
    option_c: str = ""
    option_d: str = ""
    correct_answer: str = ""

def get_embedding(text: str):
    response = openai_client.embeddings.create(
        model="text-embedding-3-small",
        input=text
    )
    return response.data[0].embedding

def search_ncert(query: str, limit: int = 5):
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

def stream_response(text: str, history: list = [], image: str = None, image_type: str = None, pdf: str = None):
    print(f"stream_response called - text: {text[:50]}, image: {bool(image)}")
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

        print(f"Image received: {bool(image)}, PDF received: {bool(pdf)}")
    if image:
        messages.append({
            "role": "user",
            "content": [
                {
                    "type": "image",
                    "source": {
                        "type": "base64",
                        "media_type": image_type or "image/jpeg",
                        "data": image
                    }
                },
                {
                    "type": "text",
                    "text": user_message
                }
            ]
        })
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
        with client.messages.stream(
            model=selected_model,
            max_tokens=1024,
            system=SYSTEM_PROMPT,
            messages=messages
        ) as stream:
            for text_chunk in stream.text_stream:
                yield text_chunk
    except Exception as e:
        print(f"STREAMING ERROR: {e}")
        yield f"Error: {str(e)}"

@app.post("/solve")
async def solve_question(req: SolveRequest):
    client = anthropic.Anthropic(api_key=ANTHROPIC_KEY)
    message = client.messages.create(
        model="claude-sonnet-4-5",
        max_tokens=1024,
        system="""You are a NEET exam expert. Solve the given NEET question step by step.

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
   - Inline: $formula$ example: $\\frac{1}{2}mv^2$
   - Display: $$formula$$ example: $$E = mc^2$$
   - Always write $H_2O$ not H₂O
   - Always write $v^2$ not v²""",
        messages=[
            {"role": "user", "content": f"Solve this NEET question:\n\nQuestion: {req.question}\n\nA) {req.option_a}\nB) {req.option_b}\nC) {req.option_c}\nD) {req.option_d}\n\nCorrect Answer: {req.correct_answer}"}
        ]
    )
    return {"solution": message.content[0].text}
  

   
@app.post("/chat")
async def chat(message: Message):
    return StreamingResponse(
        stream_response(message.text, message.history, message.image, message.image_type, message.pdf),
        media_type="text/plain"
    )

@app.post("/title")
async def generate_title(message: Message):
    client = anthropic.Anthropic(api_key=ANTHROPIC_KEY)
    response = client.messages.create(
        model="claude-haiku-4-5-20251001",
        max_tokens=15,
        system="Generate a short 3-5 word title for this NEET question. Return ONLY the title. No punctuation. No extra words.",
        messages=[{"role": "user", "content": message.text}]
    )
    return {"title": response.content[0].text}

@app.post("/pyq")
async def get_pyq(message: Message):
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