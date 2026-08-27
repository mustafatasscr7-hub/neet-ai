from fastapi import FastAPI, Header, Depends, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field
from typing import Optional, List, Dict
from contextlib import asynccontextmanager
import asyncio
from concurrent.futures import ThreadPoolExecutor
import anthropic
import requests
import httpx
import openai
from google import genai
from google.genai import types as genai_types
import fitz  # PyMuPDF -- page-count check for PDF tier limits
import base64
import re
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

DEEPSEEK_KEY = os.getenv("DEEPSEEK_KEY")
QWEN_API_KEY = os.getenv("QWEN_API_KEY")
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
OPENAI_KEY = os.getenv("OPENAI_KEY")
SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_KEY = os.getenv("SUPABASE_KEY")
# Service-role key bypasses RLS for admin writes — falls back to the anon key if not set,
# but admin updates may fail under RLS until a real service_role key is added.
SUPABASE_SERVICE_KEY = os.getenv("SUPABASE_SERVICE_KEY", SUPABASE_KEY)
ADMIN_PASSWORD = os.getenv("ADMIN_PASSWORD", "neetai-admin-2027")

openai_client = openai.OpenAI(api_key=OPENAI_KEY)
# Text doubt-answering (/chat, /solve) runs on DeepSeek V4 Flash via its Anthropic-compatible
# endpoint, not Claude -- the `anthropic` package is still used here as an SDK, just pointed at
# a different base_url; there is no live Claude/anthropic.com client left in this file. DeepSeek's
# vision support isn't reliable (confirmed via live testing -- it silently hallucinates on images
# instead of erroring), so it's never used for image-attached questions.
deepseek_client = anthropic.Anthropic(api_key=DEEPSEEK_KEY, base_url="https://api.deepseek.com/anthropic")
# Peak-hour fallback for text doubt-answering only (never images/PDFs) -- confirmed via a live
# 45-question test to be equivalent to DeepSeek on accuracy/reliability, used during DeepSeek's
# own peak-pricing windows (see _is_deepseek_peak_hour) to avoid the peak surcharge. Same
# OpenAI-compatible endpoint pattern already verified live in that test. See
# _stream_with_peak_fallback below for the actual routing/failover logic.
qwen_client = openai.OpenAI(api_key=QWEN_API_KEY, base_url="https://dashscope-intl.aliyuncs.com/compatible-mode/v1")
# Image-attached doubts (/chat) run on Gemini 3.5 Flash-Lite, swapped from Claude Sonnet after
# 4 rounds of live comparison testing: 95-100% accuracy on handwritten doubts, NCERT diagrams,
# organic mechanisms and IUPAC-adjacent naming, both on clean crops and camera-photo-style
# (tilted/blurred/shadowed) images; the one consistent weak spot is graph/curve-matching
# questions (~73% across rounds) -- mitigated, not fixed, by the graph_context prompt addition
# in stream_response() below. PDF-attached doubts moved to Gemini too, after a standalone test
# (91% accuracy across 3 real PDFs, scanned + text-based) confirmed Part.from_bytes handles
# application/pdf the same way it handles images -- see the pdf branch in stream_response().
gemini_client = genai.Client(api_key=GEMINI_API_KEY)
import requests as http_requests
from process_pyq_vision import scan_pdf_bytes, scan_mock_test_pdf

SYSTEM_PROMPT = """You are NEET-AI — an expert tutor for Indian medical entrance exam preparation.

You will be given relevant NCERT content to answer the student's question.

For EVERY answer follow this exact format:

VISUAL_INTENT: [yes or no]

NEET Importance: [N]/5

📚 Chapter: [copied verbatim from a "Retrieved from:" entry given to you — see Rule 1. Omit this whole line if no "Retrieved from:" entries were given.]

📝 Answer:
[Give the answer in clear points]

🔑 Key Points:
- [Point 1]
- [Point 2]
- [Point 3]

🧠 Easy Way to Remember: (see rule 9 below — omit this entire section if no genuinely good mnemonic exists)
[A real, well-known mnemonic if one exists, or a genuinely clever invented one]

For numerical problems, lay out Given/Formula/Solution VERTICALLY, one item per physical line —
NEVER merge values, the formula, or algebra steps into a running paragraph. A paragraph like
"Given u = 20 m/s and g = 10 m/s², using v² = u² - 2gh we get..." is exactly what NOT to do, even
though the same numbers and formula are technically present — each given value, the formula, and
each algebra step must each start on their own new line, not be woven into sentence prose. Follow
this exact worked example's layout (content differs per problem, the one-item-per-line shape does
not):

Given:
$u = 20$ m/s (initial velocity)
$g = 10$ m/s$^2$ (deceleration while rising)
$v = 0$ (velocity at maximum height)

Formula:
$$v^2 = u^2 - 2gh$$

Solution:
$0 = (20)^2 - 2(10)h$
$0 = 400 - 20h$
$h = 400 / 20$
$h = 20$ m

Answer: Maximum height = 20 m

Rules:
1. Answer ONLY from the NCERT content provided to you. This applies most strictly to the 📚
   Chapter: line — follow this exact mechanical procedure for it, in order, before writing
   anything else in that line:
   STEP A: Look at the user message. Does it literally contain the text "Retrieved from:"?
   STEP B: If NO — stop immediately, do not proceed to step C, do not try to recall the chapter
   from your own knowledge no matter how confident you are. Write "📚 Chapter: Not available —
   answering from general knowledge, not a specific retrieved NCERT chapter." (or omit the
   Chapter line entirely) and move on to the rest of the answer.
   STEP C: If YES — copy ONE of the listed entries into the Chapter line EXACTLY as given (same
   class number, same chapter number, same chapter name, character-for-character); if several
   entries are listed, pick whichever single one is most relevant to the question, but the text
   you write must still be copied verbatim from one of them, not reconstructed or blended.
   This two-step check (does "Retrieved from:" literally appear or not) is the ONLY thing that
   decides what you write — never your own confidence in what the answer "should" be. This
   matters even more for non-English questions, since NCERT content search currently only covers
   English and often returns nothing for a question asked in Hindi — a confident-sounding but
   fabricated citation is worse than honestly saying none was found.
2. Always show the NEET Importance rating AT THE TOP
3. Use bullet points — never big paragraphs
4. Answer length should match question complexity
5. If question is outside NCERT say: This is outside the NCERT NEET syllabus.
6. Rate importance CRITICALLY, not leniently. Every doubt you're asked is already restricted to
   real NCERT content (rule 1), so "is this in the NCERT syllabus" is NOT a useful signal — nearly
   everything passes that bar and rating it that way is why every answer ends up at 5/5. Instead
   judge on two things:
   - How often THIS SPECIFIC concept (not just its chapter) has actually appeared as a direct
     question in past NEET/AIPMT papers, based on your own knowledge of real exam patterns.
   - Whether it's a headline/core concept its chapter is built around (e.g. Krebs cycle, Ohm's
     law, Mendelian ratios) vs a peripheral supporting detail, footnote-level mention, or an
     obscure exception/classification within that chapter.
   Most doubts should land in the 2-4 range — reserve 5/5 for concepts that are both core to their
   chapter AND have a well-established history of direct questions almost every year, and reserve
   1/5 for real NCERT content that is genuinely peripheral, rarely if ever the direct basis of an
   actual question even though it's technically in the textbook.
   5/5 = core concept, asked almost every year
   4/5 = well-established concept, asked frequently (most years)
   3/5 = a real but secondary concept, asked occasionally
   2/5 = a peripheral/minor detail, rarely the basis of a question
   1/5 = genuinely obscure NCERT content, almost never tested directly
7. For ALL math formulas and equations use KaTeX format:
   - Inline math: $formula$ — example: $\\frac{1}{2}mv^2$
   - Display math: $$formula$$ — example: $$E = mc^2$$
   - Always write: $\\frac{1}{2}mv^2$ NOT ½mv²
   - Always write: $v^2 = u^2 + 2as$ NOT v² = u² + 2as
   - Always write: $F = ma$ for all formulas
   - Subscripts: $H_2O$ NOT H₂O
   - Superscripts: $x^2$ NOT x²
   - This applies even to a short formula mentioned parenthetically inside a sentence -- write
     "the growth rate $R = \\frac{dN}{dt}$" NOT "the growth rate (R = \\frac{dN}{dt})". Raw LaTeX
     commands wrapped in plain parentheses instead of dollar signs render as broken literal text,
     not math -- every piece of math needs its own $...$ or $$...$$, no exceptions for brevity.
8. If the student explicitly asks for a flowchart, process map, or step-by-step visualization of
   a PROCESS or SEQUENCE (e.g. "make me a flowchart of the steps in photosynthesis"), output it
   as a Mermaid flowchart in a ```mermaid code block — never say you can't create visual diagrams,
   and never draw one using plain text arrows (→, ↓) or ASCII boxes instead.
   Do NOT do this if you classified VISUAL_INTENT: yes above (rule 10) — that means the student
   wants to SEE a real NCERT figure/structure, which is handled entirely by a separate diagram-
   matching system, not by you inventing one. Inventing your own flowchart for a request that's
   really asking to see a real diagram (e.g. "show eubacteria", "what does the cell look like")
   is redundant with that system and confuses the student with two different "diagrams" for the
   same answer. If VISUAL_INTENT is yes, skip this rule entirely regardless of how the question is
   phrased.

   Build it for NEET exam prep, not as a literal restatement of the answer text turned into boxes:
   - Structure it around what NEET actually tests about this topic — the order examiners ask about
     ("which stage comes right after X"), a classification hierarchy that shows up in MCQs, or a
     cause-effect chain leading to a tested outcome. Don't just convert every sentence of the
     answer into a box; pick the structure a student would actually need to recall in an exam.
   - Node labels are short recall cues — a name, a key term, a short phrase — not full sentences.
     If a label needs more than about 4-5 words to say what happens, that detail belongs in the
     surrounding answer text, not squeezed into a box.
   - A quick fact about a step (an amount, a "used"/"formed" note, a short qualifier) goes INSIDE
     that step's own node label — e.g. `B[Glucose-6-phosphate — 1 ATP used]` — or on the single
     edge leading into it, e.g. `A -->|1 ATP used| B`. Never invent a separate node or edge whose
     only purpose is to hold that annotation.
   - Every edge connects two DIFFERENT step-nodes that are genuinely part of the sequence. A node
     is never its own source and target (no `B --> B`, no `B -.-> B`, labeled or not, solid or
     dashed) — there is no situation where that's correct, it always renders as a broken-looking
     loop. And never draw a second edge between a pair of nodes that's already connected — one
     edge per connection, with the label (if any) on that same edge.
   - Keep node label text plain — no LaTeX/KaTeX escaping (`\(`, `\)`, `$...$`) inside a node's
     brackets. Rule 7's math formatting is for the answer text outside the diagram; inside a
     Mermaid label it just renders as literal stray backslashes. Write a formula as plain text
     (`H2O`, `2 ATP`) or a plain-text label instead.
   - Where it's genuinely how NEET tests this topic, work in the specific thing that trips students
     up — a branch point examiners like to test (e.g. leading vs lagging strand), a named exception,
     a count that's a frequent MCQ answer (net ATP yield, number of sub-stages) — as a short branch
     or label, not a wall of text. Don't add this if the topic doesn't have one; an empty "gotcha"
     is worse than none.
   - Do NOT force a flowchart onto content that isn't genuinely sequential, hierarchical, or
     branching. A side-by-side comparison between two things ("compare mitosis and meiosis") is a
     comparison, not a process — cover it with a table or plain text in the normal answer instead of
     wiring two unrelated lists together into a fake flowchart. Same principle as not forcing a weak
     mnemonic (rule 9): no flowchart is better than a forced one that doesn't actually map onto a
     real sequence/hierarchy.
   Syntax example (illustrating structure/conventions only — the actual content must come from
   the topic, not be imitated from this example):
   ```mermaid
   flowchart TD
       A([Start]) --> B{Decision point?}
       B -->|Case one| C[Short outcome]
       B -->|Case two| D[Short outcome]
       C --> E([End])
       D --> E
   ```
   - Use `flowchart TD` (top-down) unless a left-right layout genuinely fits better (`flowchart LR`)
   - Use `[Rectangle]` for a step, `{Diamond}` for a branch/decision, `([Rounded])` for start/end
   - NEVER add `style`, `classDef`, `fill:`, or any other manual color/styling directives —
     the app themes the diagram automatically to match its own dark/light mode. A manually
     picked fill color fights that theming and reliably produces invisible or barely-readable
     text (e.g. light node text on a light fill you chose). Structure and labels only.
   - Still follow the normal answer format around it (NEET Importance, Chapter, etc.) — the
     diagram supplements the answer, it doesn't replace that structure
9. For "Easy Way to Remember": recall the REAL, well-known mnemonic actual NEET/coaching students
   use for this fact and give that (e.g. "King Philip Came Over For Good Soup" for taxonomic
   ranks, "OIL RIG" or "LEO says GER" for oxidation/reduction electron transfer, "Roy G. Biv" for
   the visible spectrum, "Never Eat Shredded Wheat" for compass directions) — almost every named
   list, sequence, or classification in the NCERT syllabus already has one in real use, so expect
   to find one, don't assume there isn't. Only invent your own if you're confident no real one
   exists, and only if it forms an actual memorable word/phrase with a genuine insight behind it
   (not a random acronym restating the letters).
   Actively look for reasons to SKIP this section rather than reasons to fill it — an acronym
   whose "explanation" just re-reads the acronym back out letter by letter (e.g. inventing "KUBU"
   for kidney→ureter→bladder→urethra and then explaining it as "K-U-B-U: kidney, ureter, bladder,
   urethra") is exactly the weak, forced pattern to avoid, and is worse than having no mnemonic at
   all. A short sequence that's already self-evident from its own logic (urine physically flows
   through the organs in the order they're connected; that IS the explanation, it needs no acronym
   on top) doesn't need a memory trick — skip the section for those rather than manufacture one.
   Give AT MOST ONE mnemonic for the entire answer, even when the topic has several sub-facts that
   could each get their own (e.g. one for a sequence, another for a products list, another for a
   directional rule) — pick the single most useful hook for the whole answer and stop there. Never
   stack multiple mnemonics in the same section.
   NEVER add a parenthetical or trailing sentence explaining your choice — for example, never
   write anything like "(no separate mnemonic needed since it's a natural order)" or "(there's no
   standard acronym, so this sequence is the memory aid)" or "this works because the letters spell
   out the steps." This applies in every language the answer is written in, not just English — the
   same banned pattern in Hindi (e.g. "स्वाभाविक प्रवाह क्रम होने से अलग से mnemonic की आवश्यकता
   नहीं" — "no separate mnemonic needed since it's a natural flow order") is just as forbidden as
   its English equivalent. If you decide to skip a mnemonic, the ENTIRE section — heading and all —
   must not appear at all: no explanation, no acknowledgment that one was considered, nothing. If
   you give a mnemonic, it must stand alone with zero commentary about why you chose it, why it
   works, or why alternatives weren't used.
10. VISUAL_INTENT must be the VERY FIRST LINE of your response, before anything else — not
    mentioned later, not skipped. Classify: "yes" if the student is explicitly asking to SEE,
    view, or be shown something visual (e.g. "show me the structure of X", "what does X look
    like", "draw/diagram of X") — "no" if it's a conceptual, factual, or process question that
    merely relates to a topic that happens to have a diagram (e.g. "explain how X works", "why is
    X true", "describe the function of X"). This is a hidden backend signal only — it controls
    whether a diagram is automatically shown to the student, it is never explained or referred to
    in the answer itself.
    If you classify VISUAL_INTENT: yes, two things follow for the rest of your answer:
    - Do NOT generate a Mermaid flowchart (rule 8) under any circumstance, no matter how the
      question is phrased (e.g. even "...and label its parts" or "...show it as a diagram") — a
      real diagram lookup handles this request, a self-drawn flowchart is never a substitute for
      it, and this holds regardless of exact wording.
    - Do NOT describe, narrate, or "decode" what a specific diagram/image will show, and do NOT
      make ANY claim about its display/delivery status in ANY wording or tense — you do not know
      whether a real diagram will actually be found for this question, you have no access to what
      it depicts or labels if one is found, and you have no idea whether it will end up auto-shown
      or behind a click — that decision happens in a separate system after you finish writing, with
      zero visibility into your answer. This is a blanket rule on the underlying CLAIM, not a list
      of banned sentences to avoid — rewording, softening, or hedging the same claim is still the
      same violation. All of the following are equally prohibited, however phrased: "the diagram is
      being shown/displayed now", "a diagram will be shown/displayed separately", "you'll see it
      below/above", "the system is displaying it for you", "check the diagram for this" — say NONE
      of this, in any form, even if the student has asked to see it before or repeatedly in this
      same conversation; repetition doesn't give you any more visibility into what the diagram
      system will do than a first-time request does. Also never build a table or list claiming to
      be the labelled parts visible in that specific image — that is a fabricated guess presented
      as fact, and it will often be flatly wrong about the real uploaded image. Explaining the
      topic's real structure/features as normal educational text is still fine and expected either
      way (e.g. "eubacteria have a cell wall, flagellum, ...") — just never claim or imply a
      picture is or will be on-screen, in this answer or delivered any other way.
11. BEFORE writing VISUAL_INTENT or anything else, run this exact check as a mandatory first
    step, every time — do not skip it, do not decide it doesn't apply without actually checking.
    This should trigger RARELY, but "rarely" describes the overall outcome across many doubts,
    not a reason to talk yourself out of a case that actually qualifies — if a doubt matches the
    TOPIC AMBIGUITY test below, you MUST trigger it, full stop, even though most other doubts
    won't match.

    TOPIC AMBIGUITY — closed whitelist, not a judgment call: only trigger AMBIGUOUS: yes for a
    bare word or short phrase if it matches one of these PRE-APPROVED ambiguous terms, exactly:
    resistance, cycle, potential, diffusion, current, valence. For ANY other word or phrase not on
    this exact list — including "reflex," "reflection," or anything else — NEVER trigger
    AMBIGUOUS, answer directly instead. Do not reason about whether an unlisted word might be
    ambiguous, do not evaluate it against what makes the listed words ambiguous, do not extend
    this list yourself. If it's not on the list, it's not ambiguous, full stop — this applies
    whether the doubt is in English or Hindi (match on meaning, e.g. "प्रतिरोध" matches
    "resistance", "चक्र" matches "cycle").
    A doubt that also fails to be a bare word/short phrase — a full question, a sentence with a
    verb and clear subject, even if phrased informally or with filler like "please," "I don't
    understand," "step by step" — never triggers this rule either, regardless of whether it
    contains a whitelisted word somewhere in it. "रक्त का थक्का कैसे बनता है स्टेप बाय स्टेप
    समझा दो" is a full question about one Biology process, not a bare word — not ambiguous.
    For each whitelisted word, these are its real distinct meanings — use these, don't invent
    others: "resistance" (Physics: electrical resistance / Biology: peripheral resistance in
    blood flow), "cycle" (Biology: cell cycle, Krebs cycle, menstrual cycle, nitrogen cycle),
    "potential" (Physics: electric potential / Biology: action potential, resting potential),
    "diffusion" (Biology: passive transport across membranes / Physics-Chemistry: diffusion of
    gases/molecules), "current" (Physics: electric current / Biology: current used loosely for
    flow, e.g. blood flow or transpiration stream — if no genuine second meaning applies to this
    specific doubt, answer it directly as electric current instead of forcing a clarification),
    "valence" (Chemistry: valence electrons/valency).

    FORMAT AMBIGUITY — check in this order, STEP 1 before STEP 2, every time:
    STEP 1, mechanical bypass, not a judgment call: does the doubt contain an explicit
    visual-request verb or phrase — "show me", "show", "diagram of", "picture of", "image of",
    "draw", "display", "what does X look like", or a clear equivalent (including non-English,
    e.g. "दिखाओ", "चित्र/तस्वीर दिखाओ")? If yes, this is NOT format-ambiguous, full stop — do not
    proceed to STEP 2, go straight to VISUAL_INTENT (rule 10), which will correctly classify this
    as VISUAL_INTENT: yes. This mirrors TOPIC AMBIGUITY's own closed-whitelist test above: a
    mechanical text-pattern match, not a holistic judgment about whether the doubt "feels"
    ambiguous overall.
    STEP 2, only reached if STEP 1 was no: it's genuinely unclear whether the student wants a
    text explanation or to SEE a diagram, AND a diagram has been confirmed to exist for this
    topic (only true if the user message contains the "[A relevant NCERT diagram exists for this
    topic...]" note — never offer this without that note present, even if you're sure a diagram
    should exist). Format ambiguity is ONLY for genuinely neutral phrasing that doesn't lean
    either way, e.g. a bare topic name like "eubacteria structure" with no verb indicating
    explain-vs-show (STEP 1 already ruled out the "show me X" case, so anything reaching STEP 2
    is, by construction, verb-free).

    If EITHER kind of ambiguity applies, do not answer normally at all — do not write
    VISUAL_INTENT or any of the normal answer format. Instead output EXACTLY this format and
    nothing else:

    AMBIGUOUS: yes
    CLARIFY_TYPE: [topic or format]
    QUESTION: [one short, plain-text question, no emoji]
    OPTION: [first interpretation, plain text, no emoji]
    OPTION: [second interpretation, plain text, no emoji]
    [OPTION: third interpretation, if a genuinely distinct third meaning exists]
    [OPTION: fourth interpretation, if a genuinely distinct fourth meaning exists]

    For CLARIFY_TYPE: topic — each OPTION is a short, specific label for one real interpretation
    (e.g. "Electrical resistance (Physics)" / "Peripheral resistance (Biology)") — 2 to 4
    options, only as many as there are genuinely distinct real meanings, never padded to a fixed
    count.
    For CLARIFY_TYPE: format — there are ALWAYS exactly two options, worded EXACTLY like this,
    in this order: "OPTION: Explain it" then "OPTION: Show me the diagram" — do not reword
    these two, the app matches on this exact text to decide what happens next.

    If NEITHER kind of ambiguity applies (the overwhelming majority of doubts), ignore this rule
    entirely and answer in the normal format starting with VISUAL_INTENT, exactly as before. If
    a doubt happens to qualify as ambiguous in BOTH ways at once, resolve TOPIC ambiguity only —
    never output two clarification questions in the same turn; once the student picks a topic
    interpretation, judge format ambiguity fresh on that follow-up if it still applies."""

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
    pyq_id: str = ""
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

class ReportQuestionRequest(BaseModel):
    pyq_id: str
    user_id: str
    reason: str
    optional_note: str = ""

class ReportDiagramRequest(BaseModel):
    diagram_id: int
    user_id: str
    source: str  # 'chat' or 'library' -- the two places a standalone diagrams-table row is shown
    reason: str
    optional_note: str = ""

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
    subject: Optional[str] = None
    chapter: Optional[str] = None
    correct_answer: Optional[str] = None
    is_active: Optional[bool] = None
    question: Optional[str] = None
    option_a: Optional[str] = None
    option_b: Optional[str] = None
    option_c: Optional[str] = None
    option_d: Optional[str] = None
    year: Optional[int] = None
    source_tag: Optional[str] = None
    class_: Optional[int] = Field(None, alias="class")
    diagram_url: Optional[str] = None
    option_a_diagram_url: Optional[str] = None
    option_b_diagram_url: Optional[str] = None
    option_c_diagram_url: Optional[str] = None
    option_d_diagram_url: Optional[str] = None
    reviewed: Optional[bool] = None

class AdminPyqBulkUpdate(BaseModel):
    ids: List[str]
    chapter: Optional[str] = None
    is_active: Optional[bool] = None

class ClassifyDifficultyRequest(BaseModel):
    table: str  # "pyq" or "mock_test_questions" -- the only two tables difficulty exists on
    ids: List[str]

class SetUserPlanRequest(BaseModel):
    user_id: str
    plan: str  # "free" or "pro"

class ScanPdfRequest(BaseModel):
    subject: str
    data: str  # base64-encoded PDF bytes
    filename: Optional[str] = None  # original filename, for processed_pdfs tracking

class PdfMarkProcessedRequest(BaseModel):
    id: int
    status: str  # 'completed' or 'failed'
    questions_extracted: Optional[int] = None
    error_message: Optional[str] = None

class DiagramUploadRequest(BaseModel):
    filename: str
    data: str  # base64-encoded image bytes
    media_type: str = "image/png"

class DiagramCreate(BaseModel):
    subject: str
    class_: Optional[int] = Field(None, alias="class")
    chapter: str
    subtopic: Optional[str] = None
    # Optional Hindi counterpart to subtopic -- same shared-across-the-batch field as subtopic
    # itself, just the Hindi half of it, filled in side-by-side at upload time.
    subtopic_hi: Optional[str] = None
    name: str
    description: Optional[str] = None
    # Optional Hindi counterparts, filled in side-by-side with the English fields above at
    # upload time -- never required, same as description isn't.
    name_hi: Optional[str] = None
    description_hi: Optional[str] = None
    image_url: str

class DiagramUpdate(BaseModel):
    subject: Optional[str] = None
    class_: Optional[int] = Field(None, alias="class")
    chapter: Optional[str] = None
    subtopic: Optional[str] = None
    subtopic_hi: Optional[str] = None
    name: Optional[str] = None
    description: Optional[str] = None
    name_hi: Optional[str] = None
    description_hi: Optional[str] = None
    reviewed: Optional[bool] = None

class DiagramMatchRequest(BaseModel):
    text: str   # the student's doubt text (same text /chat embeds for NCERT RAG)
    answer: str = ""  # the AI's full answer, for chapter-line extraction

class SummarizeAnswerRequest(BaseModel):
    text: str   # the AI's last full answer, to condense
    language: str = "en"
    user_id: str = ""

class PyqQuestionCreate(BaseModel):
    # Only ever set by the admin tool's undo-delete restore path -- normal question creation
    # (PDF scan review, scanned-paste parser) omits it so Postgres assigns a fresh id as before.
    id: Optional[str] = None
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
    # Also undo-restore-only -- lets a restored row keep its prior review/difficulty state
    # instead of coming back reset to "needs review" every time.
    reviewed: Optional[bool] = None
    difficulty: Optional[str] = None

class PyqBulkCreate(BaseModel):
    questions: List[PyqQuestionCreate]

class PageRange(BaseModel):
    start: int
    end: int

class MockTestScanRequest(BaseModel):
    title: str
    filename: Optional[str] = None
    data: str  # base64-encoded PDF bytes
    ranges: Dict[str, PageRange]  # keys: "Physics", "Chemistry", "Biology"

class MockTestMarkProcessedRequest(BaseModel):
    id: int
    status: str  # 'completed' or 'failed'
    questions_extracted: Optional[int] = None
    physics_count: Optional[int] = None
    chemistry_count: Optional[int] = None
    biology_count: Optional[int] = None
    error_message: Optional[str] = None

class MockTestPublishRequest(BaseModel):
    id: int

class MockTestCreateRequest(BaseModel):
    title: str

class MockTestQuestionCreate(BaseModel):
    mock_test_id: int
    question_order: int
    subject: str
    chapter: Optional[str] = None
    class_: Optional[int] = Field(None, alias="class")
    question: str
    option_a: str
    option_b: str
    option_c: str
    option_d: str
    correct_answer: str = ""
    question_type: str = "mcq"
    year: Optional[int] = None
    has_diagram: bool = False
    diagram_url: Optional[str] = None
    option_a_diagram_url: Optional[str] = None
    option_b_diagram_url: Optional[str] = None
    option_c_diagram_url: Optional[str] = None
    option_d_diagram_url: Optional[str] = None

class MockTestBulkCreate(BaseModel):
    questions: List[MockTestQuestionCreate]

class MockTestQuestionUpdate(BaseModel):
    subject: Optional[str] = None
    chapter: Optional[str] = None
    class_: Optional[int] = Field(None, alias="class")
    question: Optional[str] = None
    option_a: Optional[str] = None
    option_b: Optional[str] = None
    option_c: Optional[str] = None
    option_d: Optional[str] = None
    correct_answer: Optional[str] = None
    year: Optional[int] = None
    has_diagram: Optional[bool] = None
    diagram_url: Optional[str] = None
    option_a_diagram_url: Optional[str] = None
    option_b_diagram_url: Optional[str] = None
    option_c_diagram_url: Optional[str] = None
    option_d_diagram_url: Optional[str] = None

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

# ---------- Daily AI usage budget (per-user, rolling 24h cooldown from when the limit is hit) ----------
from datetime import datetime, timezone, timedelta, date

IST = timezone(timedelta(hours=5, minutes=30))  # fixed offset: India has no DST
DAILY_TOKEN_BUDGET_FREE = 50000  # ~9 doubts/day -- re-recalibrated 2026-08-16. The ~400
                                  # tokens/doubt this used to be built on (see git history) relied
                                  # specifically on DeepSeek's Anthropic-compat endpoint auto-
                                  # caching the repeated system prompt/NCERT context, so
                                  # usage.input_tokens only ever reflected the small "fresh"
                                  # remainder. The Qwen-Flash peak-hour fallback (shipped
                                  # 2026-08-14, AFTER that calibration) routes requests to Qwen
                                  # during DeepSeek's own peak-pricing windows -- Qwen's API gets
                                  # no equivalent caching, so input_tokens bills the FULL context
                                  # every time during those windows, exactly the failure mode the
                                  # old comment here warned about ("if DeepSeek's caching behavior
                                  # ... changes materially, re-measure") -- nobody did when the
                                  # fallback shipped, so a logged-in free user could hit "daily
                                  # limit reached" on literally their first message, every message
                                  # costing more than the entire old 3600 budget. Real per-doubt
                                  # cost measured at ~5470 tokens/doubt (pooled n=8 live /chat
                                  # generations against current code, all falling inside a peak
                                  # window -- see provider_usage_log ids 97-105, excluding one
                                  # AMBIGUOUS-clarify response that doesn't bill quota at all;
                                  # range 4509-8782). Calibrated for the peak-hour (worst-case,
                                  # uncached) cost rather than DeepSeek's off-peak cached cost, so
                                  # a student doesn't get fewer doubts just for studying during
                                  # peak hours -- this number is 9 * ~5470, rounded up. If model
                                  # routing or either provider's caching behavior changes again,
                                  # re-measure before trusting this ceiling.
COOLDOWN_HOURS = 24  # how long a block lasts once the budget is actually crossed

def _ist_today() -> str:
    return datetime.now(timezone.utc).astimezone(IST).date().isoformat()

CST = timezone(timedelta(hours=8))  # China Standard Time, fixed offset (no DST) -- DeepSeek's

# weekend-off-peak rule (effective 2026-08-23) is defined in Beijing time specifically, unlike
# the weekday windows below which are published in UTC -- these are two different reference
# timezones for the same pricing page, not an inconsistency to "fix" by converting one to match
# the other.
def _is_deepseek_peak_hour() -> bool:
    """DeepSeek's own published peak-pricing windows are 01:00-04:00 and 06:00-10:00 UTC on
    weekdays. Since 2026-08-23, Saturday and Sunday in BEIJING TIME are always off-peak all day
    (no weekday-style split) -- checked first and short-circuits the UTC-hour check entirely,
    since the weekend rule overrides it regardless of what UTC hour it happens to be. The weekday
    windows themselves stay checked directly against UTC (not IST/server-local), since UTC is the
    timezone they're actually published in and converting would just risk an offset bug for
    zero benefit."""
    now_utc = datetime.now(timezone.utc)
    if now_utc.astimezone(CST).weekday() >= 5:  # 5=Saturday, 6=Sunday
        return False
    hour = now_utc.hour
    return (1 <= hour < 4) or (6 <= hour < 10)

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

# PDF-attachment size/page limits per tier. Only "free" and "pro" are actually reachable today --
# admin_set_user_plan() (see below) only accepts those two values, so "max" is here for when a
# real Max plan value can actually be assigned, not because any user can be "max" yet.
PDF_LIMITS = {
    "free": {"max_mb": 3, "max_pages": 5},
    "pro": {"max_mb": 10, "max_pages": 20},
    "max": {"max_mb": 20, "max_pages": 25},
}
PDF_NEXT_TIER = {"free": "Pro", "pro": "Max", "max": None}

def _pdf_tier_for_plan(plan: str) -> str:
    if plan == "free":
        return "free"
    if plan == "max":
        return "max"
    return "pro"  # covers "pro" and any other non-free/non-max value defensively

async def validate_pdf_limits(pdf_b64: "str | None", user_id: str):
    """Raises HTTPException(413) before any Gemini call if the PDF exceeds the user's tier
    limit -- called from the /chat route handler (not inside stream_response's generator),
    same timing as enforce_daily_budget, so it produces a clean error response instead of
    failing mid-stream."""
    if not pdf_b64:
        return
    plan = await get_user_plan(user_id)
    tier = _pdf_tier_for_plan(plan)
    limits = PDF_LIMITS[tier]
    pdf_bytes = base64.b64decode(pdf_b64)
    size_mb = len(pdf_bytes) / (1024 * 1024)
    try:
        page_count = fitz.open(stream=pdf_bytes, filetype="pdf").page_count
    except Exception:
        page_count = None  # unreadable/corrupt -- let Gemini's own error surface instead of guessing here
    exceeds = size_mb > limits["max_mb"] or (page_count is not None and page_count > limits["max_pages"])
    if exceeds:
        next_tier = PDF_NEXT_TIER[tier]
        message = f"This PDF exceeds your plan's limit ({limits['max_mb']}MB / {limits['max_pages']} pages)."
        if next_tier:
            message += f" Upgrade to {next_tier} for a higher limit."
        raise HTTPException(status_code=413, detail={
            "message": message,
            "max_mb": limits["max_mb"], "max_pages": limits["max_pages"],
            "size_mb": round(size_mb, 2), "pages": page_count,
            "tier": tier, "next_tier": next_tier
        })

# ---------- Device-limit enforcement (Pro = 1 device, Max = 2, Free = unenforced) ----------
# Free is deliberately excluded -- no paid incentive to share a free account, and enforcing this
# adds friction/support burden for zero revenue protection (explicit product decision, not an
# oversight). "None" here means "no limit checked at all", not "unlimited devices tracked".
DEVICE_LIMITS = {"free": None, "pro": 1, "max": 2}
DEVICE_SESSION_ACTIVE_WINDOW_MINUTES = 20  # a session with no heartbeat in this long is treated
                                            # as closed/idle and stops counting against the limit
                                            # -- self-healing, no explicit logout tracking needed.
DEVICE_SESSION_GRACE_PERIOD_MINUTES = int(os.environ.get("DEVICE_SESSION_GRACE_PERIOD_MINUTES_OVERRIDE", "5"))

def _parse_ts(value: str) -> datetime:
    return datetime.fromisoformat(value.replace("Z", "+00:00"))

async def _lookup_session_location(ip: str) -> str:
    """Best-effort "City, Region" label for the Active Sessions list. Free, keyless lookup
    (ip-api.com) with a short timeout -- called only once per session (new device, or a fresh
    relogin on a previously-kicked one), never on routine heartbeats, since the result is cached
    on the row. Any failure (timeout, private/local IP during dev, service error) degrades to
    "Unknown" rather than ever blocking or failing the login/heartbeat itself."""
    if not ip or ip in ("unknown", "127.0.0.1", "localhost", "::1") or ip.startswith(("10.", "192.168.", "172.")):
        return "Unknown"
    try:
        resp = await async_client.get(
            f"http://ip-api.com/json/{ip}",
            params={"fields": "status,city,regionName"},
            timeout=3.0
        )
        data = resp.json()
        if data.get("status") != "success":
            return "Unknown"
        city = data.get("city") or ""
        region = data.get("regionName") or ""
        label = ", ".join(p for p in (city, region) if p)
        return label or "Unknown"
    except Exception:
        return "Unknown"

async def _expire_due_grace_periods(rows: list) -> list:
    """Any row whose grace period has passed but isn't marked kicked yet gets kicked_at set now
    -- checked eagerly here (not just lazily whenever THAT device's own next heartbeat happens)
    so a third device registering right as an old grace period lapses sees an accurate, current
    active count instead of briefly overcounting a session that's already effectively expired."""
    now = datetime.now(timezone.utc)
    for r in rows:
        if r.get("kicked_at"):
            continue
        deadline = r.get("kick_grace_deadline")
        if deadline and now > _parse_ts(deadline):
            r["kicked_at"] = now.isoformat()
            try:
                await async_client.patch(
                    f"{SUPABASE_URL}/rest/v1/active_sessions", headers=ADMIN_HEADERS,
                    params={"id": f"eq.{r['id']}"},
                    json={"kicked_at": r["kicked_at"]}
                )
            except Exception:
                pass
    return rows

class SessionHeartbeatRequest(BaseModel):
    user_id: str
    device_id: str
    device_label: str = ""
    is_login: bool = False  # true only for the one call made right after a fresh sign-in (see
                             # chat.html) -- distinguishes "genuine new session, re-evaluate me"
                             # from "routine poll of a device that's already been kicked", which
                             # must keep reporting kicked rather than silently un-kicking itself
                             # the moment that device's own polling happens to run again.

@app.post("/session/heartbeat")
async def session_heartbeat(req: SessionHeartbeatRequest, request: Request):
    """Called once right after login (is_login=True) and then periodically while chat.html stays
    open (is_login=False), so a kicked device finds out within one polling cycle rather than only
    whenever it next happens to send a /chat message.

    Warning-then-kick, never an instant kick: a login that would exceed the plan's device limit
    doesn't block itself or immediately kick anything -- it starts a grace period against the
    OLDEST other active session and tells the NEW device to show a countdown. Only once that
    grace period actually lapses does the old session get marked kicked, discovered on its own
    next heartbeat (or proactively by _expire_due_grace_periods if a third check-in lands first).
    Fails open throughout: any lookup/write error here returns "ok" rather than blocking a real
    login over this table's own availability."""
    if not req.user_id or not req.device_id:
        return {"status": "ok"}
    try:
        plan = await get_user_plan(req.user_id)
        limit = DEVICE_LIMITS.get(plan)
        if limit is None:
            return {"status": "ok"}

        now = datetime.now(timezone.utc)
        active_cutoff = now - timedelta(minutes=DEVICE_SESSION_ACTIVE_WINDOW_MINUTES)

        resp = await async_client.get(
            f"{SUPABASE_URL}/rest/v1/active_sessions", headers=ADMIN_HEADERS,
            params={"user_id": f"eq.{req.user_id}", "select": "*"}
        )
        rows = resp.json() if resp.status_code == 200 else []
        rows = await _expire_due_grace_periods(rows)
        existing = next((r for r in rows if r["device_id"] == req.device_id), None)

        # A routine poll (not a fresh login) of an already-kicked device always just reports
        # kicked again -- never silently un-kicks itself just because it happened to check in.
        if existing and existing.get("kicked_at") and not req.is_login:
            return {"status": "kicked"}

        if existing and not existing.get("kicked_at"):
            # Ordinary heartbeat: same device, still in good standing.
            await async_client.patch(
                f"{SUPABASE_URL}/rest/v1/active_sessions", headers=ADMIN_HEADERS,
                params={"id": f"eq.{existing['id']}"},
                json={"last_active_at": now.isoformat(), "device_label": req.device_label or existing.get("device_label")}
            )
            return {"status": "ok"}

        if existing and existing.get("kicked_at") and req.is_login:
            # A previously-kicked device logging in again for real -- clean slate. created_at
            # resets too: this is a brand new session as far as seniority/grace-period targeting
            # is concerned, it shouldn't inherit standing from the session that got kicked.
            # Location is re-resolved too -- a genuine relogin may well be from a different place.
            location = await _lookup_session_location(_client_ip(request))
            patch_resp = await async_client.patch(
                f"{SUPABASE_URL}/rest/v1/active_sessions",
                headers={**ADMIN_HEADERS, "Content-Type": "application/json", "Prefer": "return=representation"},
                params={"id": f"eq.{existing['id']}"},
                json={"kicked_at": None, "kick_grace_deadline": None, "created_at": now.isoformat(),
                      "last_active_at": now.isoformat(), "device_label": req.device_label or existing.get("device_label"),
                      "location": location}
            )
            working_row = patch_resp.json()[0] if patch_resp.status_code == 200 else {"id": existing["id"], "device_id": req.device_id, "created_at": now.isoformat()}
        else:
            # Brand-new device for this user.
            location = await _lookup_session_location(_client_ip(request))
            insert_resp = await async_client.post(
                f"{SUPABASE_URL}/rest/v1/active_sessions",
                headers={**ADMIN_HEADERS, "Content-Type": "application/json", "Prefer": "return=representation"},
                json={"user_id": req.user_id, "device_id": req.device_id, "device_label": req.device_label or "Unknown device",
                      "location": location}
            )
            if insert_resp.status_code not in (200, 201):
                return {"status": "ok"}
            working_row = insert_resp.json()[0]

        active_others = [
            r for r in rows
            if r["device_id"] != req.device_id
            and not r.get("kicked_at")
            and r.get("last_active_at") and _parse_ts(r["last_active_at"]) > active_cutoff
        ]
        if len(active_others) + 1 <= limit:
            return {"status": "ok"}

        active_others.sort(key=lambda r: r["created_at"])
        oldest = active_others[0]
        if oldest.get("kick_grace_deadline"):
            deadline = oldest["kick_grace_deadline"]
        else:
            deadline = (now + timedelta(minutes=DEVICE_SESSION_GRACE_PERIOD_MINUTES)).isoformat()
            await async_client.patch(
                f"{SUPABASE_URL}/rest/v1/active_sessions", headers=ADMIN_HEADERS,
                params={"id": f"eq.{oldest['id']}"},
                json={"kick_grace_deadline": deadline}
            )
        return {"status": "warning", "grace_period_ends_at": deadline, "plan": plan, "device_limit": limit}
    except Exception as e:
        print(f"SESSION HEARTBEAT ERROR (failing open): {e}", flush=True)
        return {"status": "ok"}

class SessionListRequest(BaseModel):
    user_id: str
    device_id: str = ""  # used only to flag which row is "this" session in the response

@app.post("/session/list")
async def session_list(req: SessionListRequest):
    """Powers the Active Sessions table in chat.html's Account settings tab. Returns only
    sessions that are actually active right now (same window used to enforce the device limit,
    so this list matches what's really counted against it) -- kicked-out and long-idle rows are
    left out rather than accumulating forever. Sorted most-recently-active first, per spec."""
    if not req.user_id:
        return {"sessions": []}
    try:
        active_cutoff = datetime.now(timezone.utc) - timedelta(minutes=DEVICE_SESSION_ACTIVE_WINDOW_MINUTES)
        resp = await async_client.get(
            f"{SUPABASE_URL}/rest/v1/active_sessions", headers=ADMIN_HEADERS,
            params={"user_id": f"eq.{req.user_id}", "select": "*"}
        )
        rows = resp.json() if resp.status_code == 200 else []
        active = [
            r for r in rows
            if not r.get("kicked_at")
            and r.get("last_active_at") and _parse_ts(r["last_active_at"]) > active_cutoff
        ]
        active.sort(key=lambda r: r["last_active_at"], reverse=True)
        return {"sessions": [
            {
                "device_id": r["device_id"],
                "device_label": r.get("device_label") or "Unknown device",
                "location": r.get("location") or "Unknown",
                "created_at": r["created_at"],
                "last_active_at": r["last_active_at"],
                "is_current": r["device_id"] == req.device_id,
            } for r in active
        ]}
    except Exception as e:
        print(f"SESSION LIST ERROR: {e}", flush=True)
        return {"sessions": []}

class SessionLogoutRequest(BaseModel):
    user_id: str
    target_device_id: str  # the session being logged out, may be the caller's own device

@app.post("/session/logout")
async def session_logout(req: SessionLogoutRequest):
    """Manual logout from the Active Sessions list -- an immediate kick (no grace period; the
    student explicitly chose this, unlike the automatic device-limit kick). Reuses the exact same
    kicked_at mechanism, so the target device is discovered and signed out via its own next
    heartbeat exactly like an over-limit kick, and the freed slot is picked up automatically by
    the device-limit check on any subsequent login (no separate bookkeeping needed)."""
    if not req.user_id or not req.target_device_id:
        raise HTTPException(status_code=400, detail="user_id and target_device_id are required")
    try:
        resp = await async_client.patch(
            f"{SUPABASE_URL}/rest/v1/active_sessions",
            headers={**ADMIN_HEADERS, "Content-Type": "application/json", "Prefer": "return=representation"},
            params={"user_id": f"eq.{req.user_id}", "device_id": f"eq.{req.target_device_id}"},
            json={"kicked_at": datetime.now(timezone.utc).isoformat()}
        )
        rows = resp.json() if resp.status_code == 200 else []
        if not rows:
            raise HTTPException(status_code=404, detail="Session not found")
        return {"status": "ok"}
    except HTTPException:
        raise
    except Exception as e:
        print(f"SESSION LOGOUT ERROR: {e}", flush=True)
        raise HTTPException(status_code=500, detail="Failed to log out session")

DAILY_TOKEN_BUDGET_GUEST = 14000  # ~2.5 doubts/day -- deliberately tight vs. the logged-in free
                                  # tier: the goal is to force a login, not to be a usable tier on
                                  # its own. Same 2026-08-16 re-recalibration and same root cause
                                  # as DAILY_TOKEN_BUDGET_FREE above (see its comment) -- guests
                                  # had it worse: at the old 1000-token budget, even the CHEAPEST
                                  # real answer measured (4509 tokens) exceeded the entire daily
                                  # allowance, so a guest could never get a single answer, ever.
                                  # This is 2.5 * the same measured ~5470 tokens/doubt, keeping the
                                  # same ratio to the free tier's budget as before.

# usage_log/guest_usage_log stay keyed by (id, usage_date) exactly as before -- tokens_used still
# resets to 0 on a fresh calendar day. What changed is WHEN a block clears: previously that was
# "whenever IST midnight next occurs" (0 minutes to ~24h depending on what time of day the limit
# was hit); now it's a fixed 24h from the moment limit_reached_at was set, regardless of the
# calendar-day boundary in between. Checking today's row plus yesterday's is sufficient: a 24h
# window that starts anywhere "today" can only still be active sometime "tomorrow", never further.
async def _fetch_usage_rows(table: str, id_field: str, id_value: str):
    """Returns (today_str, yesterday_str, rows_by_date) for the shared today+yesterday lookup
    both the enforcement check and the read-only status endpoint need."""
    today = _ist_today()
    yesterday = (date.fromisoformat(today) - timedelta(days=1)).isoformat()
    resp = await async_client.get(
        f"{SUPABASE_URL}/rest/v1/{table}", headers=ADMIN_HEADERS,
        params={id_field: f"eq.{id_value}", "usage_date": f"in.({yesterday},{today})",
                "select": "usage_date,tokens_used,limit_reached_at"}
    )
    return today, yesterday, {r["usage_date"]: r for r in resp.json()}

def _active_cooldown_seconds(today: str, yesterday: str, rows_by_date: dict) -> "int | None":
    """Read-only: does today's or yesterday's row have a limit_reached_at still within the
    COOLDOWN_HOURS window? Doesn't decide whether a *new* crossing should start one -- that's
    _check_rolling_cooldown's job, since only the enforcement path should have that side effect."""
    now = datetime.now(timezone.utc)
    for d in (yesterday, today):
        row = rows_by_date.get(d)
        if row and row.get("limit_reached_at"):
            reached = datetime.fromisoformat(row["limit_reached_at"].replace("Z", "+00:00"))
            elapsed = now - reached
            if elapsed < timedelta(hours=COOLDOWN_HOURS):
                return int((timedelta(hours=COOLDOWN_HOURS) - elapsed).total_seconds())
    return None

async def _check_rolling_cooldown(table: str, id_field: str, id_value: str, budget: int) -> "int | None":
    """Returns remaining cooldown seconds if blocked, else None. Setting limit_reached_at on the
    request that newly crosses the budget is a side effect of this check, not a separate step --
    consistent with this file's existing accepted non-atomicity for usage tracking (see
    log_token_usage's own comment on the same tradeoff)."""
    today, yesterday, rows_by_date = await _fetch_usage_rows(table, id_field, id_value)
    active = _active_cooldown_seconds(today, yesterday, rows_by_date)
    if active is not None:
        return active

    today_row = rows_by_date.get(today)
    used_today = today_row["tokens_used"] if today_row else 0
    if used_today >= budget:
        if not (today_row and today_row.get("limit_reached_at")):
            await async_client.patch(
                f"{SUPABASE_URL}/rest/v1/{table}", headers={**ADMIN_HEADERS, "Content-Type": "application/json"},
                params={id_field: f"eq.{id_value}", "usage_date": f"eq.{today}"},
                json={"limit_reached_at": datetime.now(timezone.utc).isoformat()}
            )
        return COOLDOWN_HOURS * 3600
    return None

async def enforce_daily_budget(user_id: str, ip: str = ""):
    if user_id:
        if await get_user_plan(user_id) != "free":
            return  # paid = unlimited for now
        remaining = await _check_rolling_cooldown("usage_log", "user_id", user_id, DAILY_TOKEN_BUDGET_FREE)
        if remaining is not None:
            raise HTTPException(status_code=402, detail={"message": "Daily limit reached", "retry_after_seconds": remaining})
        return
    # Guest (no account): tracked by IP instead, in a separate table -- an IP is a much weaker
    # identity than a user_id (shared behind NAT/campus wifi, changes on mobile networks), so
    # this is approximate, not airtight. Good enough to nudge toward logging in, which is the
    # actual goal here, not perfect anti-abuse.
    if not ip:
        return
    remaining = await _check_rolling_cooldown("guest_usage_log", "ip", ip, DAILY_TOKEN_BUDGET_GUEST)
    if remaining is not None:
        raise HTTPException(status_code=402, detail={"message": "Guest limit reached — log in to continue", "retry_after_seconds": remaining})

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

    # Fire-and-forget: nothing downstream needs this write to finish before the embedding
    # can be used, so awaiting it here only added latency to the critical path for no reason.
    asyncio.create_task(async_client.post(
        f"{SUPABASE_URL}/rest/v1/embedding_cache",
        headers={**headers, "Content-Type": "application/json"},
        json={"question_hash": question_hash, "embedding": embedding}
    ))

    return embedding


# ncert_content rows that are front/back matter (prelims, appendix, answer keys) rather than
# real chapter content -- excluded below rather than given a fake chapter title, since citing
# "Chapter: Appendix" to a student would be actively misleading. Filtered here in Python, not
# in the match_ncert Postgres function itself: this repo has no DDL/DATABASE_URL access to
# safely inspect or edit that function, so excluding post-hoc from a table we can't see is the
# only change we can fully verify ourselves.
#
# Also confirmed a real, separate reason to exclude Appendix from general retrieval (not just
# the citation-honesty reason above): a live diagnostic test found Appendix content -- a
# compiled answer-key/reference block spanning terse references to many different chapters --
# winning top-rank against genuinely unrelated conceptual questions (e.g. Mendel's law of
# segregation top-matching Appendix instead of the real genetics chapter), because it's topically
# diffuse enough to partially resemble almost any query. Excluding it measurably fixed that
# specific failure with no observed downside on irrelevant-query safety.
#
# Tradeoff this doesn't handle: a student directly asking a reference-lookup question the
# Appendix *should* answer (e.g. "what is sodium's atomic mass", a value that literally lives in
# an Appendix table) will no longer find it via this general search. Not fixed here -- if that
# turns out to matter in practice, the right shape is a separate, explicit reference-lookup path
# (detect that class of query and search Appendix-only, rather than lowering this general
# exclusion) rather than letting Appendix back into ordinary conceptual search.
NCERT_NON_CHAPTER_LABELS = {"Preliminary Pages", "Appendix", "Answer Key"}

# Same canonical chapter-per-class ordering as the frontend's neetSyllabus (pyqbank.html and
# friends) -- position in each list = official NCERT chapter number for that subject+class.
# ncert_content itself has no stored chapter number (only subject/class/chapter_name), so a
# live 45-question test found the model guessing/misremembering the number from its own training
# knowledge ~30% of the time even when it correctly copied the chapter NAME from retrieved
# content. This is the fix: compute the real number here, server-side, from data we already
# trust, so the model's job in the "Chapter:" line becomes "copy this string" instead of
# "recall this number" -- see _ncert_chapter_citation and its call site in stream_response().
NEET_SYLLABUS = {
    "Biology": {
        11: ['The Living World', 'Biological Classification', 'Plant Kingdom', 'Animal Kingdom', 'Morphology of Flowering Plants', 'Anatomy of Flowering Plants', 'Structural Organisation in Animals', 'Cell: The Unit of Life', 'Biomolecules', 'Cell Cycle and Cell Division', 'Transport in Plants', 'Mineral Nutrition', 'Photosynthesis in Higher Plants', 'Respiration in Plants', 'Plant Growth and Development', 'Digestion and Absorption', 'Breathing and Exchange of Gases', 'Body Fluids and Circulation', 'Excretory Products and their Elimination', 'Locomotion and Movement', 'Neural Control and Coordination', 'Chemical Coordination and Integration'],
        12: ['Reproduction in Organisms', 'Sexual Reproduction in Flowering Plants', 'Human Reproduction', 'Reproductive Health', 'Principles of Inheritance and Variation', 'Molecular Basis of Inheritance', 'Evolution', 'Human Health and Disease', 'Strategies for Enhancement in Food Production', 'Microbes in Human Welfare', 'Biotechnology: Principles and Processes', 'Biotechnology and its Applications', 'Organisms and Populations', 'Ecosystem', 'Biodiversity and Conservation', 'Environmental Issues']
    },
    "Physics": {
        11: ['Units and Measurements', 'Motion in a Straight Line', 'Motion in a Plane', 'Laws of Motion', 'Work, Energy and Power', 'System of Particles and Rotational Motion', 'Gravitation', 'Mechanical Properties of Solids', 'Mechanical Properties of Fluids', 'Thermal Properties of Matter', 'Thermodynamics', 'Kinetic Theory', 'Oscillations', 'Waves', 'Mathematical Tools'],
        12: ['Electric Charges and Fields', 'Electrostatic Potential and Capacitance', 'Current Electricity', 'Moving Charges and Magnetism', 'Magnetism and Matter', 'Electromagnetic Induction', 'Alternating Current', 'Electromagnetic Waves', 'Ray Optics and Optical Instruments', 'Wave Optics', 'Dual Nature of Radiation and Matter', 'Atoms', 'Nuclei', 'Semiconductor Electronics: Materials, Devices and Simple Circuits']
    },
    "Chemistry": {
        11: ['Some Basic Concepts of Chemistry', 'Structure of Atom', 'Classification of Elements and Periodicity in Properties of Elements', 'Chemical Bonding and Molecular Structure', 'States of Matter: Gases and Liquids', 'Thermodynamics', 'Equilibrium', 'Redox Reactions', 'Hydrogen', 'The s-Block Elements (Alkali and Alkaline Earth Metals)', 'The p-Block Elements', 'Organic Chemistry: Some Basic Principles and Techniques', 'Hydrocarbons', 'Environmental Chemistry'],
        12: ['The Solid State', 'Solutions', 'Electrochemistry', 'Chemical Kinetics', 'Surface Chemistry', 'General Principles and Processes of Isolation of Elements', 'The p-Block Elements', 'The d and f Block Elements', 'Coordination Compounds', 'Haloalkanes and Haloarenes', 'Alcohols, Phenols and Ethers', 'Aldehydes, Ketones and Carboxylic Acids', 'Amines', 'Biomolecules', 'Polymers', 'Chemistry in Everyday Life']
    }
}

# ---------------- Diagram matching (chat.html doubt-solving) ----------------
# The SYSTEM_PROMPT instructs the model to include a line like "📚 Chapter: NCERT Class X,
# Chapter X — Chapter Name" in its answer -- this is the first (and, discovered while fixing
# citation accuracy elsewhere in this file, formerly non-functional) consumer of it. The
# original regex here required literal [bracket] characters around the citation, copying the
# SYSTEM_PROMPT template's placeholder-bracket convention -- but real model output never
# actually includes brackets (confirmed against 91 real "Chapter:" lines from a live test: 0
# matched), so extract_chapter_candidate() always returned None and diagram-matching's
# chapter-filtering step never fired, silently falling through to an unfiltered similarity
# search every time. Fixed to match the rest of the line instead of a bracketed group -- the
# chapter NAME is still whatever follows the last dash-like separator ("—"/"–"/" - ", the model
# isn't perfectly consistent about which one), since "Chapter X" itself isn't a lookup key.
DIAGRAM_CHAPTER_LINE_RE = re.compile(r'chapter:\s*(.+)', re.IGNORECASE)

def extract_chapter_candidate(answer_text: str):
    m = DIAGRAM_CHAPTER_LINE_RE.search(answer_text or '')
    if not m:
        return None
    line = m.group(1).strip().strip('*').strip()
    for sep in ('—', '–', ' - '):
        if sep in line:
            candidate = line.rsplit(sep, 1)[-1].strip()
            if candidate:
                return candidate
    return line.strip() or None

DIAGRAM_CHAPTER_FUZZY_THRESHOLD = 0.3  # same spirit/magnitude as admin-pdf-review.html's FILENAME_MATCH_THRESHOLD

def _tokenize_for_chapter_match(text: str):
    return set(w.lower() for w in re.sub(r'[^a-zA-Z\s]', ' ', text or '').split() if len(w) > 1)

# Token-overlap fuzzy match, same technique as admin-pdf-review.html's matchChapterFromFilename
# (JS) -- the model's free-text chapter name won't always exactly string-match a diagrams.chapter
# value verbatim (e.g. punctuation/wording drift), so an exact-equality filter would miss real
# matches far too often.
def fuzzy_match_chapter(candidate: str, known_chapters: list):
    if not candidate or not known_chapters:
        return None
    candidate_tokens = _tokenize_for_chapter_match(candidate)
    if not candidate_tokens:
        return None
    best, best_score = None, 0.0
    for chapter in known_chapters:
        chapter_tokens = _tokenize_for_chapter_match(chapter)
        if not chapter_tokens:
            continue
        union = candidate_tokens | chapter_tokens
        score = len(candidate_tokens & chapter_tokens) / len(union) if union else 0
        if score > best_score:
            best, best_score = chapter, score
    return best if best_score >= DIAGRAM_CHAPTER_FUZZY_THRESHOLD else None

def build_diagram_embedding_text(name: str, description: str, chapter: str):
    # OPEN QUESTION (raised when the `subtopic` field was added to the diagrams table): should
    # this also take subtopic and fold it in here? Left out deliberately for now -- diagram
    # matching in chat.html is unchanged by that migration, this is intentionally a decision
    # for a human to make, not something to change automatically alongside the schema/admin-UI
    # work that added the column.
    parts = [name]
    if description:
        parts.append(description)
    parts.append(chapter)
    return " ".join(p for p in parts if p)

def build_pyq_embedding_text(question: str, chapter: str = None, option_a: str = None,
                              option_b: str = None, option_c: str = None, option_d: str = None):
    # Same join-non-empty-parts shape as build_diagram_embedding_text above. Options are included
    # (not just the question) because a real fraction of NEET questions are generic on their own
    # ("Select the correct statements.", "Which of the following is true?") -- the actual subject
    # matter often only shows up in the options, so question-only embeddings would under-match
    # exactly the rows most likely to need help matching in the first place.
    parts = [question]
    if chapter:
        parts.append(chapter)
    for opt in (option_a, option_b, option_c, option_d):
        if opt:
            parts.append(opt)
    return " ".join(p for p in parts if p)

DEVANAGARI_RE = re.compile(r'[ऀ-ॿ]')

def _detect_query_language(query: str) -> str:
    """Deliberately simple: any Devanagari character in the query counts as Hindi. No mixed-
    script handling -- a Hinglish (Latin-script Hindi) query still gets filtered as 'en', which
    is correct here since the Hindi ncert_content rows are themselves in Devanagari script, not
    transliterated Latin."""
    return "hi" if DEVANAGARI_RE.search(query or "") else "en"

NCERT_MATCH_THRESHOLD_EN = 0.5
# Hindi embeddings score systematically lower than English for genuinely correct matches --
# same underlying text-embedding-3-small model, but Hindi's much higher tokens/word density
# (see hindi_retrieval_rechunking memory) dilutes the signal even in well-formed, sentence-
# aware chunks. Confirmed via a real 15-correct/7-irrelevant diagnostic against the current
# (sentence-aware) Hindi chunks: correct queries scored 0.4046-0.5684, irrelevant queries
# scored 0.2461-0.3917 -- a clean, non-overlapping gap. 0.40 sits in that gap. Re-measure
# before trusting this if the Hindi chunking strategy changes again (word-count chunking
# measured earlier had NO clean gap at any threshold -- this number is tied to the current
# sentence-aware chunks, not a fixed property of Hindi embeddings in general).
NCERT_MATCH_THRESHOLD_HI = 0.40

async def search_ncert(query: str, limit: int = 3):
    embedding = await get_embedding(query)
    headers = {
        "apikey": SUPABASE_KEY,
        "Authorization": f"Bearer {SUPABASE_KEY}",
        "Content-Type": "application/json"
    }
    language = _detect_query_language(query)
    threshold = NCERT_MATCH_THRESHOLD_HI if language == "hi" else NCERT_MATCH_THRESHOLD_EN
    response = await async_client.post(
        f"{SUPABASE_URL}/rest/v1/rpc/match_ncert",
        headers=headers,
        json={
            "query_embedding": embedding,
            "match_threshold": threshold,
            "match_count": limit,
            "filter_language": language
        }
    )
    if response.status_code == 200:
        # chapter_name_en is what actually carries "Appendix"/etc for Hindi rows (chapter_name
        # itself is the real Hindi title, e.g. "परिशिष्ट", which would never match these English
        # labels) -- falls back to chapter_name for English rows, where chapter_name_en is null.
        return [r for r in response.json() if (r.get("chapter_name_en") or r.get("chapter_name")) not in NCERT_NON_CHAPTER_LABELS]
    return []

def _ncert_chapter_citation(subject: str, class_num, chapter_name: str, lookup_name: str = None) -> str:
    """Builds an authoritative citation string like 'NCERT Class 12, Chapter 3 -- Current
    Electricity' for one retrieved ncert_content row, by looking it up against NEET_SYLLABUS --
    the same canonical, chapter-number-by-position list the frontend uses. This is handed to the
    model as a ready-made "Retrieved from:" line (see stream_response()) so it never has to
    recall/guess the chapter number itself, which is where the real citation errors came from
    (chapter_name is already stored correctly -- only the number was ever a guess).

    chapter_name is what gets DISPLAYED in the citation (English for 'en' rows, real Hindi title
    for 'hi' rows -- a Hindi-answered student should see a Hindi citation line). lookup_name is
    what actually gets matched against NEET_SYLLABUS to find the chapter number -- defaults to
    chapter_name itself (English rows), but Hindi rows must pass chapter_name_en here instead,
    since NEET_SYLLABUS is English-only and fuzzy_match_chapter's tokenizer strips non-ASCII
    letters, so a raw Hindi title would never match anything and always fall back to numberless.

    Falls back to a numberless citation (never a guessed number) if lookup_name doesn't
    confidently match anything in the list -- e.g. Appendix content, or a genuinely new/renamed
    chapter that hasn't been added to NEET_SYLLABUS yet."""
    if not chapter_name:
        return ""
    lookup_name = lookup_name or chapter_name
    try:
        class_key = int(class_num)
    except (TypeError, ValueError):
        return f"NCERT {subject} — {chapter_name}" if subject else chapter_name
    chapters = NEET_SYLLABUS.get(subject, {}).get(class_key, [])
    if not chapters:
        return f"NCERT Class {class_key}, {subject} — {chapter_name}"
    norm = lookup_name.strip().lower()
    match = next((c for c in chapters if c.strip().lower() == norm), None) or fuzzy_match_chapter(lookup_name, chapters)
    if match:
        return f"NCERT Class {class_key}, Chapter {chapters.index(match) + 1} — {chapter_name}"
    return f"NCERT Class {class_key}, {subject} — {chapter_name}"

# Re-measured 2026-08-24 after backfilling embeddings for the ~80% of pyq rows that had none
# (984/5,084 -> 5,084/5,084) -- worth re-checking since 4x more embedded content changes what a
# given similarity score actually means. Real scores against the now-complete set: 9 genuinely
# on-topic test queries across all 3 subjects (photosynthesis, Newton's second law, SN1/SN2,
# periodic trends, enzymes, DNA structure, osmosis, escape velocity, Mendel's law) scored
# 0.416-0.657 on their real top match; 4 deliberately irrelevant queries (capital of France,
# baking a cake, movie recommendations, weather) topped out at 0.282. That's a clean ~0.13 gap
# with 0.4 sitting in the middle -- raising it would start rejecting genuine matches (escape
# velocity's real top hit is 0.416, Newton's second law's 2nd-best is 0.411), lowering it doesn't
# fix anything real (the one still-failing case, "Wheatstone bridge", has zero embedded rows on
# the topic at all -- a content gap no threshold change can paper over). Conclusion: 0.4 remains
# correct, unchanged from its original value.
PYQ_MATCH_THRESHOLD = 0.4

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
            "match_threshold": PYQ_MATCH_THRESHOLD,
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

async def _empty_bool():
    return False

async def _diagram_exists_for(text: str) -> bool:
    """Cheap existence check reusing the same embedding + match_diagrams RPC /diagram-match
    already uses (get_embedding is sha256-cached, so this is effectively free when the same
    doubt text's embedding was already computed elsewhere in this same request). Only tells the
    model WHETHER a diagram is available -- never which one -- so it can decide whether
    offering a format-clarification ("explain it, or show the diagram?") is even honest to
    offer; the real lookup still happens through the existing /diagram-match flow when the
    student actually asks to see it. Best-effort: any failure here just means the format-
    clarification option won't be offered, same fail-safe direction as /diagram-match itself."""
    try:
        embedding = await get_embedding(text)
        headers = {"apikey": SUPABASE_KEY, "Authorization": f"Bearer {SUPABASE_KEY}", "Content-Type": "application/json"}
        response = await async_client.post(
            f"{SUPABASE_URL}/rest/v1/rpc/match_diagrams",
            headers=headers,
            json={"query_embedding": embedding, "match_threshold": DIAGRAM_MATCH_THRESHOLD, "match_count": 1, "filter_chapter": None}
        )
        if response.status_code == 200:
            return bool(response.json())
    except Exception:
        pass
    return False

# Rule 11's topic-ambiguity whitelist (resistance/cycle/potential/diffusion/current/valence)
# reliably stopped false-triggering for every word it covers once prompt-only enforcement moved
# from open-ended judgment to a closed list -- except "reflex", which kept false-triggering even
# though it was never listed and the instruction explicitly says not to reason about unlisted
# words at all (44% false-trigger rate across repeated live testing, three separate prompt
# rewrites in a row). That's a compliance ceiling, not a wording problem -- this denylist is the
# deterministic backstop for words with that same confirmed, repeated-and-still-failing evidence.
# Keep this list small and evidence-based, not a general-purpose ambiguity filter: only add a
# word here after prompt-only fixes have genuinely failed on it, the way they did for "reflex".
FALSE_POSITIVE_CLARIFY_WORDS = {
    "reflex", "reflexes",
    "रिफ्लेक्स", "प्रतिवर्ती क्रिया", "प्रतिवर्ती", "प्रतिवर्त",
}

def _is_denylisted_clarify_doubt(text: str) -> bool:
    """Exact match only, not substring -- Rule 11 itself only ever fires on a bare word/short
    phrase doubt (its own full-sentence guard already handles longer doubts that merely mention
    one of these words), so an exact match on the whole stripped doubt is enough and avoids ever
    matching a denylisted word that's incidentally part of a real, different question."""
    normalized = text.strip()
    return normalized.lower() in FALSE_POSITIVE_CLARIFY_WORDS or normalized in FALSE_POSITIVE_CLARIFY_WORDS

# chat.html's streamAIResponse sends this exact string as `text` when an image doubt has no
# real typed question (`text || 'Describe this image...'`) -- must stay in sync with that
# literal. A placeholder like this has no real topic for NCERT search to match against, so
# running it just adds latency (embedding + vector search) and risks stuffing irrelevant NCERT
# content into the prompt alongside the image. Only skipped when this exact placeholder is the
# whole text -- any real typed question alongside an image still gets a real NCERT search.
IMAGE_ONLY_PLACEHOLDER_TEXT = "Describe this image and answer any NEET related content in it."

# Real per-provider pricing used only to compute provider_usage_log's `cost` column -- update
# these constants directly if a provider changes pricing; nothing else needs to change.
# DeepSeek's rates are its published peak/off-peak structure effective 2026-08-16 (fetched from
# api-docs.deepseek.com/quick_start/pricing on 2026-08-14) -- the peak-hour Qwen fallback exists
# specifically because of that change, so these are the rates that actually matter for cost
# analysis here even for the handful of days before the change formally takes effect (using the
# old flat rate instead would make the peak_window column meaningless in the cost figures).
# $ per 1M tokens throughout.
DEEPSEEK_RATES = {
    "peak": {"cache_hit": 0.014, "cache_miss": 0.44, "output": 1.32},
    "off_peak": {"cache_hit": 0.007, "cache_miss": 0.22, "output": 0.66},
}
# Qwen-Flash's published rate for the 0-256K input tier (fetched from
# alibabacloud.com/help/en/model-studio/model-pricing on 2026-08-14). Alibaba's docs mention a
# separate, lower cache-hit rate exists but don't publish a number for it, so this deliberately
# applies the flat input rate to all input tokens -- a conservative (slight over-, never under-)
# estimate rather than guessing an unconfirmed discount.
QWEN_FLASH_RATE = {"input": 0.05, "output": 0.40}
# Gemini 3.5 Flash-Lite's standard (non-batch) tier -- the tier that actually applies to live
# streaming /chat requests (fetched from ai.google.dev/gemini-api/docs/pricing on 2026-08-14).
GEMINI_FLASH_LITE_RATE = {"input": 0.30, "output": 2.50}

def _deepseek_cost(cache_miss_tokens: int, cache_hit_tokens: int, output_tokens: int, is_peak: bool) -> float:
    r = DEEPSEEK_RATES["peak" if is_peak else "off_peak"]
    return (cache_miss_tokens * r["cache_miss"] + cache_hit_tokens * r["cache_hit"] + output_tokens * r["output"]) / 1_000_000

def _qwen_cost(input_tokens: int, output_tokens: int) -> float:
    return (input_tokens * QWEN_FLASH_RATE["input"] + output_tokens * QWEN_FLASH_RATE["output"]) / 1_000_000

def _gemini_cost(input_tokens: int, output_tokens: int) -> float:
    return (input_tokens * GEMINI_FLASH_LITE_RATE["input"] + output_tokens * GEMINI_FLASH_LITE_RATE["output"]) / 1_000_000

async def log_provider_usage(provider: str, peak_window: bool, input_tokens: int, output_tokens: int, cost: float, endpoint: str, user_id: str = ""):
    """Additive to log_token_usage() (which drives per-user daily budget enforcement, keyed by
    total tokens only) -- this is purely for cost/provider analytics: one row per completed
    request in provider_usage_log (see create_provider_usage_log_table.sql; RLS locked to the
    service-role key only, no anon/authenticated policy at all, same as pyq_solution_cache --
    this is internal cost telemetry, not user-facing content). Keeps the existing print line
    too, since it's cheap and useful for a quick Railway log check without a DB round trip."""
    print(f"PROVIDER USAGE: provider={provider} peak_window={str(peak_window).lower()} input={input_tokens} output={output_tokens} cost=${cost:.6f} endpoint={endpoint}", flush=True)
    try:
        await async_client.post(
            f"{SUPABASE_URL}/rest/v1/provider_usage_log", headers={**ADMIN_HEADERS, "Content-Type": "application/json"},
            json={
                "user_id": user_id or None,
                "provider": provider,
                "peak_window": peak_window,
                "input_tokens": input_tokens,
                "output_tokens": output_tokens,
                "cost": cost,
                "endpoint": endpoint
            }
        )
    except Exception:
        pass

async def _stream_qwen(system: str, messages: list, user_id: str, ip: str, endpoint: str, billing_context: dict = None):
    """Yields text chunks from Qwen-Flash and logs real usage. Any failure -- at connection time
    or partway through the stream -- propagates to the caller as-is; _stream_with_peak_fallback
    is the one that decides whether that's safe to retry on DeepSeek (only if nothing was
    yielded yet).

    billing_context (optional, shared mutable dict): lets the caller waive the student's daily
    budget charge for this specific response -- e.g. stream_response() sets
    billing_context["bill"] = False as soon as it sees this turn is a clarifying-question
    response, not a real answer. Provider cost logging (log_provider_usage) is deliberately
    unaffected -- real tokens were genuinely spent either way, this only controls the
    user-facing quota deduction. None (the default, used by every caller that doesn't pass one,
    e.g. /solve) means "always bill", matching the pre-existing behavior."""
    qwen_stream = qwen_client.chat.completions.create(
        model="qwen-flash",
        max_tokens=1024,
        stream=True,
        stream_options={"include_usage": True},
        messages=[{"role": "system", "content": system}] + messages
    )
    input_tokens = output_tokens = 0
    try:
        for chunk in qwen_stream:
            if chunk.choices and chunk.choices[0].delta.content:
                yield chunk.choices[0].delta.content
            if chunk.usage:
                input_tokens = chunk.usage.prompt_tokens
                output_tokens = chunk.usage.completion_tokens
    finally:
        # Best-effort even on a mid-stream failure -- whatever tokens were actually used still
        # get logged/counted against the student's daily budget; log_token_usage() itself is a
        # no-op for tokens<=0, so a failure before any usage chunk arrived costs nothing extra.
        if input_tokens or output_tokens:
            cost = _qwen_cost(input_tokens, output_tokens)
            try:
                await log_provider_usage("qwen-flash", True, input_tokens, output_tokens, cost, endpoint, user_id)
            except Exception:
                pass
            if billing_context is None or billing_context.get("bill", True):
                try:
                    await log_token_usage(user_id, input_tokens + output_tokens, ip)
                except Exception:
                    pass

async def _stream_deepseek(system: str, messages: list, user_id: str, ip: str, is_peak: bool, endpoint: str, billing_context: dict = None):
    """The default off-peak provider, and also the peak-window fallback when Qwen is
    unavailable -- is_peak is only for the usage-log tag, so peak-window fallback traffic still
    shows up as such in the logs even though DeepSeek ended up serving it. See _stream_qwen's
    docstring for what billing_context does."""
    with deepseek_client.messages.stream(
        model="deepseek-v4-flash",
        max_tokens=1024,
        thinking={"type": "disabled"},
        system=system,
        messages=messages
    ) as stream:
        try:
            for text_chunk in stream.text_stream:
                yield text_chunk
        finally:
            try:
                usage = stream.get_final_message().usage
                # cache_creation_input_tokens (writing a fresh cache entry) is billed at the same
                # rate as a cache miss, not a discount -- only cache_read_input_tokens (an actual
                # hit against an already-cached system prompt) gets the cheaper rate.
                cache_miss_tokens = usage.input_tokens + (usage.cache_creation_input_tokens or 0)
                cache_hit_tokens = usage.cache_read_input_tokens or 0
                cost = _deepseek_cost(cache_miss_tokens, cache_hit_tokens, usage.output_tokens, is_peak)
                try:
                    await log_provider_usage("deepseek-v4-flash", is_peak, cache_miss_tokens + cache_hit_tokens, usage.output_tokens, cost, endpoint, user_id)
                except Exception:
                    pass
                if billing_context is None or billing_context.get("bill", True):
                    await log_token_usage(user_id, usage.input_tokens + usage.output_tokens, ip)
            except Exception:
                pass

async def _stream_with_peak_fallback(system: str, messages: list, user_id: str, ip: str, endpoint: str, billing_context: dict = None):
    """Routes text-doubt generation (/chat, /solve -- never image/PDF doubts, those stay on
    Gemini regardless of time of day) to Qwen-Flash during DeepSeek's published peak-pricing
    windows (_is_deepseek_peak_hour), DeepSeek everywhere else. Confirmed via a live 45-question
    test that the two are equivalent on accuracy/reliability, so this is purely a cost-avoidance
    swap, not a quality tradeoff.

    Failover: if Qwen fails before yielding anything, the same request is retried on DeepSeek
    transparently -- the student never sees the difference, and this accepts DeepSeek's peak
    cost rather than failing the request outright (availability over cost optimization). If Qwen
    fails after already streaming part of an answer, there's no clean way to hand off to a
    different model mid-response without duplicating or garbling what the student already saw,
    so the failure just propagates like any other mid-stream error in this file -- the caller's
    own try/except turns it into a yielded "Error: ..." message and correctly skips caching a
    partial answer, exactly as it already does for a mid-stream DeepSeek failure."""
    if _is_deepseek_peak_hour():
        sent_any = False
        try:
            async for chunk in _stream_qwen(system, messages, user_id, ip, endpoint, billing_context):
                sent_any = True
                yield chunk
            return
        except Exception as e:
            if sent_any:
                raise
            print(f"QWEN UNAVAILABLE BEFORE FIRST TOKEN, FALLING BACK TO DEEPSEEK: {e}", flush=True)
        async for chunk in _stream_deepseek(system, messages, user_id, ip, True, endpoint, billing_context):
            yield chunk
        return
    async for chunk in _stream_deepseek(system, messages, user_id, ip, False, endpoint, billing_context):
        yield chunk

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
    # Student context, NCERT search, and the diagram-existence check are independent of each
    # other -- run them concurrently instead of stacking their latency sequentially. Skip the
    # search entirely for an image-only doubt (see IMAGE_ONLY_PLACEHOLDER_TEXT above) -- there's
    # no real question text to search on. The diagram-existence check is only meaningful for
    # text doubts (format-ambiguity -- "explain it or show the diagram?" -- doesn't apply once
    # an image is already attached), so it's skipped for images/pdf the same way.
    student_context_coro = get_student_context(user_id) if (personalize and user_id) else _empty_str()
    skip_ncert_search = bool(images) and text.strip() == IMAGE_ONLY_PLACEHOLDER_TEXT
    diagram_exists_coro = _diagram_exists_for(text) if not images and not pdf else _empty_bool()
    if skip_ncert_search:
        results = []
        student_context = await student_context_coro
        diagram_exists = False
    else:
        results, student_context, diagram_exists = await asyncio.gather(search_ncert(text), student_context_coro, diagram_exists_coro)

    if results:
        context = "\n\n".join([
            f"[{r.get('subject', '')} - Class {r.get('class', '')} - {r.get('chapter_name', '')}]\n{r.get('content', '')}"
            for r in results
        ])
        # Deduplicated, mechanically-correct citation for every distinct chapter actually
        # retrieved (see _ncert_chapter_citation) -- SYSTEM_PROMPT rule 1 instructs the model to
        # copy one of these verbatim into the "Chapter:" line instead of recalling/guessing it,
        # which a live 45-question test found produced a ~30% wrong-chapter-number rate.
        real_citations = []
        for r in results:
            citation = _ncert_chapter_citation(r.get('subject', ''), r.get('class'), r.get('chapter_name', ''), lookup_name=r.get('chapter_name_en'))
            if citation and citation not in real_citations:
                real_citations.append(citation)
        retrieved_from = "\n".join(f"- {c}" for c in real_citations)
        user_message = f"NCERT Content:\n{context}\n\nRetrieved from (copy EXACTLY ONE of these verbatim into your Chapter: line -- never a different name or number):\n{retrieved_from}\n\nStudent Question: {text}"
    else:
        user_message = f"Student Question: {text}"
    if diagram_exists:
        user_message += "\n\n[A relevant NCERT diagram exists for this topic -- this is the ONLY signal that tells you whether offering a format-clarification (rule 11) is honest to offer. If this note is absent, a diagram is not available, so never offer to show one.]"

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
        pass  # Gemini branch below builds its own Part.from_bytes content -- doesn't use `messages`
    else:
        messages.append({"role": "user", "content": user_message})
    try:
        import sys
        name_context = f"\n\nThe student name is {student_name}. Use their name naturally and occasionally in responses to make it personal." if student_name else ""
        style_context = "\n\nIMPORTANT: The student has selected CONCISE mode. Give a very short answer — maximum 3 sentences only. No bullet points, no key points section, no memory tricks. Just the core answer." if answer_style == "concise" else ""
        lang_context = "\n\nIMPORTANT: Respond ONLY in Hindi (Devanagari script). Every word — headings, key points, explanations, memory tricks — must be in Hindi. Do not mix in English words or Hinglish, even for common scientific terms (e.g. write \"गुणसूत्र\" not \"chromosome\"). The ONLY exceptions are: LaTeX/KaTeX math notation, chemical formulas/symbols (e.g. $H_2O$), units (e.g. m/s, kg), and proper nouns like NEET or NCERT — keep those exactly as-is, do not translate or romanize them." if language == "hi" else ""
        # Graph/curve-matching questions are Gemini's one consistently weak category (~73%
        # across 4 rounds of testing, vs 95-100% on everything else) -- this can't be gated on
        # chapter/question-type ahead of time since the chapter is an OUTPUT of the answer (the
        # 📚 Chapter: line), not something known before generation, so the instruction is just
        # always-on for the image branch. Harmless for non-graph images, a cheap mitigation
        # (not a fix) for the ones that are.
        graph_context = "\n\nIMPORTANT: If this question involves matching a labeled curve, point, or line in a graph/diagram to a specific value, property, or identity (e.g. \"which curve represents gas X\", \"identify point Y on the graph\"), carefully re-examine which curve/point corresponds to which label before answering -- this type of graph-reading question has a measurably higher error rate, so double-check your reading of the labels against the image before finalizing your answer." if images else ""
        # Tested live against 15 real image doubts (10 handwritten-style, 5 diagram) before
        # shipping: 15/15 accuracy held vs. the untightened prompt, ~20% fewer output tokens,
        # ~7% lower cost per image doubt. Scoped to images only -- text doubts (DeepSeek) weren't
        # part of that test, so this doesn't touch them.
        conciseness_context = """

IMPORTANT -- BE CONCISE:
- Do not restate or rephrase the question before answering.
- For conceptual/definitional/factual questions (no calculation required), give the Answer in
  2-4 tight bullet points maximum -- do not add a separate lengthy walkthrough on top of the Key
  Points section already required by the format above.
- For numerical/derivation questions, keep Given/Formula/Solution as short as correctness allows
  -- show the necessary steps only, never restate the same substitution twice.
- Do not repeat the same fact in both the Answer section and the Key Points section -- each
  should add distinct information, not duplicate it.
- Keep the Easy Way to Remember line to one short sentence.
- Never use filler transition phrases like "Let's break this down" or "To understand this, we
  need to first" -- start directly with the substantive content.""" if images else ""
        full_system = SYSTEM_PROMPT + name_context + style_context + lang_context + student_context + graph_context + conciseness_context
        if images:
            # See the gemini_client comment above for why images specifically moved off Claude.
            selected_model = "gemini-3.5-flash-lite"
            print(f"MODEL SELECTED: {selected_model}", flush=True)
            sys.stdout.flush()
            image_parts = [
                genai_types.Part.from_bytes(
                    data=base64.b64decode(img.data),
                    mime_type=img.media_type or "image/jpeg"
                )
                for img in images
            ]
            gemini_stream = gemini_client.models.generate_content_stream(
                model=selected_model,
                contents=image_parts + [user_message],
                config=genai_types.GenerateContentConfig(
                    system_instruction=full_system,
                    max_output_tokens=1024
                )
            )
            full_answer = ""
            last_usage = None
            try:
                for chunk in gemini_stream:
                    if chunk.text:
                        full_answer += chunk.text
                        yield chunk.text
                    if chunk.usage_metadata:
                        last_usage = chunk.usage_metadata
            finally:
                # Same reasoning as the Anthropic branch below: log on the way out (including on
                # an early client disconnect) rather than only on a clean finish. Gemini reports
                # usage_metadata cumulatively on every streamed chunk, so the last chunk seen
                # (even a partial stream) already holds the running totals -- no separate
                # "final message" fetch needed the way Anthropic's SDK requires.
                if last_usage:
                    cost = _gemini_cost(last_usage.prompt_token_count, last_usage.candidates_token_count)
                    try:
                        await log_provider_usage("gemini-3.5-flash-lite", False, last_usage.prompt_token_count, last_usage.candidates_token_count, cost, "/chat", user_id)
                    except Exception:
                        pass
                    try:
                        await log_token_usage(user_id, last_usage.prompt_token_count + last_usage.candidates_token_count, ip)
                    except Exception:
                        pass
            return
        elif pdf:
            # Moved off Claude to Gemini 3.5 Flash-Lite -- same client/pattern as the images
            # branch above, confirmed working via a live standalone test (91% accuracy across
            # 3 real PDFs, ~10x cheaper input tokens than Claude Sonnet's pricing).
            selected_model = "gemini-3.5-flash-lite"
            print(f"MODEL SELECTED: {selected_model}", flush=True)
            sys.stdout.flush()
            pdf_part = genai_types.Part.from_bytes(data=base64.b64decode(pdf), mime_type="application/pdf")
            gemini_stream = gemini_client.models.generate_content_stream(
                model=selected_model,
                contents=[pdf_part, user_message],
                config=genai_types.GenerateContentConfig(
                    system_instruction=full_system,
                    max_output_tokens=1024
                )
            )
            full_answer = ""
            last_usage = None
            try:
                for chunk in gemini_stream:
                    if chunk.text:
                        full_answer += chunk.text
                        yield chunk.text
                    if chunk.usage_metadata:
                        last_usage = chunk.usage_metadata
            finally:
                # Same reasoning as the images branch above.
                if last_usage:
                    cost = _gemini_cost(last_usage.prompt_token_count, last_usage.candidates_token_count)
                    try:
                        await log_provider_usage("gemini-3.5-flash-lite", False, last_usage.prompt_token_count, last_usage.candidates_token_count, cost, "/chat", user_id)
                    except Exception:
                        pass
                    try:
                        await log_token_usage(user_id, last_usage.prompt_token_count + last_usage.candidates_token_count, ip)
                    except Exception:
                        pass
            return
        else:
            is_peak = _is_deepseek_peak_hour()
            print(f"MODEL SELECTED: {'qwen-flash (DeepSeek peak-hour fallback)' if is_peak else 'deepseek-v4-flash'}", flush=True)
            sys.stdout.flush()
            full_answer = ""
            # billing_context lets this loop waive the student's daily-budget charge the moment
            # it recognizes the response is a clarifying question (rule 11), not a real answer --
            # set BEFORE the underlying stream finishes, which is what matters: the nested
            # generator's own usage-logging finally block only runs once ITS stream is exhausted,
            # by which point this loop has already processed every chunk including the one that
            # revealed "AMBIGUOUS: yes" (that marker is always within the first line, yielded
            # many chunks before the short clarification response ends).
            billing_context = {"bill": True}
            # Cheap, one-time check against a tiny set -- decides which of the two loops below
            # runs, so a non-denylisted doubt (the overwhelming majority) takes the exact same
            # unbuffered fast path as before this guard existed, with zero added latency.
            denylisted_doubt = _is_denylisted_clarify_doubt(text)
            if not denylisted_doubt:
                # Reliably logs on normal completion. This is a native async generator, so an
                # early client disconnect propagates via GeneratorExit at the next yield point --
                # usage logging for whichever provider actually served the request happens
                # inside _stream_qwen/_stream_deepseek's own finally blocks either way, same
                # accepted partial-under-count-on-disconnect tradeoff as before this routing was
                # added.
                async for text_chunk in _stream_with_peak_fallback(full_system, messages, user_id, ip, "/chat", billing_context):
                    full_answer += text_chunk
                    if billing_context["bill"] and full_answer.strip().startswith("AMBIGUOUS: yes"):
                        billing_context["bill"] = False
                    yield text_chunk
            else:
                # Rule 11's prompt-only enforcement has confirmed, repeated evidence of not
                # holding for this doubt (see FALSE_POSITIVE_CLARIFY_WORDS above) -- buffer just
                # the first line (same one-line lookahead chat.html's own client-side VISUAL_
                # INTENT/AMBIGUOUS buffering already uses) before letting anything reach the
                # student, so a fabricated clarify response can be caught and discarded before a
                # single byte of it streams out, not just flagged after the fact.
                first_line_buffer = ""
                first_line_resolved = False
                override_needed = False
                async for text_chunk in _stream_with_peak_fallback(full_system, messages, user_id, ip, "/chat", billing_context):
                    full_answer += text_chunk
                    if not first_line_resolved:
                        first_line_buffer += text_chunk
                        nl_idx = first_line_buffer.find("\n")
                        if nl_idx == -1 and len(first_line_buffer) < 64:
                            continue
                        first_line_resolved = True
                        first_line = (first_line_buffer[:nl_idx] if nl_idx != -1 else first_line_buffer).strip()
                        if first_line == "AMBIGUOUS: yes":
                            override_needed = True
                            break
                        if billing_context["bill"] and full_answer.strip().startswith("AMBIGUOUS: yes"):
                            billing_context["bill"] = False
                        yield first_line_buffer
                        continue
                    if billing_context["bill"] and full_answer.strip().startswith("AMBIGUOUS: yes"):
                        billing_context["bill"] = False
                    yield text_chunk
                if override_needed:
                    # The discarded attempt's own tiny (one-line) token cost still gets logged by
                    # _stream_qwen/_stream_deepseek's own finally block when the `break` above
                    # closes it via GeneratorExit -- same accepted partial-cost tradeoff noted
                    # above, and negligible next to a full response's cost either way. This
                    # second call is a completely fresh request/response cycle, billed normally.
                    override_system = full_system + (
                        "\n\nOVERRIDE (server-enforced, not optional): this exact doubt has been "
                        "confirmed through repeated real testing to NOT be genuinely ambiguous, "
                        "despite rule 11 above. Answer it directly and normally in the standard "
                        "format starting with VISUAL_INTENT -- do not output AMBIGUOUS/"
                        "CLARIFY_TYPE under any circumstances for this doubt."
                    )
                    full_answer = ""
                    billing_context = {"bill": True}
                    async for text_chunk in _stream_with_peak_fallback(override_system, messages, user_id, ip, "/chat", billing_context):
                        full_answer += text_chunk
                        yield text_chunk
            if not images and not pdf and use_shared_cache:
                await async_client.post(
                    f"{SUPABASE_URL}/rest/v1/answer_cache",
                    headers={**headers, "Content-Type": "application/json"},
                    json={"question_hash": answer_hash, "answer": full_answer}
                )
    except Exception as e:
        print(f"STREAMING ERROR: {e}")
        yield f"Error: {str(e)}"

SOLVE_CACHE_HEADERS = {
    "apikey": SUPABASE_SERVICE_KEY,
    "Authorization": f"Bearer {SUPABASE_SERVICE_KEY}"
}

async def get_cached_pyq_solution(pyq_id: str, language: str):
    """A PYQ's correct solution never changes once generated, so it's cached forever, keyed on
    the question's own DB row id (not a text hash -- it's already a fixed row) plus language,
    since an English and Hindi solution for the same question are different content. Looked up
    here in the route handler (not inside the generator) so /solve can tell the client up front,
    via the X-Cache header, whether this request will be instant -- letting the frontend hold its
    thinking indicator for a fixed ~1s on a hit instead of an instant, jarring pop-in."""
    if not pyq_id:
        return None
    resp = await async_client.get(
        f"{SUPABASE_URL}/rest/v1/pyq_solution_cache",
        headers=SOLVE_CACHE_HEADERS,
        params={"pyq_id": f"eq.{pyq_id}", "language": f"eq.{language}", "select": "solution"}
    )
    rows = resp.json()
    return rows[0]["solution"] if rows else None

async def stream_solve_response(pyq_id: str, cached_solution, question: str, option_a: str, option_b: str, option_c: str, option_d: str, correct_answer: str, language: str = "en", user_id: str = "", ip: str = ""):
    if cached_solution is not None:
        yield cached_solution
        return
    lang_instruction = "\n5. Respond ONLY in Hindi (Devanagari script) — every word in Hindi, no English words or Hinglish mixing. The ONLY exceptions are LaTeX/KaTeX math notation, chemical formulas/symbols, and units, which stay exactly as-is." if language == "hi" else ""
    solve_system = f"""You are a NEET exam expert. Solve the given NEET question step by step.

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
   - Always write $v^2$ not v²{lang_instruction}"""
    solve_messages = [
        {"role": "user", "content": f"Solve this NEET question:\n\nQuestion: {question}\n\nA) {option_a}\nB) {option_b}\nC) {option_c}\nD) {option_d}\n\nCorrect Answer: {correct_answer}"}
    ]
    try:
        full_solution = ""
        # Same routing (Qwen during DeepSeek's peak windows, DeepSeek otherwise, with failover)
        # as stream_response()'s text branch above -- see _stream_with_peak_fallback for the
        # shared logic and its accepted early-disconnect / mid-stream-failure tradeoffs.
        async for text_chunk in _stream_with_peak_fallback(solve_system, solve_messages, user_id, ip, "/solve"):
            full_solution += text_chunk
            yield text_chunk
        if pyq_id and full_solution:
            await async_client.post(
                f"{SUPABASE_URL}/rest/v1/pyq_solution_cache",
                headers={**SOLVE_CACHE_HEADERS, "Content-Type": "application/json"},
                json={"pyq_id": pyq_id, "language": language, "solution": full_solution}
            )
    except Exception as e:
        yield f"Error: {str(e)}"

@app.post("/solve")
async def solve_question(req: SolveRequest, request: Request, _: None = Depends(rate_limiter(15, 60))):
    ip = _client_ip(request)
    await enforce_daily_budget(req.user_id, ip)
    cached_solution = await get_cached_pyq_solution(req.pyq_id, req.language)
    return StreamingResponse(
        stream_solve_response(req.pyq_id, cached_solution, req.question, req.option_a, req.option_b, req.option_c, req.option_d, req.correct_answer, req.language, req.user_id, ip),
        media_type="text/plain",
        headers={"X-Cache": "HIT" if cached_solution is not None else "MISS"}
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
    await validate_pdf_limits(message.pdf, message.user_id)
    return StreamingResponse(
       stream_response(message.text, message.history, message.images, message.pdf, message.answer_style, message.student_name, message.language, message.user_id, message.personalize, message.skip_cache, ip),
        media_type="text/plain"
    )

@app.post("/title")
async def generate_title(message: Message, request: Request, _: None = Depends(rate_limiter(15, 60))):
    ip = _client_ip(request)
    await enforce_daily_budget(message.user_id, ip)
    client = deepseek_client
    title_lang = "entirely in Hindi (Devanagari script) — every word in Hindi, no English words mixed in" if message.language == "hi" else "in English"
    response = client.messages.create(
        model="deepseek-v4-flash",
        max_tokens=15,
        thinking={"type": "disabled"},
        system=(
            f"Generate a short 3-5 word title {title_lang} summarizing the TOPIC of this message, "
            "the way a chat app names a conversation in its sidebar. The message is the first thing "
            "a student sent -- it may be a NEET question, but it may just as easily be a greeting, "
            "small talk, or something unrelated to NEET.\n\n"
            "Always output a short topic label. NEVER reply to, answer, or continue the message "
            "itself -- you are naming the conversation, not participating in it.\n"
            "Examples: \"hi\" -> Greeting. \"what's your name\" -> Asking My Name. \"explain "
            "photosynthesis\" -> Photosynthesis Explanation.\n\n"
            "Return ONLY the title. No punctuation. No extra words. No markdown."
        ),
        messages=[{"role": "user", "content": message.text}]
    )
    await log_token_usage(message.user_id, response.usage.input_tokens + response.usage.output_tokens, ip)
    return {"title": response.content[0].text}

# Backs chat.html's "/summarize" command -- same lightweight, no-NCERT-search, low-max_tokens
# shape as /title above (a small transform of already-known text, not a fresh doubt-answering
# call), not the full /chat pipeline. Deliberately uncached: /title has run this same pattern
# uncached since it shipped, and a per-answer summary cache would be new infra for a cheap,
# infrequent call -- not worth adding for this.
@app.post("/summarize-answer")
async def summarize_answer(req: SummarizeAnswerRequest, request: Request, _: None = Depends(rate_limiter(15, 60))):
    ip = _client_ip(request)
    await enforce_daily_budget(req.user_id, ip)
    client = deepseek_client
    lang_instruction = "Respond entirely in Hindi (Devanagari script) -- every word in Hindi, no English words mixed in, except LaTeX/KaTeX math notation and units which stay as-is." if req.language == "hi" else "Respond in English."
    response = client.messages.create(
        model="deepseek-v4-flash",
        max_tokens=220,
        thinking={"type": "disabled"},
        system=(
            "Condense the given NEET doubt answer into 3-5 short bullet points suitable for "
            "last-minute revision. Genuinely condense and re-synthesize -- do not just copy the "
            "answer's existing 'Key Points' section verbatim. Each bullet is one short line: no "
            "sub-bullets, no headings, no restating the question, no filler. Preserve any "
            "LaTeX/KaTeX math notation ($...$ or $$...$$) exactly as it appears in the source if "
            f"it's essential to a point. {lang_instruction} Return ONLY the bullet points as a "
            "markdown list ('- ' prefix), nothing before or after."
        ),
        messages=[{"role": "user", "content": req.text}]
    )
    await log_token_usage(req.user_id, response.usage.input_tokens + response.usage.output_tokens, ip)
    return {"summary": response.content[0].text}

# Guests have no Supabase session, so they can't use RLS to read their own usage row the way a
# logged-in user does (auth.uid() is null for anonymous requests) -- this endpoint is the only
# way a guest's remaining count can be shown, computed from the server's own view of their IP.
@app.get("/guest-usage-status")
async def guest_usage_status(request: Request):
    ip = _client_ip(request)
    today, yesterday, rows_by_date = await _fetch_usage_rows("guest_usage_log", "ip", ip)
    used = rows_by_date[today]["tokens_used"] if today in rows_by_date else 0
    retry_after_seconds = _active_cooldown_seconds(today, yesterday, rows_by_date)
    return {"tokens_used": used, "budget": DAILY_TOKEN_BUDGET_GUEST, "retry_after_seconds": retry_after_seconds}

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

USAGE_PROVIDER_BREAKDOWN_DAYS = 30  # no real billing-cycle concept exists yet (flat subscription,
                                     # Razorpay not live) -- a rolling 30-day window is used as the
                                     # closest honest stand-in for "current billing period" and is
                                     # labeled as "last 30 days" client-side rather than implying a
                                     # real cycle boundary that doesn't exist.

# Powers the Settings > Usage tab in chat.html. Every doubt COUNT here (today/weekly/provider
# breakdown) comes from provider_usage_log -- one real row per completed /chat or /solve request,
# so these are exact counts, not estimates. The one thing that ISN'T doubt-denominated is the
# actual daily budget itself (DAILY_TOKEN_BUDGET_FREE is a TOKEN ceiling, and tokens/doubt varies
# by question) -- so the progress bar/percent below is computed from usage_log's tokens_used
# against that real token budget (the same numbers enforce_daily_budget actually checks), while
# the "N doubts today" label next to it is a separate, exact count. Deliberately doesn't invent a
# fake "doubts allowed" ceiling by dividing the token budget by an average -- that would look
# precise while actually being a guess.
@app.get("/usage/summary")
async def usage_summary(user_id: str):
    if not user_id:
        raise HTTPException(status_code=400, detail="user_id is required")
    try:
        plan = await get_user_plan(user_id)
        unlimited = plan != "free"  # matches enforce_daily_budget's own "paid = unlimited for now"

        today, yesterday, rows_by_date = await _fetch_usage_rows("usage_log", "user_id", user_id)
        today_row = rows_by_date.get(today)
        tokens_used = today_row["tokens_used"] if today_row else 0

        cooldown_remaining = _active_cooldown_seconds(today, yesterday, rows_by_date)
        if cooldown_remaining is not None:
            reset_in_seconds = cooldown_remaining
        else:
            # No active block: "reset" just means the next IST midnight, when a fresh usage_date
            # row starts and today's count naturally reads back to 0.
            now_ist = datetime.now(timezone.utc).astimezone(IST)
            next_ist_midnight = datetime.combine(now_ist.date() + timedelta(days=1), datetime.min.time(), tzinfo=IST)
            reset_in_seconds = int((next_ist_midnight - now_ist).total_seconds())

        percent_used = None if unlimited else round(min(100, tokens_used / DAILY_TOKEN_BUDGET_FREE * 100), 1)
        budget = None if unlimited else DAILY_TOKEN_BUDGET_FREE

        window_start_iso = (datetime.now(timezone.utc) - timedelta(days=USAGE_PROVIDER_BREAKDOWN_DAYS)).isoformat()
        resp = await async_client.get(
            f"{SUPABASE_URL}/rest/v1/provider_usage_log", headers=ADMIN_HEADERS,
            params={"user_id": f"eq.{user_id}", "created_at": f"gte.{window_start_iso}",
                    "select": "provider,created_at", "order": "created_at.asc"}
        )
        rows = resp.json() if resp.status_code == 200 else []

        today_start_utc = _ist_today_start_utc_iso()
        doubts_today = sum(1 for r in rows if r["created_at"] >= today_start_utc)

        # Last 7 IST calendar days including today -- real doubt count, purely informational
        # (product decision: no weekly cap enforced yet). The no-longer-used per-day breakdown
        # (that used to back a bar-chart UI) was dropped; only the 7-day total is needed now.
        week_day_labels = [(date.fromisoformat(today) - timedelta(days=i)).isoformat() for i in range(7)]
        weekly_doubts = sum(1 for r in rows if
            datetime.fromisoformat(r["created_at"].replace("Z", "+00:00")).astimezone(IST).date().isoformat() in week_day_labels)

        # Weekly token total, same table/column the daily figure above uses -- lets the Weekly
        # section show a bar on the same real footing as Today's (tokens against the real budget,
        # here projected across 7 days) rather than a different, doubt-count-only metric.
        week_resp = await async_client.get(
            f"{SUPABASE_URL}/rest/v1/usage_log", headers=ADMIN_HEADERS,
            params={"user_id": f"eq.{user_id}", "usage_date": f"in.({','.join(week_day_labels)})",
                    "select": "tokens_used"}
        )
        weekly_tokens_used = sum(r["tokens_used"] for r in week_resp.json()) if week_resp.status_code == 200 else 0
        weekly_budget = None if unlimited else DAILY_TOKEN_BUDGET_FREE * 7
        weekly_percent_used = None if unlimited else round(min(100, weekly_tokens_used / weekly_budget * 100), 1)

        provider_counts = {}
        for r in rows:
            provider_counts[r["provider"]] = provider_counts.get(r["provider"], 0) + 1
        total_requests = len(rows)
        breakdown = [
            {"provider": p, "count": c, "percent": round(c / total_requests * 100, 1) if total_requests else 0}
            for p, c in sorted(provider_counts.items(), key=lambda kv: -kv[1])
        ]

        return {
            "plan": plan,
            "today": {
                "tokens_used": tokens_used, "budget": budget, "percent_used": percent_used,
                "doubts_today": doubts_today, "reset_in_seconds": max(0, reset_in_seconds), "unlimited": unlimited
            },
            "weekly": {
                "tokens_used": weekly_tokens_used, "budget": weekly_budget, "percent_used": weekly_percent_used,
                "total_doubts": weekly_doubts, "unlimited": unlimited
            },
            "providers": {"period_days": USAGE_PROVIDER_BREAKDOWN_DAYS, "total_requests": total_requests, "breakdown": breakdown}
        }
    except Exception as e:
        print(f"USAGE SUMMARY ERROR: {e}", flush=True)
        raise HTTPException(status_code=500, detail="Failed to load usage summary")

REPORT_REASONS = {"wrong_answer", "unclear", "diagram_issue", "duplicate", "other"}
MAX_REPORTS_PER_DAY = 20
# mustafatasscr7@gmail.com (owner account) -- exempt from the daily report cap by request, so
# testing/QA-flagging real content issues isn't throttled like an ordinary student account.
UNLIMITED_REPORT_USER_IDS = {"105c4f7f-ec62-492d-a4b5-2a6e5bf31b5f"}

def _ist_today_start_utc_iso() -> str:
    # datetime.min.time() (not a bare `time` import) deliberately -- `time` the module is
    # already imported above for rate_limiter's time.time() calls, and `from datetime import
    # time` would silently shadow it.
    today = date.fromisoformat(_ist_today())
    ist_midnight = datetime.combine(today, datetime.min.time(), tzinfo=IST)
    return ist_midnight.astimezone(timezone.utc).isoformat()

# Logged-in students only (need to know who, both to prevent spam and to rate-limit) -- the
# generic per-IP rate_limiter below guards against rapid-fire bursts, this endpoint's own
# per-user daily count enforces the actual 20/day business rule on top of that.
@app.post("/report-question")
async def report_question(req: ReportQuestionRequest, _: None = Depends(rate_limiter(20, 60))):
    if not req.user_id:
        raise HTTPException(status_code=401, detail="Please log in to report a question.")
    if req.reason not in REPORT_REASONS:
        return {"error": "Invalid reason"}
    try:
        if req.user_id not in UNLIMITED_REPORT_USER_IDS:
            today_rows = await async_client.get(
                f"{SUPABASE_URL}/rest/v1/question_reports", headers=ADMIN_HEADERS,
                params={"user_id": f"eq.{req.user_id}", "created_at": f"gte.{_ist_today_start_utc_iso()}",
                        "select": "id", "limit": MAX_REPORTS_PER_DAY}
            )
            if len(today_rows.json()) >= MAX_REPORTS_PER_DAY:
                raise HTTPException(status_code=429, detail="You've reached today's report limit. Try again tomorrow.")
        note = (req.optional_note or "").strip()[:500] or None
        response = await async_client.post(
            f"{SUPABASE_URL}/rest/v1/question_reports",
            headers={**ADMIN_HEADERS, "Content-Type": "application/json", "Prefer": "return=minimal"},
            json={"pyq_id": req.pyq_id, "user_id": req.user_id, "reason": req.reason, "optional_note": note}
        )
        if response.status_code >= 400:
            return {"error": response.text}
        return {"success": True}
    except HTTPException:
        raise
    except Exception as e:
        return {"error": str(e)}

DIAGRAM_REPORT_REASONS = {"wrong_image", "mislabeled", "low_quality", "incorrect_content", "other"}
DIAGRAM_REPORT_SOURCES = {"chat", "library"}

# Separate table/cap from /report-question -- see create_diagram_reports_table.sql for why this
# isn't just question_reports with an extra column. Own independent MAX_REPORTS_PER_DAY budget
# (not blended with question reports) rather than querying both tables on every submit -- 20/day
# is already a generous anti-spam ceiling for a feature this size on its own.
@app.post("/report-diagram")
async def report_diagram(req: ReportDiagramRequest, _: None = Depends(rate_limiter(20, 60))):
    if not req.user_id:
        raise HTTPException(status_code=401, detail="Please log in to report a diagram.")
    if req.reason not in DIAGRAM_REPORT_REASONS:
        return {"error": "Invalid reason"}
    if req.source not in DIAGRAM_REPORT_SOURCES:
        return {"error": "Invalid source"}
    try:
        if req.user_id not in UNLIMITED_REPORT_USER_IDS:
            today_rows = await async_client.get(
                f"{SUPABASE_URL}/rest/v1/diagram_reports", headers=ADMIN_HEADERS,
                params={"user_id": f"eq.{req.user_id}", "created_at": f"gte.{_ist_today_start_utc_iso()}",
                        "select": "id", "limit": MAX_REPORTS_PER_DAY}
            )
            if len(today_rows.json()) >= MAX_REPORTS_PER_DAY:
                raise HTTPException(status_code=429, detail="You've reached today's report limit. Try again tomorrow.")
        note = (req.optional_note or "").strip()[:500] or None
        response = await async_client.post(
            f"{SUPABASE_URL}/rest/v1/diagram_reports",
            headers={**ADMIN_HEADERS, "Content-Type": "application/json", "Prefer": "return=minimal"},
            json={"diagram_id": req.diagram_id, "user_id": req.user_id, "source": req.source, "reason": req.reason, "optional_note": note}
        )
        if response.status_code >= 400:
            return {"error": response.text}
        return {"success": True}
    except HTTPException:
        raise
    except Exception as e:
        return {"error": str(e)}

@app.post("/pyq")
async def get_pyq(message: Message, _: None = Depends(rate_limiter(15, 60))):
    results = await search_pyq(message.text)
    return {"pyqs": results}

# Threshold is a similarity floor (1 - cosine distance), same convention as match_ncert's own
# match_threshold. Configurable here -- no separate settings store in this project, every other
# tunable constant (MIN_CONFIDENCE_SCORE, FILENAME_MATCH_THRESHOLD, etc) lives as a plain
# module-level constant too. Lowered from an initial 0.75 after live measurement showed even a
# near-exact topical match only scored ~0.72 for this embedding model (text-embedding-3-small's
# absolute similarity scale runs lower than 0.75 for real paraphrased matches) -- 0.75 would have
# meant this feature almost never fired. 0.5 matches match_ncert's own threshold for the same model.
#
# Auto-embed confidence split (chat.html, DIAGRAM_HIGH_CONFIDENCE_THRESHOLD): originally a
# separate, higher tier (0.65) above this floor, re-measured live against the actual
# reviewed-diagram catalog before picking that number -- genuine on-topic matches against real
# rows scored 0.56-0.79, with topical closeness (not "visual" vs "conceptual" phrasing) driving
# the score; a non-visual query ("explain how bacteria are classified by shape", 0.787) scored
# higher than a visually-phrased one on the same topic ("show me the different shapes of
# bacteria", 0.675). So similarity alone can't stand in for visual intent -- VISUAL_INTENT (rule
# 10) is what gates that. Collapsed down to this same 0.5 floor: with VISUAL_INTENT: yes, any
# match at all now auto-embeds -- "matched" and "similarity >= 0.5" are already the same
# condition, since match_diagrams itself never returns a row below this threshold, so a separate
# higher auto-embed bar was just adding a distinct "matched but not confident enough" tier on
# top, not a real accuracy safeguard. The "Show Diagram" button is still very much alive for
# VISUAL_INTENT: "no"/missing (the fail-safe path) and for "yes but nothing matched at all" --
# only that middle tier is now unreachable.
DIAGRAM_MATCH_THRESHOLD = 0.5

# Mirrors the "always-available action button, lazily fetched on click" pattern the PYQ button
# (/pyq) already uses -- chat.html calls this on demand when the student clicks "Show Diagram",
# not during the /chat stream itself, so no change was needed to /chat's plain-text streaming
# response format.
#
# Embedding reuse: get_embedding() caches by sha256(text) in the embedding_cache table (see
# above). /chat already computed and cached an embedding for this exact doubt text via
# search_ncert()'s own get_embedding(text) call before this button can even be clicked (the
# button only exists after the full answer has streamed in) -- so this call is a cache hit in
# practice, not a second real OpenAI charge. The one theoretical exception is a click landing
# before /chat's fire-and-forget cache write (asyncio.create_task, not awaited) has completed;
# that's a same-second race with negligible cost (one extra ~$0.00002 embedding call), not a
# correctness issue.
@app.post("/diagram-match")
async def diagram_match(req: DiagramMatchRequest, _: None = Depends(rate_limiter(15, 60))):
    try:
        chapter = None
        candidate = extract_chapter_candidate(req.answer)
        if candidate:
            chapters_resp = await async_client.get(
                f"{SUPABASE_URL}/rest/v1/diagrams",
                headers={"apikey": SUPABASE_KEY, "Authorization": f"Bearer {SUPABASE_KEY}"},
                params={"select": "chapter", "reviewed": "eq.true"}
            )
            known_chapters = sorted(set(r["chapter"] for r in chapters_resp.json() if r.get("chapter"))) if chapters_resp.status_code == 200 else []
            chapter = fuzzy_match_chapter(candidate, known_chapters)
            # Falls through to filter_chapter=None (search all reviewed diagrams) when
            # extraction/fuzzy-match fails or is ambiguous, per spec.

        embedding = await get_embedding(req.text)
        response = await async_client.post(
            f"{SUPABASE_URL}/rest/v1/rpc/match_diagrams",
            headers={"apikey": SUPABASE_KEY, "Authorization": f"Bearer {SUPABASE_KEY}", "Content-Type": "application/json"},
            json={
                "query_embedding": embedding,
                "match_threshold": DIAGRAM_MATCH_THRESHOLD,
                "match_count": 1,
                "filter_chapter": chapter
            }
        )
        if response.status_code != 200:
            return {"matched": False}
        rows = response.json()
        if not rows:
            return {"matched": False}
        top = rows[0]
        # similarity is already computed by match_diagrams for its own threshold filter above --
        # exposing it here doesn't touch the matching logic itself, it just lets the caller (the
        # frontend's confidence-tier split) see the same number the RPC already had.
        return {"matched": True, "diagram_id": top["id"], "image_url": top["image_url"], "name": top.get("name"), "description": top.get("description"), "similarity": top.get("similarity")}
    except Exception:
        return {"matched": False}

# Powers diagram-library.html's browsable gallery -- the student-facing counterpart to
# /admin/diagrams-list, which requires the admin password. Uses the anon key (not
# ADMIN_HEADERS), same as diagram_match's own chapter lookup above -- reviewed=true rows are
# already anon-readable under this table's RLS policy. reviewed=true is hardcoded, never a
# caller-supplied param, so this can never leak an unreviewed diagram to a student regardless
# of what query params are sent.
@app.get("/diagrams")
async def list_diagrams(subject: str = "", class_num: int = 0, _: None = Depends(rate_limiter(30, 60))):
    try:
        params = {
            "select": "id,subject,class,chapter,name,description,name_hi,description_hi,image_url",
            "reviewed": "eq.true",
            "order": "chapter.asc,name.asc",
            "limit": 500
        }
        if subject:
            params["subject"] = f"eq.{subject}"
        if class_num:
            params["class"] = f"eq.{class_num}"
        response = await async_client.get(
            f"{SUPABASE_URL}/rest/v1/diagrams",
            headers={"apikey": SUPABASE_KEY, "Authorization": f"Bearer {SUPABASE_KEY}"},
            params=params
        )
        if response.status_code >= 400:
            return {"error": response.text}
        return {"diagrams": response.json()}
    except Exception as e:
        return {"error": str(e)}

@app.get("/mock-test-questions")
async def get_mock_test_questions():
    try:
        import random
        headers = {
            "apikey": SUPABASE_KEY,
            "Authorization": f"Bearer {SUPABASE_KEY}"
        }
        bio = http_requests.get(f"{SUPABASE_URL}/rest/v1/pyq?subject=eq.Biology&select=*&limit=200", headers=headers).json()
        phy = http_requests.get(f"{SUPABASE_URL}/rest/v1/pyq?subject=eq.Physics&select=*&limit=200", headers=headers).json()
        che = http_requests.get(f"{SUPABASE_URL}/rest/v1/pyq?subject=eq.Chemistry&select=*&limit=200", headers=headers).json()
        bio_q = random.sample(bio, min(90, len(bio)))
        phy_q = random.sample(phy, min(45, len(phy)))
        che_q = random.sample(che, min(45, len(che)))
        questions = bio_q + phy_q + che_q
        return {"questions": questions, "total": len(questions)}
    except Exception as e:
        return {"error": str(e)}

@app.get("/mock-tests/available")
async def get_available_mock_tests(user_id: str = ""):
    # Published tests minus ones this user has already completed. RLS on mock_tests
    # already restricts the anon key to is_published=true rows, so the published-only
    # filter here is belt-and-suspenders, not the only guard.
    try:
        headers = {"apikey": SUPABASE_KEY, "Authorization": f"Bearer {SUPABASE_KEY}"}
        response = await async_client.get(
            f"{SUPABASE_URL}/rest/v1/mock_tests",
            headers=headers,
            params={
                "is_published": "eq.true",
                "select": "id,title,questions_extracted,physics_count,chemistry_count,biology_count,published_at",
                "order": "published_at.desc"
            }
        )
        if response.status_code >= 400:
            return {"error": response.text}
        published = response.json()

        taken_ids = set()
        if user_id:
            # Service-role key: mock_results is RLS-scoped to the owning user, and this
            # lookup runs server-side on the student's own behalf (already filtered to
            # their own user_id below) -- same justification as get_student_context() above.
            taken_resp = await async_client.get(
                f"{SUPABASE_URL}/rest/v1/mock_results",
                headers={"apikey": SUPABASE_SERVICE_KEY, "Authorization": f"Bearer {SUPABASE_SERVICE_KEY}"},
                params={
                    "user_id": f"eq.{user_id}",
                    "mock_test_id": "not.is.null",
                    "select": "mock_test_id"
                }
            )
            if taken_resp.status_code < 400:
                taken_ids = {r["mock_test_id"] for r in taken_resp.json()}

        available = [t for t in published if t["id"] not in taken_ids]
        return {"available": available, "total_published": len(published)}
    except Exception as e:
        return {"error": str(e)}

@app.get("/mock-tests/{mock_test_id}/questions")
async def get_mock_test_questions_by_id(mock_test_id: int):
    try:
        headers = {"apikey": SUPABASE_KEY, "Authorization": f"Bearer {SUPABASE_KEY}"}
        # Double-gated on is_published, on top of the RLS policy already enforcing it --
        # a draft test's questions should never be fetchable even by guessing its id.
        test_resp = await async_client.get(
            f"{SUPABASE_URL}/rest/v1/mock_tests",
            headers=headers,
            params={"id": f"eq.{mock_test_id}", "is_published": "eq.true", "select": "id"}
        )
        if test_resp.status_code >= 400 or not test_resp.json():
            return {"error": "Mock test not found"}
        response = await async_client.get(
            f"{SUPABASE_URL}/rest/v1/mock_test_questions",
            headers=headers,
            params={
                "mock_test_id": f"eq.{mock_test_id}",
                "select": "id,subject,chapter,year,question,option_a,option_b,option_c,option_d,correct_answer,difficulty,diagram_url,option_a_diagram_url,option_b_diagram_url,option_c_diagram_url,option_d_diagram_url",
                "order": "question_order.asc"
            }
        )
        if response.status_code >= 400:
            return {"error": response.text}
        questions = response.json()
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
        # await async_client, not the blocking http_requests (sync `requests`) this used to call
        # directly from an async def route -- with uvicorn.run() here started with no workers=
        # arg (one process, one event loop) and Railway's numReplicas: 1, a blocking call ties up
        # the ONLY event loop for its full duration, stalling every other concurrent request
        # (including this same admin's own second Promise.all call to pyq-classifier-data below)
        # until it returns. Found while investigating a "could not reach the server" report for
        # Chemistry specifically -- couldn't reproduce a hard failure directly (all three subjects
        # returned 200 in isolation, concurrently, and via a real browser against production), but
        # this is a genuine root-cause candidate: Chemistry's classifier-data payload is the
        # largest of the three subjects, making it the one most likely to hold the shared event
        # loop hostage long enough to trip a timeout if the server happens to be busy with real
        # traffic at that moment. Fixed regardless, since it's wrong either way.
        response = await async_client.get(
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
    id: str = None,
    subject: str = None,
    chapter: str = None,
    search: str = None,
    is_active: str = None,
    with_uploaded_diagram: str = None,
    reviewed: str = None,
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
            "option_b_diagram_url,option_c_diagram_url,option_d_diagram_url,reviewed,created_at"
            if full else
            "id,subject,chapter,question,correct_answer,is_active,year"
        )
        params = {
            "select": select_fields,
            "order": f"{sort_col}.{sort_direction}",
            "limit": page_size,
            "offset": offset
        }
        # Direct id lookup (used by the "jump to this question" link from Reported Questions)
        # overrides every other filter -- the point is fetching one exact row, not narrowing
        # a search.
        if id:
            params["id"] = f"eq.{id}"
        if subject in ("Biology", "Physics", "Chemistry"):
            params["subject"] = f"eq.{subject}"
        if chapter:
            params["chapter"] = f"eq.{chapter}"
        if is_active in ("true", "false"):
            params["is_active"] = f"eq.{is_active}"
        if reviewed in ("true", "false"):
            params["reviewed"] = f"eq.{reviewed}"
        # Both filters below are OR-groups across the same 4-5 columns, and both need to be
        # applyable at once (e.g. searching while also filtering to uploaded-diagram rows) --
        # collected as separate or(...) groups and combined via and=() at the end instead of
        # each just assigning params["or"], which would let the second one silently clobber
        # the first if both filters were active together.
        or_groups = []
        if with_uploaded_diagram == "true":
            # An actually-uploaded image, not just the AI's has_diagram guess from extraction --
            # that flag just means "this looked like it needed one," independent of whether
            # anyone's uploaded the image yet. neq. (not equal to empty string) rather than
            # not.is.null: some rows have '' instead of NULL for a never-uploaded slot, and
            # NULL <> '' is not TRUE in SQL's 3-valued logic, so neq. alone excludes both.
            or_groups.append(
                "or(diagram_url.neq.,option_a_diagram_url.neq.,"
                "option_b_diagram_url.neq.,option_c_diagram_url.neq.,"
                "option_d_diagram_url.neq.)"
            )
        if search:
            # Previously only checked `question`, so searching for a term that's only in one
            # of the options (very common -- e.g. a specific answer choice) silently returned
            # nothing. Quoting the value is required, not cosmetic: an unquoted comma or
            # parenthesis in the search term breaks PostgREST's or=() parsing entirely (400
            # error) since those characters are also this list's own separators/grouping.
            safe_search = search.replace("\\", "\\\\").replace('"', '\\"')
            or_groups.append(
                f'or(question.ilike."*{safe_search}*",'
                f'option_a.ilike."*{safe_search}*",'
                f'option_b.ilike."*{safe_search}*",'
                f'option_c.ilike."*{safe_search}*",'
                f'option_d.ilike."*{safe_search}*")'
            )
        if len(or_groups) == 1:
            params["or"] = "(" + or_groups[0][3:]  # strip the "or(" wrapper, top-level or= wants bare (...)
        elif len(or_groups) > 1:
            params["and"] = "(" + ",".join(or_groups) + ")"

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
    if body.subject is not None and body.subject not in ("Biology", "Physics", "Chemistry"):
        return {"error": "Invalid subject"}
    update_fields = {}
    if body.subject is not None:
        update_fields["subject"] = body.subject
    if body.chapter is not None:
        update_fields["chapter"] = body.chapter
    if body.correct_answer is not None:
        update_fields["correct_answer"] = body.correct_answer
    if body.is_active is not None:
        update_fields["is_active"] = body.is_active
    if body.question is not None:
        update_fields["question"] = body.question
    if body.option_a is not None:
        update_fields["option_a"] = body.option_a
    if body.option_b is not None:
        update_fields["option_b"] = body.option_b
    if body.option_c is not None:
        update_fields["option_c"] = body.option_c
    if body.option_d is not None:
        update_fields["option_d"] = body.option_d
    if body.year is not None:
        update_fields["year"] = body.year
    if body.source_tag is not None:
        update_fields["source_tag"] = body.source_tag
    if body.class_ is not None:
        update_fields["class"] = body.class_
    if body.reviewed is not None:
        update_fields["reviewed"] = body.reviewed
    # Diagram fields need to distinguish "not sent" from "sent as null" -- removing a photo in
    # the edit form means explicitly clearing the URL, and `is not None` would silently ignore
    # that. model_fields_set has whatever keys were actually present in the request JSON.
    for diagram_field in ("diagram_url", "option_a_diagram_url", "option_b_diagram_url",
                           "option_c_diagram_url", "option_d_diagram_url"):
        if diagram_field in body.model_fields_set:
            update_fields[diagram_field] = getattr(body, diagram_field)
    if not update_fields:
        return {"error": "No fields to update"}
    try:
        # async_client, not the blocking http_requests used elsewhere -- same reasoning as the
        # /admin/pyq-delete fix: this is async def, so a blocking call here stalls the whole
        # server's event loop, and this endpoint is now called from the PYQ Preview edit form
        # right alongside delete/duplicate-scan traffic on the same page.
        response = await async_client.patch(
            f"{SUPABASE_URL}/rest/v1/pyq",
            headers={**ADMIN_HEADERS, "Content-Type": "application/json", "Prefer": "return=representation"},
            params={"id": f"eq.{pyq_id}", "select": "id,subject,chapter,question,option_a,option_b,option_c,"
                    "option_d,correct_answer,is_active,year,source_tag,class,has_diagram,diagram_url,"
                    "option_a_diagram_url,option_b_diagram_url,option_c_diagram_url,option_d_diagram_url,reviewed,created_at"},
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
    # Uses the shared httpx.AsyncClient (already set up for /chat) instead of the blocking
    # `requests` library other admin endpoints use -- this one's async def, so a synchronous
    # call here would stall the whole event loop, and every other in-flight request with it,
    # for as long as Supabase takes to respond. Directly relevant here: this endpoint gets
    # called from the duplicate-finder UI, sometimes right after a several-second-long
    # /admin/pyq-duplicates scan, when the server is otherwise busiest.
    try:
        response = await async_client.delete(
            f"{SUPABASE_URL}/rest/v1/pyq",
            headers={**ADMIN_HEADERS, "Prefer": "return=representation"},
            params={"id": f"eq.{pyq_id}"}
        )
        if response.status_code >= 400:
            return {"error": response.text}
        deleted = response.json()
        if not deleted:
            return {"error": "Row not found"}
        # Returning the deleted row (not just success:true) is what lets the admin tool's
        # Ctrl+Z undo restore it -- this is a hard DELETE, so this response is the only place
        # that data still exists once the request completes.
        return {"success": True, "deleted": deleted[0]}
    except Exception as e:
        return {"error": str(e)}

# ---------- Difficulty classification (Easy/Moderate/Difficult), cached permanently ----------
# Computed once per question by DeepSeek-V4-Flash, never recomputed once the difficulty column
# is set -- both the on-display path (/classify-difficulty) and the one-time sweep
# (/admin/backfill-difficulty) always check for an existing value first and skip the DeepSeek
# call entirely if one's already there.

def classify_difficulty(question: str, option_a: str, option_b: str, option_c: str, option_d: str,
                         chapter: str = None, source_tag: str = None):
    """Single DeepSeek call classifying one question's difficulty. Returns (label, tokens_used)
    where label is exactly 'Easy'/'Moderate'/'Difficult', or (None, tokens_used) if the model's
    reply can't be parsed cleanly -- callers must leave the DB column null in that case rather
    than cache a bad value, so it gets retried on the next display/backfill pass instead of
    permanently sticking with a wrong guess."""
    context_lines = [f"Question: {question}", f"(A) {option_a}", f"(B) {option_b}",
                      f"(C) {option_c}", f"(D) {option_d}"]
    if chapter:
        context_lines.append(f"Chapter: {chapter}")
    if source_tag:
        context_lines.append(f"Source: {source_tag}")
    try:
        response = deepseek_client.messages.create(
            model="deepseek-v4-flash",
            max_tokens=10,
            thinking={"type": "disabled"},
            system=(
                "You are classifying the difficulty of a NEET (Indian medical entrance exam) "
                "multiple-choice question for a student-facing difficulty badge.\n\n"
                "Classify as:\n"
                "- Easy: single-fact recall, directly stated in NCERT, answerable from memory alone.\n"
                "- Moderate: requires applying one NCERT concept, a single calculation step, or "
                "distinguishing between similar-sounding options.\n"
                "- Difficult: requires combining multiple NCERT concepts, multi-step reasoning, "
                "or is from a source exam type known to run harder than standard NEET (e.g. AIIMS-"
                "pattern, JIPMER-pattern) -- weigh the Source line if present, but the question's "
                "own content matters more than the label on it.\n\n"
                "Reply with EXACTLY one word: Easy, Moderate, or Difficult. Nothing else -- no "
                "punctuation, no explanation."
            ),
            messages=[{"role": "user", "content": "\n".join(context_lines)}]
        )
        raw = response.content[0].text.strip()
        tokens = response.usage.input_tokens + response.usage.output_tokens
        for label in ("Easy", "Moderate", "Difficult"):
            if label.lower() in raw.lower():
                return label, tokens
        print(f"DIFFICULTY CLASSIFY: unparseable reply {raw!r}")
        return None, tokens
    except Exception as e:
        print(f"DIFFICULTY CLASSIFY ERROR: {e}")
        return None, 0

ALLOWED_DIFFICULTY_TABLES = {"pyq", "mock_test_questions"}

@app.post("/classify-difficulty")
async def classify_difficulty_endpoint(req: ClassifyDifficultyRequest, _: None = Depends(rate_limiter(30, 60))):
    """The on-display trigger: called by every student-facing page right after it loads a batch
    of questions. Reads current difficulty for all requested ids in one query: already-set ones
    are returned as-is (zero DeepSeek calls), null ones get classified now and cached before
    returning, so the badge is present on first paint rather than popping in later."""
    if req.table not in ALLOWED_DIFFICULTY_TABLES:
        return {"error": f"table must be one of {sorted(ALLOWED_DIFFICULTY_TABLES)}"}
    if not req.ids:
        return {"difficulties": {}}
    id_list = ",".join(req.ids)
    try:
        resp = await async_client.get(
            f"{SUPABASE_URL}/rest/v1/{req.table}",
            headers=ADMIN_HEADERS,
            params={"id": f"in.({id_list})",
                    "select": "id,difficulty,question,option_a,option_b,option_c,option_d,chapter,source_tag"}
        )
        if resp.status_code >= 400:
            return {"error": resp.text}
        rows = resp.json()
    except Exception as e:
        return {"error": str(e)}

    result = {}
    to_classify = []
    for row in rows:
        if row.get("difficulty"):
            result[row["id"]] = row["difficulty"]
        else:
            to_classify.append(row)

    if to_classify:
        loop = asyncio.get_event_loop()
        with ThreadPoolExecutor(max_workers=min(5, len(to_classify))) as executor:
            futures = {
                row["id"]: loop.run_in_executor(
                    executor, classify_difficulty, row["question"], row.get("option_a", ""),
                    row.get("option_b", ""), row.get("option_c", ""), row.get("option_d", ""),
                    row.get("chapter"), row.get("source_tag")
                )
                for row in to_classify
            }
            classified = await asyncio.gather(*futures.values())
            for row_id, (label, _tokens) in zip(futures.keys(), classified):
                if label:
                    result[row_id] = label
                    try:
                        await async_client.patch(
                            f"{SUPABASE_URL}/rest/v1/{req.table}",
                            headers={**ADMIN_HEADERS, "Content-Type": "application/json"},
                            params={"id": f"eq.{row_id}"},
                            json={"difficulty": label}
                        )
                    except Exception as e:
                        print(f"DIFFICULTY SAVE ERROR ({req.table}/{row_id}): {e}")

    return {"difficulties": result}

@app.post("/admin/backfill-difficulty")
async def admin_backfill_difficulty(table: str = "pyq", limit: int = 200, _: None = Depends(verify_admin)):
    """One-time bulk sweep so the whole bank gets classified upfront instead of trickling in via
    organic page views. Processes up to `limit` null-difficulty rows per call (not the whole
    table in one request -- pyq alone is thousands of rows, a single request classifying all of
    them would time out long before finishing); call it repeatedly (e.g. from a small script)
    until remaining_null hits 0."""
    if table not in ALLOWED_DIFFICULTY_TABLES:
        return {"error": f"table must be one of {sorted(ALLOWED_DIFFICULTY_TABLES)}"}
    try:
        resp = await async_client.get(
            f"{SUPABASE_URL}/rest/v1/{table}",
            headers=ADMIN_HEADERS,
            params={"difficulty": "is.null",
                    "select": "id,question,option_a,option_b,option_c,option_d,chapter,source_tag",
                    "limit": str(limit)}
        )
        if resp.status_code >= 400:
            return {"error": resp.text}
        rows = resp.json()
    except Exception as e:
        return {"error": str(e)}

    if not rows:
        return {"classified": 0, "failed": 0, "remaining_null": 0, "tokens_used": 0}

    loop = asyncio.get_event_loop()
    with ThreadPoolExecutor(max_workers=8) as executor:
        futures = [
            loop.run_in_executor(
                executor, classify_difficulty, row["question"], row.get("option_a", ""),
                row.get("option_b", ""), row.get("option_c", ""), row.get("option_d", ""),
                row.get("chapter"), row.get("source_tag")
            )
            for row in rows
        ]
        classified = await asyncio.gather(*futures)

    classified_count = 0
    failed_count = 0
    total_tokens = 0
    for row, (label, tokens) in zip(rows, classified):
        total_tokens += tokens
        if label:
            try:
                save_resp = await async_client.patch(
                    f"{SUPABASE_URL}/rest/v1/{table}",
                    headers={**ADMIN_HEADERS, "Content-Type": "application/json"},
                    params={"id": f"eq.{row['id']}"},
                    json={"difficulty": label}
                )
                if save_resp.status_code < 400:
                    classified_count += 1
                else:
                    failed_count += 1
            except Exception:
                failed_count += 1
        else:
            failed_count += 1

    count_resp = await async_client.head(
        f"{SUPABASE_URL}/rest/v1/{table}",
        headers={**ADMIN_HEADERS, "Prefer": "count=exact"},
        params={"difficulty": "is.null", "select": "id"}
    )
    remaining = int(count_resp.headers.get("content-range", "*/0").split("/")[-1])

    return {"classified": classified_count, "failed": failed_count, "remaining_null": remaining,
            "tokens_used": total_tokens}

# ---------- On-demand correct_answer solve, cached permanently ----------
# A real chunk of the PYQ bank was extracted with correct_answer left blank (never verified
# during scanning). Rather than a UI-side workaround, this actually resolves it: the first time
# a student clicks an option on a blank-answer question, DeepSeek-V4-Flash determines and caches
# the real answer -- same "check first, solve once, save" shape as classify_difficulty above.
# Every click after that (any student, any time) is a pure DB read, no DeepSeek call.

def solve_correct_answer(question: str, option_a: str, option_b: str, option_c: str, option_d: str):
    """Single DeepSeek call determining which option is correct. Returns (letter, tokens_used)
    where letter is exactly 'a'/'b'/'c'/'d', or (None, tokens_used) if the reply can't be parsed
    -- caller must leave correct_answer blank in that case so it's retried next time rather than
    permanently caching a wrong guess."""
    try:
        response = deepseek_client.messages.create(
            model="deepseek-v4-flash",
            max_tokens=10,
            thinking={"type": "disabled"},
            system=(
                "You are answering a NEET (Indian medical entrance exam) multiple-choice question. "
                "Determine which option is correct based on NCERT-level knowledge.\n\n"
                "Reply with EXACTLY one letter: A, B, C, or D. Nothing else -- no punctuation, "
                "no explanation."
            ),
            messages=[{"role": "user", "content": f"Question: {question}\n(A) {option_a}\n(B) {option_b}\n(C) {option_c}\n(D) {option_d}"}]
        )
        raw = response.content[0].text.strip().upper()
        tokens = response.usage.input_tokens + response.usage.output_tokens
        for letter in ("A", "B", "C", "D"):
            if letter in raw:
                return letter.lower(), tokens
        print(f"SOLVE ANSWER: unparseable reply {raw!r}")
        return None, tokens
    except Exception as e:
        print(f"SOLVE ANSWER ERROR: {e}")
        return None, 0

class EnsureCorrectAnswerRequest(BaseModel):
    pyq_id: str

@app.post("/ensure-correct-answer")
async def ensure_correct_answer_endpoint(req: EnsureCorrectAnswerRequest, _: None = Depends(rate_limiter(30, 60))):
    try:
        resp = await async_client.get(
            f"{SUPABASE_URL}/rest/v1/pyq",
            headers=ADMIN_HEADERS,
            params={"id": f"eq.{req.pyq_id}",
                    "select": "id,correct_answer,question,option_a,option_b,option_c,option_d"}
        )
        if resp.status_code >= 400:
            return {"error": resp.text}
        rows = resp.json()
        if not rows:
            return {"error": "Question not found"}
        row = rows[0]
    except Exception as e:
        return {"error": str(e)}

    if row.get("correct_answer"):
        return {"correct_answer": row["correct_answer"]}

    loop = asyncio.get_event_loop()
    label, _tokens = await loop.run_in_executor(
        None, solve_correct_answer, row["question"], row.get("option_a", ""),
        row.get("option_b", ""), row.get("option_c", ""), row.get("option_d", "")
    )
    if not label:
        return {"error": "Could not determine the answer"}

    try:
        await async_client.patch(
            f"{SUPABASE_URL}/rest/v1/pyq",
            headers={**ADMIN_HEADERS, "Content-Type": "application/json"},
            params={"id": f"eq.{req.pyq_id}"},
            json={"correct_answer": label}
        )
    except Exception as e:
        print(f"CORRECT_ANSWER SAVE ERROR ({req.pyq_id}): {e}")

    return {"correct_answer": label}

# ---------- Multiple Solution Methods (alternate solving approach, on-demand + permanently cached) ----------

def generate_alternate_method(question: str, primary_answer: str, options_context: str = "", language: str = "en"):
    """Single DeepSeek call that both decides whether a genuinely different, pedagogically valid
    method exists for this specific question AND generates it if so -- not every doubt has a real
    second method (most don't), so this must not fabricate one just to fill the feature.

    The judgment/content split is a deliberate two-line protocol (ALTERNATE EXISTS / NO ALTERNATE
    on line 1, content after) rather than a single "reply NONE or the answer" instruction -- live
    testing during development found the single-instruction version defaulted to NONE almost every
    time, even on textbook energy-vs-kinematics cases, because with thinking disabled the model
    had no room to reason before committing to the cheap token. Forcing an explicit judgment line
    first measurably fixed that without needing to enable thinking.

    Returns (text, tokens_used) where text is '' (not None) when the model determined no genuine
    alternate exists -- a real, cacheable verdict, not a retry state -- or (None, 0) on a hard
    failure the caller must NOT cache, so it's retried next time instead of permanently
    remembering a failure as "no alternate exists"."""
    lang_instruction = (
        "\n\nRespond ONLY in Hindi (Devanagari script) -- every word in Hindi, no English words or "
        "Hinglish mixing. The ONLY exceptions are LaTeX/KaTeX math notation, chemical formulas/"
        "symbols, and units, which stay exactly as-is."
    ) if language == "hi" else ""
    try:
        response = deepseek_client.messages.create(
            model="deepseek-v4-flash",
            max_tokens=800,
            thinking={"type": "disabled"},
            system=f"""You are a NEET exam expert reviewing a solved question. This feature only applies to numerical or derivation-style questions that involve a calculation or step-by-step derivation to reach the answer. If the question is a simple factual recall, identification, or definition question with no calculation/derivation involved (e.g. "which organelle is X", "what is the term for Y", "name the scientist who discovered Z"), always reply NO ALTERNATE -- restating supporting facts through a different angle is NOT a genuine alternate solving method, only a real second calculation or derivation path counts.

For questions that DO involve a calculation or derivation, find a genuinely different valid method to solve the SAME question, starting from a different core PHYSICAL OR CHEMICAL PRINCIPLE than the one already used (e.g. energy conservation instead of force/kinematics, mole method instead of equivalent-weight method, angular momentum instead of force analysis).

A valid alternate must use a genuinely different principle, not just a different algebraic route to the identical relationship. If your alternate ends up re-deriving the same underlying formula the primary method already used (even via a different intermediate path, like re-deriving a known formula from its definition instead of quoting it), that is NOT a genuine alternate -- it is the same method in disguise. A valid alternate must also be a GENERAL, repeatable technique that would still work if the numbers changed, not a shortcut based on a memorized reference value. Simple single-formula plug-in questions (a basic unit conversion, direct substitution into one named formula with no derivation choice) genuinely have only one method -- do not invent an alternate for these.

First, on one line, write your judgment: either "ALTERNATE EXISTS" or "NO ALTERNATE".
Then on the next line, if ALTERNATE EXISTS, write the full alternate method as bullet-point steps. Every piece of math, however short, must be wrapped in KaTeX delimiters -- $formula$ inline, $$formula$$ display -- with NO exceptions, including a short formula mentioned parenthetically inside a sentence. Writing raw LaTeX commands inside plain parentheses instead of dollar signs (e.g. "the growth rate (R = \\frac{{dN}}{{dt}})" or "(N = -\\frac{{b}}{{2a}})") is wrong and renders as broken literal text, not math -- it must be "the growth rate $R = \\frac{{dN}}{{dt}}$" and "$N = -\\frac{{b}}{{2a}}$" instead. If NO ALTERNATE, write nothing else.

Do not default to "NO ALTERNATE" just because the final numeric answer is the same -- a different starting PRINCIPLE that reaches the same answer IS a genuine alternate and has real teaching value, for questions that involve a real calculation. Only say NO ALTERNATE if you truly cannot identify any different underlying principle, if the only "alternate" you can think of just re-derives the same formula a different way, or if the question is factual recall with no calculation at all.{lang_instruction}""",
            messages=[{"role": "user", "content": f"Question: {question}\n{options_context}\n\nPrimary method already given to the student:\n{primary_answer}\n\nFind a genuinely different valid alternate method."}]
        )
        raw = response.content[0].text.strip()
        tokens = response.usage.input_tokens + response.usage.output_tokens
        first_line, _, rest = raw.partition("\n")
        if first_line.strip().upper().startswith("NO ALTERNATE"):
            return "", tokens
        if first_line.strip().upper().startswith("ALTERNATE EXISTS"):
            return rest.strip(), tokens
        # Unparseable judgment line -- don't guess, don't cache a false verdict either way.
        print(f"ALTERNATE METHOD: unparseable judgment line {first_line!r}")
        return None, tokens
    except Exception as e:
        print(f"ALTERNATE METHOD ERROR: {e}")
        return None, 0

class ChatAlternateMethodRequest(BaseModel):
    question: str
    primary_answer: str
    language: str = "en"
    user_id: str = ""

@app.post("/chat-alternate-method")
async def chat_alternate_method_endpoint(req: ChatAlternateMethodRequest, request: Request, _: None = Depends(rate_limiter(20, 60))):
    if not req.question or not req.primary_answer:
        return {"alternate_method": ""}

    # Same hash the /chat shared cache already uses (language:text.strip().lower()) -- computed
    # here rather than trusting a client-supplied hash, so this can never be pointed at the wrong
    # cache row.
    answer_hash = hashlib.sha256(f"{req.language}:{req.question.strip().lower()}".encode()).hexdigest()
    cache_headers = {"apikey": SUPABASE_SERVICE_KEY, "Authorization": f"Bearer {SUPABASE_SERVICE_KEY}"}

    resp = await async_client.get(
        f"{SUPABASE_URL}/rest/v1/answer_cache",
        headers=cache_headers,
        params={"question_hash": f"eq.{answer_hash}", "select": "alternate_method"}
    )
    rows = resp.json()
    if rows and rows[0].get("alternate_method") is not None:
        return {"alternate_method": rows[0]["alternate_method"]}

    loop = asyncio.get_event_loop()
    text, tokens = await loop.run_in_executor(None, generate_alternate_method, req.question, req.primary_answer, "", req.language)
    if text is None:
        return {"alternate_method": ""}

    if tokens:
        await log_token_usage(req.user_id, tokens, _client_ip(request))

    # PATCH, never upsert/insert: a row only exists here at all if the primary answer went
    # through the SHARED cache (personalize=false or guest) -- writing a new row keyed on this
    # hash would risk a personalized answer's alternate method leaking into the shared cache for
    # a totally different user asking the same question text later. If there's no shared row,
    # this just doesn't persist -- the feature still works for this one render, it just isn't
    # free next time, exactly mirroring how the primary answer itself already isn't cached for
    # personalized requests either.
    if rows:
        try:
            await async_client.patch(
                f"{SUPABASE_URL}/rest/v1/answer_cache",
                headers={**cache_headers, "Content-Type": "application/json"},
                params={"question_hash": f"eq.{answer_hash}"},
                json={"alternate_method": text}
            )
        except Exception as e:
            print(f"ALTERNATE METHOD SAVE ERROR (chat, hash={answer_hash}): {e}")

    return {"alternate_method": text}

class SolveAlternateMethodRequest(BaseModel):
    pyq_id: str
    question: str
    option_a: str = ""
    option_b: str = ""
    option_c: str = ""
    option_d: str = ""
    primary_solution: str
    language: str = "en"
    user_id: str = ""

@app.post("/solve-alternate-method")
async def solve_alternate_method_endpoint(req: SolveAlternateMethodRequest, request: Request, _: None = Depends(rate_limiter(20, 60))):
    if not req.pyq_id or not req.primary_solution:
        return {"alternate_method": ""}

    resp = await async_client.get(
        f"{SUPABASE_URL}/rest/v1/pyq_solution_cache",
        headers=SOLVE_CACHE_HEADERS,
        params={"pyq_id": f"eq.{req.pyq_id}", "language": f"eq.{req.language}", "select": "alternate_method"}
    )
    rows = resp.json()
    if rows and rows[0].get("alternate_method") is not None:
        return {"alternate_method": rows[0]["alternate_method"]}

    options_context = f"A) {req.option_a}\nB) {req.option_b}\nC) {req.option_c}\nD) {req.option_d}"
    loop = asyncio.get_event_loop()
    text, tokens = await loop.run_in_executor(
        None, generate_alternate_method, req.question, req.primary_solution, options_context, req.language
    )
    if text is None:
        return {"alternate_method": ""}

    if tokens:
        await log_token_usage(req.user_id, tokens, _client_ip(request))

    # A pyq_solution_cache row always exists by the time this is called (the primary solution is
    # unconditionally cached the first time /solve runs for this pyq_id+language, no
    # personalization gate the way /chat has), so PATCH-only is safe here too.
    if rows:
        try:
            await async_client.patch(
                f"{SUPABASE_URL}/rest/v1/pyq_solution_cache",
                headers={**SOLVE_CACHE_HEADERS, "Content-Type": "application/json"},
                params={"pyq_id": f"eq.{req.pyq_id}", "language": f"eq.{req.language}"},
                json={"alternate_method": text}
            )
        except Exception as e:
            print(f"ALTERNATE METHOD SAVE ERROR (solve, pyq_id={req.pyq_id}): {e}")

    return {"alternate_method": text}

# ---------- Admin: PDF scan -> review -> save pipeline (admin-pdf-review.html) ----------

def _create_processed_pdf_row(filename: str, subject: str):
    # Tracking is best-effort and must never block the actual scan -- swallow failures here
    # rather than let a processed_pdfs write error surface as a scan failure to the admin.
    try:
        response = http_requests.post(
            f"{SUPABASE_URL}/rest/v1/processed_pdfs",
            headers={**ADMIN_HEADERS, "Content-Type": "application/json", "Prefer": "return=representation"},
            json={"filename": filename, "subject": subject, "status": "processing"}
        )
        if response.status_code < 400:
            rows = response.json()
            if rows:
                return rows[0]["id"]
    except Exception:
        pass
    return None

def _update_processed_pdf_row(row_id, status: str, questions_extracted=None, error_message: str = None):
    if row_id is None:
        return
    body = {"status": status, "processed_at": datetime.now(timezone.utc).isoformat()}
    if questions_extracted is not None:
        body["questions_extracted"] = questions_extracted
    if error_message is not None:
        body["error_message"] = error_message[:2000]
    try:
        http_requests.patch(
            f"{SUPABASE_URL}/rest/v1/processed_pdfs",
            headers={**ADMIN_HEADERS, "Content-Type": "application/json"},
            params={"id": f"eq.{row_id}"},
            json=body
        )
    except Exception:
        pass

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

    processed_pdf_id = _create_processed_pdf_row(req.filename, req.subject) if req.filename else None
    try:
        result = scan_pdf_bytes(pdf_bytes, req.subject)
    except Exception as e:
        _update_processed_pdf_row(processed_pdf_id, "failed", error_message=str(e))
        return {"error": str(e)}
    if result.get("error"):
        _update_processed_pdf_row(processed_pdf_id, "failed", error_message=result["error"])
    result["processed_pdf_id"] = processed_pdf_id
    return result

@app.get("/admin/pyq-classifier-data")
async def admin_pyq_classifier_data(subject: str, _: None = Depends(verify_admin)):
    if subject not in ("Biology", "Physics", "Chemistry"):
        return {"error": "Invalid subject"}
    try:
        # await async_client (not blocking http_requests) -- same reasoning as pyq-chapters above.
        # Also paginated now, not a single limit:3000 request: PostgREST caps a single response at
        # 1000 rows regardless of what limit a caller asks for, which was silently truncating this
        # for Chemistry (2305 active rows) and Biology (2249) -- confirmed live, both were quietly
        # returning only 1000 rows to the frontend. That starved buildClassifier()'s chapter
        # suggestions and checkDuplicates' exact-match check of more than half the real data for
        # those two subjects, on every successful request, not just an intermittent failure.
        # Same page-through-until-short-page pattern as /admin/question-reports.
        all_rows = []
        page_size = 1000
        offset = 0
        while True:
            response = await async_client.get(
                f"{SUPABASE_URL}/rest/v1/pyq",
                headers=ADMIN_HEADERS,
                params={
                    "subject": f"eq.{subject}",
                    "is_active": "eq.true",
                    "select": "question,chapter,class",
                    "order": "id.asc",
                    "limit": page_size,
                    "offset": offset
                }
            )
            page = response.json()
            all_rows.extend(page)
            if len(page) < page_size:
                break
            offset += page_size
        # Includes rows with no chapter yet too, so the frontend's exact-duplicate
        # check can catch dupes against still-untagged rows. buildClassifier() itself
        # skips rows with no chapter since they're useless as labeled training examples.
        return {"rows": all_rows}
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

DIAGRAMS_BUCKET = "ncert-daigrams"  # bucket name as actually created in Supabase (matches the existing Q-Daigrams-BIO typo convention)
ALLOWED_DIAGRAM_EXTENSIONS = {"png", "jpg", "jpeg", "webp"}

# Reference diagrams (admin-diagram-upload.html) -- separate bucket from Q-Daigrams-BIO
# since these are standalone NCERT-style reference images for future chat doubt-matching,
# not attached to a specific pyq/mock_test_questions row. Must be created manually as a
# PUBLIC bucket in the Supabase dashboard before this endpoint works -- there is no
# programmatic bucket-creation call anywhere in this codebase.
@app.post("/admin/diagram-upload")
async def admin_diagram_upload(body: DiagramUploadRequest, _: None = Depends(verify_admin)):
    ext = body.filename.rsplit(".", 1)[-1].lower() if "." in body.filename else ""
    if ext not in ALLOWED_DIAGRAM_EXTENSIONS or not body.media_type.startswith("image/"):
        return {"error": "Only PNG, JPG, and WEBP images are allowed"}
    try:
        file_bytes = base64.b64decode(body.data)
    except Exception:
        return {"error": "Could not decode image data"}
    import uuid
    path = f"{uuid.uuid4().hex}.{ext}"
    try:
        response = http_requests.post(
            f"{SUPABASE_URL}/storage/v1/object/{DIAGRAMS_BUCKET}/{path}",
            headers={
                "apikey": SUPABASE_SERVICE_KEY,
                "Authorization": f"Bearer {SUPABASE_SERVICE_KEY}",
                "Content-Type": body.media_type
            },
            data=file_bytes
        )
        if response.status_code >= 400:
            return {"error": f"Storage upload failed (is the '{DIAGRAMS_BUCKET}' bucket created and public?): {response.text}"}
        return {"url": f"{SUPABASE_URL}/storage/v1/object/public/{DIAGRAMS_BUCKET}/{path}"}
    except Exception as e:
        return {"error": str(e)}

@app.post("/admin/diagrams-create")
async def admin_diagrams_create(body: DiagramCreate, _: None = Depends(verify_admin)):
    if body.subject not in ("Biology", "Physics", "Chemistry"):
        return {"error": "Invalid subject"}
    if not body.chapter.strip() or not body.name.strip() or not body.image_url.strip():
        return {"error": "chapter, name, and image_url are required"}
    name = body.name.strip()
    chapter = body.chapter.strip()
    description = body.description.strip() if body.description else None
    subtopic = body.subtopic.strip() if body.subtopic else None
    subtopic_hi = body.subtopic_hi.strip() if body.subtopic_hi else None
    name_hi = body.name_hi.strip() if body.name_hi else None
    description_hi = body.description_hi.strip() if body.description_hi else None
    try:
        # Embedding is a match-quality enhancement for chat.html's diagram-matching, not a hard
        # requirement for the row to exist -- a failure here shouldn't block the upload itself.
        # Not fed subtopic yet -- see build_diagram_embedding_text's comment. Hindi fields
        # deliberately not fed in either -- English-only match-text, same reasoning.
        embedding = await get_embedding(build_diagram_embedding_text(name, description, chapter))
    except Exception:
        embedding = None
    payload = {
        "subject": body.subject,
        "class": body.class_,
        "chapter": chapter,
        "subtopic": subtopic,
        "subtopic_hi": subtopic_hi,
        "name": name,
        "description": description,
        "name_hi": name_hi,
        "description_hi": description_hi,
        "image_url": body.image_url,
        "embedding": embedding
    }
    try:
        response = await async_client.post(
            f"{SUPABASE_URL}/rest/v1/diagrams",
            headers={**ADMIN_HEADERS, "Content-Type": "application/json", "Prefer": "return=representation"},
            json=payload
        )
        if response.status_code >= 400:
            return {"error": response.text}
        created = response.json()
        return {"diagram": created[0] if created else None}
    except Exception as e:
        return {"error": str(e)}

@app.get("/admin/diagrams-list")
async def admin_diagrams_list(_: None = Depends(verify_admin)):
    try:
        response = await async_client.get(
            f"{SUPABASE_URL}/rest/v1/diagrams",
            headers=ADMIN_HEADERS,
            params={
                "select": "id,subject,class,chapter,subtopic,subtopic_hi,name,description,name_hi,description_hi,image_url,reviewed,created_at",
                "order": "created_at.desc",
                "limit": 1000
            }
        )
        if response.status_code >= 400:
            return {"error": response.text}
        return {"diagrams": response.json()}
    except Exception as e:
        return {"error": str(e)}

@app.patch("/admin/diagram-update/{diagram_id}")
async def admin_diagram_update(diagram_id: int, body: DiagramUpdate, _: None = Depends(verify_admin)):
    if body.subject is not None and body.subject not in ("Biology", "Physics", "Chemistry"):
        return {"error": "Invalid subject"}
    update_fields = {}
    if body.subject is not None:
        update_fields["subject"] = body.subject
    if body.class_ is not None:
        update_fields["class"] = body.class_
    if body.chapter is not None:
        update_fields["chapter"] = body.chapter
    if body.name is not None:
        update_fields["name"] = body.name
    # description/subtopic/Hindi fields can legitimately be cleared to null (removing a
    # caption/tag/translation), so distinguish "not sent" from "sent as null" the same way
    # /admin/pyq-update does for diagram fields.
    if "description" in body.model_fields_set:
        update_fields["description"] = body.description
    if "subtopic" in body.model_fields_set:
        update_fields["subtopic"] = body.subtopic
    if "subtopic_hi" in body.model_fields_set:
        update_fields["subtopic_hi"] = body.subtopic_hi
    if "name_hi" in body.model_fields_set:
        update_fields["name_hi"] = body.name_hi
    if "description_hi" in body.model_fields_set:
        update_fields["description_hi"] = body.description_hi
    if body.reviewed is not None:
        update_fields["reviewed"] = body.reviewed
    if not update_fields:
        return {"error": "No fields to update"}

    # Keep the embedding in sync whenever the text it's derived from changes -- otherwise an
    # edited diagram would silently keep matching doubts on its OLD name/description/chapter
    # forever. Best-effort: an embedding refresh failure shouldn't block the metadata edit.
    # subtopic is deliberately NOT included here yet -- see build_diagram_embedding_text's
    # comment on whether it should feed the embedding at all.
    if any(k in update_fields for k in ("name", "description", "chapter")):
        try:
            current_resp = await async_client.get(
                f"{SUPABASE_URL}/rest/v1/diagrams",
                headers=ADMIN_HEADERS,
                params={"id": f"eq.{diagram_id}", "select": "name,description,chapter"}
            )
            current_rows = current_resp.json() if current_resp.status_code < 400 else []
            if current_rows:
                current = current_rows[0]
                new_name = update_fields.get("name", current["name"])
                new_description = update_fields["description"] if "description" in update_fields else current.get("description")
                new_chapter = update_fields.get("chapter", current["chapter"])
                update_fields["embedding"] = await get_embedding(build_diagram_embedding_text(new_name, new_description, new_chapter))
        except Exception:
            pass

    try:
        response = await async_client.patch(
            f"{SUPABASE_URL}/rest/v1/diagrams",
            headers={**ADMIN_HEADERS, "Content-Type": "application/json", "Prefer": "return=representation"},
            params={"id": f"eq.{diagram_id}", "select": "id,subject,class,chapter,subtopic,subtopic_hi,name,description,name_hi,description_hi,image_url,reviewed,created_at"},
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

@app.delete("/admin/diagram-delete/{diagram_id}")
async def admin_diagram_delete(diagram_id: int, _: None = Depends(verify_admin)):
    try:
        row_resp = await async_client.get(
            f"{SUPABASE_URL}/rest/v1/diagrams",
            headers=ADMIN_HEADERS,
            params={"id": f"eq.{diagram_id}", "select": "image_url"}
        )
        if row_resp.status_code >= 400:
            return {"error": row_resp.text}
        rows = row_resp.json()
        if not rows:
            return {"error": "Row not found"}
        image_url = rows[0].get("image_url") or ""
        prefix = f"{SUPABASE_URL}/storage/v1/object/public/{DIAGRAMS_BUCKET}/"
        if image_url.startswith(prefix):
            storage_path = image_url[len(prefix):]
            # Best-effort: a storage delete failure (e.g. file already gone) shouldn't block
            # removing the DB row, since an orphaned storage object is harmless either way.
            await async_client.delete(
                f"{SUPABASE_URL}/storage/v1/object/{DIAGRAMS_BUCKET}/{storage_path}",
                headers=ADMIN_HEADERS
            )
        del_resp = await async_client.delete(
            f"{SUPABASE_URL}/rest/v1/diagrams",
            headers={**ADMIN_HEADERS, "Prefer": "return=representation"},
            params={"id": f"eq.{diagram_id}"}
        )
        if del_resp.status_code >= 400:
            return {"error": del_resp.text}
        deleted = del_resp.json()
        if not deleted:
            return {"error": "Row not found"}
        return {"success": True}
    except Exception as e:
        return {"error": str(e)}

# One-off/ongoing backfill for diagram rows created before embedding generation existed (or any
# future edge case, e.g. a manual DB import) -- diagrams-create and diagram-update both keep the
# embedding in sync going forward, so this only ever has work to do for rows that predate them.
@app.post("/admin/diagrams-backfill-embeddings")
async def admin_diagrams_backfill_embeddings(_: None = Depends(verify_admin)):
    try:
        resp = await async_client.get(
            f"{SUPABASE_URL}/rest/v1/diagrams",
            headers=ADMIN_HEADERS,
            params={"select": "id,name,description,chapter", "embedding": "is.null"}
        )
        if resp.status_code >= 400:
            return {"error": resp.text}
        rows = resp.json()
        updated = 0
        for row in rows:
            embed_text = build_diagram_embedding_text(row["name"], row.get("description"), row["chapter"])
            embedding = await get_embedding(embed_text)
            patch_resp = await async_client.patch(
                f"{SUPABASE_URL}/rest/v1/diagrams",
                headers={**ADMIN_HEADERS, "Content-Type": "application/json"},
                params={"id": f"eq.{row['id']}"},
                json={"embedding": embedding}
            )
            if patch_resp.status_code < 400:
                updated += 1
        return {"updated": updated, "total_missing": len(rows)}
    except Exception as e:
        return {"error": str(e)}

# One-off/ongoing backfill, same role as diagrams-backfill-embeddings above -- covers rows from
# before embedding-on-create existed (the 2026-08 gap this whole feature was built to close) and
# doubles as the recovery path for admin_pyq_bulk_create's background embedding attempts below
# that failed (a failed attempt deliberately leaves embedding NULL rather than writing anything
# partial, so it's indistinguishable from -- and automatically caught by -- this same query).
# Paginated (batch_size below) since the real backlog here can be thousands of rows, not the few
# dozen diagrams-backfill-embeddings was written for -- a single unbounded fetch would either hit
# PostgREST's default row cap or time out the request well before finishing.
@app.post("/admin/pyq-backfill-embeddings")
async def admin_pyq_backfill_embeddings(batch_size: int = 200, _: None = Depends(verify_admin)):
    try:
        resp = await async_client.get(
            f"{SUPABASE_URL}/rest/v1/pyq",
            headers=ADMIN_HEADERS,
            params={"select": "id,question,chapter,option_a,option_b,option_c,option_d",
                    "embedding": "is.null", "order": "id.asc", "limit": batch_size}
        )
        if resp.status_code >= 400:
            return {"error": resp.text}
        rows = resp.json()
        updated = 0
        for row in rows:
            embed_text = build_pyq_embedding_text(
                row["question"], row.get("chapter"),
                row.get("option_a"), row.get("option_b"), row.get("option_c"), row.get("option_d")
            )
            embedding = await get_embedding(embed_text)
            patch_resp = await async_client.patch(
                f"{SUPABASE_URL}/rest/v1/pyq",
                headers={**ADMIN_HEADERS, "Content-Type": "application/json"},
                params={"id": f"eq.{row['id']}"},
                json={"embedding": embedding}
            )
            if patch_resp.status_code < 400:
                updated += 1
        return {"updated": updated, "batch_size": len(rows), "done": len(rows) < batch_size}
    except Exception as e:
        return {"error": str(e)}

# Fire-and-forget embedding generation for one freshly-created pyq row -- scheduled via
# asyncio.create_task() from admin_pyq_bulk_create below rather than awaited, same pattern
# get_embedding() already uses for its own cache write, so the admin's save request returns as
# soon as the row is in the DB instead of waiting on an extra per-question OpenAI round-trip.
# On any failure this deliberately leaves embedding NULL rather than writing something partial --
# a NULL embedding is exactly what /admin/pyq-backfill-embeddings already selects for, so a
# failed row here needs no separate failure-tracking table, it's just picked up by the next
# backfill run. Still printed clearly (PYQ EMBEDDING FAILED) so a failure is visible in a Railway
# log scan rather than only discoverable by noticing it during a backfill.
async def _embed_pyq_row_background(row_id, question, chapter, option_a, option_b, option_c, option_d):
    try:
        embed_text = build_pyq_embedding_text(question, chapter, option_a, option_b, option_c, option_d)
        embedding = await get_embedding(embed_text)
        patch_resp = await async_client.patch(
            f"{SUPABASE_URL}/rest/v1/pyq",
            headers={**ADMIN_HEADERS, "Content-Type": "application/json"},
            params={"id": f"eq.{row_id}"},
            json={"embedding": embedding}
        )
        if patch_resp.status_code >= 400:
            print(f"PYQ EMBEDDING FAILED (write) id={row_id}: {patch_resp.status_code} {patch_resp.text[:200]}", flush=True)
    except Exception as e:
        print(f"PYQ EMBEDDING FAILED (generation) id={row_id}: {e}", flush=True)

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
        # Undo-delete restore passes these through to make the restored row match what was
        # actually deleted; normal creation never sets them, so this stays a no-op for every
        # other caller (PDF review, scanned-paste parser).
        if q.id:
            item["id"] = q.id
        if q.reviewed is not None:
            item["reviewed"] = q.reviewed
        if q.difficulty is not None:
            item["difficulty"] = q.difficulty
        payload.append(item)
    try:
        response = http_requests.post(
            f"{SUPABASE_URL}/rest/v1/pyq",
            headers={**ADMIN_HEADERS, "Content-Type": "application/json", "Prefer": "return=representation"},
            json=payload
        )
        if response.status_code >= 400:
            return {"error": response.text}
        created = response.json()
        for row in created:
            asyncio.create_task(_embed_pyq_row_background(
                row["id"], row.get("question"), row.get("chapter"),
                row.get("option_a"), row.get("option_b"), row.get("option_c"), row.get("option_d")
            ))
        return {"created": created}
    except Exception as e:
        return {"error": str(e)}

@app.post("/admin/pdf-mark-processed")
def admin_pdf_mark_processed(req: PdfMarkProcessedRequest, _: None = Depends(verify_admin)):
    if req.status not in ("completed", "failed"):
        return {"error": "Invalid status"}
    _update_processed_pdf_row(req.id, req.status, questions_extracted=req.questions_extracted, error_message=req.error_message)
    return {"success": True}

@app.get("/admin/pdf-check-status")
def admin_pdf_check_status(filename: str, _: None = Depends(verify_admin)):
    # Most recent processed_pdfs row for this filename, if any -- lets the frontend warn
    # before reprocessing a PDF that's already gone through the pipeline.
    try:
        response = http_requests.get(
            f"{SUPABASE_URL}/rest/v1/processed_pdfs",
            headers=ADMIN_HEADERS,
            params={
                "filename": f"eq.{filename}",
                "select": "id,status,questions_extracted,upload_date",
                "order": "upload_date.desc",
                "limit": 1
            }
        )
        if response.status_code >= 400:
            return {"error": response.text}
        rows = response.json()
        return {"previous": rows[0] if rows else None}
    except Exception as e:
        return {"error": str(e)}

@app.get("/admin/processed-pdfs")
def admin_processed_pdfs(_: None = Depends(verify_admin)):
    try:
        response = http_requests.get(
            f"{SUPABASE_URL}/rest/v1/processed_pdfs",
            headers=ADMIN_HEADERS,
            params={
                "select": "id,filename,subject,status,questions_extracted,error_message,upload_date,processed_at",
                "order": "upload_date.desc",
                "limit": 500
            }
        )
        if response.status_code >= 400:
            return {"error": response.text}
        return {"pdfs": response.json()}
    except Exception as e:
        return {"error": str(e)}

# ---------- Admin: official mock test upload pipeline (admin-mocktest-upload.html) ----------

def _create_mock_test_row(title: str, filename: str):
    # Tracking is best-effort and must never block the actual scan, same reasoning as
    # _create_processed_pdf_row above.
    try:
        response = http_requests.post(
            f"{SUPABASE_URL}/rest/v1/mock_tests",
            headers={**ADMIN_HEADERS, "Content-Type": "application/json", "Prefer": "return=representation"},
            json={"title": title, "filename": filename, "status": "processing"}
        )
        if response.status_code < 400:
            rows = response.json()
            if rows:
                return rows[0]["id"]
    except Exception:
        pass
    return None

def _update_mock_test_row(row_id, status: str, questions_extracted=None, physics_count=None,
                           chemistry_count=None, biology_count=None, error_message: str = None):
    if row_id is None:
        return
    body = {"status": status}
    if questions_extracted is not None:
        body["questions_extracted"] = questions_extracted
    if physics_count is not None:
        body["physics_count"] = physics_count
    if chemistry_count is not None:
        body["chemistry_count"] = chemistry_count
    if biology_count is not None:
        body["biology_count"] = biology_count
    if error_message is not None:
        body["error_message"] = error_message[:2000]
    try:
        http_requests.patch(
            f"{SUPABASE_URL}/rest/v1/mock_tests",
            headers={**ADMIN_HEADERS, "Content-Type": "application/json"},
            params={"id": f"eq.{row_id}"},
            json=body
        )
    except Exception:
        pass

@app.post("/admin/mock-test-create")
def admin_mock_test_create(req: MockTestCreateRequest, _: None = Depends(verify_admin)):
    # For admin-scanned-paste-parser.html -- a mock_tests row created directly from a title,
    # with no PDF/scan step. Reuses the exact same _create_mock_test_row() helper /admin/mock-
    # test-scan uses, just without the scanning that normally precedes it.
    if not req.title or not req.title.strip():
        return {"error": "Title is required"}
    mock_test_id = _create_mock_test_row(req.title.strip(), None)
    if mock_test_id is None:
        return {"error": "Could not create mock test row"}
    return {"mock_test_id": mock_test_id}

@app.post("/admin/mock-test-scan")
def admin_mock_test_scan(req: MockTestScanRequest, _: None = Depends(verify_admin), __: None = Depends(rate_limiter(5, 300))):
    # Sync (not async def), same reasoning as /admin/scan-pdf: this call is ~3x the vision
    # work of a single-subject scan, so it's rate-limited lower (5/300s vs 10/300s there).
    missing = [s for s in ("Physics", "Chemistry", "Biology") if s not in req.ranges]
    if missing:
        return {"error": f"Missing page range(s) for: {', '.join(missing)}"}
    try:
        pdf_bytes = base64.b64decode(req.data)
    except Exception:
        return {"error": "Could not decode PDF data"}

    mock_test_id = _create_mock_test_row(req.title, req.filename)
    ranges = {s: (req.ranges[s].start, req.ranges[s].end) for s in req.ranges}
    try:
        result = scan_mock_test_pdf(pdf_bytes, ranges)
    except Exception as e:
        _update_mock_test_row(mock_test_id, "failed", error_message=str(e))
        return {"error": str(e)}
    if result.get("errors"):
        import json
        _update_mock_test_row(mock_test_id, "failed", error_message=json.dumps(result["errors"]))
    result["mock_test_id"] = mock_test_id
    return result

@app.post("/admin/mock-test-bulk-create")
async def admin_mock_test_bulk_create(body: MockTestBulkCreate, _: None = Depends(verify_admin)):
    if not body.questions:
        return {"error": "No questions provided"}
    if any(q.subject not in ("Biology", "Physics", "Chemistry") for q in body.questions):
        return {"error": "Invalid subject"}
    payload = [
        {
            "mock_test_id": q.mock_test_id,
            "question_order": q.question_order,
            "subject": q.subject,
            "chapter": q.chapter,
            "class": q.class_,
            "question": q.question,
            "option_a": q.option_a,
            "option_b": q.option_b,
            "option_c": q.option_c,
            "option_d": q.option_d,
            "correct_answer": q.correct_answer,
            "question_type": q.question_type,
            "year": q.year,
            "has_diagram": q.has_diagram,
            "diagram_url": q.diagram_url,
            "option_a_diagram_url": q.option_a_diagram_url,
            "option_b_diagram_url": q.option_b_diagram_url,
            "option_c_diagram_url": q.option_c_diagram_url,
            "option_d_diagram_url": q.option_d_diagram_url
        }
        for q in body.questions
    ]
    try:
        response = http_requests.post(
            f"{SUPABASE_URL}/rest/v1/mock_test_questions",
            headers={**ADMIN_HEADERS, "Content-Type": "application/json", "Prefer": "return=representation"},
            json=payload
        )
        if response.status_code >= 400:
            return {"error": response.text}
        return {"created": response.json()}
    except Exception as e:
        return {"error": str(e)}

@app.patch("/admin/mock-test-question-update/{question_id}")
async def admin_mock_test_question_update(question_id: int, body: MockTestQuestionUpdate, _: None = Depends(verify_admin)):
    # Mirrors /admin/pyq-update/{pyq_id}'s exact shape -- lets an already-saved question (from
    # admin-scanned-paste-parser.html or elsewhere) be edited and re-saved as an UPDATE instead
    # of a duplicate INSERT.
    if body.subject is not None and body.subject not in ("Biology", "Physics", "Chemistry"):
        return {"error": "Invalid subject"}
    update_fields = {}
    if body.subject is not None:
        update_fields["subject"] = body.subject
    if body.chapter is not None:
        update_fields["chapter"] = body.chapter
    if body.class_ is not None:
        update_fields["class"] = body.class_
    if body.question is not None:
        update_fields["question"] = body.question
    if body.option_a is not None:
        update_fields["option_a"] = body.option_a
    if body.option_b is not None:
        update_fields["option_b"] = body.option_b
    if body.option_c is not None:
        update_fields["option_c"] = body.option_c
    if body.option_d is not None:
        update_fields["option_d"] = body.option_d
    if body.correct_answer is not None:
        update_fields["correct_answer"] = body.correct_answer
    if body.year is not None:
        update_fields["year"] = body.year
    if body.has_diagram is not None:
        update_fields["has_diagram"] = body.has_diagram
    # Same distinguish-"not sent"-from-"sent as null" reasoning as /admin/pyq-update/{pyq_id}:
    # removing a photo means explicitly clearing the URL, and `is not None` would ignore that.
    for diagram_field in ("diagram_url", "option_a_diagram_url", "option_b_diagram_url",
                           "option_c_diagram_url", "option_d_diagram_url"):
        if diagram_field in body.model_fields_set:
            update_fields[diagram_field] = getattr(body, diagram_field)
    if not update_fields:
        return {"error": "No fields to update"}
    try:
        response = await async_client.patch(
            f"{SUPABASE_URL}/rest/v1/mock_test_questions",
            headers={**ADMIN_HEADERS, "Content-Type": "application/json", "Prefer": "return=representation"},
            params={"id": f"eq.{question_id}", "select": "id"},
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

@app.post("/admin/mock-test-mark-processed")
def admin_mock_test_mark_processed(req: MockTestMarkProcessedRequest, _: None = Depends(verify_admin)):
    if req.status not in ("completed", "failed"):
        return {"error": "Invalid status"}
    _update_mock_test_row(
        req.id, req.status, questions_extracted=req.questions_extracted,
        physics_count=req.physics_count, chemistry_count=req.chemistry_count,
        biology_count=req.biology_count, error_message=req.error_message
    )
    return {"success": True}

@app.post("/admin/mock-test-publish")
def admin_mock_test_publish(req: MockTestPublishRequest, _: None = Depends(verify_admin)):
    try:
        response = http_requests.patch(
            f"{SUPABASE_URL}/rest/v1/mock_tests",
            headers={**ADMIN_HEADERS, "Content-Type": "application/json"},
            params={"id": f"eq.{req.id}"},
            json={"is_published": True, "published_at": datetime.now(timezone.utc).isoformat()}
        )
        if response.status_code >= 400:
            return {"error": response.text}
        return {"success": True}
    except Exception as e:
        return {"error": str(e)}

@app.get("/admin/mock-test-check-status")
def admin_mock_test_check_status(filename: str, _: None = Depends(verify_admin)):
    try:
        response = http_requests.get(
            f"{SUPABASE_URL}/rest/v1/mock_tests",
            headers=ADMIN_HEADERS,
            params={
                "filename": f"eq.{filename}",
                "select": "id,title,status,questions_extracted,created_at",
                "order": "created_at.desc",
                "limit": 1
            }
        )
        if response.status_code >= 400:
            return {"error": response.text}
        rows = response.json()
        return {"previous": rows[0] if rows else None}
    except Exception as e:
        return {"error": str(e)}

@app.get("/admin/mock-tests")
def admin_mock_tests_list(_: None = Depends(verify_admin)):
    try:
        response = http_requests.get(
            f"{SUPABASE_URL}/rest/v1/mock_tests",
            headers=ADMIN_HEADERS,
            params={
                "select": "id,title,filename,status,is_published,questions_extracted,physics_count,chemistry_count,biology_count,error_message,created_at,published_at",
                "order": "created_at.desc",
                "limit": 200
            }
        )
        if response.status_code >= 400:
            return {"error": response.text}
        return {"tests": response.json()}
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

@app.get("/admin/question-reports")
async def admin_question_reports(resolved: str = "false", _: None = Depends(verify_admin)):
    try:
        # Same 1000-row PostgREST cap as /admin/pyq-duplicates -- page through until a page
        # comes back short, otherwise a busy report queue would silently truncate.
        all_reports = []
        page_size = 1000
        offset = 0
        resolved_filter = "eq.true" if resolved == "true" else "eq.false"
        while True:
            response = await async_client.get(
                f"{SUPABASE_URL}/rest/v1/question_reports",
                headers=ADMIN_HEADERS,
                params={
                    "resolved": resolved_filter,
                    "select": "id,pyq_id,user_id,reason,optional_note,created_at",
                    "order": "created_at.desc",
                    "limit": page_size,
                    "offset": offset
                }
            )
            page = response.json()
            all_reports.extend(page)
            if len(page) < page_size:
                break
            offset += page_size

        # Grouped by pyq_id in Python -- PostgREST has no GROUP BY, same reasoning as
        # /admin/pdf-upload-history's per-filename aggregation.
        groups_by_pyq = {}
        for r in all_reports:
            pid = r["pyq_id"]
            g = groups_by_pyq.setdefault(pid, {
                "pyq_id": pid, "count": 0, "reasons": {}, "notes": [], "latest_report_at": r["created_at"]
            })
            g["count"] += 1
            g["reasons"][r["reason"]] = g["reasons"].get(r["reason"], 0) + 1
            if r["optional_note"]:
                g["notes"].append(r["optional_note"])
            if r["created_at"] > g["latest_report_at"]:
                g["latest_report_at"] = r["created_at"]

        pyq_ids = list(groups_by_pyq.keys())
        questions_by_id = {}
        for i in range(0, len(pyq_ids), 200):
            chunk = pyq_ids[i:i + 200]
            resp = await async_client.get(
                f"{SUPABASE_URL}/rest/v1/pyq", headers=ADMIN_HEADERS,
                # Same full field set as /admin/pyq-search's full=true -- admin-pyq-preview.html
                # renders and edits these rows in place (options, diagrams, year, source tag),
                # not just a text snippet, so the partial select used to leave those blank.
                params={"id": f"in.({','.join(chunk)})",
                        "select": "id,subject,chapter,question,option_a,option_b,option_c,option_d,"
                                   "correct_answer,is_active,year,source_tag,class,has_diagram,"
                                   "diagram_url,option_a_diagram_url,option_b_diagram_url,"
                                   "option_c_diagram_url,option_d_diagram_url,reviewed,created_at"}
            )
            for row in resp.json():
                questions_by_id[row["id"]] = row

        groups = []
        for pid, g in groups_by_pyq.items():
            g["question"] = questions_by_id.get(pid)  # None if the question was since deleted
            groups.append(g)
        groups.sort(key=lambda g: g["count"], reverse=True)
        return {"groups": groups}
    except Exception as e:
        return {"error": str(e)}

@app.patch("/admin/question-reports/{pyq_id}/resolve")
async def admin_resolve_question_reports(pyq_id: str, _: None = Depends(verify_admin)):
    try:
        response = await async_client.patch(
            f"{SUPABASE_URL}/rest/v1/question_reports",
            headers={**ADMIN_HEADERS, "Content-Type": "application/json", "Prefer": "return=representation"},
            params={"pyq_id": f"eq.{pyq_id}", "resolved": "eq.false"},
            json={"resolved": True}
        )
        if response.status_code >= 400:
            return {"error": response.text}
        return {"success": True, "resolved_count": len(response.json())}
    except Exception as e:
        return {"error": str(e)}

# Mirrors /admin/question-reports exactly, grouped by diagram_id instead of pyq_id and joined
# against the diagrams table instead of pyq -- surfaces in admin-pyq-preview.html's existing
# Diagram Review section via a new "Reported" status filter, not a separate top-level view.
@app.get("/admin/diagram-reports")
async def admin_diagram_reports(resolved: str = "false", _: None = Depends(verify_admin)):
    try:
        all_reports = []
        page_size = 1000
        offset = 0
        resolved_filter = "eq.true" if resolved == "true" else "eq.false"
        while True:
            response = await async_client.get(
                f"{SUPABASE_URL}/rest/v1/diagram_reports",
                headers=ADMIN_HEADERS,
                params={
                    "resolved": resolved_filter,
                    "select": "id,diagram_id,user_id,source,reason,optional_note,created_at",
                    "order": "created_at.desc",
                    "limit": page_size,
                    "offset": offset
                }
            )
            page = response.json()
            all_reports.extend(page)
            if len(page) < page_size:
                break
            offset += page_size

        groups_by_diagram = {}
        for r in all_reports:
            did = r["diagram_id"]
            g = groups_by_diagram.setdefault(did, {
                "diagram_id": did, "count": 0, "reasons": {}, "sources": {}, "notes": [], "latest_report_at": r["created_at"]
            })
            g["count"] += 1
            g["reasons"][r["reason"]] = g["reasons"].get(r["reason"], 0) + 1
            g["sources"][r["source"]] = g["sources"].get(r["source"], 0) + 1
            if r["optional_note"]:
                g["notes"].append(r["optional_note"])
            if r["created_at"] > g["latest_report_at"]:
                g["latest_report_at"] = r["created_at"]

        diagram_ids = list(groups_by_diagram.keys())
        diagrams_by_id = {}
        for i in range(0, len(diagram_ids), 200):
            chunk = diagram_ids[i:i + 200]
            resp = await async_client.get(
                f"{SUPABASE_URL}/rest/v1/diagrams", headers=ADMIN_HEADERS,
                params={"id": f"in.({','.join(str(d) for d in chunk)})",
                        "select": "id,subject,class,chapter,name,description,name_hi,description_hi,image_url,reviewed,created_at"}
            )
            for row in resp.json():
                diagrams_by_id[row["id"]] = row

        groups = []
        for did, g in groups_by_diagram.items():
            g["diagram"] = diagrams_by_id.get(did)  # None if the diagram was since deleted
            groups.append(g)
        groups.sort(key=lambda g: g["count"], reverse=True)
        return {"groups": groups}
    except Exception as e:
        return {"error": str(e)}

@app.patch("/admin/diagram-reports/{diagram_id}/resolve")
async def admin_resolve_diagram_reports(diagram_id: int, _: None = Depends(verify_admin)):
    try:
        response = await async_client.patch(
            f"{SUPABASE_URL}/rest/v1/diagram_reports",
            headers={**ADMIN_HEADERS, "Content-Type": "application/json", "Prefer": "return=representation"},
            params={"diagram_id": f"eq.{diagram_id}", "resolved": "eq.false"},
            json={"resolved": True}
        )
        if response.status_code >= 400:
            return {"error": response.text}
        return {"success": True, "resolved_count": len(response.json())}
    except Exception as e:
        return {"error": str(e)}

if __name__ == "__main__":
    import uvicorn
    import os
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port)