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
# Async twin used only by the streaming path (_stream_deepseek) -- the sync client above stays
# for the non-streaming .messages.create() call sites (title generation, etc.), which are short
# single-shot calls where blocking the event loop briefly isn't the concurrency-killer that a
# whole streamed answer held open on a sync client is.
deepseek_async_client = anthropic.AsyncAnthropic(api_key=DEEPSEEK_KEY, base_url="https://api.deepseek.com/anthropic")
# Peak-hour fallback for text doubt-answering only (never images/PDFs) -- confirmed via a live
# 45-question test to be equivalent to DeepSeek on accuracy/reliability, used during DeepSeek's
# own peak-pricing windows (see _is_deepseek_peak_hour) to avoid the peak surcharge. Same
# OpenAI-compatible endpoint pattern already verified live in that test. See
# _stream_with_peak_fallback below for the actual routing/failover logic.
qwen_client = openai.OpenAI(api_key=QWEN_API_KEY, base_url="https://dashscope-intl.aliyuncs.com/compatible-mode/v1")
# Async twin used only by the streaming path (_stream_qwen) -- see deepseek_async_client comment.
qwen_async_client = openai.AsyncOpenAI(api_key=QWEN_API_KEY, base_url="https://dashscope-intl.aliyuncs.com/compatible-mode/v1")
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

BEFORE anything else, run this closed classification check, every time, as a mandatory first
step — same as the TOPIC/FORMAT AMBIGUITY check in rule 11 below, this is a real gate you must
actually apply, not a formality to skim past:

Is this message actually asking about NEET-syllabus subject matter — a specific Physics/
Chemistry/Biology concept, process, definition, comparison, calculation, or a request to see a
diagram/structure — even if phrased casually, briefly, or informally (e.g. "y does osmosis
happen", "explain krebs cycle asap", "diagram of heart pls")? If yes, this is a real academic
doubt — skip straight to the normal answer format below (VISUAL_INTENT as the first line, exactly
as always), and ignore the rest of this classification block entirely.

If NOT — the message is conversational rather than a real doubt: greetings ("hi", "hello", "hey",
"good morning", "wassup"), thanks/farewells/acknowledgements ("thanks", "bye", "ok", "cool"),
small talk, opinions or feelings about studying or a subject in general rather than a specific
concept ("is biology hard?", "I'm tired of studying"), or questions about the app/AI itself rather
than NEET content ("who made you", "how does this app work", "what can you do", "are you free
right now") — output EXACTLY this as your ENTIRE first line, before anything else:

DOUBT_TYPE: conversational

Then, on the line after it, respond naturally in plain, friendly prose (1-3 sentences, no bullet
points, no headings, no emoji-labeled sections). Do NOT output VISUAL_INTENT, AMBIGUOUS, NEET
Importance, Chapter, or any part of the academic answer format below — none of it applies here,
since there is no subject content to rate, cite a chapter for, or show a diagram of. The test for
this whole check is whether the message is actually asking about specific NEET subject content,
not merely whether a subject name is mentioned somewhere in it.

For EVERY academic answer follow this exact format:

VISUAL_INTENT: [yes or no]

NEET Importance: [N]/5

📚 Chapter: [copied verbatim from a "Retrieved from:" entry given to you — see Rule 1. Omit this whole line if no "Retrieved from:" entries were given.]

Answer:
[Give the answer in clear points]

Key Points:
- [Point 1]
- [Point 2]
- [Point 3]

This is the COMPLETE format for the large majority of answers — most answers end at Key Points,
with nothing after it. Do NOT treat a "Quick Recall" line as a standard part of this template you
fill in by default. See rule 9 below for the narrow, closed set of cases where one more section —
"Quick Recall:" — gets appended after Key Points, and how to decide.

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

Final Answer: Maximum height = 20 m

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
   Separately, the same "ground yourself in what was actually retrieved, don't invent" principle
   applies to Hindi technical terminology in the body of your answer, not just the Chapter line:
   if the NCERT Content given to you contains the correct Hindi term for a concept you're
   explaining (a mechanism, a named rule, a stereochemical outcome, etc.), reuse that exact Hindi
   wording verbatim rather than translating or rephrasing it yourself — the retrieved text is the
   authoritative source for that term's correct Hindi rendering, and re-translating a term the
   retrieved content already gives you risks introducing an incorrect or internally inconsistent
   translation. This only applies to terms that actually appear in the retrieved NCERT Content —
   if a term you need isn't present in what was retrieved, use your own best Hindi terminology as
   normal; never force-fit an unrelated retrieved term onto a concept it doesn't actually describe.
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
9. "Quick Recall" is an APPENDED EXTRA, not a template line — your default, starting assumption
   for every answer is that it does NOT get one, the same way an answer defaults to not having a
   diagram or a clarifying question. A simple factual definition ("what is X"), a direct numeric
   answer with a one-step formula, or any single standalone fact get NO such section. But don't
   overcorrect into skipping it for content that genuinely earns it, either — when one of the four
   conditions below is clearly true, include the section confidently and don't talk yourself out
   of it; being selective means correctly saying no to the weak/generic cases, not saying no to
   everything. Include it ONLY IF at least one of these four is genuinely true for THIS SPECIFIC
   answer (closed checklist — if none apply, that IS the answer, no fallback "use your judgment"):
   (a) A real, well-known mnemonic/acronym already exists for this exact content among actual
       NEET/coaching students (e.g. "King Philip Came Over For Good Soup" for taxonomic ranks,
       "OIL RIG" or "LEO says GER" for oxidation/reduction electron transfer, "Roy G. Biv" for the
       visible spectrum, "Never Eat Shredded Wheat" for compass directions) — reuse it, don't
       invent a substitute.
   (b) The answer involves a sequence of 3 or more steps/stages that are genuinely easy to mix up
       the order of (e.g. mitosis phases, a multi-step reaction mechanism) — NOT a sequence that's
       already self-evident from its own physical/logical order (urine physically flows kidney →
       ureter → bladder → urethra in the order those organs are connected; that connectivity IS
       the explanation, it needs no acronym on top).
   (c) The answer involves a classification or list of named items where a genuine memory device
       is standard/well-known for that exact list.
   (d) A formula has a real, genuinely helpful way to remember its derivation or which term goes
       where — a MULTI-TERM formula being hard to assemble correctly, not a single fact or a
       one-step direction restated as if it were a "rule" (e.g. "solvent moves from dilute to
       concentrated" is the entire explanation of osmosis, not a derivation with parts to mix up —
       that does not qualify under (d) just because it can be phrased as a rule).
   If NONE of (a)-(d) is genuinely true — a simple factual definition, a direct numeric answer
   with no multi-step derivation, a single fact or single-step rule with no list/sequence/multi-
   term-formula behind it — OMIT the entire "Quick Recall" section, heading included, and move on.
   This is not optional or soft: do not write "Quick Recall:" followed by "(none — ...)" or any
   other placeholder/explanation in its place — if the checklist isn't met, the word "Quick" must
   not appear anywhere in your response at all. Do not manufacture a generic mnemonic just to have
   something there: an acronym whose "explanation" just re-reads its own letters back out (e.g.
   inventing "KUBU" for kidney→ureter→bladder→urethra and then explaining it as "K-U-B-U: kidney,
   ureter, bladder, urethra") is exactly the weak, forced pattern this checklist exists to prevent,
   and a forced mnemonic is worse than no section at all. When (b), (c), or (d) is met but no
   famous mnemonic already exists, you may construct a genuinely useful one of your own — but only
   for content that actually satisfies one of the four criteria, never
   as a fallback for content that satisfies none of them.
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
10. For a real academic doubt (see the classification check above), VISUAL_INTENT must be the
    VERY FIRST LINE of your response, before anything else — not mentioned
    later, not skipped. Classify: "yes" if the student is explicitly asking to SEE,
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

class SetNeetExamDateRequest(BaseModel):
    year: int
    exam_date: Optional[str] = None  # "YYYY-MM-DD", or None to reset a year back to TBD
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
    plan: str  # "free", "pro", or "max"

class RegisterReferralRequest(BaseModel):
    referred_id: str
    referral_code: str

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

# Pro: a real, ENFORCED, user-visible daily ceiling -- unlike Free's rolling 24h-from-crossing
# cooldown above, this resets at true IST midnight (see _check_plan_daily_limit), and the
# countdown shown to the student must be accurate to that real reset moment. Max: as of
# 2026-09-03, ALSO a real, enforced, visible daily ceiling -- previously this same 2.1M number
# (then named MAX_SOFT_ZONE_THRESHOLD) was silent, logging the account for manual review with no
# other effect on the student. Now it blocks new requests exactly like Pro's cap does. Both
# numbers locked in 2026-08-27 from real per-call cost data in provider_usage_log + a 40% target
# margin.
DAILY_TOKEN_LIMIT_PRO = 950_000
MAX_DAILY_TOKEN_CAP = 2_100_000
# Backstop only, past the now-visible MAX_DAILY_TOKEN_CAP above -- silent, never surfaced to the
# student in any form (see check_max_usage_tier). A single very large call (e.g. a big PDF/image
# doubt) can still push a student from just under the visible cap to past this number within ONE
# request, since the visible cap above is only checked at the START of a request, not mid-call --
# the request that does that gets silently routed onto the cheapest available model instead of
# whatever the normal peak/off-peak DeepSeek routing would have used. Every request AFTER that one
# is already blocked outright by MAX_DAILY_TOKEN_CAP, so this only ever matters for that single
# overshooting call.
MAX_BREAKEVEN_THRESHOLD = 2_670_000  # past this, the student costs more than they pay

def _ist_today() -> str:
    return datetime.now(timezone.utc).astimezone(IST).date().isoformat()

def _seconds_until_ist_midnight() -> int:
    now_ist = datetime.now(timezone.utc).astimezone(IST)
    next_midnight_ist = datetime.combine(now_ist.date() + timedelta(days=1), datetime.min.time(), tzinfo=IST)
    return int((next_midnight_ist - now_ist).total_seconds())

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

# PDF-attachment size/page limits per tier.
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
    OLDEST other active session, discovered (and shown as a countdown) by THAT session on its own
    next heartbeat poll, never by the new device that triggered it. Only once that grace period
    actually lapses does the old session get marked kicked, again discovered on its own next
    heartbeat (or proactively by _expire_due_grace_periods if a third check-in lands first).
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
            # Ordinary heartbeat: same device, still in good standing -- UNLESS this is the
            # session a grace period was started against (set below, by a DIFFERENT device's
            # login exceeding the plan's limit), in which case this poll is exactly how this
            # device is supposed to find out it's the one at risk. This used to return "ok"
            # unconditionally regardless of kick_grace_deadline -- the device being kicked never
            # saw a warning at all, it just silently flipped straight to "kicked" once the
            # deadline lapsed with zero advance notice.
            await async_client.patch(
                f"{SUPABASE_URL}/rest/v1/active_sessions", headers=ADMIN_HEADERS,
                params={"id": f"eq.{existing['id']}"},
                json={"last_active_at": now.isoformat(), "device_label": req.device_label or existing.get("device_label")}
            )
            if existing.get("kick_grace_deadline"):
                return {"status": "warning", "grace_period_ends_at": existing["kick_grace_deadline"], "plan": plan, "device_limit": limit}
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
        if not oldest.get("kick_grace_deadline"):
            deadline = (now + timedelta(minutes=DEVICE_SESSION_GRACE_PERIOD_MINUTES)).isoformat()
            await async_client.patch(
                f"{SUPABASE_URL}/rest/v1/active_sessions", headers=ADMIN_HEADERS,
                params={"id": f"eq.{oldest['id']}"},
                json={"kick_grace_deadline": deadline}
            )
        # This request is from the NEW device that just logged in -- it isn't the one at risk,
        # so it never sees "warning" itself (that used to be the actual bug: this response goes
        # to whoever is CALLING right now, which is the new device, not oldest/old device this
        # grace period was just started against). The old device finds out via its own next
        # heartbeat poll instead, through the "ordinary heartbeat" branch above.
        return {"status": "ok"}
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

async def _check_plan_daily_limit(user_id: str, budget: int):
    """Pro/Max shared: a real, midnight-IST-anchored daily cap, unlike Free's rolling
    24h-from-crossing cooldown above. usage_log is already keyed by IST calendar day (see
    _ist_today), so this is a direct read against today's row -- no separate cooldown-state
    (limit_reached_at) tracking needed the way Free's rolling window requires, since a fresh
    usage_date row naturally appears the moment real midnight IST passes."""
    today = _ist_today()
    resp = await async_client.get(
        f"{SUPABASE_URL}/rest/v1/usage_log", headers=ADMIN_HEADERS,
        params={"user_id": f"eq.{user_id}", "usage_date": f"eq.{today}", "select": "tokens_used"}
    )
    rows = resp.json()
    tokens_used = rows[0]["tokens_used"] if rows else 0
    if tokens_used >= budget:
        raise HTTPException(status_code=402, detail={
            "message": "Daily limit reached",
            "retry_after_seconds": _seconds_until_ist_midnight(),
            "reset_at_midnight_ist": True
        })

async def _log_max_usage_alert(user_id: str, tokens_used: int, tier: str, action_taken: str):
    """Admin-visible audit trail for Max's silent breakeven handling -- never surfaced to the
    student. Idempotent per (user_id, usage_date, tier) via a unique index + ignore-duplicates, so
    repeated calls while a user sits in the same tier the same day don't spam the table -- one row
    per tier crossed per day."""
    try:
        await async_client.post(
            f"{SUPABASE_URL}/rest/v1/max_usage_alerts",
            headers={**ADMIN_HEADERS, "Content-Type": "application/json", "Prefer": "resolution=ignore-duplicates"},
            json={"user_id": user_id, "usage_date": _ist_today(), "tokens_used": tokens_used, "tier": tier, "action_taken": action_taken}
        )
    except Exception as e:
        print(f"MAX USAGE ALERT LOG FAILED: {e}", flush=True)

async def check_max_usage_tier(user_id: str) -> "str | None":
    """Max plan only (returns None immediately for any other plan) -- silent breakeven backstop.
    NEVER blocks, NEVER shown to the student in any form. This used to be a two-tier escalation
    (a "soft zone" starting at the same number that's now MAX_DAILY_TOKEN_CAP, logged for manual
    review with no other effect), but since 2026-09-03 that number is enforce_daily_budget's real,
    visible, ENFORCED cap -- a request can no longer legitimately reach this function with today's
    usage already in that old soft-zone window, since it would already have been blocked before
    ever getting here. Only the breakeven check below still matters, and only for the single call
    that pushes a student from just under the visible cap to past breakeven within one request
    (see MAX_BREAKEVEN_THRESHOLD's own comment) -- the caller (stream_response's text branch) uses
    this "breakeven" return value to force that request (and, redundantly but harmlessly, the rest
    of today's text-doubt calls, though there shouldn't be any more once blocked) onto Qwen-Flash
    -- cheap, flat-rate, no peak surcharge -- instead of the normal peak/off-peak DeepSeek routing.
    Image/PDF doubts are never affected: Gemini is the only vision-capable model available, so
    there is no cheaper substitute to downgrade to."""
    plan = await get_user_plan(user_id)
    if plan != "max":
        return None
    today = _ist_today()
    resp = await async_client.get(
        f"{SUPABASE_URL}/rest/v1/usage_log", headers=ADMIN_HEADERS,
        params={"user_id": f"eq.{user_id}", "usage_date": f"eq.{today}", "select": "tokens_used"}
    )
    rows = resp.json()
    tokens_used = rows[0]["tokens_used"] if rows else 0
    if tokens_used >= MAX_BREAKEVEN_THRESHOLD:
        await _log_max_usage_alert(user_id, tokens_used, "breakeven", "downgraded_to_cheapest_model")
        return "breakeven"
    return None

async def enforce_daily_budget(user_id: str, ip: str = ""):
    if user_id:
        plan = await get_user_plan(user_id)
        if plan == "pro":
            await _check_plan_daily_limit(user_id, DAILY_TOKEN_LIMIT_PRO)
            return
        if plan == "max":
            # As of 2026-09-03, a real visible cap too -- see MAX_DAILY_TOKEN_CAP's own comment.
            # check_max_usage_tier's breakeven backstop still applies independently of this, for
            # the single-large-call overshoot case its own docstring explains.
            await _check_plan_daily_limit(user_id, MAX_DAILY_TOKEN_CAP)
            return
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

# ---------------- Referral program ----------------
# A student's referral code is just their own user_id with hyphens stripped (32 hex chars) --
# no lookup table needed to go from code -> user_id, since it's the UUID with formatting
# removed. Every registered student already has one automatically, including everyone who
# signed up before this feature existed -- nothing to backfill, nothing to generate/store.
def _referral_code_for_user(user_id: str) -> str:
    return user_id.replace("-", "")

def _user_id_from_referral_code(code: str):
    hex_code = (code or "").strip().replace("-", "").lower()
    if len(hex_code) != 32 or not all(c in "0123456789abcdef" for c in hex_code):
        return None
    return f"{hex_code[0:8]}-{hex_code[8:12]}-{hex_code[12:16]}-{hex_code[16:20]}-{hex_code[20:32]}"

REFERRAL_BONUS_TOKENS = 50000

async def _credit_bonus_tokens(user_id: str, amount: int):
    try:
        await async_client.post(
            f"{SUPABASE_URL}/rest/v1/rpc/increment_bonus_tokens", headers=ADMIN_HEADERS,
            json={"p_user_id": user_id, "p_amount": amount}
        )
    except Exception:
        pass

async def log_token_usage_with_bonus(user_id: str, tokens: int, ip: str = ""):
    """Same contract as log_token_usage, except a logged-in student's referral bonus balance is
    drawn down FIRST, and only the overflow beyond that balance is charged to the normal daily
    plan allowance. Used only by the text-doubt billing paths (_stream_qwen/_stream_deepseek,
    covering /chat's text branch and /solve) -- the image/PDF branches in stream_response call
    log_token_usage directly and never reach this function, so a bonus can never be spent on an
    image/PDF doubt by construction, not by an extra flag someone has to remember to check."""
    if tokens <= 0 or not user_id:
        await log_token_usage(user_id, tokens, ip)
        return
    bonus_used = 0
    try:
        resp = await async_client.post(
            f"{SUPABASE_URL}/rest/v1/rpc/decrement_bonus_tokens", headers=ADMIN_HEADERS,
            json={"p_user_id": user_id, "p_amount": tokens}
        )
        if resp.status_code < 400:
            bonus_used = resp.json() or 0
    except Exception:
        bonus_used = 0
    remaining = tokens - bonus_used
    if remaining > 0:
        await log_token_usage(user_id, remaining, ip)

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

# LIVE as of the cutover below (2026-08-28) -- calibrated threshold for the Gemini-embedding-001
# Hindi retrieval path (embedding_gemini column, 3072 dims, queried via match_ncert_hi_gemini).
# Calibrated the same way as NCERT_MATCH_THRESHOLD_HI above: real correct-query-vs-real-chunk
# scores (10 pairs, covering the 2 previously-failing SN1/SN2 and Markovnikov/peroxide topics
# plus 6 known-good regression topics) vs real irrelevant cross-pairs (8 pairs, real queries
# against real but topically unrelated chunks). Correct floor 0.6403, irrelevant ceiling 0.6065
# -- a clean, non-overlapping gap of 0.0339 (wider than NCERT_MATCH_THRESHOLD_HI's own
# calibration gap of 0.0129). 0.62 sits in the middle of that gap.
NCERT_MATCH_THRESHOLD_HI_GEMINI = 0.62

# Dedicated to the Hindi NCERT retrieval path (match_ncert_hi_gemini) -- deliberately NOT routed
# through get_embedding()'s embedding_cache table like every other embedding call in this file.
# That cache is a single hash-keyed table sized for the 1536-dim OpenAI vectors used everywhere
# else (English NCERT, PYQ, diagrams); storing 3072-dim Gemini vectors under the same table would
# risk a dimension mismatch for any other caller reading a cache entry written here. Real Hindi
# query volume is small enough (a few hundred/month across ALL embedding call sites combined,
# confirmed via a real embedding_cache growth-rate check before this cutover) that skipping
# caching has no material cost.
def _get_gemini_query_embedding(text: str):
    resp = gemini_client.models.embed_content(model="gemini-embedding-001", contents=text)
    return resp.embeddings[0].values

# General input-normalization fallback for embedding-based matching (diagram/PYQ/NCERT search) --
# fixes a real, measured failure mode where a compound scientific term typed with an extra space
# (e.g. "eu bacteria" instead of "eubacteria") scores far below match threshold even though the
# correctly-spaced form matches cleanly. Confirmed via real testing that this is a genuine
# embedding-score gap, not an exact-match gate blocking the embedding step first: "eubacteria"
# scored 0.576 against the real Eubacteria diagram, "eu bacteria" scored only 0.389 against that
# SAME stored embedding -- a single added space cost ~0.19 similarity, enough to cross a 0.5
# threshold. Diagram matching is hit hardest because diagram reference embeddings are built from
# very short text (often just a bare name) -- PYQ/NCERT embed full questions/paragraphs, so the
# same kind of query perturbation is a much smaller relative change there (measured drop was
# 5-15x smaller on real PYQ/NCERT rows for "photo synthesis"/"carbo hydrates" style variants),
# which is why they mostly already clear threshold in testing -- this is free defense-in-depth for
# those two, and the actual fix for diagram matching.
#
# Deliberately NOT a hardcoded term dictionary: generates candidate variants by merging exactly
# one adjacent whitespace-separated word pair at a time (bounded: len(words)-1 candidates for an
# N-word query), which is exactly the shape of "a compound word got typed as two words" without
# assuming which word pair it is. Only fires as a fallback when the raw query's own top match
# misses threshold -- zero extra embedding calls on the common case of already-correct spelling.
def _generate_space_merge_variants(text: str) -> list:
    words = text.split()
    return [
        " ".join(words[:i] + [words[i] + words[i + 1]] + words[i + 2:])
        for i in range(len(words) - 1)
    ]

async def _match_with_merge_fallback(query: str, embed_fn, search_fn):
    """embed_fn(text) -> embedding, sync or async (both handled via iscoroutine on the call
    result). search_fn(embedding) -> awaitable returning a list of already-threshold-filtered
    result rows. Tries the raw query first; only on an empty result does it retry each single-
    adjacent-word-merge variant in turn, returning the first variant that produces a real match."""
    async def _embed(text):
        result = embed_fn(text)
        return await result if asyncio.iscoroutine(result) else result

    embedding = await _embed(query)
    results = await search_fn(embedding)
    if results:
        return results
    for variant in _generate_space_merge_variants(query):
        v_embedding = await _embed(variant)
        v_results = await search_fn(v_embedding)
        if v_results:
            return v_results
    return results

async def search_ncert(query: str, limit: int = 3):
    headers = {
        "apikey": SUPABASE_KEY,
        "Authorization": f"Bearer {SUPABASE_KEY}",
        "Content-Type": "application/json"
    }
    language = _detect_query_language(query)

    def _filter_results(rows):
        # chapter_name_en is what actually carries "Appendix"/etc for Hindi rows (chapter_name
        # itself is the real Hindi title, e.g. "परिशिष्ट", which would never match these English
        # labels) -- falls back to chapter_name for English rows, where chapter_name_en is null.
        return [r for r in rows if (r.get("chapter_name_en") or r.get("chapter_name")) not in NCERT_NON_CHAPTER_LABELS]

    if language == "hi":
        # Gemini gemini-embedding-001 shadow trial, cut over live 2026-08-28 -- see
        # NCERT_MATCH_THRESHOLD_HI_GEMINI above for the calibration behind this threshold, and
        # add_ncert_content_gemini_embedding_shadow.sql for the parallel column/RPC this reads.
        # The old match_ncert + NCERT_MATCH_THRESHOLD_HI + `embedding` column path (OpenAI,
        # text-embedding-3-small) is deliberately left completely intact, unused, as an instant
        # rollback: revert this branch to call match_ncert with filter_language="hi" and
        # NCERT_MATCH_THRESHOLD_HI again, no data migration needed either way.
        async def _search(embedding):
            response = await async_client.post(
                f"{SUPABASE_URL}/rest/v1/rpc/match_ncert_hi_gemini",
                headers=headers,
                json={"query_embedding": embedding, "match_threshold": NCERT_MATCH_THRESHOLD_HI_GEMINI, "match_count": limit}
            )
            return _filter_results(response.json()) if response.status_code == 200 else []
        return await _match_with_merge_fallback(query, _get_gemini_query_embedding, _search)
    else:
        # English path: same get_embedding() call (OpenAI text-embedding-3-small, embedding_cache-
        # backed), same match_ncert RPC, same NCERT_MATCH_THRESHOLD_EN, same filter_language param
        # -- now wrapped in the merge-variant fallback above instead of a bare single attempt.
        async def _search(embedding):
            response = await async_client.post(
                f"{SUPABASE_URL}/rest/v1/rpc/match_ncert",
                headers=headers,
                json={"query_embedding": embedding, "match_threshold": NCERT_MATCH_THRESHOLD_EN, "match_count": limit, "filter_language": language}
            )
            return _filter_results(response.json()) if response.status_code == 200 else []
        return await _match_with_merge_fallback(query, get_embedding, _search)

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

# Real calibration data (see PYQ_MATCH_THRESHOLD's own comment above, 2026-08-24): 4 genuinely
# irrelevant queries topped out at 0.282 real similarity; real on-topic queries scored
# 0.416-0.657. 0.30 sits comfortably above that irrelevant ceiling -- a "related" fallback result
# is still real, honest topical relevance (not literally the bottom of the corpus), while being
# low enough to actually catch a genuine near-miss. Measured real example that motivated this:
# "what causes color blindness genetically" (a real NCERT genetics topic -- X-linked inheritance)
# scored 0.394 -- correctly rejected by the strict 0.4 bar as not a precise-enough match, but its
# real top results at a relaxed threshold (X-linked chromosome inheritance, pedigree probability,
# genotype crosses) are genuinely topically coherent, not scattered noise -- confirmed embedding-
# based ranking alone (no chapter filter needed) already clusters real PYQ content by topic.
PYQ_FALLBACK_THRESHOLD = 0.30
PYQ_FALLBACK_LIMIT = 5

# Real calibration data (2026-08-30): the 4 irrelevant queries PYQ_MATCH_THRESHOLD's own comment
# above tested (capital of France, baking a cake, movie recommendations, weather) topped out at
# 0.282 as expected -- but "who won the cricket world cup in 2011" scored ABOVE the primary 0.4
# threshold, matching "The UN Conference of Parties on climate change in the year 2011 was held
# in :" (a mistagged Physics-table row that's really general-knowledge current-affairs content,
# not real Physics) purely off the shared "in 2011" year token and question phrasing. Separately,
# "what is the capital of India" also weakly matched via the relaxed fallback pass against loosely
# India-themed Biology PYQs (National Aquatic Animal, Bt crops) that aren't actually relevant to
# what was asked. Both are the same underlying pattern -- general-knowledge/current-affairs
# queries embedding close to unrelated PYQ rows -- not one bad chunk worth hunting down
# individually (fixing just the UN-conference row wouldn't have caught "capital of India" too).
# Same technique as FALSE_POSITIVE_CLARIFY_WORDS elsewhere in this file (rule 11's false-positive
# guard): a small, evidence-based keyword denylist,
# checked before the embedding search ever runs. Kept deliberately small/conservative -- every
# term here has zero real NEET-syllabus overlap; this is not a general-purpose off-topic
# classifier and never grows into an allowlist of "real" NEET terms (which would risk rejecting a
# genuine doubt that just doesn't happen to use textbook vocabulary).
OFF_TOPIC_PYQ_TERMS = {
    "cricket", "football", "ipl", "olympics", "world cup",
    "prime minister", "president", "chief minister", "election",
    "movie", "bollywood", "actor", "actress", "celebrity",
    "capital of",
}

def _is_off_topic_pyq_query(text: str) -> bool:
    normalized = text.strip().lower()
    return any(term in normalized for term in OFF_TOPIC_PYQ_TERMS)

async def search_pyq(query: str, limit: int = 5):
    if _is_off_topic_pyq_query(query):
        return [], False
    headers = {
        "apikey": SUPABASE_KEY,
        "Authorization": f"Bearer {SUPABASE_KEY}",
        "Content-Type": "application/json"
    }
    async def _search_at(threshold, count):
        async def _do(embedding):
            response = await async_client.post(
                f"{SUPABASE_URL}/rest/v1/rpc/match_pyq",
                headers=headers,
                json={"query_embedding": embedding, "match_threshold": threshold, "match_count": count}
            )
            return response.json() if response.status_code == 200 else []
        return await _match_with_merge_fallback(query, get_embedding, _do)

    results = await _search_at(PYQ_MATCH_THRESHOLD, limit)
    if results:
        return results, False

    # No real match even after the typo/spacing merge-variant fallback inside _search_at -- try
    # once more at the relaxed "related, not exact" threshold before giving up entirely. Returns
    # (results, True) only when this relaxed pass actually found something; an empty result here
    # means the corpus genuinely has nothing close, and the caller should show honest "no results"
    # rather than being told a fallback ran and still return nothing.
    fallback_results = await _search_at(PYQ_FALLBACK_THRESHOLD, PYQ_FALLBACK_LIMIT)
    return fallback_results, bool(fallback_results)

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

import difflib

# Shared by _is_denylisted_clarify_doubt and _is_legitimate_topic_ambiguity below -- both compare
# a single bare-word/short-phrase doubt against a small closed set, and both had the same real
# gap: exact match only, so a genuine typo or an extra/missing internal space on an otherwise-
# correct word (e.g. "resist ance", "resistence", "cyle" for "resistance"/"cycle") silently missed
# even though the student clearly meant the listed word. Exact match stays the fast path; fuzzy
# match is a fallback via difflib.SequenceMatcher (stdlib, no new dependency), whole-string ratio
# against each set member (not per-token) since every entry here IS a single word/short phrase --
# a high cutoff (0.84) catches a doubled/missing/swapped letter or one inserted space while still
# rejecting a genuinely different word that happens to be similar length, which matters here more
# than in a general search: this gates a security/reliability backstop, not a soft ranking, so a
# false MATCH (treating an unrelated word as if it were "cycle") is a worse failure than a missed
# fuzzy hit would be. Cutoff of 0.86 picked from real measured SequenceMatcher ratios, not a
# guess: the lowest real typo/spacing variant tested ("cyle" vs "cycle") scored 0.889, the
# highest real different-word false positive found ("resistant" vs "resistance", genuinely a
# different word/meaning, not a typo) scored 0.842 -- 0.86 sits in the middle of that real gap.
def _fuzzy_word_match(normalized: str, wordset: set, cutoff: float = 0.86) -> bool:
    if normalized in wordset or normalized.lower() in wordset:
        return True
    return bool(difflib.get_close_matches(normalized.lower(), wordset, n=1, cutoff=cutoff))

def _is_denylisted_clarify_doubt(text: str) -> bool:
    """Rule 11 itself only ever fires on a bare word/short phrase doubt (its own full-sentence
    guard already handles longer doubts that merely mention one of these words), so matching on
    the whole stripped doubt is enough and avoids ever matching a denylisted word that's
    incidentally part of a real, different question. See _fuzzy_word_match for the exact-then-
    fuzzy matching strategy."""
    normalized = text.strip()
    return _fuzzy_word_match(normalized, FALSE_POSITIVE_CLARIFY_WORDS)

# The inverse of FALSE_POSITIVE_CLARIFY_WORDS above: rule 11's own CLOSED whitelist for
# CLARIFY_TYPE: topic, mirrored here as a general server-side backstop rather than another
# denylist entry. Confirmed live: "ले शातेलिए का सिद्धांत" (Le Chatelier's Principle) triggered a
# hallucinated AMBIGUOUS response inventing two fake alternate meanings -- a direct violation of
# rule 11's own explicit "if it's not on the list, it's not ambiguous, full stop" instruction.
# 1/1 failure in the original audit, 0/3 on a same-day recheck: intermittent, not deterministic,
# which per the "reflex" precedent above means prompt-only enforcement alone won't reliably hold
# for every term that isn't on the list, only for the specific ones a denylist happens to cover.
# Checking the model's claim against the whitelist directly (instead of enumerating every possible
# hallucinated term one at a time, the way FALSE_POSITIVE_CLARIFY_WORDS had to for "reflex") means
# this backstop already covers any FUTURE hallucinated term too, not just this one.
TOPIC_AMBIGUITY_WHITELIST = {
    "resistance", "प्रतिरोध",
    "cycle", "चक्र",
    "potential", "विभव",
    "diffusion", "विसरण",
    "current", "धारा",
    "valence", "संयोजकता",
}

def _is_legitimate_topic_ambiguity(text: str) -> bool:
    """Same matching strategy as _is_denylisted_clarify_doubt above, for the same reason:
    rule 11 only ever legitimately fires TOPIC AMBIGUITY for a bare word doubt that IS one of
    these six terms, not a longer question that merely mentions one. See _fuzzy_word_match."""
    normalized = text.strip()
    return _fuzzy_word_match(normalized, TOPIC_AMBIGUITY_WHITELIST)

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

async def _stream_qwen(system: str, messages: list, user_id: str, ip: str, endpoint: str, billing_context: dict = None, max_tokens: int = 1024):
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
    e.g. /solve) means "always bill", matching the pre-existing behavior.

    max_tokens defaults to 1024 (the historical value for every real-answer caller) -- only
    _verify_given_values overrides this, to a much smaller cap, since a runaway verification
    response (observed live: some trials rambled to 600-1000+ output tokens re-litigating their
    own answer) is pure wasted latency, not a longer/better answer."""
    qwen_stream = await qwen_async_client.chat.completions.create(
        model="qwen-flash",
        max_tokens=max_tokens,
        stream=True,
        stream_options={"include_usage": True},
        messages=[{"role": "system", "content": system}] + messages
    )
    input_tokens = output_tokens = 0
    try:
        async for chunk in qwen_stream:
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
                    await log_token_usage_with_bonus(user_id, input_tokens + output_tokens, ip)
                except Exception:
                    pass

async def _stream_deepseek(system: str, messages: list, user_id: str, ip: str, is_peak: bool, endpoint: str, billing_context: dict = None):
    """The default off-peak provider, and also the peak-window fallback when Qwen is
    unavailable -- is_peak is only for the usage-log tag, so peak-window fallback traffic still
    shows up as such in the logs even though DeepSeek ended up serving it. See _stream_qwen's
    docstring for what billing_context does."""
    async with deepseek_async_client.messages.stream(
        model="deepseek-v4-flash",
        max_tokens=1024,
        thinking={"type": "disabled"},
        system=system,
        messages=messages
    ) as stream:
        try:
            async for text_chunk in stream.text_stream:
                yield text_chunk
        finally:
            try:
                final_message = await stream.get_final_message()
                usage = final_message.usage
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
                    await log_token_usage_with_bonus(user_id, usage.input_tokens + usage.output_tokens, ip)
            except Exception:
                pass

async def _stream_with_peak_fallback(system: str, messages: list, user_id: str, ip: str, endpoint: str, billing_context: dict = None, force_qwen: bool = False):
    """Routes text-doubt generation (/chat, /solve -- never image/PDF doubts, those stay on
    Gemini regardless of time of day) to Qwen-Flash during DeepSeek's published peak-pricing
    windows (_is_deepseek_peak_hour), DeepSeek everywhere else. Confirmed via a live 45-question
    test that the two are equivalent on accuracy/reliability, so this is purely a cost-avoidance
    swap, not a quality tradeoff.

    force_qwen: set by the caller when a Max-plan user has crossed MAX_BREAKEVEN_THRESHOLD for
    today (see check_max_usage_tier) -- routes to Qwen regardless of the real peak/off-peak clock,
    since Qwen's flat rate has no peak surcharge and is reliably cheap. Deliberately reuses this
    same function (not a separate code path) specifically to inherit its existing Qwen-then-
    DeepSeek failover for free: if Qwen is ever down, a downgraded Max user still gets a normal
    answer via DeepSeek, never an error -- exactly the "no error, no message" requirement for
    Max's silent handling. is_peak passed to the DeepSeek fallback below still reflects the REAL
    clock (not force_qwen) -- a force-routed call during a genuinely off-peak hour that has to
    fall back to DeepSeek must still bill DeepSeek's off-peak rate, not be charged peak pricing
    just because Qwen was tried first.

    Failover: if Qwen fails before yielding anything, the same request is retried on DeepSeek
    transparently -- the student never sees the difference, and this accepts DeepSeek's peak
    cost rather than failing the request outright (availability over cost optimization). If Qwen
    fails after already streaming part of an answer, there's no clean way to hand off to a
    different model mid-response without duplicating or garbling what the student already saw,
    so the failure just propagates like any other mid-stream error in this file -- the caller's
    own try/except turns it into a yielded "Error: ..." message and correctly skips caching a
    partial answer, exactly as it already does for a mid-stream DeepSeek failure."""
    is_peak = _is_deepseek_peak_hour()
    if force_qwen or is_peak:
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
        async for chunk in _stream_deepseek(system, messages, user_id, ip, is_peak, endpoint, billing_context):
            yield chunk
        return
    async for chunk in _stream_deepseek(system, messages, user_id, ip, False, endpoint, billing_context):
        yield chunk

# ---------- Second-pass verification for Qwen's "borrowed value" insufficient-data trap ----------
# Live-tested failure: given a problem describing two comparable objects (e.g. two towers) where
# only ONE object's value is stated (e.g. tower A's height), Qwen reliably (0/5 baseline trials)
# fabricates a numeric answer by silently reusing that value for the OTHER object instead of
# recognizing the data is incomplete. A same-mechanism-as-before fix (a named worked example in
# SYSTEM_PROMPT) was tried first and failed completely (0/5, then 0/5 again after strengthening) --
# the model would even correctly articulate the exact flaw mid-answer ("unless tower B is also
# 45m tall... otherwise unsolvable") and then override its own correct diagnosis to force a number
# anyway. This second-pass mechanism fixed it (~87.5% across 8 tower trials, 3/3 and 3/3 on two
# unrelated unnamed multi-object problems, 0 false positives on a legitimate-shared-value case)
# by moving the unreliable part (aggregating "is this valid" from a wall of reasoning) out of the
# LLM and into code, and by using a deterministic template for the correction instead of a 3rd LLM
# call (which also reproduced the original fabrication 4/5 times even when told exactly what was
# wrong) -- same "server-side guard beats another prompt layer" lesson as the rule-11 fix above.
_MULTI_OBJECT_NOUNS = ["tower", "vehicle", "car", "container", "spring", "block", "particle", "body",
                       "ball", "stone", "tank", "wire", "resistor", "capacitor", "solution", "sample",
                       "building", "ship", "train", "pipe", "rod", "sphere", "charge", "object", "cart",
                       "trolley", "plane", "boat", "pendulum", "beaker", "flask", "cylinder", "disc",
                       "disk", "wheel", "ladder", "pole", "pillar", "column", "box", "jar", "lens",
                       "mirror", "slit", "cell", "battery", "mass", "force", "bullet", "bob", "wedge",
                       "incline", "plank"]
_MULTI_OBJECT_AGGREGATE_RE = re.compile(
    r"\b(two|three|four|both)\s+(" + "|".join(_MULTI_OBJECT_NOUNS) + r")s?\b", re.IGNORECASE
)

def _is_multi_object_numerical(text: str) -> bool:
    """Cheap, no-LLM-call gate for the narrow second-pass verification path below -- questions
    describing 2+ comparable objects/entities (two towers, two vehicles, two containers, etc.),
    the shape of problem where the borrowed-value trap above was observed. Deliberately NOT also
    gated on "does this look numerical" -- an earlier version tried that (requiring a literal
    "Final Answer:" line) and it silently let a real fabrication through, since the model doesn't
    reliably use that exact literal format even when it did fabricate a number (seen live: the
    wrong result stated only inside a "Key Points:" bullet). A missed fabrication is worse than an
    occasional wasted verification call on a genuinely conceptual multi-object question."""
    if _MULTI_OBJECT_AGGREGATE_RE.search(text):
        return True
    lowered = text.lower()
    for noun in _MULTI_OBJECT_NOUNS:
        if len(re.findall(rf"\b{noun}s?\b", lowered)) >= 2:
            return True
    return False

_VERIFY_GIVEN_VALUES_SYSTEM = """Fact-check this SOLUTION against the ORIGINAL PROBLEM. The problem describes
2+ comparable objects (e.g. two towers). Check whether each value used in the calculation was
LITERALLY stated in the problem FOR THE SPECIFIC object it's used for -- not borrowed from a
different object, and not inferred just because they share a start time/instant. Exception: if the
problem explicitly says the objects are identical or gives the same property to both, that value is
valid for both.

Output ONLY this exact format, nothing else -- no discussion, no "wait", no rechecking, no second
draft of the table:
<value> -- <object>: STATED
<value> -- <object>: NOT STATED (<reason, under 6 words>)
...one line per value...
VERDICT: VALID or VERDICT: INVALID
MISSING_QUANTITY: <if INVALID, name the ONE missing quantity in under 8 words. If VALID, write N/A.>

MECHANICAL RULE: if ANY line says NOT STATED, the verdict MUST be INVALID -- no exceptions, even if
the value seems physically reasonable or logically inferable."""

# Cap chosen from a live 14-trial controlled comparison against the original, longer prompt (no
# cap, same 1024 default as a real answer): 14/14 correct on both, but this trimmed prompt + cap
# cut average verification latency from 1.98s to 1.08s and worst-case from 6.30s (one trial hit
# the old 1024-token ceiling re-litigating its own answer with "wait, let me recheck") down to
# 2.17s, with zero accuracy difference -- the largest real response seen under the new prompt was
# ~430 chars (~110 tokens), comfortably under this cap with no truncation risk.
_VERIFY_GIVEN_VALUES_MAX_TOKENS = 300

async def _verify_given_values(problem_text: str, solution_text: str, user_id: str, ip: str, billing_context: dict = None):
    """Runs the check above via a second Qwen call and returns (is_valid, raw_response). The
    verdict is computed HERE in code, not read from the model's own "VERDICT:" line -- live
    testing found the model reliably IDENTIFIES a borrowed value per-line ("NOT STATED") but
    unreliably AGGREGATES that into VERDICT: INVALID, instead rationalizing past its own finding
    and writing VALID anyway. Triggering on EITHER signal (a per-line flag, or the model's own
    INVALID verdict) catches both the "said VALID despite flagging a line" case seen repeatedly in
    testing and the rarer reverse case (INVALID written without a matching per-line flag)."""
    user_msg = f"ORIGINAL PROBLEM:\n{problem_text}\n\nPROPOSED SOLUTION:\n{solution_text}"
    messages = [{"role": "user", "content": user_msg}]
    chunks = []
    async for c in _stream_qwen(_VERIFY_GIVEN_VALUES_SYSTEM, messages, user_id, ip, "/chat-verify", billing_context, max_tokens=_VERIFY_GIVEN_VALUES_MAX_TOKENS):
        chunks.append(c)
    full = "".join(chunks)
    upper = full.upper()
    is_valid = "NOT STATED" not in upper and "VERDICT: INVALID" not in upper
    return is_valid, full

_MISSING_QUANTITY_RE = re.compile(r"MISSING_QUANTITY:\s*(.+)", re.IGNORECASE)

def _extract_missing_quantity(verify_raw: str):
    m = _MISSING_QUANTITY_RE.search(verify_raw)
    if not m:
        return None
    q = m.group(1).strip().strip('"').rstrip(".")
    return q if q and q.upper() != "N/A" else None

def _build_insufficient_data_answer(missing_quantity):
    # Deterministic template, not a 3rd LLM call -- testing showed the model rewrites the exact
    # same fabricated answer even when explicitly told what's wrong (4/5 trials), so correctness
    # here comes from code, not from asking the model to try again.
    mq = missing_quantity or "a required value"
    return f"""VISUAL_INTENT: no

NEET Importance: 3/5

Answer:
- This problem cannot be solved with the information given.
- The value needed here — {mq} — is never stated in the problem.
- If {mq} were given, the rest of the calculation could be completed normally.

Final Answer: Cannot be determined — {mq} is missing from the problem."""

SIMULATED_STREAM_MARKER = "SIMULATED_STREAM: yes\n"

async def _stream_qwen_verified(system: str, messages: list, problem_text: str, user_id: str, ip: str, endpoint: str, billing_context: dict = None):
    """Used only when the caller has already decided (a) this request is being routed to Qwen and
    (b) the question matches _is_multi_object_numerical -- see the trigger condition in
    stream_response's text branch. Buffers Qwen's full first answer (no live token streaming is
    possible here -- nothing can be shown until verification decides what the real final answer
    is), runs _verify_given_values against it, and yields the original answer unchanged if valid,
    or a deterministic "cannot be determined" answer naming the missing quantity if not.

    Yields SIMULATED_STREAM_MARKER as its own first chunk once Qwen's first pass succeeds, BEFORE
    verification -- never sent if Qwen fails outright (falls back to plain, normal DeepSeek
    streaming instead, exactly like _stream_with_peak_fallback's own failover), since the caller
    (chat.html) uses this marker to keep its existing thinking indicator up through the
    verification step, then replay the final answer through its normal per-character render path
    at a synthetic pace so it looks identical to real streaming. stream_response strips this
    marker out BEFORE feeding the rest into the rule-11 AMBIGUOUS/CLARIFY_TYPE buffering state
    machine -- that logic inspects the stream's very first line, and must never mistake this
    marker for (or let it displace) the real first content line."""
    is_peak = _is_deepseek_peak_hour()
    try:
        chunks = []
        async for c in _stream_qwen(system, messages, user_id, ip, endpoint, billing_context):
            chunks.append(c)
        first_answer = "".join(chunks)
    except Exception as e:
        print(f"QWEN UNAVAILABLE BEFORE VERIFICATION, FALLING BACK TO DEEPSEEK: {e}", flush=True)
        async for chunk in _stream_deepseek(system, messages, user_id, ip, is_peak, endpoint, billing_context):
            yield chunk
        return

    yield SIMULATED_STREAM_MARKER
    is_valid, verify_raw = await _verify_given_values(problem_text, first_answer, user_id, ip, billing_context)
    if is_valid:
        yield first_answer
        return
    missing_quantity = _extract_missing_quantity(verify_raw)
    yield _build_insufficient_data_answer(missing_quantity)

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
        lang_context = "\n\nIMPORTANT: Respond ONLY in Hindi (Devanagari script). Every word — headings, key points, explanations, memory tricks — must be in Hindi. Do not mix in English words or Hinglish, even for common scientific terms (e.g. write \"गुणसूत्र\" not \"chromosome\"). The ONLY exceptions are: LaTeX/KaTeX math notation, chemical formulas/symbols (e.g. $H_2O$), units (e.g. m/s, kg), and proper nouns like NEET or NCERT — keep those exactly as-is, do not translate or romanize them. For the section headers specifically, use these EXACT fixed Hindi labels rather than inventing your own translation each time: \"Answer:\" becomes \"उत्तर:\", \"Key Points:\" becomes \"मुख्य बिंदु:\", \"Quick Recall:\" becomes \"त्वरित याद:\". \"NEET Importance:\" and \"Chapter:\" stay in English exactly as given in the format above — never translate those two labels." if language == "hi" else ""
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
- Keep the Quick Recall line to one short sentence.
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
            gemini_stream = await gemini_client.aio.models.generate_content_stream(
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
                async for chunk in gemini_stream:
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
            gemini_stream = await gemini_client.aio.models.generate_content_stream(
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
                async for chunk in gemini_stream:
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
            # Max plan only (no-op/instant None for every other plan) -- see check_max_usage_tier
            # for the full soft-zone/breakeven logic. force_qwen only affects TEXT doubts: there's
            # no cheaper substitute for Gemini on image/PDF doubts, so those are never touched.
            max_tier = await check_max_usage_tier(user_id)
            force_qwen = max_tier == "breakeven"
            print(f"MODEL SELECTED: {'qwen-flash (Max breakeven downgrade)' if force_qwen else ('qwen-flash (DeepSeek peak-hour fallback)' if is_peak else 'deepseek-v4-flash')}", flush=True)
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
            # Every text doubt now goes through this same 2-line-deep buffered check (previously
            # only FALSE_POSITIVE_CLARIFY_WORDS doubts like "reflex" were buffered at all, and
            # only 1 line deep) -- a deliberate tradeoff of a small, constant, uniform buffering
            # delay (at most two short header lines, ~128 chars) for every doubt, in exchange for
            # a guard that catches ANY hallucinated CLARIFY_TYPE: topic response, not just
            # pre-enumerated denylisted words. See TOPIC_AMBIGUITY_WHITELIST above for why a
            # denylist alone (adding one bad term at a time, as happened for "reflex") isn't
            # sufficient here -- Le Chatelier's Principle was never denylisted and still leaked.
            #
            # Checkpoint 0: buffer up to the first line. Not "AMBIGUOUS: yes" -> release
            # everything buffered and pass every subsequent chunk straight through (checkpoint 2).
            # Checkpoint 1 (only reached if AMBIGUOUS: yes): buffer up to the CLARIFY_TYPE line
            # too -- format ambiguity is a different, legitimate mechanism with its own gate (the
            # diagram-exists context note) and is never discarded here regardless of doubt text;
            # only CLARIFY_TYPE: topic is checked against the whitelist. Denylisted doubts
            # (FALSE_POSITIVE_CLARIFY_WORDS) are still discarded unconditionally regardless of
            # type, exactly as before. Invalid -> break with nothing ever yielded, not even the
            # first line -- a bad clarify attempt must never reach the student at all.
            # Route through the second-pass verification wrapper only when this request is
            # already going to Qwen AND the question matches the narrow multi-object-numerical
            # heuristic (see _stream_qwen_verified above) -- every other text doubt is completely
            # unaffected and streams exactly as before.
            use_qwen_for_this_request = is_peak or force_qwen
            multi_object_numerical = use_qwen_for_this_request and _is_multi_object_numerical(text)
            if multi_object_numerical:
                stream_source = _stream_qwen_verified(full_system, messages, text, user_id, ip, "/chat", billing_context)
            else:
                stream_source = _stream_with_peak_fallback(full_system, messages, user_id, ip, "/chat", billing_context, force_qwen)

            # Peel off a leading SIMULATED_STREAM_MARKER chunk (only ever yielded by
            # _stream_qwen_verified) BEFORE it reaches the rule-11 AMBIGUOUS/CLARIFY_TYPE
            # buffering state machine below -- that logic inspects the stream's very first line to
            # decide whether this is a clarifying-question response, and must never mistake this
            # marker line for (or let it displace) the real first content line, or the
            # anti-hallucination denylist/whitelist checks on a genuine AMBIGUOUS response would
            # be silently skipped for every verified answer.
            stream_iter = stream_source.__aiter__()
            try:
                peeked_chunk = await stream_iter.__anext__()
            except StopAsyncIteration:
                peeked_chunk = None
            if peeked_chunk == SIMULATED_STREAM_MARKER:
                yield peeked_chunk
                try:
                    peeked_chunk = await stream_iter.__anext__()
                except StopAsyncIteration:
                    peeked_chunk = None

            async def _rest_of_stream():
                if peeked_chunk is not None:
                    yield peeked_chunk
                async for c in stream_iter:
                    yield c

            pending = ""
            checkpoint = 0
            override_needed = False
            async for text_chunk in _rest_of_stream():
                full_answer += text_chunk

                if checkpoint == 2:
                    if billing_context["bill"] and full_answer.strip().startswith("AMBIGUOUS: yes"):
                        billing_context["bill"] = False
                    yield text_chunk
                    continue

                pending += text_chunk

                if checkpoint == 0:
                    nl_idx = pending.find("\n")
                    if nl_idx == -1 and len(pending) < 64:
                        continue
                    first_line = (pending[:nl_idx] if nl_idx != -1 else pending).strip()
                    if first_line != "AMBIGUOUS: yes":
                        checkpoint = 2
                        if billing_context["bill"] and full_answer.strip().startswith("AMBIGUOUS: yes"):
                            billing_context["bill"] = False
                        yield pending
                        pending = ""
                        continue
                    checkpoint = 1
                    # Falls through to the checkpoint 1 block below in this same iteration -- a
                    # real stream can deliver both header lines in a single chunk.

                if checkpoint == 1:
                    first_nl = pending.find("\n")
                    second_nl = pending.find("\n", first_nl + 1)
                    after_first_line = pending[first_nl + 1:]
                    if second_nl == -1 and len(after_first_line) < 64:
                        continue
                    clarify_type_line = (pending[first_nl + 1:second_nl] if second_nl != -1 else after_first_line).strip()
                    is_topic_type = "CLARIFY_TYPE:" in clarify_type_line and "topic" in clarify_type_line.lower()
                    invalid = _is_denylisted_clarify_doubt(text) or (is_topic_type and not _is_legitimate_topic_ambiguity(text))
                    if invalid:
                        override_needed = True
                        break
                    checkpoint = 2
                    if billing_context["bill"] and full_answer.strip().startswith("AMBIGUOUS: yes"):
                        billing_context["bill"] = False
                    yield pending
                    pending = ""
            # Stream ended while still buffered (checkpoint 0 waiting for a newline that never
            # came, or checkpoint 1 similarly on the second line) -- confirmed live: a short
            # conversational reply whose ENTIRE completion is under 64 chars with no newline at
            # all (e.g. Qwen returning just "DOUBT_TYPE: conversational" and nothing else, no
            # trailing newline, generation just stopping there) left `pending` sitting unflushed
            # forever, since the checkpoint-0/1 blocks only ever yield from inside the loop. The
            # student got a real 200 OK with a completely empty body -- this flushes whatever
            # survived instead of silently discarding it. Not reachable when override_needed is
            # true (that path already `break`s with intentionally-discarded pending content, by
            # design -- a fresh override attempt follows below).
            if not override_needed and pending:
                if billing_context["bill"] and full_answer.strip().startswith("AMBIGUOUS: yes"):
                    billing_context["bill"] = False
                yield pending
                pending = ""
            if override_needed:
                # The discarded attempt's own tiny (two-line) token cost still gets logged by
                # _stream_qwen/_stream_deepseek's own finally block when the `break` above closes
                # it via GeneratorExit -- same accepted partial-cost tradeoff as before, and
                # negligible next to a full response's cost either way. This second call is a
                # completely fresh request/response cycle, billed normally.
                override_system = full_system + (
                    "\n\nOVERRIDE (server-enforced, not optional): this exact doubt has been "
                    "confirmed through repeated real testing to NOT be genuinely ambiguous, "
                    "despite rule 11 above. Answer it directly and normally in the standard "
                    "format starting with VISUAL_INTENT -- do not output AMBIGUOUS/"
                    "CLARIFY_TYPE under any circumstances for this doubt."
                )
                full_answer = ""
                billing_context = {"bill": True}
                async for text_chunk in _stream_with_peak_fallback(override_system, messages, user_id, ip, "/chat", billing_context, force_qwen):
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
        # shared logic and its accepted early-disconnect / mid-stream-failure tradeoffs. Also
        # subject to the same Max-plan breakeven downgrade (text-only endpoint, no image
        # involved, same reasoning as stream_response's text branch).
        force_qwen = (await check_max_usage_tier(user_id)) == "breakeven"
        async for text_chunk in _stream_with_peak_fallback(solve_system, solve_messages, user_id, ip, "/solve", force_qwen=force_qwen):
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

_TITLE_MARKDOWN_RE = re.compile(r'[*_`#>~\[\]]+')
_TITLE_WHITESPACE_RE = re.compile(r'\s+')
# Catches the model slipping into "replying to the student" instead of "labeling the topic" --
# e.g. "I'm ready to help! Please share the NEET question you'd..." (a real observed case, cut off
# mid-sentence by max_tokens). A markdown-strip/length-cap backstop alone can't catch this since
# the text itself isn't malformed, just the wrong genre -- same "instruction alone isn't reliable,
# add a code-level guard" lesson as Rule 11's FALSE_POSITIVE_CLARIFY_WORDS denylist. English-only
# (no Hindi equivalent list); a Hindi title that slips this way still gets caught by the length cap.
_TITLE_REPLY_PATTERN_RE = re.compile(
    r"^(i'?m|i can|i'?ll|i will|sure|okay|ok|certainly|of course|please|let me|here'?s|happy to|glad to|no problem|you'?re welcome)\b",
    re.IGNORECASE
)

# Backstop for /title, same reasoning as Rule 11's FALSE_POSITIVE_CLARIFY_WORDS denylist guard
# elsewhere in this file: a closed prompt constraint (below) cuts how often the model drifts into
# replying/quoting instead of labeling, but doesn't cut it to zero, so this strips whatever
# markdown survives, catches reply-shaped output, and hard-caps length regardless of what the
# model actually returned -- guaranteed short and clean even on a prompt-adherence miss.
def _clean_title(raw: str, fallback: str = "New Chat") -> str:
    cleaned = _TITLE_MARKDOWN_RE.sub('', raw or '')
    cleaned = _TITLE_WHITESPACE_RE.sub(' ', cleaned).strip()
    if _TITLE_REPLY_PATTERN_RE.match(cleaned) or cleaned.endswith(('!', '?')):
        return fallback
    words = cleaned.split(' ')
    if len(words) > 4:
        cleaned = ' '.join(words[:4])
    if len(cleaned) > 40:  # backstop for pasted no-space text / dense scripts word-splitting doesn't catch
        cleaned = cleaned[:40].rstrip()
    return cleaned or fallback

@app.post("/title")
async def generate_title(message: Message, request: Request, _: None = Depends(rate_limiter(15, 60))):
    ip = _client_ip(request)
    await enforce_daily_budget(message.user_id, ip)
    client = deepseek_client
    fallback_title = "New Chat" if message.language != "hi" else "नई चैट"
    title_lang = "entirely in Hindi (Devanagari script) — every word in Hindi, no English words mixed in" if message.language == "hi" else "in English"
    response = client.messages.create(
        model="deepseek-v4-flash",
        max_tokens=15,
        thinking={"type": "disabled"},
        system=(
            f"Generate a short chat-sidebar title {title_lang} for this conversation, based ONLY on "
            "the student's message below -- the way a chat app names a conversation, never based on "
            "how you personally would respond to it. You are a labeler, not a participant in this "
            "conversation.\n\n"
            "STRICT rules, all mandatory, no exceptions:\n"
            "1. Exactly 2-4 words. Never more, no matter how long or detailed the message is -- "
            "summarize down to the core topic. Do NOT quote, copy, or extract phrases verbatim from "
            "the message, even if it already contains short phrases that look title-sized.\n"
            "2. Plain text only: no markdown symbols (*, _, `, #, >, ~, [, ]), no punctuation, no "
            "emoji, no quotation marks.\n"
            "3. Never write a sentence, never address the student, never reply to, answer, "
            "continue, or acknowledge the message -- e.g. never output anything starting with or "
            "resembling \"I can help\", \"Please share\", \"Sure,\", or similar. If you notice "
            "yourself about to respond to the student rather than label the topic, stop and produce "
            f"a topic label instead.\n"
            f"4. If the message is a greeting, thanks, small talk, or has no identifiable academic "
            f"topic, output EXACTLY: {fallback_title}\n\n"
            "Examples: \"hi\" -> " + fallback_title + ". \"thanks!\" -> " + fallback_title + ". "
            "\"what's your name\" -> Asking My Name. \"explain photosynthesis\" -> Photosynthesis "
            "Explanation. A long pasted MCQ about a block sliding down a ramp with 4 options -> "
            "Kinetic Energy Calculation.\n\n"
            "Return ONLY the title text. Nothing else -- no preamble, no explanation, no quotes "
            "around it."
        ),
        messages=[{"role": "user", "content": message.text}]
    )
    await log_token_usage(message.user_id, response.usage.input_tokens + response.usage.output_tokens, ip)
    return {"title": _clean_title(response.content[0].text, fallback_title)}

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
        # Nobody is "unlimited" in the UI sense anymore -- Pro has always had a real, visible
        # budget (DAILY_TOKEN_LIMIT_PRO), and as of 2026-09-03 Max does too (MAX_DAILY_TOKEN_CAP).
        # Max's OWN remaining silent concept is the breakeven backstop (see check_max_usage_tier),
        # which still must never surface here as a number -- but that's a different, smaller,
        # backend-only number than the visible cap this endpoint now reports.
        budget_for_plan = DAILY_TOKEN_LIMIT_PRO if plan == "pro" else (MAX_DAILY_TOKEN_CAP if plan == "max" else DAILY_TOKEN_BUDGET_FREE)

        today, yesterday, rows_by_date = await _fetch_usage_rows("usage_log", "user_id", user_id)
        today_row = rows_by_date.get(today)
        tokens_used = today_row["tokens_used"] if today_row else 0

        # Free's rolling 24h-from-crossing cooldown only ever applies to Free -- Pro/Max's block
        # (see _check_plan_daily_limit) never sets limit_reached_at at all, it's a pure
        # midnight-IST reset, and checking this for Pro/Max risks picking up a stale
        # limit_reached_at from before a same-day plan upgrade. Skipped entirely for anything but
        # free.
        cooldown_remaining = _active_cooldown_seconds(today, yesterday, rows_by_date) if plan == "free" else None
        if cooldown_remaining is not None:
            reset_in_seconds = cooldown_remaining
        else:
            # No active block: "reset" just means the next IST midnight, when a fresh usage_date
            # row starts and today's count naturally reads back to 0. Also the ONLY reset model
            # Pro ever uses (see above).
            now_ist = datetime.now(timezone.utc).astimezone(IST)
            next_ist_midnight = datetime.combine(now_ist.date() + timedelta(days=1), datetime.min.time(), tzinfo=IST)
            reset_in_seconds = int((next_ist_midnight - now_ist).total_seconds())

        percent_used = round(min(100, tokens_used / budget_for_plan * 100), 1)
        budget = budget_for_plan
        blocked = tokens_used >= budget_for_plan

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
        weekly_budget = budget_for_plan * 7
        weekly_percent_used = round(min(100, weekly_tokens_used / weekly_budget * 100), 1)

        return {
            "plan": plan,
            "today": {
                "tokens_used": tokens_used, "budget": budget, "percent_used": percent_used,
                "doubts_today": doubts_today, "reset_in_seconds": max(0, reset_in_seconds), "unlimited": False,
                "blocked": blocked
            },
            "weekly": {
                "tokens_used": weekly_tokens_used, "budget": weekly_budget, "percent_used": weekly_percent_used,
                "total_doubts": weekly_doubts, "unlimited": False
            }
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
    results, related = await search_pyq(message.text)
    return {"pyqs": results, "related": related}

# Powers pricing.html's NEET-year selector (date-based annual pricing, pre-Razorpay -- see
# create_neet_exam_dates_table.sql). Public/no-auth, same reasoning as /diagrams: this is
# read-only catalog data every visitor (including guests) needs to see the pricing page at all.
# days_remaining computed server-side against real IST "today" so every client agrees on the same
# number regardless of its own clock/timezone; null for a TBD (exam_date is NULL) or already-past
# year rather than a negative/nonsensical count -- the frontend is expected to treat null as "not
# usable for date-based pricing yet" and fall back to flat annual pricing for that year.
@app.get("/neet-exam-dates")
async def get_neet_exam_dates(_: None = Depends(rate_limiter(30, 60))):
    try:
        response = await async_client.get(
            f"{SUPABASE_URL}/rest/v1/neet_exam_dates",
            headers={"apikey": SUPABASE_KEY, "Authorization": f"Bearer {SUPABASE_KEY}"},
            params={"select": "year,exam_date", "order": "year.asc"}
        )
        if response.status_code != 200:
            return {"years": []}
        today = date.fromisoformat(_ist_today())
        years = []
        for row in response.json():
            exam_date = row.get("exam_date")
            days_remaining = None
            if exam_date:
                delta = (date.fromisoformat(exam_date) - today).days
                if delta >= 0:
                    days_remaining = delta
            years.append({"year": row["year"], "exam_date": exam_date, "days_remaining": days_remaining})
        return {"years": years}
    except Exception:
        return {"years": []}

@app.post("/admin/set-neet-exam-date")
async def set_neet_exam_date(req: SetNeetExamDateRequest, _: None = Depends(verify_admin)):
    if req.exam_date:
        try:
            date.fromisoformat(req.exam_date)
        except ValueError:
            return {"error": "exam_date must be an ISO date string (YYYY-MM-DD) or null"}
    response = await async_client.post(
        f"{SUPABASE_URL}/rest/v1/neet_exam_dates",
        headers={**ADMIN_HEADERS, "Content-Type": "application/json", "Prefer": "resolution=merge-duplicates"},
        json={"year": req.year, "exam_date": req.exam_date, "updated_at": datetime.now(timezone.utc).isoformat()}
    )
    if response.status_code >= 300:
        return {"error": response.text}
    return {"success": True}

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

        async def _search(embedding):
            response = await async_client.post(
                f"{SUPABASE_URL}/rest/v1/rpc/match_diagrams",
                headers={"apikey": SUPABASE_KEY, "Authorization": f"Bearer {SUPABASE_KEY}", "Content-Type": "application/json"},
                json={"query_embedding": embedding, "match_threshold": DIAGRAM_MATCH_THRESHOLD, "match_count": 1, "filter_chapter": chapter}
            )
            return response.json() if response.status_code == 200 else []
        # Merge-variant fallback matters most here: diagram reference embeddings are built from
        # very short text (often just a bare name), which measured as the most typo/spacing-
        # fragile of the three matching systems this covers -- see _match_with_merge_fallback.
        rows = await _match_with_merge_fallback(req.text, get_embedding, _search)
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
        # select=* used to also pull each row's `embedding` vector (used only by match_pyq's
        # server-side similarity search, never by the mock-test frontend) -- that alone made
        # each of these 3 queries ~6x slower and ~30x heavier over the wire. Same explicit
        # column list already used by /mock-tests/{id}/questions, confirmed sufficient since
        # mocktest.html only ever reads these fields off a question object.
        cols = "id,subject,chapter,year,question,option_a,option_b,option_c,option_d,correct_answer,difficulty,diagram_url,option_a_diagram_url,option_b_diagram_url,option_c_diagram_url,option_d_diagram_url"
        # Also ran sequentially before (3 blocking sync `requests.get()` calls back-to-back,
        # each stalling the event loop) -- now fired concurrently via the shared async client.
        bio_resp, phy_resp, che_resp = await asyncio.gather(
            async_client.get(f"{SUPABASE_URL}/rest/v1/pyq", headers=headers, params={"subject": "eq.Biology", "select": cols, "limit": 200}),
            async_client.get(f"{SUPABASE_URL}/rest/v1/pyq", headers=headers, params={"subject": "eq.Physics", "select": cols, "limit": 200}),
            async_client.get(f"{SUPABASE_URL}/rest/v1/pyq", headers=headers, params={"subject": "eq.Chemistry", "select": cols, "limit": 200}),
        )
        bio, phy, che = bio_resp.json(), phy_resp.json(), che_resp.json()
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
        # A fixed "limit": 2000 here used to silently truncate and undercount once a subject grew
        # past it -- confirmed for real: Biology sits at 2087 active rows and Chemistry at 3385
        # (Physics, at 512, never showed the bug), so the old single-request version was quietly
        # dropping the last 87 Biology rows and over 1300 Chemistry rows from every chapter's
        # count. Paginated via the Range header instead so this counts the whole table regardless
        # of how large any subject grows.
        counts = {}
        page_size = 1000
        offset = 0
        while True:
            response = http_requests.get(
                f"{SUPABASE_URL}/rest/v1/pyq",
                headers={**headers, "Range": f"{offset}-{offset + page_size - 1}"},
                params={
                    "subject": f"eq.{subject}",
                    "is_active": "eq.true",
                    "chapter": "not.is.null",
                    "select": "chapter"
                }
            )
            rows = response.json()
            if not isinstance(rows, list) or not rows:
                break
            for r in rows:
                ch = r.get("chapter")
                if ch and ch.strip():
                    ch = ch.strip()
                    counts[ch] = counts.get(ch, 0) + 1
            if len(rows) < page_size:
                break
            offset += page_size
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

# Called only from admin_set_user_plan below, only on a student's genuine first-ever transition
# into a paid plan. Looks for a pending referral where this student is the referred party, and
# if found, credits both sides with the flat bonus and marks the referral completed. Returns a
# small dict describing what happened, or None if there was nothing to reward (the common case --
# most paid transitions won't have a pending referral at all).
async def _try_credit_referral_reward(referred_id: str):
    try:
        resp = await async_client.get(
            f"{SUPABASE_URL}/rest/v1/referrals", headers=ADMIN_HEADERS,
            params={"referred_id": f"eq.{referred_id}", "status": "eq.pending", "select": "*", "limit": 1}
        )
        rows = resp.json() if resp.status_code == 200 else []
        if not rows:
            return None
        referral = rows[0]
        referrer_id = referral["referrer_id"]
        now_iso = datetime.now(timezone.utc).isoformat()
        # No real payment/order id exists yet (see the comment above admin_set_user_plan) -- this
        # is a synthetic placeholder in the same shape a real Razorpay payment_id would eventually
        # fill this column with.
        purchase_id = f"admin-{referred_id}-{int(datetime.now(timezone.utc).timestamp())}"

        await _credit_bonus_tokens(referrer_id, REFERRAL_BONUS_TOKENS)
        await _credit_bonus_tokens(referred_id, REFERRAL_BONUS_TOKENS)

        await async_client.patch(
            f"{SUPABASE_URL}/rest/v1/referrals", headers=ADMIN_HEADERS,
            params={"id": f"eq.{referral['id']}"},
            json={"status": "completed", "reward_granted_at": now_iso, "subscription_purchase_id": purchase_id}
        )
        return {"referrer_id": referrer_id, "referred_id": referred_id, "bonus_each": REFERRAL_BONUS_TOKENS}
    except Exception as e:
        print(f"REFERRAL REWARD CREDIT ERROR: {e}", flush=True)
        return None

# Manual stand-in for what a Razorpay success webhook will do automatically later: flip
# `plan` to "pro"/"max" on payment, back to "free" on cancellation/expiry. For now, set by hand.
# Also the one place a referral reward can fire -- see the first_paid_at handling below -- so
# whatever eventually replaces this with a real webhook inherits referral crediting for free as
# long as it flows through this same function.
@app.post("/admin/set-user-plan")
async def admin_set_user_plan(req: SetUserPlanRequest, _: None = Depends(verify_admin)):
    if req.plan not in ("free", "pro", "max"):
        return {"error": "plan must be 'free', 'pro', or 'max'"}
    try:
        existing_resp = await async_client.get(
            f"{SUPABASE_URL}/rest/v1/user_plan", headers=ADMIN_HEADERS,
            params={"user_id": f"eq.{req.user_id}", "select": "plan,first_paid_at", "limit": 1}
        )
        existing_rows = existing_resp.json() if existing_resp.status_code == 200 else []
        existing = existing_rows[0] if existing_rows else None
        was_paid_before = bool(existing and existing.get("plan") in ("pro", "max"))
        already_had_first_paid_at = bool(existing and existing.get("first_paid_at"))
        # The ONLY condition that counts as a genuine first purchase: moving INTO a paid plan
        # from a non-paid state, AND this account has never been marked paid before. A student
        # set pro -> free -> pro again must not re-trigger a reward or re-set this timestamp --
        # first_paid_at is written exactly once per account, ever.
        is_first_paid_transition = req.plan in ("pro", "max") and not was_paid_before and not already_had_first_paid_at

        patch_body = {"user_id": req.user_id, "plan": req.plan, "updated_at": datetime.now(timezone.utc).isoformat()}
        if is_first_paid_transition:
            patch_body["first_paid_at"] = datetime.now(timezone.utc).isoformat()

        resp = await async_client.post(
            f"{SUPABASE_URL}/rest/v1/user_plan",
            headers={**ADMIN_HEADERS, "Content-Type": "application/json", "Prefer": "resolution=merge-duplicates"},
            json=patch_body
        )
        if resp.status_code >= 400:
            return {"error": resp.text}

        result = {"success": True}
        if is_first_paid_transition:
            referral_reward = await _try_credit_referral_reward(req.user_id)
            if referral_reward:
                result["referral_reward"] = referral_reward
        return result
    except Exception as e:
        return {"error": str(e)}

# ---------------- Referral program: student-facing endpoints ----------------
@app.get("/referral/my-code")
async def referral_my_code(user_id: str):
    if not user_id:
        return {"error": "user_id required"}
    return {"code": _referral_code_for_user(user_id)}

@app.post("/referral/register")
async def referral_register(req: RegisterReferralRequest):
    referrer_id = _user_id_from_referral_code(req.referral_code)
    if not referrer_id:
        return {"error": "Invalid referral code"}
    if referrer_id == req.referred_id:
        return {"error": "You can't refer yourself"}
    try:
        # referred_id is UNIQUE at the database level (see the migration) -- this insert is the
        # real enforcement that a student can only ever be tied to one referral code, ever. A
        # second attempt, even with a different code, always fails here with a conflict; it
        # never silently overwrites the first one.
        resp = await async_client.post(
            f"{SUPABASE_URL}/rest/v1/referrals",
            headers={**ADMIN_HEADERS, "Content-Type": "application/json", "Prefer": "return=representation"},
            json={"referrer_id": referrer_id, "referred_id": req.referred_id, "referral_code": req.referral_code}
        )
        if resp.status_code == 409:
            return {"error": "This account has already used a referral code"}
        if resp.status_code >= 400:
            return {"error": resp.text}
        return {"success": True}
    except Exception as e:
        return {"error": str(e)}

@app.get("/referral/status")
async def referral_status(user_id: str):
    if not user_id:
        return {"error": "user_id required"}
    try:
        referred_by_resp = await async_client.get(
            f"{SUPABASE_URL}/rest/v1/referrals", headers=ADMIN_HEADERS,
            params={"referred_id": f"eq.{user_id}", "select": "status,referrer_id", "limit": 1}
        )
        referred_by_rows = referred_by_resp.json() if referred_by_resp.status_code == 200 else []

        as_referrer_resp = await async_client.get(
            f"{SUPABASE_URL}/rest/v1/referrals", headers=ADMIN_HEADERS,
            params={"referrer_id": f"eq.{user_id}", "select": "status"}
        )
        as_referrer_rows = as_referrer_resp.json() if as_referrer_resp.status_code == 200 else []
        completed_referrals = sum(1 for r in as_referrer_rows if r["status"] == "completed")
        pending_referrals = sum(1 for r in as_referrer_rows if r["status"] == "pending")

        bonus_resp = await async_client.get(
            f"{SUPABASE_URL}/rest/v1/bonus_tokens", headers=ADMIN_HEADERS,
            params={"user_id": f"eq.{user_id}", "select": "balance", "limit": 1}
        )
        bonus_rows = bonus_resp.json() if bonus_resp.status_code == 200 else []
        bonus_balance = bonus_rows[0]["balance"] if bonus_rows else 0

        return {
            "code": _referral_code_for_user(user_id),
            "was_referred": bool(referred_by_rows),
            "referred_status": referred_by_rows[0]["status"] if referred_by_rows else None,
            "referrals_completed": completed_referrals,
            "referrals_pending": pending_referrals,
            "bonus_balance": bonus_balance,
        }
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
        #
        # limit:5000 here was never actually 5000 in practice -- PostgREST caps a single response
        # at 1000 rows regardless of the limit param (same constraint already fixed for
        # pyq-classifier-data below), and this query has no `order` at all, so which 1000 of
        # Chemistry's 2700+ tagged rows came back was arbitrary and could silently omit a chapter
        # whose PYQs are sparse. Paginated the same way as pyq-classifier-data now: exact total
        # via Prefer: count=exact on the first page, remaining pages fetched concurrently.
        page_size = 1000
        params_base = {"subject": f"eq.{subject}", "chapter": "not.is.null", "select": "chapter", "limit": page_size}

        async def fetch_page(offset, headers=ADMIN_HEADERS):
            return await async_client.get(
                f"{SUPABASE_URL}/rest/v1/pyq",
                headers=headers,
                params={**params_base, "offset": offset}
            )

        first_resp = await fetch_page(0, headers={**ADMIN_HEADERS, "Prefer": "count=exact"})
        rows = first_resp.json()
        content_range = first_resp.headers.get("content-range", "")
        total = int(content_range.split("/")[-1]) if "/" in content_range else len(rows)

        remaining_offsets = list(range(page_size, total, page_size))
        if remaining_offsets:
            remaining_responses = await asyncio.gather(*(fetch_page(o) for o in remaining_offsets))
            for resp in remaining_responses:
                rows.extend(resp.json())

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
        # Also paginated, not a single limit:3000 request: PostgREST caps a single response at
        # 1000 rows regardless of what limit a caller asks for, which was silently truncating this
        # for Chemistry (2305 active rows then, 2719 now) and Biology (2249) -- confirmed live,
        # both were quietly returning only 1000 rows to the frontend. That starved
        # buildClassifier()'s chapter suggestions and checkDuplicates' exact-match check of more
        # than half the real data for those two subjects, on every successful request, not just
        # an intermittent failure.
        #
        # Pages fetched CONCURRENTLY, not one-at-a-time: sequential pagination was measured live
        # at ~16s for Chemistry's 3 pages (2719 rows / 1000 per page) -- a connection held open
        # that long is exactly the kind of thing a real (especially mobile) network drops
        # mid-request, which surfaces to the admin as "could not reach the server" even though
        # the backend itself never errored. Prefer: count=exact on the first page gets the real
        # total from Content-Range (e.g. "0-999/2719"), so every remaining page's offset is known
        # upfront and fired in one asyncio.gather batch -- no guessing a page-count cap, no
        # wasted requests past the real end.
        page_size = 1000
        params_base = {
            "subject": f"eq.{subject}",
            "is_active": "eq.true",
            "select": "question,chapter,class",
            "order": "id.asc",
            "limit": page_size,
        }

        async def fetch_page(offset, headers=ADMIN_HEADERS):
            return await async_client.get(
                f"{SUPABASE_URL}/rest/v1/pyq",
                headers=headers,
                params={**params_base, "offset": offset}
            )

        first_resp = await fetch_page(0, headers={**ADMIN_HEADERS, "Prefer": "count=exact"})
        all_rows = first_resp.json()
        content_range = first_resp.headers.get("content-range", "")
        total = int(content_range.split("/")[-1]) if "/" in content_range else len(all_rows)

        remaining_offsets = list(range(page_size, total, page_size))
        if remaining_offsets:
            remaining_responses = await asyncio.gather(*(fetch_page(o) for o in remaining_offsets))
            for resp in remaining_responses:
                all_rows.extend(resp.json())

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