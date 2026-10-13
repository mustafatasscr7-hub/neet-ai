"""Scan the shared answer caches for visible self-correction ("Wait -- let me re-check...") and,
only when explicitly asked, repair them.

Run with: python scan_solution_cache.py            (report only -- reads, never writes)
          python scan_solution_cache.py --apply    (regenerate/delete the flagged rows)

Uses server.py's own _has_self_correction detector and _generate_solve_solution, so a
regenerated solution goes through exactly the prompt/model routing a live /solve would (Gemini
with the real diagram image when the PYQ has one), plus the same SELF_CORRECTION_RETRY_GUARD the
live background retry uses.

pyq_solution_cache: each flagged row is regenerated (up to 2 attempts, not billed to anyone) and
replaced in place only if the new solution is clean; its alternate_method is cleared because it
was generated from the old solution. A row that still comes back flagged is deleted instead (the
next live request regenerates it through the same gate) and listed for PYQ data review -- that
pattern almost always means a wrong answer key or badly extracted options, which no amount of
regenerating fixes.

Rows whose PYQ has a BLANK answer key are skipped by --apply unless --include-blank-key is given:
the 2026-10-13 live check found the model confidently wrong on exactly these (unununium answered
"8th period"; the 5-ohm circuit answered 0.2 A instead of 2.8 A), and a clean-but-wrong solution
passes the self-correction check. Fill in the answer key first, then --apply.

answer_cache: rows hold only a question hash, not the question text, so they can't be
regenerated here -- --apply deletes flagged rows and the next student asking that question gets a
fresh, gated generation.
"""
import argparse
import asyncio
import io
import os
import re
import sys

import httpx

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import server

H = {"apikey": server.SUPABASE_SERVICE_KEY, "Authorization": f"Bearer {server.SUPABASE_SERVICE_KEY}"}
URL = server.SUPABASE_URL

# Rough token estimates for the cost preview -- ~4 chars/token, Gemini bills ~258 tokens per image.
CHARS_PER_TOKEN = 4
GEMINI_TOKENS_PER_IMAGE = 258
MAX_ATTEMPTS = 2


async def fetch_all(table: str, select: str) -> list:
    rows, offset = [], 0
    while True:
        resp = await server.async_client.get(
            f"{URL}/rest/v1/{table}", headers={**H, "Range": f"{offset}-{offset + 999}"},
            params={"select": select}, timeout=120
        )
        batch = resp.json()
        if not isinstance(batch, list):
            raise RuntimeError(f"{table}: {batch}")
        rows += batch
        if len(batch) < 1000:
            return rows
        offset += 1000


def matched_snippet(text: str) -> str:
    for p in server._SELF_CORRECTION_PATTERNS:
        m = p.search(text or "")
        if m:
            start = max(0, m.start() - 50)
            return "..." + re.sub(r"\s+", " ", text[start:m.end() + 50]) + "..."
    return ""


def solve_request_for(pyq: dict, language: str) -> "server.SolveRequest":
    return server.SolveRequest(
        pyq_id=pyq["id"], question=pyq.get("question") or "",
        option_a=pyq.get("option_a") or "", option_b=pyq.get("option_b") or "",
        option_c=pyq.get("option_c") or "", option_d=pyq.get("option_d") or "",
        correct_answer=pyq.get("correct_answer") or "", language=language,
        diagram_url=pyq.get("diagram_url"),
        option_a_diagram_url=pyq.get("option_a_diagram_url"), option_b_diagram_url=pyq.get("option_b_diagram_url"),
        option_c_diagram_url=pyq.get("option_c_diagram_url"), option_d_diagram_url=pyq.get("option_d_diagram_url"),
    )


def estimate_cost(req, old_solution: str) -> float:
    """One attempt, priced at the most expensive model that path can hit (DeepSeek peak for text,
    Gemini Flash-Lite for diagrams) -- an upper-bound estimate, not a quote."""
    has_diagram = server._solve_has_diagram(req)
    system = server._solve_system_prompt(req.language, has_diagram) + server.SELF_CORRECTION_RETRY_GUARD
    in_tokens = (len(system) + len(server._solve_text_messages(req)[0]["content"])) // CHARS_PER_TOKEN
    out_tokens = max(len(old_solution) // CHARS_PER_TOKEN, 300)
    if has_diagram:
        images = sum(1 for u in (req.diagram_url, req.option_a_diagram_url, req.option_b_diagram_url,
                                 req.option_c_diagram_url, req.option_d_diagram_url) if u)
        return server._gemini_cost(in_tokens + images * GEMINI_TOKENS_PER_IMAGE, 0, out_tokens)
    return server._deepseek_cost(in_tokens, 0, out_tokens, True)


async def regenerate(req) -> "str | None":
    for _ in range(MAX_ATTEMPTS):
        solution = await server._generate_solve_solution(req, ip="", extra_system=server.SELF_CORRECTION_RETRY_GUARD, bill=False)
        if solution.strip() and not server._has_self_correction(solution):
            return solution
    return None


async def main(apply: bool, include_blank_key: bool):
    # Normally created by server.py's FastAPI lifespan, which doesn't run for a standalone script.
    server.async_client = httpx.AsyncClient()
    solve_rows = await fetch_all("pyq_solution_cache", "pyq_id,language,solution")
    flagged = [r for r in solve_rows if server._has_self_correction(r["solution"])]
    print(f"pyq_solution_cache: {len(flagged)} of {len(solve_rows)} rows contain self-correction text\n")

    plans, total_cost = [], 0.0
    for row in flagged:
        resp = await server.async_client.get(f"{URL}/rest/v1/pyq", headers=H, params={"id": f"eq.{row['pyq_id']}", "select": "*"})
        pyq = (resp.json() or [None])[0]
        if not pyq:
            print(f"  {row['pyq_id']} [{row['language']}]  PYQ row no longer exists -> would delete cache row")
            plans.append((row, None))
            continue
        req = solve_request_for(pyq, row["language"])
        cost = estimate_cost(req, row["solution"])
        total_cost += cost
        plans.append((row, req))
        print(f"  {row['pyq_id']} [{row['language']}]  {pyq.get('subject')} / {pyq.get('chapter')}")
        print(f"      answer key: {pyq.get('correct_answer') or '(blank -- --apply skips this row until it is filled in)'}   path: {'Gemini + diagram' if server._solve_has_diagram(req) else 'text (DeepSeek/Qwen)'}   est. ${cost:.4f}/attempt")
        print(f"      matched: {matched_snippet(row['solution'])}")
    if flagged:
        print(f"\n  Estimated regeneration cost: ${total_cost:.4f} (one attempt each), "
              f"${total_cost * MAX_ATTEMPTS:.4f} worst case ({MAX_ATTEMPTS} attempts each)")

    answer_rows = await fetch_all("answer_cache", "question_hash,answer")
    flagged_answers = [r for r in answer_rows if server._has_self_correction(r["answer"])]
    print(f"\nanswer_cache: {len(flagged_answers)} of {len(answer_rows)} rows contain self-correction text "
          f"(no question text stored -- --apply deletes these, no model cost)")
    for r in flagged_answers:
        print(f"  {r['question_hash'][:12]}  matched: {matched_snippet(r['answer'])}")

    if not apply:
        print("\nReport only -- nothing was changed. Re-run with --apply to regenerate/delete the rows above.")
        return

    print("\nApplying...")
    needs_review = []
    for row, req in plans:
        key = {"pyq_id": f"eq.{row['pyq_id']}", "language": f"eq.{row['language']}"}
        if req and not req.correct_answer.strip() and not include_blank_key:
            print(f"  SKIPPED   {row['pyq_id']} [{row['language']}] -- blank answer key, fill it in first")
            continue
        solution = await regenerate(req) if req else None
        if solution:
            await server._store_cached_pyq_solution(row["pyq_id"], row["language"], solution, reset_alternate=True)
            print(f"  REPLACED  {row['pyq_id']} [{row['language']}]")
        else:
            await server.async_client.delete(f"{URL}/rest/v1/pyq_solution_cache", headers=H, params=key)
            needs_review.append(row["pyq_id"])
            print(f"  DELETED   {row['pyq_id']} [{row['language']}] -- still flagged after {MAX_ATTEMPTS} attempts, review the PYQ data")
    for r in flagged_answers:
        await server.async_client.delete(f"{URL}/rest/v1/answer_cache", headers=H, params={"question_hash": f"eq.{r['question_hash']}"})
        print(f"  DELETED   answer_cache {r['question_hash'][:12]}")
    if needs_review:
        print("\nAdd to PYQ_DATA_REVIEW.md: " + ", ".join(needs_review))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--apply", action="store_true", help="regenerate/delete flagged rows (default: report only)")
    parser.add_argument("--include-blank-key", action="store_true", help="with --apply, also regenerate rows whose PYQ has no answer key")
    args = parser.parse_args()
    asyncio.run(main(args.apply, args.include_blank_key))
