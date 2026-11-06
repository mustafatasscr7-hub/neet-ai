"""Side-by-side comparison of search_ncert (old, live) vs search_ncert_v2 (new, ncert_content_v2
-- see ncert_chunks_v2.sql and ncert_rechunk_v2.py) for one query. Meant to be run locally by an
admin against the service-role key (same credential server.py's own retrieval code uses) -- not
exposed as an HTTP endpoint.

Run with:
    python ncert_v2_compare.py "function of golgi apparatus"

Forces NCERT_V2_ENABLED on for this process only (env var set before importing server, not
written back to .env) so search_ncert_v2 runs regardless of the live flag's value. Requires
ncert_chunks_v2.sql to have been run and ncert_rechunk_v2.py --load to have populated
ncert_content_v2 -- otherwise the new side will just come back empty, not error.
"""
import asyncio
import io
import os
import sys

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")
os.environ["NCERT_V2_ENABLED"] = "true"
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import server


def _print_rows(label, rows):
    print(f"\n--- {label} ({len(rows)} results) ---")
    if not rows:
        print("  (none)")
    for i, r in enumerate(rows, 1):
        heading, page = r.get("heading"), r.get("page")
        extra = f" | heading={heading!r} page={page}" if heading or page else ""
        print(f"{i}. [{r.get('similarity', 0):.3f}] {r.get('subject')} class {r.get('class')} "
              f"| {r.get('chapter_name')}{extra}")
        print(f"   {(r.get('content') or '')[:220]}")


async def main(query: str):
    old_rows, new_rows = await asyncio.gather(
        server.search_ncert(query, limit=5),
        server.search_ncert_v2(query, limit=5),
    )
    print(f"Query: {query!r}")
    _print_rows("search_ncert (old, live)", old_rows)
    _print_rows("search_ncert_v2 (new, ncert_content_v2)", new_rows)


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print('Usage: python ncert_v2_compare.py "<query>"', file=sys.stderr)
        sys.exit(1)
    asyncio.run(main(" ".join(sys.argv[1:])))
