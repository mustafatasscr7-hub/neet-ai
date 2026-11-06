"""Regression test for /pyq-browse pagination (chapter-loading perf work).

Run with: python test_pyq_browse_pagination.py
Reuses FakeSupabase/pyq_row from test_pyq_review_gate.py (same offline fake-Supabase approach as
the other pyq test files) -- its fake PostgREST already understands order/limit/offset and
Prefer: count=exact, so this only has to exercise the real /pyq-browse endpoint against it.

Covers:
  * no limit/offset: response shape is unchanged (no total_count/has_more key) -- every existing
    caller (savedquestions, tests, chat, the app) must see exactly today's behaviour.
  * limit/offset: pages are gapless and duplicate-free across the whole chapter, in a stable order,
    and total_count/has_more are correct for the whole chapter and narrowed by year.
"""
import io
import os
import sys

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import httpx
import server
from fastapi.testclient import TestClient
from test_pyq_review_gate import CHAPTER, FakeSupabase, pyq_row

_passed = []
_failed = []


def check(label, condition, detail=""):
    if condition:
        _passed.append(label)
        print(f"  PASS: {label}")
    else:
        _failed.append(label)
        print(f"  FAIL: {label}  {detail}")


def reset_limits():
    server._rate_buckets.clear()
    server._rate_windows.clear()


def seed(fake, n=47):
    # Odd count and two years so page boundaries never line up with a round number or a single
    # year, and both the whole-chapter and year-narrowed total_count paths get exercised for real.
    # language="en" is required -- PYQ_STUDENT_VISIBLE now filters on it (Hindi PYQ rollout) and
    # pyq_row's own default fixture predates that column, so every row needs it set explicitly here.
    for i in range(n):
        year = 2020 if i % 3 else 2021
        fake.tables["pyq"].append(pyq_row(subject="Biology", chapter=CHAPTER, year=year, language="en"))
    # A different chapter/subject, and a hidden row in the same chapter -- neither may leak into
    # this chapter's pages or its total_count.
    fake.tables["pyq"].append(pyq_row(subject="Biology", chapter="Human Health and Disease", language="en"))
    fake.tables["pyq"].append(pyq_row("pending", subject="Biology", chapter=CHAPTER, uploaded_by="contractor", language="en"))
    return n


def test_default_shape_unchanged(client):
    print("\n# No limit/offset: response shape is byte-for-byte what every existing caller gets today")
    reset_limits()
    r = client.get("/pyq-browse", params={"subject": "Biology", "chapter": CHAPTER})
    body = r.json()
    check("200 OK", r.status_code == 200, r.status_code)
    check("only the 'questions' key is present", set(body.keys()) == {"questions"}, body.keys())
    check("every approved+active row for the chapter came back", len(body["questions"]) == 47, len(body["questions"]))


def test_pagination_gapless(client, n):
    print("\n# limit/offset: gapless, duplicate-free pages covering the whole chapter")
    reset_limits()
    page_size = 7  # doesn't divide 47 evenly -- forces a short last page
    seen_ids = []
    offset = 0
    last_total = None
    pages = 0
    while True:
        r = client.get("/pyq-browse", params={"subject": "Biology", "chapter": CHAPTER, "limit": page_size, "offset": offset})
        body = r.json()
        pages += 1
        if pages > 20:
            check("pagination terminates", False, "ran away past 20 pages")
            break
        ids = [q["id"] for q in body["questions"]]
        seen_ids.extend(ids)
        last_total = body.get("total_count")
        expect_has_more = offset + len(ids) < n
        check(f"page at offset {offset} reports has_more={expect_has_more}", body.get("has_more") == expect_has_more, body)
        if not body.get("has_more"):
            break
        offset += len(ids)
    check("total_count is the whole chapter's approved+active count", last_total == n, last_total)
    check("no duplicate ids across pages", len(seen_ids) == len(set(seen_ids)), len(seen_ids) - len(set(seen_ids)))
    check("every row was returned exactly once (no gaps)", len(seen_ids) == n, (len(seen_ids), n))

    reset_limits()
    r = client.get("/pyq-browse", params={"subject": "Biology", "chapter": CHAPTER})
    full_ids = [q["id"] for q in r.json()["questions"]]
    check("paginated ids are the same set as the unpaginated call", set(seen_ids) == set(full_ids), None)


def test_year_filter_across_pages(client):
    print("\n# year filter narrows total_count and pages correctly")
    reset_limits()
    r = client.get("/pyq-browse", params={"subject": "Biology", "chapter": CHAPTER, "year": 2021, "limit": 5, "offset": 0})
    body = r.json()
    reset_limits()
    unfiltered = client.get("/pyq-browse", params={"subject": "Biology", "chapter": CHAPTER}).json()["questions"]
    expected_2021 = sum(1 for q in unfiltered if q["year"] == 2021)
    check("year=2021 total_count matches the real count of 2021 rows in the chapter",
          body.get("total_count") == expected_2021, (body.get("total_count"), expected_2021))
    check("every row on the page is actually year 2021", all(q["year"] == 2021 for q in body["questions"]), body["questions"])


def test_stable_order(client):
    print("\n# order is stable across repeated page fetches (no reshuffle between requests)")
    reset_limits()
    r1 = client.get("/pyq-browse", params={"subject": "Biology", "chapter": CHAPTER, "limit": 10, "offset": 10}).json()
    reset_limits()
    r2 = client.get("/pyq-browse", params={"subject": "Biology", "chapter": CHAPTER, "limit": 10, "offset": 10}).json()
    check("the same offset/limit returns the same ids in the same order",
          [q["id"] for q in r1["questions"]] == [q["id"] for q in r2["questions"]], (r1["questions"], r2["questions"]))


def test_bad_params(client):
    print("\n# limit/offset validation")
    reset_limits()
    r = client.get("/pyq-browse", params={"subject": "Biology", "chapter": CHAPTER, "limit": 0})
    check("limit=0 is refused", "error" in r.json(), r.json())
    reset_limits()
    r = client.get("/pyq-browse", params={"subject": "Biology", "chapter": CHAPTER, "limit": 500})
    check("limit over the max is refused", "error" in r.json(), r.json())
    reset_limits()
    r = client.get("/pyq-browse", params={"subject": "Biology", "chapter": CHAPTER, "limit": 5, "offset": -1})
    check("negative offset is refused", "error" in r.json(), r.json())


def main():
    fake = FakeSupabase()
    n = seed(fake)
    server.async_client = httpx.AsyncClient(transport=httpx.MockTransport(fake.handle))
    client = TestClient(server.app)

    test_default_shape_unchanged(client)
    test_pagination_gapless(client, n)
    test_year_filter_across_pages(client)
    test_stable_order(client)
    test_bad_params(client)

    print(f"\n{len(_passed)} passed, {len(_failed)} failed")
    if _failed:
        sys.exit(1)


if __name__ == "__main__":
    main()
