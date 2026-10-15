# Mock-test grading: investigation and secure path

Written 2026-10-15. Part 1 is what the code did before this change (the investigation). Part 2 is
the secure path that was added, and Part 3 is how the mobile app should switch over.

## Part 1: what happens today (before this change)

### Endpoints that hand out test questions

All four return `correct_answer` for every question, to anyone. Three need no login at all.

| Endpoint | Used for | Auth | Answer-revealing fields returned |
|---|---|---|---|
| `GET /mock-test-questions` | Random full mock (90 Bio / 45 Phy / 45 Chem from `pyq`) | none | `correct_answer` |
| `GET /mock-tests/{id}/questions` | Official uploaded mock test (`mock_test_questions`) | none | `correct_answer` |
| `POST /personalised-test-questions` | Personalised test by subject/chapters | none | `correct_answer` |
| `POST /personalised-catalog-start` | Personalised catalog test | none | `correct_answer`, plus `select=*` (whole pyq row incl. the 1536-float `embedding`, `source_pdf_filename`, `source_tag`, `reviewed`) |

There are no stored solutions or explanations: the `pyq` and `mock_test_questions` tables have
no such columns. Explanations are generated on demand by the AI `/solve` endpoint (cached in
`pyq_solution_cache`), which `scoreboard.html` calls after a test for wrong answers.

Direct database reads (anon key, which is embedded in every page):
- `pyq`: closed to the anon key (policy dropped 2026-09-20), confirmed live: returns `[]`.
- `mock_test_questions`: **anon/authenticated SELECT policy is still open for published tests and
  covers every column, including `correct_answer`.** Today it returns nothing only because there
  are 0 mock tests in the database. The moment one is published, any student could read its whole
  answer key with the page's own anon key, bypassing every endpoint.

### How grading and saving work today

All grading is client-side.

- `mocktest.html` (random + official): loads questions with answers, the timer runs only in the
  browser (`totalSeconds = 12000`, i.e. 3h20m; resume uses an end timestamp kept in localStorage),
  `submitTest()` scores in the browser (+4 / -1 / 0, `correct_answer` normalised from a-d or 1-4,
  anything else counts as skipped), stores the result in `sessionStorage.mockTestResults`, then
  goes to `mockresults.html`. In-progress state, *including every question's `correct_answer`*, is
  saved in localStorage.
- `mockresults.html` inserts the `mock_results` row itself from those client-computed numbers
  (every page load, so a reload inserts a duplicate).
- `personalised-test.html`: same idea, timer `max(300, n * 70)` seconds in the browser, scores
  in the browser (without the 1-4 normalisation, so a malformed key counts as wrong), inserts
  `mock_results` with `source: 'personalised'` itself, and shows results inline.
- Both pages insert into `saved_questions` with `correct_answer` (mid-test "Save" button, and
  automatic "Mistake Journal" saves of wrong answers after submit).
- `scoreboard.html` reads `mock_results` (incl. `question_details`) with the anon key + session.

`mock_results` row shape (what every client writes): `user_id, score (max 0), correct, wrong,
skipped, total, time_taken, subject_{biology,physics,chemistry}_score (each max 0 of 4c - w),
mock_test_id, question_details (trimmed per-question array with correct_answer, userAnswer,
status), source` (`'mock'` default, `'personalised'` for personalised tests).

`mock_results` RLS, probed live with a throwaway account (deleted afterwards): a student **can
insert their own row with any score** (a fake 999 was accepted), cannot update or delete rows,
cannot insert for another user.

### Who calls these

- Web: `mocktest.html`, `personalised-test.html`, `mockresults.html`, `scoreboard.html`.
- Mobile (`neet-ai-mobile`, read only): `src/lib/mockTestApi.ts` and `personalisedTestApi.ts`
  call all four endpoints above plus `/mock-tests/available` and `/personalised-catalog`; scoring
  is ported in `src/lib/mockTestScoring.ts`; `take.tsx` inserts `mock_results` and
  `saved_questions` directly. The app depends on `correct_answer` being in those responses.

### Can students see the answers?

Yes. Every answer key is in the question response, visible in the browser's network tab and in
localStorage while the test is running. The timer is browser-only; the score saved to the
scoreboard is whatever the browser computed.

## Part 2: the secure path (added 2026-10-15)

Web pages (`mocktest.html`, `personalised-test.html`) now use two new endpoints. Both require a
valid Supabase JWT (`Authorization: Bearer <access_token>`); the user id always comes from the
token. The four legacy endpoints above are unchanged for clients that don't opt in.

### `POST /mock-tests/attempts` (start)

Request, one of:

```json
{ "kind": "random" }
{ "kind": "official", "mock_test_id": 7 }
{ "kind": "personalised", "selections": [{ "subject": "Biology", "chapters": ["Cell Cycle and Cell Division"] }], "count": 20 }
{ "kind": "personalised_catalog", "subject": "Biology", "test_number": 1 }
```

The server picks the questions exactly as the matching legacy endpoint does, records an attempt
(user, ordered question ids, server start time, duration), and returns:

```
{
  "attempt_id": "uuid",
  "kind": "random",
  "mock_test_id": null,
  "questions": [ { id, subject, chapter, year, question, option_a, option_b, option_c, option_d,
                   difficulty, diagram_url, option_a_diagram_url, option_b_diagram_url,
                   option_c_diagram_url, option_d_diagram_url } ],
  "total": 180,
  "duration_seconds": 12000,
  "started_at": "2026-10-15T10:00:00+00:00",
  "deadline_at": "2026-10-15T13:20:00+00:00",
  "server_now": "2026-10-15T10:00:00.120+00:00",
  "grace_seconds": 60,
  "requested": 20, "available": 57,    (personalised only)
  "title": "..."                        (personalised_catalog only)
}
```

Questions pass through an allowlist (`MOCK_TEST_PUBLIC_FIELDS`), so no answer field, and no
future column, can reach the client. Question order and option order are exactly as the legacy
endpoints serve them. Duration: 12000 s (3h20m) for random/official; `max(300, 70 * n)` for
personalised, same as the pages' old timers. Show remaining time as `deadline_at - server_now`,
not from the device clock.

Errors: 401 no/invalid token; 400 bad `kind`, missing `mock_test_id` / subjects / `test_number`,
invalid subject; 404 unpublished/missing test or no matching questions; 422 malformed body;
429 rate limit (10 starts per user per 10 min, plus 20/min per IP); 502 database trouble.

### `POST /mock-tests/attempts/{attempt_id}/submit`

```json
{ "question_ids": ["<id>", "..."], "answers": { "<question id>": "a", "<another id>": "c" } }
```

`question_ids` must be the attempt's ids in the served order (409 otherwise: tampered, swapped,
reordered, extra or missing). `answers` is keyed by question id, values `a`-`d` (any case) or
empty; leave unanswered questions out. Anything else the client sends (score, start time) is
ignored.

The server grades against the stored keys (+4 correct, -1 wrong, 0 skipped; a blank, `EMPTY` or
otherwise unusable key counts as skipped, never wrong, the same rule the web and app used),
writes one `mock_results` row in exactly the existing shape (`source` = `personalised` for
personalised kinds, column default `mock` otherwise; `mock_test_id` for official tests), and
returns:

```
{
  "attempt_id": "uuid", "result_id": "uuid (= mock_results.id)", "kind": "random", "mock_test_id": null,
  "already_submitted": false, "late": false,
  "score": 11, "correct": 3, "wrong": 1, "skipped": 176, "total": 180, "time_taken": 5400,
  "subjects": { "Biology": { "correct": 1, "wrong": 0, "skipped": 89, "total": 90, "score": 4 }, "Physics": {...}, "Chemistry": {...} },
  "results": [ { ...question fields..., "correct_answer": "b", "userAnswer": "a", "status": "wrong" } ]
}
```

`score` is unclamped (as the pages always showed it); `mock_results.score` stores `max(0, score)`.
`results` items have exactly the `mock_results.question_details` item shape, in served order.

Rules:
- Only the attempt's owner can submit; anyone else gets 404 (no hint that the id exists).
- Idempotent: the first submit atomically claims the attempt (stores the answers, a late flag and
  a pre-assigned `result_id`); any later submit returns the saved result with
  `already_submitted: true` and ignores the answers sent that time. Two submits at the same moment
  write one row. If the server dies between the claim and the write, the next submit grades from
  the stored answers under the same `result_id`.
- Timer: a submit after `started_at + duration + MOCK_TEST_SUBMIT_GRACE_SECONDS` (env, default 60)
  is still graded and saved but flagged `late: true` (stored on the attempt, shown on the results
  page). Rejecting it would throw away a genuine attempt, e.g. a laptop that slept through the
  timer and submitted on wake.
- Errors: 401, 404 (not yours / unknown), 409 (question set mismatch, or still being graded), 400
  (answer for a question outside the test, or not a-d), 422 (malformed), 429 (30 submits per
  user per 10 min), 502 (database). Answers and question ids are never logged.

### Web pages

- `mocktest.html`: start goes through the endpoint above; the timer counts down to the server
  deadline; saved progress (localStorage) keeps the `attemptId` and holds no answer key, and old
  saved progress without an `attemptId` is discarded. Submit sends the answers keyed by id and
  shows the server's result; if it fails, the answers are kept and the student can retry or go
  back to the test. The mid-test "Save" button queues the question and writes it to
  `saved_questions` after grading (the answer key isn't known before then).
- `personalised-test.html`: the same for catalog and custom tests; its client-side
  `mock_results` insert was removed (the server writes the row).
- `mockresults.html`: skips its own `mock_results` insert when the result came from the server
  (`serverSaved`), so no duplicate rows (this also stops the duplicate-on-reload for these
  attempts); older client-graded results still save as before. Shows a note for late submits.
- `scoreboard.html`: unchanged; server rows have the same shape as client rows.
- Guests: tests now need an account on the web (a login prompt replaces the test), because the
  submit endpoint requires a JWT.
- Neither page showed per-question correctness during a test, so there was nothing to keep there.

### Database

`create_mock_test_attempts_table.sql`: the new `mock_test_attempts` table (RLS: students can read
only their own attempts; no client insert, update or delete), plus dropping the anon/authenticated
SELECT policy on `mock_test_questions` that exposed published official tests' `correct_answer`.
Every reader of that table now uses the service key (`/mock-tests/{id}/questions` was switched in
this change). `mock_results` needs no change.

## Part 3: moving the mobile app over (neet-ai-mobile)

The app keeps working unchanged against the legacy endpoints. To move it to server grading:

1. `src/lib/mockTestApi.ts` / `personalisedTestApi.ts`: replace `fetchRandomMockTestQuestions`,
   `fetchOfficialMockTestQuestions`, `fetchPersonalisedTestQuestions` and
   `startPersonalisedCatalogTest` with one `startAttempt(body)` that calls
   `POST /mock-tests/attempts` through `apiJsonAsUser` (it needs the bearer token), using the
   `kind` bodies shown in Part 2. Make `correct_answer` optional on `MockTestQuestion` (it is
   absent before submit).
2. Keep `attempt_id`, `deadline_at` and `server_now` in the attempt draft, and drive the timer from
   `deadline_at - server_now` instead of the device's own countdown.
3. `take.tsx` `handleSubmit`: call `POST /mock-tests/attempts/{attempt_id}/submit` with
   `{ question_ids: questions.map(q => String(q.id)), answers: { [String(q.id)]: letter } }`.
   Use the response's `results` / `subjects` / counts for the results screen. Take
   `scoreMockTest` out of the submit path and remove `saveMockResultWithRetry` (the server has
   already written `mock_results`). Retrying a failed submit is safe (idempotent), so keep a
   retry button.
4. Mid-test "Save question" and the "Mistake Journal" auto-save: write them from the submit
   response's `results`, which carry `correct_answer`.
5. Guests: require a login before starting a test, as the web now does.
6. Once a release with this change has replaced the old app versions in use, do the legacy
   cleanup in Part 4.

## Part 4: remaining risk, and when to remove the legacy path

- Until the app moves, the four legacy endpoints still return `correct_answer` to anyone who
  calls them, and `mock_results` still accepts a client-written row with any score (a fake 999
  was accepted in the probe). The web no longer uses either.
- The PYQ bank (`/pyq-browse`) returns `correct_answer` by design (it is a practice bank), and
  random and personalised tests draw from the same `pyq` table, so a determined student can still
  look an answer up there. The secure path stops answers being shipped inside the test and stops
  forged scores from the web; it can't stop a student searching the PYQ bank or asking the AI
  tutor during a test.
- 92% of active `pyq` rows (7,584 of 8,274, measured 2026-10-15) have an empty `correct_answer`,
  so roughly 60% of a random mock is always graded as skipped. That was true before this change
  and is unchanged by it; filling those keys (e.g. via `/ensure-correct-answer`) is a separate task.
- Removal: once the updated app has been out long enough that old versions are rare (suggested
  date: 2027-01-15, three months out), remove `correct_answer` from the four legacy endpoints (or
  remove the endpoints) and drop the client INSERT policy on `mock_results`.
