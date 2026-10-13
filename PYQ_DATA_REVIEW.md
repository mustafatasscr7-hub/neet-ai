# PYQ data review

PYQ rows whose data (answer key or option text) looks wrong. These can't be fixed by regenerating a solution, because the model ends up arguing with the bad data. Fix the `pyq` row in the admin tools first. The next `/solve` for that PYQ then regenerates and caches a clean solution automatically.

Server logs flag new candidates with `SOLVE SELF-CORRECTION UNRESOLVED, NOT CACHED (review this PYQ's answer key/options)`.

| PYQ id | Subject / chapter | Problem | Cache status |
|---|---|---|---|
| `cbf4e2c3-948e-409c-afdd-381764447f7b` | Chemistry / Classification of Elements | **Blank answer key.** Unununium is element 111 (Rg): group 11, period 7, so it is a transition element. Suggested key: `c`. In one live run the text model answered "8th period" (B) 3 times out of 4. | Cached solution has visible self-correction. `scan_solution_cache.py --apply` skips it until the key is filled in. |
| `a994e1e1-2c82-48bb-99a8-439a14f35b95` | Chemistry / States of Matter | **Blank answer key.** In the V–T graph, P₁ is the steeper line, and slope = nR/P, so P₁ < P₂. Suggested key: `b`. The current cached solution says A, which is **wrong**. | Same as above. |
| `2f78ede0-0121-47b3-aa21-9ed5b36c56cc` | Physics / Current Electricity | **Blank answer key.** Taking the bottom-right node as 0 V, the top-right node is at 4 V, and the 6 V and 8 V cells put the top-left node at −10 V, so I(5 Ω) = 14/5 = 2.8 A. Suggested key: `b`, but please check the cell polarities against the source. Gemini answers 0.2 A (D) or 0.8 A (C) every time, and the cached solution says D. | Same as above. |
| `8edfce62-5212-4251-a86a-588c6488d64c` | Chemistry / Some Basic Concepts of Chemistry | Match-the-prefixes question (Micro, Mega, Giga, Femto). The standard answer is a-ii, b-iv, c-iii, d-i, but none of the 4 stored options matches it, and the stored answer key `d` (a-i, b-iii, c-iv, d-ii) gets every pair wrong. The options were probably extracted wrongly. Check against the source paper. | Cached solution deleted 2026-10-13. It argued with the answer key and concluded "correct option is not listed". |
