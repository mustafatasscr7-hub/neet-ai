"""Known NCERT textbook errors, Hindi numerical labels, and the flip-pair review scan.

Run with: python test_hindi_content_fixes.py
Plain pass/fail harness like test_cost_fixes.py. Fully offline: the Supabase RPC and the embedding
are fakes.

1. NCERT_ERRATA corrects retrieved chunk text (Hindi Henle/Krebs/pH, English NAD/NADH), only in the
   chunk's own language, without touching the row it was given.
2. search_ncert hands the corrected text on.
3. Hindi answers get fixed Hindi labels for Given / Formula / Solution / Final Answer.
4. scan_ncert_flip_pairs: whole-word matching, both-sides sentences, ranking, errata chunks skipped.
"""
import asyncio
import io
import os
import sys

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import server
import scan_ncert_flip_pairs as scan

_passed = []
_failed = []


def check(label, condition, detail=""):
    if condition:
        _passed.append(label)
        print(f"  PASS: {label}")
    else:
        _failed.append(label)
        print(f"  FAIL: {label}" + (f" -- {detail}" if detail else ""))


HENLE_WRONG = ("यह भाग मध्यांश में उच्च अंतराकाशी तरल की परासरणता के नियमन में महत्वपूर्ण भूमिका निभाता है। "
               "हेन्ले-लूप की अवरोही भुजा जल के लिए अपारगम्य होती है, परंतु वैद्युत अपघट्य के सक्रियता से या धीरे-धीरे पारगम्य होती है। "
               "यह नीचे की ओर जाते हुए निष्यंद को सांद्र करती है।")
NAD_WRONG = "electrons are excited and transferred through PS II and PS I and finally to NAD forming NADH. During this process"


def test_errata():
    print("\n1. NCERT_ERRATA")
    hi = {"id": "h", "language": "hi", "content": HENLE_WRONG}
    en = {"id": "e", "language": "en", "content": NAD_WRONG}
    other = {"id": "o", "language": "en", "content": "Krebs cycle forms NADH and FADH2."}
    fixed = server._with_ncert_errata([hi, en, other])
    check("Hindi descending limb: permeable to water",
          "अवरोही भुजा जल के लिए पारगम्य होती है, परंतु वैद्युत अपघट्य के लिए लगभग अपारगम्य होती है।" in fixed[0]["content"], fixed[0]["content"])
    check("rest of the Hindi chunk kept", fixed[0]["content"].startswith("यह भाग") and fixed[0]["content"].endswith("सांद्र करती है।"))
    check("English: NADP+ forming NADPH", "finally to NADP+ forming NADPH." in fixed[1]["content"] and "NAD forming" not in fixed[1]["content"], fixed[1]["content"])
    check("unrelated chunk is the same object", fixed[2] is other)
    check("input rows not mutated", hi["content"] == HENLE_WRONG and en["content"] == NAD_WRONG)
    krebs = {"language": "hi", "content": r"क्रेब्स चक्र में $\mathrm{NADPH} + \mathrm{H}^{+}$तथा $\mathrm{NADH}_{2}$ बनते हैं।"}
    ph = {"language": "hi", "content": r"इस प्रकार, अम्लीय विलयन की $\text{pH} < 7$ क्षारीय विलयन की $\text{pH} < 7$ उदासीन विलयन की $\text{pH} = 7$"}
    krebs_fixed, ph_fixed = (r["content"] for r in server._with_ncert_errata([krebs, ph]))
    check("Krebs cycle: NADH + H+ and FADH2",
          r"$\mathrm{NADH} + \mathrm{H}^{+}$ तथा $\mathrm{FADH}_{2}$" in krebs_fixed and "NADPH" not in krebs_fixed, krebs_fixed)
    check("basic solution pH > 7, acidic stays < 7",
          r"क्षारीय विलयन की $\text{pH} > 7$" in ph_fixed and r"अम्लीय विलयन की $\text{pH} < 7$" in ph_fixed, ph_fixed)
    check("entry only applies in its own language",
          server._with_ncert_errata([{"language": "hi", "content": NAD_WRONG}])[0]["content"] == NAD_WRONG)


class FakeResp:
    def __init__(self, body):
        self.status_code = 200
        self._body = body

    def json(self):
        return self._body


def test_search_ncert_applies_errata():
    print("\n2. search_ncert returns corrected text")
    row = {"id": "e", "subject": "Biology", "class": 11, "chapter_name": "Photosynthesis in Higher Plants",
           "chapter_name_en": None, "language": "en", "content": NAD_WRONG + " a proton gradient is created.", "similarity": 0.7}

    class FakeClient:
        async def post(self, url, headers=None, json=None):
            return FakeResp([row])

    async def fake_embed(text):
        return [0.0] * 1536

    saved = server.async_client, server.get_embedding
    server.async_client, server.get_embedding = FakeClient(), fake_embed
    try:
        rows = asyncio.run(server.search_ncert("what happens in the light reaction of photosynthesis"))
    finally:
        server.async_client, server.get_embedding = saved
    check("corrected chunk returned", len(rows) == 1 and "forming NADPH" in rows[0]["content"], rows)


def test_hindi_numerical_labels():
    print("\n3. Hindi labels for the numerical template")
    ctx = server.HINDI_LANG_CONTEXT
    for en, hi in [("Given:", "दिया गया है:"), ("Formula:", "सूत्र:"), ("Solution:", "हल:"), ("Final Answer:", "अंतिम उत्तर:")]:
        check(f'"{en}" -> "{hi}"', f'"{en}" becomes "{hi}"' in ctx)
    check("earlier label mappings kept", '"Answer:" becomes "उत्तर:"' in ctx and '"Key Points:" becomes "मुख्य बिंदु:"' in ctx)
    check("English template itself unchanged", "Given:\n" in server.SYSTEM_PROMPT and "Final Answer: Maximum height" in server.SYSTEM_PROMPT)


def test_flip_scan():
    print("\n4. scan_ncert_flip_pairs")
    terms = scan.flip_terms("अर्धपारगम्य नहीं, अपारगम्य झिल्ली; पारगम्यता; अपचयन तथा अपचय")
    check("अपारगम्य is its own word, inflections count",
          [t for _, t in terms] == ["अपारगम्य", "पारगम्य", "अपचयन", "अपचय"], terms)
    check("अपचयन is oxidation/reduction, अपचय is catabolism",
          [n for n, _ in terms][2:] == ["oxidation/reduction", "anabolism/catabolism"], terms)

    both = scan.scan_chunk("सेलुलोस ऐसीटेट जल के लिए पारगम्य है परंतु आयनों के लिए अपारगम्य है। अगला वाक्य।")
    check("sentence naming both sides is flagged", [n for n, _ in both["both"]] == ["permeable/impermeable"], both)

    rows = [
        {"id": "one-side", "content": "यह झिल्ली पारगम्य है।"},
        {"id": "sign", "content": "धनात्मक तथा ऋणात्मक दोनों।"},
        {"id": "perm", "content": "जल के लिए पारगम्य परंतु आयनों के लिए अपारगम्य।"},
        {"id": "errata", "content": HENLE_WRONG},
        {"id": "none", "content": "कोई युग्म नहीं।"},
    ]
    ranked = [r["id"] for r, _ in scan.rank(rows, server.NCERT_ERRATA, 20)]
    check("permeability pair first, one-sided last, errata/no-pair skipped", ranked == ["perm", "sign", "one-side"], ranked)
    check("max respected", len(scan.rank(rows, server.NCERT_ERRATA, 2)) == 2)


if __name__ == "__main__":
    test_errata()
    test_search_ncert_applies_errata()
    test_hindi_numerical_labels()
    test_flip_scan()
    print(f"\n{len(_passed)} passed, {len(_failed)} failed")
    sys.exit(1 if _failed else 0)
