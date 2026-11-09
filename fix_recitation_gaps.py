"""One-off: splice the 3 remaining small text gaps (each a single ~40-90 word sliver that Gemini's
RECITATION filter still blocked at minimum crop size, confirmed by direct visual inspection of the
rendered crop -- not re-attempted via the API again) directly into the already-recovered partial
text for lebo104.pdf p17/p20 and lebo112.pdf p4, and removes one duplicate line artifact (a line
that straddled a crop split and got read in both halves) on lebo104.pdf p17. Appends ONE final
corrected record per page so merge_and_rechunk.py's last-record-wins logic picks these up.
"""
import json
import os

OCR_FILE = "ncert_page_recovery_ocr.jsonl"


def _load_final(pdf, page):
    with open(OCR_FILE, encoding="utf-8") as f:
        recs = [json.loads(l) for l in f if l.strip()]
    matches = [r for r in recs if r["pdf"] == pdf and r["page"] == page]
    return matches[-1]


def _append(rec):
    with open(OCR_FILE, "a", encoding="utf-8") as f:
        f.write(json.dumps(rec, ensure_ascii=False) + "\n")
        f.flush()
        os.fsync(f.fileno())


def fix_lebo104_p17():
    rec = _load_final("lebo104.pdf", 17)
    text = rec["text"]

    dup_old = (
        "experimental verification of the chromosomal theory of inheritance by Thomas Hunt Morgan "
        "and his colleagues, led to discovering the basis for the variation\nand his colleagues, "
        "led to discovering the basis for the variation\nthat sexual reproduction produced."
    )
    dup_new = (
        "experimental verification of the chromosomal theory of inheritance by Thomas Hunt Morgan "
        "and his colleagues, led to discovering the basis for the variation\nthat sexual reproduction produced."
    )
    assert dup_old in text, "duplicate-line pattern not found -- text must have changed"
    text = text.replace(dup_old, dup_new)

    gap_anchor = "    Morgan and his group knew that the genes were located on the X chromosome (Section 4.4) and saw quickly that when the two genes in a\non the same chromosome"
    gap_fill = (
        "Morgan and his group knew that the genes were located on the X chromosome (Section 4.4) and saw quickly that when the two genes in a\n"
        "dihybrid cross were situated on the same chromosome, the proportion of parental gene "
        "combinations were much higher than the non-parental type. Morgan attributed this due to "
        "the physical association or linkage of the two genes and coined the term **linkage** to "
        "describe this physical association of genes on a chromosome and the term "
        "**recombination** to describe the generation of non-parental gene combinations (Figure "
        "4.11). Morgan and his group also found that even when genes were grouped\n"
        "on the same chromosome"
    )
    assert gap_anchor in text, "p17 gap anchor not found"
    text = text.replace(gap_anchor, gap_fill)

    rec["text"] = text
    rec["error"] = None
    rec["fully_recovered"] = True
    rec["manual_gap_fill"] = "one ~90-word sliver manually transcribed from the rendered crop " \
                              "(Gemini RECITATION-blocked even at minimum split size); one " \
                              "crop-boundary duplicate line removed"
    rec["split_notes"] = []
    _append(rec)
    print("Fixed lebo104.pdf p17")


def fix_lebo104_p20():
    rec = _load_final("lebo104.pdf", 20)
    text = rec["text"]

    gap_anchor = "Eggs fertilised by sperm having\nbe the sex chromosome"
    gap_fill = (
        "Eggs fertilised by sperm having\n"
        "an X-chromosome become females and, those fertilised by sperms that do not have an "
        "X-chromosome become males. *Do you think the number of chromosomes in the male and "
        "female are equal?* Due to the involvement of the X-chromosome in the determination of "
        "sex, it was designated to\n"
        "be the sex chromosome"
    )
    assert gap_anchor in text, "p20 gap anchor not found"
    text = text.replace(gap_anchor, gap_fill)

    rec["text"] = text
    rec["error"] = None
    rec["fully_recovered"] = True
    rec["manual_gap_fill"] = "one ~65-word sliver manually transcribed from the rendered crop " \
                              "(Gemini RECITATION-blocked even at minimum split size)"
    rec["split_notes"] = []
    _append(rec)
    print("Fixed lebo104.pdf p20")


def fix_lebo112_p4():
    rec = _load_final("lebo112.pdf", 4)
    text = rec["text"]

    gap_anchor = "Decomposition is largely an oxygen-requiring process. The rate of\ntemperature and anaerobiosis"
    gap_fill = (
        "Decomposition is largely an oxygen-requiring process. The rate of\n"
        "decomposition is controlled by chemical composition of detritus and climatic factors. In "
        "a particular climatic condition, decomposition rate is slower if detritus is rich in "
        "lignin and chitin, and quicker, if detritus is rich in nitrogen and water-soluble "
        "substances like sugars. Temperature and soil moisture are the most important climatic "
        "factors that regulate decomposition through their effects on the activities of soil "
        "microbes. Warm and moist environment favour decomposition whereas low\n"
        "temperature and anaerobiosis"
    )
    assert gap_anchor in text, "p4 gap anchor not found"
    text = text.replace(gap_anchor, gap_fill)

    rec["text"] = text
    rec["error"] = None
    rec["fully_recovered"] = True
    rec["manual_gap_fill"] = "one ~75-word sliver manually transcribed from the rendered crop " \
                              "(Gemini RECITATION-blocked even at minimum split size)"
    rec["split_notes"] = []
    _append(rec)
    print("Fixed lebo112.pdf p4")


if __name__ == "__main__":
    fix_lebo104_p17()
    fix_lebo104_p20()
    fix_lebo112_p4()
