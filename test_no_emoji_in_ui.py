"""Guards the 2026-11-07 emoji-removal pass: fails if any emoji/pictograph character is found in
this repo's HTML or JS files (which includes i18n.js). AI-generated answer text is handled
separately (server.py prompts + chat.html's/admin-dashboard.html's stripAnswerEmojiMarkers at
render time for old saved/cached answers) -- this test only covers literal characters sitting in
the UI source files themselves.

Run with: python test_no_emoji_in_ui.py
"""
import glob
import os
import re
import sys
import unicodedata

sys.stdout = __import__("io").TextIOWrapper(sys.stdout.buffer, encoding="utf-8")
ROOT = os.path.dirname(os.path.abspath(__file__))
EXTS = (".html", ".js")
EXCLUDE_DIRS = {"node_modules", ".git", "test-fixtures"}

# Broad net over the common emoji/pictograph/symbol blocks -- includes flags (regional indicator
# pairs), the variation selector (U+FE0F) that forces emoji presentation, the zero-width joiner
# used in combined emoji, and the misc-technical block (hourglass/stopwatch and friends) that a
# narrower "emoji" range misses.
EMOJI_RE = re.compile(
    "["
    "\U0001F000-\U0001FFFF"
    "\U00002300-\U000023FF"
    "\U00002460-\U000027BF"
    "\U00002B00-\U00002BFF"
    "\U0001F1E6-\U0001F1FF"
    "\U0000FE0F"
    "\U0000200D"
    "]"
)

# Intentional exceptions -- plain, monochrome, text-presentation symbols the 2026-11-07 pass
# explicitly kept (never colorful pictographs): star rating glyphs, a plain check/cross/x used as
# a correctness or remove-button glyph, a fullscreen-toggle glyph, disclosure-triangle chevrons,
# and a plain circled-i info glyph.
ALLOWED_NAMES = {
    "CHECK MARK", "MULTIPLICATION X", "BALLOT X", "BLACK STAR", "WHITE STAR",
    "SQUARE FOUR CORNERS", "BLACK RIGHT-POINTING SMALL TRIANGLE",
    "CIRCLED LATIN SMALL LETTER I", "BLACK UP-POINTING TRIANGLE", "BLACK DOWN-POINTING TRIANGLE",
}


def find_emoji():
    hits = []
    for ext in EXTS:
        for path in glob.glob(os.path.join(ROOT, "**", "*" + ext), recursive=True):
            rel = os.path.relpath(path, ROOT)
            if any(part in EXCLUDE_DIRS for part in rel.split(os.sep)):
                continue
            try:
                with open(path, encoding="utf-8") as f:
                    lines = f.readlines()
            except UnicodeDecodeError:
                continue
            for i, line in enumerate(lines, 1):
                for m in EMOJI_RE.finditer(line):
                    ch = m.group()
                    name = unicodedata.name(ch, "<no name>")
                    if name in ALLOWED_NAMES:
                        continue
                    hits.append((rel, i, ch, name, line.strip()[:140]))
    return hits


def main():
    hits = find_emoji()
    if hits:
        print(f"FAIL: {len(hits)} emoji character(s) found outside the allowed exceptions:")
        for rel, i, ch, name, snippet in hits:
            print(f"  {rel}:{i}  {ch!r} ({name})  {snippet}")
        sys.exit(1)
    print(f"PASS: no emoji found in any .html/.js file under {ROOT} (allowed exceptions: {sorted(ALLOWED_NAMES)})")


if __name__ == "__main__":
    main()
