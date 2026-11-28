"""Guards the "Sourced from NCERT" footer line added under the existing
"NEET-AI can make mistakes." disclaimer in chat.html (web only, no mobile app).

Checks:
1. Every place chat.html renders the disclaimer line also renders the new
   source line right alongside it (the two static input-hint divs and the
   showWelcome() JS template rebuild).
2. The chat.disclaimerSource i18n key exists for every language chat.disclaimerMistakes
   is defined in, so no language silently falls back to the English string.

Run with: python test_chat_footer_source_line.py
"""
import os
import re
import sys

ROOT = os.path.dirname(os.path.abspath(__file__))


def extract_balanced_block(text, start_brace_index):
    """Given the index of a '{' in text, return the substring up to and
    including its matching '}'."""
    depth = 0
    for i in range(start_brace_index, len(text)):
        if text[i] == "{":
            depth += 1
        elif text[i] == "}":
            depth -= 1
            if depth == 0:
                return text[start_brace_index:i + 1]
    raise ValueError("unbalanced braces")


def check_chat_html():
    path = os.path.join(ROOT, "chat.html")
    with open(path, encoding="utf-8") as f:
        html = f.read()

    mistakes_attr = html.count('data-i18n="chat.disclaimerMistakes"')
    source_attr = html.count('data-i18n="chat.disclaimerSource"')
    mistakes_call = html.count("t('chat','disclaimerMistakes')")
    source_call = html.count("t('chat','disclaimerSource')")

    errors = []
    if mistakes_attr == 0:
        errors.append("chat.html: no static data-i18n=\"chat.disclaimerMistakes\" found")
    if mistakes_attr != source_attr:
        errors.append(
            f"chat.html: {mistakes_attr} static disclaimerMistakes div(s) but "
            f"{source_attr} disclaimerSource div(s) -- every render site must carry both lines"
        )
    if mistakes_call == 0:
        errors.append("chat.html: no t('chat','disclaimerMistakes') JS render call found")
    if mistakes_call != source_call:
        errors.append(
            f"chat.html: {mistakes_call} t(...'disclaimerMistakes') call(s) but "
            f"{source_call} t(...'disclaimerSource') call(s) -- JS rebuild must render both lines"
        )
    return errors


def check_i18n_js():
    path = os.path.join(ROOT, "i18n.js")
    with open(path, encoding="utf-8") as f:
        js = f.read()

    chat_key_match = re.search(r"\bchat:\s*\{", js)
    if not chat_key_match:
        return ["i18n.js: could not find the top-level 'chat:' section"]
    chat_block = extract_balanced_block(js, chat_key_match.end() - 1)

    lang_matches = list(re.finditer(r"\b([a-z]{2}):\s*\{", chat_block))
    if not lang_matches:
        return ["i18n.js: no language sub-objects found inside chat: {...}"]

    errors = []
    checked_langs = []
    for m in lang_matches:
        lang = m.group(1)
        lang_block = extract_balanced_block(chat_block, m.end() - 1)
        has_mistakes = "disclaimerMistakes:" in lang_block
        has_source = "disclaimerSource:" in lang_block
        if not has_mistakes:
            continue  # not a disclaimer-bearing language block
        checked_langs.append(lang)
        if not has_source:
            errors.append(f"i18n.js: chat.{lang} has disclaimerMistakes but is missing disclaimerSource")

    if not checked_langs:
        errors.append("i18n.js: no chat.<lang>.disclaimerMistakes key found anywhere to check against")

    return errors


def main():
    errors = check_chat_html() + check_i18n_js()
    if errors:
        print(f"FAIL: {len(errors)} issue(s) found:")
        for e in errors:
            print(f"  - {e}")
        sys.exit(1)
    print("PASS: both footer lines render at every site in chat.html, and chat.disclaimerSource exists for every language chat.disclaimerMistakes does")


if __name__ == "__main__":
    main()
