// Regression test for the chat.html half of the NCERT "Chapter:" line / "From NCERT" quote card
// citation-conflict fix (see server.py's NCERT_OLD_CHAPTER_MIN_SCORE + show_old_chapter for the
// backend half -- test_ncert_old_chapter_confidence.py covers that side). Run with:
//   node test-chat-ncert-citation-conflict.js
//
// Same no-framework convention as test-paste-parser.js: loads the real functions straight out of
// chat.html via Node's `vm` module and asserts against them directly, rather than a hand-copied
// duplicate of the logic that could silently drift out of sync with the real file. Two separate
// source blocks are extracted (extractAnswerMeta/renderNcertSectionsCardHtml/getActiveNcertQuotes
// are contiguous; escapeHtml lives elsewhere in the file) and concatenated into one sandboxed
// function body before being run -- order between the two doesn't matter, since plain `function`
// declarations are hoisted within that shared scope either way.
'use strict';
const fs = require('fs');
const path = require('path');
const vm = require('vm');
const assert = require('assert');

const html = fs.readFileSync(path.join(__dirname, 'chat.html'), 'utf8');

function extractBlock(startMarker, endMarker) {
  const startIdx = html.indexOf(startMarker);
  const endIdx = html.indexOf(endMarker);
  assert(startIdx !== -1 && endIdx !== -1 && endIdx > startIdx,
    `could not locate block in chat.html -- did a surrounding comment move? start=${startIdx} end=${endIdx}`);
  return html.slice(startIdx, endIdx);
}

// extractAnswerMeta, renderNcertSectionsCardHtml, getActiveNcertQuotes (plus their small internal
// helpers stripAnswerEmojiMarkers/renderInlineMathText/LEGACY_ANSWER_MARKER_RE/buildPyqSearchQuery)
const mainBlock = extractBlock(
  '// For short inline labels (like the chapter pill below) that may contain $...$ math --',
  '// Parses the CLARIFY_TYPE/QUESTION/OPTION lines that follow an AMBIGUOUS: yes first line'
);
// escapeHtml lives elsewhere in the file (used by many unrelated features too), so it's a
// separate block.
const escapeHtmlBlock = extractBlock(
  '    function escapeHtml(str) {',
  '    // ---------------- Match List-I / List-II formatting ----------------'
);

const exportNames = ['extractAnswerMeta', 'renderNcertSectionsCardHtml', 'getActiveNcertQuotes', 'escapeHtml'];
const sandbox = {
  // Minimal createElement('div') stub for escapeHtml's textContent->innerHTML round trip --
  // mirrors real browser/DOM behavior for the three entities a text node's innerHTML actually
  // escapes (&, <, >); none of this test's fixtures need attribute-context quote-escaping fidelity.
  document: {
    createElement() {
      let text = '';
      return {
        set textContent(v) { text = v == null ? '' : String(v); },
        get textContent() { return text; },
        get innerHTML() { return text.replace(/&/g, '&amp;').replace(/</g, '&lt;').replace(/>/g, '&gt;'); }
      };
    }
  }
};
vm.createContext(sandbox);
const wrapped = `(function() { ${escapeHtmlBlock}\n${mainBlock}\nreturn { ${exportNames.join(', ')} }; })()`;
const mod = vm.runInContext(wrapped, sandbox, { filename: 'chat.html (extracted NCERT-citation block)' });

let passed = 0, failed = 0;
function test(name, fn) {
  try {
    fn();
    passed++;
    console.log(`  ok - ${name}`);
  } catch (e) {
    failed++;
    console.log(`  FAIL - ${name}`);
    console.log('    ' + e.message);
  }
}

const SAMPLE_ANSWER_TEXT =
  'VISUAL_INTENT: no\n\nNEET Importance: 4/5\n\nChapter: NCERT Class 11, Chapter 8 — Cell: The Unit of Life\n\n' +
  'Answer:\nThe cell is the basic structural and functional unit of life.\n\nKey Points:\n- Point 1';

const SAMPLE_QUOTES = [
  { quote: 'The cell is the basic structural and functional unit of all living organisms.', citation: 'Cell: The Unit of Life, Cell Theory' }
];

console.log('extractAnswerMeta: chapter pill vs. From NCERT card (rule 1 -- never both)');

test('no card (suppressChapter=false, today\'s behavior): chapter pill IS rendered', () => {
  const { metaHtml, chapterText } = mod.extractAnswerMeta(SAMPLE_ANSWER_TEXT, false);
  assert(metaHtml.includes('answer-chapter-pill'), 'expected the chapter pill to render: ' + metaHtml);
  assert.strictEqual(chapterText, 'Cell: The Unit of Life');
});

test('card shown (suppressChapter=true): chapter pill is dropped, not replaced with mismatched text', () => {
  const { metaHtml, chapterText } = mod.extractAnswerMeta(SAMPLE_ANSWER_TEXT, true);
  assert(!metaHtml.includes('answer-chapter-pill'), 'chapter pill should have been suppressed: ' + metaHtml);
  // chapterText itself (PYQ-search-query sharpening, never shown to the student on its own) is
  // still computed -- suppression only affects the visible pill.
  assert.strictEqual(chapterText, 'Cell: The Unit of Life');
});

test('card shown: the stars row (unrelated to chapter) is unaffected', () => {
  const { metaHtml } = mod.extractAnswerMeta(SAMPLE_ANSWER_TEXT, true);
  assert(metaHtml.includes('answer-stars'), 'NEET Importance stars should still render: ' + metaHtml);
});

test('card shown + rendered together: the combined bubble HTML never carries two chapter references', () => {
  const { metaHtml, cleanText } = mod.extractAnswerMeta(SAMPLE_ANSWER_TEXT, true);
  const cardHtml = mod.renderNcertSectionsCardHtml(SAMPLE_QUOTES);
  const combined = metaHtml + cleanText + cardHtml;
  assert(combined.includes('ncert-sections-card'), 'card should be present: ' + combined);
  assert(!combined.includes('answer-chapter-pill'), 'pill must not also be present alongside the card: ' + combined);
  // The raw "Chapter: ..." line itself is always stripped into cleanText regardless of
  // suppression (pre-existing behavior, unrelated to this fix) -- never shown as inline prose.
  assert(!cleanText.includes('Chapter:'), 'raw Chapter: line must never leak into the answer body: ' + cleanText);
});

test('no quote card available at all (Gemini/Free -- rule 3): behaves exactly like no card', () => {
  const noNcertQuotes = null; // Gemini (images/pdf) and Free-plan requests never produce this
  const noCard = mod.extractAnswerMeta(SAMPLE_ANSWER_TEXT, !!(noNcertQuotes && noNcertQuotes.length));
  const todayDefault = mod.extractAnswerMeta(SAMPLE_ANSWER_TEXT, false);
  assert.strictEqual(noCard.metaHtml, todayDefault.metaHtml, 'a falsy/absent quotes array must suppress nothing, same as before this fix');
});

console.log('\nrenderNcertSectionsCardHtml');

test('empty/missing quotes list renders nothing', () => {
  assert.strictEqual(mod.renderNcertSectionsCardHtml(null), '');
  assert.strictEqual(mod.renderNcertSectionsCardHtml([]), '');
  assert.strictEqual(mod.renderNcertSectionsCardHtml(undefined), '');
});

test('renders the quote and its citation, HTML-escaped', () => {
  const htmlOut = mod.renderNcertSectionsCardHtml([{ quote: 'a <script>alert(1)</script> quote', citation: 'Ch & Heading' }]);
  assert(htmlOut.includes('ncert-sections-card-quote'));
  assert(!htmlOut.includes('<script>'), 'quote text must be HTML-escaped: ' + htmlOut);
  assert(htmlOut.includes('&lt;script&gt;'));
  assert(htmlOut.includes('Ch &amp; Heading'));
});

console.log('\ngetActiveNcertQuotes (history reload / version-switch consistency -- rule 4)');

test('a message with versions[] returns the ACTIVE version\'s own quotes, not another version\'s', () => {
  const msg = {
    versions: [
      { text: 'v0', ncertQuotes: [{ quote: 'v0 quote here is six words long', citation: 'A' }] },
      { text: 'v1', ncertQuotes: null },
      { text: 'v2', ncertQuotes: [{ quote: 'v2 quote here is six words long', citation: 'B' }] }
    ],
    activeVersionIndex: 2
  };
  const q = mod.getActiveNcertQuotes(msg);
  assert(q && q.length === 1 && q[0].citation === 'B', 'expected version 2\'s own quotes, got: ' + JSON.stringify(q));
});

test('a version with no card (quotes null) returns null, not an empty-but-truthy value', () => {
  const msg = { versions: [{ text: 'v0', ncertQuotes: null }], activeVersionIndex: 0 };
  assert.strictEqual(mod.getActiveNcertQuotes(msg), null);
});

test('a pre-version-history message (no .versions at all) falls back to the top-level field (or null)', () => {
  assert.strictEqual(mod.getActiveNcertQuotes({ text: 'old message, saved before versions shipped' }), null);
  assert.strictEqual(mod.getActiveNcertQuotes(null), null);
  assert.strictEqual(mod.getActiveNcertQuotes(undefined), null);
});

test('reload render built from getActiveNcertQuotes matches a fresh live render for the same version', () => {
  const activeQuotes = SAMPLE_QUOTES;
  const msg = { versions: [{ text: SAMPLE_ANSWER_TEXT, ncertQuotes: activeQuotes }], activeVersionIndex: 0 };
  // "Reload" path: derive suppression from the saved message the way addMessageRow now does.
  const reloadMeta = mod.extractAnswerMeta(SAMPLE_ANSWER_TEXT, !!(mod.getActiveNcertQuotes(msg) || []).length);
  const reloadCard = mod.renderNcertSectionsCardHtml(mod.getActiveNcertQuotes(msg));
  // "Live" path: the trailer decoded the same quotes the moment generation finished.
  const liveMeta = mod.extractAnswerMeta(SAMPLE_ANSWER_TEXT, !!(activeQuotes && activeQuotes.length));
  const liveCard = mod.renderNcertSectionsCardHtml(activeQuotes);
  assert.strictEqual(reloadMeta.metaHtml, liveMeta.metaHtml, 'a reload must suppress the chapter pill exactly like the live render did');
  assert.strictEqual(reloadCard, liveCard, 'a reload must show the identical card the live render showed');
  assert(!reloadMeta.metaHtml.includes('answer-chapter-pill') && reloadCard.includes('ncert-sections-card'),
    'the reloaded message must show the card, not a conflicting chapter pill');
});

console.log(`\n${passed} passed, ${failed} failed`);
process.exit(failed > 0 ? 1 : 0);
