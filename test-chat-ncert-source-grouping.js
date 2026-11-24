// Regression test for grouping "From NCERT" quotes by section instead of repeating the citation
// after every quote (renderNcertSectionsCardHtml in chat.html). Run with:
//   node test-chat-ncert-source-grouping.js
//
// Same no-framework convention as test-chat-ncert-citation-conflict.js: loads the real function
// straight out of chat.html via Node's `vm` module rather than a hand-copied duplicate.
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

const mainBlock = extractBlock(
  '// For short inline labels (like the chapter pill below) that may contain $...$ math --',
  '// Parses the CLARIFY_TYPE/QUESTION/OPTION lines that follow an AMBIGUOUS: yes first line'
);
const escapeHtmlBlock = extractBlock(
  '    function escapeHtml(str) {',
  '    // ---------------- Match List-I / List-II formatting ----------------'
);

const exportNames = ['renderNcertSectionsCardHtml', 'getActiveNcertQuotes'];
const sandbox = {
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
const mod = vm.runInContext(wrapped, sandbox, { filename: 'chat.html (extracted NCERT-grouping block)' });

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

function countOccurrences(haystack, needle) {
  return haystack.split(needle).length - 1;
}

console.log('renderNcertSectionsCardHtml: source grouping');

test('a single section shows its citation line once, not after every quote', () => {
  const quotes = [
    { quote: 'First quote from the section', citation: 'Biological Classification - 2.2.5 Protozoans' },
    { quote: 'Second quote from the same section', citation: 'Biological Classification - 2.2.5 Protozoans' },
    { quote: 'Third quote from the same section', citation: 'Biological Classification - 2.2.5 Protozoans' }
  ];
  const out = mod.renderNcertSectionsCardHtml(quotes);
  assert.strictEqual(countOccurrences(out, 'ncert-sections-card-citation'), 1,
    'expected exactly one citation group line: ' + out);
  assert(out.includes('Biological Classification - 2.2.5 Protozoans'), 'citation text missing: ' + out);
  assert.strictEqual(countOccurrences(out, 'ncert-sections-card-quote'), 3, 'all 3 quotes should render: ' + out);
});

test('two sections show two citation lines, each above its own group', () => {
  const quotes = [
    { quote: 'Quote from section A', citation: 'Section A' },
    { quote: 'Another quote from section A', citation: 'Section A' },
    { quote: 'Quote from section B', citation: 'Section B' }
  ];
  const out = mod.renderNcertSectionsCardHtml(quotes);
  assert.strictEqual(countOccurrences(out, 'ncert-sections-card-citation'), 2,
    'expected exactly two citation group lines: ' + out);
  const idxA = out.indexOf('Section A');
  const idxB = out.indexOf('Section B');
  const idxQA2 = out.indexOf('Another quote from section A');
  const idxQB = out.indexOf('Quote from section B');
  assert(idxA !== -1 && idxA < idxQA2, 'Section A header must precede its own quotes: ' + out);
  assert(idxB !== -1 && idxB < idxQB && idxQA2 < idxB, 'Section B header must come after section A\'s quotes, before its own: ' + out);
});

test('no per-quote citation markup remains (old inline "(citation)" after each quote is gone)', () => {
  const quotes = [{ quote: 'Some quoted text', citation: 'Some Chapter, Some Heading' }];
  const out = mod.renderNcertSectionsCardHtml(quotes);
  assert(!/&rdquo;\s*\(/.test(out), 'citation must not trail the quote inline any more: ' + out);
});

test('toggle still works: beyond the 5 visible quotes, "Show N more" wraps the rest in a hidden block', () => {
  const quotes = Array.from({ length: 7 }, (_, i) => ({ quote: `Quote number ${i + 1} of seven total`, citation: 'Same Section' }));
  const out = mod.renderNcertSectionsCardHtml(quotes);
  assert(out.includes('Show 2 more'), 'expected a toggle for the remaining 2 quotes: ' + out);
  assert(out.includes('ncert-sections-card-rest'), 'hidden rest block missing: ' + out);
  assert(out.includes('style="display:none"'), 'rest block should start hidden: ' + out);
  assert.strictEqual(countOccurrences(out, 'ncert-sections-card-quote'), 7, 'all 7 quotes should still be present in the markup: ' + out);
});

test('toggle across a section boundary: 2 sections, first visible-then-hidden split keeps both headers', () => {
  const quotes = [
    ...Array.from({ length: 4 }, (_, i) => ({ quote: `A quote ${i + 1}`, citation: 'Section A' })),
    { quote: 'A fifth quote still in section A', citation: 'Section A' },
    { quote: 'First quote in section B', citation: 'Section B' }
  ];
  const out = mod.renderNcertSectionsCardHtml(quotes);
  assert.strictEqual(countOccurrences(out, 'Section A'), 1, 'Section A is fully within the visible 5, one header: ' + out);
  assert(out.includes('Section B'), 'Section B header should appear in the hidden "rest" block: ' + out);
  assert(out.includes('Show 1 more'), 'expected a toggle for the single remaining quote: ' + out);
});

test('5 quotes from one section show the source exactly once through every render path', () => {
  const quotes = Array.from({ length: 5 }, (_, i) => ({
    quote: `Quote number ${i + 1} of five total`,
    citation: 'Biological Classification - 2.2.5 Protozoans'
  }));
  // "Live" path: renderNcertSectionsCardHtml called directly on the quotes the NCSEC trailer
  // just decoded (see newNcertSectionsQuotes in chat.html's streaming completion).
  const liveCard = mod.renderNcertSectionsCardHtml(quotes);
  // "Version switch" path: the quotes live on the version object itself (see the versions[]
  // render at bubble.innerHTML = ... + renderNcertSectionsCardHtml(v.ncertQuotes)).
  const msg = { versions: [{ text: 'v0', ncertQuotes: quotes }], activeVersionIndex: 0 };
  const versionSwitchCard = mod.renderNcertSectionsCardHtml(msg.versions[msg.activeVersionIndex].ncertQuotes);
  // "History reload" path: same message, read back through getActiveNcertQuotes the way
  // addMessageRow does on a fresh page load.
  const reloadCard = mod.renderNcertSectionsCardHtml(mod.getActiveNcertQuotes(msg));

  for (const [name, out] of [['live', liveCard], ['version switch', versionSwitchCard], ['reload', reloadCard]]) {
    assert.strictEqual(countOccurrences(out, 'ncert-sections-card-citation'), 1,
      `${name} render: expected exactly one source line: ${out}`);
    assert(out.includes('Biological Classification - 2.2.5 Protozoans'), `${name} render: source text missing: ${out}`);
    assert.strictEqual(countOccurrences(out, 'ncert-sections-card-quote'), 5, `${name} render: all 5 quotes should render: ${out}`);
  }
  assert.strictEqual(liveCard, versionSwitchCard, 'live and version-switch renders must be identical');
  assert.strictEqual(versionSwitchCard, reloadCard, 'version-switch and reload renders must be identical');
});

test('8 quotes from one section: expanded, the source line still appears exactly once (not again after the Show-more split)', () => {
  const quotes = Array.from({ length: 8 }, (_, i) => ({
    quote: `Quote number ${i + 1} of eight total`,
    citation: 'Biological Classification - 2.2.5 Protozoans'
  }));
  const out = mod.renderNcertSectionsCardHtml(quotes);
  assert.strictEqual(countOccurrences(out, 'ncert-sections-card-citation'), 1,
    'a single section spanning the 5-quote visible/hidden split must still print its source line only once: ' + out);
  assert(out.includes('Show 3 more'), 'expected a toggle for the remaining 3 quotes: ' + out);
  assert.strictEqual(countOccurrences(out, 'ncert-sections-card-quote'), 8, 'all 8 quotes should render: ' + out);
});

test('quotes from two sections (split by the 5-quote cut): two source lines total, collapsed and expanded', () => {
  const quotes = [
    ...Array.from({ length: 3 }, (_, i) => ({ quote: `Section A quote ${i + 1}`, citation: 'Section A' })),
    ...Array.from({ length: 4 }, (_, i) => ({ quote: `Section B quote ${i + 1}`, citation: 'Section B' }))
  ];
  const out = mod.renderNcertSectionsCardHtml(quotes);
  // The toggle only flips style.display on the already-rendered "rest" block -- nothing
  // re-renders -- so this one HTML string covers both the collapsed and expanded view.
  assert.strictEqual(countOccurrences(out, 'ncert-sections-card-citation'), 2,
    'exactly two source lines total; section B\'s header must not repeat when its tail quotes fall past the cut: ' + out);
  assert(out.includes('Show 2 more'), 'expected a toggle for the remaining 2 quotes: ' + out);
  assert.strictEqual(countOccurrences(out, 'ncert-sections-card-quote'), 7, 'all 7 quotes should render: ' + out);
});

console.log(`\n${passed} passed, ${failed} failed`);
process.exit(failed > 0 ? 1 : 0);
