// Unit tests for the inline match-the-following pair splitter ("(a) Pila (i) Flame cells")
// used by pyqbank.html, savedquestions.html, personalised-test.html, mocktest.html,
// mockresults.html, chat.html and admin-pyq-preview.html. Run with: node test-match-pair-split.js
//
// No test framework/build step in this repo (see package.json), so -- same approach as
// test-paste-parser.js -- this loads the splitter functions straight out of the real HTML file
// (pyqbank.html, the one all the other copies were mirrored from) via Node's `vm` module and
// asserts against them directly, rather than keeping a copy-pasted duplicate of the logic that
// could drift out of sync with the real file.
'use strict';
const fs = require('fs');
const path = require('path');
const vm = require('vm');
const assert = require('assert');

const html = fs.readFileSync(path.join(__dirname, 'pyqbank.html'), 'utf8');
const startMarker = '// ---------------- Inline match-the-following pairs (no List-I/List-II header) ----------------';
const endMarker = '// ---------------- Assertion-Reason / Statement I-II formatting ----------------';
const startIdx = html.indexOf(startMarker);
const endIdx = html.indexOf(endMarker);
assert(startIdx !== -1 && endIdx !== -1 && endIdx > startIdx, 'could not locate the match-pair splitter block in pyqbank.html -- did the surrounding comments move?');
const splitterSource = html.slice(startIdx, endIdx);

const exportNames = ['parseMatchPairLine_MLP', 'findMatchPairRun_MLP', 'renderMatchPairsHtml_MLP'];
const sandbox = {};
vm.createContext(sandbox);
const wrapped = `(function() { ${splitterSource}\nreturn { ${exportNames.join(', ')} }; })()`;
const splitter = vm.runInContext(wrapped, sandbox, { filename: 'pyqbank.html (extracted match-pair splitter block)' });

const identity = x => x;

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

console.log('renderMatchPairsHtml_MLP');

test('splits the real Pila/Bombyx/Pleurobrachia/Taenia question into a 2-column grid', () => {
  const text = 'Match the following columns.\n(a) Pila (i) Flame cells\n(b) Bombyx (ii) Comb plates\n(c) Pleurobrachia (iii) Nematocysts\n(d) Taenia (iv) Scolex\nChoose the correct answer.';
  const html = splitter.renderMatchPairsHtml_MLP(text, identity);
  assert(html !== null, 'expected a match to be found');
  assert(html.includes('match-pair-preamble'), 'missing preamble wrapper');
  assert(html.includes('match-pair-grid'), 'missing grid wrapper');
  assert(html.includes('match-pair-trailer'), 'missing trailer wrapper');
  // 4 rows x 2 cells = 8 match-pair-cell divs.
  assert.strictEqual((html.match(/match-pair-cell/g) || []).length, 8);
  // Markers are kept in the rendered text (matches the List-I/List-II column renderer's own
  // "(a) Pila" convention) -- students need them to read lettered answer options like "a-iv".
  assert(html.includes('(a) Pila') && html.includes('(i) Flame cells'), 'left/right marker+text missing');
  assert(html.includes('(d) Taenia') && html.includes('(iv) Scolex'), 'left/right marker+text missing');
});

test('leaves a question with markers already on separate lines untouched', () => {
  const text = 'Match the following.\n(a) Pila\n(i) Flame cells\n(b) Bombyx\n(ii) Comb plates';
  assert.strictEqual(splitter.renderMatchPairsHtml_MLP(text, identity), null);
});

test('leaves normal question text (a single stray "(a)") untouched', () => {
  const text = 'Which of the following is true? (a) Only statement I is correct.';
  assert.strictEqual(splitter.renderMatchPairsHtml_MLP(text, identity), null);
});

test('requires at least 2 consecutive matching lines -- a single pair line is left alone', () => {
  const text = 'Intro text.\n(a) Pila (i) Flame cells\nSome unrelated continuation line.';
  assert.strictEqual(splitter.renderMatchPairsHtml_MLP(text, identity), null);
});

test('handles the dotted form without brackets ("a." / "i.") on both markers', () => {
  const text = 'a. Enzyme A i. Cofactor 1\nb. Enzyme B ii. Cofactor 2';
  const html = splitter.renderMatchPairsHtml_MLP(text, identity);
  assert(html !== null, 'expected a match to be found');
  assert(html.includes('a. Enzyme A') && html.includes('i. Cofactor 1'));
  assert(html.includes('b. Enzyme B') && html.includes('ii. Cofactor 2'));
});

test('never includes the options block -- unrelated trailing text after the run is kept as a trailer, not split', () => {
  const text = '(a) Pila (i) Flame cells\n(b) Bombyx (ii) Comb plates\nChoose the correct answer from the options below.';
  const html = splitter.renderMatchPairsHtml_MLP(text, identity);
  assert(html !== null);
  assert(html.includes('match-pair-trailer'));
  assert(html.includes('Choose the correct answer from the options below.'));
  assert.strictEqual((html.match(/match-pair-cell/g) || []).length, 4);
});

console.log(`\n${passed} passed, ${failed} failed`);
process.exit(failed > 0 ? 1 : 0);
