// Unit tests for the SimpleTex/Mathpix paste-text parser used by admin-pdf-review.html's
// "Paste LaTeX" mode. Run with: node test-paste-parser.js
//
// There's no test framework/build step in this repo (see package.json), so this loads the
// parser functions straight out of the real HTML file (extracted from the same block described
// below, verified to be free of document/window/fetch/localStorage references) via Node's `vm`
// module and asserts against them directly -- no copy-pasted duplicate of the parsing logic to
// drift out of sync with the real file.
'use strict';
const fs = require('fs');
const path = require('path');
const vm = require('vm');
const assert = require('assert');

const html = fs.readFileSync(path.join(__dirname, 'admin-pdf-review.html'), 'utf8');
const startMarker = '// ---------------- Mathpix/SimpleTex paste parsing ----------------';
const endMarker = "// Feeds parsed questions into this page's EXISTING block model";
const startIdx = html.indexOf(startMarker);
const endIdx = html.indexOf(endMarker);
assert(startIdx !== -1 && endIdx !== -1 && endIdx > startIdx, 'could not locate the parser function block in admin-pdf-review.html -- did the surrounding comments move?');
const parserSource = html.slice(startIdx, endIdx);

const exportNames = [
  'cleanupLatexWrappers', 'extractSourceTag', 'extractOptions', 'parseQuestionChunk', 'parseMathpixText',
  'extractOptionsSimpleTex', 'insertBoundarySpaces', 'normalizeRunOnSpacing', 'splitByPipeDelimiter',
  'splitIntoQuestionChunksByPipe', 'extractQuestionsFromPipeChunks',
  'extractOptionsNumbered', 'extractOptionsWithPriority', 'normalizeSimpleTexHeaders',
  'parseSimpleTexText', 'parseSimpleTexBareParagraphs'
];
const sandbox = {};
vm.createContext(sandbox);
const wrapped = `(function() { ${parserSource}\nreturn { ${exportNames.join(', ')} }; })()`;
const parser = vm.runInContext(wrapped, sandbox, { filename: 'admin-pdf-review.html (extracted parser block)' });

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

console.log('normalizeRunOnSpacing / insertBoundarySpaces');
test('does NOT insert a space at a lower->upper boundary (would corrupt plain-text formulas)', () => {
  // Regression guard: this rule was tried and reverted after it turned the real, previously
  // verified "(SiH$_3$)$_2$" option into "(Si H$_3$)$_2$" -- "Si"+"H" is indistinguishable from
  // a genuine run-on word boundary by regex alone, so it's intentionally not attempted at all.
  assert.strictEqual(parser.insertBoundarySpaces('bondLength'), 'bondLength');
  assert.strictEqual(parser.insertBoundarySpaces('(SiH3)2'), '(SiH 3)2'); // digit boundary still fires -- see next test
});
test('inserts a space at letter->digit and digit->letter boundaries', () => {
  assert.strictEqual(parser.insertBoundarySpaces('HOCl2gives3He'), 'HOCl 2 gives 3 He');
});
test('does not touch content inside $...$ math spans', () => {
  const input = 'Given $a^{2}b_3$ and text4here outside';
  const out = parser.normalizeRunOnSpacing(input);
  assert(out.includes('$a^{2}b_3$'), 'math span was altered: ' + out);
  assert(out.includes('text 4 here'), 'boundary outside math span was not normalized: ' + out);
});
test('does not touch content inside $$...$$ display math spans', () => {
  const input = 'stem $$x^{2}y_1$$ moreText';
  const out = parser.normalizeRunOnSpacing(input);
  assert(out.includes('$$x^{2}y_1$$'), 'display math span was altered: ' + out);
});

console.log('\nsplitByPipeDelimiter (priority tier a)');
test('splits a || delimited stem+4options block', () => {
  const r = parser.splitByPipeDelimiter('What is X? || Option one || Option two || Option three || Option four');
  assert(r, 'expected a non-null result');
  assert.strictEqual(r.stem, 'What is X?');
  assert.strictEqual(r.options.a, 'Option one');
  assert.strictEqual(r.options.d, 'Option four');
});
test('returns null when there are fewer than 5 segments', () => {
  assert.strictEqual(parser.splitByPipeDelimiter('stem || only one option'), null);
});
test('returns null when there is no || at all', () => {
  assert.strictEqual(parser.splitByPipeDelimiter('no delimiter here'), null);
});

console.log('\nextractOptionsNumbered (priority tier c)');
test('splits a clean 1. 2. 3. 4. numbered block', () => {
  const r = parser.extractOptionsNumbered('Stem text 1.Option A 2.Option B 3.Option C 4.Option D');
  assert.strictEqual(r.options.a, 'Option A');
  assert.strictEqual(r.options.b, 'Option B');
  assert.strictEqual(r.options.c, 'Option C');
  assert.strictEqual(r.options.d, 'Option D');
});
test('leaves options blank when markers are out of order/incomplete', () => {
  const r = parser.extractOptionsNumbered('Stem with only 1.one and 2.two');
  assert.strictEqual(r.options.a, '');
});

console.log('\nextractOptionsWithPriority (a -> b -> c cascade + confidence tagging)');
test('prefers || delimiter over markers when both could apply', () => {
  const r = parser.extractOptionsWithPriority('stem || (A) fake-looking opt || b || c || d');
  assert.strictEqual(r.confidence, 'delimiter');
});
test('falls to (A)(B)(C)(D) markers when no || present', () => {
  const r = parser.extractOptionsWithPriority('Which is correct?(A) opt a(B) opt b(C) opt c(D) opt d');
  assert.strictEqual(r.confidence, 'marker');
  assert.strictEqual(r.options.a, 'opt a');
});
test('falls to numbered markers when neither || nor (A)(B)(C)(D) present', () => {
  const r = parser.extractOptionsWithPriority('Which is correct? 1.opt a 2.opt b 3.opt c 4.opt d');
  assert.strictEqual(r.confidence, 'marker');
  assert.strictEqual(r.options.a, 'opt a');
});
test('returns null confidence with blank options when nothing matches', () => {
  const r = parser.extractOptionsWithPriority('just a plain sentence with no markers at all');
  assert.strictEqual(r.confidence, null);
  assert.strictEqual(r.options.a, '');
});

console.log('\nregression: existing (A)(B)(C)(D) marker parsing still works end to end');
test('parseSimpleTexText parses a normal lettered question with a source tag', () => {
  const input = `A dimensionally consistent relation for volumetric flow rate Q can be written as(A) no units(B) no units and no dimensions(C) some units but no dimensions(D) some units and also dimensions\n\n[NEET 2016]`;
  const out = parser.parseSimpleTexText(input);
  assert.strictEqual(out.length, 1);
  assert.strictEqual(out[0].option_a, 'no units');
  assert.strictEqual(out[0].source_tag, 'NEET 2016');
  assert.strictEqual(out[0].parse_confidence, 'marker');
});

console.log('\nthe reported bug: run-on/glued SimpleTex output with numbered markers');
test('parses a run-on numbered-marker block into separate options, not one run-on block', () => {
  // Simulates the reported failure ("159.Thecorrectstatementabouthydrolysisof...") but with a
  // full, realistic body -- the fragment given in the bug report is too short on its own to
  // contain a real option boundary to assert against. Math spans are deliberately left glued to
  // surrounding text (as SimpleTex actually emits them) to confirm the math-span skip doesn't
  // block numbered-marker detection immediately outside a $...$ span.
  const input = '159.Thecorrectstatementabouthydrolysisof$BCl_3$and$NCl_3$is-' +
    '1.$NCl_3$ishydrolysedandgivesHOClbut$BCl_3$isnothydrolysed' +
    '2.Both$NCl_3$and$BCl_3$onhydrolysisgivesHCl' +
    '3.$NCl_3$onhydrolysisgivesHOClbut$BCl_3$givesHCl' +
    '4.Both$NCl_3$and$BCl_3$onhydrolysisgivesHOCl';
  const out = parser.parseSimpleTexText(input);
  assert.strictEqual(out.length, 1, 'expected exactly one parsed question, got ' + out.length);
  const q = out[0];
  assert.strictEqual(q.parse_confidence, 'marker');
  assert(q.option_a.length > 0 && q.option_b.length > 0 && q.option_c.length > 0 && q.option_d.length > 0,
    'expected all four options to be non-empty, got: ' + JSON.stringify(q));
  assert.notStrictEqual(q.option_a, q.option_b, 'options collapsed into the same text -- splitting failed');
  assert(q.option_a.includes('NCl'), 'option a lost its content: ' + q.option_a);
  assert(q.option_d.includes('HOCl'), 'option d lost its content: ' + q.option_d);
});

console.log('\nparseSimpleTexBareParagraphs (tier d, only reached when a/b/c all fail)');
test('tags positionally-grouped questions as fallback confidence', () => {
  // Subscripts wrapped in $...$, matching real SimpleTex output (e.g. "(SiH$_3$)$_2$") -- this
  // is also what caught the case-boundary regression during manual testing, see above.
  const input = 'Which among the following is an electron-deficient compound?\n\n(SiH$_3$)$_2$\n\n(BH$_3$)$_2$\n\nPH$_3$\n\n(CH$_3$)$_2$';
  const out = parser.parseSimpleTexBareParagraphs(input);
  assert.strictEqual(out.length, 1);
  assert.strictEqual(out[0].parse_confidence, 'fallback');
  assert.strictEqual(out[0].option_a, '(SiH$_3$)$_2$');
});
test('does not misattach the next question as options when the current one is short', () => {
  // Regression for the exact misalignment caught during manual testing: a question whose real
  // options don't cleanly fill 4 paragraphs must NOT swallow the next question's stem/diagram.
  const input = 'Electron pairs that do not participate in bonding.\n\n' +
    'Electron pairs that are present in the valence shell.\n\n' +
    'The correct Lewis structure of acetic acid is -\n\n' +
    '<div><img src="x"/></div>\n\n' +
    'None of the above.';
  const out = parser.parseSimpleTexBareParagraphs(input);
  const stems = out.map(o => o.question);
  assert(!stems.some(s => s.includes('Lewis structure') && s.includes('do not participate')),
    'a later question\'s stem got merged into an earlier one\'s options');
});

console.log('\nsplitIntoQuestionChunksByPipe / the 183->19 regression (flattened multi-question || blob)');
test('splits on "||N." as a question boundary, keeping the number with the new question', () => {
  const input = '146.Stem A 1.opt1 2.opt2 3.opt3 4.opt4||158.Stem B 1.optA 2.optB 3.optC 4.optD';
  const chunks = parser.splitIntoQuestionChunksByPipe(input);
  assert.strictEqual(chunks.length, 2);
  assert(chunks[0].startsWith('146.'));
  assert(chunks[1].startsWith('158.'), 'the "158." must stay attached to the new question, not get eaten by the boundary match');
});
test('a plain "||" with no following question number is NOT treated as a question boundary', () => {
  assert.strictEqual(parser.splitIntoQuestionChunksByPipe('stem || opt1 || opt2 || opt3 || opt4'), null);
});
test('a chunk\'s own remaining "||" still splits into options as before (not eaten by boundary detection)', () => {
  const input = '146.What is the SI unit of force?||Newton||Joule||Watt||Pascal||158.Next stem 1.a 2.b 3.c 4.d';
  const chunks = parser.splitIntoQuestionChunksByPipe(input);
  assert.strictEqual(chunks.length, 2);
  const questions = parser.extractQuestionsFromPipeChunks(chunks, '');
  assert.strictEqual(questions[0].parse_confidence, 'delimiter');
  assert.strictEqual(questions[0].option_a, 'Newton');
  assert.strictEqual(questions[1].parse_confidence, 'marker');
  assert.strictEqual(questions[1].option_a, 'a');
});
test('the reported real block (Pentane question + 158/159/160) parses into 4 separate questions, not 1', () => {
  // Reconstructed from the actual sample text shown earlier in this conversation, joined with
  // "||" the way a flattened markdown table (row = question, cells lose their newlines) would
  // actually glue consecutive questions together in a real paste.
  const input =
    '145.The incorrect order of the force of attraction among the given species is -' +
    '1.HI>HBr>$Cl_2$' +
    '2.$CCl_4$>$CH_2Cl_2$>$CH_3Cl$>$CH_4$' +
    '3.n-Pentane>iso-Pentane>neo-Pentane' +
    '4.$OH_2$>$O(CH_3)_2$>$OBr_2$' +
    '||158.O—O bond length (in Å) in $H_2O_2$ and $O_2F_2$ is respectively:' +
    '1.1.22,1.48' +
    '2.1.48,1.22' +
    '3.1.22,1.22' +
    '4.1.48,1.48' +
    '||159.The correct statement about hydrolysis of $BCl_3$ and $NCl_3$ is-' +
    '1.$NCl_3$ is hydrolysed and gives HOCl but $BCl_3$ is not hydrolysed' +
    '2.Both $NCl_3$ and $BCl_3$ on hydrolysis gives $HCl$' +
    '3.$NCl_3$ on hydrolysis gives HOCl but $BCl_3$ gives $HCl$' +
    '4.Both $NCl_3$ and $BCl_3$ on hydrolysis gives HOCl' +
    '||160.The correct order of solubility in an aqueous medium is:' +
    '1.$CuS>ZnS>Na_2S$' +
    '2.$ZnS>Na_2S>CuS$' +
    '3.$Na_2S>CuS>ZnS$' +
    '4.$Na_2S>ZnS>CuS$';
  const out = parser.parseSimpleTexText(input);
  assert.strictEqual(out.length, 4, 'expected 4 separate questions, got ' + out.length + ': ' + JSON.stringify(out.map(q => q.question.slice(0, 30))));
  assert(out[0].question.startsWith('145.'));
  assert(out[1].question.startsWith('158.'));
  assert(out[2].question.startsWith('159.'));
  assert(out[3].question.startsWith('160.'));
  out.forEach((q, i) => {
    assert(q.option_a && q.option_b && q.option_c && q.option_d, `question ${i} (${q.question.slice(0, 20)}...) has a blank option: ${JSON.stringify(q)}`);
    assert.strictEqual(q.parse_confidence, 'marker');
  });
  assert(out[1].option_a.includes('1.22'), 'Q158 option a lost its content: ' + out[1].option_a);
  assert(out[2].option_a.includes('NCl'), 'Q159 option a lost its content: ' + out[2].option_a);
  assert(out[3].option_a.includes('CuS'), 'Q160 option a lost its content: ' + out[3].option_a);
});
test('regression guard: a single self-contained "stem || 4 options" question (no ||N. boundary) still works via the existing tier', () => {
  const input = 'What is the SI unit of force? || Newton || Joule || Watt || Pascal';
  const out = parser.parseSimpleTexText(input);
  assert.strictEqual(out.length, 1);
  assert.strictEqual(out[0].parse_confidence, 'delimiter');
  assert.strictEqual(out[0].option_a, 'Newton');
});
test('a flattened || blob mixed with ordinary lettered questions in the same document all count -- not just the blob', () => {
  // Simulates the actual 183-question document shape: most of it is normal paragraph-formatted
  // questions, with one flattened multi-question table blob embedded partway through. Before the
  // fix, that one blob paragraph alone would eat ~everything after it into a single bogus
  // question; the fix must leave the surrounding ordinary questions completely unaffected.
  const before = 'Which among the following is an electron-deficient compound?(A) opt a(B) opt b(C) opt c(D) opt d';
  const blob =
    '145.Stem one 1.a1 2.a2 3.a3 4.a4' +
    '||158.Stem two 1.b1 2.b2 3.b3 4.b4' +
    '||159.Stem three 1.c1 2.c2 3.c3 4.c4' +
    '||160.Stem four 1.d1 2.d2 3.d3 4.d4';
  const after = 'Which of the following is a paramagnetic compound?(A) N2(B) H2(C) Li2(D) O2';
  const input = before + '\n\n' + blob + '\n\n' + after;
  const out = parser.parseSimpleTexText(input);
  assert.strictEqual(out.length, 6, 'expected 1 (before) + 4 (blob) + 1 (after) = 6, got ' + out.length);
});

console.log(`\n${passed} passed, ${failed} failed`);
process.exit(failed > 0 ? 1 : 0);
