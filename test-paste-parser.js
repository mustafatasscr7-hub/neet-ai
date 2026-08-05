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
  'maskMathSpans', 'unmaskMathSpans', 'insertBoundarySpaces', 'segmentWord', 'segmentFusedRuns',
  'countSequentialQuadruples', 'detectPatterns',
  'splitByPipeDelimiter', 'splitByTableRow', 'extractOptionsSimpleTex', 'extractOptionsNumbered',
  'extractOptionsByDetectedPattern', 'splitIntoQuestionChunksByPipe', 'extractQuestionsFromChunks',
  'normalizeSimpleTexHeaders', 'groupLeftoverParagraphsPositionally', 'parseSimpleTexPipeline'
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

// ==================== Stage A: math masking ====================
console.log('maskMathSpans / unmaskMathSpans (Stage A)');
test('masks and restores a $...$ span exactly', () => {
  const { masked, spans } = parser.maskMathSpans('Given $a^{2}b_3$ and text here');
  assert(!masked.includes('$'), 'math span was not masked: ' + masked);
  assert.strictEqual(parser.unmaskMathSpans(masked, spans), 'Given $a^{2}b_3$ and text here');
});
test('masks and restores a $$...$$ display span exactly', () => {
  const { masked, spans } = parser.maskMathSpans('stem $$x^{2}y_1$$ more');
  assert.strictEqual(parser.unmaskMathSpans(masked, spans), 'stem $$x^{2}y_1$$ more');
});
test('round-trips multiple math spans in one string, in order', () => {
  const raw = '$a$ and $b$ and $c$';
  const { masked, spans } = parser.maskMathSpans(raw);
  assert.strictEqual(parser.unmaskMathSpans(masked, spans), raw);
});
test('the placeholder token contains no letters -- immune to later letter<->digit boundary repair', () => {
  const { masked } = parser.maskMathSpans('$a^{2}b_3$');
  assert(!/[a-zA-Z]/.test(masked), 'placeholder token contains a letter, which Stage B1 could mangle: ' + JSON.stringify(masked));
});

// ==================== Stage B1: letter<->digit boundary repair ====================
console.log('\ninsertBoundarySpaces (Stage B1)');
test('does NOT insert a space at a lower->upper boundary (would corrupt plain-text formulas)', () => {
  // Regression guard: this rule was tried and reverted after it turned the real, previously
  // verified "(SiH$_3$)$_2$" option into "(Si H$_3$)$_2$" -- "Si"+"H" is indistinguishable from
  // a genuine run-on word boundary by regex alone. Stage B2's dictionary segmenter now owns that
  // class of decision instead (see below).
  assert.strictEqual(parser.insertBoundarySpaces('bondLength'), 'bondLength');
});
test('inserts a space at letter->digit and digit->letter boundaries', () => {
  assert.strictEqual(parser.insertBoundarySpaces('HOCl2gives3He'), 'HOCl 2 gives 3 He');
});

// ==================== Stage B2: dictionary word segmentation ====================
console.log('\nsegmentFusedRuns / segmentWord (Stage B2)');
test('segments a genuine run-on into its real words', () => {
  assert.strictEqual(parser.segmentFusedRuns('hasnovacantd'), 'has no vacant d');
});
test('segments the reported failing example into readable words', () => {
  assert.strictEqual(parser.segmentFusedRuns('thecorrectstatementabouthydrolysisof'), 'the correct statement about hydrolysis of');
});
test('does NOT split "SiH" -- the exact case-boundary regression, now owned by the segmenter', () => {
  // Neither "Si" nor "H" alone is a recognized dictionary word, so no confident split exists --
  // segmentFusedRuns must leave it untouched rather than guess.
  assert.strictEqual(parser.segmentFusedRuns('SiH'), 'SiH');
});
test('leaves a short, no-case-transition run below the 12-char threshold untouched', () => {
  assert.strictEqual(parser.segmentFusedRuns('isunstable'), 'isunstable');
});
test('leaves an unrecognized long technical term untouched rather than mis-splitting it', () => {
  const weird = 'zzqxvbnmqwrtyplkjhgfdsazzz'; // not a real word under any split
  assert.strictEqual(parser.segmentFusedRuns(weird), weird);
});
test('segmentWord returns null when no full-coverage dictionary split exists', () => {
  assert.strictEqual(parser.segmentWord('sih'), null);
});

// ==================== Stage C: pattern detection ====================
console.log('\ndetectPatterns / countSequentialQuadruples (Stage C)');
test('detects a single complete (A)(B)(C)(D) cycle', () => {
  const r = parser.detectPatterns('stem(A) a(B) b(C) c(D) d');
  assert.strictEqual(r.scores.letter, 1);
  assert.strictEqual(r.winner, 'letter');
});
test('detects a single complete numbered cycle', () => {
  const r = parser.detectPatterns('stem 1.a 2.b 3.c 4.d');
  assert.strictEqual(r.scores.numbered, 1);
  assert.strictEqual(r.winner, 'numbered');
});
test('detects a pipe-delimited block (4 pipes = 1 question)', () => {
  const r = parser.detectPatterns('stem || a || b || c || d');
  assert.strictEqual(r.scores.pipe, 1);
  assert.strictEqual(r.winner, 'pipe');
});
test('counts multiple back-to-back quadruple cycles, not just one', () => {
  const r = parser.detectPatterns('(A)a(B)b(C)c(D)d (A)a(B)b(C)c(D)d');
  assert.strictEqual(r.scores.letter, 2);
});
test('picks the highest-scoring pattern, not a fixed priority order, when multiple are present', () => {
  // Two full numbered cycles vs one full letter cycle -- numbered should win on count, even
  // though letter markers are earlier in the tie-break order.
  const r = parser.detectPatterns('(A)a(B)b(C)c(D)d 1.a 2.b 3.c 4.d 1.a 2.b 3.c 4.d');
  assert.strictEqual(r.winner, 'numbered');
});
test('table-row score only counts when no "||" is present at all', () => {
  const withDouble = parser.detectPatterns('| a | b || c | d | e |');
  assert.strictEqual(withDouble.scores.table, 0);
  const singleOnly = parser.detectPatterns('| a | b | c | d | e |');
  assert(singleOnly.scores.table >= 1);
});

// ==================== Stage E: option extraction per chunk ====================
console.log('\nextractOptionsByDetectedPattern (Stage E)');
test('prefers || delimiter over markers when both could apply', () => {
  const r = parser.extractOptionsByDetectedPattern('stem || (A) fake-looking opt || b || c || d');
  assert.strictEqual(r.confidence, 'delimiter');
  assert.strictEqual(r.detected_pattern, 'pipe');
});
test('falls to (A)(B)(C)(D) markers when no || present', () => {
  const r = parser.extractOptionsByDetectedPattern('Which is correct?(A) opt a(B) opt b(C) opt c(D) opt d');
  assert.strictEqual(r.confidence, 'marker');
  assert.strictEqual(r.detected_pattern, 'letter');
  assert.strictEqual(r.options.a, 'opt a');
});
test('falls to numbered markers when neither || nor (A)(B)(C)(D) present', () => {
  const r = parser.extractOptionsByDetectedPattern('Which is correct? 1.opt a 2.opt b 3.opt c 4.opt d');
  assert.strictEqual(r.confidence, 'marker');
  assert.strictEqual(r.detected_pattern, 'numbered');
});
test('falls to a markdown table row when nothing else applies', () => {
  const r = parser.extractOptionsByDetectedPattern('| stem | opt a | opt b | opt c | opt d |');
  assert.strictEqual(r.confidence, 'delimiter');
  assert.strictEqual(r.detected_pattern, 'table');
  assert.strictEqual(r.options.a, 'opt a');
});
test('returns null confidence/pattern with blank options when nothing matches', () => {
  const r = parser.extractOptionsByDetectedPattern('just a plain sentence with no markers at all');
  assert.strictEqual(r.confidence, null);
  assert.strictEqual(r.detected_pattern, null);
  assert.strictEqual(r.options.a, '');
});

// ==================== Full pipeline: regression coverage from prior rounds ====================
console.log('\nparseSimpleTexPipeline -- regression: existing (A)(B)(C)(D) marker parsing still works end to end');
test('parses a normal lettered question with a source tag', () => {
  const input = `A dimensionally consistent relation for volumetric flow rate Q can be written as(A) no units(B) no units and no dimensions(C) some units but no dimensions(D) some units and also dimensions\n\n[NEET 2016]`;
  const out = parser.parseSimpleTexPipeline(input);
  assert.strictEqual(out.length, 1);
  assert.strictEqual(out[0].option_a, 'no units');
  assert.strictEqual(out[0].source_tag, 'NEET 2016');
  assert.strictEqual(out[0].parse_confidence, 'marker');
  assert.strictEqual(out[0].detected_pattern, 'letter');
});

console.log('\nparseSimpleTexPipeline -- regression: run-on/glued SimpleTex output with numbered markers');
test('parses a run-on numbered-marker block into separate options, not one run-on block', () => {
  const input = '159.Thecorrectstatementabouthydrolysisof$BCl_3$and$NCl_3$is-' +
    '1.$NCl_3$ishydrolysedandgivesHOClbut$BCl_3$isnothydrolysed' +
    '2.Both$NCl_3$and$BCl_3$onhydrolysisgivesHCl' +
    '3.$NCl_3$onhydrolysisgivesHOClbut$BCl_3$givesHCl' +
    '4.Both$NCl_3$and$BCl_3$onhydrolysisgivesHOCl';
  const out = parser.parseSimpleTexPipeline(input);
  assert.strictEqual(out.length, 1, 'expected exactly one parsed question, got ' + out.length);
  const q = out[0];
  assert.strictEqual(q.parse_confidence, 'marker');
  assert(q.option_a.length > 0 && q.option_b.length > 0 && q.option_c.length > 0 && q.option_d.length > 0,
    'expected all four options to be non-empty, got: ' + JSON.stringify(q));
  assert.notStrictEqual(q.option_a, q.option_b, 'options collapsed into the same text -- splitting failed');
  assert(q.option_a.includes('NCl'), 'option a lost its content: ' + q.option_a);
  assert(q.option_d.includes('HOCl'), 'option d lost its content: ' + q.option_d);
  // Now also readable, thanks to Stage B2 -- the stem reads as real words, not a run-on blob.
  assert(q.question.includes('The correct statement about hydrolysis'), 'stem was not word-segmented: ' + q.question);
});

console.log('\nparseSimpleTexPipeline -- regression: position-based fallback (no markers anywhere)');
test('tags positionally-grouped questions as fallback confidence with detected_pattern "position"', () => {
  const input = 'Which among the following is an electron-deficient compound?\n\n(SiH$_3$)$_2$\n\n(BH$_3$)$_2$\n\nPH$_3$\n\n(CH$_3$)$_2$';
  const out = parser.parseSimpleTexPipeline(input);
  assert.strictEqual(out.length, 1);
  assert.strictEqual(out[0].parse_confidence, 'fallback');
  assert.strictEqual(out[0].detected_pattern, 'position');
  assert.strictEqual(out[0].option_a, '(SiH$_3$)$_2$'); // SiH formula still intact -- no case-boundary corruption
});
test('does not misattach the next question as options when the current one is short', () => {
  // Regression for the exact misalignment caught during manual testing: a question whose real
  // options don't cleanly fill 4 paragraphs must NOT swallow the next question's stem/diagram.
  const input = 'Electron pairs that do not participate in bonding.\n\n' +
    'Electron pairs that are present in the valence shell.\n\n' +
    'The correct Lewis structure of acetic acid is -\n\n' +
    '<div><img src="x"/></div>\n\n' +
    'None of the above.';
  const out = parser.parseSimpleTexPipeline(input);
  const stems = out.map(o => o.question);
  assert(!stems.some(s => s.includes('Lewis structure') && s.includes('do not participate')),
    "a later question's stem got merged into an earlier one's options");
});
test('a paragraph that fails markers now gets a real shot at Stage C/D/E before falling to position (recovers previously-dropped questions)', () => {
  // Before this round: the octet-rule paragraph below only got tried against markers if the
  // WHOLE document found zero questions elsewhere; mixed in with other successfully-parsed
  // paragraphs, it used to be silently dropped instead of recognized as a numbered-in-stem
  // question. Now every paragraph gets its own shot regardless of what else succeeded nearby.
  const input = 'Which among the following is an electron-deficient compound?(A) opt a(B) opt b(C) opt c(D) opt d\n\n' +
    'The significance of the octet rule is: 1. To determine stability. 2. To find hybridization. 3. To calculate lone pairs. 4. None of the above.';
  const out = parser.parseSimpleTexPipeline(input);
  assert.strictEqual(out.length, 2);
  const octet = out.find(q => q.question.includes('octet'));
  assert(octet, 'octet-rule question was dropped entirely');
  assert.strictEqual(octet.parse_confidence, 'marker');
  assert.strictEqual(octet.detected_pattern, 'numbered');
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
  const questions = parser.extractQuestionsFromChunks(chunks, '');
  assert.strictEqual(questions[0].parse_confidence, 'delimiter');
  assert.strictEqual(questions[0].option_a, 'Newton');
  assert.strictEqual(questions[1].parse_confidence, 'marker');
  assert.strictEqual(questions[1].option_a, 'a');
});
test('the reported real block (Pentane question + 158/159/160) parses into 4 separate questions, not 1', () => {
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
  const out = parser.parseSimpleTexPipeline(input);
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
test('regression guard: a single self-contained "stem || 4 options" question (no ||N. boundary) still works', () => {
  const input = 'What is the SI unit of force? || Newton || Joule || Watt || Pascal';
  const out = parser.parseSimpleTexPipeline(input);
  assert.strictEqual(out.length, 1);
  assert.strictEqual(out[0].parse_confidence, 'delimiter');
  assert.strictEqual(out[0].option_a, 'Newton');
});
test('a flattened || blob mixed with ordinary lettered questions in the same document all count -- not just the blob', () => {
  const before = 'Which among the following is an electron-deficient compound?(A) opt a(B) opt b(C) opt c(D) opt d';
  const blob =
    '145.Stem one 1.a1 2.a2 3.a3 4.a4' +
    '||158.Stem two 1.b1 2.b2 3.b3 4.b4' +
    '||159.Stem three 1.c1 2.c2 3.c3 4.c4' +
    '||160.Stem four 1.d1 2.d2 3.d3 4.d4';
  const after = 'Which of the following is a paramagnetic compound?(A) N2(B) H2(C) Li2(D) O2';
  const input = before + '\n\n' + blob + '\n\n' + after;
  const out = parser.parseSimpleTexPipeline(input);
  assert.strictEqual(out.length, 6, 'expected 1 (before) + 4 (blob) + 1 (after) = 6, got ' + out.length);
});

// ==================== Fixture library: combined regression run ====================
console.log('\nfixture library (test-fixtures/*.json)');
const fixturesDir = path.join(__dirname, 'test-fixtures');
const fixtureFiles = fs.readdirSync(fixturesDir).filter(f => f.endsWith('.json'));
assert(fixtureFiles.length >= 5, `expected at least 5 fixture files, found ${fixtureFiles.length}`);

let combinedTotal = 0;
let combinedFallback = 0;
for (const file of fixtureFiles) {
  const fixture = JSON.parse(fs.readFileSync(path.join(fixturesDir, file), 'utf8'));
  test(`${file}: provenance is tagged real or synthetic`, () => {
    assert(fixture.provenance === 'real' || fixture.provenance === 'synthetic', `invalid provenance "${fixture.provenance}" in ${file} -- must be "real" or "synthetic"`);
  });
  test(`${file}: parses within its expected question-count range`, () => {
    const out = parser.parseSimpleTexPipeline(fixture.raw);
    combinedTotal += out.length;
    combinedFallback += out.filter(q => q.parse_confidence === 'fallback').length;
    assert(
      out.length >= fixture.expected.minQuestions && out.length <= fixture.expected.maxQuestions,
      `expected ${fixture.expected.minQuestions}-${fixture.expected.maxQuestions} questions, got ${out.length}`
    );
  });
}

// The combined fallback rate is reported on every run (so a real regression -- e.g. a change
// that suddenly makes a previously-clean fixture fall back -- is visible in the numbers) but
// deliberately does NOT hard-fail the suite at a fixed threshold like <10%: the one real fixture
// (chemistry-chemical-bonding-real) is genuinely ~91% fallback on its own merits -- image-only
// options and ambiguous embedded lists that no parser can resolve without guessing -- and forcing
// the aggregate under an arbitrary number with a small fixture set would mean padding with clean
// synthetic content to game the number, not actually fixing anything. Grow the real-fixture side
// of this list over time and this number should trend down on its own.
console.log(`\n  combined fixture stats: ${combinedTotal} questions total, ${combinedFallback} fallback (${combinedTotal ? (100 * combinedFallback / combinedTotal).toFixed(1) : '0.0'}%) -- informational, not a hard gate`);

console.log(`\n${passed} passed, ${failed} failed`);
process.exit(failed > 0 ? 1 : 0);
