// Unit tests for the "Scan failed: [object Object]" fix in admin-pdf-review.html and
// contractor-pdf-review.html: the HTTP error formatter (formatErrorDetail/describeHttpError)
// and the mock-exam mock-name length validation (mockNameValidationError).
//
// Same no-framework convention as test-paste-parser.js: loads the real functions straight out of
// both HTML files via Node's `vm` module (extracted by the marker comments around their
// definitions) and runs identical assertions against both copies, so the two duplicated pages
// can't silently drift apart. Run with: node test-error-formatter.js
'use strict';
const fs = require('fs');
const path = require('path');
const vm = require('vm');
const assert = require('assert');

function extractBlock(html, startMarker, endMarker, file) {
  const startIdx = html.indexOf(startMarker);
  const endIdx = html.indexOf(endMarker, startIdx);
  assert(startIdx !== -1 && endIdx !== -1 && endIdx > startIdx,
    `could not locate block in ${file} -- did the surrounding comments move? (start=${startIdx}, end=${endIdx})`);
  return html.slice(startIdx, endIdx);
}

function loadPage(file) {
  const html = fs.readFileSync(path.join(__dirname, file), 'utf8');
  const errorSource = extractBlock(
    html,
    '// ---------------- HTTP error formatting ----------------',
    'async function verifyPassword()',
    file
  );
  const mockNameSource = extractBlock(
    html,
    '// ---------------- Mock name validation ----------------',
    'function onMockNameInput()',
    file
  );
  const sandbox = {};
  vm.createContext(sandbox);
  const wrapped = `(function() { ${errorSource}\n${mockNameSource}\nreturn { formatErrorDetail, describeHttpError, mockNameValidationError }; })()`;
  return vm.runInContext(wrapped, sandbox, { filename: `${file} (extracted error/mock-name block)` });
}

const pages = {
  'admin-pdf-review.html': loadPage('admin-pdf-review.html'),
  'contractor-pdf-review.html': loadPage('contractor-pdf-review.html'),
};

let passed = 0, failed = 0;
async function test(name, fn) {
  try {
    await fn();
    passed++;
    console.log(`  ok - ${name}`);
  } catch (e) {
    failed++;
    console.log(`  FAIL - ${name}`);
    console.log('    ' + e.message);
  }
}

function fakeRes(status, body, { jsonThrows = false } = {}) {
  return {
    status,
    json: async () => {
      if (jsonThrows) throw new Error('not json');
      return body;
    },
  };
}

(async () => {
  for (const [file, mod] of Object.entries(pages)) {
    console.log(`\n=== ${file} ===`);

    console.log('formatErrorDetail: string');
    await test('returns a plain string unchanged', () => {
      assert.strictEqual(mod.formatErrorDetail('Mock name is required for the mock exam layout'), 'Mock name is required for the mock exam layout');
    });

    console.log('formatErrorDetail: list');
    await test('a FastAPI validation-error list formats as one readable line, never [object Object]', () => {
      const detail = [{ type: 'missing', loc: ['body', 'subject'], msg: 'Field required', input: {} }];
      const out = mod.formatErrorDetail(detail);
      assert.strictEqual(out, 'subject: Field required');
      assert(!out.includes('[object Object]'));
    });
    await test('multiple list entries join with a separator, never [object Object]', () => {
      const detail = [
        { loc: ['body', 'subject'], msg: 'Field required' },
        { loc: ['body', 'mock_name'], msg: 'Field required' },
      ];
      const out = mod.formatErrorDetail(detail);
      assert.strictEqual(out, 'subject: Field required; mock_name: Field required');
      assert(!out.includes('[object Object]'));
    });

    console.log('formatErrorDetail: object');
    await test('a single error object with msg formats using its loc + msg', () => {
      assert.strictEqual(mod.formatErrorDetail({ loc: ['body', 'mock_name'], msg: 'Field required' }), 'mock_name: Field required');
    });
    await test('a plain object without msg falls back to readable JSON, never [object Object]', () => {
      const out = mod.formatErrorDetail({ code: 'rate_limited', retry_after: 30 });
      assert.strictEqual(out, JSON.stringify({ code: 'rate_limited', retry_after: 30 }));
      assert(!out.includes('[object Object]'));
    });

    console.log('formatErrorDetail: empty');
    await test('null/undefined/empty string/empty list/empty object all format as empty', () => {
      assert.strictEqual(mod.formatErrorDetail(null), '');
      assert.strictEqual(mod.formatErrorDetail(undefined), '');
      assert.strictEqual(mod.formatErrorDetail(''), '');
      assert.strictEqual(mod.formatErrorDetail([]), '');
      assert.strictEqual(mod.formatErrorDetail({}), '');
    });

    console.log('describeHttpError');
    await test('a 422 validation-error body never describes as [object Object]', async () => {
      const res = fakeRes(422, { detail: [{ loc: ['body', 'subject'], msg: 'Field required' }] });
      const out = await mod.describeHttpError(res);
      assert.strictEqual(out, 'subject: Field required');
    });
    await test('a plain string error field is returned as-is', async () => {
      const res = fakeRes(400, { error: 'Mock name is required for the mock exam layout' });
      assert.strictEqual(await mod.describeHttpError(res), 'Mock name is required for the mock exam layout');
    });
    await test('an empty/unparseable body falls back to the HTTP status text', async () => {
      assert.strictEqual(await mod.describeHttpError(fakeRes(500, {})), 'HTTP 500');
      assert.strictEqual(await mod.describeHttpError(fakeRes(500, null, { jsonThrows: true })), 'HTTP 500');
    });

    console.log('mockNameValidationError');
    await test('empty or whitespace-only name has no error (covered by the separate "pick a name" hint)', () => {
      assert.strictEqual(mod.mockNameValidationError(''), null);
      assert.strictEqual(mod.mockNameValidationError('   '), null);
    });
    await test('a one-character name is rejected with a clear message', () => {
      assert.strictEqual(mod.mockNameValidationError('M'), 'Mock name must be at least 3 characters.');
    });
    await test('a two-character name is rejected', () => {
      assert(mod.mockNameValidationError('Mo'));
    });
    await test('a three-character name is accepted', () => {
      assert.strictEqual(mod.mockNameValidationError('Moc'), null);
    });
    await test('a real mock name is accepted', () => {
      assert.strictEqual(mod.mockNameValidationError('Mock Test 1'), null);
    });
  }

  console.log(`\n${passed} passed, ${failed} failed`);
  process.exit(failed ? 1 : 0);
})();
