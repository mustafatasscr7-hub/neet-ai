// Unit tests for the Settings > Usage panel's "Resets ..." sub-line formatter
// (formatResetText / formatUsageDuration), added when the weekly reset line switched from a
// duration ("Resets in 4 d 0 hr") to an absolute weekday+time once more than a day away
// ("Resets Mon 5:30 AM"), falling back to an hours/minutes countdown under 24h.
//
// Same no-framework convention as test-error-formatter.js/test-paste-parser.js: loads the real
// functions straight out of chat.html and settings.js via Node's `vm` module (extracted by the
// surrounding marker comments) and runs identical assertions against both copies, so the two
// duplicated implementations can't silently drift apart. t()/localizeNum() are stubbed with the
// real English template strings/identity digits -- this is a test of the date math (weekday,
// 12-hour clock, the 24h/1h thresholds, local-timezone handling), not of the i18n layer.
// Run with: node test-usage-reset-formatter.js
'use strict';
const fs = require('fs');
const path = require('path');
const vm = require('vm');
const assert = require('assert');

function extractBlock(source, startMarker, endMarker, file) {
  const startIdx = source.indexOf(startMarker);
  const endIdx = source.indexOf(endMarker, startIdx);
  assert(startIdx !== -1 && endIdx !== -1 && endIdx > startIdx,
    `could not locate block in ${file} -- did the surrounding comments move? (start=${startIdx}, end=${endIdx})`);
  return source.slice(startIdx, endIdx);
}

const TEMPLATES = {
  usageResetsIn: 'Resets in {duration}',
  usageResetsAt: 'Resets {datetime}',
  usageLessThanMinute: 'less than a minute',
  usageDayUnit: 'd',
  usageHrUnit: 'hr',
  usageMinUnit: 'min',
};

function loadFormatter(file, startMarker, endMarker) {
  const source = fs.readFileSync(path.join(__dirname, file), 'utf8');
  const block = extractBlock(source, startMarker, endMarker, file);
  const sandbox = {
    t: (_ns, key) => TEMPLATES[key],
    localizeNum: (v) => String(v),
  };
  vm.createContext(sandbox);
  const wrapped = `(function() { ${block}\nreturn { formatUsageDuration, formatResetText }; })()`;
  const mod = vm.runInContext(wrapped, sandbox, { filename: `${file} (extracted usage-reset block)` });
  // The extracted code's own `Date` global -- a distinct realm from this test file's `Date`, so
  // "now" has to be stubbed on THIS reference, not on the outer process's Date.
  mod._sandboxDate = vm.runInContext('Date', sandbox);
  return mod;
}

const pages = {
  'settings.js': loadFormatter('settings.js', '// ---------- Usage (Usage tab) ----------', 'function usageBarColor(percent) {'),
  'chat.html': loadFormatter('chat.html', '// ---------- Usage tab (Settings > Usage) ----------', 'function usageBarColor(percent) {'),
};

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

// Fixes the extracted module's own notion of "now" (local-time components, so this exercises
// the same local-timezone path the real code takes, whatever timezone this test runs in).
function withFixedNow(mod, y, mo, d, h, mi, s, fn) {
  const SandboxDate = mod._sandboxDate;
  const fixedNow = new SandboxDate(y, mo, d, h, mi, s).getTime();
  const real = SandboxDate.now;
  SandboxDate.now = () => fixedNow;
  try {
    fn();
  } finally {
    SandboxDate.now = real;
  }
}

for (const [file, mod] of Object.entries(pages)) {
  console.log(`\n=== ${file} ===`);

  console.log('formatResetText: >= 24h shows an absolute weekday + 12-hour time');
  // Wed 2026-11-25 10:00 local + 2 days 7h30m -> Fri 2026-11-27 5:30 PM local.
  withFixedNow(mod, 2026, 10, 25, 10, 0, 0, () => {
    test('more than 24h away formats as "Resets <weekday> <h>:<mm> <AM/PM>"', () => {
      const totalSeconds = 2 * 86400 + 7 * 3600 + 30 * 60;
      assert.strictEqual(mod.formatResetText(totalSeconds), 'Resets Fri 5:30 PM');
    });
  });
  // Mon 2026-11-23 23:00 local + 25h -> Wed 2026-11-25 00:00 local (midnight -> 12 AM, not 0 AM).
  withFixedNow(mod, 2026, 10, 23, 23, 0, 0, () => {
    test('midnight on the target day shows 12 AM, not 0 AM', () => {
      const totalSeconds = 25 * 3600;
      assert.strictEqual(mod.formatResetText(totalSeconds), 'Resets Wed 12:00 AM');
    });
  });

  console.log('formatResetText: under 24h falls back to an hours/minutes countdown');
  test('just under 24h away shows hours and minutes, not a day count', () => {
    assert.strictEqual(mod.formatResetText(86400 - 1), 'Resets in 23 hr 59 min');
  });
  test('over an hour away shows hours and minutes', () => {
    assert.strictEqual(mod.formatResetText(3 * 3600 + 12 * 60), 'Resets in 3 hr 12 min');
  });

  console.log('formatResetText: under 1h shows minutes only');
  test('under an hour away shows minutes only, no "0 hr"', () => {
    assert.strictEqual(mod.formatResetText(33 * 60), 'Resets in 33 min');
  });
  test('under a minute away falls back to the "less than a minute" text', () => {
    assert.strictEqual(mod.formatResetText(10), 'Resets in less than a minute');
  });

  console.log('formatResetText: local-timezone handling');
  test('the same absolute instant renders a different local weekday/time in a different zone', () => {
    const SandboxDate = mod._sandboxDate;
    const fixedNowUtc = SandboxDate.UTC(2026, 10, 25, 20, 0, 0); // 2026-11-25 20:00 UTC
    const real = SandboxDate.now;
    const realTZ = process.env.TZ;
    try {
      SandboxDate.now = () => fixedNowUtc;
      const totalSeconds = 2 * 86400; // lands on 2026-11-27 20:00 UTC

      process.env.TZ = 'Asia/Kolkata'; // UTC+5:30 -> 2026-11-28 01:30 local (Saturday)
      assert.strictEqual(mod.formatResetText(totalSeconds), 'Resets Sat 1:30 AM');

      process.env.TZ = 'America/Los_Angeles'; // UTC-8 -> 2026-11-27 12:00 local (Friday)
      assert.strictEqual(mod.formatResetText(totalSeconds), 'Resets Fri 12:00 PM');
    } finally {
      SandboxDate.now = real;
      if (realTZ === undefined) delete process.env.TZ; else process.env.TZ = realTZ;
    }
  });
}

console.log(`\n${passed} passed, ${failed} failed`);
process.exit(failed ? 1 : 0);
