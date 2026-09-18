// Shared LaTeX pre-processing applied right before every katex.renderToString call, site-wide.
// Single source of truth so a fix here (or a future one) never needs copy-pasting into the ~13
// pages that each have their own renderMathInText/parseWithMath implementation.

// Two adjacent \frac{...}{...} blocks with nothing between them (no \times, \cdot, or existing
// space) render pressed together in KaTeX, unlike standard textbook typesetting which shows a
// small gap between two multiplied fraction terms (e.g. \frac{1}{4\pi\varepsilon_0}\frac{2q}{L}).
// Each argument group tolerates one level of nested braces (\sqrt{5}, \varepsilon_{0}, v_{0}^{2})
// so it still matches real-world questions, without trying to fully balance arbitrarily nested
// LaTeX -- deeper nesting (e.g. a \frac inside a \frac's own argument) falls back to no-op rather
// than mis-spacing anything.
const FRAC_ARG = String.raw`(?:[^{}]|\{[^{}]*\})*`;
const ADJACENT_FRAC_RE = new RegExp(String.raw`(\\frac\{${FRAC_ARG}\}\{${FRAC_ARG}\})(?=\\frac)`, 'g');

function addFracSpacing(latex) {
  if (!latex) return latex;
  return latex.replace(ADJACENT_FRAC_RE, '$1\\,');
}
