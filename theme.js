// Shared dark/light theme engine for NEET-AI. Include on every page:
// <script src="./theme.js"></script>
// Persists to the same 'theme' localStorage key chat.html's Settings panel already uses.
// Applies 'light-mode' on <body> so each page's own `body.light-mode { ... }` CSS takes over.

const THEME_KEY = 'theme';

function getTheme() {
  return localStorage.getItem(THEME_KEY) || 'dark';
}

function setTheme(theme) {
  localStorage.setItem(THEME_KEY, theme);
  applyThemeClass();
  document.dispatchEvent(new CustomEvent('themechange', { detail: { theme } }));
}

function toggleTheme() {
  setTheme(getTheme() === 'light' ? 'dark' : 'light');
}

function applyThemeClass() {
  const isLight = getTheme() === 'light';
  document.body.classList.toggle('light-mode', isLight);
  document.body.classList.toggle('dark-mode', !isLight);
}

document.addEventListener('DOMContentLoaded', applyThemeClass);
