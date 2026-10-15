// Shared by privacy.html, terms.html and delete-account.html. Load it in <head> (not deferred),
// so a bilingual page shows the right language on first paint.
//
// 1. SUPPORT_EMAIL is the one place the public support / grievance address lives -- change it
//    here. Fills every .support-email (text) and .support-email-link (mailto; an optional
//    data-subject attribute becomes the subject line).
// 2. English/Hindi switch, only on pages whose <html> has data-bilingual: ?lang=hi|en wins, else
//    the site-wide 'neet_lang' preference i18n.js uses. The switch changes this page and its URL
//    only, never that preference. Content is in .lang-en / .lang-hi blocks, hidden by each page's
//    own CSS keyed on html[data-lang]; titles come from data-title-en / data-title-hi.
const SUPPORT_EMAIL = 'neetai0123@gmail.com';

(function () {
  const root = document.documentElement;
  const bilingual = root.hasAttribute('data-bilingual');

  function initialLang() {
    const fromUrl = new URLSearchParams(location.search).get('lang');
    if (fromUrl === 'hi' || fromUrl === 'en') return fromUrl;
    try {
      return localStorage.getItem('neet_lang') === 'hi' ? 'hi' : 'en';
    } catch (e) {
      return 'en';
    }
  }

  function showLang(lang) {
    root.lang = lang;
    root.dataset.lang = lang;
    const title = lang === 'hi' ? root.dataset.titleHi : root.dataset.titleEn;
    if (title) document.title = title;
    document.querySelectorAll('.lang-switch button').forEach((btn) => {
      btn.setAttribute('aria-pressed', String(btn.dataset.lang === lang));
    });
  }

  if (bilingual) showLang(initialLang());

  document.addEventListener('DOMContentLoaded', () => {
    if (bilingual) {
      showLang(root.dataset.lang);
      document.querySelectorAll('.lang-switch button').forEach((btn) => {
        btn.addEventListener('click', () => {
          showLang(btn.dataset.lang);
          const url = new URL(location.href);
          url.searchParams.set('lang', btn.dataset.lang);
          history.replaceState(null, '', url);
        });
      });
    }
    document.querySelectorAll('.support-email').forEach((el) => { el.textContent = SUPPORT_EMAIL; });
    document.querySelectorAll('.support-email-link').forEach((el) => {
      const subject = el.dataset.subject;
      el.href = `mailto:${SUPPORT_EMAIL}${subject ? `?subject=${encodeURIComponent(subject)}` : ''}`;
    });
  });
})();
