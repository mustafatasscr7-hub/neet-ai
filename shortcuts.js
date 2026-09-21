// Shared keyboard-shortcut infrastructure (keyboard-shortcuts audit/rollout, 2026-09-20): the
// typing-context guard every page-specific shortcut handler must check first, plus a generic
// "?"-triggered cheat-sheet modal. Page-specific key bindings (which action, which key, which
// page/screen it's valid on) live in each page's own inline script -- the actions and selectors
// differ too much per page to share, same reasoning confirm-modal.js's own header gives for why
// IT is shared but page-specific confirm copy isn't. This file only holds what's common to all of
// them: the guard, and the modal that lists whatever window.PAGE_SHORTCUTS that page populated.
//
// Include via <script src="./shortcuts.js"></script>. Each page sets window.PAGE_SHORTCUTS to an
// array of { keys: 'S' | '1-4 / A-D' | '← / →', desc: '...' } objects (order = display
// order) any time before the user might press "?" -- typically right where that page's own
// shortcut handler is wired up. Pages with nothing to list (none of the shortcuts below touch
// them) simply never set it, and "?" does nothing there.

// True while focus is on anything the user could be actively typing into -- every page-specific
// shortcut handler below must return early when this is true, or a single-letter shortcut would
// fire mid-keystroke instead of typing the letter. Mirrors server.py's own DEVANAGARI_RE-style
// gate-first pattern, just guarding DOM focus instead of query language.
function isTypingTarget(el) {
  if (!el) return false;
  if (el.isContentEditable) return true;
  var tag = el.tagName;
  return tag === 'INPUT' || tag === 'TEXTAREA' || tag === 'SELECT';
}

(function () {
  var STYLE_ID = 'shortcuts-modal-shared-styles';
  if (!document.getElementById(STYLE_ID)) {
    var style = document.createElement('style');
    style.id = STYLE_ID;
    style.textContent =
      '.shortcuts-modal-overlay { position: fixed; inset: 0; background: rgba(0,0,0,0.6); z-index: 3000; display: flex; align-items: center; justify-content: center; }' +
      '.shortcuts-modal { background: #1a1a1a; border: 1px solid #2a2a2a; border-radius: 14px; padding: 22px 24px; width: 360px; max-width: calc(100vw - 32px); max-height: 80vh; overflow-y: auto; box-shadow: 0 12px 32px rgba(0,0,0,0.5); font-family: "Inter", sans-serif; }' +
      '.shortcuts-modal-header { display: flex; justify-content: space-between; align-items: center; margin-bottom: 14px; }' +
      '.shortcuts-modal-title { font-size: 15px; font-weight: 700; color: #ececec; margin: 0; }' +
      '.shortcuts-modal-close { background: none; border: none; color: #888; font-size: 18px; cursor: pointer; line-height: 1; padding: 2px; }' +
      '.shortcuts-modal-close:hover { color: #ccc; }' +
      '.shortcuts-modal-row { display: flex; justify-content: space-between; align-items: center; gap: 14px; padding: 9px 0; border-bottom: 1px solid #2a2a2a; }' +
      '.shortcuts-modal-row:last-child { border-bottom: none; }' +
      '.shortcuts-modal-desc { font-size: 13px; color: #ccc; }' +
      '.shortcuts-modal-keys { display: flex; align-items: center; gap: 5px; flex-shrink: 0; }' +
      '.shortcuts-modal-key { background: #2a2a2a; border: 1px solid #3a3a3a; border-radius: 6px; padding: 2px 8px; font-size: 12px; font-family: "SFMono-Regular", Consolas, monospace; color: #ececec; white-space: nowrap; }' +
      '.shortcuts-modal-or { font-size: 11px; color: #666; }' +
      '.shortcuts-modal-hint { font-size: 12px; color: #777; margin-top: 14px; text-align: center; }' +
      'body.light-mode .shortcuts-modal { background: #ffffff !important; border-color: #e3e3ea !important; }' +
      'body.light-mode .shortcuts-modal-title { color: #16161d !important; }' +
      'body.light-mode .shortcuts-modal-desc { color: #3a3a42 !important; }' +
      'body.light-mode .shortcuts-modal-row { border-color: #e3e3ea !important; }' +
      'body.light-mode .shortcuts-modal-key { background: #f2f2f5 !important; border-color: #d5d5dd !important; color: #16161d !important; }' +
      'body.light-mode .shortcuts-modal-hint { color: #8a8a92 !important; }';
    document.head.appendChild(style);
  }

  function closeShortcutsModal() {
    var existing = document.getElementById('shortcutsModalOverlay');
    if (existing) existing.remove();
  }
  window.closeShortcutsModal = closeShortcutsModal;

  window.showShortcutsModal = function () {
    var list = window.PAGE_SHORTCUTS || [];
    if (!list.length) return;
    closeShortcutsModal();

    var overlay = document.createElement('div');
    overlay.id = 'shortcutsModalOverlay';
    overlay.className = 'shortcuts-modal-overlay';

    var modal = document.createElement('div');
    modal.className = 'shortcuts-modal';

    var header = document.createElement('div');
    header.className = 'shortcuts-modal-header';
    var title = document.createElement('h3');
    title.className = 'shortcuts-modal-title';
    title.textContent = 'Keyboard Shortcuts';
    var closeBtn = document.createElement('button');
    closeBtn.type = 'button';
    closeBtn.className = 'shortcuts-modal-close';
    closeBtn.textContent = '✕';
    closeBtn.onclick = closeShortcutsModal;
    header.appendChild(title);
    header.appendChild(closeBtn);
    modal.appendChild(header);

    list.forEach(function (s) {
      var row = document.createElement('div');
      row.className = 'shortcuts-modal-row';
      var desc = document.createElement('span');
      desc.className = 'shortcuts-modal-desc';
      desc.textContent = s.desc;
      var keysWrap = document.createElement('div');
      keysWrap.className = 'shortcuts-modal-keys';
      String(s.keys).split('/').forEach(function (k, i) {
        if (i > 0) {
          var or = document.createElement('span');
          or.className = 'shortcuts-modal-or';
          or.textContent = 'or';
          keysWrap.appendChild(or);
        }
        var keyEl = document.createElement('span');
        keyEl.className = 'shortcuts-modal-key';
        keyEl.textContent = k.trim();
        keysWrap.appendChild(keyEl);
      });
      row.appendChild(desc);
      row.appendChild(keysWrap);
      modal.appendChild(row);
    });

    var hint = document.createElement('div');
    hint.className = 'shortcuts-modal-hint';
    hint.textContent = 'Press ? anytime to see this again';
    modal.appendChild(hint);

    overlay.appendChild(modal);
    overlay.addEventListener('mousedown', function (e) { if (e.target === overlay) closeShortcutsModal(); });
    document.body.appendChild(overlay);
  };

  // "?" opens the cheat sheet (a second press closes it, same toggle feel as most apps' own "?"
  // help overlay); Escape closes it too. Both skip entirely while the user is typing, same guard
  // every page-specific shortcut below uses -- "?" is Shift+/ on a US layout, which would
  // otherwise land a literal "?" character in whatever field has focus.
  document.addEventListener('keydown', function (e) {
    if (e.key === 'Escape') { closeShortcutsModal(); return; }
    if (e.key !== '?' || e.ctrlKey || e.metaKey || e.altKey) return;
    if (isTypingTarget(document.activeElement)) return;
    if (document.getElementById('shortcutsModalOverlay')) { closeShortcutsModal(); return; }
    e.preventDefault();
    window.showShortcutsModal();
  });
})();
