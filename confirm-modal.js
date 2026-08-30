// Shared, reusable styled confirmation modal -- replaces the browser's native confirm() sitewide.
// Native confirm() renders as a plain OS-chrome dialog (shows the raw page hostname in its own
// title bar, can't be styled at all) which looks broken next to this app's dark theme. This is
// the same visual language mocktest.html's and personalised-test.html's own pre-existing
// showConfirmModal already used (identical CSS class names/values) -- extracted here into one
// shared file instead of being copy-pasted per page, and extended with a `danger` option (red
// confirm button) for genuinely destructive actions like Remove/Delete, which those two files
// never needed (Exit/Submit aren't destructive in the same sense a delete is).
//
// Include via <script src="./confirm-modal.js"></script> on any page -- no other setup needed,
// CSS is injected into <head> on load.
//
// Two calling conventions, both supported:
//   1. New style (returns a Promise<boolean>, true = confirmed):
//        const ok = await showConfirmModal('Remove this question block?', { confirmLabel: 'Remove', danger: true });
//        if (!ok) return;
//   2. Old style (callback-based, exactly what mocktest.html/personalised-test.html already
//      used before this file existed -- kept working unchanged, no call-site changes needed):
//        showConfirmModal('Are you sure you want to exit?', function() { ... }, 'Exit');
//
// Both styles also support Escape-to-cancel and click-outside-to-cancel, matching chat.html's
// own showDeleteConfirmModal (the other hand-built styled dialog already in this codebase).
(function () {
  var STYLE_ID = 'confirm-modal-shared-styles';
  if (!document.getElementById(STYLE_ID)) {
    var style = document.createElement('style');
    style.id = STYLE_ID;
    style.textContent =
      '.confirm-modal-overlay { position: fixed; inset: 0; background: rgba(0,0,0,0.6); z-index: 3000; display: flex; align-items: center; justify-content: center; }' +
      '.confirm-modal { background: #1a1a1a; border: 1px solid #2a2a2a; border-radius: 14px; padding: 24px; width: 340px; max-width: calc(100vw - 32px); box-shadow: 0 12px 32px rgba(0,0,0,0.5); font-family: "Inter", sans-serif; }' +
      '.confirm-modal-message { font-size: 14px; color: #ececec; line-height: 1.6; white-space: pre-line; margin-bottom: 20px; }' +
      '.confirm-modal-actions { display: flex; justify-content: flex-end; gap: 10px; }' +
      '.confirm-modal-cancel { background: transparent; border: 1px solid #2a2a2a; color: #999; padding: 8px 16px; border-radius: 8px; font-size: 13px; font-weight: 600; cursor: pointer; font-family: "Inter", sans-serif; }' +
      '.confirm-modal-cancel:hover { border-color: #3a3a3a; color: #ccc; }' +
      '.confirm-modal-ok { background: #4f8ef7; border: none; color: white; padding: 8px 18px; border-radius: 8px; font-size: 13px; font-weight: 600; cursor: pointer; font-family: "Inter", sans-serif; }' +
      '.confirm-modal-ok:hover { background: #3f7ee0; }' +
      '.confirm-modal-ok.danger { background: #ef4444; }' +
      '.confirm-modal-ok.danger:hover { background: #dc3838; }' +
      'body.light-mode .confirm-modal { background: #ffffff !important; border-color: #e3e3ea !important; }' +
      'body.light-mode .confirm-modal-message { color: #16161d !important; }' +
      'body.light-mode .confirm-modal-cancel { border-color: #d5d5dd !important; color: #55555f !important; }';
    document.head.appendChild(style);
  }

  window.showConfirmModal = function (message, onConfirmOrOpts, confirmLabel, cancelLabel) {
    var onConfirm = null;
    var danger = false;
    if (typeof onConfirmOrOpts === 'function') {
      onConfirm = onConfirmOrOpts; // old callback-based calling convention
    } else if (onConfirmOrOpts && typeof onConfirmOrOpts === 'object') {
      confirmLabel = onConfirmOrOpts.confirmLabel || confirmLabel;
      cancelLabel = onConfirmOrOpts.cancelLabel || cancelLabel;
      danger = !!onConfirmOrOpts.danger;
    }

    return new Promise(function (resolve) {
      var existing = document.getElementById('confirmModalOverlay');
      if (existing) existing.remove();

      var overlay = document.createElement('div');
      overlay.id = 'confirmModalOverlay';
      overlay.className = 'confirm-modal-overlay';
      overlay.innerHTML =
        '<div class="confirm-modal">' +
          '<div class="confirm-modal-message"></div>' +
          '<div class="confirm-modal-actions">' +
            '<button type="button" class="confirm-modal-cancel"></button>' +
            '<button type="button" class="confirm-modal-ok' + (danger ? ' danger' : '') + '"></button>' +
          '</div>' +
        '</div>';
      overlay.querySelector('.confirm-modal-message').textContent = message;
      overlay.querySelector('.confirm-modal-cancel').textContent = cancelLabel || 'Cancel';
      overlay.querySelector('.confirm-modal-ok').textContent = confirmLabel || 'OK';

      function close(result) {
        overlay.remove();
        document.removeEventListener('keydown', onKeydown);
        resolve(result);
      }
      function onKeydown(e) {
        if (e.key === 'Escape') close(false);
      }

      overlay.querySelector('.confirm-modal-cancel').onclick = function () { close(false); };
      overlay.querySelector('.confirm-modal-ok').onclick = function () {
        close(true);
        if (onConfirm) onConfirm();
      };
      overlay.addEventListener('mousedown', function (e) { if (e.target === overlay) close(false); });
      document.addEventListener('keydown', onKeydown);

      document.body.appendChild(overlay);
    });
  };
})();
