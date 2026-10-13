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
      'body.light-mode .confirm-modal-cancel { border-color: #d5d5dd !important; color: #55555f !important; }' +
      '.confirm-modal-title { color: #ececec; }' +
      'body.light-mode .confirm-modal-title { color: #16161d !important; }' +
      'body.light-mode .confirm-modal-input { background: #f6f6f9 !important; border-color: #d5d5dd !important; color: #16161d !important; }';
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

  // Same modal, for actions serious enough to need the user to type a word first (Delete
  // Account). The confirm button stays disabled until the input matches opts.confirmWord exactly.
  // opts.onConfirm is async and runs with the modal still open (buttons disabled, label switched
  // to opts.busyLabel): resolve true to close, or an error string to show it and let the user
  // retry or cancel. Resolves true only after onConfirm succeeded.
  //   const ok = await showTypedConfirmModal({ title, message, confirmWord: 'DELETE', prompt,
  //     confirmLabel, busyLabel, cancelLabel, danger: true, onConfirm: async () => true });
  window.showTypedConfirmModal = function (opts) {
    return new Promise(function (resolve) {
      var existing = document.getElementById('confirmModalOverlay');
      if (existing) existing.remove();

      var overlay = document.createElement('div');
      overlay.id = 'confirmModalOverlay';
      overlay.className = 'confirm-modal-overlay';
      overlay.innerHTML =
        '<div class="confirm-modal" style="width:400px;">' +
          '<div class="confirm-modal-title" style="font-size:16px;font-weight:700;margin-bottom:10px;"></div>' +
          '<div class="confirm-modal-message" style="margin-bottom:16px;"></div>' +
          '<label class="confirm-modal-prompt" style="display:block;font-size:12px;color:#999;margin-bottom:6px;"></label>' +
          '<input class="confirm-modal-input" type="text" autocomplete="off" autocapitalize="characters" spellcheck="false" style="width:100%;box-sizing:border-box;background:#0d0d0d;border:1px solid #2a2a2a;border-radius:8px;padding:8px 12px;color:#ececec;font-size:13px;font-family:Inter,sans-serif;outline:none;" />' +
          '<div class="confirm-modal-error" style="display:none;font-size:12px;color:#ef4444;margin-top:8px;"></div>' +
          '<div class="confirm-modal-actions" style="margin-top:20px;">' +
            '<button type="button" class="confirm-modal-cancel"></button>' +
            '<button type="button" class="confirm-modal-ok' + (opts.danger ? ' danger' : '') + '" disabled style="opacity:0.5;cursor:not-allowed;"></button>' +
          '</div>' +
        '</div>';
      var titleEl = overlay.querySelector('.confirm-modal-title');
      var input = overlay.querySelector('.confirm-modal-input');
      var errorEl = overlay.querySelector('.confirm-modal-error');
      var cancelBtn = overlay.querySelector('.confirm-modal-cancel');
      var okBtn = overlay.querySelector('.confirm-modal-ok');
      titleEl.textContent = opts.title || '';
      overlay.querySelector('.confirm-modal-message').textContent = opts.message || '';
      overlay.querySelector('.confirm-modal-prompt').textContent = opts.prompt || '';
      cancelBtn.textContent = opts.cancelLabel || 'Cancel';
      okBtn.textContent = opts.confirmLabel || 'OK';
      var busy = false;

      function matches() { return input.value.trim() === opts.confirmWord; }
      function setOkEnabled(enabled) {
        okBtn.disabled = !enabled;
        okBtn.style.opacity = enabled ? '' : '0.5';
        okBtn.style.cursor = enabled ? '' : 'not-allowed';
      }
      function close(result) {
        overlay.remove();
        document.removeEventListener('keydown', onKeydown);
        resolve(result);
      }
      function onKeydown(e) {
        if (e.key === 'Escape' && !busy) close(false);
      }
      async function submit() {
        if (busy || !matches()) return;
        busy = true;
        errorEl.style.display = 'none';
        setOkEnabled(false);
        cancelBtn.disabled = true;
        input.disabled = true;
        okBtn.textContent = opts.busyLabel || okBtn.textContent;
        var result;
        try { result = opts.onConfirm ? await opts.onConfirm() : true; } catch (e) { result = false; }
        if (result === true) { close(true); return; }
        busy = false;
        cancelBtn.disabled = false;
        input.disabled = false;
        okBtn.textContent = opts.confirmLabel || 'OK';
        setOkEnabled(matches());
        errorEl.textContent = typeof result === 'string' ? result : 'Something went wrong. Please try again.';
        errorEl.style.display = 'block';
      }

      input.addEventListener('input', function () { setOkEnabled(matches()); });
      input.addEventListener('keydown', function (e) { if (e.key === 'Enter') submit(); });
      cancelBtn.onclick = function () { if (!busy) close(false); };
      okBtn.onclick = submit;
      overlay.addEventListener('mousedown', function (e) { if (e.target === overlay && !busy) close(false); });
      document.addEventListener('keydown', onKeydown);

      document.body.appendChild(overlay);
      input.focus();
    });
  };
})();
