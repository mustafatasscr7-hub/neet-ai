// Shared "Add question above" / "Add question below" modal for the mock exam review surfaces --
// admin-pyq-preview.html's main review queue and contractor-pdf-review.html's Needs fix queue.
// Both pages already define API_BASE, authHeaders(), escapeHtml() and fileToBase64() identically
// (same inclusion pattern as confirm-modal.js) -- this file only adds window.MockQuestionAdd, it
// assumes those four already exist on the page.
//
// Include via <script src="./mock-question-add.js"></script>, then from a button's onclick:
//   MockQuestionAdd.open({
//     mockName: 'Mock Test 3', neighborPosition: 46, direction: 'above',
//     addEndpoint: '/admin/mock-question-add',       // or '/contractor/mock-question-add'
//     extraFields: { uploader_name: currentUploaderName },  // optional, merged into the POST body
//     onSaved: function(createdRow) { loadResults(); }      // called after a successful save
//   });
(function () {
  var STYLE_ID = 'mock-question-add-styles';
  if (!document.getElementById(STYLE_ID)) {
    var style = document.createElement('style');
    style.id = STYLE_ID;
    style.textContent =
      '.mqa-overlay { position: fixed; inset: 0; background: rgba(0,0,0,0.6); z-index: 3100; display: flex; align-items: center; justify-content: center; padding: 20px; overflow-y: auto; }' +
      '.mqa-modal { background: #1a1a1a; border: 1px solid #2a2a2a; border-radius: 14px; padding: 22px; width: 520px; max-width: calc(100vw - 32px); max-height: calc(100vh - 32px); overflow-y: auto; box-shadow: 0 12px 32px rgba(0,0,0,0.5); font-family: "Inter", sans-serif; }' +
      '.mqa-title { color: #ececec; font-size: 15px; font-weight: 700; margin-bottom: 4px; }' +
      '.mqa-subtitle { color: #888; font-size: 12px; margin-bottom: 16px; }' +
      '.mqa-field { margin-bottom: 12px; }' +
      '.mqa-field label { display: block; font-size: 11px; color: #999; margin-bottom: 4px; text-transform: uppercase; letter-spacing: 0.3px; }' +
      '.mqa-field input[type=text], .mqa-field input[type=number], .mqa-field textarea, .mqa-field select { width: 100%; box-sizing: border-box; background: #0d0d0d; border: 1px solid #2a2a2a; border-radius: 8px; padding: 8px 10px; color: #ececec; font-size: 13px; font-family: "Inter", sans-serif; outline: none; }' +
      '.mqa-field input:focus, .mqa-field textarea:focus, .mqa-field select:focus { border-color: #4f8ef7; }' +
      '.mqa-row { display: flex; gap: 12px; }' +
      '.mqa-row > .mqa-field { flex: 1; }' +
      '.mqa-readonly { padding: 8px 10px; font-size: 13px; color: #ccc; }' +
      '.mqa-hint { font-size: 11px; color: #888; margin-top: 4px; line-height: 1.5; }' +
      '.mqa-hint.taken { color: #f59e0b; }' +
      '.mqa-options { display: grid; grid-template-columns: 1fr 1fr; gap: 10px; }' +
      '.mqa-image-row { display: flex; align-items: center; gap: 10px; }' +
      '.mqa-image-row img { max-width: 120px; max-height: 80px; border-radius: 6px; border: 1px solid #2a2a2a; }' +
      '.mqa-remove-img { background: transparent; border: 1px solid #2a2a2a; color: #999; padding: 4px 10px; border-radius: 6px; font-size: 12px; cursor: pointer; font-family: "Inter", sans-serif; }' +
      '.mqa-error { font-size: 12px; color: #ef4444; margin-top: 10px; min-height: 0; }' +
      '.mqa-actions { display: flex; justify-content: flex-end; gap: 10px; margin-top: 18px; }' +
      '.mqa-cancel { background: transparent; border: 1px solid #2a2a2a; color: #999; padding: 8px 16px; border-radius: 8px; font-size: 13px; font-weight: 600; cursor: pointer; font-family: "Inter", sans-serif; }' +
      '.mqa-cancel:hover { border-color: #3a3a3a; color: #ccc; }' +
      '.mqa-save { background: #4f8ef7; border: none; color: white; padding: 8px 18px; border-radius: 8px; font-size: 13px; font-weight: 600; cursor: pointer; font-family: "Inter", sans-serif; }' +
      '.mqa-save:hover { background: #3f7ee0; }' +
      '.mqa-save:disabled { opacity: 0.5; cursor: not-allowed; }' +
      '.mqa-add-btn { background: transparent; border: 1px solid #2a2a2a; color: #4f8ef7; padding: 4px 10px; border-radius: 6px; font-size: 11px; font-weight: 600; cursor: pointer; font-family: "Inter", sans-serif; }' +
      '.mqa-add-btn:hover { border-color: #4f8ef7; }' +
      'body.light-mode .mqa-modal { background: #ffffff !important; border-color: #e3e3ea !important; }' +
      'body.light-mode .mqa-title, body.light-mode .mqa-readonly { color: #16161d !important; }' +
      'body.light-mode .mqa-field input, body.light-mode .mqa-field textarea, body.light-mode .mqa-field select { background: #f6f6f9 !important; border-color: #d5d5dd !important; color: #16161d !important; }';
    document.head.appendChild(style);
  }

  var DIAGRAM_UPLOAD_ENDPOINT = '/admin/pyq-diagram-upload';
  var DEFAULT_POSITIONS_ENDPOINT = '/admin/mock-question-positions';

  function subjectForPosition(pos) {
    if (!pos || pos < 1 || pos > 180) return '';
    if (pos <= 45) return 'Physics';
    if (pos <= 90) return 'Chemistry';
    return 'Biology';
  }

  function formatPositionList(positions) {
    if (!positions || !positions.length) return 'none';
    var shown = positions.slice(0, 30);
    var text = shown.join(', ');
    if (positions.length > shown.length) text += ', +' + (positions.length - shown.length) + ' more';
    return text;
  }

  function fetchJson(url, opts) {
    return fetch(url, opts).then(function (res) {
      return res.json().catch(function () { return {}; }).then(function (data) {
        return { res: res, data: data };
      });
    });
  }

  window.MockQuestionAdd = {
    open: function (opts) {
      var mockName = opts.mockName;
      var direction = opts.direction === 'above' ? 'above' : 'below';
      var neighborPosition = opts.neighborPosition;
      var addEndpoint = opts.addEndpoint;
      var positionsEndpoint = opts.positionsEndpoint || DEFAULT_POSITIONS_ENDPOINT;
      var extraFields = opts.extraFields || {};
      var onSaved = opts.onSaved || function () {};

      var candidate = direction === 'above' ? neighborPosition - 1 : neighborPosition + 1;
      var diagramUrl = null;
      var positionsUsed = [];
      var positionsFree = [];

      var existing = document.getElementById('mqaOverlay');
      if (existing) existing.remove();

      var overlay = document.createElement('div');
      overlay.id = 'mqaOverlay';
      overlay.className = 'mqa-overlay';
      overlay.innerHTML =
        '<div class="mqa-modal">' +
          '<div class="mqa-title">Add question ' + direction + ' position ' + neighborPosition + '</div>' +
          '<div class="mqa-subtitle">' + escapeHtml(mockName) + '</div>' +
          '<div class="mqa-row">' +
            '<div class="mqa-field"><label>Position (1-180)</label><input type="number" id="mqaPosition" min="1" max="180" /></div>' +
            '<div class="mqa-field"><label>Subject</label><div class="mqa-readonly" id="mqaSubject">-</div></div>' +
          '</div>' +
          '<div class="mqa-hint" id="mqaPositionHint">Checking positions...</div>' +
          '<div class="mqa-field"><label>Question</label><textarea id="mqaQuestion" rows="3"></textarea></div>' +
          '<div class="mqa-options">' +
            '<div class="mqa-field"><label>Option A</label><input type="text" id="mqaOptionA" /></div>' +
            '<div class="mqa-field"><label>Option B</label><input type="text" id="mqaOptionB" /></div>' +
            '<div class="mqa-field"><label>Option C</label><input type="text" id="mqaOptionC" /></div>' +
            '<div class="mqa-field"><label>Option D</label><input type="text" id="mqaOptionD" /></div>' +
          '</div>' +
          '<div class="mqa-field" style="max-width:160px;"><label>Correct answer</label>' +
            '<select id="mqaAnswer"><option value="">-</option><option value="A">A</option><option value="B">B</option><option value="C">C</option><option value="D">D</option></select></div>' +
          '<div class="mqa-field"><label>Image (optional -- diagram or math crop)</label>' +
            '<div class="mqa-image-row" id="mqaImageRow"><input type="file" accept="image/*" id="mqaImageFile" /></div>' +
            '<div class="mqa-hint" id="mqaImageStatus"></div></div>' +
          '<div class="mqa-error" id="mqaError"></div>' +
          '<div class="mqa-actions">' +
            '<button type="button" class="mqa-cancel" id="mqaCancel">Cancel</button>' +
            '<button type="button" class="mqa-save" id="mqaSave">Add question</button>' +
          '</div>' +
        '</div>';
      document.body.appendChild(overlay);

      var positionInput = overlay.querySelector('#mqaPosition');
      var subjectEl = overlay.querySelector('#mqaSubject');
      var hintEl = overlay.querySelector('#mqaPositionHint');
      var errorEl = overlay.querySelector('#mqaError');
      var saveBtn = overlay.querySelector('#mqaSave');
      var imageStatusEl = overlay.querySelector('#mqaImageStatus');
      var imageRow = overlay.querySelector('#mqaImageRow');

      function updateSubject() {
        subjectEl.textContent = subjectForPosition(parseInt(positionInput.value, 10)) || '-';
      }

      function renderHint() {
        if (candidate < 1 || candidate > 180 || positionsUsed.indexOf(candidate) !== -1) {
          hintEl.className = 'mqa-hint taken';
          hintEl.textContent = (candidate < 1 || candidate > 180
            ? 'No position directly ' + direction + ' this question (out of range 1-180).'
            : 'Position ' + candidate + ' is already used in this mock.') +
            ' Choose a free position. Free positions: ' + formatPositionList(positionsFree);
          positionInput.value = '';
        } else {
          hintEl.className = 'mqa-hint';
          hintEl.textContent = 'Free positions in this mock: ' + formatPositionList(positionsFree);
          positionInput.value = candidate;
        }
        updateSubject();
      }

      fetchJson(API_BASE + positionsEndpoint + '?mock_name=' + encodeURIComponent(mockName), { headers: authHeaders() })
        .then(function (r) {
          if (r.res.status === 401) { overlay.remove(); return; }
          if (!r.res.ok || r.data.error) {
            hintEl.className = 'mqa-hint taken';
            hintEl.textContent = 'Could not load positions: ' + (r.data.error || r.data.detail || ('HTTP ' + r.res.status));
            return;
          }
          positionsUsed = r.data.positions_used || [];
          positionsFree = r.data.positions_free || [];
          renderHint();
        })
        .catch(function () {
          hintEl.className = 'mqa-hint taken';
          hintEl.textContent = 'Could not reach server to check positions.';
        });

      positionInput.addEventListener('input', updateSubject);

      function wireImageInput() {
        var fileInput = imageRow.querySelector('#mqaImageFile');
        if (fileInput) fileInput.addEventListener('change', onImageFileChange);
      }

      function onImageFileChange(e) {
        var file = e.target.files && e.target.files[0];
        if (!file) return;
        imageStatusEl.textContent = 'Uploading...';
        fileToBase64(file).then(function (b64) {
          return fetchJson(API_BASE + DIAGRAM_UPLOAD_ENDPOINT, {
            method: 'POST',
            headers: authHeaders({ 'Content-Type': 'application/json' }),
            body: JSON.stringify({ filename: file.name || 'image.png', data: b64, media_type: file.type || 'image/png' })
          });
        }).then(function (r) {
          if (!r.res.ok || r.data.error) {
            imageStatusEl.textContent = 'Upload failed: ' + (r.data.error || r.data.detail || ('HTTP ' + r.res.status));
            return;
          }
          diagramUrl = r.data.url;
          imageStatusEl.textContent = '';
          renderImageRow();
        }).catch(function () {
          imageStatusEl.textContent = 'Upload failed -- check connection.';
        });
      }

      function renderImageRow() {
        if (diagramUrl) {
          imageRow.innerHTML = '<img src="' + escapeHtml(diagramUrl) + '" alt=""><button type="button" class="mqa-remove-img" id="mqaRemoveImg">Remove</button>';
          imageRow.querySelector('#mqaRemoveImg').onclick = function () { diagramUrl = null; renderImageRow(); };
        } else {
          imageRow.innerHTML = '<input type="file" accept="image/*" id="mqaImageFile" />';
          wireImageInput();
        }
      }

      wireImageInput();

      function closeModal() {
        overlay.remove();
        document.removeEventListener('keydown', onKeydown);
      }
      function onKeydown(e) { if (e.key === 'Escape') closeModal(); }
      document.addEventListener('keydown', onKeydown);
      overlay.addEventListener('mousedown', function (e) { if (e.target === overlay) closeModal(); });
      overlay.querySelector('#mqaCancel').onclick = closeModal;

      overlay.querySelector('#mqaSave').onclick = function () {
        errorEl.textContent = '';
        var position = parseInt(positionInput.value, 10);
        var question = overlay.querySelector('#mqaQuestion').value.trim();
        var a = overlay.querySelector('#mqaOptionA').value.trim();
        var b = overlay.querySelector('#mqaOptionB').value.trim();
        var c = overlay.querySelector('#mqaOptionC').value.trim();
        var d = overlay.querySelector('#mqaOptionD').value.trim();
        var answer = overlay.querySelector('#mqaAnswer').value;

        if (!position || position < 1 || position > 180) { errorEl.textContent = 'Choose a position between 1 and 180.'; return; }
        if (positionsUsed.indexOf(position) !== -1) { errorEl.textContent = 'Position ' + position + ' is already used in this mock. Free positions: ' + formatPositionList(positionsFree); return; }
        if (!question) { errorEl.textContent = 'Question text is required.'; return; }
        var options = [a, b, c, d];
        if (options.some(function (o) { return !o; })) { errorEl.textContent = 'All four options are required.'; return; }
        if (new Set(options).size < 4) { errorEl.textContent = 'Options must be different from each other.'; return; }
        if (!answer) { errorEl.textContent = 'Choose the correct answer.'; return; }

        saveBtn.disabled = true;
        saveBtn.textContent = 'Adding...';
        var payload = Object.assign({
          mock_name: mockName, mock_position: position,
          question: question, option_a: a, option_b: b, option_c: c, option_d: d,
          correct_answer: answer, diagram_url: diagramUrl
        }, extraFields);
        fetchJson(API_BASE + addEndpoint, {
          method: 'POST',
          headers: authHeaders({ 'Content-Type': 'application/json' }),
          body: JSON.stringify(payload)
        }).then(function (r) {
          if (r.res.status === 401) { closeModal(); return; }
          if (!r.res.ok || r.data.error) {
            errorEl.textContent = r.data.error || r.data.detail || ('HTTP ' + r.res.status);
            saveBtn.disabled = false;
            saveBtn.textContent = 'Add question';
            return;
          }
          closeModal();
          onSaved(r.data.created);
        }).catch(function () {
          errorEl.textContent = 'Could not reach server.';
          saveBtn.disabled = false;
          saveBtn.textContent = 'Add question';
        });
      };
    }
  };
})();
