/* Tags each option image (.q-opt-img) with data-orient="portrait" (height >= width, e.g. a
   chemical structure or a square graph) or "wide" from its natural size, so
   question-images.css can give portrait images a higher minimum height cap. Linked in <head>
   right after question-images.css on every page that links that file.

   Until an image is tagged it stays display:none (see the gate rule in question-images.css),
   so it appears once at its final size instead of first at the wide cap and then growing.
   The gate is keyed on the html.q-img-orient class this script adds, so if the script ever
   fails to load, images just keep the plain cap. */
(function () {
  document.documentElement.classList.add('q-img-orient');

  function tag(img) {
    if (img.dataset.orient) return;
    if (!img.complete) return;
    if (!img.naturalWidth || !img.naturalHeight) { img.dataset.orient = 'none'; return; }
    img.dataset.orient = img.naturalHeight >= img.naturalWidth ? 'portrait' : 'wide';
  }
  function isOptImg(el) {
    return el && el.tagName === 'IMG' && el.classList.contains('q-opt-img');
  }

  // load/error don't bubble, so listen in the capture phase. A src change resets the tag.
  document.addEventListener('load', function (e) {
    if (isOptImg(e.target)) { delete e.target.dataset.orient; tag(e.target); }
  }, true);
  document.addEventListener('error', function (e) {
    if (isOptImg(e.target)) e.target.dataset.orient = 'none';
  }, true);

  // Images that finished loading before they were inserted (cached, or built in a detached
  // element) fire their load event where the document listener can't see it, so tag them on
  // insertion. MutationObserver callbacks run before the next paint.
  function scan(root) {
    if (isOptImg(root)) tag(root);
    else if (root.querySelectorAll) root.querySelectorAll('img.q-opt-img').forEach(tag);
  }
  new MutationObserver(function (records) {
    records.forEach(function (r) { r.addedNodes.forEach(scan); });
  }).observe(document.documentElement, { childList: true, subtree: true });
})();
