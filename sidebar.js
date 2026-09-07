// Shared sidebar behavior for every standalone page (pyqbank.html, savedquestions.html,
// diagram-library.html, mocktest.html, scoreboard.html). Adapted from chat.html's own sidebar JS
// so the visuals and core navigation match exactly, but DELIBERATELY narrower in scope than
// chat.html's own copy:
//   - Clicking a Pinned/Recent chat navigates to chat.html?open=<id> (chat.html loads it there)
//     rather than rendering it inline -- these pages have no chat window to render into.
//   - No drag-to-reorder, no per-item rename/pin/delete menu, no Settings modal. All of that
//     already exists and works in chat.html; opening a chat from here hands off to the one real
//     implementation instead of re-building a second one here.
// Requires: this page's own `client` (Supabase) already created, and i18n.js already loaded,
// before this script runs.

const SIDEBAR_MIN_WIDTH = 200;
const SIDEBAR_MAX_WIDTH = 400;
const SIDEBAR_DEFAULT_WIDTH = 260;

function getSavedSidebarWidth() {
  const saved = parseInt(localStorage.getItem('sidebarWidth'), 10);
  return (saved >= SIDEBAR_MIN_WIDTH && saved <= SIDEBAR_MAX_WIDTH) ? saved : SIDEBAR_DEFAULT_WIDTH;
}

function toggleSidebar() {
  const sidebar = document.querySelector('.sidebar');
  const collapsing = !sidebar.classList.contains('collapsed');
  sidebar.classList.toggle('collapsed');
  sidebar.style.width = collapsing ? '' : (getSavedSidebarWidth() + 'px');
}

function toggleSearchChats() {
  const box = document.getElementById('searchChatsBox');
  const input = document.getElementById('searchChatsInput');
  const isVisible = box.style.display !== 'none';
  box.style.display = isVisible ? 'none' : 'block';
  if (!isVisible) input.focus();
}

function displayableTitle(title) {
  const stripped = (title || '').replace(/[\s\u200B-\u200F\u2028-\u202F\uFEFF]/g, '');
  return stripped ? title.trim() : 'Untitled chat';
}

function createHistoryItem(id, title) {
  const item = document.createElement('div');
  item.className = 'history-item';
  item.id = 'conv-' + id;
  const titleSpan = document.createElement('span');
  titleSpan.className = 'item-title';
  titleSpan.textContent = displayableTitle(title);
  item.appendChild(titleSpan);
  item.addEventListener('click', function() {
    window.location.href = './chat.html?open=' + encodeURIComponent(id);
  });
  return item;
}

function positionItemMenu(menu, anchorRect, { belowInsteadOfRight = false } = {}) {
  const menuRect = menu.getBoundingClientRect();
  let left = belowInsteadOfRight ? anchorRect.left : anchorRect.right + 4;
  if (left + menuRect.width > window.innerWidth - 8) {
    left = belowInsteadOfRight ? window.innerWidth - menuRect.width - 8 : anchorRect.left - menuRect.width - 4;
  }
  left = Math.max(8, left);
  let top = belowInsteadOfRight ? anchorRect.bottom + 4 : anchorRect.top;
  if (top + menuRect.height > window.innerHeight - 8) {
    top = window.innerHeight - menuRect.height - 8;
  }
  top = Math.max(8, top);
  menu.style.left = left + 'px';
  menu.style.top = top + 'px';
}

function positionProfileMenu() {
  const menu = document.getElementById('profileMenu');
  const anchorRect = document.getElementById('profileSection').getBoundingClientRect();
  const menuWidth = 228;
  let left = Math.min(anchorRect.left, window.innerWidth - menuWidth - 8);
  left = Math.max(8, left);
  const bottom = Math.max(8, window.innerHeight - anchorRect.top + 8);
  menu.style.left = left + 'px';
  menu.style.bottom = bottom + 'px';
}

const PROFILE_SUBMENU_CONTENT = {
  language: () =>
    `<div class="item-menu-option" onclick="setLang('en')">English</div>` +
    `<div class="item-menu-option" onclick="setLang('hi')">हिंदी</div>`,
  learnMore: () =>
    `<div class="item-menu-option" onclick="window.open('./index.html','_blank')">${typeof t === 'function' ? t('chat', 'learnMoreNeetAi') : 'About NEET-AI'}</div>` +
    `<div class="item-menu-option" onclick="window.open('./privacy.html','_blank')">${typeof t === 'function' ? t('chat', 'learnMorePrivacy') : 'Privacy Policy'}</div>` +
    `<div class="item-menu-option" onclick="window.open('./terms.html','_blank')">${typeof t === 'function' ? t('chat', 'learnMoreTerms') : 'Terms of Service'}</div>`,
};

function openProfileSubmenu(key, anchorEl) {
  document.querySelectorAll('.item-menu').forEach(m => m.style.display = 'none');
  let menu = document.getElementById('profileSubmenu-' + key);
  if (!menu) {
    menu = document.createElement('div');
    menu.id = 'profileSubmenu-' + key;
    menu.className = 'item-menu';
    menu.style.display = 'none';
    document.body.appendChild(menu);
  }
  menu.innerHTML = PROFILE_SUBMENU_CONTENT[key]();
  menu.style.display = 'block';
  positionItemMenu(menu, anchorEl.getBoundingClientRect());
}

document.addEventListener('click', function(e) {
  if (!e.target.closest('#profileSection') && !e.target.closest('#profileMenu')) {
    const m = document.getElementById('profileMenu');
    if (m) m.style.display = 'none';
  }
  if (!e.target.closest('.profile-menu-item') && !e.target.closest('.item-menu')) {
    document.querySelectorAll('[id^="profileSubmenu-"]').forEach(m => m.style.display = 'none');
  }
});

function initSidebar() {
  const sidebar = document.querySelector('.sidebar');
  if (!sidebar) return;
  sidebar.style.width = getSavedSidebarWidth() + 'px';

  const searchInput = document.getElementById('searchChatsInput');
  if (searchInput) {
    searchInput.addEventListener('input', function() {
      const query = this.value.toLowerCase().trim();
      document.querySelectorAll('.history-item').forEach(item => {
        const title = item.querySelector('.item-title');
        if (!title) return;
        const match = title.textContent.toLowerCase().includes(query);
        item.style.display = match || query === '' ? 'flex' : 'none';
      });
    });
  }

  // Same fixed-anchor delta math as chat.html's own resize handle.
  (function setupSidebarResize() {
    const handle = document.getElementById('sidebarResizeHandle');
    if (!handle) return;
    let startX = 0, startWidth = 0, dragging = false;
    function onPointerMove(e) {
      if (!dragging) return;
      const next = Math.max(SIDEBAR_MIN_WIDTH, Math.min(SIDEBAR_MAX_WIDTH, startWidth + (e.clientX - startX)));
      sidebar.style.width = next + 'px';
    }
    function endDrag(e) {
      if (!dragging) return;
      dragging = false;
      sidebar.classList.remove('resizing');
      handle.classList.remove('active');
      handle.removeEventListener('pointermove', onPointerMove);
      handle.removeEventListener('pointerup', endDrag);
      handle.removeEventListener('pointercancel', endDrag);
      try { handle.releasePointerCapture(e.pointerId); } catch (err) {}
      localStorage.setItem('sidebarWidth', parseInt(sidebar.style.width, 10));
    }
    handle.addEventListener('pointerdown', function(e) {
      if (sidebar.classList.contains('collapsed')) return;
      dragging = true;
      startX = e.clientX;
      startWidth = sidebar.getBoundingClientRect().width;
      sidebar.classList.add('resizing');
      handle.classList.add('active');
      handle.setPointerCapture(e.pointerId);
      handle.addEventListener('pointermove', onPointerMove);
      handle.addEventListener('pointerup', endDrag);
      handle.addEventListener('pointercancel', endDrag);
      e.preventDefault();
    });
  })();

  const profileSection = document.getElementById('profileSection');
  if (profileSection) {
    profileSection.addEventListener('click', function(e) {
      e.stopPropagation();
      const m = document.getElementById('profileMenu');
      const willOpen = (m.style.display === 'none' || m.style.display === '');
      if (willOpen) positionProfileMenu();
      m.style.display = willOpen ? 'block' : 'none';
    });
  }

  const logoutBtn = document.getElementById('logoutBtn');
  if (logoutBtn) {
    logoutBtn.addEventListener('click', async () => {
      await client.auth.signOut();
      window.location.href = './login.html';
    });
  }

  // Populate real Pinned/Recent chats + account info, same source of truth (the `chats` table)
  // chat.html's own sidebar reads from.
  client.auth.getSession().then(async ({ data: { session } }) => {
    if (session) {
      const user = session.user;
      const name = user.user_metadata.full_name || user.email || user.phone || 'Student';
      const profileNameEl = document.querySelector('.profile-name');
      const profileAvatarEl = document.querySelector('.profile-avatar');
      if (profileNameEl) profileNameEl.textContent = name;
      if (profileAvatarEl) profileAvatarEl.textContent = name[0].toUpperCase();
      const emailEl = document.getElementById('profileMenuEmail');
      if (emailEl && user.email) {
        emailEl.textContent = user.email;
        emailEl.style.display = 'block';
      }
      document.getElementById('loginNudge').style.display = 'none';
      document.getElementById('guestMenu').style.display = 'none';
      document.getElementById('guestAvatarBtn').style.display = 'none';
      document.getElementById('profileSection').style.display = 'flex';

      try {
        const { data: planRow } = await client.from('user_plan').select('plan').eq('user_id', user.id).single();
        const plan = (planRow && planRow.plan) || 'free';
        const planEl = document.querySelector('.profile-plan');
        if (planEl) planEl.textContent = plan === 'free' ? 'Free Plan' : (plan.charAt(0).toUpperCase() + plan.slice(1) + ' Plan');
      } catch (e) {}

      const { data: chats } = await client.from('chats').select('id,title,is_pinned').eq('user_id', user.id).order('sort_order', { ascending: true });
      if (chats && chats.length > 0) {
        const pinnedList = document.getElementById('pinnedList');
        const historyList = document.getElementById('historyList');
        chats.forEach(chat => {
          const item = createHistoryItem(chat.id, chat.title);
          (chat.is_pinned ? pinnedList : historyList).appendChild(item);
        });
        if (chats.some(c => c.is_pinned)) {
          document.getElementById('pinnedLabel').style.display = 'block';
          pinnedList.style.display = 'flex';
        }
      }
    } else {
      document.getElementById('loginNudge').style.display = 'flex';
      document.getElementById('guestMenu').style.display = 'flex';
      document.getElementById('guestAvatarBtn').style.display = 'flex';
      document.getElementById('profileSection').style.display = 'none';
    }
  });
}

if (document.readyState === 'loading') {
  document.addEventListener('DOMContentLoaded', initSidebar);
} else {
  initSidebar();
}
