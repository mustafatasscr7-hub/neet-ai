// Shared Settings modal -- ported from chat.html's own inline openSettings()/showSettingsSection()
// implementation so the "Settings" item in the profile menu works the same way on every page that
// shares that sidebar (pyqbank, scoreboard, mocktest, diagram-library, savedquestions), instead of
// just navigating to chat.html. chat.html keeps its own original inline copy untouched -- this is
// a deliberate duplication (not a shared include) to avoid any risk of regressing chat.html's
// already-live account system while porting it elsewhere.
//
// Requires (already present on every page this is loaded on): `client` (Supabase client),
// `API_BASE`, and i18n.js's t()/applyTranslations()/getLang()/setLang(). Also requires
// confirm-modal.js's showConfirmModal() (used only by deleteAccount()) and settings.css.
//
// Intentionally NOT ported: openProfileSubmenu()/PROFILE_SUBMENU_CONTENT (the profile menu's
// Language/Learn More flyout) -- sidebar.js already provides an equivalent on every page this
// file targets, and defining it again here would silently overwrite that one.

let isLoggedIn = false;
let userPlan = 'free';

// Resolved once, at load, exactly like chat.html resolves the same two facts inside its own
// client.auth.getSession() callback -- openSettings() awaits this so a fast click right after
// page load still sees the real logged-in/plan state instead of the isLoggedIn=false default.
const settingsBootstrapPromise = (async () => {
  try {
    const { data: { session } } = await client.auth.getSession();
    isLoggedIn = !!session;
    if (session) {
      const { data: planRow } = await client.from('user_plan').select('plan').eq('user_id', session.user.id).single();
      userPlan = (planRow && planRow.plan) || 'free';
    }
  } catch (e) { /* leave isLoggedIn/userPlan at their guest defaults */ }
})();

function isPro() {
  return !!userPlan && userPlan !== 'free';
}

function escapeHtml(str) {
  const div = document.createElement('div');
  div.textContent = str == null ? '' : String(str);
  return div.innerHTML;
}

async function openSettings() {
  await settingsBootstrapPromise;

  const existing = document.getElementById('settingsModal');
  if (existing) existing.remove();

  const modal = document.createElement('div');
  modal.id = 'settingsModal';
  modal.style.cssText = 'position:fixed;top:0;left:0;width:100%;height:100%;background:rgba(0,0,0,0.7);z-index:1000;display:flex;align-items:center;justify-content:center;';

  modal.innerHTML = `
    <div class="settings-modal-panel">

      <!-- Close button -->
      <button onclick="document.getElementById('settingsModal').remove()" style="position:absolute;top:16px;right:16px;background:transparent;border:none;color:#666;cursor:pointer;font-size:20px;z-index:10;">✕</button>

      <!-- Left sidebar (becomes a horizontal scrollable tab strip on mobile, see .settings-sidebar's @media rule) -->
      <div class="settings-sidebar">
        <div class="settings-sidebar-title" data-i18n="chat.settingsTitle">Settings</div>
       <div class="settings-nav-item active" onclick="showSettingsSection('general', this)" style="padding:9px 14px;border-radius:8px;cursor:pointer;font-size:13px;color:#ececec;background:#2a2a2a;display:flex;align-items:center;gap:10px;">
          <svg width="15" height="15" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><circle cx="12" cy="12" r="3"/><path d="M19.4 15a1.65 1.65 0 0 0 .33 1.82l.06.06a2 2 0 0 1-2.83 2.83l-.06-.06a1.65 1.65 0 0 0-1.82-.33 1.65 1.65 0 0 0-1 1.51V21a2 2 0 0 1-4 0v-.09A1.65 1.65 0 0 0 9 19.4a1.65 1.65 0 0 0-1.82.33l-.06.06a2 2 0 0 1-2.83-2.83l.06-.06A1.65 1.65 0 0 0 4.68 15a1.65 1.65 0 0 0-1.51-1H3a2 2 0 0 1 0-4h.09A1.65 1.65 0 0 0 4.6 9a1.65 1.65 0 0 0-.33-1.82l-.06-.06a2 2 0 0 1 2.83-2.83l.06.06A1.65 1.65 0 0 0 9 4.68a1.65 1.65 0 0 0 1-1.51V3a2 2 0 0 1 4 0v.09a1.65 1.65 0 0 0 1 1.51 1.65 1.65 0 0 0 1.82-.33l.06-.06a2 2 0 0 1 2.83 2.83l-.06.06A1.65 1.65 0 0 0 19.4 9a1.65 1.65 0 0 0 1.51 1H21a2 2 0 0 1 0 4h-.09a1.65 1.65 0 0 0-1.51 1z"/></svg>
          <span data-i18n="chat.settingsGeneral">General</span>
        </div>
        ${isLoggedIn ? `
        <div class="settings-nav-item" onclick="showSettingsSection('account', this)" style="padding:9px 14px;border-radius:8px;cursor:pointer;font-size:13px;color:#666;display:flex;align-items:center;gap:10px;">
          <svg width="15" height="15" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><path d="M20 21v-2a4 4 0 0 0-4-4H8a4 4 0 0 0-4 4v2"/><circle cx="12" cy="7" r="4"/></svg>
          <span data-i18n="chat.settingsAccount">Account</span>
        </div>
        <div class="settings-nav-item" onclick="showSettingsSection('plan', this)" style="padding:9px 14px;border-radius:8px;cursor:pointer;font-size:13px;color:#666;display:flex;align-items:center;gap:10px;">
          <svg width="15" height="15" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><polygon points="12 2 15.09 8.26 22 9.27 17 14.14 18.18 21.02 12 17.77 5.82 21.02 7 14.14 2 9.27 8.91 8.26 12 2"/></svg>
          <span data-i18n="chat.settingsPlan">Plan</span>
        </div>
        <div class="settings-nav-item" onclick="showSettingsSection('usage', this)" style="padding:9px 14px;border-radius:8px;cursor:pointer;font-size:13px;color:#666;display:flex;align-items:center;gap:10px;">
          <svg width="15" height="15" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><line x1="18" y1="20" x2="18" y2="10"/><line x1="12" y1="20" x2="12" y2="4"/><line x1="6" y1="20" x2="6" y2="14"/></svg>
          <span data-i18n="chat.settingsUsage">Usage</span>
        </div>
        <div class="settings-nav-item" onclick="showSettingsSection('privacy', this)" style="padding:9px 14px;border-radius:8px;cursor:pointer;font-size:13px;color:#666;display:flex;align-items:center;gap:10px;">
          <svg width="15" height="15" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><path d="M12 22s8-4 8-10V5l-8-3-8 3v7c0 6 8 10 8 10z"/></svg>
          <span data-i18n="chat.settingsPrivacy">Privacy</span>
        </div>
        ` : ''}
      </div>

      <!-- Right content -->
      <div class="settings-content-area" id="settingsContent">
      </div>
    </div>
  `;

  document.body.appendChild(modal);
  modal.addEventListener('click', function(e) { if (e.target === modal) modal.remove(); });
  showSettingsSection('general', document.querySelector('.settings-nav-item.active'));
}

async function showSettingsSection(section, clickedEl) {
  document.querySelectorAll('.settings-nav-item').forEach(el => {
    el.classList.remove('active');
    el.style.background = 'transparent';
    el.style.color = '#666';
  });
  if (clickedEl) {
    clickedEl.classList.add('active');
    clickedEl.style.background = '#2a2a2a';
    clickedEl.style.color = '#ececec';
  }

  const content = document.getElementById('settingsContent');

  if (section === 'general') {
    content.innerHTML = `
      <h2 style="font-size:18px;font-weight:700;color:#fff;margin-bottom:24px;" data-i18n="chat.settingsGeneral">General</h2>

      <div style="margin-bottom:32px;">
        <h3 style="font-size:13px;font-weight:600;color:#888;text-transform:uppercase;letter-spacing:0.05em;margin-bottom:16px;" data-i18n="chat.appearance">Appearance</h3>
        <div style="display:flex;align-items:center;justify-content:space-between;padding:16px 0;border-bottom:1px solid #1e1e1e;">
          <div>
            <div style="font-size:14px;color:#ececec;margin-bottom:2px;" data-i18n="chat.theme">Theme</div>
            <div style="font-size:12px;color:#555;" data-i18n="chat.themeDesc">Choose your preferred look</div>
          </div>
          ${renderSettingsDropdown('theme')}
        </div>
        <div style="display:flex;align-items:center;justify-content:space-between;padding:16px 0;border-bottom:1px solid #1e1e1e;">
          <div>
            <div style="font-size:14px;color:#ececec;margin-bottom:2px;" data-i18n="chat.fontSize">Font Size</div>
            <div style="font-size:12px;color:#555;" data-i18n="chat.fontSizeDesc">Adjust text size across the app</div>
          </div>
          ${renderSettingsDropdown('fontSize')}
        </div>
        <div style="display:flex;align-items:center;justify-content:space-between;padding:16px 0;border-bottom:1px solid #1e1e1e;">
          <div>
            <div style="font-size:14px;color:#ececec;margin-bottom:2px;" data-i18n="chat.chatFont">Chat Font</div>
            <div style="font-size:12px;color:#555;" data-i18n="chat.chatFontDesc">Choose the font used for AI responses</div>
          </div>
          ${renderSettingsDropdown('chatFont')}
        </div>
      </div>

     <div style="margin-bottom:32px;">
        <h3 style="font-size:13px;font-weight:600;color:#888;text-transform:uppercase;letter-spacing:0.05em;margin-bottom:16px;" data-i18n="chat.aiBehaviour">AI Behaviour</h3>
        <div style="display:flex;align-items:center;justify-content:space-between;padding:16px 0;border-bottom:1px solid #1e1e1e;">
          <div>
            <div style="font-size:14px;color:#ececec;margin-bottom:2px;" data-i18n="chat.answerStyle">Answer Style</div>
            <div style="font-size:12px;color:#555;" data-i18n="chat.answerStyleDesc">How detailed should AI responses be</div>
          </div>
          ${renderSettingsDropdown('answerStyle')}
        </div>
        ${isLoggedIn ? `
        <div style="display:flex;align-items:center;justify-content:space-between;padding:16px 0;border-bottom:1px solid #1e1e1e;">
          <div>
            <div style="font-size:14px;color:#ececec;margin-bottom:2px;" data-i18n="chat.personalize">Personalize my AI tutor</div>
            <div style="font-size:12px;color:#555;" data-i18n="chat.personalizeDesc">Use your mock test scores and mistake history to tailor answers to you</div>
          </div>
          <label class="toggle-switch">
            <input type="checkbox" id="personalizeToggle" ${localStorage.getItem('personalize') !== 'false' ? 'checked' : ''} onchange="setPersonalize(this.checked)"/>
            <span class="toggle-track"><span class="toggle-thumb"></span></span>
          </label>
        </div>
        ` : ''}
      </div>

      <div style="margin-bottom:32px;">
        <h3 style="font-size:13px;font-weight:600;color:#888;text-transform:uppercase;letter-spacing:0.05em;margin-bottom:16px;" data-i18n="chat.language">Language</h3>
        <div style="display:flex;align-items:center;justify-content:space-between;padding:16px 0;border-bottom:1px solid #1e1e1e;">
          <div>
            <div style="font-size:14px;color:#ececec;margin-bottom:2px;" data-i18n="chat.appLanguage">App Language</div>
            <div style="font-size:12px;color:#555;" data-i18n="chat.appLanguageDesc">Switch the whole site and AI answers between English and Hindi</div>
          </div>
          ${renderSettingsDropdown('language')}
        </div>
      </div>
    `;
    applyTranslations();
  } else if (section === 'account') {
    content.innerHTML = `
      <h2 style="font-size:18px;font-weight:700;color:#fff;margin-bottom:24px;" data-i18n="chat.settingsAccount">Account</h2>
      <div style="margin-bottom:40px;">
        <h3 style="font-size:13px;font-weight:600;color:#888;text-transform:uppercase;letter-spacing:0.05em;margin-bottom:16px;" data-i18n="chat.profile">Profile</h3>
        <div class="account-row">
          <div class="account-row-label" data-i18n="chat.avatar">Avatar</div>
          <div class="account-row-avatar">${(document.querySelector('.profile-avatar')?.textContent || '?').trim()}</div>
        </div>
        <div class="account-row">
          <div class="account-row-label" data-i18n="chat.displayName">Display Name</div>
          <div id="displayNameInputGroup" style="display:flex;gap:8px;align-items:center;">
            <input id="displayNameInput" type="text" class="account-pill-input" value="${document.querySelector('.profile-name')?.textContent || ''}" oninput="onDisplayNameInput()" />
            <button id="displayNameSaveBtn" class="dn-save-btn" onclick="saveDisplayName()" style="padding:8px 16px;border-radius:999px;background:linear-gradient(135deg,#4f8ef7,#7c3aed);border:none;color:white;cursor:pointer;font-size:12px;font-family:Inter,sans-serif;flex-shrink:0;">Save</button>
          </div>
        </div>
        ${(() => {
          const email = document.getElementById('profileMenuEmail')?.textContent || '';
          const isMax = userPlan === 'max';
          const planLabel = !isLoggedIn ? '' : isMax ? t('chat', 'maxPlan') : isPro() ? t('chat', 'proPlan') : t('chat', 'free');
          return `
        <div class="account-row">
          <div class="account-row-label" data-i18n="chat.email">Email</div>
          <div style="display:flex;align-items:center;gap:8px;min-width:0;">
            ${email ? `<span class="account-pill-value" style="min-width:0;overflow:hidden;text-overflow:ellipsis;white-space:nowrap;" title="${escapeHtml(email)}">${escapeHtml(email)}</span>` : ''}
            ${planLabel ? `<span class="account-pill-value" style="flex-shrink:0;">${escapeHtml(planLabel)}</span>` : ''}
            ${(!email && !planLabel) ? `<span class="account-pill-value">Not logged in</span>` : ''}
          </div>
        </div>`;
        })()}
        <div class="account-row">
          <div class="account-row-label" data-i18n="chat.password">Password</div>
          <button onclick="resetPassword()" class="account-pill-btn" data-i18n="chat.sendResetEmail">Send Reset Email</button>
        </div>
        <div class="account-row">
          <div>
            <div class="account-row-label" data-i18n="chat.phoneNumber">Phone Number</div>
            <div class="account-row-sub" id="phoneLinkStatus">…</div>
          </div>
          <div id="phoneLinkAction"></div>
        </div>
        <div id="phoneLinkPanel" style="display:none;padding:12px 0;border-bottom:1px solid rgba(255,255,255,0.07);"></div>
      </div>

      <div style="margin-bottom:32px;">
        <h3 style="font-size:13px;font-weight:600;color:#888;text-transform:uppercase;letter-spacing:0.05em;margin-bottom:16px;" data-i18n="chat.referrals">Referrals</h3>
        <div id="referralContainer">
          <div style="font-size:13px;color:#555;padding:16px 0;">Loading…</div>
        </div>
      </div>

      <div style="margin-bottom:32px;">
        <h3 style="font-size:13px;font-weight:600;color:#888;text-transform:uppercase;letter-spacing:0.05em;margin-bottom:16px;" data-i18n="chat.activeSessions">Active Sessions</h3>
        <div id="activeSessionsContainer">
          <div style="font-size:13px;color:#555;padding:16px 0;">Loading…</div>
        </div>
      </div>
    `;
    applyTranslations();
    renderPhoneLinkStatus();
    initDisplayNameSaveState();
    renderActiveSessions();
    renderReferralSection();
  } else if (section === 'plan') {
    const pro = isPro();
    const isMax = userPlan === 'max';
    content.innerHTML = `
      <h2 style="font-size:18px;font-weight:700;color:#fff;margin-bottom:24px;" data-i18n="chat.settingsPlan">Plan</h2>
      <div style="background:#111;border:1px solid #1e1e1e;border-radius:12px;padding:24px;margin-bottom:16px;">
        <div style="font-size:13px;color:#555;margin-bottom:4px;" data-i18n="chat.currentPlan">Current Plan</div>
        ${isMax ? `
        <div style="font-size:22px;font-weight:700;color:#fff;margin-bottom:8px;" data-i18n="chat.maxPlan">Max Plan</div>
        <div style="font-size:13px;color:#666;margin-bottom:20px;" data-i18n="chat.maxPlanDesc">Truly unlimited · Every feature</div>
        ` : pro ? `
        <div style="font-size:22px;font-weight:700;color:#fff;margin-bottom:8px;" data-i18n="chat.proPlan">Pro Plan</div>
        <div style="font-size:13px;color:#666;margin-bottom:20px;" data-i18n="chat.proPlanDesc">Generous daily usage · All 3 subjects unlocked</div>
        ` : `
        <div style="font-size:22px;font-weight:700;color:#fff;margin-bottom:8px;" data-i18n="chat.free">Free</div>
        <div style="font-size:13px;color:#666;margin-bottom:20px;" data-i18n="chat.freePlanDesc">Limited daily doubts · Basic access</div>
        `}
        <button onclick="window.open('./pricing.html','_blank')" style="background:${pro ? 'transparent' : '#4f8ef7'};border:${pro ? '1px solid #2a2a2a' : 'none'};color:${pro ? '#ececec' : 'white'};padding:10px 24px;border-radius:8px;cursor:pointer;font-size:13px;font-weight:600;font-family:Inter,sans-serif;" data-i18n="${isMax ? 'chat.viewPlans' : pro ? 'chat.upgradeToMaxShort' : 'chat.upgradeToProShort'}">${isMax ? 'View Plans' : pro ? '⭐ Upgrade to Max' : '⭐ Upgrade to Pro'}</button>
      </div>
      ${isMax ? '' : '<div style="font-size:12px;color:#333;text-align:center;" data-i18n="chat.razorpayNote">Payments powered by Razorpay · Launching soon</div>'}
    `;
    applyTranslations();
  } else if (section === 'usage') {
    const planLabel = userPlan === 'max' ? t('chat', 'usageMaxName') : userPlan === 'pro' ? t('chat', 'usageProName') : t('chat', 'free');
    content.innerHTML = `
      <div style="display:flex;align-items:baseline;justify-content:space-between;margin-bottom:24px;flex-wrap:wrap;gap:8px;">
        <h2 style="font-size:18px;font-weight:700;color:#fff;display:flex;align-items:center;gap:8px;">
          <span data-i18n="chat.settingsUsage">Usage</span>
          <span class="usage-plan-badge">${planLabel}</span>
        </h2>
        <div style="display:flex;align-items:center;gap:6px;font-size:12px;color:#555;">
          <span id="usageLastUpdated" data-i18n="chat.usageLastUpdatedJustNow">Last updated: just now</span>
          <button onclick="renderUsageSection(true)" data-i18n-title="chat.usageRefresh" title="Refresh" style="background:transparent;border:none;color:#666;cursor:pointer;padding:2px;display:flex;align-items:center;">
            <svg width="13" height="13" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><polyline points="23 4 23 10 17 10"/><polyline points="1 20 1 14 7 14"/><path d="M20.49 9A9 9 0 0 0 5.64 5.64L1 10m22 4l-4.64 4.36A9 9 0 0 1 3.51 15"/></svg>
          </button>
        </div>
      </div>
      <div id="usageSummaryContainer">
        <div style="font-size:13px;color:#555;padding:16px 0;">Loading…</div>
      </div>
    `;
    applyTranslations();
    renderUsageSection();
  } else if (section === 'privacy') {
    content.innerHTML = `
      <h2 style="font-size:18px;font-weight:700;color:#fff;margin-bottom:24px;" data-i18n="chat.settingsPrivacy">Privacy</h2>
      <div style="margin-bottom:32px;">
        <h3 style="font-size:13px;font-weight:600;color:#888;text-transform:uppercase;letter-spacing:0.05em;margin-bottom:16px;" data-i18n="chat.data">Data</h3>
        <div style="display:flex;align-items:center;justify-content:space-between;padding:16px 0;border-bottom:1px solid #1e1e1e;">
          <div>
            <div style="font-size:14px;color:#ececec;margin-bottom:2px;" data-i18n="chat.chatHistory">Chat History</div>
            <div style="font-size:12px;color:#555;" data-i18n="chat.chatHistoryDesc">All your chats are saved securely</div>
          </div>
          <div style="font-size:12px;color:#4ade80;" data-i18n="chat.protected">Protected</div>
        </div>
        <div style="display:flex;align-items:center;justify-content:space-between;padding:16px 0;border-bottom:1px solid #1e1e1e;">
          <div>
            <div style="font-size:14px;color:#ececec;margin-bottom:2px;" data-i18n="chat.deleteAccount">Delete Account</div>
            <div style="font-size:12px;color:#555;" data-i18n="chat.deleteAccountDesc">Permanently delete your account and data</div>
          </div>
          <button onclick="deleteAccount()" style="padding:6px 14px;border-radius:8px;border:1px solid #ef4444;background:transparent;color:#ef4444;cursor:pointer;font-size:12px;font-family:Inter,sans-serif;" data-i18n="chat.deleteAccount">Delete Account</button>
        </div>
      </div>
    `;
    applyTranslations();
  }
}

function setAppLanguage(lang) {
  setLang(lang);
}

function setTheme(theme) {
  localStorage.setItem('theme', theme);
  const root = document.documentElement;
  if (theme === 'light') {
    root.style.setProperty('--bg', '#f5f5f5');
    root.style.setProperty('--bg2', '#ffffff');
    root.style.setProperty('--text', '#0d0d0d');
    root.style.setProperty('--border', '#e0e0e0');
    root.style.setProperty('--sidebar', '#ebebeb');
    document.body.classList.add('light-mode');
    document.body.classList.remove('dark-mode');
  } else {
    root.style.setProperty('--bg', '#0d0d0d');
    root.style.setProperty('--bg2', '#111111');
    root.style.setProperty('--text', '#ececec');
    root.style.setProperty('--border', '#1e1e1e');
    root.style.setProperty('--sidebar', '#111111');
    document.body.classList.add('dark-mode');
    document.body.classList.remove('light-mode');
  }
}

function setAnswerStyle(style) {
  localStorage.setItem('answerStyle', style);
}

function setPersonalize(enabled) {
  localStorage.setItem('personalize', enabled ? 'true' : 'false');
}

function setFontSize(size) {
  const sizes = { small: '13px', medium: '14px', large: '16px' };
  document.documentElement.style.setProperty('--chat-font-size', sizes[size] || sizes.medium);
  localStorage.setItem('fontSize', size);
}

// Scoped to --chat-font-family (read only by .message-bubble.ai on chat.html) -- on pages with
// no chat bubbles this is a harmless no-op, same as Answer Style having no visible surface here.
const CHAT_FONT_STACKS = {
  sans: "'Inter', sans-serif",
  serif: "'Source Serif 4', Georgia, serif",
  system: "-apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, Helvetica, Arial, sans-serif",
  dyslexic: "'OpenDyslexic', 'Inter', sans-serif"
};
function setChatFont(font) {
  document.documentElement.style.setProperty('--chat-font-family', CHAT_FONT_STACKS[font] || CHAT_FONT_STACKS.sans);
  localStorage.setItem('chatFont', font);
}

function dropdownOptionLabel(opt) {
  return opt.label || t('chat', opt.labelKey);
}

const THEME_ICONS = {
  dark: '<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M21 12.79A9 9 0 1 1 11.21 3 7 7 0 0 0 21 12.79z"/></svg>',
  light: '<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><circle cx="12" cy="12" r="5"/><line x1="12" y1="1" x2="12" y2="3"/><line x1="12" y1="21" x2="12" y2="23"/><line x1="4.22" y1="4.22" x2="5.64" y2="5.64"/><line x1="18.36" y1="18.36" x2="19.78" y2="19.78"/><line x1="1" y1="12" x2="3" y2="12"/><line x1="21" y1="12" x2="23" y2="12"/><line x1="4.22" y1="19.78" x2="5.64" y2="18.36"/><line x1="18.36" y1="5.64" x2="19.78" y2="4.22"/></svg>'
};

const SETTINGS_DROPDOWNS = {
  theme: {
    options: [{ value: 'dark', labelKey: 'themeDark', icon: THEME_ICONS.dark }, { value: 'light', labelKey: 'themeLight', icon: THEME_ICONS.light }],
    get: () => localStorage.getItem('theme') || 'dark',
    set: (v) => setTheme(v)
  },
  fontSize: {
    options: [{ value: 'small', labelKey: 'small' }, { value: 'medium', labelKey: 'medium' }, { value: 'large', labelKey: 'large' }],
    get: () => localStorage.getItem('fontSize') || 'medium',
    set: (v) => setFontSize(v)
  },
  chatFont: {
    options: [
      { value: 'serif', label: 'NEET-AI Serif' },
      { value: 'sans', label: 'NEET-AI Sans' },
      { value: 'system', label: 'System' },
      { value: 'dyslexic', label: 'Dyslexic Friendly' }
    ],
    get: () => localStorage.getItem('chatFont') || 'sans',
    set: (v) => setChatFont(v)
  },
  answerStyle: {
    options: [{ value: 'concise', labelKey: 'concise' }, { value: 'detailed', labelKey: 'detailed' }],
    get: () => localStorage.getItem('answerStyle') || 'detailed',
    set: (v) => setAnswerStyle(v)
  },
  language: {
    options: [{ value: 'en', label: 'English' }, { value: 'hi', label: 'हिंदी' }],
    get: () => getLang(),
    set: (v) => setAppLanguage(v)
  }
};

function renderSettingsDropdown(key) {
  const cfg = SETTINGS_DROPDOWNS[key];
  const current = cfg.get();
  const currentOpt = cfg.options.find(o => o.value === current) || cfg.options[0];
  return `
    <div class="settings-dropdown" data-dropdown-key="${key}">
      <button type="button" class="dropdown-trigger" onclick="event.stopPropagation(); toggleSettingsDropdown('${key}')">
        <span class="dropdown-option-label">${currentOpt.icon ? `<span class="item-menu-icon">${currentOpt.icon}</span>` : ''}<span>${dropdownOptionLabel(currentOpt)}</span></span>
        <svg class="dropdown-chevron" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><polyline points="6 9 12 15 18 9"/></svg>
      </button>
      <div class="dropdown-menu">
        ${cfg.options.map(o => `
          <div class="dropdown-option${o.value === current ? ' selected' : ''}" onclick="event.stopPropagation(); chooseSettingsDropdown('${key}','${o.value}')">
            <span class="dropdown-option-label">${o.icon ? `<span class="item-menu-icon">${o.icon}</span>` : ''}<span>${dropdownOptionLabel(o)}</span></span>
            ${o.value === current ? '<span class="dropdown-check">✓</span>' : ''}
          </div>
        `).join('')}
      </div>
    </div>
  `;
}

function toggleSettingsDropdown(key) {
  const el = document.querySelector(`.settings-dropdown[data-dropdown-key="${key}"]`);
  if (!el) return;
  const isOpen = el.classList.contains('open');
  document.querySelectorAll('.settings-dropdown.open').forEach(d => d.classList.remove('open'));
  if (!isOpen) el.classList.add('open');
}

function refreshSettingsDropdown(key) {
  const el = document.querySelector(`.settings-dropdown[data-dropdown-key="${key}"]`);
  if (!el) return;
  el.outerHTML = renderSettingsDropdown(key);
}

function chooseSettingsDropdown(key, value) {
  SETTINGS_DROPDOWNS[key].set(value);
  refreshSettingsDropdown(key);
}

document.addEventListener('click', function(e) {
  if (!e.target.closest('.settings-dropdown')) {
    document.querySelectorAll('.settings-dropdown.open').forEach(d => d.classList.remove('open'));
  }
});

document.addEventListener('keydown', function(e) {
  if (e.key === 'Escape') {
    document.querySelectorAll('.settings-dropdown.open').forEach(d => d.classList.remove('open'));
  }
});

// Language affects every dropdown's labels, not just its own -- refresh all of them when it
// changes. (chat.html's version also calls renderWelcomeGreeting(), a welcome-screen-only concept
// that doesn't exist on these pages, so that call is intentionally dropped here.)
document.addEventListener('langchange', function() {
  if (document.getElementById('settingsModal')) {
    Object.keys(SETTINGS_DROPDOWNS).forEach(refreshSettingsDropdown);
  }
});

// Tracks the last-saved value so input can be compared against it on every keystroke -- re-set
// whenever the Account tab (re)renders, since content.innerHTML rebuilds the field fresh each
// time and a stale baseline from a previous render would show a false "changed".
let displayNameSavedValue = '';

function initDisplayNameSaveState() {
  const input = document.getElementById('displayNameInput');
  if (!input) return;
  displayNameSavedValue = input.value;
}

function onDisplayNameInput() {
  const btn = document.getElementById('displayNameSaveBtn');
  const input = document.getElementById('displayNameInput');
  if (!btn || !input) return;
  btn.classList.remove('success', 'error');
  btn.classList.toggle('visible', input.value.trim() !== displayNameSavedValue.trim());
}

function saveDisplayName() {
  const btn = document.getElementById('displayNameSaveBtn');
  const input = document.getElementById('displayNameInput');
  const name = input.value.trim();
  if (!name) return;
  btn.disabled = true;
  btn.classList.remove('success', 'error');
  btn.textContent = 'Saving...';
  try {
    document.querySelector('.profile-name').textContent = name;
    document.querySelector('.profile-avatar').textContent = name[0].toUpperCase();
    displayNameSavedValue = name;
    setTimeout(() => {
      btn.classList.add('success');
      btn.textContent = '✓ Saved';
      setTimeout(() => {
        btn.disabled = false;
        btn.classList.remove('success', 'visible');
        btn.textContent = 'Save';
      }, 1300);
    }, 250);
  } catch (e) {
    btn.disabled = false;
    btn.classList.add('error');
    btn.textContent = 'Failed, retry';
  }
}

async function resetPassword() {
  const { data: { session } } = await client.auth.getSession();
  if (!session) { alert('Please log in first.'); return; }
  const { error } = await client.auth.resetPasswordForEmail(session.user.email);
  if (error) { alert('Error: ' + error.message); return; }
  alert('Password reset email sent!');
}

// ---------- Phone linking (attach a phone number to the CURRENT logged-in account) ----------
let pendingLinkPhone = '';

async function renderPhoneLinkStatus() {
  const statusEl = document.getElementById('phoneLinkStatus');
  const actionEl = document.getElementById('phoneLinkAction');
  if (!statusEl || !actionEl) return;
  const { data: { session } } = await client.auth.getSession();
  const phone = session?.user?.phone;
  if (phone) {
    statusEl.textContent = '+' + phone;
    actionEl.innerHTML = '';
  } else {
    statusEl.textContent = t('chat', 'phoneNotLinked');
    actionEl.innerHTML = `<button onclick="showPhoneLinkPanel()" class="account-pill-btn" data-i18n="chat.linkPhone">Link phone number</button>`;
    applyTranslations();
  }
}

function showPhoneLinkPanel() {
  const panel = document.getElementById('phoneLinkPanel');
  panel.style.display = 'block';
  panel.innerHTML = `
    <div style="display:flex;gap:8px;">
      <select id="linkPhoneCountryCode" style="background:#0d0d0d;border:1px solid #2a2a2a;border-radius:8px;padding:6px 10px;color:#ececec;font-size:13px;font-family:Inter,sans-serif;">
        <option value="+91">+91</option>
        <option value="+1">+1</option>
        <option value="+965">+965</option>
      </select>
      <input id="linkPhoneInput" type="tel" data-i18n-placeholder="login.phonePlaceholder" placeholder="Phone number" style="flex:1;background:#0d0d0d;border:1px solid #2a2a2a;border-radius:8px;padding:6px 12px;color:#ececec;font-size:13px;font-family:Inter,sans-serif;outline:none;" />
      <button onclick="sendLinkPhoneOtp()" style="padding:6px 14px;border-radius:8px;background:#4f8ef7;border:none;color:white;cursor:pointer;font-size:12px;font-family:Inter,sans-serif;" data-i18n="chat.sendCode">Send Code</button>
    </div>
    <div id="phoneLinkMsg" style="font-size:12px;margin-top:8px;"></div>
  `;
  applyTranslations();
}

async function sendLinkPhoneOtp() {
  const code = document.getElementById('linkPhoneCountryCode').value;
  const local = document.getElementById('linkPhoneInput').value.trim().replace(/\D/g, '');
  const msgEl = document.getElementById('phoneLinkMsg');
  if (!local) { msgEl.style.color = '#ef4444'; msgEl.textContent = t('login', 'enterPhone'); return; }
  const phone = code + local;
  const { error } = await client.auth.updateUser({ phone });
  if (error) { msgEl.style.color = '#ef4444'; msgEl.textContent = error.message; return; }
  pendingLinkPhone = phone;
  const panel = document.getElementById('phoneLinkPanel');
  panel.innerHTML = `
    <div style="display:flex;gap:8px;">
      <input id="linkPhoneOtpInput" type="text" maxlength="6" placeholder="123456" style="flex:1;background:#0d0d0d;border:1px solid #2a2a2a;border-radius:8px;padding:6px 12px;color:#ececec;font-size:13px;font-family:Inter,sans-serif;outline:none;" />
      <button onclick="verifyLinkPhoneOtp()" style="padding:6px 14px;border-radius:8px;background:#4f8ef7;border:none;color:white;cursor:pointer;font-size:12px;font-family:Inter,sans-serif;" data-i18n="chat.verifyCode">Verify</button>
    </div>
    <div id="phoneLinkMsg" style="font-size:12px;margin-top:8px;color:#555;">${t('login', 'otpSentTo')} ${pendingLinkPhone}</div>
  `;
  applyTranslations();
}

async function verifyLinkPhoneOtp() {
  const token = document.getElementById('linkPhoneOtpInput').value.trim();
  const msgEl = document.getElementById('phoneLinkMsg');
  const { error } = await client.auth.verifyOtp({ phone: pendingLinkPhone, token, type: 'phone_change' });
  if (error) { msgEl.style.color = '#ef4444'; msgEl.textContent = t('login', 'invalidOtp'); return; }
  document.getElementById('phoneLinkPanel').style.display = 'none';
  document.getElementById('phoneLinkPanel').innerHTML = '';
  await renderPhoneLinkStatus();
  alert(t('chat', 'phoneLinkedSuccess'));
}

async function deleteAccount() {
  if (!(await showConfirmModal('Are you sure? This will permanently delete your account and all data.', { confirmLabel: 'Delete Account', danger: true }))) return;
  alert('Account deletion coming soon. Contact support for now.');
}

// ---------- Active sessions (Account tab) ----------
function getDeviceId() {
  let id = localStorage.getItem('deviceId');
  if (!id) {
    id = (crypto.randomUUID ? crypto.randomUUID() : 'dev-' + Date.now() + '-' + Math.random().toString(36).slice(2));
    localStorage.setItem('deviceId', id);
  }
  return id;
}

function formatSessionTime(iso) {
  const d = new Date(iso);
  const diffMs = Date.now() - d.getTime();
  const mins = Math.floor(diffMs / 60000);
  if (mins < 1) return 'Just now';
  if (mins < 60) return `${mins} minute${mins === 1 ? '' : 's'} ago`;
  const hours = Math.floor(mins / 60);
  if (hours < 24) return `${hours} hour${hours === 1 ? '' : 's'} ago`;
  const days = Math.floor(hours / 24);
  if (days < 7) return `${days} day${days === 1 ? '' : 's'} ago`;
  return d.toLocaleDateString(undefined, { year: 'numeric', month: 'short', day: 'numeric' });
}

async function renderActiveSessions() {
  const container = document.getElementById('activeSessionsContainer');
  if (!container) return;
  const { data: { session } } = await client.auth.getSession();
  if (!session) return;
  try {
    const res = await fetch(`${API_BASE}/session/list`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ user_id: session.user.id, device_id: getDeviceId() })
    });
    const data = await res.json();
    const sessions = data.sessions || [];
    if (document.getElementById('activeSessionsContainer') !== container) return; // settings tab changed mid-fetch

    if (sessions.length === 0) {
      container.innerHTML = `<div style="font-size:13px;color:#555;padding:16px 0;">No active sessions found.</div>`;
      return;
    }

    container.innerHTML = `
      <div class="sessions-table">
        <div class="sessions-row sessions-head">
          <div>Device</div><div>Location</div><div>Created</div><div>Updated</div><div></div>
        </div>
        ${sessions.map(s => `
          <div class="sessions-row">
            <div class="sessions-device-cell">
              <span class="sessions-device-name" title="${escapeHtml(s.device_label)}">${escapeHtml(s.device_label)}</span>
              ${s.is_current ? `<span class="sessions-current-badge">Current</span>` : ''}
            </div>
            <div class="sessions-cell muted sessions-meta" data-label="Location" title="${escapeHtml(s.location)}">${escapeHtml(s.location)}</div>
            <div class="sessions-cell muted sessions-meta" data-label="Created">${formatSessionTime(s.created_at)}</div>
            <div class="sessions-cell muted sessions-meta" data-label="Updated">${formatSessionTime(s.last_active_at)}</div>
            <div>
              ${s.is_current ? '' : `<button class="sessions-logout-btn" onclick="logoutSession('${s.device_id}', this)">Log out</button>`}
            </div>
          </div>
        `).join('')}
      </div>
    `;
  } catch (e) {
    container.innerHTML = `<div style="font-size:13px;color:#555;padding:16px 0;">Couldn't load sessions.</div>`;
  }
}

async function logoutSession(targetDeviceId, btn) {
  const { data: { session } } = await client.auth.getSession();
  if (!session) return;
  if (btn) { btn.disabled = true; btn.textContent = '…'; }
  try {
    const res = await fetch(`${API_BASE}/session/logout`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ user_id: session.user.id, target_device_id: targetDeviceId })
    });
    if (!res.ok) throw new Error('logout failed');
  } catch (e) {
    alert("Couldn't log out that session. Please try again.");
  }
  renderActiveSessions();
}

// ---------- Referrals (Account tab) ----------
function copyReferralValue(btn, value) {
  navigator.clipboard.writeText(value).then(() => {
    const original = btn.textContent;
    btn.textContent = 'Copied!';
    setTimeout(() => { btn.textContent = original; }, 1500);
  });
}

async function renderReferralSection() {
  const container = document.getElementById('referralContainer');
  if (!container) return;
  const { data: { session } } = await client.auth.getSession();
  if (!session) return;
  try {
    const res = await fetch(`${API_BASE}/referral/status?user_id=${encodeURIComponent(session.user.id)}`);
    const data = await res.json();
    if (document.getElementById('referralContainer') !== container) return; // settings tab changed mid-fetch
    if (data.error) throw new Error(data.error);

    container.innerHTML = `
      <div class="account-row">
        <div>
          <div class="account-row-label" data-i18n="chat.referralCode">Your referral code</div>
          <div class="account-row-sub" data-i18n="chat.referralCodeDesc">Both of you get 50,000 bonus tokens once they subscribe to Pro or Max</div>
        </div>
        <div style="display:flex;align-items:center;gap:10px;">
          <span class="account-pill-value" style="font-family:'Courier New',monospace;font-weight:700;letter-spacing:2px;font-size:15px;">${escapeHtml(data.code)}</span>
          <button class="account-pill-btn" onclick="copyReferralValue(this, '${data.code}')" data-i18n="chat.copyCode">Copy code</button>
        </div>
      </div>
      <div class="account-row">
        <div class="account-row-label" data-i18n="chat.referralsCompleted">Successful referrals</div>
        <span class="account-pill-value">${data.referrals_completed}${data.referrals_pending ? ` (+${data.referrals_pending} pending)` : ''}</span>
      </div>
      ${data.was_referred ? `
      <div class="account-row">
        <div class="account-row-label" data-i18n="chat.youWereReferred">You were referred</div>
        <span class="account-pill-value">${data.referred_status === 'completed' ? '✓ Reward granted' : 'Pending your first subscription'}</span>
      </div>
      ` : ''}
    `;
  } catch (e) {
    container.innerHTML = `<div style="font-size:13px;color:#555;padding:16px 0;">Couldn't load referral info.</div>`;
  }
}

// ---------- Usage (Usage tab) ----------
function formatUsageDuration(totalSeconds) {
  const h = Math.floor(totalSeconds / 3600);
  const m = Math.floor((totalSeconds % 3600) / 60);
  if (h <= 0 && m <= 0) return t('chat', 'usageLessThanMinute');
  const hUnit = t('chat', 'usageHrUnit');
  const mUnit = t('chat', 'usageMinUnit');
  if (h <= 0) return `${localizeNum(m)} ${mUnit}`;
  if (m <= 0) return `${localizeNum(h)} ${hUnit}`;
  return `${localizeNum(h)} ${hUnit} ${localizeNum(m)} ${mUnit}`;
}

function usageBarColor(percent) {
  if (percent >= 90) return '#ef4444';
  if (percent >= 75) return '#f59e0b';
  return '#4f8ef7';
}

function usageBarGradient(percent) {
  if (percent >= 90) return 'linear-gradient(90deg,#f8b4ac,#ef4444)';
  if (percent >= 75) return 'linear-gradient(90deg,#fbd39a,#f59e0b)';
  return 'linear-gradient(90deg,#a9c9fb,#4f8ef7)';
}

async function renderUsageSection(isManualRefresh) {
  const container = document.getElementById('usageSummaryContainer');
  if (!container) return;
  const { data: { session } } = await client.auth.getSession();
  if (!session) return;

  if (isManualRefresh) container.style.opacity = '0.6';

  try {
    const [res, referralRes] = await Promise.all([
      fetch(`${API_BASE}/usage/summary?user_id=${encodeURIComponent(session.user.id)}`),
      fetch(`${API_BASE}/referral/status?user_id=${encodeURIComponent(session.user.id)}`)
    ]);
    const data = await res.json();
    const referralData = await referralRes.json().catch(() => null);
    if (document.getElementById('usageSummaryContainer') !== container) return; // tab changed mid-fetch

    const todayData = data.today;
    const w = data.weekly;

    const bonusBalance = referralData && !referralData.error ? (referralData.bonus_balance || 0) : 0;
    const bonusTotal = referralData ? (referralData.bonus_total_granted || 0) : 0;
    const bonusPercent = bonusTotal > 0 ? Math.max(0, Math.min(100, Math.round((bonusBalance / bonusTotal) * 100))) : 0;
    const bonusSectionHtml = bonusBalance > 0 ? `
      <div class="usage-section">
        <div class="usage-section-head">
          <span class="usage-section-title">${t('chat', 'referralBonusHeading')}</span>
          <span class="usage-section-stat" style="color:#4ade80;">${t('chat', 'referralBonusActive')}</span>
        </div>
        <div class="usage-bar-track">
          <div class="usage-bar-fill" style="width:${bonusPercent}%;background:linear-gradient(90deg,#a78bfa,#7c3aed);"></div>
        </div>
        <div class="usage-section-sub">${t('chat', 'referralBonusSub')}</div>
      </div>
    ` : '';

    const todayPercent = Math.round(todayData.percent_used || 0);
    const weekPercent = Math.round(w.percent_used || 0);

    const todayColor = todayData.unlimited ? '#4ade80' : usageBarColor(todayPercent);
    const weekColor = w.unlimited ? '#4ade80' : usageBarColor(weekPercent);
    const todayGradient = todayData.unlimited ? 'linear-gradient(90deg,#86e6ae,#4ade80)' : usageBarGradient(todayPercent);
    const weekGradient = w.unlimited ? 'linear-gradient(90deg,#86e6ae,#4ade80)' : usageBarGradient(weekPercent);

    container.innerHTML = `
      ${bonusSectionHtml}
      <div class="usage-section">
        <div class="usage-section-head">
          <span class="usage-section-title">${t('chat', 'usageTodayHeading')}</span>
          <span class="usage-section-stat" style="color:${todayColor};">${todayData.unlimited ? t('chat', 'usageUnlimited') : t('chat', 'usagePercentUsed').replace('{n}', localizeNum(todayPercent))}</span>
        </div>
        <div class="usage-bar-track">
          <div class="usage-bar-fill" style="width:${todayData.unlimited ? 100 : todayPercent}%;background:${todayGradient};"></div>
        </div>
        <div class="usage-section-sub">
          ${todayData.unlimited ? t('chat', 'usageNoDailyLimit') : t('chat', 'usageResetsIn').replace('{duration}', formatUsageDuration(todayData.reset_in_seconds))}
        </div>
      </div>

      <div class="usage-section">
        <div class="usage-section-head">
          <span class="usage-section-title">${t('chat', 'usageWeeklyHeading')}</span>
          <span class="usage-section-stat" style="color:${weekColor};">${w.unlimited ? t('chat', 'usageUnlimited') : t('chat', 'usagePercentUsed').replace('{n}', localizeNum(weekPercent))}</span>
        </div>
        <div class="usage-bar-track">
          <div class="usage-bar-fill" style="width:${w.unlimited ? 100 : weekPercent}%;background:${weekGradient};"></div>
        </div>
        <div class="usage-section-sub">${t('chat', 'usageDoubtsInWeek').replace('{n}', localizeNum(w.total_doubts))}</div>
      </div>
    `;

    const lastUpdatedEl = document.getElementById('usageLastUpdated');
    if (lastUpdatedEl) lastUpdatedEl.textContent = t('chat', 'usageLastUpdatedJustNow');
  } catch (e) {
    container.innerHTML = `<div style="font-size:13px;color:#555;padding:16px 0;">${t('chat', 'usageCouldntLoad')}</div>`;
  } finally {
    container.style.opacity = '1';
  }
}

// Apply all saved settings on load (theme/font size/chat font/answer style) -- same call chat.html
// makes immediately after defining this function, so it runs as soon as settings.js loads.
function applyAllSettings() {
  const savedTheme = localStorage.getItem('theme');
  const savedFontSize = localStorage.getItem('fontSize');
  const savedChatFont = localStorage.getItem('chatFont');
  const savedAnswerStyle = localStorage.getItem('answerStyle');
  if (savedTheme === 'light') setTheme('light');
  if (savedFontSize) setFontSize(savedFontSize);
  if (savedChatFont) setChatFont(savedChatFont);
  if (savedAnswerStyle) setAnswerStyle(savedAnswerStyle);
}
applyAllSettings();
