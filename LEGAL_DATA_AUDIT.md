# NEET-AI data audit for the privacy policy, terms and Play Data Safety form

Drafted 2026-10-15 from the code in `neet-ai` (website + backend) and `neet-ai-mobile` (Android app).
**Not legal advice. Have a lawyer review this file, `privacy.html` and `terms.html` before publishing.**

Keep this file, `privacy.html`, `terms.html`, `delete-account.html` and `server.py`'s
`ACCOUNT_DELETE_TABLES` / `ACCOUNT_ANONYMISE_COLUMNS` in step. Any new table, provider or tracked
field means updating all of them, plus the Play Data Safety form.

## 1. What is collected

"Retention today" is what the code actually does. Nothing in the codebase deletes old rows on a
schedule; data is only removed by the user's own actions or by `POST /account/delete`.

| Data | Where it comes from | Purpose | Stored in | Retention today | Processed by |
|---|---|---|---|---|---|
| Phone number | Sign-in (app: phone + SMS OTP only; web: optional) | Sign-in, account recovery | Supabase Auth `auth.users` | Until account deletion | Supabase, Twilio (SMS, via Supabase Phone Auth), Railway (`/auth/send-otp` passes it through) |
| Email + password | Web email sign-up, or "link email" in Settings | Sign-in | Supabase Auth (password hashed by Supabase) | Until account deletion | Supabase |
| Google name + email | Web "Continue with Google" | Sign-in, display name | Supabase Auth | Until account deletion | Google OAuth, Supabase |
| Display name | Settings > Account (optional) | Shown in UI; first name added to the AI prompt when personalisation is on | `auth.users.user_metadata.full_name` | Until account deletion | Supabase; AI providers (prompt only) |
| Account ID (UUID) | Supabase | Links all rows | Every user table | Until account deletion (nulled in anonymised tables) | Supabase; Telegram alert text |
| Chat messages, AI answers, answer versions, quotes, titles, pins, diagram refs, per-version votes | Chat (web `chat.html`, app `chatStore.tsx`) | Core feature, history, Retry versions | `chats.messages` (JSONB) | Until the user deletes the chat or the account | Supabase, Railway |
| Question text + last 4 turns + running summary | Each `/chat` request | Generate the answer | Not stored server-side beyond `chats` (client-written) | Transient on our side; provider terms apply | DeepSeek (default), Qwen (peak/fallback), DeepSeek also writes the running summary |
| Images (incl. camera photos) | Chat attachments | Image doubts | Supabase Storage, **public** bucket `ncert-daigrams`, path `chat-uploads/<user_id>/…` (legacy uploads flat under `chat-uploads/`) | Until account deletion (deleting a single chat does NOT delete its files) | Supabase, Railway, **Google Gemini** |
| PDFs | Chat attachments (≤20 MB) | PDF doubts | Same bucket, `chat-uploads/<user_id>/pdfs/` | Until account deletion | Supabase, Railway, **Google Gemini** |
| Image/PDF doubt log | Written by server after every image/PDF answer | Audit wrong answers / abuse | `media_doubt_log`: user_id **or IP for guests**, file SHA-256 + type + size, full AI answer | **Forever** for guests; until account deletion for users | Supabase |
| Saved questions | PYQ bank, chat | Feature; weak-chapter personalisation | `saved_questions` | Until user deletes / account deletion | Supabase |
| Mock / personalised test attempts and results | Mock tests | Feature, scoreboard, personalisation | `mock_test_attempts` (answers, timing), `mock_results` | Until account deletion (attempts cascade on auth-user delete) | Supabase |
| Streaks | Daily activity | Feature | `user_streaks` | Until account deletion | Supabase |
| Answer feedback (thumbs + comment) | Chat | Quality review | `response_feedback` (incl. full question + answer text) | Until user removes vote / account deletion | Supabase |
| Session rating (Bad/Fine/Good + comment) | Chat strip | Quality review | `session_feedback` | Until account deletion (cascade) | Supabase |
| Question / diagram reports (reason + note ≤500 chars) | Report buttons | Fix content | `question_reports`, `diagram_reports` | Forever; `user_id` nulled on account deletion | Supabase |
| Plan, first-paid date, limit timestamps | Server | Plans and limits | `user_plan` | Until account deletion | Supabase |
| Daily token usage | Server, every AI call | Limits, usage page | `usage_log` (user_id, IST date, tokens) | **Forever** until account deletion | Supabase |
| Max-plan breakeven alerts | Server | Internal cost control | `max_usage_alerts` | Forever until account deletion | Supabase |
| AI cost telemetry | Server | Cost tracking | `provider_usage_log` (user_id, provider, token counts, cost) | Forever; `user_id` nulled on account deletion | Supabase |
| Bonus tokens | Referrals | Reward | `bonus_tokens` | Until account deletion | Supabase |
| Referral code / referral links | Referral program | Reward, abuse checks | `user_referral_codes`, `referrals` (referrer_id, referred_id, status) | Codes deleted on account deletion; `referrals` kept with the id nulled | Supabase |
| Device ID, device label, approx. location, sign-in/last-active times | `/session/heartbeat` (web every page; app every 45 s) | Pro/Max device limits, Active Sessions list | `active_sessions` (random ID from localStorage / SecureStore; label = phone model or browser+OS; "City, Region" from IP) | Until the user logs that device out / account deletion | Supabase, **ip-api.com** (receives the IP over plain HTTP) |
| Guest IP + daily tokens | Guest `/chat` | Guest limit | `guest_usage_log` (IP, IST date, tokens) | **Forever** (keyed by IP, not touched by account deletion) | Supabase |
| Shared answer cache / embedding cache | Non-personalised `/chat` | Cost/latency | `answer_cache` (question hash → answer), `embedding_cache` (question hash → embedding vector) | Forever | Supabase; not linked to a user |
| Server logs | Railway stdout | Debugging | Railway logs: user IDs + errors, no question text (checked `print(` calls) | Railway's log retention | Railway |
| Server error events | `sentry_sdk` (send_default_pii=False) | Debugging | Sentry, **only if `SENTRY_DSN` is set on Railway** (empty in local `.env`; Railway unknown) | Sentry plan retention | Sentry (if on) |
| "All AI providers down" alert | Server | Ops | Telegram message: endpoint, user_id, error text | Telegram | Telegram (only if bot token set on Railway; empty locally) |
| On-device only | Web localStorage: `neet_lang`, `theme`, `fontSize`, `chatFont`, `answerStyle`, `personalize`, `showQuestionTimer`, `sidebarWidth`, `deviceId`, session-feedback dismissals. App: `deviceId` (SecureStore), `studyReminder.*`, `rateUs.*`, `intro.seen.v1`, PYQ chapter-count cache | Preferences | Device | Until cleared | None (except `deviceId`, sent to heartbeat) |
| Study reminders | App | Local notifications | `expo-notifications` scheduled locally; **no push token is requested or sent** | n/a | None |
| Voice input | Web Speech API / `expo-speech-recognition` pinned to Google's speech service | Dictation | Audio handled by the OS/browser speech service; we receive text only | n/a | Google (OS service, not us) |
| Read aloud | `speechSynthesis` / `expo-speech` | TTS | On-device | n/a | None |
| Crash reports (app) | `@sentry/react-native` | — | **Off**: `EXPO_PUBLIC_SENTRY_DSN` unset in EAS; heavy PII scrubbing if ever turned on | — | — |

Not collected anywhere: contacts, precise location, advertising ID, analytics SDKs (no GA/Firebase),
SMS inbox (OTP autofill uses the keyboard's `sms-otp` autocomplete, no SMS permission), payment data
(Razorpay not live; plans are set manually).

## 2. Third-party processors

| Processor | Code / env | Role | User content sent? | Location | Notes |
|---|---|---|---|---|---|
| Supabase | `SUPABASE_URL`, `SUPABASE_KEY`, `SUPABASE_SERVICE_KEY` | DB, Auth, Storage | Yes, everything | **India, AWS ap-south-1 (Mumbai)**, verified by matching the DB host's IP against AWS's published ranges | Backups: delete-account.html promises expiry within 30 days; check the Supabase plan's backup window matches |
| Railway | `railway.json` | Backend (FastAPI) | Yes, every request incl. uploads (base64) | **USA, `sfo` (US West)** per `multiRegionConfig` | Consider Railway's Singapore region: data closer to India and lower latency |
| Vercel | `.vercel/`, `vercel.json` | Static website | Only request metadata (IP, UA) | Global edge | |
| DeepSeek | `DEEPSEEK_KEY`, `api.deepseek.com/anthropic` | Default text answers (`/chat`, `/solve`), running chat summary, admin difficulty/answer-key/alt-method jobs | Yes: question, recent turns, summary, first name + weak chapters if personalised | **China** | Third-party analyses say DeepSeek's privacy policy covers API traffic, data is stored in the PRC and may be used for training, with no clear API opt-out. Not confirmed against DeepSeek's own current Open Platform terms. **Biggest risk item for minors' data.** |
| Alibaba Cloud Model Studio (Qwen) | `QWEN_API_KEY`, `dashscope-intl.aliyuncs.com` | Text answers in DeepSeek peak windows / failover | Same as DeepSeek | **Singapore** (international deployment) | Alibaba's docs say it won't use customer data for training |
| Google Gemini API | `GEMINI_API_KEY`, `google-genai` | Image/PDF doubts, `/solve` with diagrams, Hindi NCERT embeddings (`gemini-embedding-001`), context caches | Yes: images, PDFs, question text | USA / Google global | **Confirm the key's project has billing enabled (paid tier).** On the free tier Google may use inputs/outputs to improve products and human reviewers may read them; the paid tier excludes them from training |
| OpenAI | `OPENAI_KEY` | `text-embedding-3-small` for English retrieval + diagram matching | Question text | USA | |
| Twilio | Configured in the Supabase dashboard (not in code) | SMS OTP | Phone number + code | USA | From the 2026-07-29 phone-auth setup notes; **confirm in Supabase > Auth > Phone** |
| Google OAuth | `login.html` | Web sign-in | Google returns name/email to us | — | |
| ip-api.com | `server.py` `_lookup_session_location` | IP → city/region | IP address | Outside India | **Called over plain HTTP; ip-api's free tier is for non-commercial use only** (per ip-api.com's own pricing page; verify). Switch to a paid HTTPS plan or an offline GeoLite2 DB |
| Telegram | `TELEGRAM_BOT_TOKEN`, `TELEGRAM_CHAT_ID` | Ops alert | user_id + error text only | Global | Only if set on Railway |
| Sentry | `SENTRY_DSN` (server), `EXPO_PUBLIC_SENTRY_DSN` (app) | Error reporting | Server: stack traces, `user_id` in the providers-down context; app: scrubbed | US/EU | App: off. Server: **confirm `SENTRY_DSN` is unset on Railway**, or add Sentry to the policy |
| Expo (EAS Update) | `app.json` `updates.url` | OTA app updates | IP, app/runtime version, random EAS client ID | USA | |
| Google Play | — | Distribution, in-app review prompt (`expo-store-review`) | Google's own policy | — | |
| Google Fonts, jsDelivr | `<link>` / `<script>` tags | Web assets | IP + UA on page load | Global | |
| Anthropic | `ANTHROPIC_KEY` in `.env` | **Not called.** The `anthropic` package is only the SDK pointed at DeepSeek's endpoint | No | — | The old privacy policy wrongly listed Anthropic; remove the unused key |

## 3. Issues to fix or decide before publishing

1. **No retention cleanup exists.** `guest_usage_log` (IPs), `media_doubt_log`, `usage_log`,
   `max_usage_alerts` and `provider_usage_log` grow forever. The new privacy policy promises "up to one
   year, then deleted" (an HTML comment marks the line). Add a daily job that deletes rows older than
   365 days before publishing, or change the wording. One year also matches DPDP Rules 6(1)(e) and 8(3),
   which from May 2027 require keeping processing logs for **at least** one year.
2. **DPDP Rule 8(3) vs instant account deletion (from May 2027).** Rule 8(3) requires keeping "personal
   data, associated traffic data and other logs of the processing" for at least one year, even after the
   account is deleted (Illustration, Case 1). `/account/delete` currently hard-deletes `usage_log` and
   `media_doubt_log` immediately. Ask the lawyer whether those need keeping (possibly anonymised) for a
   year after deletion.
3. **Chat uploads are in a public bucket.** Anyone with the URL can open a student's photo or PDF. The
   URLs are random and unlisted, and the policy says exactly that, but a private bucket with signed URLs
   is the better fix.
4. **DeepSeek handles most answers, including minors' questions, on servers in China.** It's disclosed
   in the policy. The Data Safety answers below treat it as "sharing". To reduce risk: confirm DeepSeek's
   current API data-use terms, or route under-18 accounts to Qwen or paid-tier Gemini.
5. **Gemini tier.** Confirm the paid tier. If it's the free tier, switch before launch, since it receives
   students' photos.
6. **ip-api.com** is HTTP-only and free for non-commercial use only.
7. **Grievance Officer name.** DPDP Rule 9 only needs contact details, but the IT (SPDI) Rules 2011,
   Rule 5(9), which still apply until the DPDP Act fully replaces them, ask for the Grievance Officer's
   **name**. Passwords count as sensitive personal data there. Ask the lawyer whether to name a person.
8. **Minimum age changed from 16 to 13** in both pages (see section 4). Revert if you'd rather keep 16.
9. **Governing law:** the old terms named the courts of Kuwait City. The draft says "competent courts in
   India". Choose a city where the business is registered.
10. **Effective date** is set to 15 October 2026. Set it to the actual publish date.
11. **`help.html` still shows `mustafatasscr7@gmail.com`** as the contact address. It wasn't changed, since
    it was out of scope.
12. **App → Hindi policy:** `ProfileMenu.tsx` opens `privacy.html` / `terms.html` without `?lang=`, and an
    in-app browser has no `neet_lang`, so Hindi users land on English. Append `?lang=${lang}` (JS-only,
    can ship OTA).
13. `delete-account.html` is still English-only.

## 4. Minors: what's required and the simplest compliant approach

### DPDP Act 2023 + DPDP Rules 2025 (official Gazette text, notified 13–14 Nov 2025)
- A **child is anyone under 18** (s.2(f)). Most of NEET-AI's users are children in law.
- **Verifiable parental consent before processing any of a child's personal data** (s.9(1), Rule 10).
  You must take "technical and organisational measures" and do due diligence that the person consenting
  is an **identifiable adult**. That can rest on (a) reliable identity/age details you already hold, or
  (b) details provided voluntarily by the parent, or a virtual token from an authorised entity, with
  DigiLocker named. Rule 10's illustrations, for a parent who isn't already a verified user, point to
  government-issued identity/age details or a DigiLocker token. A self-declared checkbox is not enough.
- **No tracking or behavioural monitoring of children, and no targeted advertising at them** (s.9(3)).
  "Personalize my AI tutor" (weak chapters from mistakes) could be read as behavioural monitoring.
- Exemptions (Fourth Schedule): an "educational institution" ("an institution of learning that imparts
  education") is exempt only for tracking/behavioural monitoring tied to its educational activities.
  Whether an online AI tutor qualifies is untested, so **don't rely on it**. Part B item 6 exempts
  processing done to confirm a user is *not* a child, which means **an age gate is itself permitted**.
- **When:** Rules 3 and 5–16 (notice, security, breach, retention, children, rights, grievances) apply
  **18 months after publication, around 13–14 May 2027**. Penalty for children's-data breaches: up to
  ₹200 crore. Rights requests and grievances must be answered within 90 days. Breaches go to the Board
  within 72 hours.

### Google Play
- **Target audience and content:** declare honest age groups. If any group is **under 13**, the
  **Families Policy** applies (Families-certified SDKs only, extra ads/data rules, teacher-approved
  scrutiny). Targeting **13–15, 16–17 and 18+** keeps you outside the Families Policy, though Google can
  still review an app that appeals to children. Misdeclaring can get the app removed.
- **Generative AI apps** must let users report or flag offensive AI output in the app. The thumbs-down
  plus comment on every answer (`response_feedback`) should cover this. Make sure it stays visible.
- Account deletion: in-app plus web URL. Already done.
- Distribute in **India only** at launch. Other countries (for example some US states' app-store age
  laws) add their own age-verification duties.

### Proposed approach (not built)
1. **Minimum age 13.** Play target audience: 13–15, 16–17, 18+.
2. **Neutral age gate** at sign-up, and once for existing accounts: ask date or year of birth with no
   default value and no "I'm 18+" pre-checked. Store `birth_year` and `age_confirmed_at`. Under 13: block.
3. **18+:** unchanged.
4. **13–17: "parent consent pending" state.** The student enters a parent's mobile number. The parent
   gets an SMS link to a consent page with the notice (EN/HI: data, purposes, AI providers incl. China,
   cross-border, rights, withdrawal). The parent verifies their own number by OTP, states they're the
   parent/guardian and an adult, and consents. Store a `parental_consents` row: child id, parent phone,
   parent name, timestamp, notice version, method. Until consent arrives, the student gets guest-mode
   limits only, with nothing saved to an account.
5. **Before May 2027:** add DigiLocker (or another authorised entity's) age-token check for the parent,
   to meet Rule 10's "identifiable adult" due diligence. Step 4 alone is a good-faith interim measure,
   not full Rule 10 compliance.
6. **For under-18 accounts:** personalisation off and hidden unless the lawyer clears it under s.9(3).
   Consider routing their doubts away from DeepSeek. No ads (already true).
7. **Parent rights:** parents email the Grievance Officer to see or delete data. A parent withdrawing
   consent means deleting the account.

Cost: one table, age-gate and parent-consent screens on web and app (pure JS, no native package), and an
SMS sender for parent links. Supabase phone OTP is tied to sign-in, so this needs direct Twilio/MSG91
access, i.e. new env keys.

## 5. Play Console Data Safety answers

Website for account deletion: `https://neet-ai-jet.vercel.app/delete-account.html`.
Privacy policy URL: `https://neet-ai-jet.vercel.app/privacy.html`.

### Data collection and security

| Question | Answer |
|---|---|
| Does your app collect or share any of the required user data types? | **Yes** |
| Is all of the user data collected by your app encrypted in transit? | **Yes** (app → Supabase / Railway / Expo are all HTTPS; the HTTP ip-api call is server-side) |
| Which methods of account creation does your app support? | **Username and other authentication** (phone number + SMS one-time code). No OAuth or password sign-up in the app |
| Delete account URL | `https://neet-ai-jet.vercel.app/delete-account.html` |
| Do you provide a way for users to request that some or all of their data is deleted, without requiring them to delete their account? | **Yes**: delete individual chats and saved questions, remove answer votes, log out devices |
| Independent security review | No |
| Committed to Play Families Policy | Not applicable (no under-13 target) |
| UPI payments badge | No (payments not live) |

### Data types

`Shared = Yes*` means the data goes to an AI provider that may use it for its own purposes (DeepSeek,
and Gemini if on the free tier), so it shouldn't be claimed under the service-provider exemption. If you
confirm every AI provider is a pure processor (no training, paid tier), these can become **No**.
Over-declaring is safe; under-declaring gets apps pulled.

| Category | Data type | Collected | Shared | Ephemeral | Required / optional | Purposes (collected) | Purposes (shared) |
|---|---|---|---|---|---|---|---|
| Location | Approximate location | Yes (city/region from IP, Active Sessions) | No | No | Required | App functionality; Fraud prevention, security, and compliance | — |
| Location | Precise location | No | No | | | | |
| Personal info | Name | Yes | Yes* (first name in AI prompt when personalisation is on) | No | Optional | App functionality; Personalization; Account management | App functionality; Personalization |
| Personal info | Email address | Yes | No | No | Optional (app sign-in is phone) | Account management | — |
| Personal info | User IDs | Yes | No | No | Required | App functionality; Account management; Fraud prevention, security, and compliance | — |
| Personal info | Phone number | Yes | No (Twilio is a service provider) | No | Required | Account management; Fraud prevention, security, and compliance | — |
| Personal info | Address, race/ethnicity, political/religious, sexual orientation, other info | No | No | | | | |
| Financial info | All (payment info, purchase history, credit score, other) | No (update to **Purchase history** when Razorpay goes live) | No | | | | |
| Health and fitness | Health info, fitness info | No | No | | | | |
| Messages | Emails; SMS or MMS | No | No | | | | |
| Messages | Other in-app messages (chats with the AI tutor) | Yes | Yes* | No | Required | App functionality; Personalization | App functionality |
| Photos and videos | Photos | Yes (doubt images, camera) | Yes* (Gemini) | No | Optional | App functionality | App functionality |
| Photos and videos | Videos | No | No | | | | |
| Audio | Voice or sound recordings | No (the OS speech service converts speech; the app receives text only) | No | | | | |
| Audio | Music files, other audio | No | No | | | | |
| Files and docs | Files and docs | Yes (PDF doubts) | Yes* (Gemini) | No | Optional | App functionality | App functionality |
| Calendar | Calendar events | No | No | | | | |
| Contacts | Contacts | No | No | | | | |
| App activity | App interactions (test answers/scores, saved questions, streaks, votes, ratings, usage counts, last active) | Yes | Yes* (weak-chapter names in the AI prompt when personalisation is on) | No | Required | App functionality; Analytics; Personalization; Fraud prevention, security, and compliance | Personalization |
| App activity | In-app search history | No (search is local) | No | | | | |
| App activity | Installed apps | No | No | | | | |
| App activity | Other user-generated content (feedback comments, rating comments, report notes) | Yes | No | No | Optional | App functionality; Analytics | — |
| App activity | Other actions | No | No | | | | |
| Web browsing | Web browsing history | No | No | | | | |
| App info and performance | Crash logs, diagnostics, other performance data | No (Sentry off; switch to **Yes / Analytics** if `EXPO_PUBLIC_SENTRY_DSN` is ever set) | No | | | | |
| Device or other IDs | Device or other IDs | Yes (random per-install device ID for device limits; Expo update client ID) | No | No | Required | App functionality; Fraud prevention, security, and compliance | — |

## Sources
- DPDP Rules, 2025, official Gazette text (MeitY): https://www.meity.gov.in/static/uploads/2025/11/53450e6e5dc0bfa85ebd78686cadad39.pdf
- PIB explainer, "DPDP Rules, 2025 Notified": https://static.pib.gov.in/WriteReadData/specificdocs/documents/2025/nov/doc20251117695301.pdf
- Play Data safety form: https://support.google.com/googleplay/android-developer/answer/10787469
- Play account deletion requirements: https://support.google.com/googleplay/android-developer/answer/13327111
- Play target audience / Families: https://support.google.com/googleplay/android-developer/answer/9867159
- Play AI-generated content policy: https://support.google.com/googleplay/android-developer/answer/13985936
- Gemini API additional terms (free vs paid data use): https://ai.google.dev/terms
- Alibaba Model Studio regions: https://help.aliyun.com/en/model-studio/regions/
- DeepSeek API data use (third-party analysis, not primary): https://meetily.ai/llm-privacy/deepseek
