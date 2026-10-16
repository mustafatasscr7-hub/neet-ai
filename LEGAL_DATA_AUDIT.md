# NEET-AI data audit for the privacy policy, terms and Play Data Safety form

Drafted 2026-10-15 from the code in `neet-ai` (website + backend) and `neet-ai-mobile` (Android app).
Updated 2026-10-16: private chat uploads, the retention schedule (section 1a), the DB-IP location
lookup, storage-location wording, and 5,000-student sizing notes (section 6).
**Not legal advice. Have a lawyer review this file, `privacy.html` and `terms.html` before publishing.**

Keep this file, `privacy.html`, `terms.html`, `delete-account.html` and `server.py`'s
`ACCOUNT_DELETE_TABLES` / `ACCOUNT_ANONYMISE_COLUMNS` in step. Any new table, provider or tracked
field means updating all of them, plus the Play Data Safety form.

## 1. What is collected

"Retention" is what the code does once `retention_cleanup.sql` is scheduled (daily pg_cron jobs,
section 1a). Until the owner runs that SQL, nothing deletes old rows on a schedule; data is then
only removed by the user's own actions or by `POST /account/delete`.

| Data | Where it comes from | Purpose | Stored in | Retention | Processed by |
|---|---|---|---|---|---|
| Phone number | Sign-in (app: phone + SMS OTP only; web: optional) | Sign-in, account recovery | Supabase Auth `auth.users` | Until account deletion | Supabase, Twilio (SMS, via Supabase Phone Auth), Railway (`/auth/send-otp` passes it through) |
| Email + password | Web email sign-up, or "link email" in Settings | Sign-in | Supabase Auth (password hashed by Supabase) | Until account deletion | Supabase |
| Google name + email | Web "Continue with Google" | Sign-in, display name | Supabase Auth | Until account deletion | Google OAuth, Supabase |
| Display name | Settings > Account (optional) | Shown in UI; first name added to the AI prompt when personalisation is on | `auth.users.user_metadata.full_name` | Until account deletion | Supabase; AI providers (prompt only) |
| Account ID (UUID) | Supabase | Links all rows | Every user table | Until account deletion (nulled in anonymised tables) | Supabase; Telegram alert text |
| Chat messages, AI answers, answer versions, quotes, titles, pins, diagram refs, per-version votes | Chat (web `chat.html`, app `chatStore.tsx`) | Core feature, history, Retry versions | `chats.messages` (JSONB) | Until the user deletes the chat or the account | Supabase, Railway |
| Question text + last 4 turns + running summary | Each `/chat` request | Generate the answer | Not stored server-side beyond `chats` (client-written) | Transient on our side; provider terms apply | DeepSeek (default), Qwen (peak/fallback), DeepSeek also writes the running summary |
| Images (incl. camera photos) | Chat attachments | Image doubts | Supabase Storage, **private** bucket `chat-uploads`, key `<user_id>/<hex>.<ext>` (since 2026-10-16; moved from the public `ncert-daigrams/chat-uploads/` by `migrate_chat_uploads_private.py`). Students get 1-hour signed URLs from `POST /chat-uploads/sign`, own folder only | While linked from one of the owner's chats; unlinked files deleted once 7 days old (daily sweep); all deleted on account deletion | Supabase, Railway, **Google Gemini** |
| PDFs | Chat attachments (≤20 MB) | PDF doubts | Same private bucket, `<user_id>/pdfs/` | Same as images | Supabase, Railway, **Google Gemini** |
| Image/PDF doubt log | Written by server after every image/PDF answer | Audit wrong answers / abuse | `media_doubt_log`: user_id **or IP for guests**, file SHA-256 + type + size, full AI answer | 90 days; deleted on account deletion | Supabase |
| Saved questions | PYQ bank, chat | Feature; weak-chapter personalisation | `saved_questions` | Until user deletes / account deletion | Supabase |
| Mock / personalised test attempts and results | Mock tests | Feature, scoreboard, personalisation | `mock_test_attempts` (answers, timing), `mock_results` | Until account deletion (attempts cascade on auth-user delete) | Supabase |
| Streaks | Daily activity | Feature | `user_streaks` | Until account deletion | Supabase |
| Answer feedback (thumbs + comment) | Chat | Quality review | `response_feedback` (incl. full question + answer text) | Until user removes vote / account deletion; `user_id` + `chat_id` cleared after 12 months, content kept | Supabase |
| Session rating (Bad/Fine/Good + comment) | Chat strip | Quality review | `session_feedback` | Until account deletion (cascade); `user_id` + `chat_id` cleared after 12 months, content kept | Supabase |
| Question / diagram reports (reason + note ≤500 chars) | Report buttons | Fix content | `question_reports`, `diagram_reports` | Content kept; `user_id` cleared after 12 months or on account deletion | Supabase |
| Plan, first-paid date, limit timestamps | Server | Plans and limits | `user_plan` | Until account deletion | Supabase |
| Daily token usage | Server, every AI call | Limits, usage page | `usage_log` (user_id, IST date, tokens) | 12 months; deleted on account deletion | Supabase |
| Max-plan breakeven alerts | Server | Internal cost control | `max_usage_alerts` | 12 months; deleted on account deletion | Supabase |
| AI cost telemetry | Server | Cost tracking | `provider_usage_log` (user_id, provider, token counts, cost) | 24 months; `user_id` cleared after 12 months or on account deletion | Supabase |
| Bonus tokens | Referrals | Reward | `bonus_tokens` | Until account deletion | Supabase |
| Referral code / referral links | Referral program | Reward, abuse checks | `user_referral_codes`, `referrals` (referrer_id, referred_id, status) | Codes deleted on account deletion; `referrals` kept with the id nulled | Supabase |
| Device ID, device label, approx. location, sign-in/last-active times | `/session/heartbeat` (web every page; app every 45 s) | Pro/Max device limits, Active Sessions list | `active_sessions` (random ID from localStorage / SecureStore; label = phone model or browser+OS; "City, Region" from IP, the IP itself is not stored) | 90 days after last activity (logged-out/kicked rows included); deleted on account deletion | Supabase. Location looked up on our server in DB-IP's offline database: **no third party receives the IP** (replaced ip-api.com 2026-10-16) |
| Guest IP + daily tokens | Guest `/chat` | Guest limit | `guest_usage_log` (IP, IST date, tokens) | 30 days (keyed by IP, not touched by account deletion) | Supabase |
| Supabase sign-in log | Supabase Auth | Auth audit trail | `auth.audit_log_entries` (sign-in events, IP address) | 12 months | Supabase |
| Shared answer cache / embedding cache | Non-personalised `/chat` | Cost/latency | `answer_cache` (question hash → answer), `embedding_cache` (question hash → embedding vector), `pyq_solution_cache` | No limit (not personal) | Supabase; not linked to a user |
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

## 1a. Retention schedule: every table (implemented 2026-10-16)

The owner's chosen periods, enforced by `retention_cleanup.sql`: two daily pg_cron jobs (03:00 and
03:15 IST). Row rules are batched (10,000 rows × up to 20 batches per table per run; a larger backlog
drains over the following days and the run summary says "capped"). Unlinked chat files are deleted
through the Storage API via pg_net, with the service key in Vault. Every run writes its counts to
`retention_runs`. Verified on Postgres 17 (PGlite) with production-shaped tables. Production state:
**SQL written, not yet run by the owner.**

| Rule | Tables / storage |
|---|---|
| Until the user deletes it or the account | `chats`, `saved_questions`, `mock_results`, `mock_test_attempts`, `user_streaks` |
| While linked from one of the owner's chats; unlinked files deleted once 7 days old | `chat-uploads` bucket. "Linked" = the file's 32-character name appears in one of **its owner's** chats (every stored link form contains it). `chat_upload_refs` (kept by a trigger on `chats`) finds candidates fast; each one is re-checked against the chats before deletion, and a trigger error never blocks a chat save |
| Deleted accounts: removed immediately | `POST /account/delete` (unchanged, now also covers the private bucket). **No backups**: the project is on Supabase Free, which takes none (owner, 2026-10-16). Storage files are never in Supabase backups on any plan |
| 30 days | `guest_usage_log` |
| 90 days after last activity | `active_sessions` |
| 90 days | `media_doubt_log` |
| 12 months | `usage_log`, `max_usage_alerts`, `auth.audit_log_entries` (owner chose 12 months; see section 6 on its size) |
| 24 months, `user_id` cleared at 12 months | `provider_usage_log`. It holds `user_id`, so it was not "no personal data" as it stood; clearing the id at 12 months makes the second year anonymous |
| Anonymise at 12 months (`user_id`, and `chat_id` where present, cleared), keep content | `question_reports`, `diagram_reports`, `response_feedback`, `session_feedback` (their `user_id` was NOT NULL; `retention_cleanup.sql` relaxes that) |
| No limit (not personal) | `answer_cache`, `embedding_cache`, `pyq_solution_cache` |
| Housekeeping | `cron.job_run_details` 30 days; `retention_runs` 90 days (the file sweep logs deleted file names, `<user_id>/<hex>.<ext>`, so a wrong deletion could be traced) |

Tables that didn't fit the owner's list, and what was decided:

| Table | Decision |
|---|---|
| `user_plan`, `bonus_tokens`, `user_referral_codes` | Account state, not logs: kept until account deletion (already deleted by `/account/delete`) |
| `referrals` | Kept; both ids cleared on either student's account deletion (unchanged) |
| `usage_limits` | Legacy per-user table (13 rows, last written 2026-07-21), unused by both repos and missed by `/account/delete`: **dropped** by `drop_unused_tables.sql` |
| `"ncert-daigrams"` (a table, 0 rows) | Created by mistake with the bucket's name: **dropped** by `drop_unused_tables.sql` |
| `Q-Images` storage bucket (public, 0 files) | Unused: **delete from the dashboard** (step 3 of `drop_unused_tables.sql`) |
| `auth.users`, `auth.sessions`, `auth.refresh_tokens` | Supabase-managed sign-in data, removed with the account. Supabase Pro can time out inactive sessions (Auth settings); 90 days would match `active_sessions` |
| `pyq`, `diagrams`, `ncert_content`, `mock_tests`, `mock_test_questions`, `personalised_test_sets`, `neet_exam_dates`, `pyq_dismissed_duplicates`, `processed_pdfs` | Content, no student data (`processed_pdfs.processed_by` is staff): no limit |
| `retention_runs`, `chat_upload_refs` | Internal to the cleanup (counts / chat-file links), RLS on with no policies; refs cascade with their chat |

## 2. Third-party processors

The Location column is for our own records. **privacy.html names no server or provider country**
(owner's decision, 2026-10-16): it says data is stored in India and that service providers may
process it in other countries.

| Processor | Code / env | Role | User content sent? | Location | Notes |
|---|---|---|---|---|---|
| Supabase | `SUPABASE_URL`, `SUPABASE_KEY`, `SUPABASE_SERVICE_KEY` | DB, Auth, Storage | Yes, everything | **India, AWS ap-south-1 (Mumbai)**, verified by matching the DB host's IP against AWS's published ranges | **Free plan: no automatic backups** (owner, 2026-10-16), so privacy.html and delete-account.html say deleted data is removed immediately with no backup copies. **The owner will move to Pro before going live**: Pro keeps daily backups for 7 days (point-in-time recovery add-on: 7/14/28 days), so on that day change both pages to "backup copies are deleted within 7 days" (or the PITR window). Storage files are never in Supabase backups |
| Railway | `railway.json` | Backend (FastAPI) | Yes, every request incl. uploads (base64) | **USA, `sfo` (US West)** per `multiRegionConfig` | Consider Railway's Singapore region: data closer to India and lower latency |
| Vercel | `.vercel/`, `vercel.json` | Static website | Only request metadata (IP, UA) | Global edge | |
| DeepSeek | `DEEPSEEK_KEY`, `api.deepseek.com/anthropic` | Default text answers (`/chat`, `/solve`), running chat summary, admin difficulty/answer-key/alt-method jobs | Yes: question, recent turns, summary, first name + weak chapters if personalised | **China** | Third-party analyses say DeepSeek's privacy policy covers API traffic, data is stored in the PRC and may be used for training, with no clear API opt-out. Not confirmed against DeepSeek's own current Open Platform terms. **Biggest risk item for minors' data.** |
| Alibaba Cloud Model Studio (Qwen) | `QWEN_API_KEY`, `dashscope-intl.aliyuncs.com` | Text answers in DeepSeek peak windows / failover | Same as DeepSeek | **Singapore** (international deployment) | Alibaba's docs say it won't use customer data for training |
| Google Gemini API | `GEMINI_API_KEY`, `google-genai` | Image/PDF doubts, `/solve` with diagrams, Hindi NCERT embeddings (`gemini-embedding-001`), context caches | Yes: images, PDFs, question text | USA / Google global | **Confirm the key's project has billing enabled (paid tier).** On the free tier Google may use inputs/outputs to improve products and human reviewers may read them; the paid tier excludes them from training |
| OpenAI | `OPENAI_KEY` | `text-embedding-3-small` for English retrieval + diagram matching | Question text | USA | |
| Twilio | Configured in the Supabase dashboard (not in code) | SMS OTP | Phone number + code | USA | From the 2026-07-29 phone-auth setup notes; **confirm in Supabase > Auth > Phone** |
| Google OAuth | `login.html` | Web sign-in | Google returns name/email to us | — | |
| DB-IP (data source, **not a processor**) | `server.py` `_refresh_geoip_db` / `_lookup_session_location` | IP → city/region, on our own server | **Nothing**: the "IP to City Lite" database (~120 MB, monthly) is downloaded to the server and read locally | — | Replaced ip-api.com (plain HTTP, non-commercial free tier) on 2026-10-16. Licence CC BY 4.0, commercial use allowed **with attribution**: "IP Geolocation by DB-IP" is linked under every Active Sessions list (web + app), in privacy.html section 1, and on the app's licences screen. Keep all three if the lookup stays |
| Telegram | `TELEGRAM_BOT_TOKEN`, `TELEGRAM_CHAT_ID` | Ops alert | user_id + error text only | Global | Only if set on Railway |
| Sentry | `SENTRY_DSN` (server), `EXPO_PUBLIC_SENTRY_DSN` (app) | Error reporting | Server: stack traces, `user_id` in the providers-down context; app: scrubbed | US/EU | App: off. Server: **confirm `SENTRY_DSN` is unset on Railway**, or add Sentry to the policy |
| Expo (EAS Update) | `app.json` `updates.url` | OTA app updates | IP, app/runtime version, random EAS client ID | USA | |
| Google Play | — | Distribution, in-app review prompt (`expo-store-review`) | Google's own policy | — | |
| Google Fonts, jsDelivr | `<link>` / `<script>` tags | Web assets | IP + UA on page load | Global | |
| Anthropic | `ANTHROPIC_KEY` in `.env` | **Not called.** The `anthropic` package is only the SDK pointed at DeepSeek's endpoint | No | — | The old privacy policy wrongly listed Anthropic; remove the unused key |

## 3. Issues to fix or decide before publishing

1. **Retention cleanup: built, not live yet.** `retention_cleanup.sql` (section 1a) implements the
   owner's periods. Publish privacy.html only after the owner has run it and both cron jobs show up in
   `cron.job`. Note the owner's periods for guest IPs (30 days), device sessions (90 days) and image/PDF
   doubt records (90 days) are **shorter than one year**. DPDP Rules 6(1)(e) and 8(3) (from May 2027)
   require keeping certain logs of processing for **at least** one year. Ask the lawyer which of these
   tables count as such logs.
2. **DPDP Rule 8(3) vs instant account deletion (from May 2027).** Rule 8(3) requires keeping "personal
   data, associated traffic data and other logs of the processing" for at least one year, even after the
   account is deleted (Illustration, Case 1). `/account/delete` currently hard-deletes `usage_log` and
   `media_doubt_log` immediately. Ask the lawyer whether those need keeping (possibly anonymised) for a
   year after deletion.
3. **Chat uploads: fixed (2026-10-16), pending deploy + migration.** Private `chat-uploads` bucket,
   1-hour signed URLs for the owner only, old links in `chats.messages` still resolve. Live only after:
   `create_chat_uploads_bucket.sql` → backend deploy → `migrate_chat_uploads_private.py --apply` →
   website + EAS Update → owner's device check → `--delete-public` (owner: no waiting period, the app
   isn't public). Until `--delete-public`, the old public copies are still reachable by URL.
4. **DeepSeek handles most answers, including minors' questions, on servers in China.** privacy.html
   names DeepSeek but, by the owner's decision, no provider country. Ask the lawyer whether naming no
   country is enough for students under 18. The Data Safety answers below treat DeepSeek as "sharing".
   To reduce risk: confirm DeepSeek's current API data-use terms, or route under-18 accounts to Qwen or
   paid-tier Gemini.
5. **Gemini tier.** Confirm the paid tier. If it's the free tier, switch before launch, since it receives
   students' photos.
6. **ip-api.com: replaced (2026-10-16)** by DB-IP's offline database on our own server (section 2).
   Keep the attribution links.
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
14. **Supabase Free has no backups.** Both pages now say deleted data is removed immediately and no
    backup copies are kept. When the project moves to Pro before going live, update that sentence in
    privacy.html (EN + HI) and delete-account.html to the Pro backup window (7 days, or the PITR window).
15. Found and fixed during the 2026-10-16 work:
    - The image-correction follow-up ("no, this is a dicot") made the **server fetch any URL** taken from
      the client-sent chat history (SSRF, guests included). It now only downloads the caller's own
      private files with the service key.
    - **Active Sessions was empty on every page except chat.html** (`settings.js` called
      `/session/list` and `/session/logout` without the Bearer token since the 2026-10-07 IDOR fix).
    - `/account/delete` missed chat uploads saved with a `.jpeg` extension (two such files, Sept 2026).

## 3a. Old app versions during the upload move

The app has no public install yet (owner, 2026-10-16). For the record, a build from before the move
that never receives the EAS Update will, once `--delete-public` has run: show a blank thumbnail for
older photos; open a browser page saying the object isn't found from an older PDF chip; and resend
Retry/Edit of those messages without the attachment (`downloadFileAsync` throws on the error, so
nothing wrong is sent). Photos it sends after the move work for an hour; their thumbnails reappear once
it updates, because the new code re-signs the saved link. Nothing crashes, and text, answers and
diagrams are unaffected.

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
| Is all of the user data collected by your app encrypted in transit? | **Yes** (app → Supabase / Railway / Expo are all HTTPS; the plain-HTTP ip-api.com call is gone since 2026-10-16) |
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

## 6. Sizing for 5,000 active students (2026-10-16)

Assumptions: 5,000 **daily** active students; ~1 KB per chat message (measured), 10 messages per
student per day (assumed); 10 AI calls per student per day (measured median 3.5, 90th percentile 17);
1 in 5 students sends one photo a day at ~410 KB (measured average upload), each viewed ~3 times.

Built for this scale on 2026-10-16:
- **Signed URLs:** one batched call per opened chat. The server reuses a URL while at least 10 minutes
  are left (LRU of 20,000 entries, ~8 MB), so reopening a chat hits the browser/app image cache.
  Limit: 120 calls a minute per student (by user, not by IP).
- **Uploads** are written asynchronously now. The old blocking call stalled every streaming answer
  while a file uploaded.
- **Cleanup**, one day's volume measured on Postgres 17 in WASM (slower than Supabase):
  - row rules: 2.1 s for ~340k rows
  - file sweep: 3.6 s over 32k files and 150k chats
  - chat-save trigger: under 1 ms
  - caps: 200k rows per table and 1,000 files per run
- **DB-IP:** ~40 µs per lookup, made only for new sessions. ~120 MB on disk, memory-mapped (+0 MB when
  opened, ~60 MB after 50k random lookups, reclaimable).

Not changed by this work. These break or cost money at 5,000:

| Item | At 5,000 students | Free plan | Pro plan |
|---|---|---|---|
| Database: `chats` | ~50 MB/day, ~1.5 GB/month | 500 MB: full in ~10 days | 8 GB included: ~5 months, then $0.125/GB-month |
| Database: `provider_usage_log` (kept 24 months) | 50k rows/day, ~36M rows | — | The largest table. Consider rolling rows older than 12 months into daily per-provider totals |
| Storage: chat uploads (kept while linked) | ~410 MB/day, ~12 GB/month | 1 GB: full in ~3 days | 100 GB: ~8 months, then $0.021/GB-month |
| Egress: viewing uploads | ~1.2 GB/day, ~37 GB/month | 5 GB/month | 250 GB/month |
| `auth.audit_log_entries` (12 months) | If Supabase logs each hourly token refresh: 100k+ rows/day, tens of millions | | Check the count in `retention_cleanup.sql` PART 3. A shorter period, or turning off database audit logs in Auth settings, may be needed |
| Per-IP rate limits (`/chat` 15/min, uploads 10/min) | Indian carriers put many users behind one shared IP, so students get false "Too many requests" | | Key logged-in limits by user id |
| `_rate_limit_buckets` in server.py | One entry per IP and endpoint, never pruned: slow memory growth | | Prune old keys |

Above 5,000:
- Raise the cleanup caps (`p_max_batches`) if the run summaries say "capped" every day.
- The file sweep reads the bucket's whole object list daily. That's fine up to a few hundred thousand
  files; beyond that, limit it to a `created_at` window.
- With one Railway replica, the URL cache and rate limits live in that one process. More replicas
  would each keep their own.

## Sources
- Supabase backups (Free: none; Pro 7 days; storage objects never included): https://supabase.com/docs/guides/platform/backups
- Supabase pricing (quotas in section 6): https://supabase.com/pricing
- Supabase Storage, deleting objects (API only, not SQL): https://supabase.com/docs/guides/storage/management/delete-objects
- Supabase pg_net (`net.http_delete`): https://supabase.com/docs/guides/database/extensions/pg_net
- DB-IP IP to City Lite (CC BY 4.0, attribution): https://db-ip.com/db/download/ip-to-city-lite
- DPDP Rules, 2025, official Gazette text (MeitY): https://www.meity.gov.in/static/uploads/2025/11/53450e6e5dc0bfa85ebd78686cadad39.pdf
- PIB explainer, "DPDP Rules, 2025 Notified": https://static.pib.gov.in/WriteReadData/specificdocs/documents/2025/nov/doc20251117695301.pdf
- Play Data safety form: https://support.google.com/googleplay/android-developer/answer/10787469
- Play account deletion requirements: https://support.google.com/googleplay/android-developer/answer/13327111
- Play target audience / Families: https://support.google.com/googleplay/android-developer/answer/9867159
- Play AI-generated content policy: https://support.google.com/googleplay/android-developer/answer/13985936
- Gemini API additional terms (free vs paid data use): https://ai.google.dev/terms
- Alibaba Model Studio regions: https://help.aliyun.com/en/model-studio/regions/
- DeepSeek API data use (third-party analysis, not primary): https://meetily.ai/llm-privacy/deepseek
