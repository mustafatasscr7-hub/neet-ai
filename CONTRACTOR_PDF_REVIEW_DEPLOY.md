# Deploying the isolated contractor PDF-review tool

This is the one-time setup for the unbranded, standalone copy of the PDF review tool
(`contractor-pdf-review.html`). It's separate from the main site on purpose — a new Vercel
project, a new URL, its own access key — so a contractor using it never sees the main app or
learns the business name behind it. These steps need your Vercel account; nothing here can be
done from inside this repo alone.

## 1. Generate and set the contractor's access key

Pick a long random string (e.g. `openssl rand -hex 24`) — this is the contractor's password,
separate from your own admin password. Set it as a Railway environment variable on the backend:

```
CONTRACTOR_PDF_KEY=<the random string>
```

Redeploy the backend (or let Railway pick it up automatically, however it's configured) so the
new env var takes effect. Until this is set, the isolated tool's login will always reject every
password — it's opt-in by design.

## 2. Create a new, separate Vercel project

In the Vercel dashboard, create a **new project** (not a new page/route inside the existing
`neet-ai` project) with a name that doesn't reference the business, e.g. `pdf-review-portal` or
similar. This is what gives you a `<your-chosen-name>.vercel.app` URL with nothing recognizable
in it. A brand-new project needs its own git repo or a manual folder upload — either works.

## 3. Deploy exactly two files to that new project

- `contractor-pdf-review.html` → rename to `index.html`
- `contractor-pdf-review-vercel.json` → rename to `vercel.json`

That's the entire deployment — no build step, no other files needed. The `vercel.json` rewrite
(`/api/:path* → https://neet-ai-production.up.railway.app/:path*`) is what proxies API calls
through the new domain, so a contractor's browser DevTools only ever sees requests to
`pdf-review-portal.vercel.app/api/...`, never the real Railway hostname.

## 4. Test

- Visit the new URL, enter the `CONTRACTOR_PDF_KEY` value from step 1 — should log in.
- Confirm the old admin password does *not* work here (it shouldn't — this tool checks only
  `CONTRACTOR_PDF_KEY`, independent of your own admin password).
- Confirm your own admin password still works, unaffected, on the original
  `admin-pdf-review.html` on the main site — this change never touched that page or its login.
- Scan a real test PDF end-to-end to confirm the proxy is working (network requests should show
  `/api/admin/scan-pdf` on the new domain, not the Railway one).

## Rotating or revoking access later

Change or unset `CONTRACTOR_PDF_KEY` on Railway at any time — takes effect immediately, no
redeploy of the frontend needed, and never affects your own admin login.
