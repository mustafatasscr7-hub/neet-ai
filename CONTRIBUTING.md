# Contributing

## Secret scanning (pre-commit hook)

This repo has had real API keys leak into git history more than once (keys
were rotated after the fact each time). To catch that before it happens
again instead of relying on everyone remembering, every commit is scanned
for secrets by [`detect-secrets`](https://github.com/Yelp/detect-secrets)
via the [`pre-commit`](https://pre-commit.com) framework.

### One-time setup (per machine)

```
pip install pre-commit detect-secrets
pre-commit install
```

That's it — `git commit` now runs the scan automatically from then on.

### What happens when it fires

Every `git commit` scans the staged files for anything shaped like a
credential (API keys, tokens, high-entropy strings, etc.) and blocks the
commit if it finds something new that isn't already recorded in
`.secrets.baseline`. `.secrets.baseline` is the list of findings that existed
in the repo at the time the hook was added (mostly the Supabase anon key,
which is meant to ship in frontend code and is safe by design) — it's what
lets the hook flag genuinely *new* secrets without choking on everything
already there.

- **If it blocks a real secret:** don't commit it. Remove it from the file,
  rotate the credential if it was ever pushed anywhere (even to a local
  branch), and load it from `.env` (see the `python-dotenv` usage in
  `server.py`) instead of hardcoding it.
- **If it blocks a false positive** (a test fixture, an intentionally public
  key, something clearly not sensitive): update the baseline rather than
  disabling the hook —
  ```
  detect-secrets scan --exclude-files node_modules --baseline .secrets.baseline
  git add .secrets.baseline
  ```
  then commit again.

Don't bypass the hook with `git commit --no-verify` to get around a finding
you haven't actually looked at — that defeats the point of having it.
