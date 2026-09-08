"""One-time backfill: generate a short referral code for every existing registered user.

Not load-bearing for correctness -- _referral_code_for_user in server.py already does
get-or-create lazily, so any account without a row here just gets one generated the first time
it's requested. This script only exists so that moment doesn't happen on someone's live first
page load; every existing account gets its code up front instead.

Run once, after applying referral_short_code_migration.sql:
    python migrate_referral_codes.py
"""
import asyncio
import httpx
import server


async def main():
    server.async_client = httpx.AsyncClient()
    headers = {"apikey": server.SUPABASE_SERVICE_KEY, "Authorization": f"Bearer {server.SUPABASE_SERVICE_KEY}"}

    resp = await server.async_client.get(
        f"{server.SUPABASE_URL}/auth/v1/admin/users", headers=headers, params={"per_page": 200}
    )
    resp.raise_for_status()
    data = resp.json()
    users = data.get("users", data) if isinstance(data, dict) else data
    print(f"Found {len(users)} registered users.")

    for u in users:
        user_id = u["id"]
        code = await server._referral_code_for_user(user_id)
        label = u.get("email") or u.get("phone") or user_id
        print(f"  {label} -> {code}")

    await server.async_client.aclose()
    print("Done.")


if __name__ == "__main__":
    asyncio.run(main())
