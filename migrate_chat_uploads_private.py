"""One-time move of students' chat uploads out of the PUBLIC ncert-daigrams bucket
(chat-uploads/...) into the private chat-uploads bucket (server.py CHAT_UPLOADS_BUCKET).

Run after create_chat_uploads_bucket.sql AND after the backend that writes to the private bucket
is live (so nothing new lands in the public location after the copy):

    python migrate_chat_uploads_private.py                  # dry run: prints the plan, changes nothing
    python migrate_chat_uploads_private.py --apply          # copies into the private bucket, deletes nothing
    python migrate_chat_uploads_private.py --delete-public  # then removes ncert-daigrams/chat-uploads/*

Where each file goes:
  - chat-uploads/<user_id>/[pdfs/]<hex>.<ext>  ->  <user_id>/[pdfs/]<hex>.<ext>
  - chat-uploads/[pdfs/]<hex>.<ext> (flat, no owner in the path) -> the folder of the ONE student
    whose chats link it. Linked by nobody: an orphan nobody can see -- not copied, deleted by
    --delete-public. Linked by more than one student: not copied or deleted, reported for a manual
    decision.
chats.messages is never rewritten: server.py's _chat_upload_path resolves the old public URLs
against the caller's own private folder.

--delete-public refuses to run while any owned file is missing (or a different size) in the
private bucket, and only ever deletes keys under chat-uploads/ -- never a diagram.
"""
import argparse
import collections
import json
import re
import sys

import httpx
import server

URL = server.SUPABASE_URL
HEADERS = {"apikey": server.SUPABASE_SERVICE_KEY, "Authorization": f"Bearer {server.SUPABASE_SERVICE_KEY}"}
PUBLIC_BUCKET = server.DIAGRAMS_BUCKET
PRIVATE_BUCKET = server.CHAT_UPLOADS_BUCKET
PUBLIC_KEY_RE = re.compile(rf"chat-uploads/(?:(?P<uid>{server._CHAT_UPLOAD_UUID})/)?(?P<pdfs>pdfs/)?(?P<hex>[0-9a-f]{{32}})\.(?P<ext>jpg|jpeg|png|webp|gif|pdf)")
# Any stored form of an attachment, for "which students link this file" -- the hex is unique.
CHAT_REF_HEX_RE = re.compile(rf"chat-upload(?:s/|://)(?:{server._CHAT_UPLOAD_UUID}/)?(?:pdfs/)?([0-9a-f]{{32}})\.")


def list_objects(client, bucket, prefix):
    """{key: size} under prefix, recursive."""
    out, folders = {}, [prefix]
    while folders:
        folder = folders.pop()
        offset = 0
        while True:
            resp = client.post(f"{URL}/storage/v1/object/list/{bucket}", headers=HEADERS,
                               json={"prefix": folder, "limit": 1000, "offset": offset})
            resp.raise_for_status()
            items = resp.json()
            for item in items:
                key = f"{folder}/{item['name']}" if folder else item["name"]
                if item.get("id"):
                    out[key] = (item.get("metadata") or {}).get("size")
                else:
                    folders.append(key)
            if len(items) < 1000:
                break
            offset += 1000
    return out


def chat_links(client):
    """{hex: {user_id}} across every chat."""
    links, offset = collections.defaultdict(set), 0
    while True:
        resp = client.get(f"{URL}/rest/v1/chats", headers=HEADERS,
                          params={"select": "user_id,messages", "order": "id.asc", "limit": 100, "offset": offset})
        resp.raise_for_status()
        rows = resp.json()
        for row in rows:
            for h in CHAT_REF_HEX_RE.findall(json.dumps(row.get("messages"))):
                if row.get("user_id"):
                    links[h].add(row["user_id"])
        if len(rows) < 100:
            return links
        offset += 100


def make_plan(client):
    public = list_objects(client, PUBLIC_BUCKET, "chat-uploads")
    links = chat_links(client)
    plan = {"copy": {}, "orphan": [], "conflict": [], "unexpected": []}
    for key, size in sorted(public.items()):
        m = PUBLIC_KEY_RE.fullmatch(key)
        if not m or (m["ext"] == "pdf") != bool(m["pdfs"]):
            plan["unexpected"].append(key)
            continue
        owners = {m["uid"]} if m["uid"] else links.get(m["hex"], set())
        if len(owners) == 1:
            owner = next(iter(owners))
            plan["copy"][key] = (f"{owner}/{m['pdfs'] or ''}{m['hex']}.{m['ext']}", size)
        elif not owners:
            plan["orphan"].append(key)
        else:
            plan["conflict"].append((key, sorted(owners)))
    return public, plan


def private_sizes(client, targets):
    folders = {t.rsplit("/", 1)[0] for t in targets}
    sizes = {}
    for folder in folders:
        sizes.update(list_objects(client, PRIVATE_BUCKET, folder))
    return sizes


def copy_one(client, src, dst):
    resp = client.post(f"{URL}/storage/v1/object/copy", headers=HEADERS,
                       json={"bucketId": PUBLIC_BUCKET, "sourceKey": src, "destinationBucket": PRIVATE_BUCKET, "destinationKey": dst})
    if resp.status_code < 400:
        return "copy"
    # Fallback for a Storage version without cross-bucket copy: download + upload, same bytes and type.
    got = client.get(f"{URL}/storage/v1/object/{PUBLIC_BUCKET}/{src}", headers=HEADERS)
    got.raise_for_status()
    put = client.post(f"{URL}/storage/v1/object/{PRIVATE_BUCKET}/{dst}", content=got.content,
                      headers={**HEADERS, "Content-Type": got.headers.get("content-type", "application/octet-stream"),
                               "cache-control": f"max-age={server.CHAT_UPLOAD_URL_TTL_SECONDS}"})
    put.raise_for_status()
    return f"download+upload (copy said {resp.status_code}: {resp.text[:80]})"


def main():
    parser = argparse.ArgumentParser()
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--apply", action="store_true", help="copy owned files into the private bucket")
    mode.add_argument("--delete-public", action="store_true", help="delete ncert-daigrams/chat-uploads/* after a verified --apply")
    args = parser.parse_args()

    with httpx.Client(timeout=60) as client:
        public, plan = make_plan(client)
        print(f"Public chat-uploads/ objects: {len(public)}")
        print(f"  to copy into {PRIVATE_BUCKET}/: {len(plan['copy'])}")
        print(f"  orphans (no owner, linked from no chat): {len(plan['orphan'])}")
        print(f"  linked by more than one student (left alone): {len(plan['conflict'])}")
        print(f"  unexpected names (left alone): {len(plan['unexpected'])}")
        for key, owners in plan["conflict"]:
            print(f"    CONFLICT {key}: {owners}")
        for key in plan["unexpected"]:
            print(f"    UNEXPECTED {key}")

        existing = private_sizes(client, [dst for dst, _ in plan["copy"].values()])
        pending = {src: (dst, size) for src, (dst, size) in plan["copy"].items() if existing.get(dst) != size}
        print(f"  already in the private bucket with the same size: {len(plan['copy']) - len(pending)}")

        if args.apply:
            for src, (dst, size) in pending.items():
                how = copy_one(client, src, dst)
                print(f"  copied {src} -> {dst} ({size} bytes, {how})")
            existing = private_sizes(client, [dst for dst, _ in plan["copy"].values()])
            bad = [src for src, (dst, size) in plan["copy"].items() if existing.get(dst) != size]
            print(f"Verified: {len(plan['copy']) - len(bad)}/{len(plan['copy'])} present with matching size.")
            if bad:
                print("NOT verified:", *bad, sep="\n  ")
                sys.exit(1)
        elif args.delete_public:
            if pending:
                print("Refusing: these owned files aren't (correctly) in the private bucket yet -- run --apply first:")
                print(*pending, sep="\n  ")
                sys.exit(1)
            doomed = sorted(list(plan["copy"]) + plan["orphan"])
            assert all(k.startswith("chat-uploads/") for k in doomed)
            for i in range(0, len(doomed), 100):
                resp = client.request("DELETE", f"{URL}/storage/v1/object/{PUBLIC_BUCKET}", headers=HEADERS,
                                      json={"prefixes": doomed[i:i + 100]})
                resp.raise_for_status()
            left = list_objects(client, PUBLIC_BUCKET, "chat-uploads")
            print(f"Deleted {len(doomed)} public objects. Left under chat-uploads/: {len(left)}"
                  + (" (the conflicts/unexpected above)" if left else ""))
        else:
            print("Dry run -- nothing changed. Re-run with --apply, then --delete-public.")


if __name__ == "__main__":
    main()
