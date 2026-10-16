"""Regression test for private chat attachments: /chat-image-upload, /chat-pdf-upload,
/chat-uploads/sign, the image-correction re-fetch and /account/delete's upload cleanup.

Run with: python test_chat_uploads_private.py
Plain pass/fail harness (no pytest), fully offline: server.async_client is swapped for an httpx
client whose transport is a small in-memory stand-in for Supabase Auth, Storage and PostgREST, so
the real endpoint code runs without touching any real bucket or table.
"""
import asyncio
import base64
import io
import json
import os
import sys
import time
from contextlib import redirect_stdout

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import httpx
import server
from fastapi.testclient import TestClient
from PIL import Image

_passed = []
_failed = []


def check(label, condition, detail=""):
    if condition:
        _passed.append(label)
        print(f"  PASS: {label}")
    else:
        _failed.append(label)
        print(f"  FAIL: {label}  {detail}")


UID_A = "aaaaaaaa-0000-4000-8000-00000000000a"
UID_B = "bbbbbbbb-0000-4000-8000-00000000000b"
TOKENS = {"token-a": UID_A, "token-b": UID_B}
AUTH_A = {"Authorization": "Bearer token-a"}
AUTH_B = {"Authorization": "Bearer token-b"}
HEX = ["%032x" % i for i in range(1, 20)]
BASE = server.SUPABASE_URL
PUBLIC = f"{BASE}/storage/v1/object/public/ncert-daigrams"


class FakeSupabase:
    def __init__(self):
        self.objects = {"chat-uploads": {}, "ncert-daigrams": {}}  # bucket -> {path: (bytes, content type)}
        self.chats = []
        self.requests = []      # (method, url) for every request
        self.sign_calls = 0
        self.fail_sign = False

    def handle(self, request):
        url, path, method = str(request.url), request.url.path, request.method
        self.requests.append((method, url))
        if not url.startswith(BASE):
            return httpx.Response(200, content=b"internal metadata")
        service = request.headers.get("apikey") == server.SUPABASE_SERVICE_KEY
        if path == "/auth/v1/user":
            uid = TOKENS.get(request.headers.get("authorization", "")[7:])
            return httpx.Response(200, json={"id": uid}) if uid else httpx.Response(401, json={"msg": "invalid JWT"})
        if path.startswith("/auth/v1/admin/users/"):
            return httpx.Response(200, json={})
        if path.startswith("/storage/v1/object/sign/"):
            assert service
            self.sign_calls += 1
            if self.fail_sign:
                return httpx.Response(500, json={"message": "boom"})
            bucket = path.split("/")[5]
            body = json.loads(request.content)
            return httpx.Response(200, json=[
                {"path": p, "signedURL": f"/object/sign/{bucket}/{p}?token=t{self.sign_calls}", "error": None}
                if p in self.objects[bucket] else
                {"path": p, "signedURL": None, "error": "Either the object does not exist or you do not have access to it"}
                for p in body["paths"]
            ])
        if path.startswith("/storage/v1/object/list/"):
            bucket = path.split("/")[5]
            prefix = json.loads(request.content)["prefix"].rstrip("/") + "/"
            names = {p[len(prefix):] for p in self.objects[bucket] if p.startswith(prefix)}
            items = [{"name": n, "id": "x"} for n in names if "/" not in n] + [{"name": n.split("/")[0], "id": None} for n in names if "/" in n]
            return httpx.Response(200, json=items)
        if path.startswith("/storage/v1/object/"):
            rest = path[len("/storage/v1/object/"):]
            bucket, _, key = rest.partition("/")
            if method == "POST":
                assert service
                self.objects[bucket][key] = (request.content, request.headers.get("content-type"))
                return httpx.Response(200, json={"Key": f"{bucket}/{key}"})
            if method == "GET":
                assert service
                if key not in self.objects[bucket]:
                    return httpx.Response(400, json={"error": "not_found"})
                data, ctype = self.objects[bucket][key]
                return httpx.Response(200, content=data, headers={"content-type": ctype})
            if method == "DELETE":
                for p in json.loads(request.content)["prefixes"]:
                    self.objects[bucket].pop(p, None)
                return httpx.Response(200, json=[])
        if path == "/rest/v1/chats" and method == "GET":
            params = dict(request.url.params)
            uid = params["user_id"][3:]
            rows = [c for c in self.chats if c["user_id"] == uid]
            off, lim = int(params.get("offset", 0)), int(params.get("limit", 100))
            return httpx.Response(200, json=[{"messages": c["messages"]} for c in rows[off:off + lim]])
        if path.startswith("/rest/v1/") and method in ("PATCH", "DELETE"):
            if path == "/rest/v1/chats":
                uid = dict(request.url.params)["user_id"][3:]
                self.chats = [c for c in self.chats if c["user_id"] != uid]
            return httpx.Response(204, headers={"content-range": "*/0"})
        raise AssertionError(f"fake Supabase: unhandled {method} {path}")


def jpeg_b64():
    out = io.BytesIO()
    Image.new("RGB", (8, 8), (200, 10, 10)).save(out, format="JPEG")
    return base64.b64encode(out.getvalue()).decode()


def reset():
    server._rate_limit_buckets.clear()
    server._mock_test_user_buckets.clear()
    server._chat_upload_url_cache.clear()


# ---------------------------------------------------------------- tests
def test_parser():
    print("\n# _chat_upload_path: every stored form, only ever inside the caller's folder")
    p = server._chat_upload_path
    h = HEX[0]
    cases = [
        (f"chat-upload://{UID_A}/{h}.jpg", UID_A, f"{UID_A}/{h}.jpg"),
        (f"chat-upload://{UID_A}/pdfs/{h}.pdf", UID_A, f"{UID_A}/pdfs/{h}.pdf"),
        (f"{PUBLIC}/chat-uploads/{UID_A}/{h}.png", UID_A, f"{UID_A}/{h}.png"),
        (f"{PUBLIC}/chat-uploads/{UID_A}/pdfs/{h}.pdf", UID_A, f"{UID_A}/pdfs/{h}.pdf"),
        (f"{PUBLIC}/chat-uploads/{h}.webp", UID_A, f"{UID_A}/{h}.webp"),
        (f"{PUBLIC}/chat-uploads/{h}.jpeg", UID_A, f"{UID_A}/{h}.jpeg"),        # Sept 2026 uploads used .jpeg
        (f"{PUBLIC}/chat-uploads/pdfs/{h}.pdf", UID_B, f"{UID_B}/pdfs/{h}.pdf"),
        (f"{BASE}/storage/v1/object/sign/chat-uploads/{UID_A}/{h}.gif?token=abc.def-ghi", UID_A, f"{UID_A}/{h}.gif"),
        (f"  chat-upload://{UID_A}/{h}.jpg  ", UID_A, f"{UID_A}/{h}.jpg"),
    ]
    for ref, uid, want in cases:
        check(f"maps {ref[-40:]!r}", p(ref, uid) == want, str(p(ref, uid)))
    rejected = [
        (f"chat-upload://{UID_B}/{h}.jpg", UID_A),                               # someone else's folder
        (f"{PUBLIC}/chat-uploads/{UID_B}/{h}.jpg", UID_A),
        (f"{BASE}/storage/v1/object/sign/chat-uploads/{UID_B}/{h}.jpg?token=x", UID_A),
        (f"chat-upload://{UID_A}/{h}.pdf", UID_A),                               # pdf outside pdfs/
        (f"chat-upload://{UID_A}/pdfs/{h}.jpg", UID_A),                          # image inside pdfs/
        (f"chat-upload://{UID_A}/../{UID_B}/{h}.jpg", UID_A),                    # traversal
        (f"chat-upload://{UID_A}/{'AB' * 16}.jpg", UID_A),                    # hex is lowercase only
        (f"chat-upload://{UID_A}/{h}.svg", UID_A),
        (f"{PUBLIC}/{h}.png", UID_A),                                            # a diagram, stays public
        (f"https://evil.example/storage/v1/object/public/ncert-daigrams/chat-uploads/{h}.jpg", UID_A),
        (f"chat-upload://{UID_A}/{h}.jpg", ""),                                  # guest
        (None, UID_A), (123, UID_A), ("data:image/jpeg;base64,AAAA", UID_A), ("file:///x.jpg", UID_A),
    ]
    for ref, uid in rejected:
        check(f"rejects {str(ref)[-45:]!r} for {'guest' if not uid else uid[:8]}", p(ref, uid) is None, str(p(ref, uid)))


def test_uploads(client, fake):
    print("\n# uploads go to the private bucket and return {ref, url}")
    reset()
    r = client.post("/chat-image-upload", json={"data": jpeg_b64(), "media_type": "image/jpeg", "user_id": UID_B}, headers=AUTH_A).json()
    ref = r.get("ref", "")
    key = ref[len("chat-upload://"):]
    check("image: ref is chat-upload://<own id>/<hex>.jpg (body user_id ignored)", key.startswith(f"{UID_A}/") and key.endswith(".jpg") and len(key.split("/")[1]) == 36, ref)
    check("image: stored only in chat-uploads, re-encoded JPEG", key in fake.objects["chat-uploads"] and not fake.objects["ncert-daigrams"]
          and fake.objects["chat-uploads"][key][1] == "image/jpeg" and fake.objects["chat-uploads"][key][0][:2] == b"\xff\xd8")
    check("image: url is a signed URL for old app builds", (r.get("url") or "").startswith(f"{BASE}/storage/v1/object/sign/chat-uploads/{key}?token="), str(r.get("url")))
    upload = next(u for m, u in fake.requests if m == "POST" and u.endswith(key))
    check("image: write is authenticated with the service key", upload is not None)
    r = client.post("/chat-pdf-upload", json={"data": base64.b64encode(b"%PDF-1.4 test").decode(), "user_id": UID_A}, headers=AUTH_A).json()
    check("pdf: ref is chat-upload://<own id>/pdfs/<hex>.pdf", r.get("ref", "").startswith(f"chat-upload://{UID_A}/pdfs/") and r["ref"].endswith(".pdf")
          and fake.objects["chat-uploads"][r["ref"][14:]][1] == "application/pdf", str(r))
    check("no token -> 401 (image and pdf)", client.post("/chat-image-upload", json={"data": jpeg_b64(), "user_id": UID_A}).status_code == 401
          and client.post("/chat-pdf-upload", json={"data": "JVBERg==", "user_id": UID_A}).status_code == 401)
    fake.fail_sign = True
    r = client.post("/chat-image-upload", json={"data": jpeg_b64(), "media_type": "image/jpeg", "user_id": UID_A}, headers=AUTH_A).json()
    fake.fail_sign = False
    check("signing failure after a good write still returns the ref (url null)", r.get("ref", "").startswith("chat-upload://") and r.get("url") is None, str(r))
    reset()


def test_sign(client, fake):
    print("\n# /chat-uploads/sign")
    reset()
    own = [f"{UID_A}/{HEX[1]}.jpg", f"{UID_A}/pdfs/{HEX[2]}.pdf", f"{UID_A}/{HEX[3]}.png"]
    for p in own:
        fake.objects["chat-uploads"][p] = (b"x", "image/jpeg")
    fake.objects["chat-uploads"][f"{UID_B}/{HEX[4]}.jpg"] = (b"x", "image/jpeg")
    refs = [f"chat-upload://{own[0]}", f"chat-upload://{own[1]}",
            f"{PUBLIC}/chat-uploads/{HEX[3]}.png",                 # legacy flat, migrated into A's folder
            f"chat-upload://{UID_B}/{HEX[4]}.jpg",                 # B's file
            f"chat-upload://{UID_A}/{HEX[5]}.jpg",                 # A's, but gone
            f"{PUBLIC}/{HEX[6]}.png", "not a ref", f"chat-upload://{own[0]}"]
    before = fake.sign_calls
    r = client.post("/chat-uploads/sign", json={"refs": refs}, headers=AUTH_A)
    body = r.json()
    urls = body.get("urls", {})
    check("200 with one entry per distinct ref", r.status_code == 200 and len(urls) == 7, r.text[:200])
    check("own refs (new + legacy flat) get signed URLs", all((urls.get(x) or "").startswith(f"{BASE}/storage/v1/object/sign/chat-uploads/{UID_A}/") for x in refs[:3]), str(urls))
    check("another student's file -> null", urls.get(refs[3]) is None)
    check("missing file / diagram / junk -> null", urls.get(refs[4]) is None and urls.get(refs[5]) is None and urls.get(refs[6]) is None)
    check("one Storage sign call for the whole batch", fake.sign_calls - before == 1, str(fake.sign_calls - before))
    check("expires_in close to the TTL", 3500 <= body.get("expires_in", 0) <= 3600, str(body.get("expires_in")))
    r2 = client.post("/chat-uploads/sign", json={"refs": refs[:3]}, headers=AUTH_A).json()
    check("reopening the chat: same URLs from cache, no new Storage call", r2["urls"] == {x: urls[x] for x in refs[:3]} and fake.sign_calls - before == 1)
    p0 = server._chat_upload_path(refs[0], UID_A)
    url0, _ = server._chat_upload_url_cache[p0]
    server._chat_upload_url_cache[p0] = (url0, time.time() + 300)   # under 10 minutes left
    r3 = client.post("/chat-uploads/sign", json={"refs": refs[:1]}, headers=AUTH_A).json()
    check("a URL with under 10 minutes left is re-signed", r3["urls"][refs[0]] != url0 and fake.sign_calls - before == 2)
    r = client.post("/chat-uploads/sign", json={"refs": [f"chat-upload://{UID_A}/{HEX[1]}.jpg"]}, headers=AUTH_B).json()
    check("student B can't sign A's reference", r["urls"][f"chat-upload://{UID_A}/{HEX[1]}.jpg"] is None)
    many = [f"chat-upload://{UID_A}/{'%032x' % (1000 + i)}.jpg" for i in range(150)]
    r = client.post("/chat-uploads/sign", json={"refs": many}, headers=AUTH_A).json()
    check("capped at 100 refs per request", len(r["urls"]) == 100)
    check("no token -> 401", client.post("/chat-uploads/sign", json={"refs": refs}).status_code == 401)
    check("bad body -> 422", client.post("/chat-uploads/sign", json={"refs": "x"}, headers=AUTH_A).status_code == 422)
    server._chat_upload_url_cache.clear()
    fake.fail_sign = True
    with redirect_stdout(io.StringIO()):
        code = client.post("/chat-uploads/sign", json={"refs": refs[:1]}, headers=AUTH_A).status_code
    fake.fail_sign = False
    check("Storage failure -> 502 (clients fall back)", code == 502, str(code))
    reset()
    codes = [client.post("/chat-uploads/sign", json={"refs": []}, headers=AUTH_A).status_code for _ in range(121)]
    check("120 per minute per student, the 121st is 429", codes[:120] == [200] * 120 and codes[120] == 429, str(codes[-3:]))
    check("another student is unaffected", client.post("/chat-uploads/sign", json={"refs": []}, headers=AUTH_B).status_code == 200)
    reset()


def test_refetch(fake):
    print("\n# image-correction re-fetch: own private files only, never an arbitrary URL")
    fake.objects["chat-uploads"][f"{UID_A}/{HEX[7]}.png"] = (b"\x89PNG own", "image/png")
    fake.objects["chat-uploads"][f"{UID_B}/{HEX[8]}.png"] = (b"\x89PNG other", "image/png")
    fake.requests.clear()
    refs = ["http://169.254.169.254/latest/meta-data/", f"chat-upload://{UID_B}/{HEX[8]}.png",
            f"chat-upload://{UID_A}/{HEX[7]}.png", f"chat-upload://{UID_A}/pdfs/{HEX[9]}.pdf"]
    got = asyncio.run(server._refetch_images_as_attachments(refs, UID_A))
    check("SSRF: the arbitrary URL is never requested", not any("169.254" in u for _, u in fake.requests), str(fake.requests))
    check("another student's image is not fetched", not any(UID_B in u for _, u in fake.requests))
    check("only the student's own image comes back", len(got) == 1 and base64.b64decode(got[0].data) == b"\x89PNG own" and got[0].media_type == "image/png")
    fake.requests.clear()
    check("guest (no user id): nothing fetched", asyncio.run(server._refetch_images_as_attachments(refs, "")) == [] and not fake.requests)
    history = [{"role": "user", "text": "", "images": [f"chat-upload://{UID_A}/{HEX[7]}.png"]}, {"role": "ai", "text": "That's a monocot."}]
    check("correction detection hands the stored references through unchanged",
          server._detect_image_correction("no, this is a dicot", history, [], None) == [f"chat-upload://{UID_A}/{HEX[7]}.png"])


def test_account_delete(client, fake):
    print("\n# /account/delete removes the student's uploads from both buckets, nobody else's")
    reset()
    fake.objects = {"chat-uploads": {}, "ncert-daigrams": {}}
    own_private = [f"{UID_A}/{HEX[10]}.jpg", f"{UID_A}/pdfs/{HEX[11]}.pdf"]
    own_public = [f"chat-uploads/{UID_A}/{HEX[12]}.jpg", f"chat-uploads/{HEX[13]}.png"]
    others = {"chat-uploads": [f"{UID_B}/{HEX[14]}.jpg"], "ncert-daigrams": [f"chat-uploads/{UID_B}/{HEX[15]}.jpg", f"{HEX[16]}.png", f"chat-uploads/{HEX[17]}.png"]}
    for p in own_private + others["chat-uploads"]:
        fake.objects["chat-uploads"][p] = (b"x", "image/jpeg")
    for p in own_public + others["ncert-daigrams"]:
        fake.objects["ncert-daigrams"][p] = (b"x", "image/jpeg")
    fake.chats = [{"user_id": UID_A, "messages": [
        {"role": "user", "images": [f"{PUBLIC}/{own_public[1]}", f"chat-upload://{own_private[0]}"]},
        {"role": "user", "images": [f"{PUBLIC}/chat-uploads/{UID_B}/{HEX[15]}.jpg"]},   # crafted: B's file
        {"role": "ai", "diagram": {"image_url": f"{PUBLIC}/{HEX[16]}.png"}},             # a diagram
    ]}]
    server._chat_upload_url_cache[own_private[0]] = ("https://signed", time.time() + 3000)
    server._chat_upload_url_cache[others["chat-uploads"][0]] = ("https://signed-b", time.time() + 3000)
    with redirect_stdout(io.StringIO()):
        r = client.post("/account/delete", json={"confirm": "DELETE"}, headers=AUTH_A)
    check("200 deleted", r.status_code == 200 and r.json() == {"deleted": True}, r.text[:200])
    check("own private files gone (images + pdfs/)", not any(p in fake.objects["chat-uploads"] for p in own_private))
    check("own public leftovers gone (own folder + legacy flat referenced in chats)", not any(p in fake.objects["ncert-daigrams"] for p in own_public))
    check("other student's files and the diagram untouched",
          all(p in fake.objects[b] for b, ps in others.items() for p in ps), str(fake.objects))
    check("own cached signed URLs forgotten, others kept",
          own_private[0] not in server._chat_upload_url_cache and others["chat-uploads"][0] in server._chat_upload_url_cache)
    reset()


if __name__ == "__main__":
    fake = FakeSupabase()
    server.async_client = httpx.AsyncClient(transport=httpx.MockTransport(fake.handle))
    tc = TestClient(server.app)  # no `with`: skips startup hooks, nothing here needs them
    test_parser()
    test_uploads(tc, fake)
    test_sign(tc, fake)
    test_refetch(fake)
    test_account_delete(tc, fake)
    print(f"\nTOTAL: {len(_passed)} passed, {len(_failed)} failed")
    sys.exit(1 if _failed else 0)
