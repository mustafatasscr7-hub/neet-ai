"""/todos CRUD (Study To-do, Pro/Max only): plan gate, ownership isolation, 200-task cap, title
length. Fully offline, same shape as test_referral_register.py -- Supabase is an
httpx.MockTransport over an in-memory todo_tasks list, user_plan lookups and auth/v1/user are
faked too, so nothing here ever reaches the real Supabase project.

Run with: python test_todo_tasks.py
"""
import io
import json
import os
import sys

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import httpx
import server
from fastapi.testclient import TestClient

_passed = []
_failed = []


def check(label, condition, detail=""):
    if condition:
        _passed.append(label)
        print(f"  PASS: {label}")
    else:
        _failed.append(label)
        print(f"  FAIL: {label}  {detail}")


UID_FREE = "aaaaaaaa-0000-4000-8000-00000000000a"
UID_PRO = "bbbbbbbb-0000-4000-8000-00000000000b"
UID_PRO2 = "cccccccc-0000-4000-8000-00000000000c"
UID_PRO3 = "dddddddd-0000-4000-8000-00000000000d"
TOKENS = {"token-free": UID_FREE, "token-pro": UID_PRO, "token-pro2": UID_PRO2, "token-pro3": UID_PRO3}
PLANS = {UID_FREE: "free", UID_PRO: "pro", UID_PRO2: "pro", UID_PRO3: "max"}


class FakeSupabase:
    def __init__(self):
        self.todos = []
        self._next_id = 1

    def _param(self, request, name):
        raw = request.url.params.get(name)
        return raw[3:] if raw and raw.startswith("eq.") else raw

    def handle(self, request: httpx.Request) -> httpx.Response:
        path = request.url.path
        if path == "/auth/v1/user":
            uid = TOKENS.get(request.headers.get("authorization", "")[7:])
            return httpx.Response(200, json={"id": uid}) if uid else httpx.Response(401, json={})
        if path == "/rest/v1/user_plan":
            uid = self._param(request, "user_id")
            return httpx.Response(200, json=[{"plan": PLANS.get(uid, "free")}])
        if path == "/rest/v1/todo_tasks":
            user_id = self._param(request, "user_id")
            task_id = self._param(request, "id")
            if request.method == "GET":
                rows = [r for r in self.todos if r["user_id"] == user_id]
                total = len(rows)
                return httpx.Response(200, json=rows, headers={"content-range": f"0-{max(total - 1, 0)}/{total}"})
            if request.method == "POST":
                body = json.loads(request.content)
                row = {"id": f"task-{self._next_id}", "user_id": body["user_id"], "title": body["title"],
                       "due_date": body.get("due_date"), "done": False,
                       "created_at": "2026-10-20T00:00:00Z", "done_at": None}
                self._next_id += 1
                self.todos.append(row)
                return httpx.Response(201, json=[row])
            if request.method == "PATCH":
                body = json.loads(request.content)
                matched = [r for r in self.todos if r["id"] == task_id and r["user_id"] == user_id]
                for r in matched:
                    r.update(body)
                return httpx.Response(200, json=matched)
            if request.method == "DELETE":
                matched = [r for r in self.todos if r["id"] == task_id and r["user_id"] == user_id]
                self.todos = [r for r in self.todos if r is not matched[0]] if matched else self.todos
                return httpx.Response(200, json=matched)
        return httpx.Response(200, json=[])


def auth(token):
    return {"Authorization": f"Bearer {token}", "X-Real-IP": "203.0.113.20"}


def test_free_plan_gate(client):
    print("\n# Free user -> 403 plan_required on all four endpoints, no data")
    r = client.get("/todos", headers=auth("token-free"))
    check("GET /todos -> 403", r.status_code == 403 and r.json()["detail"]["code"] == "plan_required", r.text)
    r = client.post("/todos", json={"title": "x"}, headers=auth("token-free"))
    check("POST /todos -> 403", r.status_code == 403 and r.json()["detail"]["code"] == "plan_required", r.text)
    r = client.patch("/todos/whatever", json={"done": True}, headers=auth("token-free"))
    check("PATCH /todos/{id} -> 403", r.status_code == 403 and r.json()["detail"]["code"] == "plan_required", r.text)
    r = client.delete("/todos/whatever", headers=auth("token-free"))
    check("DELETE /todos/{id} -> 403", r.status_code == 403 and r.json()["detail"]["code"] == "plan_required", r.text)


def test_pro_crud(client):
    print("\n# Pro user: create, list, tick, edit, delete")
    r = client.post("/todos", json={"title": "Revise Mendel's laws", "due_date": "2026-10-21"}, headers=auth("token-pro"))
    check("create -> 200 with task", r.status_code == 200 and r.json()["task"]["title"] == "Revise Mendel's laws", r.text)
    task_id = r.json()["task"]["id"]

    r = client.get("/todos", headers=auth("token-pro"))
    check("list -> contains created task", any(x["id"] == task_id for x in r.json()["tasks"]), r.text)

    r = client.patch(f"/todos/{task_id}", json={"done": True}, headers=auth("token-pro"))
    check("tick done -> task.done true", r.status_code == 200 and r.json()["task"]["done"] is True, r.text)

    r = client.patch(f"/todos/{task_id}", json={"title": "Revise Mendel's laws (ch 5)"}, headers=auth("token-pro"))
    check("edit title -> updated", r.status_code == 200 and r.json()["task"]["title"] == "Revise Mendel's laws (ch 5)", r.text)

    r = client.delete(f"/todos/{task_id}", headers=auth("token-pro"))
    check("delete -> {deleted: true}", r.status_code == 200 and r.json()["deleted"] is True, r.text)

    r = client.get("/todos", headers=auth("token-pro"))
    check("deleted task gone from list", not any(x["id"] == task_id for x in r.json()["tasks"]), r.text)


def test_cross_user_isolation(client):
    print("\n# One user cannot read/edit/delete another user's task by id")
    r = client.post("/todos", json={"title": "Pro user 1's private task"}, headers=auth("token-pro"))
    task_id = r.json()["task"]["id"]

    r = client.get("/todos", headers=auth("token-pro2"))
    check("other user's list does not include it", not any(x["id"] == task_id for x in r.json()["tasks"]), r.text)

    r = client.patch(f"/todos/{task_id}", json={"done": True}, headers=auth("token-pro2"))
    check("other user patching it by id -> 404", r.status_code == 404, r.text)

    r = client.delete(f"/todos/{task_id}", headers=auth("token-pro2"))
    check("other user deleting it by id -> 404", r.status_code == 404, r.text)

    r = client.get("/todos", headers=auth("token-pro"))
    mine = [x for x in r.json()["tasks"] if x["id"] == task_id]
    check("owner still has it, untouched (still not done)", len(mine) == 1 and mine[0]["done"] is False, r.text)


def test_task_cap(client, fake):
    print("\n# 200-task cap")
    for i in range(200):
        fake.todos.append({"id": f"seed-{i}", "user_id": UID_PRO3, "title": f"seed {i}", "due_date": None,
                            "done": False, "created_at": "2026-01-01T00:00:00Z", "done_at": None})
    r = client.post("/todos", json={"title": "one task too many"}, headers=auth("token-pro3"))
    check("201st task rejected with 400", r.status_code == 400, r.text)


def test_title_length(client):
    print("\n# Title length validation (max 200 chars)")
    r = client.post("/todos", json={"title": "x" * 201}, headers=auth("token-pro2"))
    check("create with 201-char title -> 400", r.status_code == 400, r.text)

    r = client.post("/todos", json={"title": "short enough"}, headers=auth("token-pro2"))
    task_id = r.json()["task"]["id"]
    r = client.patch(f"/todos/{task_id}", json={"title": "y" * 201}, headers=auth("token-pro2"))
    check("update to 201-char title -> 400", r.status_code == 400, r.text)


if __name__ == "__main__":
    fake = FakeSupabase()
    server.async_client = httpx.AsyncClient(transport=httpx.MockTransport(fake.handle))
    tc = TestClient(server.app)
    test_free_plan_gate(tc)
    test_pro_crud(tc)
    test_cross_user_isolation(tc)
    test_task_cap(tc, fake)
    test_title_length(tc)
    print(f"\nTOTAL: {len(_passed)} passed, {len(_failed)} failed")
    sys.exit(1 if _failed else 0)
