"""Regression test for the Active Sessions location lookup (local DB-IP database, no third party).

Run with: python test_session_location.py
Plain pass/fail harness, offline: the download goes to an httpx MockTransport and
maxminddb.open_database is swapped for a fake reader, so no real ~120 MB file is needed. Set
GEOIP_TEST_DB to a real dbip-city-lite-*.mmdb to also check a few real addresses.
"""
import asyncio
import gzip
import io
import os
import sys
import tempfile
from contextlib import redirect_stdout
from datetime import date

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import httpx
import maxminddb
import server

_passed = []
_failed = []


def check(label, condition, detail=""):
    if condition:
        _passed.append(label)
        print(f"  PASS: {label}")
    else:
        _failed.append(label)
        print(f"  FAIL: {label}  {detail}")


class FakeReader:
    def __init__(self, records, name=""):
        self.records, self.name, self.closed = records, name, False

    def get(self, ip):
        value = self.records.get(ip)
        if isinstance(value, Exception):
            raise value
        return value

    def close(self):
        self.closed = True


def lookup(ip):
    return asyncio.run(server._lookup_session_location(ip))


def test_months():
    print("\n# _geoip_months: this month's release, then last month's")
    check("mid-month", server._geoip_months(date(2026, 10, 16)) == ["2026-10", "2026-09"])
    check("January rolls back to December", server._geoip_months(date(2027, 1, 2)) == ["2027-01", "2026-12"])


def test_lookup():
    print("\n# _lookup_session_location: in-memory only, always a label")
    requests = []
    server.async_client = httpx.AsyncClient(transport=httpx.MockTransport(lambda r: requests.append(r) or httpx.Response(500)))
    server._geoip.update(reader=None, month=None)
    check("database not loaded yet -> Unknown", lookup("49.36.0.1") == "Unknown")
    server._geoip.update(reader=FakeReader({
        "49.36.0.1": {"city": {"names": {"en": "Navi Mumbai (Ghansoli)"}}, "subdivisions": [{"names": {"en": "Maharashtra"}}]},
        "106.51.0.1": {"city": {"names": {"en": "Bengaluru"}}, "subdivisions": [{"names": {"en": "Karnataka"}}]},
        "2405:201:4000::1": {"city": {"names": {"en": "Delhi"}}, "subdivisions": [{"names": {"en": "Delhi"}}]},
        "8.8.4.4": {"country": {"iso_code": "US"}, "subdivisions": [{"names": {"en": "California"}}]},
        "1.1.1.1": {"city": {"names": {"en": "Sydney"}}},
        "9.9.9.9": {},
        "4.4.4.4": RuntimeError("corrupt node"),
    }), month="2026-10")
    check("city + region", lookup("106.51.0.1") == "Bengaluru, Karnataka", lookup("106.51.0.1"))
    check("a neighbourhood in brackets is dropped", lookup("49.36.0.1") == "Navi Mumbai, Maharashtra", lookup("49.36.0.1"))
    check("IPv6", lookup("2405:201:4000::1") == "Delhi, Delhi")
    check("region only / city only", lookup("8.8.4.4") == "California" and lookup("1.1.1.1") == "Sydney")
    check("empty record / unknown IP / reader error -> Unknown", lookup("9.9.9.9") == "Unknown" and lookup("5.5.5.5") == "Unknown" and lookup("4.4.4.4") == "Unknown")
    check("private, loopback, junk, empty -> Unknown",
          all(lookup(ip) == "Unknown" for ip in ("10.1.2.3", "192.168.1.5", "172.16.0.9", "127.0.0.1", "::1", "unknown", "not an ip", "")))
    check("no network request is ever made", requests == [], str(requests))


def test_download_and_refresh():
    print("\n# download + refresh: background, monthly, never a half-written file")
    tmp = tempfile.mkdtemp(prefix="geoip-test-")
    server.GEOIP_DB_DIR = tmp
    published = {}   # month -> gz bytes
    hits = []

    def handle(request):
        hits.append(request.url.path)
        month = request.url.path.rsplit("dbip-city-lite-", 1)[1][:7]
        return httpx.Response(200, content=published[month]) if month in published else httpx.Response(404)

    server.async_client = httpx.AsyncClient(transport=httpx.MockTransport(handle))
    opened = []
    real_open = maxminddb.open_database

    def fake_open(path):
        data = open(path, "rb").read()
        if not data.startswith(b"MMDB"):
            raise ValueError("not a MaxMind DB")
        reader = FakeReader({}, name=os.path.basename(path))
        opened.append(reader)
        return reader

    maxminddb.open_database = fake_open
    current, previous = server._geoip_months()
    try:
        server._geoip.update(reader=None, month=None)
        with redirect_stdout(io.StringIO()):
            asyncio.run(server._refresh_geoip_db())
        check("nothing published -> stays unloaded (lookups say Unknown)", server._geoip["reader"] is None and lookup("49.36.0.1") == "Unknown")
        published[previous] = gzip.compress(b"MMDB previous")
        with redirect_stdout(io.StringIO()):
            asyncio.run(server._refresh_geoip_db())
        check("this month not out yet -> last month's release is used", server._geoip["month"] == previous and opened[-1].name == f"dbip-city-lite-{previous}.mmdb")
        check("unpacked atomically: no .part files left", sorted(os.listdir(tmp)) == [f"dbip-city-lite-{previous}.mmdb"], str(os.listdir(tmp)))
        hits.clear()
        with redirect_stdout(io.StringIO()):
            asyncio.run(server._refresh_geoip_db())
        check("daily re-check: asks only for the new month, never re-downloads the loaded one",
              len(hits) == 1 and current in hits[0] and server._geoip["month"] == previous, str(hits))
        published[current] = gzip.compress(b"MMDB current")
        old_reader = server._geoip["reader"]
        with redirect_stdout(io.StringIO()):
            asyncio.run(server._refresh_geoip_db())
        check("new month published -> switched, old reader closed, old file deleted",
              server._geoip["month"] == current and old_reader.closed and os.listdir(tmp) == [f"dbip-city-lite-{current}.mmdb"], str(os.listdir(tmp)))
        hits.clear()
        with redirect_stdout(io.StringIO()):
            asyncio.run(server._refresh_geoip_db())
        check("up to date -> no request at all", hits == [])
        # a corrupt download is deleted and doesn't replace a working database
        server._geoip.update(month=None)
        os.remove(os.path.join(tmp, f"dbip-city-lite-{current}.mmdb"))
        published[current] = gzip.compress(b"garbage")
        good = server._geoip["reader"]
        published.pop(previous)
        with redirect_stdout(io.StringIO()):
            asyncio.run(server._refresh_geoip_db())
        check("corrupt database -> deleted, the working reader stays", server._geoip["reader"] is good and os.listdir(tmp) == [], str(os.listdir(tmp)))

        def boom(request):
            raise httpx.ConnectError("db-ip.com unreachable")
        server.async_client = httpx.AsyncClient(transport=httpx.MockTransport(boom))
        out = io.StringIO()
        with redirect_stdout(out):
            asyncio.run(server._refresh_geoip_db())
        check("network failure -> logged, nothing raised, reader unchanged", "GEOIP DOWNLOAD FAILED" in out.getvalue() and server._geoip["reader"] is good)
    finally:
        maxminddb.open_database = real_open
        server._geoip.update(reader=None, month=None)


def test_real_database():
    path = os.getenv("GEOIP_TEST_DB")
    if not path:
        print("\n# real database: skipped (set GEOIP_TEST_DB to a dbip-city-lite-*.mmdb)")
        return
    print(f"\n# real database: {os.path.basename(path)}")
    server._geoip.update(reader=maxminddb.open_database(path), month="test")
    for ip, region in (("8.8.8.8", "California"), ("49.36.0.1", "Maharashtra"), ("106.51.0.1", "Karnataka")):
        got = lookup(ip)
        check(f"{ip} -> ... {region}", got.endswith(region) and "(" not in got, got)
    server._geoip["reader"].close()
    server._geoip.update(reader=None, month=None)


if __name__ == "__main__":
    test_months()
    test_lookup()
    test_download_and_refresh()
    test_real_database()
    print(f"\nTOTAL: {len(_passed)} passed, {len(_failed)} failed")
    sys.exit(1 if _failed else 0)
