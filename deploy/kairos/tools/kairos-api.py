"""Minimal Kairos console API client. Reads credentials from ~/key/kairos_login.json
in memory only; never prints secrets. Usage: python3 kairos-api.py <action> [args]"""
import json
import sys
import urllib.request
import http.cookiejar
from pathlib import Path

BASE = "http://127.0.0.1:30800"
CREDS = Path.home() / "key" / "kairos_login.json"

TOKEN = None


def req(method, path, payload=None, *, _reauthenticated=False):
    data = json.dumps(payload).encode() if payload is not None else None
    headers = {"Content-Type": "application/json"}
    if TOKEN:
        headers["Authorization"] = "Bearer " + TOKEN
    r = urllib.request.Request(BASE + path, data=data, method=method, headers=headers)
    try:
        with urllib.request.urlopen(r, timeout=30) as resp:
            body = resp.read().decode("utf-8", "replace")
    except urllib.error.HTTPError as e:
        body = e.read().decode("utf-8", "replace")
        # Console sessions expire after 12 hours. Refresh only an authenticated
        # request rejected with 401, once; never retry transport errors or the
        # login endpoint. Credentials/token remain in memory as in login().
        if e.code == 401 and TOKEN and path != "/api/auth/login" and not _reauthenticated:
            login()
            return req(method, path, payload, _reauthenticated=True)
        try:
            return e.code, json.loads(body)
        except json.JSONDecodeError:
            return e.code, body
    try:
        return resp.status, json.loads(body)
    except json.JSONDecodeError:
        return resp.status, body


def login():
    global TOKEN
    creds = json.loads(CREDS.read_text(encoding="utf-8-sig"))
    status, body = req("POST", "/api/auth/login",
                       {"username": creds["username"], "password": creds["password"]})
    if status != 200 or not isinstance(body, dict) or not body.get("token"):
        print("login failed:", status, body)
        raise SystemExit(1)
    TOKEN = body["token"]
    print("login ok, namespace:", body.get("namespace"))


def main():
    action = sys.argv[1]
    login()
    if action == "state":
        for path in ("/api/me", "/api/cluster/overview", "/api/images", "/api/jobs"):
            status, body = req("GET", path)
            print(f"== {path} [{status}]")
            print(json.dumps(body, ensure_ascii=False, indent=1)[:3000])
    elif action == "submit":
        job = json.loads(sys.argv[2])
        status, body = req("POST", "/api/jobs", job)
        print("submit:", status, json.dumps(body, ensure_ascii=False)[:1000])
    elif action == "get":
        status, body = req("GET", sys.argv[2])
        print(f"[{status}]")
        print(json.dumps(body, ensure_ascii=False, indent=1)[:6000] if not isinstance(body, str) else body[:6000])
    elif action == "job":
        ns, name = sys.argv[2], sys.argv[3]
        status, body = req("GET", f"/api/jobs/{ns}/{name}")
        print("job:", status, json.dumps(body, ensure_ascii=False, indent=1)[:3000])
    elif action == "logs":
        ns, name = sys.argv[2], sys.argv[3]
        status, body = req("GET", f"/api/jobs/{ns}/{name}/logs")
        print("logs:", status)
        print(json.dumps(body, ensure_ascii=False, indent=1)[:4000] if not isinstance(body, str) else body[:4000])
    else:
        print("unknown action")
        raise SystemExit(2)


if __name__ == "__main__":
    main()
