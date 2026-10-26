"""Local test server for the survey in docs/ (the published version is static, see survey/README.md).
Run from the project root: python survey/server.py [port]. Submissions are appended to survey/responses.jsonl."""
import json
import os
import sys
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.join(os.path.dirname(HERE), "docs")
OUT = os.path.join(HERE, "responses.jsonl")
PUBLIC = ("/", "/index.html", "/items.json", "/config.js")


class Handler(SimpleHTTPRequestHandler):
    def __init__(self, *a, **k):
        super().__init__(*a, directory=ROOT, **k)

    def do_GET(self):
        path = self.path.split("?")[0]
        if path == "/submit":
            counts = {}
            if os.path.exists(OUT):
                for line in open(OUT):
                    for t in json.loads(line).get("trials", []):
                        counts[t["item"]] = counts.get(t["item"], 0) + 1
            body = json.dumps(counts).encode()
            self.send_response(200); self.send_header("Content-Type", "application/json"); self.end_headers(); self.wfile.write(body)
            return
        if path in PUBLIC or (path.startswith("/audio/") and ".." not in path):
            return super().do_GET()
        self.send_error(404)

    def do_POST(self):
        if self.path != "/submit":
            return self.send_error(404)
        body = self.rfile.read(int(self.headers.get("Content-Length", 0)))
        try:
            json.loads(body)
        except ValueError:
            return self.send_error(400)
        with open(OUT, "ab") as f:
            f.write(body.replace(b"\n", b" ") + b"\n")
        bs = json.loads(body).get("besample")  # mirrors the Apps Script: code = bnum * BESAMPLE_BKEY
        key = os.environ.get("BESAMPLE_BKEY")
        code = str(int(bs["bnum"]) * int(key)) if bs and key and str(bs.get("bnum", "")).isdigit() else ""
        out = json.dumps({"ok": True, "code": code}).encode()
        self.send_response(200); self.send_header("Content-Type", "application/json"); self.send_header("Content-Length", str(len(out))); self.end_headers(); self.wfile.write(out)


if __name__ == "__main__":
    port = int(sys.argv[1]) if len(sys.argv) > 1 else 8000
    print(f"survey on http://0.0.0.0:{port}")
    ThreadingHTTPServer(("0.0.0.0", port), Handler).serve_forever()
