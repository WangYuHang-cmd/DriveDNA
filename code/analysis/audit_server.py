#!/usr/bin/env python3
"""Serve the maneuver-audit pages and save answers to disk in real time.
    python audit_server.py   ->  http://127.0.0.1:8765/index.html
POST /save {page, answers[]} -> results/maneuver_audit/answers/<page>.json (atomic)."""
import json, os
from http.server import HTTPServer, SimpleHTTPRequestHandler

ROOT = "./results/maneuver_audit"
os.makedirs(f"{ROOT}/answers", exist_ok=True)

class H(SimpleHTTPRequestHandler):
    def __init__(self, *a, **k):
        super().__init__(*a, directory=ROOT, **k)
    def end_headers(self):
        self.send_header("Cache-Control", "no-store, must-revalidate")
        super().end_headers()
    def handle_one_request(self):
        try:
            super().handle_one_request()
        except (BrokenPipeError, ConnectionResetError):
            pass                      # browser aborted a video request; harmless
    def handle(self):
        try:
            super().handle()
        except (BrokenPipeError, ConnectionResetError):
            pass
    def do_POST(self):
        if self.path != "/save":
            self.send_error(404); return
        body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
        page = "".join(c for c in body["page"] if c.isalnum() or c in "_-")
        tmp = f"{ROOT}/answers/{page}.json.tmp"
        json.dump(body["answers"], open(tmp, "w"), indent=1)
        os.replace(tmp, f"{ROOT}/answers/{page}.json")
        self.send_response(200); self.end_headers(); self.wfile.write(b"ok")
    def log_message(self, *a):
        pass

print("serving http://127.0.0.1:8765/index.html  (answers -> answers/)")
HTTPServer(("127.0.0.1", 8765), H).serve_forever()
