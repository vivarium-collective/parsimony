#!/usr/bin/env python3
"""No-cache static file server for the parsimony viewer.

Disables HTTP caching so edits to viewer.js / index.html and freshly
regenerated packs always show up on a normal reload — no hard-refresh
dance. Serves the current working directory (run it from the project
root so the viewer can fetch root-relative mesh URLs like
`/examples/pdb_meshes/x.obj`).

    python scripts/serve.py [PORT]

`parsimony viewer` and `view_pack.sh` both launch this.
"""
import sys
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer

PORT = int(sys.argv[1]) if len(sys.argv) > 1 else 8123


class NoCacheHandler(SimpleHTTPRequestHandler):
    def end_headers(self):
        self.send_header("Cache-Control", "no-store, no-cache, must-revalidate, max-age=0")
        self.send_header("Pragma", "no-cache")
        self.send_header("Expires", "0")
        super().end_headers()


if __name__ == "__main__":
    httpd = ThreadingHTTPServer(("", PORT), NoCacheHandler)
    print(f"serving (no-cache) on http://localhost:{PORT}/  (Ctrl-C to stop)")
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        pass
