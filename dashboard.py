"""dashboard.py — live dashboard for learning_log_*.csv files.

Run from the project root:   python dashboard.py [port]
Then open                    http://localhost:8000
"""
import glob
import http.server
import os
import socketserver
import sys

PORT = int(sys.argv[1]) if len(sys.argv) > 1 else 8000
ROOT = os.path.dirname(os.path.abspath(__file__))

PAGE = """<!doctype html><html><head><meta charset="utf-8">
<title>Sorting Robots - Live Dashboard</title><style>
 body{background:#111;color:#ddd;font-family:monospace;margin:20px}
 h1{font-size:18px} canvas{background:#181818;border:1px solid #333}
 #stats div{margin:6px 0;font-size:14px}
 .b{color:#7cf}.g{color:#7f7}.r{color:#f77}
</style></head><body>
<h1>Sorting robots - live training</h1><div id="stats"></div>
<canvas id="c" width="900" height="360"></canvas>
<p>refreshes every 5 s &mdash; thin: raw return &nbsp; thick: 10-episode average</p>
<script>
const csvs = __FILES__;
async function load(){
  const all = [];
  for (const f of csvs){
    try{
      const txt = await (await fetch(f + "?t=" + Date.now())).text();
      const rows = txt.trim().split("\\n").slice(1).map(l => l.split(","));
      if (rows.length && rows[0].length >= 5) all.push({name: f.split("/").pop(), rows});
    }catch(e){}
  }
  draw(all);
}
function draw(all){
  const st = document.getElementById("stats"); st.innerHTML = "";
  const cv = document.getElementById("c"), ctx = cv.getContext("2d");
  ctx.clearRect(0, 0, cv.width, cv.height);
  if (!all.length){ st.textContent = "no logs yet - start a training run"; return; }
  let mx = 10;
  all.forEach(s => s.rows.forEach(r => mx = Math.max(mx, parseFloat(r[2]) + 5)));
  const nMax = Math.max(...all.map(s => s.rows.length));
  const X = i => 40 + i / Math.max(1, nMax) * (cv.width - 60);
  const Y = v => cv.height - 30 - (v + 10) / (mx + 10) * (cv.height - 50);
  ctx.strokeStyle = "#333"; ctx.beginPath();
  ctx.moveTo(40, Y(0)); ctx.lineTo(cv.width - 20, Y(0)); ctx.stroke();
  all.forEach((s, k) => {
    const ret = s.rows.map(r => parseFloat(r[2]));
    const col = "hsl(" + (k * 137 % 360) + ",80%,65%)";
    ctx.strokeStyle = col; ctx.fillStyle = col; ctx.lineWidth = 1;
    ctx.beginPath();
    ret.forEach((v, i) => i ? ctx.lineTo(X(i), Y(v)) : ctx.moveTo(X(0), Y(v)));
    ctx.stroke();
    ctx.lineWidth = 2.5; ctx.beginPath();
    ret.forEach((v, i) => {
      if (i >= 9){
        const m = ret.slice(i - 9, i + 1).reduce((a, b) => a + b) / 10;
        i == 9 ? ctx.moveTo(X(i), Y(m)) : ctx.lineTo(X(i), Y(m));
      }
    });
    ctx.stroke(); ctx.lineWidth = 1;
    const last = s.rows[s.rows.length - 1];
    const d = document.createElement("div");
    d.innerHTML = "<span class='b'>" + s.name + "</span> episode " + last[1] +
      " | return <span class='g'>" + last[2] + "</span> | eps " + last[5];
    st.appendChild(d);
  });
}
setInterval(load, 5000); load();
</script></body></html>"""


def main():
    files = sorted(glob.glob(os.path.join(ROOT, "controllers", "*",
                                          "learning_log_*.csv")))
    rel = [os.path.relpath(f, ROOT).replace(os.sep, "/") for f in files]
    html = PAGE.replace("__FILES__", repr(rel))

    class Handler(http.server.SimpleHTTPRequestHandler):
        def __init__(self, *a, **kw):
            super().__init__(*a, directory=ROOT, **kw)

        def do_GET(self):
            if self.path in ("/", "/index.html"):
                body = html.encode()
                self.send_response(200)
                self.send_header("Content-Type", "text/html")
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)
            else:
                super().do_GET()

    with socketserver.TCPServer(("", PORT), Handler) as httpd:
        print("dashboard: http://localhost:%d   (Ctrl+C to stop)" % PORT)
        print("watching:", rel or "(no logs yet)")
        httpd.serve_forever()


if __name__ == "__main__":
    main()
