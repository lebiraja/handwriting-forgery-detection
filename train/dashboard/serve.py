"""Live training dashboard. Run: python3 serve.py  ->  http://localhost:8765  (stdlib only, read-only)."""
import json, re, subprocess, time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from statistics import median

TRAIN = Path(__file__).resolve().parent.parent
RUNS = TRAIN / "runs"
QUEUE = TRAIN / "queue_resume.sh"
DEFAULT_EPOCHS = 40
SEC_FALLBACK = 25.0


def queue_runs():
    out = []
    for line in QUEUE.read_text().splitlines():
        m = re.match(r"\./run_train\.sh\s+(\S+)", line.strip())
        if m:
            out.append(m.group(1))
    return out


def read_run(name):
    log = RUNS / f"{name}.log"
    info = {"name": name, "epoch": 0, "epochs": DEFAULT_EPOCHS, "sec": None, "eer": None,
            "best": None, "state": "queued", "loss": None}
    if not log.exists():
        return info
    secs, evals, hist, vhist = [], [], [], []
    for line in log.read_text(errors="ignore").splitlines():
        line = line.strip()
        if line.startswith("{") and '"epoch"' in line:
            try:
                d = json.loads(line)
            except ValueError:
                continue
            info["epoch"] = d["epoch"] + 1 if d["epoch"] < info["epochs"] else d["epoch"]
            info["loss"] = d.get("loss")
            secs.append(d.get("sec", 0))
            hist.append([d["epoch"] + 1, d.get("loss"), d.get("sec")])
            if "val" in d:
                vhist.append([d["epoch"] + 1, d["val"].get("mean_eer"), d["val"].get("iam", {}).get("eer"),
                              d["val"].get("cvl", {}).get("eer")])
            if "val" in d:
                evals.append(d["val"].get("mean_eer"))
                info["eer"] = d["val"]
        elif line.startswith("exit "):
            info["exit"] = line
    info["hist"], info["vhist"] = hist, vhist
    evals = [e for e in evals if e is not None]
    info["best"] = min(evals) if evals else None
    info["sec"] = sum(secs[-20:]) / len(secs[-20:]) if secs else None
    info["avg_sec"] = sum(secs) / len(secs) if secs else None
    mtime = log.stat().st_mtime
    if "exit" in info:
        info["state"] = "done" if info["exit"].endswith("0") else "failed"
        info["epoch"] = max(info["epoch"], info["epochs"]) if info["state"] == "done" else info["epoch"]
    elif time.time() - mtime < 300:
        info["state"] = "running"
    else:
        info["state"] = "stalled" if secs else "queued"
    return info


def gpu():
    try:
        o = subprocess.run(["nvidia-smi", "--query-gpu=temperature.gpu,utilization.gpu,memory.used,memory.total",
                            "--format=csv,noheader,nounits"], capture_output=True, text=True, timeout=5).stdout
        t, u, mu, mt = [x.strip() for x in o.split(",")]
        return {"temp": int(t), "util": int(u), "mem_used": int(mu), "mem_total": int(mt)}
    except Exception:
        return None


def status():
    names = queue_runs()
    runs = [read_run(n) for n in names]
    done_secs = [r["avg_sec"] for r in runs if r["state"] == "done" and r["avg_sec"]]
    # distillation-off runs are faster; use median of finished runs as the estimate for queued ones
    est = median(done_secs) if done_secs else SEC_FALLBACK
    overhead = 45  # container start + validation per run, seconds (rough)
    remaining = 0.0
    for r in runs:
        if r["state"] == "running" or r["state"] == "stalled":
            remaining += (r["epochs"] - r["epoch"]) * (r["sec"] or est)
        elif r["state"] == "queued":
            remaining += r["epochs"] * est + overhead
        r["eta_sec"] = ((r["epochs"] - r["epoch"]) * (r["sec"] or est)
                        if r["state"] in ("running", "stalled") else None)
    total_epochs = sum(r["epochs"] for r in runs)
    done_epochs = sum(min(r["epoch"], r["epochs"]) for r in runs)
    cur = next((r for r in runs if r["state"] in ("running", "stalled")), None)
    anchor = (RUNS / f"{cur['name']}.log").stat().st_mtime if cur else time.time()
    for r in runs:
        r["eta_at"] = anchor + r["eta_sec"] if r.get("eta_sec") is not None else None
    return {"now": time.time(), "runs": runs, "remaining_sec": remaining,
            "finish_at": anchor + remaining, "progress": done_epochs / total_epochs if total_epochs else 0,
            "queue_done": (RUNS / "queue.done").exists(), "gpu": gpu(), "est_sec_epoch": est}


class H(BaseHTTPRequestHandler):
    def do_GET(self):
        if self.path.startswith("/status"):
            body, ctype = json.dumps(status()).encode(), "application/json"
        elif self.path.startswith("/stream"):
            self.send_response(200)
            self.send_header("Content-Type", "text/event-stream")
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            try:
                while True:
                    self.wfile.write(b"data: " + json.dumps(status()).encode() + b"\n\n")
                    self.wfile.flush()
                    time.sleep(1)
            except (BrokenPipeError, ConnectionResetError):
                return
        else:
            body, ctype = (Path(__file__).parent / "index.html").read_bytes(), "text/html; charset=utf-8"
        if self.path.startswith("/stream"):
            return
        self.send_response(200)
        self.send_header("Content-Type", ctype)
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, *a):
        pass


if __name__ == "__main__":
    print("http://localhost:8765")
    ThreadingHTTPServer(("127.0.0.1", 8765), H).serve_forever()
