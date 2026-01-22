import json
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

LOG_FILE = Path("metrics/latency_log.jsonl")
OUT_FILE = Path("metrics/comparison.png")

def load_jsonl(path: Path):
    rows = []
    if not path.exists():
        return rows
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows

def pct(a, p):
    return float(np.percentile(np.array(a, dtype=float), p))

def ms_label(ms):
    ms = float(ms)
    return f"{ms/1000:.2f}s" if ms >= 1000 else f"{ms:.0f}ms"

if __name__ == "__main__":
    rows = load_jsonl(LOG_FILE)
    if not rows:
        raise SystemExit("No logs found. Hit /insights a few times first.")

    retrieval = [r["retrieval_ms"] for r in rows if "retrieval_ms" in r]
    e2e = [r["e2e_ms"] for r in rows if "e2e_ms" in r]

    n = min(len(retrieval), len(e2e))
    retrieval = retrieval[:n]
    e2e = e2e[:n]

    r_p50, r_p95 = pct(retrieval, 50), pct(retrieval, 95)
    e_p50, e_p95 = pct(e2e, 50), pct(e2e, 95)

    plt.figure(figsize=(10, 6))

    # Top: p50/p95 bars
    ax1 = plt.subplot(2, 1, 1)
    labels = ["Retrieval", "End-to-End"]
    p50_vals = [r_p50, e_p50]
    p95_vals = [r_p95, e_p95]
    x = np.arange(len(labels))
    w = 0.35

    ax1.bar(x - w/2, p50_vals, w, label="p50")
    ax1.bar(x + w/2, p95_vals, w, label="p95")
    ax1.set_ylabel("Latency (ms)")
    ax1.set_title(f"Latency Summary (n={n})")
    ax1.set_xticks(x)
    ax1.set_xticklabels(labels)
    ax1.legend()

    for i, v in enumerate(p50_vals):
        ax1.text(i - w/2, v, ms_label(v), ha="center", va="bottom", fontsize=9)
    for i, v in enumerate(p95_vals):
        ax1.text(i + w/2, v, ms_label(v), ha="center", va="bottom", fontsize=9)

    ax2 = plt.subplot(2, 1, 2)
    idx = np.arange(1, n + 1)
    ax2.plot(idx, retrieval, marker="o", linewidth=1, label="Retrieval (ms)")
    ax2.plot(idx, e2e, marker="o", linewidth=1, label="End-to-End (ms)")
    ax2.set_xlabel("Request #")
    ax2.set_ylabel("Latency (ms)")
    ax2.set_title("Latency Across Requests")
    ax2.legend()

    plt.tight_layout()
    OUT_FILE.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(OUT_FILE, dpi=200)
    plt.close()

    print(f"Saved: {OUT_FILE.resolve()}")