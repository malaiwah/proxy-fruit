#!/usr/bin/env python3
"""Publish training progress to the HF checkpoint repo, resumption-proof.

1. Merges this node's metric lines (NODE_LINES file: `[val N] ...` and
   `[N/M] loss ...` lines) with the durable copy at
   `logs/train_metrics.log` in the HF repo — so history survives node
   changes (a fresh node's run.log starts empty; HF is the ledger).
2. Renders a 2x2 progress PNG: val curves; train/mtp loss (EMA); lr+gnorm;
   aux + s/step.
3. Uploads log + PNG + README.

Env: NODE_LINES, WORK (output dir), HF_TOKEN, REPO (default
malaiwah/fruit-phase1-ckpt), PLOT_TITLE.
"""
import os
import re
from datetime import datetime, timezone

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

REPO = os.environ.get("REPO", "malaiwah/fruit-phase1-ckpt")
WORK = os.environ.get("WORK", ".")
TITLE = os.environ.get("PLOT_TITLE", "GLM-5.2-SIQ-Fruit — Phase-1")

VAL_RE = re.compile(r"^\[val (\d+)\] (.*)")
STEP_RE = re.compile(
    r"^\[(\d+)/\d+\] loss ([\d.]+) mtp ([\d.]+) aux ([\d.]+)"
    r"(?: gnorm ([\d.]+))? lr ([\d.e+-]+) ([\d.]+)s/step")
INC_RE = re.compile(r"^\[incarnation\] step=(\d+) .*?gpu=(\S+) .*?"
                    r"time=(\S+)")
TELEM_RE = re.compile(r"^\[telem\] step=(\d+) time=(\S+) (.*)")
KV = re.compile(r"(\w+)=([\d.]+)")


def parse_key(line):
    m = VAL_RE.match(line)
    if m:
        return ("val", int(m.group(1)))
    m = STEP_RE.match(line)
    if m:
        return ("step", int(m.group(1)))
    m = INC_RE.match(line)
    if m:  # additive: unique per start event (keyed by timestamp)
        return ("inc", m.group(3))
    m = TELEM_RE.match(line)
    if m:
        return ("telem", m.group(2))
    return None


def ema(xs, a=0.15):
    out, m = [], None
    for x in xs:
        m = x if m is None else (1 - a) * m + a * x
        out.append(m)
    return out


def main():
    merged = {}
    try:
        from huggingface_hub import hf_hub_download
        prev = hf_hub_download(REPO, "logs/train_metrics.log",
                               token=os.environ.get("HF_TOKEN"))
        for line in open(prev):
            k = parse_key(line)
            if k:
                merged[k] = line.rstrip()
        print(f"[merge] {len(merged)} lines from HF ledger", flush=True)
    except Exception as exc:
        print(f"[merge] no prior ledger ({type(exc).__name__})", flush=True)
    n_node = 0
    for line in open(os.environ["NODE_LINES"]):
        k = parse_key(line.strip())
        if k:
            merged[k] = line.strip()
            n_node += 1
    print(f"[merge] +{n_node} node lines -> {len(merged)} total", flush=True)
    lines = [merged[k] for k in
             sorted(merged, key=lambda k: (str(k[1]), k[0]))]
    lines.sort(key=lambda ln: (int((VAL_RE.match(ln) or STEP_RE.match(ln)
                                    or INC_RE.match(ln)).group(1))))
    logp = os.path.join(WORK, "train_metrics.log")
    open(logp, "w").write("\n".join(lines) + "\n")

    incs = []
    telem = {}
    vals, steps = {}, {"step": [], "loss": [], "mtp": [], "aux": [],
                       "gnorm": [], "lr": [], "sps": []}
    for line in lines:
        m = INC_RE.match(line)
        if m:
            incs.append((int(m.group(1)), m.group(2)))
            continue
        m = TELEM_RE.match(line)
        if m:
            from datetime import datetime as _dt
            try:
                ts = _dt.fromisoformat(m.group(2).replace("Z", "+00:00"))
            except ValueError:
                continue
            for k, v in KV.findall(m.group(3)):
                telem.setdefault(k, []).append((ts, float(v)))
            continue
        m = VAL_RE.match(line)
        if m:
            for k, v in KV.findall(m.group(2)):
                vals.setdefault(k, []).append((int(m.group(1)), float(v)))
            continue
        m = STEP_RE.match(line)
        if m:
            steps["step"].append(int(m.group(1)))
            steps["loss"].append(float(m.group(2)))
            steps["mtp"].append(float(m.group(3)))
            steps["aux"].append(float(m.group(4)))
            steps["gnorm"].append(float(m.group(5)) if m.group(5) else None)
            steps["lr"].append(float(m.group(6)))
            steps["sps"].append(float(m.group(7)))

    has_throttle = telem and telem.get("gpuclk")
    nrows = (4 if has_throttle else 3) if telem else 2
    fig, ax = plt.subplots(nrows, 2, figsize=(13, 4.25 * nrows), dpi=120)
    a = ax[0][0]
    for k in sorted(vals):
        if k == "global":
            continue
        xs, ys = zip(*sorted(set(vals[k])))
        a.plot(xs, ys, lw=1.0, alpha=0.7, label=k)
    if vals.get("global"):
        xs, ys = zip(*sorted(set(vals["global"])))
        a.plot(xs, ys, lw=2.6, color="black", label="global")
        a.set_title(f"val loss (global {ys[-1]:.3f} @ step {xs[-1]:,})")
    a.legend(fontsize=7, ncol=2)
    a.grid(alpha=0.3)

    s = steps
    a = ax[0][1]
    a.plot(s["step"], s["loss"], lw=0.4, alpha=0.3, color="tab:blue")
    a.plot(s["step"], ema(s["loss"]), lw=1.8, color="tab:blue",
           label="train loss (EMA)")
    a.plot(s["step"], ema(s["mtp"]), lw=1.2, color="tab:orange",
           label="mtp loss (EMA)")
    a.set_title("train loss")
    a.legend(fontsize=8)
    a.grid(alpha=0.3)

    a = ax[1][0]
    a.plot(s["step"], s["lr"], lw=1.6, color="tab:green", label="lr")
    a.set_yscale("log")
    a.set_ylabel("lr")
    g = [(x, y) for x, y in zip(s["step"], s["gnorm"]) if y is not None]
    if g:
        a2 = a.twinx()
        gx, gy = zip(*g)
        a2.plot(gx, gy, lw=0.5, alpha=0.6, color="tab:red", label="gnorm")
        a2.set_ylabel("gnorm")
        a2.set_ylim(0, 3)
    a.set_title("lr + gnorm")
    a.grid(alpha=0.3)

    a = ax[1][1]
    a.plot(s["step"], ema(s["aux"]), lw=1.4, color="tab:purple",
           label="aux (EMA)")
    a.set_ylabel("aux")
    a.set_yscale("log")
    a2 = a.twinx()
    a2.plot(s["step"], s["sps"], lw=0.8, alpha=0.7, color="tab:gray")
    a2.set_ylabel("s/step")
    a.set_title("router aux + step time")
    a.grid(alpha=0.3)

    if telem:
        def tl(key):
            pts = sorted(telem.get(key, []))
            return ([p[0] for p in pts], [p[1] for p in pts])
        a = ax[2][0]
        x, y = tl("gpuutil")
        a.plot(x, y, lw=1.0, color="tab:green", label="GPU util %")
        a.set_ylim(0, 105)
        a.set_ylabel("GPU util %")
        a2 = a.twinx()
        x, y = tl("vram")
        a2.plot(x, y, lw=1.0, color="tab:blue", alpha=0.7, label="VRAM GB")
        a2.set_ylabel("VRAM GB (total)")
        a.set_title("GPU utilization + VRAM")
        a.grid(alpha=0.3)
        a = ax[2][1]
        x, y = tl("cpu")
        a.plot(x, y, lw=0.9, color="tab:orange", label="CPU %")
        x, y = tl("mem")
        a.plot(x, y, lw=0.9, color="tab:purple", label="RAM GB")
        a.set_ylabel("CPU % / RAM GB")
        a2 = a.twinx()
        for k, c in (("rx", "tab:cyan"), ("tx", "tab:pink")):
            x, y = tl(k)
            a2.plot(x, y, lw=0.7, alpha=0.7, color=c, label=f"net {k} MB/s")
        a2.set_ylabel("net MB/s")
        a.legend(fontsize=7, loc="upper left")
        a2.legend(fontsize=7, loc="upper right")
        a.set_title("host CPU / RAM / network")
        a.grid(alpha=0.3)
    if has_throttle:
        a = ax[3][0]
        for k, c in (("gpuclk", "tab:green"), ("memclk", "tab:olive"),
                     ("cpumhz", "tab:gray")):
            x, y = tl(k)
            if x:
                a.plot(x, y, lw=0.9, color=c, label=f"{k} MHz")
        a.set_ylabel("MHz")
        a2 = a.twinx()
        for k, c in (("gputemp", "tab:red"), ("cputemp", "tab:orange")):
            x, y = tl(k)
            if x and max(y) > 0:
                a2.plot(x, y, lw=0.9, ls=":", color=c, label=f"{k} °C")
        a2.set_ylabel("°C")
        a.legend(fontsize=7, loc="upper left")
        a2.legend(fontsize=7, loc="upper right")
        a.set_title("clocks + temps (throttle forensics)")
        a.grid(alpha=0.3)
        a = ax[3][1]
        x, y = tl("pdraw")
        a.plot(x, y, lw=1.1, color="tab:red", label="power draw W (sum)")
        x, y = tl("plim")
        if x:
            a.plot(x, y, lw=1.0, ls="--", color="black",
                   label="power limit W (sum)")
        a.set_ylabel("W")
        a.legend(fontsize=7)
        a.set_title("power draw vs limit")
        a.grid(alpha=0.3)
    for i, row in enumerate(ax):
        for a in row:
            if i < 2:
                a.set_xlabel("step")
                for xstep, gpu in incs:
                    a.axvline(xstep, color="crimson", lw=0.8, ls="--",
                              alpha=0.6)
            else:
                a.set_xlabel("time (UTC)")
                a.tick_params(axis="x", labelsize=6, rotation=30)
    if incs:  # annotate shapes on the val panel only (avoid clutter)
        for xstep, gpu in incs:
            ax[0][0].annotate(gpu, (xstep, ax[0][0].get_ylim()[1]),
                              rotation=90, fontsize=6, va="top",
                              color="crimson", alpha=0.8)
    fig.suptitle(f"{TITLE} · updated "
                 f"{datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M')}Z")
    fig.tight_layout()
    png = os.path.join(WORK, "val_progress.png")
    fig.savefig(png)

    from huggingface_hub import HfApi
    api = HfApi(token=os.environ["HF_TOKEN"])
    for f, dest in ((png, "val_progress.png"),
                    (logp, "logs/train_metrics.log"),
                    (os.path.join(WORK, "README.md"), "README.md")):
        if os.path.exists(f):
            api.upload_file(path_or_fileobj=f, path_in_repo=dest,
                            repo_id=REPO,
                            commit_message="progress update")
    print(f"PROGRESS-PUBLISHED ({len(lines)} ledger lines)", flush=True)


if __name__ == "__main__":
    main()
