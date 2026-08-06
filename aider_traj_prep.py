#!/usr/bin/env python3
"""Convert Aider benchmark trajectories into ShareGPT-style SFT data.

Reusable corpus-refresh tool (part of the published Fruit toolchain): point it
at an Aider `tmp.benchmarks/` tree (the run dirs containing per-exercise
`.aider.chat.history.md` + `.aider.results.json`) and it emits one JSONL of
{"conversations": [{"from": "human"|"gpt", "value": ...}, ...], "meta": {...}}
rows, consumable by sft_data_prep.py (set AIDER_JSONL) or any ShareGPT loader.

History format parsed: lines starting '#### ' are user text; lines starting
'> ' are aider announcements (dropped); everything else is assistant output.
A file may contain several sessions ('# aider chat started at ...'); each
session becomes one conversation. Consecutive same-role blocks are merged.

Filtering: PASS_ONLY=1 (default) keeps only exercises whose results JSON shows
at least one passing test outcome — we distill behavior worth imitating.
MIN_CHARS drops trivial/empty sessions.

CONTAMINATION NOTE (disclose in any model card): these are Aider/Exercism
benchmark problems. A model trained on this data is permanently contaminated
for Aider-style evaluations.

Env: BENCH_DIR, OUT_JSONL, PASS_ONLY=1, MIN_CHARS=500.
"""
import json
import os
import re
from pathlib import Path

BENCH = Path(os.environ.get("BENCH_DIR",
                            "/mnt/vault/llm/fruit-pilot/aider-bench"))
OUT = Path(os.environ.get("OUT_JSONL",
                          "/mnt/vault/llm/fruit-pilot/sft-aider.jsonl"))
PASS_ONLY = os.environ.get("PASS_ONLY", "1") == "1"
MIN_CHARS = int(os.environ.get("MIN_CHARS", "500"))
# Skip run dirs whose files changed in the last N minutes: a benchmark still
# in progress has half-written histories/results (0 disables the guard).
QUIESCENT_MIN = float(os.environ.get("QUIESCENT_MIN", "30"))
# Giant single turns (megabytes of compiler/test spam with heavy n-gram
# repetition — a measured loss-spike risk) get middle-elided to head+tail.
MAX_TURN_CHARS = int(os.environ.get("MAX_TURN_CHARS", "32000"))
SESSION_RE = re.compile(r"^# aider chat started at ", re.M)


def active_runs(bench):
    import time
    skip = set()
    if QUIESCENT_MIN <= 0:
        return skip
    now = time.time()
    for run in (d for d in bench.iterdir() if d.is_dir()):
        newest = max((f.stat().st_mtime
                      for f in run.rglob(".aider.*")), default=0)
        if now - newest < QUIESCENT_MIN * 60:
            skip.add(run.name)
            print(f"[aider-prep] SKIP in-progress run: {run.name}",
                  flush=True)
    return skip


def parse_session(text):
    """One session's markdown -> list of {'from','value'} turns."""
    turns = []

    def add(role, buf):
        val = "\n".join(buf).strip()
        if not val:
            return
        if MAX_TURN_CHARS and len(val) > MAX_TURN_CHARS:
            half = MAX_TURN_CHARS // 2
            val = (val[:half] + f"\n... [{len(val) - MAX_TURN_CHARS} "
                   "chars elided] ...\n" + val[-half:])
        if turns and turns[-1]["from"] == role:
            turns[-1]["value"] += "\n" + val
        else:
            turns.append({"from": role, "value": val})

    role, buf = None, []
    for line in text.splitlines():
        line = line.rstrip()
        if line.startswith("> "):        # aider announcements: drop
            continue
        if line.startswith("#### "):
            new_role, content = "human", line[5:]
        elif line.startswith("####"):
            new_role, content = "human", line[4:]
        else:
            new_role, content = "gpt", line
        if new_role != role and buf:
            add(role, buf)
            buf = []
        role = new_role
        buf.append(content)
    if buf:
        add(role, buf)
    return turns


def main():
    n_files = n_conv = n_skip = chars = 0
    live = active_runs(BENCH)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    with open(OUT, "w") as out:
        for hist in sorted(BENCH.rglob(".aider.chat.history.md")):
            if hist.relative_to(BENCH).parts[0] in live:
                continue
            n_files += 1
            res = hist.parent / ".aider.results.json"
            meta = {}
            if res.exists():
                try:
                    r = json.loads(res.read_text())
                    meta = {"exercise": r.get("testcase"),
                            "edit_format": r.get("edit_format"),
                            "passed": any(r.get("tests_outcomes") or [])}
                except Exception:
                    pass
            if PASS_ONLY and not meta.get("passed"):
                n_skip += 1
                continue
            meta["run"] = hist.relative_to(BENCH).parts[0]
            for session in SESSION_RE.split(hist.read_text(errors="replace")):
                turns = parse_session(session)
                # must start with a user turn and contain a reply
                while turns and turns[0]["from"] != "human":
                    turns.pop(0)
                if len(turns) < 2 or \
                        sum(len(t["value"]) for t in turns) < MIN_CHARS:
                    continue
                if turns[-1]["from"] != "gpt":
                    turns.pop()
                if len(turns) < 2:
                    continue
                out.write(json.dumps({"conversations": turns,
                                      "meta": meta}) + "\n")
                n_conv += 1
                chars += sum(len(t["value"]) for t in turns)
    print(f"AIDER-PREP-DONE: {n_conv} conversations from {n_files} files "
          f"({n_skip} filtered non-passing), {chars/1e6:.1f}M chars "
          f"-> {OUT}", flush=True)


if __name__ == "__main__":
    main()
