"""Wrapper that runs eval.py N times with different seeds and aggregates.

Reports mean +/- std for each stochastic metric. Use this for any number
you'd actually quote in a paper or PR — single-seed numbers carry hidden
variance on the order of +/-0.1 nats for ELBO and +/-15% for sample-based
metrics at our sample budget.

Usage:
    python eval_multi.py --n-runs 3   # 3 seeds, ~5 min on 3090
"""
import argparse
import re
import subprocess
import sys
from pathlib import Path

import numpy as np


def parse_table(text: str):
    """Extract the eval.py main + schedule + blockwise tables into dicts.

    Returns:
        headline: dict[label -> (mdm_str, ar_str)]
        sched:    dict[schedule -> ppl]
        blocks:   dict[label -> ppl]
    """
    headline, sched, blocks = {}, {}, {}
    section = None
    for line in text.splitlines():
        line = line.strip()
        if line.startswith("| Metric | MDM | AR |"):
            section = "headline"
            continue
        if line.startswith("| Schedule | PPL under AR"):
            section = "sched"
            continue
        if line.startswith("| Block setup |"):
            section = "blocks"
            continue
        if not line.startswith("|") or "---" in line:
            continue
        parts = [p.strip() for p in line.strip("|").split("|")]
        if section == "headline" and len(parts) == 3:
            headline[parts[0]] = (parts[1], parts[2])
        elif section == "sched" and len(parts) == 3:
            sched[parts[0]] = parts[1].split()[0]
        elif section == "blocks" and len(parts) == 3:
            blocks[parts[0]] = parts[2]
    return headline, sched, blocks


def parse_float(s: str):
    """Pull the leading float out of a cell like '<= 1.617 (ELBO)' or '3.35'."""
    m = re.search(r"-?\d+\.\d+", s)
    return float(m.group()) if m else None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-runs", type=int, default=3)
    ap.add_argument("--n-samples", type=int, default=16)
    ap.add_argument("--steps", type=int, default=64)
    ap.add_argument("--temperature", type=float, default=1.0)
    ap.add_argument("--top-p", type=float, default=0.9)
    args = ap.parse_args()

    headline_all = {}    # label -> [(mdm_float, ar_float), ...]
    sched_all = {}       # schedule -> [ppl, ...]
    blocks_all = {}      # label -> [ppl, ...]

    here = Path(__file__).parent
    for seed in range(args.n_runs):
        print(f"\n========== run {seed + 1}/{args.n_runs} (seed={seed}) ==========")
        out = subprocess.run(
            [sys.executable, str(here / "eval.py"),
             "--seed", str(seed),
             "--n-samples", str(args.n_samples),
             "--steps", str(args.steps),
             "--temperature", str(args.temperature),
             "--top-p", str(args.top_p)],
            capture_output=True, text=True, encoding="utf-8",
        )
        if out.returncode != 0:
            print("eval.py failed:")
            print(out.stdout)
            print(out.stderr)
            sys.exit(1)
        # Echo the tail so the user sees progress
        print(out.stdout.splitlines()[-20:])

        headline, sched, blocks = parse_table(out.stdout)
        for k, (m, a) in headline.items():
            headline_all.setdefault(k, []).append((parse_float(m), parse_float(a)))
        for k, v in sched.items():
            sched_all.setdefault(k, []).append(parse_float(v))
        for k, v in blocks.items():
            blocks_all.setdefault(k, []).append(parse_float(v))

    def fmt(values):
        clean = [v for v in values if v is not None]
        if not clean:
            return "n/a"
        if len(clean) == 1:
            return f"{clean[0]:.3f}"
        return f"{np.mean(clean):.3f} +/- {np.std(clean):.3f}"

    print("\n" + "=" * 72)
    print(f"AGGREGATED ({args.n_runs} seeds)")
    print("=" * 72)
    print("\n| Metric | MDM | AR |")
    print("|---|---|---|")
    for label, runs in headline_all.items():
        m_vals = [r[0] for r in runs]
        a_vals = [r[1] for r in runs if r[1] is not None]
        print(f"| {label} | {fmt(m_vals)} | {fmt(a_vals)} |")

    print("\n| Schedule | PPL under AR (mean +/- std) |")
    print("|---|---|")
    for k, vs in sched_all.items():
        print(f"| {k} | {fmt(vs)} |")

    print("\n| Block setup | PPL under AR (mean +/- std) |")
    print("|---|---|")
    for k, vs in blocks_all.items():
        print(f"| {k} | {fmt(vs)} |")


if __name__ == "__main__":
    main()
