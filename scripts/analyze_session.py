#!/usr/bin/env python3
"""Summarise one or many mcbio session directories (experiment analysis).

Usage::

    python scripts/analyze_session.py sessions/                 # every session under sessions/
    python scripts/analyze_session.py sessions/S1_* sessions/S2_* --out analysis/
    python scripts/analyze_session.py sessions/ --no-plots

A *session directory* contains ``session.json``, ``frames.csv``, ``events.csv`` and
(for ``--protocol`` runs) ``cues.csv``. Sessions are grouped by ``subject``.

Outputs (in ``--out``, default ``<first input>/analysis``):

* ``summary_subjects.csv``  per subject: trials, accuracy, reaction time median / mean / SD
  (correct action cues), false activations per minute of REST, actions/min, latency p50/p95, fps.
* ``summary_actions.csv``   per cued action across subjects: mean ± SD of the per-subject mean RT,
  accuracy mean ± SD, n subjects.
* ``subject_action.csv``    the long table (subject x action) the above is built from.
* ``rt_by_action.png``      bar chart: mean reaction time per action, SD error bars across subjects.
* ``latency_hist.png``      capture->output latency histogram with p50 / p95 markers.

Only the standard library, numpy and (for plots) matplotlib are needed.
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable

import numpy as np

ACTION_ORDER = ["forward", "back", "left", "right", "jump", "attack", "interact",
                "look_left", "look_right"]


# ---------------------------------------------------------------------------- loading
def find_sessions(paths: Iterable[str | Path]) -> list[Path]:
    """Expand inputs: a session dir itself, or a folder containing session dirs (recursive)."""
    out: list[Path] = []
    for p in map(Path, paths):
        if (p / "session.json").exists():
            out.append(p)
        elif p.is_dir():
            out.extend(sorted(q.parent for q in p.rglob("session.json")))
        else:
            print(f"warning: {p} is not a session directory", file=sys.stderr)
    seen: set[Path] = set()
    uniq = []
    for p in out:
        r = p.resolve()
        if r not in seen:
            seen.add(r)
            uniq.append(p)
    return uniq


def _read_csv(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        return []
    with path.open(newline="") as f:
        return list(csv.DictReader(f))


def _f(v: str | None) -> float | None:
    try:
        return float(v) if v not in (None, "") else None
    except ValueError:
        return None


def load_session(d: Path) -> dict[str, Any]:
    """Read one session directory into plain Python structures."""
    meta = json.loads((d / "session.json").read_text())
    frames = _read_csv(d / "frames.csv")
    events = _read_csv(d / "events.csv")
    cues = _read_csv(d / "cues.csv")
    t = np.array([_f(r["t"]) or 0.0 for r in frames]) if frames else np.zeros(0)
    lat = np.array([_f(r["latency_ms"]) or 0.0 for r in frames]) if frames else np.zeros(0)
    paused = np.array([r.get("paused") == "1" for r in frames]) if frames else np.zeros(0, bool)
    return {"dir": d, "meta": meta, "subject": str(meta.get("subject", d.name)),
            "t": t, "latency": lat, "paused": paused, "events": events, "cues": cues}


# ---------------------------------------------------------------------------- stats
def _mean_sd(x: list[float]) -> tuple[float, float]:
    if not x:
        return math.nan, math.nan
    a = np.asarray(x, float)
    return float(a.mean()), float(a.std(ddof=1)) if len(a) > 1 else 0.0


def summarize(sessions: list[dict[str, Any]]) -> tuple[list[dict], list[dict], list[dict], np.ndarray]:
    """Compute (subject rows, action rows, subject x action rows, pooled latencies)."""
    by_subj: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for s in sessions:
        by_subj[s["subject"]].append(s)

    subj_rows: list[dict] = []
    long_rows: list[dict] = []
    all_lat: list[np.ndarray] = []
    for subj, ss in sorted(by_subj.items()):
        lat = np.concatenate([s["latency"] for s in ss]) if ss else np.zeros(0)
        all_lat.append(lat)
        n_frames = sum(len(s["t"]) for s in ss)
        dur = sum(float(s["t"][-1] - s["t"][0]) for s in ss if len(s["t"]) > 1)
        live_min = sum(float(np.sum(~s["paused"][1:] * np.diff(s["t"]))) for s in ss if len(s["t"]) > 1) / 60.0
        onsets = sum(1 for s in ss for e in s["events"]
                     if e["state"] == "on" and e["action"] != "paused")
        cues = [c for s in ss for c in s["cues"]]
        action_cues = [c for c in cues if c["expected"]]
        rest_cues = [c for c in cues if not c["expected"]]
        rts = [_f(c["rt_ms"]) for c in action_cues if c["correct"] == "1" and _f(c["rt_ms"]) is not None]
        rest_min = sum(_f(c["window_s"]) or 0.0 for c in rest_cues) / 60.0
        false_n = sum(int(_f(c["false_activations"]) or 0) for c in rest_cues)
        m, sd = _mean_sd(rts)  # type: ignore[arg-type]
        subj_rows.append({
            "subject": subj, "sessions": len(ss), "frames": n_frames,
            "cues": len(cues), "action_cues": len(action_cues),
            "accuracy_pct": 100.0 * sum(c["correct"] == "1" for c in action_cues) / len(action_cues)
            if action_cues else math.nan,
            "rt_median_ms": float(np.median(rts)) if rts else math.nan,
            "rt_mean_ms": m, "rt_sd_ms": sd,
            "rest_min": rest_min, "rest_false_activations": false_n,
            "false_per_min_rest": false_n / rest_min if rest_min > 0 else math.nan,
            "actions_per_min": onsets / live_min if live_min > 0 else math.nan,
            "latency_p50_ms": float(np.percentile(lat, 50)) if len(lat) else math.nan,
            "latency_p95_ms": float(np.percentile(lat, 95)) if len(lat) else math.nan,
            "fps": (n_frames - len(ss)) / dur if dur > 0 else math.nan,
        })
        # per action for this subject
        acts = defaultdict(list)
        for c in action_cues:
            acts[c["expected"]].append(c)
        for a, cs in acts.items():
            r = [_f(c["rt_ms"]) for c in cs if c["correct"] == "1" and _f(c["rt_ms"]) is not None]
            am, asd = _mean_sd(r)  # type: ignore[arg-type]
            long_rows.append({"subject": subj, "action": a, "n": len(cs),
                              "accuracy_pct": 100.0 * sum(c["correct"] == "1" for c in cs) / len(cs),
                              "rt_median_ms": float(np.median(r)) if r else math.nan,
                              "rt_mean_ms": am, "rt_sd_ms": asd})

    # across subjects, per action: mean ± SD of the per-subject means
    action_rows: list[dict] = []
    actions = sorted({r["action"] for r in long_rows},
                     key=lambda a: (ACTION_ORDER.index(a) if a in ACTION_ORDER else 99, a))
    for a in actions:
        rows = [r for r in long_rows if r["action"] == a]
        rt_means = [r["rt_mean_ms"] for r in rows if not math.isnan(r["rt_mean_ms"])]
        m, sd = _mean_sd(rt_means)
        am, asd = _mean_sd([r["accuracy_pct"] for r in rows])
        action_rows.append({"action": a, "subjects": len(rows), "trials": sum(r["n"] for r in rows),
                            "rt_mean_ms": m, "rt_sd_ms": sd,
                            "rt_median_ms": float(np.median(rt_means)) if rt_means else math.nan,
                            "accuracy_mean_pct": am, "accuracy_sd_pct": asd})
    pooled = np.concatenate(all_lat) if all_lat else np.zeros(0)
    return subj_rows, action_rows, long_rows, pooled


# ---------------------------------------------------------------------------- output
def _fmt(v: Any) -> str:
    if isinstance(v, float):
        return "-" if math.isnan(v) else f"{v:.1f}"
    return str(v)


def print_table(rows: list[dict], cols: list[str], title: str) -> None:
    """Fixed-width console table."""
    print(f"\n{title}")
    if not rows:
        print("  (no data)")
        return
    w = {c: max(len(c), *(len(_fmt(r.get(c, ""))) for r in rows)) for c in cols}
    print("  " + "  ".join(c.rjust(w[c]) for c in cols))
    for r in rows:
        print("  " + "  ".join(_fmt(r.get(c, "")).rjust(w[c]) for c in cols))


def write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        return
    with path.open("w", newline="") as f:
        wr = csv.DictWriter(f, fieldnames=list(rows[0]))
        wr.writeheader()
        for r in rows:
            wr.writerow({k: (f"{v:.3f}" if isinstance(v, float) and not math.isnan(v) else
                             ("" if isinstance(v, float) else v)) for k, v in r.items()})


# Plot styling: one series, one hue; recessive grid and axes; text in ink colours.
SERIES = "#2a78d6"
SURFACE = "#fcfcfb"
INK = "#0b0b0b"
INK2 = "#52514e"
GRID = "#e4e3df"


def _style(ax: Any) -> None:
    ax.set_facecolor(SURFACE)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(INK2)
    ax.tick_params(colors=INK2)
    ax.yaxis.grid(True, color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)


def make_plots(action_rows: list[dict], latencies: np.ndarray, out: Path, n_subjects: int) -> list[Path]:
    """Reaction-time bar chart (± SD across subjects) and latency histogram."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    written: list[Path] = []
    rows = [r for r in action_rows if not math.isnan(r["rt_mean_ms"])]
    if rows:
        fig, ax = plt.subplots(figsize=(9, 4.8), facecolor=SURFACE)
        _style(ax)
        x = np.arange(len(rows))
        means = [r["rt_mean_ms"] for r in rows]
        sds = [0.0 if math.isnan(r["rt_sd_ms"]) else r["rt_sd_ms"] for r in rows]
        ax.bar(x, means, width=0.6, color=SERIES, edgecolor=SURFACE, linewidth=2,
               yerr=sds, capsize=4, error_kw={"ecolor": INK2, "elinewidth": 1.2})
        for xi, m, sd in zip(x, means, sds):
            ax.text(xi, m + sd + 15, f"{m:.0f}", ha="center", va="bottom", color=INK, fontsize=9)
        ax.set_xticks(x, [r["action"].replace("_", " ") for r in rows], rotation=20, color=INK)
        ax.set_ylabel("reaction time (ms)", color=INK)
        ax.set_title(f"Cue -> action reaction time (mean ± SD across {n_subjects} subject"
                     f"{'s' if n_subjects != 1 else ''})", color=INK, loc="left")
        fig.tight_layout()
        p = out / "rt_by_action.png"
        fig.savefig(p, dpi=150, facecolor=SURFACE)
        plt.close(fig)
        written.append(p)
    if len(latencies):
        fig, ax = plt.subplots(figsize=(8, 4.2), facecolor=SURFACE)
        _style(ax)
        hi = float(np.percentile(latencies, 99.5))
        ax.hist(np.clip(latencies, 0, hi), bins=50, color=SERIES, edgecolor=SURFACE, linewidth=1)
        for q, ls in ((50, "-"), (95, "--")):
            v = float(np.percentile(latencies, q))
            ax.axvline(v, color=INK, linestyle=ls, linewidth=1.2)
            ax.text(v, ax.get_ylim()[1] * 0.95, f" p{q} {v:.1f} ms", color=INK, fontsize=9, va="top")
        ax.set_xlabel("capture -> output latency (ms)", color=INK)
        ax.set_ylabel("frames", color=INK)
        ax.set_title(f"Per-frame latency ({len(latencies)} frames)", color=INK, loc="left")
        fig.tight_layout()
        p = out / "latency_hist.png"
        fig.savefig(p, dpi=150, facecolor=SURFACE)
        plt.close(fig)
        written.append(p)
    return written


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("paths", nargs="+", help="session dirs, or folders containing them")
    ap.add_argument("--out", default=None, help="output folder (default: <first path>/analysis)")
    ap.add_argument("--no-plots", action="store_true", help="skip matplotlib plots")
    a = ap.parse_args(argv)

    dirs = find_sessions(a.paths)
    if not dirs:
        print("error: no session directories found (looking for session.json)", file=sys.stderr)
        return 1
    sessions = [load_session(d) for d in dirs]
    subj, actions, long_rows, lat = summarize(sessions)
    first = Path(a.paths[0])
    if a.out:
        out = Path(a.out)
    else:   # a single session dir -> its parent/analysis; a folder of sessions -> folder/analysis
        out = (first.parent if (first / "session.json").exists() else first) / "analysis"
    out.mkdir(parents=True, exist_ok=True)

    print(f"{len(sessions)} session(s), {len(subj)} subject(s)")
    print_table(subj, ["subject", "cues", "accuracy_pct", "rt_median_ms", "rt_mean_ms", "rt_sd_ms",
                       "false_per_min_rest", "actions_per_min", "latency_p50_ms", "latency_p95_ms", "fps"],
                "Per subject")
    print_table(actions, ["action", "subjects", "trials", "rt_mean_ms", "rt_sd_ms",
                          "accuracy_mean_pct", "accuracy_sd_pct"],
                "Per action (mean ± SD of subject means)")
    if len(subj) > 1:
        accs = [r["accuracy_pct"] for r in subj if not math.isnan(r["accuracy_pct"])]
        rts = [r["rt_mean_ms"] for r in subj if not math.isnan(r["rt_mean_ms"])]
        fam = [r["false_per_min_rest"] for r in subj if not math.isnan(r["false_per_min_rest"])]
        print("\nGroup: accuracy {:.1f} ± {:.1f} %   RT {:.0f} ± {:.0f} ms   false/min(rest) {:.2f} ± {:.2f}"
              .format(*_mean_sd(accs), *_mean_sd(rts), *_mean_sd(fam)))

    write_csv(out / "summary_subjects.csv", subj)
    write_csv(out / "summary_actions.csv", actions)
    write_csv(out / "subject_action.csv", long_rows)
    written = [out / "summary_subjects.csv", out / "summary_actions.csv", out / "subject_action.csv"]
    if not a.no_plots:
        try:
            written += make_plots(actions, lat, out, len(subj))
        except ImportError:
            print("note: matplotlib not installed -> no plots (pip install matplotlib)", file=sys.stderr)
    print("\nwrote:\n  " + "\n  ".join(str(p) for p in written if p.exists()))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
