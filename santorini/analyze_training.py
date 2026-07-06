"""
Export TensorBoard scalars to CSV and analyze Santorini training runs for
convergence.

CSV format: long-form with columns `run,tag,step,wall_time,value`.

Usage:
    # TB event files -> per-run scalars.csv + combined all_scalars.csv
    uv run python -m santorini.analyze_training export models/tb

    # Analyze a CSV
    uv run python -m santorini.analyze_training analyze models/tb/all_scalars.csv

    # Export then analyze in one go (auto-detects TB dir vs CSV file)
    uv run python -m santorini.analyze_training run models/tb
"""

from __future__ import annotations

import argparse
import csv
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path

import numpy as np


# Direction of "better" for each metric when judging training health.
# +1 = higher is better, -1 = lower is better, 0 = context-dependent / diagnostic only.
METRIC_DIRECTION: dict[str, int] = {
    "rollout/ep_rew_mean": +1,
    "eval/winrate_vs_random": +1,
    "eval/winrate_vs_random_p0": +1,
    "eval/winrate_vs_random_p1": +1,
    "eval/winrate_vs_greedy_p0": +1,
    "eval/winrate_vs_greedy_p1": +1,
    "eval/winrate_vs_latest_snapshot": 0,  # ~0.5 is healthy vs a recent snapshot
    "eval/traj_diversity": +1,
    "train/explained_variance": +1,
    "train/value_loss": -1,
    "train/loss": -1,
    "train/approx_kl": 0,        # want bounded, not necessarily monotonic
    "train/clip_fraction": 0,
    "train/entropy_loss": 0,     # drifts toward 0 as policy commits; interpret with rewards
    "train/policy_gradient_loss": 0,
    "train/learning_rate": 0,
    "train/clip_range": 0,
    "rollout/ep_len_mean": 0,
    "time/fps": 0,
    "eval/mean_ep_length": 0,
    "eval/ep_len_mean": 0,
    "eval/ep_len_std": 0,
    "selfplay/learner_winrate": 0,       # ~0.5 healthy; ~0 or ~1 is a red flag
    "selfplay/learner_winrate_seat0": 0,
    "selfplay/learner_winrate_seat1": 0,
    "selfplay/ep_len_mean": 0,
    "selfplay/terminal_reward_mean": 0,
    "selfplay/shaping_total_mean": 0,
}


@dataclass
class MetricStats:
    tag: str
    n: int
    first_step: int
    last_step: int
    first_val: float
    last_val: float
    tail_mean: float
    tail_std: float
    tail_cv: float          # coefficient of variation over the tail window
    slope_per_1k: float     # linear regression slope, per 1000 env steps, over tail
    rel_slope_per_1k: float # slope / |tail_mean|, per 1000 steps (unitless)


def export_tb_to_csv(
    tb_root: Path,
    combined_csv: Path | None = None,
) -> tuple[dict[str, Path], Path | None]:
    """Walk `tb_root` for TB event-file directories and write scalar CSVs.

    For each immediate subdirectory of `tb_root` containing TB event files,
    writes `<subdir>/scalars.csv` with columns `run,tag,step,wall_time,value`.
    If `combined_csv` is provided, also writes a single combined CSV with the
    same schema across all runs (defaults to `<tb_root>/all_scalars.csv`).

    Returns a `({run_name: per_run_csv_path}, combined_path | None)` tuple.
    """
    from tensorboard.backend.event_processing.event_accumulator import (
        SCALARS,
        EventAccumulator,
    )

    combined_csv = combined_csv if combined_csv is not None else tb_root / "all_scalars.csv"
    per_run: dict[str, Path] = {}
    all_rows: list[tuple[str, str, int, float, float]] = []

    for run_dir in sorted(p for p in tb_root.iterdir() if p.is_dir()):
        ea = EventAccumulator(str(run_dir), size_guidance={SCALARS: 0})
        ea.Reload()
        tags = ea.Tags().get("scalars", [])
        if not tags:
            continue
        rows: list[tuple[str, str, int, float, float]] = []
        for tag in tags:
            for ev in ea.Scalars(tag):
                rows.append((run_dir.name, tag, ev.step, ev.wall_time, ev.value))
        out = run_dir / "scalars.csv"
        with out.open("w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["run", "tag", "step", "wall_time", "value"])
            w.writerows(rows)
        per_run[run_dir.name] = out
        all_rows.extend(rows)
        print(f"{run_dir.name}: {len(tags)} tags, {len(rows)} rows -> {out}")

    if all_rows:
        with combined_csv.open("w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["run", "tag", "step", "wall_time", "value"])
            w.writerows(all_rows)
        print(f"combined -> {combined_csv} ({len(all_rows)} rows)")
        return per_run, combined_csv
    return per_run, None


def load_scalars(csv_path: Path) -> dict[tuple[str, str], np.ndarray]:
    """Return {(run, tag): array of shape (n, 2) with columns [step, value]}."""
    series: dict[tuple[str, str], list[tuple[int, float]]] = defaultdict(list)
    with csv_path.open() as f:
        reader = csv.DictReader(f)
        for row in reader:
            series[(row["run"], row["tag"])].append(
                (int(row["step"]), float(row["value"]))
            )
    out: dict[tuple[str, str], np.ndarray] = {}
    for key, pts in series.items():
        arr = np.array(sorted(pts), dtype=float)
        out[key] = arr
    return out


def compute_stats(steps: np.ndarray, values: np.ndarray, window: int) -> MetricStats:
    n = len(values)
    tail_n = min(window, n)
    tail_s = steps[-tail_n:]
    tail_v = values[-tail_n:]
    tail_mean = float(tail_v.mean())
    tail_std = float(tail_v.std())
    tail_cv = tail_std / abs(tail_mean) if tail_mean != 0 else float("inf")
    if tail_n >= 2 and np.ptp(tail_s) > 0:
        # Simple least-squares slope of value vs step.
        slope = float(np.polyfit(tail_s, tail_v, 1)[0])
    else:
        slope = 0.0
    slope_per_1k = slope * 1000.0
    rel_slope_per_1k = slope_per_1k / abs(tail_mean) if tail_mean != 0 else 0.0
    return MetricStats(
        tag="",
        n=n,
        first_step=int(steps[0]),
        last_step=int(steps[-1]),
        first_val=float(values[0]),
        last_val=float(values[-1]),
        tail_mean=tail_mean,
        tail_std=tail_std,
        tail_cv=tail_cv,
        slope_per_1k=slope_per_1k,
        rel_slope_per_1k=rel_slope_per_1k,
    )


def verdict(tag: str, s: MetricStats) -> str:
    """One-word health verdict per metric based on direction and tail trend."""
    direction = METRIC_DIRECTION.get(tag, 0)
    # Treat |rel slope| < 0.5% per 1k steps as "flat".
    flat_thresh = 5e-3
    improving = s.rel_slope_per_1k * direction > flat_thresh
    regressing = s.rel_slope_per_1k * direction < -flat_thresh
    stable = abs(s.rel_slope_per_1k) < flat_thresh

    if direction == 0:
        # Diagnostic metric: just describe trend.
        if stable:
            return "stable"
        return "rising" if s.rel_slope_per_1k > 0 else "falling"

    if improving:
        return "improving"
    if regressing:
        return "regressing"
    return "plateau"


def _mean_of(results: dict[str, MetricStats], tags: list[str]) -> float | None:
    """Average tail_mean across whichever of `tags` are present."""
    vals = [results[t].tail_mean for t in tags if t in results]
    return sum(vals) / len(vals) if vals else None


def degenerate_flags(results: dict[str, MetricStats]) -> list[str]:
    """Detect the deterministic self-play collapse failure mode: every game
    funnels into the same line of play while gradients flatline. The old
    shared-stream runs look exactly like this (ep_len variance ~0, entropy ~0)
    while still reporting a high winrate vs random."""
    flags: list[str] = []
    diversity = results.get("eval/traj_diversity")
    if diversity and diversity.tail_mean < 0.3:
        flags.append(
            f"trajectory diversity collapsed ({diversity.tail_mean:.0%} unique)"
        )
    ep_len = results.get("rollout/ep_len_mean")
    entropy = results.get("train/entropy_loss")
    if (
        ep_len
        and entropy
        and ep_len.tail_std < 0.1
        and abs(entropy.tail_mean) < 0.05
    ):
        flags.append(
            f"policy frozen (ep_len std={ep_len.tail_std:.3f}, "
            f"entropy={entropy.tail_mean:.4f})"
        )
    return flags


def convergence_verdict(results: dict[str, MetricStats]) -> str:
    """Top-level verdict that weights the most load-bearing signals."""
    rew = results.get("rollout/ep_rew_mean")
    wr = results.get("eval/winrate_vs_random")
    ev = results.get("train/explained_variance")
    kl = results.get("train/approx_kl")

    notes: list[str] = []

    # A saturated winrate vs random is meaningless if the policy is
    # degenerate, so check for collapse first.
    collapse = degenerate_flags(results)
    wr_random = _mean_of(
        results,
        ["eval/winrate_vs_random_p0", "eval/winrate_vs_random_p1"],
    )
    wr_greedy = _mean_of(
        results,
        ["eval/winrate_vs_greedy_p0", "eval/winrate_vs_greedy_p1"],
    )
    if collapse:
        detail = "; ".join(collapse)
        if wr_random is not None or wr:
            shown = wr_random if wr_random is not None else wr.tail_mean
            detail += f"; winrate vs random ({shown:.1%}) is NOT trustworthy"
        return f"DEGENERATE (self-play collapse) — {detail}"

    if rew:
        if rew.rel_slope_per_1k > 5e-3:
            notes.append("reward rising")
        elif rew.rel_slope_per_1k < -5e-3:
            notes.append("reward regressing")
        else:
            notes.append("reward plateau")

    # Prefer the per-seat winrates from the self-play trainer; fall back to
    # the legacy single-seat eval/winrate_vs_random tag.
    if wr_random is None and wr:
        wr_random = wr.tail_mean
    if wr_random is not None:
        if wr_random >= 0.95:
            notes.append(f"winrate vs random saturated at {wr_random:.1%}")
        elif wr and wr.rel_slope_per_1k > 5e-3:
            notes.append(f"winrate vs random still rising ({wr_random:.1%})")
        elif wr and wr.rel_slope_per_1k < -5e-3:
            notes.append(f"winrate vs random regressing ({wr_random:.1%})")
        else:
            notes.append(f"winrate vs random @ {wr_random:.1%}")
    if wr_greedy is not None:
        notes.append(f"winrate vs greedy @ {wr_greedy:.1%}")

    if ev and ev.tail_mean < 0.1:
        notes.append(f"value fn weak (EV={ev.tail_mean:.2f})")

    if kl and kl.tail_mean > 0.05:
        notes.append(f"KL high (avg={kl.tail_mean:.3f}) — updates may be unstable")

    # Top-level label
    improving = any("rising" in n for n in notes)
    saturated = any("saturated" in n for n in notes)
    regressing = any("regressing" in n for n in notes)

    # "Converged" requires beating a non-trivial baseline, not just random:
    # random never blocks a threat, so degenerate policies still crush it.
    strong = wr_greedy is not None and wr_greedy >= 0.7
    if saturated and strong and not regressing:
        label = "CONVERGED (beats greedy baseline)"
    elif saturated and wr_greedy is None and not regressing:
        label = "BEATS RANDOM ONLY (no greedy-baseline eval in this run)"
    elif saturated and not regressing:
        label = f"BEATS RANDOM ONLY (greedy baseline @ {wr_greedy:.1%})"
    elif improving and not regressing:
        label = "CONVERGING"
    elif regressing:
        label = "REGRESSING"
    else:
        label = "PLATEAUED (not yet converged)"

    return f"{label} — " + "; ".join(notes)


def analyze_run(
    run: str,
    run_series: dict[str, np.ndarray],
    window: int,
) -> None:
    header = f"=== {run} ==="
    print(header)
    results: dict[str, MetricStats] = {}
    # Print in a stable order: main signals first, diagnostics after.
    order = [
        "rollout/ep_rew_mean",
        "eval/winrate_vs_random",
        "eval/winrate_vs_random_p0",
        "eval/winrate_vs_random_p1",
        "eval/winrate_vs_greedy_p0",
        "eval/winrate_vs_greedy_p1",
        "eval/winrate_vs_latest_snapshot",
        "eval/traj_diversity",
        "selfplay/learner_winrate",
        "selfplay/learner_winrate_seat0",
        "selfplay/learner_winrate_seat1",
        "eval/mean_ep_length",
        "eval/ep_len_mean",
        "eval/ep_len_std",
        "rollout/ep_len_mean",
        "selfplay/ep_len_mean",
        "selfplay/terminal_reward_mean",
        "selfplay/shaping_total_mean",
        "train/explained_variance",
        "train/entropy_loss",
        "train/approx_kl",
        "train/clip_fraction",
        "train/value_loss",
        "train/policy_gradient_loss",
        "train/loss",
        "train/learning_rate",
        "train/clip_range",
        "time/fps",
    ]
    known = [t for t in order if t in run_series]
    extras = sorted(t for t in run_series if t not in order)
    tags = known + extras

    print(
        f"{'tag':<32} {'n':>5} {'last':>12} {'tail_mean':>12} "
        f"{'tail_std':>10} {'Δ/1k':>12} {'Δ%/1k':>9} verdict"
    )
    for tag in tags:
        arr = run_series[tag]
        steps, values = arr[:, 0], arr[:, 1]
        if len(values) == 0:
            continue
        stats = compute_stats(steps, values, window)
        stats.tag = tag
        results[tag] = stats
        v = verdict(tag, stats)
        print(
            f"{tag:<32} {stats.n:>5} {stats.last_val:>12.4g} "
            f"{stats.tail_mean:>12.4g} {stats.tail_std:>10.3g} "
            f"{stats.slope_per_1k:>12.4g} {stats.rel_slope_per_1k*100:>8.2f}% "
            f"{v}"
        )

    total_steps = max(s.last_step for s in results.values())
    print(f"\nTotal env steps: {total_steps:,}")
    print(f"Tail window: last {window} rollouts")
    print(f"Convergence verdict: {convergence_verdict(results)}")
    print()


def _analyze_csv(csv_path: Path, window: int) -> None:
    series = load_scalars(csv_path)
    runs = sorted({run for run, _ in series})
    for run in runs:
        run_series = {tag: arr for (r, tag), arr in series.items() if r == run}
        if not run_series:
            continue
        analyze_run(run, run_series, window)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="cmd", required=True)

    p_export = sub.add_parser(
        "export", help="Export TB event files under a directory to scalars.csv."
    )
    p_export.add_argument("tb_dir", type=Path, help="Directory containing TB runs.")
    p_export.add_argument(
        "--combined",
        type=Path,
        default=None,
        help="Combined CSV path (default: <tb_dir>/all_scalars.csv).",
    )

    p_analyze = sub.add_parser(
        "analyze", help="Analyze a long-format scalars CSV for convergence."
    )
    p_analyze.add_argument(
        "csv",
        type=Path,
        help="Long-format scalars CSV (run,tag,step,wall_time,value).",
    )
    p_analyze.add_argument(
        "--window",
        type=int,
        default=40,
        help="Number of most-recent samples used for tail stats (default: 40).",
    )

    p_run = sub.add_parser(
        "run", help="Export TB dir to CSV then analyze (one-shot)."
    )
    p_run.add_argument("tb_dir", type=Path, help="Directory containing TB runs.")
    p_run.add_argument("--window", type=int, default=40)

    args = parser.parse_args()

    if args.cmd == "export":
        export_tb_to_csv(args.tb_dir, args.combined)
    elif args.cmd == "analyze":
        _analyze_csv(args.csv, args.window)
    elif args.cmd == "run":
        _, combined = export_tb_to_csv(args.tb_dir)
        if combined is None:
            print("No scalar data found; nothing to analyze.")
            return
        print()
        _analyze_csv(combined, args.window)


if __name__ == "__main__":
    main()
