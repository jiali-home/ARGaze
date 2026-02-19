#!/usr/bin/env python3
"""
Dataset Exploration Utility

Generates:
1) Basic table of dataset stats:
   - Total Samples, Unique Tasks, Unique Sites, Unique Participants,
     Avg. Samples per Task, Avg. Samples per Site, Avg. Samples per Participant
2) Frequency distribution tables:
   - Samples per Task, per Site, per Participant
3) Cross-tabulation (pivot) matrix:
   - Task vs. Site (rows: task_id, cols: site_id, values: sample counts)
4) Visualizations:
   - Bar charts for distributions, heatmap for pivot, box plots for distributions

Outputs are saved into the given output directory (default: ./explore).

Usage:
  python explore/generate_exploration.py \
      --csv /path/to/your.csv \
      --outdir explore \
      [--task-col task_id] [--site-col site_id] [--participant-col participant_id]

If column names differ, the script attempts to auto-detect from common synonyms.
"""

from __future__ import annotations

import argparse
import os
import sys
from typing import List, Optional, Tuple

import pandas as pd
import numpy as np


def _infer_column(df: pd.DataFrame, candidates: List[str], provided: Optional[str], label: str) -> str:
    """Infer a column name from user-provided value or candidate list.

    Raises a clear error if no suitable column is found.
    """
    if provided:
        if provided in df.columns:
            return provided
        raise ValueError(f"Provided {label} column '{provided}' not found in CSV. Available: {list(df.columns)}")
    for c in candidates:
        if c in df.columns:
            return c
    raise ValueError(
        f"Could not infer {label} column. Looked for any of: {candidates}. Available columns: {list(df.columns)}"
    )


def _ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def _save_table(df: pd.DataFrame, outdir: str, basename: str) -> str:
    path = os.path.join(outdir, f"{basename}.csv")
    df.to_csv(path, index=hasattr(df, "index") and df.index.name is not None)
    return path


def _series_for(df: pd.DataFrame, colname: str) -> pd.Series:
    """Return a 1-D Series for a possibly duplicated column name.

    If multiple columns share the same name, take the first occurrence.
    """
    matches = [i for i, c in enumerate(df.columns) if c == colname]
    if not matches:
        raise KeyError(f"Column not found: {colname}")
    s = df.iloc[:, matches[0]]
    # Ensure the name is the requested label for downstream readability
    s = s.copy()
    s.name = colname
    return s


def _sanitize(value) -> str:
    s = str(value)
    s = s.strip().replace("/", "-").replace("\\", "-").replace(" ", "_")
    s = s.replace(":", "-").replace("|", "-").replace("\n", "-")
    if len(s) > 80:
        s = s[:80]
    return s


def compute_and_save_basic_table(df: pd.DataFrame, task_col: str, site_col: str, participant_col: str, outdir: str) -> pd.DataFrame:
    total_samples = len(df)
    unique_tasks = _series_for(df, task_col).nunique(dropna=True)
    unique_sites = _series_for(df, site_col).nunique(dropna=True)
    unique_participants = _series_for(df, participant_col).nunique(dropna=True)

    avg_samples_per_task = total_samples / unique_tasks if unique_tasks else float("nan")
    avg_samples_per_site = total_samples / unique_sites if unique_sites else float("nan")
    avg_samples_per_participant = total_samples / unique_participants if unique_participants else float("nan")

    basic = pd.DataFrame(
        {
            "Metric": [
                "Total Samples",
                "Unique Tasks",
                "Unique Sites",
                "Unique Participants",
                "Avg. Samples per Task",
                "Avg. Samples per Site",
                "Avg. Samples per Participant",
            ],
            "Value": [
                total_samples,
                unique_tasks,
                unique_sites,
                unique_participants,
                round(avg_samples_per_task, 3) if pd.notna(avg_samples_per_task) else avg_samples_per_task,
                round(avg_samples_per_site, 3) if pd.notna(avg_samples_per_site) else avg_samples_per_site,
                round(avg_samples_per_participant, 3) if pd.notna(avg_samples_per_participant) else avg_samples_per_participant,
            ],
        }
    )

    _save_table(basic, outdir, "basic_stats")
    return basic


def freq_tables(df: pd.DataFrame, task_col: str, site_col: str, participant_col: str, outdir: str) -> dict:
    tables = {}
    tables["by_task"] = _series_for(df, task_col).value_counts(dropna=False).rename_axis(task_col).reset_index(name="samples")
    tables["by_site"] = _series_for(df, site_col).value_counts(dropna=False).rename_axis(site_col).reset_index(name="samples")
    tables["by_participant"] = (
        _series_for(df, participant_col).value_counts(dropna=False).rename_axis(participant_col).reset_index(name="samples")
    )

    for name, tdf in tables.items():
        _save_table(tdf, outdir, f"freq_{name}")
    return tables


def pivot_task_site(df: pd.DataFrame, task_col: str, site_col: str, outdir: str) -> pd.DataFrame:
    # Use crosstab on 1-D Series to avoid issues with duplicate column names
    p = pd.crosstab(_series_for(df, task_col), _series_for(df, site_col))
    p = p.astype(int)
    p.to_csv(os.path.join(outdir, "pivot_task_vs_site.csv"))
    return p


def pivot_site_task(df: pd.DataFrame, task_col: str, site_col: str, outdir: Optional[str] = None) -> pd.DataFrame:
    p = pd.crosstab(_series_for(df, site_col), _series_for(df, task_col)).astype(int)
    if outdir is not None:
        p.to_csv(os.path.join(outdir, "pivot_site_vs_task.csv"))
    return p


def try_import_seaborn():
    try:
        import seaborn as sns  # type: ignore

        return sns
    except Exception:
        return None


def draw_plots(
    tables: dict,
    pivot: pd.DataFrame,
    outdir: str,
    task_col: str,
    site_col: str,
    participant_col: str,
    top_n: int = 50,
) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    sns = try_import_seaborn()

    # Bar charts (sorted descending). Limit to top_n to keep charts readable.
    for key, label in (
        ("by_task", task_col),
        ("by_site", site_col),
        ("by_participant", participant_col),
    ):
        df_counts = tables[key].copy()
        df_counts_sorted = df_counts.sort_values("samples", ascending=False).head(top_n)
        plt.figure(figsize=(10, max(4, 0.25 * len(df_counts_sorted))))
        plt.barh(df_counts_sorted[label].astype(str), df_counts_sorted["samples"], color="steelblue")
        plt.xlabel("Samples")
        plt.ylabel(label)
        plt.title(f"Samples per {label} (top {min(top_n, len(df_counts))})")
        plt.tight_layout()
        plt.savefig(os.path.join(outdir, f"bar_{key}.png"), dpi=200)
        plt.close()

    # Heatmap for task vs site
    figsize = (max(6, 0.4 * pivot.shape[1]), max(6, 0.3 * pivot.shape[0]))
    figsize = (12, 12)
    plt.figure(figsize=figsize)
    if sns is not None:
        sns.heatmap(pivot, cmap="YlGnBu")
    else:
        # Fallback: imshow without annotations
        plt.imshow(pivot.values, aspect="auto", cmap="YlGnBu")
        plt.colorbar()
        plt.xticks(ticks=range(pivot.shape[1]), labels=pivot.columns.astype(str), rotation=90)
        plt.yticks(ticks=range(pivot.shape[0]), labels=pivot.index.astype(str))
    plt.title("Task vs. Site (samples)")
    plt.tight_layout()
    plt.savefig(os.path.join(outdir, "heatmap_task_vs_site.png"), dpi=200)
    plt.close()

    # Additional heatmap: Site vs Task (as drill-down entry point)
    site_task = pivot.T
    plt.figure(figsize=(max(6, 0.4 * site_task.shape[1]), max(6, 0.3 * site_task.shape[0])))
    figsize = (12, 12)
    if sns is not None:
        sns.heatmap(site_task, cmap="YlOrRd")
    else:
        plt.imshow(site_task.values, aspect="auto", cmap="YlOrRd")
        plt.colorbar()
        plt.xticks(ticks=range(site_task.shape[1]), labels=site_task.columns.astype(str), rotation=90)
        plt.yticks(ticks=range(site_task.shape[0]), labels=site_task.index.astype(str))
    plt.title("Site vs. Task (samples)")
    plt.tight_layout()
    plt.savefig(os.path.join(outdir, "heatmap_site_vs_task.png"), dpi=200)
    plt.close()


def drilldown_analysis(
    df: pd.DataFrame,
    task_col: str,
    site_col: str,
    participant_col: str,
    outdir: str,
    top_pairs_n: int = 10,
    pair_to_drill: Optional[Tuple[str, str]] = None,
) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    drill_dir = os.path.join(outdir, "drilldown")
    _ensure_dir(drill_dir)

    # Overall Site-Task pivot and heatmap
    p_site_task = pivot_site_task(df, task_col, site_col, outdir=drill_dir)

    # Top-N pairs by count
    pairs_long = (
        p_site_task.stack().reset_index().rename(columns={0: "samples", site_col: "site", task_col: "task"})
    )
    pairs_long = pairs_long[pairs_long["samples"] > 0]
    top_pairs = pairs_long.sort_values("samples", ascending=False).head(top_pairs_n)
    _save_table(top_pairs, drill_dir, "top_site_task_pairs")

    # Determine which pairs to analyze
    if pair_to_drill:
        pairs_to_analyze: List[Tuple[str, str]] = [pair_to_drill]
    else:
        pairs_to_analyze = list(zip(top_pairs["site"].astype(str), top_pairs["task"].astype(str)))

    for site_val, task_val in pairs_to_analyze:
        mask = (_series_for(df, site_col).astype(str) == str(site_val)) & (
            _series_for(df, task_col).astype(str) == str(task_val)
        )
        sub = df[mask]
        if sub.empty:
            continue
        counts = _series_for(sub, participant_col).value_counts(dropna=False).reset_index()
        counts.columns = [participant_col, "samples"]

        pair_dir = os.path.join(drill_dir, f"{_sanitize(site_val)}__{_sanitize(task_val)}")
        _ensure_dir(pair_dir)
        _save_table(counts, pair_dir, "participants_distribution")

        # Plot bar chart for participants
        plt.figure(figsize=(10, max(4, 0.25 * len(counts))))
        plt.barh(counts[participant_col].astype(str), counts["samples"], color="teal")
        plt.xlabel("Samples")
        plt.ylabel("Participant")
        plt.title(f"Participants for {site_val} - {task_val}")
        plt.tight_layout()
        plt.savefig(os.path.join(pair_dir, "participants_distribution.png"), dpi=200)
        plt.close()


def participant_centric_analysis(
    df: pd.DataFrame,
    task_col: str,
    site_col: str,
    participant_col: str,
    outdir: str,
    top_k: int = 3,
    mid_k: int = 3,
) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    pc_dir = os.path.join(outdir, "participant_centric")
    _ensure_dir(pc_dir)

    counts = _series_for(df, participant_col).value_counts().reset_index()
    counts.columns = [participant_col, "samples"]
    if counts.empty:
        _save_table(counts, pc_dir, "selected_participants")
        return

    # Top-K by samples
    top_participants = counts.head(max(0, top_k))[participant_col].astype(str).tolist()

    # Mid-K near median
    median_samples = float(np.median(counts["samples"]))
    counts["median_diff"] = (counts["samples"] - median_samples).abs()
    mid_candidates = (
        counts.sort_values(["median_diff", "samples"], ascending=[True, False])[participant_col]
        .astype(str)
        .tolist()
    )
    mid_participants = [p for p in mid_candidates if p not in set(top_participants)][: max(0, mid_k)]

    selected = pd.DataFrame(
        {
            "participant": top_participants + mid_participants,
            "role": ["top"] * len(top_participants) + ["mid"] * len(mid_participants),
        }
    )
    _save_table(selected, pc_dir, "selected_participants")

    for pid in selected["participant"].astype(str).tolist():
        sub = df[_series_for(df, participant_col).astype(str) == pid]
        if sub.empty:
            continue
        pst = pivot_site_task(sub, task_col, site_col)
        indiv_dir = os.path.join(pc_dir, _sanitize(pid))
        _ensure_dir(indiv_dir)
        pst.to_csv(os.path.join(indiv_dir, "pivot_site_vs_task.csv"))

        plt.figure(figsize=(max(6, 0.4 * pst.shape[1]), max(4, 0.3 * pst.shape[0])))
        sns = try_import_seaborn()
        if sns is not None:
            sns.heatmap(pst, cmap="Blues")
        else:
            plt.imshow(pst.values, aspect="auto", cmap="Blues")
            plt.colorbar()
            plt.xticks(ticks=range(pst.shape[1]), labels=pst.columns.astype(str), rotation=90)
            plt.yticks(ticks=range(pst.shape[0]), labels=pst.index.astype(str))
        plt.title(f"{pid}: Site vs. Task")
        plt.tight_layout()
        plt.savefig(os.path.join(indiv_dir, "heatmap_site_vs_task.png"), dpi=200)
        plt.close()

    # Box plots for distribution of samples per group
    for key, label in (
        ("by_task", task_col),
        ("by_site", site_col),
        ("by_participant", participant_col),
    ):
        df_counts = tables[key]
        # Work with the counts only
        series = df_counts["samples"]
        plt.figure(figsize=(6, 4))
        if sns is not None:
            sns.boxplot(x=series, orient="h", color="lightsteelblue")  # type: ignore
        else:
            plt.boxplot(series, vert=False)
        plt.xlabel("Samples")
        plt.title(f"Distribution: Samples per {label}")
        plt.tight_layout()
        plt.savefig(os.path.join(outdir, f"boxplot_{key}.png"), dpi=200)
        plt.close()


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="Generate dataset exploration tables and charts.")
    parser.add_argument("--csv", default="/mnt/sdc1/jiali/znotebook/GLC/output/aria_video_analysis_v2/aria_video_clips_for_benchmark.csv", help="Path to input CSV file.")
    parser.add_argument("--outdir", default="explore", help="Directory to save outputs (default: explore)")
    parser.add_argument("--task-col", dest="task_col", default=None, help="Task column name (optional)")
    parser.add_argument("--site-col", dest="site_col", default=None, help="Site column name (optional)")
    parser.add_argument(
        "--participant-col", dest="participant_col", default=None, help="Participant column name (optional)"
    )
    parser.add_argument("--top-n", dest="top_n", type=int, default=50, help="Top-N items to plot in bar charts.")
    # Drill-down analysis options
    parser.add_argument("--drill-top-n", dest="drill_top_n", type=int, default=10, help="Top-N Site-Task pairs to analyze.")
    parser.add_argument("--drill-pair-site", dest="drill_pair_site", default=None, help="Specific Site to drill.")
    parser.add_argument("--drill-pair-task", dest="drill_pair_task", default=None, help="Specific Task to drill.")
    # Participant-centric analysis options
    parser.add_argument("--participant-top-k", dest="participant_top_k", type=int, default=3, help="Top-K participants by samples.")
    parser.add_argument("--participant-mid-k", dest="participant_mid_k", type=int, default=3, help="Mid-K participants near median samples.")

    args = parser.parse_args(argv)

    # Load CSV
    if not os.path.isfile(args.csv):
        print(f"ERROR: CSV not found: {args.csv}", file=sys.stderr)
        return 2

    df = pd.read_csv(args.csv)
    if df.empty:
        print("ERROR: CSV has no rows.", file=sys.stderr)
        return 2

    # Infer columns
    task_col = _infer_column(
        df,
        # Prefer human-readable names over IDs
        candidates=["task_name", "parent_task_name", "task", "task_id", "parent_task_id"],
        provided=args.task_col,
        label="task",
    )
    # If an ID column was selected but a name column exists, switch to name for tables/figures
    if task_col in {"task_id", "parent_task_id"}:
        for name_col in ["task_name", "parent_task_name", "task"]:
            if name_col in df.columns:
                task_col = name_col
                break
    site_col = _infer_column(
        df,
        candidates=["site_id", "site", "location", "location_id", "venue", "scene_id", "university_name"],
        provided=args.site_col,
        label="site",
    )
    participant_col = _infer_column(
        df,
        candidates=["participant_id", "participant_uid", "subject_id", "person_id", "user_id", "participant", "subject"],
        provided=args.participant_col,
        label="participant",
    )

    _ensure_dir(args.outdir)

    # 1) Basic table
    basic = compute_and_save_basic_table(df, task_col, site_col, participant_col, args.outdir)

    # 2) Frequency distribution tables
    tables = freq_tables(df, task_col, site_col, participant_col, args.outdir)

    # 3) Cross-tabulation
    p = pivot_task_site(df, task_col, site_col, args.outdir)

    # 4) Plots
    try:
        draw_plots(tables, p, args.outdir, task_col, site_col, participant_col, top_n=args.top_n)
    except Exception as e:
        # Do not fail the whole run if plotting backend or seaborn is missing.
        print(f"WARNING: Plotting failed: {e}", file=sys.stderr)

    # 5) Drill-down Analysis (Method 1)
    try:
        pair = None
        if args.drill_pair_site and args.drill_pair_task:
            pair = (str(args.drill_pair_site), str(args.drill_pair_task))
        drilldown_analysis(
            df,
            task_col=task_col,
            site_col=site_col,
            participant_col=participant_col,
            outdir=args.outdir,
            top_pairs_n=int(args.drill_top_n),
            pair_to_drill=pair,
        )
    except Exception as e:
        print(f"WARNING: Drill-down analysis failed: {e}", file=sys.stderr)

    # 6) Participant-Centric Analysis (Method 2)
    try:
        participant_centric_analysis(
            df,
            task_col=task_col,
            site_col=site_col,
            participant_col=participant_col,
            outdir=args.outdir,
            top_k=int(args.participant_top_k),
            mid_k=int(args.participant_mid_k),
        )
    except Exception as e:
        print(f"WARNING: Participant-centric analysis failed: {e}", file=sys.stderr)

    # Also write a short README for convenience
    readme_path = os.path.join(args.outdir, "README.txt")
    with open(readme_path, "w", encoding="utf-8") as f:
        f.write(
            "Generated files:\n"
            "- basic_stats.csv\n"
            "- freq_by_task.csv, freq_by_site.csv, freq_by_participant.csv\n"
            "- pivot_task_vs_site.csv\n"
            "- bar_by_task.png, bar_by_site.png, bar_by_participant.png\n"
            "- heatmap_task_vs_site.png, heatmap_site_vs_task.png\n"
            "- boxplot_by_task.png, boxplot_by_site.png, boxplot_by_participant.png\n"
            "\nDrill-down (Site-Task) in explore/drilldown/:\n"
            "- pivot_site_vs_task.csv, heatmap_site_vs_task.png\n"
            "- top_site_task_pairs.csv\n"
            "- <site>__<task>/participants_distribution.csv/.png\n"
            "\nParticipant-centric in explore/participant_centric/:\n"
            "- selected_participants.csv (top + mid)\n"
            "- <participant>/pivot_site_vs_task.csv/.png\n"
            f"\nDetected columns: task={task_col}, site={site_col}, participant={participant_col}\n"
        )

    print("Done. Outputs written to:", os.path.abspath(args.outdir))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
