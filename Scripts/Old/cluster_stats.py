#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import os
from pathlib import Path

import pandas as pd
import matplotlib.pyplot as plt


def summarize_column(series: pd.Series) -> pd.DataFrame:
    """Return value counts and proportions for a Series, including NaN as a category."""
    counts = series.fillna("<MISSING>").astype(str).value_counts(dropna=False)
    proportions = counts / counts.sum()
    summary = (
        pd.DataFrame({
            "count": counts,
            "proportion": proportions,
        })
        .reset_index()
        .rename(columns={"index": "value"})
        .sort_values(["count", "value"], ascending=[False, True])
        .reset_index(drop=True)
    )
    return summary


def plot_bar(summary: pd.DataFrame, title: str, out_png: Path) -> None:
    """Plot a horizontal bar chart for counts with proportions annotated."""
    plt.figure(figsize=(10, max(3, 0.4 * len(summary))))
    values = summary["value"].astype(str)
    counts = summary["count"]
    proportions = summary["proportion"].round(3)

    bars = plt.barh(values, counts, color="#4C78A8")
    plt.xlabel("Count")
    plt.title(title)
    plt.gca().invert_yaxis()

    for bar, prop in zip(bars, proportions):
        width = bar.get_width()
        label = f"{prop:.3f}"
        plt.text(width + max(counts) * 0.01, bar.get_y() + bar.get_height() / 2, label, va="center")

    plt.tight_layout()
    out_png.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_png, dpi=200)
    plt.close()


def main():
    parser = argparse.ArgumentParser(description="Compute counts and proportions for selected columns.")
    parser.add_argument(
        "--csv",
        default="/Users/emmanuelle.coutu-nadeau/Code/NED LAB/GENiAL/Data/proband_clusters_kmeans_complete_cases.csv",
        help="Path to input CSV",
    )
    parser.add_argument(
        "--columns",
        nargs="+",
        default=["family_ethnicity", "highest_education_level"],
        help="Columns to summarize",
    )
    parser.add_argument(
        "--outdir",
        default="/Users/emmanuelle.coutu-nadeau/Code/NED LAB/GENiAL/Data/Results/SPR-2025/demographics",
        help="Directory to write summary CSVs and charts",
    )
    args = parser.parse_args()

    csv_path = Path(args.csv)
    outdir = Path(args.outdir)

    if not csv_path.exists():
        raise FileNotFoundError(f"CSV not found: {csv_path}")

    df = pd.read_csv(csv_path)

    print(f"Loaded {csv_path} with shape {df.shape}")

    for col in args.columns:
        if col not in df.columns:
            print(f"[WARN] Column not found and will be skipped: {col}")
            continue

        summary = summarize_column(df[col])
        print("\n" + "=" * 80)
        print(f"Summary for '{col}':")
        print(summary.to_string(index=False))

        # Save CSV
        outdir.mkdir(parents=True, exist_ok=True)
        out_csv = outdir / f"{col}_value_counts.csv"
        summary.to_csv(out_csv, index=False)

        # Save plot
        out_png = outdir / f"{col}_value_counts.png"
        plot_bar(summary, title=f"{col} value counts", out_png=out_png)
        print(f"Saved: {out_csv}")
        print(f"Saved: {out_png}")


if __name__ == "__main__":
    main()
