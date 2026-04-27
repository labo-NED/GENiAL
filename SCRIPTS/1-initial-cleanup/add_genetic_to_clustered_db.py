from pathlib import Path
import pandas as pd

# ------------------------------------------------------------
# PATHS / CONSTANTS - EDIT THESE
# ------------------------------------------------------------
ROOT_DIR = Path("/Users/emmanuelle.coutu-nadeau/Code/NED LAB/GENiAL")

# Final clustered/final DB used in stats
CLUSTERED_DB_PATH = ROOT_DIR / "DATA/OUTPUTS/Clustered/clustered_SOM_Q1K_CHU_MHC_BC_DATA_MAR_09_2026.csv"

# New REDCap export (initial format)
REDCAP_DB_PATH = ROOT_DIR / "DATA/REDCAP_REPORTS/Q1K/Q1KDatabase-ECNDEMEEGDIABEHIQGEN_DATA_2026-04-23_1000.csv"

# Output merged file
OUTPUT_PATH = ROOT_DIR / "DATA/OUTPUTS/Clustered/clustered_SOM_Q1K_CHU_MHC_BC_DATA_APR_23_2026_with_genetic.csv"


def _first_non_empty(series: pd.Series):
    """Return first non-empty value from a series, else NA."""
    for value in series:
        if pd.isna(value):
            continue
        if str(value).strip() != "":
            return value
    return pd.NA


def _normalize_text(series: pd.Series) -> pd.Series:
    """Normalize values for emptiness checks."""
    return (
        series.astype("string")
        .str.strip()
        .str.upper()
        .replace({"": pd.NA, "NAN": pd.NA, "NONE": pd.NA, "NA": pd.NA})
    )


def _get_redcap_participant_column(redcap_df: pd.DataFrame) -> str:
    if "participant_id" in redcap_df.columns:
        return "participant_id"
    if "eeg_participant_code" in redcap_df.columns:
        return "eeg_participant_code"
    raise ValueError(
        "No participant identifier found in REDCap file. "
        "Expected `participant_id` or `eeg_participant_code`."
    )


def build_genetic_table(redcap_df: pd.DataFrame) -> pd.DataFrame:
    participant_col = _get_redcap_participant_column(redcap_df)
    required_cols = ["gt_cnv_status", "gt_snv_mut"]

    missing_cols = [col for col in required_cols if col not in redcap_df.columns]
    if missing_cols:
        raise ValueError(
            f"Missing required genetic columns in REDCap file: {', '.join(missing_cols)}"
        )

    needs_record_id = "record_id" in redcap_df.columns
    base_cols = [participant_col] + required_cols + (["record_id"] if needs_record_id else [])
    subset = redcap_df[base_cols].copy()
    subset = subset.rename(columns={participant_col: "participant_id"})
    subset["participant_id"] = _normalize_text(subset["participant_id"])

    # REDCap repeat instruments often store genetics on rows where participant_id is empty.
    # Propagate participant_id within each record_id before filtering.
    if "record_id" in subset.columns:
        subset["record_id"] = _normalize_text(subset["record_id"])
        subset["participant_id"] = (
            subset.groupby("record_id")["participant_id"].transform(_first_non_empty)
        )

    subset = subset.dropna(subset=["participant_id"])

    subset["cnv_non_empty"] = _normalize_text(subset["gt_cnv_status"]).notna()
    subset["snv_non_empty"] = _normalize_text(subset["gt_snv_mut"]).notna()
    subset["genetic_present"] = subset["cnv_non_empty"] | subset["snv_non_empty"]

    grouped = (
        subset.groupby("participant_id", as_index=False)
        .agg(
            {
                "gt_cnv_status": _first_non_empty,
                "gt_snv_mut": _first_non_empty,
                "genetic_present": "any",
            }
        )
        .copy()
    )

    grouped["genetic"] = grouped["genetic_present"].map({True: "yes", False: ""})
    grouped = grouped.drop(columns=["genetic_present"])

    return grouped


def merge_genetic_info(
    clustered_db_path: Path, redcap_db_path: Path, output_path: Path
) -> None:
    clustered_df = pd.read_csv(clustered_db_path)
    redcap_df = pd.read_csv(redcap_db_path)

    if "participant_id" not in clustered_df.columns:
        raise ValueError(
            "No `participant_id` column found in clustered database."
        )

    clustered_df["participant_id"] = _normalize_text(clustered_df["participant_id"])
    genetic_df = build_genetic_table(redcap_df)

    merged_df = clustered_df.drop(
        columns=[col for col in ["gt_cnv_status", "gt_snv_mut", "genetic"] if col in clustered_df.columns]
    ).merge(genetic_df, on="participant_id", how="left", indicator=True)

    matched_count = int((merged_df["_merge"] == "both").sum())
    merged_df = merged_df.drop(columns=["_merge"])

    output_path.parent.mkdir(parents=True, exist_ok=True)
    merged_df.to_csv(output_path, index=False)

    positive_count = (merged_df["genetic"] == "yes").sum()

    print(f"Saved merged file: {output_path}")
    print(f"Rows in output: {len(merged_df)}")
    print(f"Participants with genetic info matched: {matched_count}")
    print(f"Participants flagged genetic=yes: {positive_count}")

    if "cluster" in merged_df.columns:
        cluster_summary = (
            merged_df.groupby("cluster", dropna=False)
            .agg(
                total_n=("participant_id", "size"),
                genetic_n=("genetic", lambda s: (s == "yes").sum()),
            )
            .sort_index()
        )
        cluster_summary["genetic_pct"] = (
            cluster_summary["genetic_n"] / cluster_summary["total_n"] * 100
        )

        print("\nGenetic summary per cluster:")
        for cluster_value, row in cluster_summary.iterrows():
            print(
                f"  Cluster {cluster_value}: total n = {int(row['total_n'])}, "
                f"genetic n = {int(row['genetic_n'])}, "
                f"% gen = {row['genetic_pct']:.1f}"
            )


if __name__ == "__main__":
    merge_genetic_info(
        clustered_db_path=CLUSTERED_DB_PATH,
        redcap_db_path=REDCAP_DB_PATH,
        output_path=OUTPUT_PATH,
    )
