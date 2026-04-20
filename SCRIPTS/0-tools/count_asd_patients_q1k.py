import argparse
import pandas as pd


def first_non_null(series: pd.Series):
    """Return first non-null value in a Series, or None."""
    values = series.dropna()
    return values.iloc[0] if not values.empty else None


def to_int_or_none(value):
    """Safely convert a value to int when possible."""
    if pd.isna(value):
        return None
    try:
        return int(float(value))
    except (ValueError, TypeError):
        return None


def has_asd(row: pd.Series) -> bool:
    """
    ASD-positive logic copied from preprocessing conventions:
    - ghf_asd: 1 means yes
    - diag_asd: 2 means yes
    """
    ghf_asd = to_int_or_none(row.get("ghf_asd"))
    diag_asd = to_int_or_none(row.get("diag_asd"))
    return ghf_asd == 1 or diag_asd == 2


def count_asd_patients(input_csv: str) -> tuple[int, int]:
    """
    Return:
    - number of unique participants flagged ASD
    - total number of unique participants assessed
    """
    df = pd.read_csv(input_csv)

    required = ["record_id", "ghf_asd", "diag_asd"]
    missing = [col for col in required if col not in df.columns]
    if missing:
        raise ValueError(
            f"Missing required column(s): {', '.join(missing)}. "
            "Please upload a Q1K file containing record_id, ghf_asd, and diag_asd."
        )

    # Collapse multiple REDCap rows per participant to one row.
    collapsed = (
        df[required]
        .groupby("record_id", as_index=False)
        .agg({"ghf_asd": first_non_null, "diag_asd": first_non_null})
    )

    asd_mask = collapsed.apply(has_asd, axis=1)
    asd_count = int(asd_mask.sum())
    total_count = int(len(collapsed))
    return asd_count, total_count


def main():
    parser = argparse.ArgumentParser(
        description="Quickly count ASD-positive participants in a Q1K CSV."
    )
    parser.add_argument(
        "--input",
        required=True,
        help="Absolute or relative path to the uploaded Q1K CSV file.",
    )
    args = parser.parse_args()

    asd_count, total_count = count_asd_patients(args.input)
    percent = (asd_count / total_count * 100) if total_count else 0.0

    print("ASD patient count (unique record_id):", asd_count)
    print("Total participants (unique record_id):", total_count)
    print(f"ASD percentage: {percent:.2f}%")


if __name__ == "__main__":
    main()
import argparse
import pandas as pd


def first_non_null(series: pd.Series):
    """Return first non-null value in a Series, or None."""
    values = series.dropna()
    return values.iloc[0] if not values.empty else None


def to_int_or_none(value):
    """Safely convert a value to int when possible."""
    if pd.isna(value):
        return None
    try:
        return int(float(value))
    except (ValueError, TypeError):
        return None


def has_asd(row: pd.Series) -> bool:
    """
    ASD-positive logic copied from preprocessing conventions:
    - ghf_asd: 1 means yes
    - diag_asd: 2 means yes
    """
    ghf_asd = to_int_or_none(row.get("ghf_asd"))
    diag_asd = to_int_or_none(row.get("diag_asd"))
    return ghf_asd == 1 or diag_asd == 2


def count_asd_patients(input_csv: str) -> tuple[int, int]:
    """
    Return:
    - number of unique participants flagged ASD
    - total number of unique participants assessed
    """
    df = pd.read_csv(input_csv)

    required = ["record_id", "ghf_asd", "diag_asd"]
    missing = [col for col in required if col not in df.columns]
    if missing:
        raise ValueError(
            f"Missing required column(s): {', '.join(missing)}. "
            "Please upload a Q1K file containing record_id, ghf_asd, and diag_asd."
        )

    # Collapse multiple REDCap rows per participant to one row.
    collapsed = (
        df[required]
        .groupby("record_id", as_index=False)
        .agg({"ghf_asd": first_non_null, "diag_asd": first_non_null})
    )

    asd_mask = collapsed.apply(has_asd, axis=1)
    asd_count = int(asd_mask.sum())
    total_count = int(len(collapsed))
    return asd_count, total_count


def main():
    parser = argparse.ArgumentParser(
        description="Quickly count ASD-positive participants in a Q1K CSV."
    )
    parser.add_argument(
        "--input",
        required=True,
        help="Absolute or relative path to the uploaded Q1K CSV file.",
    )
    args = parser.parse_args()

    asd_count, total_count = count_asd_patients(args.input)
    percent = (asd_count / total_count * 100) if total_count else 0.0

    print("ASD patient count (unique record_id):", asd_count)
    print("Total participants (unique record_id):", total_count)
    print(f"ASD percentage: {percent:.2f}%")


if __name__ == "__main__":
    main()
