"""
create_splits.py

Create train/validation/test splits from metadata.csv.
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

import pandas as pd


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    datefmt="%H:%M:%S"
)

logger = logging.getLogger(__name__)


# ==========================================================
# Arguments
# ==========================================================

def parse_args():

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--metadata",
        type=Path,
        required=True,
    )

    parser.add_argument(
        "--output",
        type=Path,
        required=True,
    )

    parser.add_argument(
        "--weather",
        nargs="+",
        default=None,
    )

    parser.add_argument(
        "--timeofday",
        nargs="+",
        default=None,
    )

    parser.add_argument(
        "--scene",
        nargs="+",
        default=None,
    )

    parser.add_argument(
        "--min_objects",
        type=int,
        default=1,
    )

    parser.add_argument(
        "--train_ratio",
        type=float,
        default=0.8,
    )
    
    parser.add_argument(
        "--test_size",
        type=int,
        default=5000,
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=42,
    )

    return parser.parse_args()


# ==========================================================
# Filtering
# ==========================================================

def filter_train_val(df, args):
    """
    Images used for training/validation.
    """

    logger.info(f"Original images : {len(df):,}")

    if args.weather is not None:
        df = df[df.weather.isin(args.weather)]

    if args.timeofday is not None:
        df = df[df.timeofday.isin(args.timeofday)]

    if args.scene is not None:
        df = df[df.scene.isin(args.scene)]

    df = df[df.num_objects >= args.min_objects]

    logger.info(f"Train/Val images : {len(df):,}")

    return df


def create_test_set(df, train_val_df, test_size, seed):
    """
    Create a representative test set from all remaining images.
    Sampling is proportional to weather × timeofday × scene.
    """

    test = df.loc[~df.image_path.isin(train_val_df.image_path)].copy()

    logger.info(f"Candidate test images: {len(test):,}")

    # Create stratification key
    test["stratum"] = (
        test["weather"].fillna("undefined").astype(str)
        + "_"
        + test["timeofday"].fillna("undefined").astype(str)
        + "_"
        + test["scene"].fillna("undefined").astype(str)
    )

    total = len(test)

    sampled = []

    for _, group in test.groupby("stratum"):

        proportion = len(group) / total
        n = max(1, round(proportion * test_size))

        sampled.append(
            group.sample(
                n=min(n, len(group)),
                random_state=seed,
            )
        )

    test = (
        pd.concat(sampled)
        .sample(frac=1, random_state=seed)
        .reset_index(drop=True)
    )

    # Ensure exactly test_size images
    if len(test) > test_size:
        test = test.iloc[:test_size]

    elif len(test) < test_size:
        remaining = df.loc[
            (~df.image_path.isin(train_val_df.image_path))
            & (~df.image_path.isin(test.image_path))
        ]

        extra = remaining.sample(
            n=test_size - len(test),
            random_state=seed,
        )

        test = pd.concat([test, extra]).reset_index(drop=True)

    test = test.drop(columns="stratum")

    logger.info(f"Final test images: {len(test):,}")

    return test


# ==========================================================
# Split
# ==========================================================

def split_dataset(df, ratio, seed):

    df = df.sample(
        frac=1,
        random_state=seed
    ).reset_index(drop=True)

    n_train = int(len(df) * ratio)

    train = df.iloc[:n_train]

    val = df.iloc[n_train:]

    return train, val


# ==========================================================
# Save
# ==========================================================

def save_split(train, val, test, output):

    output.mkdir(
        parents=True,
        exist_ok=True,
    )

    train.to_csv(
        output / "split_train.csv",
        index=False,
    )

    val.to_csv(
        output / "split_val.csv",
        index=False,
    )

    test.to_csv(
        output / "split_test.csv",
        index=False,
    )

    train["image_path"].to_csv(
        output / "train.txt",
        index=False,
        header=False,
    )

    val["image_path"].to_csv(
        output / "val.txt",
        index=False,
        header=False,
    )

    test["image_path"].to_csv(
        output / "test.txt",
        index=False,
        header=False,
    )

    with open(output / "split_summary.txt", "w") as f:

        f.write(f"Train : {len(train)}\n")
        f.write(f"Val   : {len(val)}\n")
        f.write(f"Test  : {len(test)}\n")

def print_dataset_statistics(
    train: pd.DataFrame,
    val: pd.DataFrame,
    test: pd.DataFrame,
):
    """
    Print detailed statistics for all dataset splits.
    """

    logger.info("")
    logger.info("=" * 80)
    logger.info("DATASET SPLIT SUMMARY")
    logger.info("=" * 80)

    total = len(train) + len(val) + len(test)

    logger.info(f"Train      : {len(train):6d} ({100*len(train)/total:5.2f}%)")
    logger.info(f"Validation : {len(val):6d} ({100*len(val)/total:5.2f}%)")
    logger.info(f"Test       : {len(test):6d} ({100*len(test)/total:5.2f}%)")
    logger.info(f"Total      : {total:6d}")

    datasets = {
        "TRAIN": train,
        "VALIDATION": val,
        "TEST": test,
    }

    categorical_columns = [
        "weather",
        "timeofday",
        "scene",
    ]

    object_columns = [
        "car",
        "pedestrian",
        "traffic light",
        "traffic sign",
    ]

    for split_name, df in datasets.items():

        logger.info("")
        logger.info("=" * 80)
        logger.info(split_name)
        logger.info("=" * 80)

        # --------------------------------------------------
        # Environment statistics
        # --------------------------------------------------

        for column in categorical_columns:

            logger.info("")
            logger.info(column.upper())

            summary = (
                df[column]
                .fillna("undefined")
                .value_counts()
                .rename_axis(column)
                .reset_index(name="Images")
            )

            summary["Percent"] = (
                summary["Images"]
                / summary["Images"].sum()
                * 100
            ).round(2)

            logger.info("\n%s", summary.to_string(index=False))

        # --------------------------------------------------
        # Object statistics
        # --------------------------------------------------

        logger.info("")
        logger.info("OBJECT COUNTS")

        rows = []

        for cls in object_columns:

            if cls in df.columns:

                rows.append(
                    {
                        "Class": cls,
                        "Objects": int(df[cls].sum()),
                    }
                )

        object_df = pd.DataFrame(rows)

        logger.info("\n%s", object_df.to_string(index=False))

        logger.info("")
        logger.info(
            "Average objects/image: %.2f",
            object_df["Objects"].sum() / len(df),
        )


# ==========================================================
# Main
# ==========================================================

def main():

    args = parse_args()

    logger.info("Loading metadata...")

    df = pd.read_csv(args.metadata)

    train_val_df = filter_train_val(df, args)

    train, val = split_dataset(
        train_val_df,
        args.train_ratio,
        args.seed,
    )

    test = create_test_set(
        df,
        train_val_df,
        args.test_size,
        args.seed,
    )

    logger.info(f"Train : {len(train):,}")
    logger.info(f"Val   : {len(val):,}")
    logger.info(f"Test  : {len(test):,}")

    save_split(
        train,
        val,
        test,
        args.output,
    )

    print_dataset_statistics(
        train,
        val,
        test,
    )

    logger.info("Done.")


if __name__ == "__main__":
    main()
