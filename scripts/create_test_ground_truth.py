"""
create_test_ground_truth.py

Extract ground-truth annotations for the BDD100K test split.

Input
-----
- det_v2_train_release.json
- split_test.csv

Output
------
test_ground_truth.csv

The output contains one row per ground-truth object.
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

import pandas as pd

# =============================================================================
# Constants
# =============================================================================

CLASS_MAPPING = {

    "pedestrian": 0,
    "car": 1,
    "traffic light": 2,
    "traffic sign": 3,

}

TARGET_CLASSES = set(CLASS_MAPPING.keys())


# =============================================================================
# Logging
# =============================================================================

logging.basicConfig(

    level=logging.INFO,

    format="%(asctime)s | %(levelname)s | %(message)s",

    datefmt="%H:%M:%S",

)

logger = logging.getLogger(__name__)


# =============================================================================
# Arguments
# =============================================================================

def parse_args():

    parser = argparse.ArgumentParser(

        description="Create ground-truth CSV for the test split."

    )

    parser.add_argument(

        "--annotations",

        type=Path,

        required=True,

        help="BDD100K detection JSON.",

    )

    parser.add_argument(

        "--test_csv",

        type=Path,

        required=True,

        help="split_test.csv",

    )

    parser.add_argument(

        "--output",

        type=Path,

        required=True,

        help="Output CSV.",

    )

    return parser.parse_args()


# =============================================================================
# Validation
# =============================================================================

def validate_inputs(args):

    for path in [

        args.annotations,

        args.test_csv,

    ]:

        if not path.exists():

            raise FileNotFoundError(path)


# =============================================================================
# Loading
# =============================================================================

def load_test_metadata(csv_path):

    logger.info("")
    logger.info("Loading test metadata...")

    df = pd.read_csv(csv_path)

    logger.info(
        f"Loaded {len(df):,} test images."
    )

    return df


def load_annotations(json_path):

    logger.info("")
    logger.info("Loading BDD100K annotations...")

    with open(json_path, "r") as f:

        annotations = json.load(f)

    logger.info(
        f"Loaded {len(annotations):,} annotated images."
    )

    return annotations


# =============================================================================
# Test image lookup
# =============================================================================

def build_test_lookup(test_metadata):
    """
    Create a dictionary:

    image_name ->
        weather
        timeofday
        scene
        image_path
    """

    lookup = {}

    for _, row in test_metadata.iterrows():

        image_name = Path(
            row["image_path"]
        ).name

        lookup[image_name] = {

            "image_path": row["image_path"],

            "weather": row["weather"],

            "timeofday": row["timeofday"],

            "scene": row["scene"],

        }

    logger.info(
        f"Indexed {len(lookup):,} test images."
    )

    return lookup


# =============================================================================
# Ground-truth extraction
# =============================================================================

def extract_ground_truth(
    annotations,
    test_lookup,
):
    """
    Extract ground-truth objects for the test split.
    """

    logger.info("")
    logger.info("Extracting test annotations...")

    rows = []

    skipped_images = 0
    processed_images = 0

    class_counts = {
        cls: 0
        for cls in TARGET_CLASSES
    }

    for image in annotations:

        image_name = image["name"]

        if image_name not in test_lookup:

            skipped_images += 1
            continue

        processed_images += 1

        metadata = test_lookup[image_name]

        labels = image.get("labels", [])

        for label in labels:

            category = label.get("category")

            if category not in TARGET_CLASSES:
                continue

            if "box2d" not in label:
                continue

            box = label["box2d"]

            x1 = float(box["x1"])
            y1 = float(box["y1"])
            x2 = float(box["x2"])
            y2 = float(box["y2"])

            if x2 <= x1 or y2 <= y1:
                continue

            rows.append({
                "object_id": len(rows),
                
                "image_id": image_name,

                "image_path": metadata["image_path"],

                "weather": metadata["weather"],

                "timeofday": metadata["timeofday"],

                "scene": metadata["scene"],

                "class_id": CLASS_MAPPING[category],

                "class_name": category,

                "x1": x1,
                "y1": y1,
                "x2": x2,
                "y2": y2,

            })

            class_counts[category] += 1

    logger.info("")
    logger.info("Extraction finished.")

    logger.info(f"Processed images : {processed_images:,}")
    logger.info(f"Skipped images   : {skipped_images:,}")
    logger.info(f"Objects          : {len(rows):,}")

    logger.info("")
    logger.info("Objects per class")

    for cls in sorted(class_counts):

        logger.info(
            f"{cls:20s} {class_counts[cls]:8,d}"
        )

    return rows


# =============================================================================
# Statistics
# =============================================================================

def compute_statistics(rows):
    """
    Compute simple dataset statistics.
    """

    df = pd.DataFrame(rows)

    logger.info("")
    logger.info("=" * 60)
    logger.info("DATASET STATISTICS")
    logger.info("=" * 60)

    logger.info(
        f"Images : {df['image_path'].nunique():,}"
    )

    logger.info(
        f"Objects: {len(df):,}"
    )

    logger.info("")

    logger.info("Weather")

    logger.info(
        df["weather"]
        .value_counts()
        .sort_values(ascending=False)
    )

    logger.info("")

    logger.info("Time of day")

    logger.info(
        df["timeofday"]
        .value_counts()
        .sort_values(ascending=False)
    )

    logger.info("")

    logger.info("Scene")

    logger.info(
        df["scene"]
        .value_counts()
        .sort_values(ascending=False)
    )

    return df


# =============================================================================
# Save
# =============================================================================

def save_ground_truth(
    df,
    output_path,
):
    """
    Save ground-truth annotations.
    """

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    df = df.sort_values(

        by=[
            "image_path",
            "class_id",
        ]

    ).reset_index(drop=True)

    df.to_csv(
        output_path,
        index=False,
    )

    logger.info("")
    logger.info(
        f"Ground truth saved to:\n{output_path}"
    )


# =============================================================================
# Summary
# =============================================================================

def print_summary(df):

    logger.info("")
    logger.info("=" * 60)
    logger.info("SUMMARY")
    logger.info("=" * 60)

    logger.info(
        f"Images : {df['image_path'].nunique():,}"
    )

    logger.info(
        f"Objects: {len(df):,}"
    )

    logger.info("")

    logger.info("Objects per class")

    counts = (
        df["class_name"]
        .value_counts()
        .sort_index()
    )

    for cls, count in counts.items():

        logger.info(
            f"{cls:20s}{count:8,d}"
        )

    logger.info("")

    logger.info("Weather distribution")

    logger.info(
        df["weather"]
        .value_counts()
    )

    logger.info("")

    logger.info("Time-of-day distribution")

    logger.info(
        df["timeofday"]
        .value_counts()
    )

    logger.info("")

    logger.info("Scene distribution")

    logger.info(
        df["scene"]
        .value_counts()
    )


# =============================================================================
# Main
# =============================================================================

def main():

    args = parse_args()

    validate_inputs(args)

    test_metadata = load_test_metadata(
        args.test_csv,
    )

    annotations = load_annotations(
        args.annotations,
    )

    test_lookup = build_test_lookup(
        test_metadata,
    )

    rows = extract_ground_truth(

        annotations,

        test_lookup,

    )

    df = compute_statistics(
        rows,
    )

    save_ground_truth(

        df,

        args.output,

    )

    print_summary(
        df,
    )

    logger.info("")
    logger.info(
        "Finished successfully."
    )


if __name__ == "__main__":

    main()
