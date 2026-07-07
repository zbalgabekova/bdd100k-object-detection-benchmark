"""
prepare_dataset.py

Prepare the BDD100K dataset for object detection experiments.

This script:

- Loads BDD100K detection annotations
- Computes dataset statistics
- Generates metadata.csv
- Produces distribution plots
- Serves as the foundation for dataset splitting and YOLO conversion

Author: Zarema Balgabekova
"""

from __future__ import annotations

import argparse
import json
import logging
from collections import Counter
from pathlib import Path
from statistics import mean, median
from typing import Dict, List, Optional

import matplotlib.pyplot as plt
import pandas as pd
from PIL import Image
from tqdm import tqdm


# =============================================================================
# Logging
# =============================================================================

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    datefmt="%H:%M:%S"
)

logger = logging.getLogger(__name__)


# =============================================================================
# Constants
# =============================================================================

VALID_IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png"}


# =============================================================================
# Helper functions
# =============================================================================

def load_annotations(path: Path) -> List[dict]:
    """
    Load BDD100K annotation JSON.

    Parameters
    ----------
    path : Path

    Returns
    -------
    list
    """

    logger.info("Loading annotations...")

    with open(path, "r") as f:
        annotations = json.load(f)

    logger.info(f"Loaded {len(annotations):,} annotations.")

    return annotations


def get_image_ids(images_dir: Path) -> set[str]:
    """
    Return all image ids (without extension).
    """

    image_ids = {
        p.stem
        for p in images_dir.iterdir()
        if p.suffix.lower() in VALID_IMAGE_EXTENSIONS
    }

    logger.info(f"Found {len(image_ids):,} images.")

    return image_ids


def load_image_size(image_path: Path) -> tuple[Optional[int], Optional[int]]:
    """
    Safely load image dimensions.
    """

    try:

        with Image.open(image_path) as img:
            return img.width, img.height

    except Exception:

        return None, None


def compute_box_statistics(
    widths: List[float],
    heights: List[float],
    areas: List[float]
) -> Dict[str, Optional[float]]:
    """
    Compute bounding box statistics for one image.
    """

    if len(areas) == 0:

        return {

            "avg_box_width": None,
            "avg_box_height": None,

            "avg_box_area": None,
            "median_box_area": None,

            "min_box_area": None,
            "max_box_area": None
        }

    return {

        "avg_box_width": mean(widths),
        "avg_box_height": mean(heights),

        "avg_box_area": mean(areas),
        "median_box_area": median(areas),

        "min_box_area": min(areas),
        "max_box_area": max(areas)
    }


def plot_counter(
    counter: Counter,
    title: str,
    xlabel: str,
    ylabel: str,
    save_path: Path
) -> None:
    """
    Save a bar chart from Counter.
    """

    if len(counter) == 0:
        return

    labels = list(counter.keys())
    values = list(counter.values())

    plt.figure(figsize=(10, 5))

    plt.bar(labels, values)

    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)

    plt.xticks(rotation=30)

    plt.tight_layout()

    plt.savefig(save_path, dpi=200)

    plt.close()


def print_counter(counter: Counter, title: str) -> None:
    """
    Nicely print Counter contents.
    """

    logger.info("")
    logger.info(title)
    logger.info("-" * len(title))

    for key, value in counter.most_common():

        logger.info(f"{key:<20} {value:>8}")


def verify_dataset(
    annotations: List[dict],
    image_ids: set[str]
) -> None:
    """
    Verify that annotations correspond to existing images.
    """

    annotation_ids = {
        Path(a["name"]).stem
        for a in annotations
    }

    common = annotation_ids & image_ids

    logger.info("")
    logger.info("Dataset verification")
    logger.info("--------------------")

    logger.info(f"Images             : {len(image_ids):,}")
    logger.info(f"Annotations        : {len(annotation_ids):,}")
    logger.info(f"Matching           : {len(common):,}")
    logger.info(f"Missing images     : {len(annotation_ids-image_ids):,}")
    logger.info(f"Missing annotations: {len(image_ids-annotation_ids):,}")


def bbox_area_ratio(
    avg_area: Optional[float],
    width: Optional[int],
    height: Optional[int]
) -> Optional[float]:
    """
    Compute average bounding box area relative to image area.
    """

    if avg_area is None:
        return None

    if width is None or height is None:
        return None

    image_area = width * height

    if image_area == 0:
        return None

    return avg_area / image_area
    
    
    
# =============================================================================
# Annotation processing
# =============================================================================

def process_annotations(
    annotations: List[dict],
    images_dir: Path,
    image_ids: set[str],
):
    """
    Process all annotations and collect metadata.

    Returns
    -------
    metadata_rows : list[dict]
    global_stats : dict
    """

    metadata_rows = []

    weather_counter = Counter()
    time_counter = Counter()
    scene_counter = Counter()
    class_counter = Counter()

    dataset_bbox_widths = []
    dataset_bbox_heights = []
    dataset_bbox_areas = []

    missing_images = 0
    processed_images = 0

    logger.info("")
    logger.info("Processing annotations...")

    for ann in tqdm(annotations):

        image_id = Path(ann["name"]).stem

        # ---------------------------------------------------------
        # Skip annotations without corresponding image
        # ---------------------------------------------------------

        if image_id not in image_ids:
            missing_images += 1
            continue

        image_path = None

        for ext in VALID_IMAGE_EXTENSIONS:
            candidate = images_dir / f"{image_id}{ext}"
            if candidate.exists():
                image_path = candidate
                break

        if image_path is None:
            missing_images += 1
            continue

        if not image_path.exists():
            missing_images += 1
            continue

        processed_images += 1

        # ---------------------------------------------------------
        # Image attributes
        # ---------------------------------------------------------

        attrs = ann.get("attributes") or {}

        weather = attrs.get("weather", "unknown")
        timeofday = attrs.get("timeofday", "unknown")
        scene = attrs.get("scene", "unknown")

        weather_counter[weather] += 1
        time_counter[timeofday] += 1
        scene_counter[scene] += 1

        width, height = load_image_size(image_path)

        # ---------------------------------------------------------
        # Objects
        # ---------------------------------------------------------

        labels = ann.get("labels") or []

        object_counter = Counter()

        image_box_widths = []
        image_box_heights = []
        image_box_areas = []

        for obj in (labels or []):

            category = obj.get("category")

            if category is None:
                continue

            object_counter[category] += 1
            class_counter[category] += 1

            box = obj.get("box2d")

            if box is None:
                continue

            w = box["x2"] - box["x1"]
            h = box["y2"] - box["y1"]

            if w <= 0 or h <= 0:
                continue

            area = w * h

            image_box_widths.append(w)
            image_box_heights.append(h)
            image_box_areas.append(area)

            dataset_bbox_widths.append(w)
            dataset_bbox_heights.append(h)
            dataset_bbox_areas.append(area)

        # ---------------------------------------------------------
        # Bounding-box statistics
        # ---------------------------------------------------------

        bbox_stats = compute_box_statistics(
            image_box_widths,
            image_box_heights,
            image_box_areas,
        )

        # ---------------------------------------------------------
        # Metadata row
        # ---------------------------------------------------------

        row = {

            "image": image_id,

            "image_path": str(image_path),

            "width": width,
            "height": height,

            "weather": weather,
            "timeofday": timeofday,
            "scene": scene,

            "num_objects": sum(object_counter.values()),

            "avg_box_width": bbox_stats["avg_box_width"],
            "avg_box_height": bbox_stats["avg_box_height"],

            "avg_box_area": bbox_stats["avg_box_area"],
            "median_box_area": bbox_stats["median_box_area"],

            "min_box_area": bbox_stats["min_box_area"],
            "max_box_area": bbox_stats["max_box_area"],

            "avg_box_area_ratio": bbox_area_ratio(
                bbox_stats["avg_box_area"],
                width,
                height,
            ),
        }

        # ---------------------------------------------------------
        # Dynamic class columns
        # ---------------------------------------------------------

        for cls, count in object_counter.items():

            row[cls] = count
            row[f"has_{cls}"] = True

        metadata_rows.append(row)

    # -------------------------------------------------------------
    # Global statistics
    # -------------------------------------------------------------

    global_stats = {

        "weather": weather_counter,
        "timeofday": time_counter,
        "scene": scene_counter,
        "classes": class_counter,

        "bbox_widths": dataset_bbox_widths,
        "bbox_heights": dataset_bbox_heights,
        "bbox_areas": dataset_bbox_areas,

        "processed_images": processed_images,
        "missing_images": missing_images,
    }

    return metadata_rows, global_stats
    
    
    
 # =============================================================================
# Metadata
# =============================================================================

def build_metadata_dataframe(metadata_rows: List[dict]) -> pd.DataFrame:
    """
    Convert metadata rows into a clean pandas DataFrame.

    Missing object counts become 0.
    Missing has_* columns become False.
    """

    logger.info("")
    logger.info("Building metadata dataframe...")

    df = pd.DataFrame(metadata_rows)

    if df.empty:
        raise RuntimeError("No metadata was generated.")

    # ---------------------------------------------------------
    # Discover dynamic columns
    # ---------------------------------------------------------

    has_columns = [c for c in df.columns if c.startswith("has_")]

    fixed_columns = {
        "image",
        "image_path",
        "width",
        "height",
        "weather",
        "timeofday",
        "scene",
        "num_objects",
        "avg_box_width",
        "avg_box_height",
        "avg_box_area",
        "median_box_area",
        "min_box_area",
        "max_box_area",
        "avg_box_area_ratio",
    }

    class_columns = [
        c
        for c in df.columns
        if c not in fixed_columns
        and not c.startswith("has_")
    ]

    # ---------------------------------------------------------
    # Fill missing values
    # ---------------------------------------------------------

    for c in class_columns:
        df[c] = df[c].fillna(0).astype(int)

    for c in has_columns:
        df[c] = df[c].fillna(False).astype(bool)

    # ---------------------------------------------------------
    # Sort columns
    # ---------------------------------------------------------

    ordered_columns = [
        "image",
        "image_path",
        "width",
        "height",
        "weather",
        "timeofday",
        "scene",
        "num_objects",
        "avg_box_width",
        "avg_box_height",
        "avg_box_area",
        "median_box_area",
        "min_box_area",
        "max_box_area",
        "avg_box_area_ratio",
    ]

    ordered_columns.extend(sorted(class_columns))
    ordered_columns.extend(sorted(has_columns))

    df = df[ordered_columns]

    df = df.sort_values("image").reset_index(drop=True)

    logger.info(f"Metadata rows: {len(df):,}")

    return df


# =============================================================================
# Saving
# =============================================================================

def save_metadata(df: pd.DataFrame, output_dir: Path) -> None:
    """
    Save metadata CSV.
    """

    output_path = output_dir / "metadata.csv"

    df.to_csv(output_path, index=False)

    logger.info(f"Saved {output_path}")


# =============================================================================
# Plots
# =============================================================================

def save_plots(global_stats: dict, output_dir: Path) -> None:
    """
    Save distribution plots.
    """

    logger.info("")
    logger.info("Saving plots...")

    plot_counter(
        global_stats["weather"],
        "Weather Distribution",
        "Weather",
        "Images",
        output_dir / "weather.png",
    )

    plot_counter(
        global_stats["timeofday"],
        "Time of Day Distribution",
        "Time of Day",
        "Images",
        output_dir / "timeofday.png",
    )

    plot_counter(
        global_stats["scene"],
        "Scene Distribution",
        "Scene",
        "Images",
        output_dir / "scene.png",
    )

    plot_counter(
        global_stats["classes"],
        "Object Class Distribution",
        "Class",
        "Instances",
        output_dir / "classes.png",
    )

    logger.info("Plots saved.")


# =============================================================================
# Summary
# =============================================================================

def print_summary(df: pd.DataFrame, global_stats: dict) -> None:
    """
    Print dataset summary.
    """

    logger.info("")
    logger.info("=" * 60)
    logger.info("BDD100K DATASET SUMMARY")
    logger.info("=" * 60)

    logger.info(f"Images processed : {global_stats['processed_images']:,}")
    logger.info(f"Images skipped   : {global_stats['missing_images']:,}")
    logger.info(f"Metadata rows    : {len(df):,}")

    logger.info("")
    logger.info(f"Average objects/image : {df['num_objects'].mean():.2f}")
    logger.info(f"Median objects/image  : {df['num_objects'].median():.2f}")

    bbox_areas = global_stats["bbox_areas"]
    bbox_widths = global_stats["bbox_widths"]
    bbox_heights = global_stats["bbox_heights"]

    if bbox_areas:

        logger.info("")
        logger.info("Bounding Boxes")
        logger.info("------------------------")

        logger.info(f"Total boxes     : {len(bbox_areas):,}")
        logger.info(f"Average area    : {mean(bbox_areas):.1f}")
        logger.info(f"Median area     : {median(bbox_areas):.1f}")

        logger.info(f"Average width   : {mean(bbox_widths):.1f}")
        logger.info(f"Average height  : {mean(bbox_heights):.1f}")

        logger.info(f"Smallest box    : {min(bbox_areas):.1f}")
        logger.info(f"Largest box     : {max(bbox_areas):.1f}")

    print_counter(global_stats["weather"], "Weather")

    print_counter(global_stats["timeofday"], "Time of Day")

    print_counter(global_stats["scene"], "Scene")

    logger.info("")
    logger.info("Top 20 Object Classes")
    logger.info("---------------------")

    for cls, count in global_stats["classes"].most_common(20):
        logger.info(f"{cls:<25} {count:>10}")

    logger.info("")
    logger.info("=" * 60)
    
    
    
# =============================================================================
# Main
# =============================================================================

def parse_args() -> argparse.Namespace:
    """
    Parse command-line arguments.
    """

    parser = argparse.ArgumentParser(
        description="Prepare the BDD100K dataset."
    )

    parser.add_argument(
        "--images",
        type=Path,
        required=True,
        help="Path to image directory (e.g. 100k/train)"
    )

    parser.add_argument(
        "--labels",
        type=Path,
        required=True,
        help="Path to det_v2_train_release.json"
    )

    parser.add_argument(
        "--output",
        type=Path,
        default=Path("outputs"),
        help="Output directory"
    )

    return parser.parse_args()


def validate_inputs(images_dir: Path, labels_file: Path) -> None:
    """
    Validate user input paths.
    """

    if not images_dir.exists():
        raise FileNotFoundError(
            f"Images directory not found:\n{images_dir}"
        )

    if not images_dir.is_dir():
        raise NotADirectoryError(images_dir)

    if not labels_file.exists():
        raise FileNotFoundError(
            f"Annotation file not found:\n{labels_file}"
        )


def main() -> None:

    args = parse_args()

    args.output.mkdir(parents=True, exist_ok=True)

    validate_inputs(args.images, args.labels)

    logger.info("=" * 60)
    logger.info("BDD100K PREPARATION")
    logger.info("=" * 60)

    # ---------------------------------------------------------
    # Load data
    # ---------------------------------------------------------

    annotations = load_annotations(args.labels)

    image_ids = get_image_ids(args.images)

    verify_dataset(
        annotations,
        image_ids,
    )

    # ---------------------------------------------------------
    # Process dataset
    # ---------------------------------------------------------

    metadata_rows, global_stats = process_annotations(
        annotations=annotations,
        images_dir=args.images,
        image_ids=image_ids,
    )

    # ---------------------------------------------------------
    # Metadata
    # ---------------------------------------------------------

    df = build_metadata_dataframe(metadata_rows)

    save_metadata(
        df,
        args.output,
    )

    # ---------------------------------------------------------
    # Plots
    # ---------------------------------------------------------

    save_plots(
        global_stats,
        args.output,
    )

    # ---------------------------------------------------------
    # Summary
    # ---------------------------------------------------------

    print_summary(
        df,
        global_stats,
    )

    logger.info("")
    logger.info("Finished successfully.")
    logger.info(f"Output directory: {args.output.resolve()}")


if __name__ == "__main__":
    main()