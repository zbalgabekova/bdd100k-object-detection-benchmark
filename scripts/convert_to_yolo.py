"""
convert_to_yolo.py

Convert BDD100K detection annotations to YOLO format.

Features
--------
- Reads train/validation splits from CSV
- Converts BDD100K boxes to YOLO labels
- Copies images
- Generates dataset.yaml
- Supports custom class mappings
"""

from __future__ import annotations

import argparse
import json
import logging
import shutil
from pathlib import Path
from typing import Dict, List

import pandas as pd
from PIL import Image
from tqdm import tqdm


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
# Classes
# =============================================================================

# Change this dictionary if you want to train on different classes.

CLASS_MAPPING = {
    "pedestrian": 0,
    "car": 1,
    "traffic light": 2,
    "traffic sign": 3,
}

VALID_EXTENSIONS = [".jpg", ".jpeg", ".png"]


# =============================================================================
# Arguments
# =============================================================================

def parse_args():

    parser = argparse.ArgumentParser(
        description="Convert BDD100K annotations to YOLO."
    )

    parser.add_argument(
        "--train_csv",
        type=Path,
        required=True,
        help="split_train.csv",
    )

    parser.add_argument(
        "--val_csv",
        type=Path,
        required=True,
        help="split_val.csv",
    )
    
    parser.add_argument(
        "--test_csv",
        type=Path,
        required=True,
        help="split_test.csv",
    )

    parser.add_argument(
        "--labels",
        type=Path,
        required=True,
        help="det_v2_train_release.json",
    )

    parser.add_argument(
        "--images",
        type=Path,
        required=True,
        help="100k/train directory",
    )

    parser.add_argument(
        "--output",
        type=Path,
        default=Path("dataset"),
    )

    parser.add_argument(
        "--copy_images",
        action="store_true",
        help="Copy images into YOLO dataset.",
    )

    return parser.parse_args()


# =============================================================================
# Validation
# =============================================================================

def validate_inputs(args):

    if not args.train_csv.exists():
        raise FileNotFoundError(args.train_csv)

    if not args.val_csv.exists():
        raise FileNotFoundError(args.val_csv)

    if not args.labels.exists():
        raise FileNotFoundError(args.labels)

    if not args.images.exists():
        raise FileNotFoundError(args.images)
        
    if not args.test_csv.exists():
        raise FileNotFoundError(args.test_csv)


# =============================================================================
# Annotation loading
# =============================================================================

def load_annotation_index(json_path: Path) -> Dict[str, dict]:
    """
    Build

        image_id -> annotation

    dictionary.
    """

    logger.info("Loading annotations...")

    with open(json_path, "r") as f:
        annotations = json.load(f)

    annotation_index = {}

    for ann in annotations:

        image_id = Path(ann["name"]).stem

        annotation_index[image_id] = ann

    logger.info(f"Indexed {len(annotation_index):,} images.")

    return annotation_index


# =============================================================================
# Image helpers
# =============================================================================

def find_image(images_dir: Path, image_id: str) -> Path | None:

    for ext in VALID_EXTENSIONS:

        candidate = images_dir / f"{image_id}{ext}"

        if candidate.exists():
            return candidate

    return None


def load_image_size(image_path: Path):

    with Image.open(image_path) as img:
        return img.width, img.height


# =============================================================================
# YOLO conversion
# =============================================================================

def convert_box(box, image_width, image_height):
    """
    Convert

        x1,y1,x2,y2

    to

        x_center
        y_center
        width
        height

    normalized to [0,1].
    """

    x1 = box["x1"]
    y1 = box["y1"]

    x2 = box["x2"]
    y2 = box["y2"]

    w = x2 - x1
    h = y2 - y1

    x = x1 + w / 2
    y = y1 + h / 2

    return (
        x / image_width,
        y / image_height,
        w / image_width,
        h / image_height,
    )


# =============================================================================
# Directory creation
# =============================================================================

def create_directories(output_dir: Path):

    dirs = {

        "train_images": output_dir / "images" / "train",
        "val_images": output_dir / "images" / "val",
        "test_images": output_dir / "images" / "test",

        "train_labels": output_dir / "labels" / "train",
        "val_labels": output_dir / "labels" / "val",
        "test_labels": output_dir / "labels" / "test",
        }

    for d in dirs.values():
        d.mkdir(parents=True, exist_ok=True)

    return dirs



# =============================================================================
# Conversion
# =============================================================================

def convert_split(
    split_csv: Path,
    annotation_index: Dict[str, dict],
    images_dir: Path,
    image_output_dir: Path,
    label_output_dir: Path,
    copy_images: bool,
):
    """
    Convert one split (train or val).

    Returns
    -------
    dict
        Statistics for this split.
    """

    df = pd.read_csv(split_csv)

    logger.info("")
    logger.info(f"Processing {split_csv.name}")
    logger.info(f"Images: {len(df):,}")

    stats = {
        "images": 0,
        "missing_images": 0,
        "missing_annotations": 0,
        "copied_images": 0,
        "objects_total": 0,
        "objects_converted": 0,
        "objects_skipped": 0,
        "empty_labels": 0,
    }

    for _, row in tqdm(df.iterrows(), total=len(df)):

        image_id = row["image"]

        image_path = find_image(images_dir, image_id)

        if image_path is None:
            stats["missing_images"] += 1
            continue

        ann = annotation_index.get(image_id)

        if ann is None:
            stats["missing_annotations"] += 1
            continue

        width, height = load_image_size(image_path)

        label_file = label_output_dir / f"{image_id}.txt"

        yolo_lines = []

        labels = ann.get("labels") or []

        for obj in labels:

            stats["objects_total"] += 1

            category = obj.get("category")

            if category not in CLASS_MAPPING:
                stats["objects_skipped"] += 1
                continue

            box = obj.get("box2d")

            if box is None:
                stats["objects_skipped"] += 1
                continue

            x1 = max(0.0, min(box["x1"], width))
            y1 = max(0.0, min(box["y1"], height))
            x2 = max(0.0, min(box["x2"], width))
            y2 = max(0.0, min(box["y2"], height))

            if x2 <= x1 or y2 <= y1:
                stats["objects_skipped"] += 1
                continue

            x, y, w, h = convert_box(
                {
                    "x1": x1,
                    "y1": y1,
                    "x2": x2,
                    "y2": y2,
                },
                width,
                height,
            )

            cls = CLASS_MAPPING[category]

            yolo_lines.append(
                f"{cls} {x:.6f} {y:.6f} {w:.6f} {h:.6f}"
            )

            stats["objects_converted"] += 1

        with open(label_file, "w") as f:

            for line in yolo_lines:
                f.write(line + "\n")

        if len(yolo_lines) == 0:
            stats["empty_labels"] += 1

        if copy_images:

            shutil.copy2(
                image_path,
                image_output_dir / image_path.name,
            )

            stats["copied_images"] += 1

        stats["images"] += 1

    return stats


# =============================================================================
# Dataset YAML
# =============================================================================

def write_dataset_yaml(output_dir: Path):
    """
    Create dataset.yaml for Ultralytics YOLO.
    """

    yaml_path = output_dir / "dataset.yaml"

    with open(yaml_path, "w") as f:

        f.write(f"path: {output_dir.resolve()}\n")
        f.write("train: images/train\n")
        f.write("val: images/val\n")
        f.write("test: images/test\n\n")

        f.write("names:\n")

        for name, idx in sorted(
            CLASS_MAPPING.items(),
            key=lambda x: x[1],
        ):
            f.write(f"  {idx}: {name}\n")

    logger.info(f"Saved {yaml_path}")
    
    
# =============================================================================
# Statistics
# =============================================================================

def print_statistics(name: str, stats: dict):

    logger.info("")
    logger.info("=" * 60)
    logger.info(name.upper())
    logger.info("=" * 60)

    logger.info(f"Images processed      : {stats['images']:,}")
    logger.info(f"Images copied         : {stats['copied_images']:,}")

    logger.info(f"Missing images        : {stats['missing_images']:,}")
    logger.info(f"Missing annotations   : {stats['missing_annotations']:,}")

    logger.info(f"Objects total         : {stats['objects_total']:,}")
    logger.info(f"Objects converted     : {stats['objects_converted']:,}")
    logger.info(f"Objects skipped       : {stats['objects_skipped']:,}")

    logger.info(f"Empty label files     : {stats['empty_labels']:,}")

    if stats["objects_total"] > 0:

        ratio = (
            100.0
            * stats["objects_converted"]
            / stats["objects_total"]
        )

        logger.info(f"Conversion rate       : {ratio:.2f}%")


def combine_statistics(*stats_list):

    total = {}

    for stats in stats_list:

        for key, value in stats.items():

            total[key] = total.get(key, 0) + value

    return total


# =============================================================================
# Main
# =============================================================================

def main():

    args = parse_args()

    validate_inputs(args)

    logger.info("=" * 60)
    logger.info("BDD100K → YOLO CONVERTER")
    logger.info("=" * 60)

    annotation_index = load_annotation_index(args.labels)

    dirs = create_directories(args.output)

    # ---------------------------------------------------------
    # Train
    # ---------------------------------------------------------

    train_stats = convert_split(
        split_csv=args.train_csv,
        annotation_index=annotation_index,
        images_dir=args.images,
        image_output_dir=dirs["train_images"],
        label_output_dir=dirs["train_labels"],
        copy_images=args.copy_images,
    )

    # ---------------------------------------------------------
    # Validation
    # ---------------------------------------------------------

    val_stats = convert_split(
        split_csv=args.val_csv,
        annotation_index=annotation_index,
        images_dir=args.images,
        image_output_dir=dirs["val_images"],
        label_output_dir=dirs["val_labels"],
        copy_images=args.copy_images,
    )
    
    
    # ---------------------------------------------------------
    # Test
    # ---------------------------------------------------------

    test_stats = convert_split(
        split_csv=args.test_csv,
        annotation_index=annotation_index,
        images_dir=args.images,
        image_output_dir=dirs["test_images"],
        label_output_dir=dirs["test_labels"],
        copy_images=args.copy_images,
    )

    # ---------------------------------------------------------
    # YAML
    # ---------------------------------------------------------

    write_dataset_yaml(args.output)

    # ---------------------------------------------------------
    # Statistics
    # ---------------------------------------------------------

    total_stats = combine_statistics(
        train_stats,
        val_stats,
        test_stats,
    )

    print_statistics(
        "Training split",
        train_stats,
    )

    print_statistics(
        "Validation split",
        val_stats,
    )
    
    print_statistics(
        "Test split",
        test_stats,
    )

    print_statistics(
        "Overall",
        total_stats,
    )

    logger.info("")
    logger.info("Class mapping")

    for cls, idx in sorted(
        CLASS_MAPPING.items(),
        key=lambda x: x[1],
    ):
        logger.info(f"{idx:>2} -> {cls}")

    logger.info("")
    logger.info("Dataset created successfully.")

    logger.info(f"Output directory : {args.output.resolve()}")

    logger.info("")
    logger.info("Ready for YOLO training.")


if __name__ == "__main__":
    main()