"""
verify_yolo_dataset.py

Validate a YOLO dataset generated from BDD100K.

Features
--------
- Verify folder structure
- Check image/label consistency
- Validate YOLO labels
- Generate statistics
- Visualize random samples
"""

from __future__ import annotations

import argparse
import json
import logging
import random
from collections import Counter
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.patches as patches
import yaml
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
# Arguments
# =============================================================================

def parse_args():

    parser = argparse.ArgumentParser(
        description="Verify a YOLO dataset."
    )

    parser.add_argument(
        "--dataset",
        type=Path,
        required=True,
        help="Path to dataset directory",
    )

    parser.add_argument(
        "--output",
        type=Path,
        default=Path("verification"),
        help="Output directory",
    )

    parser.add_argument(
        "--samples",
        type=int,
        default=9,
        help="Number of random visualization samples",
    )

    parser.add_argument(
        "--strict",
        action="store_true",
        help="Exit immediately if an error is found.",
    )

    return parser.parse_args()


# =============================================================================
# Folder validation
# =============================================================================

def verify_structure(dataset_dir: Path):

    required = [
        dataset_dir / "dataset.yaml",

        dataset_dir / "images",
        dataset_dir / "images" / "train",
        dataset_dir / "images" / "val",
        dataset_dir / "images" / "test",


        dataset_dir / "labels",
        dataset_dir / "labels" / "train",
        dataset_dir / "labels" / "val",
        dataset_dir / "labels" / "test",
    ]

    missing = []

    for path in required:

        if not path.exists():
            missing.append(path)

    if missing:

        logger.error("Dataset structure is incomplete.")

        for path in missing:
            logger.error(f"Missing: {path}")

        raise FileNotFoundError("Dataset structure is invalid.")

    logger.info("Dataset structure verified.")


# =============================================================================
# Dataset YAML
# =============================================================================

def read_dataset_yaml(dataset_dir: Path):

    yaml_file = dataset_dir / "dataset.yaml"

    with open(yaml_file, "r") as f:
        config = yaml.safe_load(f)

    names = config.get("names")

    if names is None:
        raise ValueError("dataset.yaml does not contain 'names'.")

    if isinstance(names, list):

        class_names = names

    elif isinstance(names, dict):

        class_names = [
            names[k]
            for k in sorted(names.keys())
        ]

    else:
        raise ValueError("Unsupported 'names' format.")

    logger.info(f"Detected {len(class_names)} classes.")

    return class_names


# =============================================================================
# File indexing
# =============================================================================

def index_split(directory: Path, extensions):

    files = {}

    for ext in extensions:

        for file in directory.glob(f"*{ext}"):

            files[file.stem] = file

    return files


def index_dataset(dataset_dir: Path):

    image_exts = [".jpg", ".jpeg", ".png"]
    label_exts = [".txt"]

    train_images = index_split(
        dataset_dir / "images" / "train",
        image_exts,
    )

    val_images = index_split(
        dataset_dir / "images" / "val",
        image_exts,
    )
    
    test_images = index_split(
        dataset_dir / "images" / "test",
        image_exts,
    )

    train_labels = index_split(
        dataset_dir / "labels" / "train",
        label_exts,
    )

    val_labels = index_split(
        dataset_dir / "labels" / "val",
        label_exts,
    )
    
    test_labels = index_split(
        dataset_dir / "labels" / "test",
        label_exts,
    )

    logger.info(
        f"Train images : {len(train_images):,}"
    )

    logger.info(
        f"Validation images : {len(val_images):,}"
    )
    
    logger.info(
        f"Test images : {len(test_images):,}"
    )

    return {

        "train_images": train_images,
        "val_images": val_images,
        "test_images": test_images,

        "train_labels": train_labels,
        "val_labels": val_labels,
        "test_labels": test_labels,
    }


# =============================================================================
# Image ↔ Label consistency
# =============================================================================

def compare_sets(images, labels):

    image_ids = set(images.keys())
    label_ids = set(labels.keys())

    missing_labels = image_ids - label_ids
    missing_images = label_ids - image_ids

    matching = image_ids & label_ids

    return {

        "matching": matching,

        "missing_labels": missing_labels,

        "missing_images": missing_images,
    }


def verify_matching(index):

    train = compare_sets(
        index["train_images"],
        index["train_labels"],
    )

    val = compare_sets(
        index["val_images"],
        index["val_labels"],
    )
    
    test = compare_sets(
        index["test_images"],
        index["test_labels"],
    )

    logger.info("")
    logger.info("Train split")

    logger.info(f"Matching         : {len(train['matching']):,}")
    logger.info(f"Missing labels   : {len(train['missing_labels']):,}")
    logger.info(f"Missing images   : {len(train['missing_images']):,}")

    logger.info("")
    logger.info("Validation split")

    logger.info(f"Matching         : {len(val['matching']):,}")
    logger.info(f"Missing labels   : {len(val['missing_labels']):,}")
    logger.info(f"Missing images   : {len(val['missing_images']):,}")
    
    logger.info("")
    logger.info("Test split")

    logger.info(f"Matching         : {len(test['matching']):,}")
    logger.info(f"Missing labels   : {len(test['missing_labels']):,}")
    logger.info(f"Missing images   : {len(test['missing_images']):,}")

    return train, val, test


# =============================================================================
# Label validation
# =============================================================================

def validate_split(
    image_index: dict,
    label_index: dict,
    class_names: list,
):

    stats = {

        "images": len(image_index),

        "empty_labels": 0,

        "objects": 0,

        "invalid_files": 0,

        "invalid_lines": 0,

        "invalid_boxes": 0,

        "invalid_classes": 0,

        "objects_per_class": Counter(),

        "objects_per_image": [],

        "errors": [],
    }

    for image_id in tqdm(sorted(image_index.keys())):

        label_file = label_index.get(image_id)

        if label_file is None:
            continue

        object_count = 0

        with open(label_file, "r") as f:

            lines = [
                line.strip()
                for line in f.readlines()
                if line.strip()
            ]

        if len(lines) == 0:

            stats["empty_labels"] += 1
            stats["objects_per_image"].append(0)

            continue

        for line_number, line in enumerate(lines, start=1):

            parts = line.split()

            # -----------------------------------------------------
            # Format
            # -----------------------------------------------------

            if len(parts) != 5:

                stats["invalid_lines"] += 1

                stats["errors"].append(

                    f"{label_file.name}:{line_number} "
                    f"Expected 5 values, got {len(parts)}"

                )

                continue

            # -----------------------------------------------------
            # Parse numbers
            # -----------------------------------------------------

            try:

                class_id = int(parts[0])

                x = float(parts[1])
                y = float(parts[2])

                w = float(parts[3])
                h = float(parts[4])

            except ValueError:

                stats["invalid_lines"] += 1

                stats["errors"].append(

                    f"{label_file.name}:{line_number} "
                    f"Non-numeric value"

                )

                continue

            # -----------------------------------------------------
            # Class ID
            # -----------------------------------------------------

            if class_id < 0 or class_id >= len(class_names):

                stats["invalid_classes"] += 1

                stats["errors"].append(

                    f"{label_file.name}:{line_number} "
                    f"Invalid class {class_id}"

                )

                continue

            # -----------------------------------------------------
            # Coordinates
            # -----------------------------------------------------

            if not (0.0 <= x <= 1.0):

                stats["invalid_boxes"] += 1

                stats["errors"].append(

                    f"{label_file.name}:{line_number} "
                    f"x_center={x}"

                )

                continue

            if not (0.0 <= y <= 1.0):

                stats["invalid_boxes"] += 1

                stats["errors"].append(

                    f"{label_file.name}:{line_number} "
                    f"y_center={y}"

                )

                continue

            if not (0.0 < w <= 1.0):

                stats["invalid_boxes"] += 1

                stats["errors"].append(

                    f"{label_file.name}:{line_number} "
                    f"width={w}"

                )

                continue

            if not (0.0 < h <= 1.0):

                stats["invalid_boxes"] += 1

                stats["errors"].append(

                    f"{label_file.name}:{line_number} "
                    f"height={h}"

                )

                continue


            # -----------------------------------------------------
            # Statistics
            # -----------------------------------------------------

            stats["objects"] += 1

            object_count += 1

            stats["objects_per_class"][
                class_names[class_id]
            ] += 1

        stats["objects_per_image"].append(
            object_count
        )

    return stats


# =============================================================================
# Statistics
# =============================================================================

def summarize_statistics(stats):

    if len(stats["objects_per_image"]) == 0:

        stats["avg_objects"] = 0

        stats["max_objects"] = 0

        stats["min_objects"] = 0

    else:

        stats["avg_objects"] = round(

            sum(stats["objects_per_image"])
            / len(stats["objects_per_image"]),

            2,

        )

        stats["max_objects"] = max(

            stats["objects_per_image"]

        )

        stats["min_objects"] = min(

            stats["objects_per_image"]

        )

    return stats


# =============================================================================
# Console report
# =============================================================================

def print_statistics(title, stats):

    logger.info("")
    logger.info("=" * 60)
    logger.info(title.upper())
    logger.info("=" * 60)

    logger.info(f"Images              : {stats['images']:,}")

    logger.info(f"Objects             : {stats['objects']:,}")

    logger.info(f"Average/image       : {stats['avg_objects']}")

    logger.info(f"Maximum/image       : {stats['max_objects']}")

    logger.info(f"Minimum/image       : {stats['min_objects']}")

    logger.info(f"Empty labels        : {stats['empty_labels']:,}")

    logger.info(f"Invalid classes     : {stats['invalid_classes']:,}")

    logger.info(f"Invalid boxes       : {stats['invalid_boxes']:,}")

    logger.info(f"Invalid lines       : {stats['invalid_lines']:,}")

    logger.info("")

    logger.info("Class distribution")

    for cls, count in sorted(

        stats["objects_per_class"].items(),

        key=lambda x: x[1],

        reverse=True,

    ):

        logger.info(f"{cls:<20} {count:>10,}")
        
        
# =============================================================================
# Visualization
# =============================================================================

def visualize_samples(
    image_index: dict,
    label_index: dict,
    class_names: list,
    output_dir: Path,
    num_samples: int = 9,
):

    output_dir.mkdir(parents=True, exist_ok=True)

    available = list(image_index.keys())

    if len(available) == 0:
        logger.warning("No images available for visualization.")
        return

    num_samples = min(num_samples, len(available))

    samples = random.sample(available, num_samples)

    cols = int(num_samples ** 0.5)

    if cols * cols < num_samples:
        cols += 1

    rows = (num_samples + cols - 1) // cols

    fig, axes = plt.subplots(
        rows,
        cols,
        figsize=(5 * cols, 5 * rows)
    )

    if rows == 1 and cols == 1:
        axes = [axes]
    else:
        axes = axes.flatten()

    for ax in axes:
        ax.axis("off")

    for ax, image_id in zip(axes, samples):

        image_path = image_index[image_id]
        label_path = label_index.get(image_id)

        image = Image.open(image_path)

        width, height = image.size

        ax.imshow(image)
        ax.set_title(image_id, fontsize=8)

        if label_path is not None:

            with open(label_path) as f:

                for line in f:

                    parts = line.strip().split()

                    if len(parts) != 5:
                        continue

                    cls = int(parts[0])

                    x = float(parts[1])
                    y = float(parts[2])
                    w = float(parts[3])
                    h = float(parts[4])

                    x1 = (x - w / 2) * width
                    y1 = (y - h / 2) * height

                    rect = patches.Rectangle(
                        (x1, y1),
                        w * width,
                        h * height,
                        fill=False,
                        linewidth=2,
                    )

                    ax.add_patch(rect)

                    ax.text(
                        x1,
                        y1,
                        class_names[cls],
                        fontsize=8,
                        bbox=dict(alpha=0.6),
                    )

    plt.tight_layout()

    figure_path = output_dir / "sample_grid.png"

    plt.savefig(
        figure_path,
        dpi=200,
        bbox_inches="tight",
    )

    plt.close()

    logger.info(f"Saved {figure_path}")


# =============================================================================
# Reports
# =============================================================================

def save_reports(
    output_dir: Path,
    train_stats: dict,
    val_stats: dict,
    test_stats: dict,
):

    output_dir.mkdir(parents=True, exist_ok=True)

    report_file = output_dir / "verification_report.txt"

    with open(report_file, "w") as f:

        f.write("YOLO Dataset Verification Report\n")
        f.write("=" * 50 + "\n\n")

        for name, stats in [
            ("TRAIN", train_stats),
            ("VALIDATION", val_stats),
            ("TEST", test_stats),
        ]:

            f.write(name + "\n")
            f.write("-" * 40 + "\n")

            f.write(f"Images: {stats['images']}\n")
            f.write(f"Objects: {stats['objects']}\n")
            f.write(f"Average objects/image: {stats['avg_objects']}\n")
            f.write(f"Empty labels: {stats['empty_labels']}\n")
            f.write(f"Invalid classes: {stats['invalid_classes']}\n")
            f.write(f"Invalid boxes: {stats['invalid_boxes']}\n")
            f.write(f"Invalid lines: {stats['invalid_lines']}\n\n")

            f.write("Class distribution\n")

            for cls, count in stats["objects_per_class"].items():
                f.write(f"{cls}: {count}\n")

            f.write("\n")

            if stats["errors"]:

                f.write("Errors\n")

                for err in stats["errors"]:
                    f.write(err + "\n")

                f.write("\n")

    logger.info(f"Saved {report_file}")

    json_report = {

        "train": {
            k: v
            for k, v in train_stats.items()
            if k not in ["errors", "objects_per_class", "objects_per_image"]
        },

        "validation": {
            k: v
            for k, v in val_stats.items()
            if k not in ["errors", "objects_per_class", "objects_per_image"]
        },
        
        "test": {
            k: v
            for k, v in test_stats.items()
            if k not in ["errors","objects_per_class","objects_per_image"]
        },

        "train_class_distribution":
            dict(train_stats["objects_per_class"]),

        "validation_class_distribution":
            dict(val_stats["objects_per_class"]),
        
        "test_class_distribution":
            dict(test_stats["objects_per_class"]),
    }

    json_file = output_dir / "report.json"

    with open(json_file, "w") as f:
        json.dump(json_report, f, indent=4, default=str)

    logger.info(f"Saved {json_file}")


# =============================================================================
# Main
# =============================================================================

def main():

    args = parse_args()

    verify_structure(args.dataset)

    class_names = read_dataset_yaml(args.dataset)

    index = index_dataset(args.dataset)

    train_match, val_match, test_match = verify_matching(index)

    train_stats = summarize_statistics(
        validate_split(
            index["train_images"],
            index["train_labels"],
            class_names,
        )
    )

    val_stats = summarize_statistics(
        validate_split(
            index["val_images"],
            index["val_labels"],
            class_names,
        )
    )
    
    test_stats = summarize_statistics(
        validate_split(
            index["test_images"],
            index["test_labels"],
            class_names,
        )
    )

    print_statistics("Training", train_stats)

    print_statistics("Validation", val_stats)
    
    print_statistics("Test", test_stats)

    visualize_samples(
        index["test_images"],
        index["test_labels"],
        class_names,
        args.output / "test_samples",
        args.samples,
    )

    save_reports(
        args.output,
        train_stats,
        val_stats,
        test_stats,
    )

    total_errors = (
        train_stats["invalid_boxes"]
        + train_stats["invalid_lines"]
        + train_stats["invalid_classes"]
        + val_stats["invalid_boxes"]
        + val_stats["invalid_lines"]
        + val_stats["invalid_classes"]
        + test_stats["invalid_boxes"]
        + test_stats["invalid_lines"]
        + test_stats["invalid_classes"]
        + len(train_match["missing_labels"])
        + len(train_match["missing_images"])
        + len(val_match["missing_labels"])
        + len(val_match["missing_images"])
        + len(test_match["missing_labels"])
        + len(test_match["missing_images"])
    )

    logger.info("")
    logger.info("=" * 60)

    if total_errors == 0:

        logger.info("✓ DATASET VERIFIED SUCCESSFULLY")

    else:

        logger.warning(f"Dataset contains {total_errors} issue(s).")

        if args.strict:
            raise SystemExit(1)

    logger.info("=" * 60)


if __name__ == "__main__":
    main()