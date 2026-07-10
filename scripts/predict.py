"""
predict.py

Run inference on the BDD100K test set and save all predictions.

Output
------
predictions.csv

Columns
-------
image_path
class_id
class_name
confidence
x1
y1
x2
y2
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

import shutil
import json

import pandas as pd
import yaml

from ultralytics import YOLO, RTDETR

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
        description="Run inference on the BDD100K test set."
    )

    parser.add_argument(
        "--config",
        type=Path,
        required=True,
        help="Prediction configuration YAML.",
    )

    return parser.parse_args()


# =============================================================================
# Configuration
# =============================================================================

def load_config(config_path: Path):

    if not config_path.exists():
        raise FileNotFoundError(config_path)

    with open(config_path, "r") as f:

        config = yaml.safe_load(f)

    logger.info(f"Loaded configuration: {config_path}")

    return config


def validate_config(config):

    required = [

        "model",

        "test_csv",

        "images",

        "project",

        "name",

        "confidence",

        "imgsz",

        "device",

    ]

    missing = [

        key

        for key in required

        if key not in config

    ]

    if missing:

        raise KeyError(
            f"Missing configuration keys: {missing}"
        )


# =============================================================================
# Output directory
# =============================================================================

def prepare_output_directory(config):

    output_dir = (
        Path(config["project"])
        / config["name"]
    )

    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    logger.info(f"Output directory: {output_dir}")

    return output_dir


# =============================================================================
# Model
# =============================================================================

def load_model(config):

    model_path = config["model"]

    logger.info("")
    logger.info(f"Loading model: {model_path}")

    if "rtdetr" in model_path.lower():

        model = RTDETR(model_path)

    else:

        model = YOLO(model_path)

    logger.info("Model loaded successfully.")

    return model


# =============================================================================
# Model information
# =============================================================================

def get_model_type(model_path):
    """
    Infer model type from model filename.
    """

    model_name = Path(model_path).stem.lower()

    if "rtdetr" in model_name:
        return "RT-DETR"

    if "yolo11n" in model_name:
        return "YOLO11n"

    if "yolo11s" in model_name:
        return "YOLO11s"

    if "yolo11m" in model_name:
        return "YOLO11m"

    if "yolo11l" in model_name:
        return "YOLO11l"

    if "yolo11x" in model_name:
        return "YOLO11x"

    return model_name


# =============================================================================
# Test metadata
# =============================================================================

def load_test_metadata(config):

    csv_path = Path(config["test_csv"])

    if not csv_path.exists():

        raise FileNotFoundError(csv_path)

    df = pd.read_csv(csv_path)

    logger.info(
        f"Loaded {len(df):,} test images."
    )

    return df


# =============================================================================
# Image paths
# =============================================================================

def build_image_list(config, metadata):

    images_root = Path(config["images"])

    image_paths = []

    for relative_path in metadata["image_path"]:

        image_file = images_root / relative_path

        if not image_file.exists():

            logger.warning(
                f"Missing image: {image_file}"
            )

            continue

        image_paths.append(image_file)

    logger.info(
        f"Images to process: {len(image_paths):,}"
    )

    return image_paths


# =============================================================================
# Prediction
# =============================================================================

def predict_image(
    model,
    image_path,
    config,
):
    """
    Run inference on a single image.
    """

    results = model.predict(

        source=str(image_path),

        conf=config["confidence"],

        imgsz=config["imgsz"],

        device=config["device"],

        verbose=False,

    )

    return results[0]


# =============================================================================
# Prediction parsing
# =============================================================================

def prediction_to_rows(
    result,
    image_path,
):
    """
    Convert Ultralytics prediction to table rows.
    """

    rows = []

    names = result.names

    boxes = result.boxes

    if boxes is None or len(boxes) == 0:
        return rows

    for box in boxes:

        cls_id = int(box.cls.item())

        confidence = float(box.conf.item())

        x1, y1, x2, y2 = (
            box.xyxy[0]
            .cpu()
            .numpy()
            .tolist()
        )

        rows.append({
            
            "prediction_id": len(rows),

            "image_id": image_path.name,

            "image_path": image_path.name,

            "class_id": cls_id,

            "class_name": names[cls_id],

            "confidence": confidence,

            "x1": x1,
            "y1": y1,
            "x2": x2,
            "y2": y2,

        })

    return rows


# =============================================================================
# Whole dataset prediction
# =============================================================================

def predict_dataset(
    model,
    image_paths,
    config,
):
    """
    Predict all images.
    """

    logger.info("")
    logger.info("=" * 60)
    logger.info("RUNNING INFERENCE")
    logger.info("=" * 60)

    rows = []

    total = len(image_paths)

    for i, image_path in enumerate(image_paths, start=1):

        if i % 100 == 0 or i == total:

            logger.info(
                f"{i:,}/{total:,} images processed"
            )

        result = predict_image(

            model,

            image_path,

            config,

        )

        rows.extend(

            prediction_to_rows(

                result,

                image_path,

            )

        )

    logger.info("")
    logger.info(
        f"Finished inference on {total:,} images."
    )

    logger.info(
        f"Total predictions: {len(rows):,}"
    )

    return rows

# =============================================================================
# Save predictions
# =============================================================================

def save_predictions(
    rows,
    output_dir,
):
    """
    Save predictions to CSV.
    """

    df = pd.DataFrame(rows)

    csv_path = output_dir / "predictions.csv"

    df.to_csv(
        csv_path,
        index=False,
    )

    logger.info("")
    logger.info(f"Predictions saved to: {csv_path}")

    return csv_path

# =============================================================================
# Save configuration
# =============================================================================

def save_config(
    config,
    output_dir,
):
    """
    Save the prediction configuration.
    """

    config_path = output_dir / "prediction_config.yaml"

    with open(config_path, "w") as f:

        yaml.safe_dump(
            config,
            f,
            sort_keys=False,
        )

    logger.info(
        f"Configuration saved to: {config_path}"
    )


# =============================================================================
# Save summary
# =============================================================================

def save_summary(
    df,
    config,
    output_dir,
):
    """
    Save prediction summary as JSON.
    """

    summary = {
        
        "model_type": get_model_type(
            config["model"]
        ),

        "model": config["model"],

        "images": df["image_path"].nunique(),

        "predicted_objects": len(df),

        "average_confidence": (
            float(df["confidence"].mean())
            if len(df)
            else 0.0
        ),

        "objects_per_class": (
            df["class_name"]
            .value_counts()
            .to_dict()
            if len(df)
            else {}
        ),
        
        "inference": {

            "confidence": config["confidence"],

            "imgsz": config["imgsz"],

            "device": config["device"],

        },
    }

    summary_path = (
        output_dir
        / "prediction_summary.json"
    )

    with open(summary_path, "w") as f:

        json.dump(
            summary,
            f,
            indent=4,
        )

    logger.info(
        f"Summary saved to: {summary_path}"
    )

# =============================================================================
# Prediction summary
# =============================================================================

def print_summary(df):

    logger.info("")
    logger.info("=" * 60)
    logger.info("PREDICTION SUMMARY")
    logger.info("=" * 60)

    logger.info(f"Predicted objects : {len(df):,}")

    if len(df) == 0:

        logger.warning("No predictions were produced.")

        return

    logger.info("")
    logger.info("Objects per class")

    counts = (
        df["class_name"]
        .value_counts()
        .sort_index()
    )

    for cls, count in counts.items():

        logger.info(f"{cls:20s} {count:8,d}")

    logger.info("")
    logger.info(
        f"Average confidence: "
        f"{df['confidence'].mean():.3f}"
    )


# =============================================================================
# Main
# =============================================================================

def main():

    args = parse_args()

    config = load_config(
        args.config,
    )

    validate_config(
        config,
    )

    output_dir = prepare_output_directory(
        config,
    )

    model = load_model(
        config,
    )

    metadata = load_test_metadata(
        config,
    )

    image_paths = build_image_list(
        config,
        metadata,
    )

    rows = predict_dataset(

        model,

        image_paths,

        config,

    )

    # ---------------------------------------------------------
    # Replace filenames with relative image paths from metadata
    # ---------------------------------------------------------

    filename_to_path = {

        Path(p).name: p

        for p in metadata["image_path"]

    }

    for row in rows:

        row["image_path"] = filename_to_path[
            row["image_path"]
        ]

    csv_path = save_predictions(

        rows,

        output_dir,

    )

    predictions = pd.read_csv(
        csv_path,
    )
    
    save_config(
        config,
        output_dir,
    )

    save_summary(
        predictions,
        config,
        output_dir,
    )

    print_summary(
        predictions,
    )

    logger.info("")
    logger.info("Prediction finished successfully.")


if __name__ == "__main__":

    main()