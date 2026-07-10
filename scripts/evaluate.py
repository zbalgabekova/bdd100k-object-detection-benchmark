"""
evaluate.py

Evaluate object detection predictions on the BDD100K test set.

Inputs
------
- predictions.csv
- test_ground_truth.csv

Outputs
-------
- overall_metrics.json
-per_class_results.csv
- weather_results.csv
- timeofday_results.csv
- scene_results.csv
- matched_predictions.csv
- missed_objects.csv
-matched_ground_truth.csv
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

import numpy as np
import pandas as pd
import yaml


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

        description="Evaluate object detection predictions."

    )

    parser.add_argument(

        "--config",

        type=Path,

        required=True,

        help="Evaluation configuration YAML.",

    )

    return parser.parse_args()


# =============================================================================
# Configuration
# =============================================================================

def load_config(config_path):

    if not config_path.exists():

        raise FileNotFoundError(config_path)

    with open(config_path, "r") as f:

        config = yaml.safe_load(f)

    logger.info(
        f"Loaded configuration: {config_path}"
    )

    return config


def validate_config(config):

    required = [

        "predictions",

        "ground_truth",

        "output",

        "iou_threshold",

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

    output_dir = Path(
        config["output"]
    )

    output_dir.mkdir(

        parents=True,

        exist_ok=True,

    )

    logger.info(
        f"Output directory: {output_dir}"
    )

    return output_dir


# =============================================================================
# Data loading
# =============================================================================

def load_predictions(config):

    path = Path(
        config["predictions"]
    )

    if not path.exists():

        raise FileNotFoundError(path)

    df = pd.read_csv(path)

    logger.info(
        f"Loaded {len(df):,} predictions."
    )

    return df


def load_ground_truth(config):

    path = Path(
        config["ground_truth"]
    )

    if not path.exists():

        raise FileNotFoundError(path)

    df = pd.read_csv(path)

    logger.info(
        f"Loaded {len(df):,} ground-truth objects."
    )

    return df


# =============================================================================
# Validation
# =============================================================================

def validate_inputs(
    predictions,
    ground_truth,
):

    prediction_columns = {

        "prediction_id",

        "image_path",

        "class_id",

        "class_name",

        "confidence",

        "x1",

        "y1",

        "x2",

        "y2",

    }

    gt_columns = {

        "object_id",

        "image_path",

        "weather",

        "timeofday",

        "scene",

        "class_id",

        "class_name",

        "x1",

        "y1",

        "x2",

        "y2",

    }

    missing_predictions = (

        prediction_columns

        - set(predictions.columns)

    )

    missing_gt = (

        gt_columns

        - set(ground_truth.columns)

    )

    if missing_predictions:

        raise ValueError(

            f"Prediction file missing columns: {missing_predictions}"

        )

    if missing_gt:

        raise ValueError(

            f"Ground-truth file missing columns: {missing_gt}"

        )

    logger.info("Input validation successful.")
    
    
# =============================================================================
# IoU
# =============================================================================

def compute_iou(box1, box2):
    """
    Compute IoU between two boxes.

    Box format:
    [x1, y1, x2, y2]
    """

    x_left = max(box1[0], box2[0])
    y_top = max(box1[1], box2[1])

    x_right = min(box1[2], box2[2])
    y_bottom = min(box1[3], box2[3])

    if x_right <= x_left or y_bottom <= y_top:
        return 0.0

    intersection = (
        (x_right - x_left)
        * (y_bottom - y_top)
    )

    area1 = (
        (box1[2] - box1[0])
        * (box1[3] - box1[1])
    )

    area2 = (
        (box2[2] - box2[0])
        * (box2[3] - box2[1])
    )

    union = area1 + area2 - intersection

    if union <= 0:
        return 0.0

    return intersection / union


# =============================================================================
# Matching
# =============================================================================

def match_image(
    predictions,
    ground_truth,
    iou_threshold,
):
    """
    Match predictions and GT objects
    for one image and one class.
    """

    matched_gt = set()

    matched_predictions = []
    
    matched_ground_truth = []

    predictions = predictions.sort_values(
        "confidence",
        ascending=False,
    )
    
    # Get image metadata (all GT objects in the same image
    # share the same weather/timeofday/scene)
    if len(ground_truth) > 0:

        metadata = ground_truth.iloc[0]

        weather = metadata["weather"]
        timeofday = metadata["timeofday"]
        scene = metadata["scene"]

    else:

        weather = None
        timeofday = None
        scene = None

    for _, pred in predictions.iterrows():

        pred_box = [

            pred["x1"],
            pred["y1"],
            pred["x2"],
            pred["y2"],

        ]

        best_iou = 0.0
        best_gt = None

        for gt_index, gt in ground_truth.iterrows():

            if gt_index in matched_gt:
                continue

            gt_box = [

                gt["x1"],
                gt["y1"],
                gt["x2"],
                gt["y2"],

            ]

            iou = compute_iou(
                pred_box,
                gt_box,
            )

            if iou > best_iou:

                best_iou = iou
                best_gt = gt_index

        if (
            best_gt is not None
            and best_iou >= iou_threshold
        ):

            matched_gt.add(best_gt)
            
            gt = ground_truth.loc[best_gt]

            matched_predictions.append({

                "prediction_id": pred["prediction_id"],

                "object_id": gt["object_id"],

                "image_path": pred["image_path"],
                
                "weather": gt["weather"],

                "timeofday": gt["timeofday"],

                "scene": gt["scene"],

                "class_id": pred["class_id"],

                "class_name": pred["class_name"],

                "confidence": pred["confidence"],

                "iou": best_iou,

                "status": "TP",

            })
            
            matched_ground_truth.append({

                "object_id": gt["object_id"],

                "prediction_id": pred["prediction_id"],

                "image_path": gt["image_path"],

                "weather": gt["weather"],

                "timeofday": gt["timeofday"],
    
                "scene": gt["scene"],

                "class_id": gt["class_id"],

                "class_name": gt["class_name"],

                "iou": best_iou,

                "status": "TP",

            })

        else:
            

            matched_predictions.append({

                "prediction_id": pred["prediction_id"],

                "object_id": None,

                "image_path": pred["image_path"],
                
                "weather": weather,

                "timeofday": timeofday,

                "scene": scene,

                "class_id": pred["class_id"],

                "class_name": pred["class_name"],

                "confidence": pred["confidence"],

                "iou": best_iou,

                "status": "FP",

            })

    missed_objects = []

    for gt_index, gt in ground_truth.iterrows():

        if gt_index not in matched_gt:

            missed_objects.append({

                "object_id": gt["object_id"],

                "image_path": gt["image_path"],

                "weather": gt["weather"],

                "timeofday": gt["timeofday"],

                "scene": gt["scene"],

                "class_id": gt["class_id"],

                "class_name": gt["class_name"],

                "status": "FN",

            })

    return (

        matched_predictions,

        matched_ground_truth,

        missed_objects,

    )


# =============================================================================
# Whole dataset matching
# =============================================================================

def match_dataset(
    predictions,
    ground_truth,
    iou_threshold,
):
    """
    Match the whole dataset.
    """

    logger.info("")
    logger.info("=" * 60)
    logger.info("MATCHING PREDICTIONS")
    logger.info("=" * 60)

    matched_predictions = []

    missed_objects = []
    
    matched_ground_truth = []

    images = sorted(
        ground_truth["image_path"].unique()
    )

    for i, image in enumerate(images, start=1):

        if i % 500 == 0 or i == len(images):

            logger.info(
                f"{i:,}/{len(images):,} images"
            )

        pred_image = predictions[
            predictions["image_path"] == image
        ]

        gt_image = ground_truth[
            ground_truth["image_path"] == image
        ]

        classes = sorted(
            set(pred_image["class_id"])
            | set(gt_image["class_id"])
        )

        for cls in classes:

            pred_cls = pred_image[
                pred_image["class_id"] == cls
            ]

            gt_cls = gt_image[
                gt_image["class_id"] == cls
            ]

            matched, matched_gt, missed = match_image(

                pred_cls,

                gt_cls,

                iou_threshold,

            )

            matched_predictions.extend(
                matched
            )
            
            matched_ground_truth.extend(
                matched_gt
            )

            missed_objects.extend(
                missed
            )

    logger.info("Matching finished.")

    return (
        pd.DataFrame(matched_predictions),
        pd.DataFrame(matched_ground_truth),
        pd.DataFrame(missed_objects),
    )


# =============================================================================
# Metrics
# =============================================================================

def compute_metrics(
    matched_predictions,
    matched_ground_truth,
    missed_objects,
):
    """
    Compute evaluation metrics.

    Parameters
    ----------
    matched_predictions : DataFrame
        TP and FP predictions.

    matched_ground_truth : DataFrame
        Ground-truth objects matched to predictions (TP).

    missed_objects : DataFrame
        Unmatched ground-truth objects (FN).
    """

    # -----------------------------
    # Counts
    # -----------------------------

    tp = len(matched_ground_truth)

    fp = (
        matched_predictions["status"] == "FP"
    ).sum()

    fn = len(missed_objects)

    # -----------------------------
    # Metrics
    # -----------------------------

    precision = (
        tp / (tp + fp)
        if (tp + fp) > 0
        else 0.0
    )

    recall = (
        tp / (tp + fn)
        if (tp + fn) > 0
        else 0.0
    )

    f1 = (
        2 * precision * recall
        / (precision + recall)
        if (precision + recall) > 0
        else 0.0
    )

    if tp > 0:

        mean_iou = matched_ground_truth["iou"].mean()

    else:

        mean_iou = 0.0

    metrics = {

        "TP": int(tp),

        "FP": int(fp),

        "FN": int(fn),

        "Precision": float(precision),

        "Recall": float(recall),

        "F1": float(f1),

        "Mean IoU": float(mean_iou),

    }

    return metrics

# =============================================================================
# Per-class metrics
# =============================================================================

def evaluate_per_class(
    matched_predictions,
    matched_ground_truth,
    missed_objects,
):
    """
    Compute evaluation metrics for each object class.
    """

    # Get all classes that appear anywhere
    classes = sorted(
        set(matched_ground_truth["class_name"])
        | set(missed_objects["class_name"])
        | set(matched_predictions["class_name"])
    )

    rows = []

    for cls in classes:

        pred_cls = matched_predictions[
            matched_predictions["class_name"] == cls
        ]

        matched_gt_cls = matched_ground_truth[
            matched_ground_truth["class_name"] == cls
        ]

        missed_cls = missed_objects[
            missed_objects["class_name"] == cls
        ]

        metrics = compute_metrics(

            pred_cls,

            matched_gt_cls,

            missed_cls,

        )

        metrics["class_name"] = cls

        rows.append(metrics)

    return pd.DataFrame(rows)

# =============================================================================
# Save metrics
# =============================================================================

def save_metrics(
    metrics,
    per_class,
    output_dir,
):
    """
    Save evaluation results.
    """

    overall_path = (
        output_dir
        / "overall_metrics.json"
    )

    with open(
        overall_path,
        "w",
    ) as f:

        json.dump(
            metrics,
            f,
            indent=4,
        )

    per_class_path = (
        output_dir
        / "per_class_results.csv"
    )

    per_class.to_csv(
        per_class_path,
        index=False,
    )

    logger.info("")
    logger.info(
        f"Saved: {overall_path}"
    )

    logger.info(
        f"Saved: {per_class_path}"
    )


# =============================================================================
# Print summary
# =============================================================================

def print_metrics(metrics):

    logger.info("")
    logger.info("=" * 60)
    logger.info("OVERALL RESULTS")
    logger.info("=" * 60)

    logger.info(
        f"TP         : {metrics['TP']:,}"
    )

    logger.info(
        f"FP         : {metrics['FP']:,}"
    )

    logger.info(
        f"FN         : {metrics['FN']:,}"
    )

    logger.info("")

    logger.info(
        f"Precision  : {metrics['Precision']:.4f}"
    )

    logger.info(
        f"Recall     : {metrics['Recall']:.4f}"
    )

    logger.info(
        f"F1-score   : {metrics['F1']:.4f}"
    )

    logger.info(
        f"Mean IoU   : {metrics['Mean IoU']:.4f}"
    )
    
    
# =============================================================================
# Condition evaluation
# =============================================================================

def evaluate_by_condition(
    matched_predictions,
    matched_ground_truth,
    missed_objects,
    column_name,
):
    """
    Evaluate performance for each weather, time of day,
    or scene category.
    """

    logger.info("")
    logger.info(f"Evaluating by {column_name}...")

    values = sorted(
        set(matched_ground_truth[column_name].dropna())
        | set(missed_objects[column_name].dropna())
    )

    rows = []

    for value in values:

        pred_subset = matched_predictions[
            matched_predictions[column_name] == value
        ]

        matched_gt_subset = matched_ground_truth[
            matched_ground_truth[column_name] == value
        ]

        missed_subset = missed_objects[
            missed_objects[column_name] == value
        ]

        metrics = compute_metrics(

            pred_subset,

            matched_gt_subset,

            missed_subset,

        )

        metrics[column_name] = value

        metrics["Images"] = (
            len(
                set(matched_gt_subset["image_path"])
                | set(missed_subset["image_path"])
            )
        )

        metrics["Ground Truth Objects"] = (

            len(matched_gt_subset)

            + len(missed_subset)

        )

        rows.append(metrics)

    return pd.DataFrame(rows)


# =============================================================================
# Save condition results
# =============================================================================

def save_condition_results(
    weather_results,
    timeofday_results,
    scene_results,
    output_dir,
):
    """
    Save condition evaluation tables.
    """

    weather_path = (
        output_dir
        / "weather_results.csv"
    )

    timeofday_path = (
        output_dir
        / "timeofday_results.csv"
    )

    scene_path = (
        output_dir
        / "scene_results.csv"
    )

    weather_results.to_csv(
        weather_path,
        index=False,
    )

    timeofday_results.to_csv(
        timeofday_path,
        index=False,
    )

    scene_results.to_csv(
        scene_path,
        index=False,
    )

    logger.info("")
    logger.info("Condition results saved.")
    
# =============================================================================
# Save matched predictions
# =============================================================================

def save_debug_tables(
    matched_predictions,
    matched_ground_truth,
    missed_objects,
    output_dir,
):
    """
    Save detailed TP/FP/FN tables.
    """

    matched_path = (
        output_dir
        / "matched_predictions.csv"
    )
    
    matched_gt_path = (
        output_dir
        / "matched_ground_truth.csv"
    )

    missed_path = (
        output_dir
        / "missed_objects.csv"
    )

    matched_predictions.to_csv(
        matched_path,
        index=False,
    )
    
    matched_ground_truth.to_csv(
        matched_gt_path,
        index=False,
    )

    missed_objects.to_csv(
        missed_path,
        index=False,
    )

    logger.info("")
    logger.info(
        f"Saved: {matched_path}"
    )
    
    logger.info(
        f"Saved: {matched_gt_path}"
    )

    logger.info(
        f"Saved: {missed_path}"
    )


# =============================================================================
# Main
# =============================================================================

def main():

    args = parse_args()

    config = load_config(
        args.config,
    )

    validate_config(config)

    output_dir = prepare_output_directory(
        config,
    )

    predictions = load_predictions(
        config,
    )

    ground_truth = load_ground_truth(
        config,
    )

    validate_inputs(
        predictions,
        ground_truth,
    )

    matched_predictions, matched_ground_truth, missed_objects = match_dataset(

        predictions,

        ground_truth,

        config["iou_threshold"],

    )

    metrics = compute_metrics(

        matched_predictions,
        
        matched_ground_truth,

        missed_objects,

    )

    per_class = evaluate_per_class(

        matched_predictions,
        
        matched_ground_truth,

        missed_objects,

    )

    save_metrics(

        metrics,

        per_class,

        output_dir,

    )

    weather_results = evaluate_by_condition(

        matched_predictions,
        
        matched_ground_truth,

        missed_objects,

        "weather",

    )

    timeofday_results = evaluate_by_condition(

        matched_predictions,
        
        matched_ground_truth,

        missed_objects,

        "timeofday",

    )

    scene_results = evaluate_by_condition(

        matched_predictions,
        
        matched_ground_truth,

        missed_objects,

        "scene",

    )

    save_condition_results(

        weather_results,

        timeofday_results,

        scene_results,

        output_dir,

    )

    save_debug_tables(

        matched_predictions,
        
        matched_ground_truth,

        missed_objects,

        output_dir,

    )

    print_metrics(
        metrics,
    )

    logger.info("")
    logger.info("=" * 60)
    logger.info("Evaluation finished successfully.")
    logger.info("=" * 60)


if __name__ == "__main__":

    main()
    
    
