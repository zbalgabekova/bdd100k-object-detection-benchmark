"""
train.py

Train a YOLO/RT-DETR model using a YAML configuration file.

Example
-------
python train.py --config configs/baseline.yaml
"""

from __future__ import annotations

import argparse
import logging
import random
import shutil
from pathlib import Path

import numpy as np
import torch
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
        description="Train a YOLO model."
    )

    parser.add_argument(
        "--config",
        type=Path,
        required=True,
        help="Path to training configuration YAML.",
    )

    return parser.parse_args()


# =============================================================================
# Configuration
# =============================================================================

def load_config(config_path: Path) -> dict:

    if not config_path.exists():
        raise FileNotFoundError(config_path)

    with open(config_path, "r") as f:
        config = yaml.safe_load(f)

    logger.info(f"Loaded configuration: {config_path}")

    return config


# =============================================================================
# Validation
# =============================================================================

REQUIRED_FIELDS = [

    "data",
    "model",

    "epochs",
    "batch",
    "imgsz",

    "project",
    "name",

]


TRAIN_KEYS = [
    "data",
    "epochs",
    "batch",
    "imgsz",
    "workers",
    "device",
    "optimizer",
    "lr0",
    "weight_decay",
    "hsv_h",
    "hsv_s",
    "hsv_v",
    "degrees",
    "translate",
    "scale",
    "fliplr",
    "project",
    "name",
    "exist_ok",
    "seed",
]


def validate_config(config: dict):

    missing = []

    for field in REQUIRED_FIELDS:

        if field not in config:
            missing.append(field)

    if missing:

        raise ValueError(
            "Missing configuration fields: "
            + ", ".join(missing)
        )

    dataset = Path(config["data"])

    if not dataset.exists():

        raise FileNotFoundError(
            f"Dataset not found: {dataset}"
        )

    if config["epochs"] <= 0:
        raise ValueError("epochs must be > 0")

    if config["batch"] <= 0:
        raise ValueError("batch must be > 0")

    if config["imgsz"] <= 0:
        raise ValueError("imgsz must be > 0")

    logger.info("Configuration validated.")


# =============================================================================
# Reproducibility
# =============================================================================

def set_seed(seed: int):

    random.seed(seed)

    np.random.seed(seed)

    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    logger.info(f"Random seed set to {seed}")


# =============================================================================
# Experiment directory
# =============================================================================

def create_experiment_directory(config: dict) -> Path:

    project = Path(config["project"])

    experiment_dir = project / config["name"]

    experiment_dir.mkdir(
        parents=True,
        exist_ok=config.get("exist_ok", False),
    )

    return experiment_dir


# =============================================================================
# Summary
# =============================================================================

def print_summary(config: dict):

    logger.info("")
    logger.info("=" * 60)
    logger.info("TRAINING CONFIGURATION")
    logger.info("=" * 60)

    logger.info(f"Model        : {config['model']}")
    logger.info(f"Dataset      : {config['data']}")
    logger.info(f"Epochs       : {config['epochs']}")
    logger.info(f"Batch        : {config['batch']}")
    logger.info(f"Image size   : {config['imgsz']}")
    logger.info(f"Optimizer    : {config.get('optimizer', 'auto')}")
    logger.info(f"Device       : {config.get('device', 'auto')}")
    logger.info(f"Project      : {config['project']}")
    logger.info(f"Experiment   : {config['name']}")
    logger.info(f"Seed         : {config.get('seed', 42)}")

    logger.info("=" * 60)
    
    
# =============================================================================
# Model
# =============================================================================

def load_model(config: dict):
    """
    Load a detection model based on the configuration.
    Currently supports:
      - YOLO models
      - RT-DETR models
    """

    model_name = config["model"]

    logger.info("")
    logger.info(f"Loading model: {model_name}")

    if config["model_type"] == "rtdetr":
        model = RTDETR(model_name)
        logger.info("RT-DETR model loaded successfully.")
    else:
        model = YOLO(model_name)
        logger.info("YOLO model loaded successfully.")

    return model


# =============================================================================
# Training
# =============================================================================

def train_model(model: YOLO, config: dict):
    """
    Train the model using parameters from the configuration file.
    """

    logger.info("")
    logger.info("=" * 60)
    logger.info("STARTING TRAINING")
    logger.info("=" * 60)


    train_args = {
        key: config[key]
        for key in TRAIN_KEYS
        if key in config
    }

    # Automatically select device
    if config.get("device", "auto") == "auto":
        train_args["device"] = 0 if torch.cuda.is_available() else "cpu"

    train_args["verbose"] = True

    results = model.train(**train_args)
    
    logger.info("")
    logger.info("=" * 60)
    logger.info("TRAINING FINISHED")
    logger.info("=" * 60)

    return results


# =============================================================================
# Utilities
# =============================================================================

def print_best_weights(config: dict):
    """
    Print the expected location of the trained weights.
    """

    experiment = (
        Path(config["project"])
        / config["name"]
    )

    logger.info("")
    logger.info(f"Experiment directory : {experiment}")

    logger.info(
        f"Best weights         : {experiment / 'weights' / 'best.pt'}"
    )

    logger.info(
        f"Last weights         : {experiment / 'weights' / 'last.pt'}"
    )


def print_device():

    logger.info("")

    if torch.cuda.is_available():

        logger.info(
            f"CUDA device : {torch.cuda.get_device_name(0)}"
        )

        logger.info(
            f"GPU memory  : "
            f"{torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB"
        )

    else:

        logger.info("Running on CPU.")
        
      
    
# =============================================================================
# Save configuration
# =============================================================================

def save_experiment_config(config: dict, experiment_dir: Path):
    """
    Save a copy of the configuration used for this experiment.
    """

    output_file = experiment_dir / "config.yaml"

    with open(output_file, "w") as f:
        yaml.safe_dump(
            config,
            f,
            sort_keys=False,
        )

    logger.info(f"Saved configuration to {output_file}")


# =============================================================================
# Save training summary
# =============================================================================

def save_training_summary(
    config: dict,
    results,
    experiment_dir: Path,
):
    """
    Save training summary.
    """

    summary_file = experiment_dir / "training_summary.txt"

    summary = {
        "model": config["model"],
        "dataset": config["data"],
        "epochs": config["epochs"],
        "batch": config["batch"],
        "imgsz": config["imgsz"],
        "experiment": config["name"],
        "save_dir": str(results.save_dir),
    }

    with open(summary_file, "w") as f:

        f.write("YOLO Training Summary\n")
        f.write("=" * 50 + "\n\n")

        for key, value in summary.items():
            f.write(f"{key}: {value}\n")

    logger.info(f"Saved training summary to {summary_file}")

# =============================================================================
# Main
# =============================================================================

def main():

    args = parse_args()

    try:

        # ---------------------------------------------------------
        # Configuration
        # ---------------------------------------------------------

        config = load_config(args.config)

        validate_config(config)

        print_summary(config)

        set_seed(
            config.get("seed", 42)
        )

        experiment_dir = create_experiment_directory(
            config
        )

        # ---------------------------------------------------------
        # Save configuration
        # ---------------------------------------------------------

        save_experiment_config(
            config,
            experiment_dir,
        )

        # ---------------------------------------------------------
        # Device information
        # ---------------------------------------------------------

        print_device()

        # ---------------------------------------------------------
        # Load model
        # ---------------------------------------------------------

        model = load_model(config)

        # ---------------------------------------------------------
        # Train
        # ---------------------------------------------------------

        results = train_model(
            model,
            config,
        )

        # ---------------------------------------------------------
        # Save summary
        # ---------------------------------------------------------

        save_training_summary(
            config,
            results,
            experiment_dir,
        )

        # ---------------------------------------------------------
        # Final information
        # ---------------------------------------------------------

        print_best_weights(config)

        logger.info("")
        logger.info("=" * 60)
        logger.info("TRAINING COMPLETED SUCCESSFULLY")
        logger.info("=" * 60)

    except Exception as e:

        logger.exception("Training failed.")

        raise


if __name__ == "__main__":
    main()
