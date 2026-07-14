"""
compare_models.py

Compare the performance of multiple object detection models
trained on the BDD100K benchmark.

Author: Zarema Balgabekova
"""

# =============================================================================
# Imports
# =============================================================================

import json
import logging
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


# =============================================================================
# Logger
# =============================================================================

logging.basicConfig(

    level=logging.INFO,

    format="%(asctime)s | %(levelname)s | %(message)s",

)

logger = logging.getLogger(__name__)


# =============================================================================
# Configuration
# =============================================================================

# Root directory of the project
PROJECT_ROOT = Path(__file__).resolve().parents[1]

# -------------------------------------------------------------------------
# Evaluation folders
# -------------------------------------------------------------------------

MODEL_PATHS = {

    "YOLO11n": PROJECT_ROOT
    / "runs"
    / "yolo11n"
    / "evaluation",

    "YOLO11s": PROJECT_ROOT
    / "runs"
    / "yolo11s"
    / "evaluation",

    "RT-DETR": PROJECT_ROOT
    / "runs"
    / "rtdetr"
    / "evaluation",

}

# -------------------------------------------------------------------------
# Output directory
# -------------------------------------------------------------------------

OUTPUT_DIR = (

    PROJECT_ROOT

    / "runs"

    / "comparison"

)


# =============================================================================
# Helper Functions
# =============================================================================

def create_output_directory():
    """
    Create output directory if it does not exist.
    """

    OUTPUT_DIR.mkdir(

        parents=True,

        exist_ok=True,

    )

    logger.info("")
    logger.info(f"Output directory: {OUTPUT_DIR}")


def print_models():
    """
    Print models that will be compared.
    """

    logger.info("")
    logger.info("Models included in comparison:")

    for model_name in MODEL_PATHS:

        logger.info(f"  • {model_name}")
        
        
# =============================================================================
# Loading Functions
# =============================================================================

def load_json(file_path):
    """
    Load a JSON file.
    """

    with open(file_path, "r") as f:

        data = json.load(f)

    return data


def load_csv(file_path):
    """
    Load a CSV file.
    """

    return pd.read_csv(file_path)


def load_model_results(model_name, model_dir):
    """
    Load all evaluation files for one model.
    """

    logger.info(f"Loading results for {model_name}...")

    results = {

        "overall": load_json(
            model_dir / "overall_metrics.json"
        ),

        "per_class": load_csv(
            model_dir / "per_class_results.csv"
        ),

        "weather": load_csv(
            model_dir / "weather_results.csv"
        ),

        "timeofday": load_csv(
            model_dir / "timeofday_results.csv"
        ),

        "scene": load_csv(
            model_dir / "scene_results.csv"
        ),

    }

    return results


def load_all_results():
    """
    Load evaluation results for all models.
    """

    all_results = {}

    logger.info("")
    logger.info("Loading evaluation results...")

    for model_name, model_dir in MODEL_PATHS.items():

        all_results[model_name] = load_model_results(

            model_name,

            model_dir,

        )

    logger.info("")
    logger.info("Finished loading evaluation files.")

    return all_results


# =============================================================================
# Inspection Functions
# =============================================================================

def print_loaded_files(all_results):
    """
    Print summary of loaded data.
    """

    logger.info("")
    logger.info("=" * 60)
    logger.info("Loaded Evaluation Results")
    logger.info("=" * 60)

    for model_name, results in all_results.items():

        logger.info("")
        logger.info(model_name)

        logger.info(
            f"  Overall metrics : "
            f"{len(results['overall'])} entries"
        )

        logger.info(
            f"  Per-class rows  : "
            f"{len(results['per_class'])}"
        )

        logger.info(
            f"  Weather rows    : "
            f"{len(results['weather'])}"
        )

        logger.info(
            f"  Time rows       : "
            f"{len(results['timeofday'])}"
        )

        logger.info(
            f"  Scene rows      : "
            f"{len(results['scene'])}"
        )

    logger.info("")
    
    
    
# =============================================================================
# Comparison Table Functions
# =============================================================================

def create_overall_comparison(all_results):
    """
    Create overall comparison table.
    """

    logger.info("")
    logger.info("Creating overall comparison table...")

    rows = []

    for model_name, results in all_results.items():

        metrics = results["overall"]

        rows.append({

            "Model": model_name,

            "Precision": metrics["Precision"],

            "Recall": metrics["Recall"],

            "F1": metrics["F1"],

            "Mean IoU": metrics["Mean IoU"],

            "TP": metrics["TP"],

            "FP": metrics["FP"],

            "FN": metrics["FN"],

        })

    df = pd.DataFrame(rows)

    return df


def create_per_class_comparison(all_results):
    """
    Merge per-class evaluation results.
    """

    logger.info("Creating per-class comparison table...")

    tables = []

    for model_name, results in all_results.items():

        df = results["per_class"].copy()

        df.insert(0, "Model", model_name)

        tables.append(df)

    return pd.concat(

        tables,

        ignore_index=True,

    )


def create_weather_comparison(all_results):
    """
    Merge weather evaluation results.
    """

    logger.info("Creating weather comparison table...")

    tables = []

    for model_name, results in all_results.items():

        df = results["weather"].copy()

        df.insert(0, "Model", model_name)

        tables.append(df)

    return pd.concat(

        tables,

        ignore_index=True,

    )


def create_timeofday_comparison(all_results):
    """
    Merge time-of-day evaluation results.
    """

    logger.info("Creating time-of-day comparison table...")

    tables = []

    for model_name, results in all_results.items():

        df = results["timeofday"].copy()

        df.insert(0, "Model", model_name)

        tables.append(df)

    return pd.concat(

        tables,

        ignore_index=True,

    )


def create_scene_comparison(all_results):
    """
    Merge scene evaluation results.
    """

    logger.info("Creating scene comparison table...")

    tables = []

    for model_name, results in all_results.items():

        df = results["scene"].copy()

        df.insert(0, "Model", model_name)

        tables.append(df)

    return pd.concat(

        tables,

        ignore_index=True,

    )


# =============================================================================
# Create All Comparison Tables
# =============================================================================

def create_all_tables(all_results):
    """
    Create all comparison tables.
    """

    tables = {

        "overall": create_overall_comparison(all_results),

        "per_class": create_per_class_comparison(all_results),

        "weather": create_weather_comparison(all_results),

        "timeofday": create_timeofday_comparison(all_results),

        "scene": create_scene_comparison(all_results),

    }

    logger.info("")
    logger.info("Comparison tables created successfully.")

    return tables


# =============================================================================
# Preview Functions
# =============================================================================

def print_overall_table(overall_table):
    """
    Print overall comparison table.
    """

    logger.info("")
    logger.info("=" * 70)
    logger.info("Overall Model Comparison")
    logger.info("=" * 70)

    print(overall_table.round(4))

    logger.info("")
    
    
# =============================================================================
# Save Functions
# =============================================================================

def save_comparison_tables(tables):
    """
    Save all comparison tables.
    """

    logger.info("")
    logger.info("Saving comparison tables...")

    file_names = {

        "overall": "overall_comparison.csv",

        "per_class": "per_class_comparison.csv",

        "weather": "weather_comparison.csv",

        "timeofday": "timeofday_comparison.csv",

        "scene": "scene_comparison.csv",

    }

    for key, filename in file_names.items():

        output_file = OUTPUT_DIR / filename

        tables[key].to_csv(

            output_file,

            index=False,

        )

        logger.info(f"Saved: {filename}")


# =============================================================================
# Summary Report
# =============================================================================

def create_summary_report(overall_table):
    """
    Create a text summary of the comparison.
    """

    logger.info("")
    logger.info("Creating summary report...")

    summary_file = OUTPUT_DIR / "comparison_summary.txt"

    best_precision = overall_table.loc[
        overall_table["Precision"].idxmax()
    ]

    best_recall = overall_table.loc[
        overall_table["Recall"].idxmax()
    ]

    best_f1 = overall_table.loc[
        overall_table["F1"].idxmax()
    ]

    best_iou = overall_table.loc[
        overall_table["Mean IoU"].idxmax()
    ]

    with open(summary_file, "w") as f:

        f.write("=" * 60 + "\n")
        f.write("MODEL COMPARISON SUMMARY\n")
        f.write("=" * 60 + "\n\n")

        f.write("Overall Metrics\n")
        f.write("-" * 60 + "\n\n")

        f.write(overall_table.round(4).to_string(index=False))

        f.write("\n\n")

        f.write("Best Models\n")
        f.write("-" * 60 + "\n")

        f.write(
            f"Highest Precision : {best_precision['Model']} "
            f"({best_precision['Precision']:.4f})\n"
        )

        f.write(
            f"Highest Recall    : {best_recall['Model']} "
            f"({best_recall['Recall']:.4f})\n"
        )

        f.write(
            f"Highest F1-score  : {best_f1['Model']} "
            f"({best_f1['F1']:.4f})\n"
        )

        f.write(
            f"Highest Mean IoU  : {best_iou['Model']} "
            f"({best_iou['Mean IoU']:.4f})\n"
        )

    logger.info(f"Saved: {summary_file.name}")


# =============================================================================
# Save Everything
# =============================================================================

def save_results(tables):
    """
    Save all outputs.
    """

    save_comparison_tables(tables)

    create_summary_report(

        tables["overall"]

    )

    logger.info("")
    logger.info("All comparison files saved successfully.")
    
    
# =============================================================================
# Plot Functions
# =============================================================================

def plot_overall_metrics(overall_table):
    """
    Plot overall comparison of all models.
    """

    logger.info("")
    logger.info("Creating overall comparison figure...")

    metrics = [

        "Precision",

        "Recall",

        "F1",

        "Mean IoU",

    ]

    fig, axes = plt.subplots(

        2,

        2,

        figsize=(12, 8),

    )

    axes = axes.flatten()

    for ax, metric in zip(axes, metrics):

        ax.bar(

            overall_table["Model"],

            overall_table[metric],

        )

        ax.set_title(metric)

        ax.set_ylim(0, 1)

        ax.set_ylabel(metric)

        ax.grid(

            axis="y",

            linestyle="--",

            alpha=0.5,

        )

        # Write value above each bar
        for i, value in enumerate(overall_table[metric]):

            ax.text(

                i,

                value + 0.01,

                f"{value:.3f}",

                ha="center",

                fontsize=9,

            )

    plt.tight_layout()

    output_file = (

        OUTPUT_DIR

        / "overall_metrics.png"

    )

    plt.savefig(

        output_file,

        dpi=300,

        bbox_inches="tight",

    )

    plt.close()

    logger.info(f"Saved: {output_file.name}")
    
    
def plot_single_metric(
    overall_table,
    metric,
):
    """
    Plot one metric.
    """

    plt.figure(

        figsize=(6, 5),

    )

    plt.bar(

        overall_table["Model"],

        overall_table[metric],

    )

    plt.ylim(0, 1)

    plt.ylabel(metric)

    plt.title(metric)

    plt.grid(

        axis="y",

        linestyle="--",

        alpha=0.5,

    )

    for i, value in enumerate(overall_table[metric]):

        plt.text(

            i,

            value + 0.01,

            f"{value:.3f}",

            ha="center",

        )

    plt.tight_layout()

    output_file = (

        OUTPUT_DIR

        / f"{metric.lower().replace(' ', '_')}.png"

    )

    plt.savefig(

        output_file,

        dpi=300,

        bbox_inches="tight",

    )

    plt.close()

    logger.info(f"Saved: {output_file.name}")
    
    
def create_overall_plots(tables):
    """
    Create all overall comparison plots.
    """

    overall = tables["overall"]

    plot_overall_metrics(

        overall,

    )

    plot_single_metric(

        overall,

        "Precision",

    )

    plot_single_metric(

        overall,

        "Recall",

    )

    plot_single_metric(

        overall,

        "F1",

    )

    plot_single_metric(

        overall,

        "Mean IoU",

    )

    logger.info("")
    logger.info("Overall plots created successfully.")
    
    
 # =============================================================================
# Condition Comparison Plots
# =============================================================================

def plot_condition_comparison(
    df,
    condition_column,
    output_name,
):
    """
    Plot Precision, Recall and F1 for one condition.
    """

    logger.info(f"Creating {output_name}...")

    metrics = [

        "Precision",

        "Recall",

        "F1",

    ]

    fig, axes = plt.subplots(

        3,

        1,

        figsize=(12, 14),

        sharex=True,

    )

    models = df["Model"].unique()

    categories = df[condition_column].unique()

    x = range(len(categories))

    width = 0.25

    for ax, metric in zip(axes, metrics):

        for i, model in enumerate(models):

            subset = df[

                df["Model"] == model

            ]

            ax.bar(

                [

                    p + width * i

                    for p in x

                ],

                subset[metric],

                width=width,

                label=model,

            )

        ax.set_ylabel(metric)

        ax.set_ylim(0, 1)

        ax.set_title(metric)

        ax.grid(

            axis="y",

            linestyle="--",

            alpha=0.5,

        )

    axes[-1].set_xticks(

        [

            p + width

            for p in x

        ]

    )

    axes[-1].set_xticklabels(

        categories,

        rotation=30,

        ha="right",

    )

    axes[0].legend()

    plt.tight_layout()

    output_file = (

        OUTPUT_DIR

        / output_name

    )

    plt.savefig(

        output_file,

        dpi=300,

        bbox_inches="tight",

    )

    plt.close()

    logger.info(f"Saved: {output_name}")
    
    
def create_condition_plots(
    tables,
):
    """
    Create all robustness comparison plots.
    """

    plot_condition_comparison(

        tables["weather"],

        "weather",

        "weather_comparison.png",

    )

    plot_condition_comparison(

        tables["timeofday"],

        "timeofday",

        "timeofday_comparison.png",

    )

    plot_condition_comparison(

        tables["scene"],

        "scene",

        "scene_comparison.png",

    )

    logger.info("")
    logger.info("Condition plots created successfully.")
    
    
# =============================================================================
# Model Ranking
# =============================================================================

def create_model_ranking(overall_table):
    """
    Rank models according to overall performance.
    """

    logger.info("")
    logger.info("Creating model ranking...")

    ranking = overall_table.copy()

    ranking["Precision Rank"] = (
        ranking["Precision"]
        .rank(ascending=False, method="min")
        .astype(int)
    )

    ranking["Recall Rank"] = (
        ranking["Recall"]
        .rank(ascending=False, method="min")
        .astype(int)
    )

    ranking["F1 Rank"] = (
        ranking["F1"]
        .rank(ascending=False, method="min")
        .astype(int)
    )

    ranking["Mean IoU Rank"] = (
        ranking["Mean IoU"]
        .rank(ascending=False, method="min")
        .astype(int)
    )

    ranking["Average Rank"] = (

        ranking[
            [
                "Precision Rank",
                "Recall Rank",
                "F1 Rank",
                "Mean IoU Rank",
            ]
        ]

        .mean(axis=1)

    )

    ranking = ranking.sort_values(

        "Average Rank"

    ).reset_index(drop=True)

    ranking.to_csv(

        OUTPUT_DIR / "model_ranking.csv",

        index=False,

    )

    logger.info("Saved: model_ranking.csv")

    return ranking


def print_model_ranking(ranking):
    """
    Print ranking table.
    """

    logger.info("")
    logger.info("=" * 70)
    logger.info("Overall Model Ranking")
    logger.info("=" * 70)

    print(

        ranking[
            [
                "Model",
                "Average Rank",
                "Precision Rank",
                "Recall Rank",
                "F1 Rank",
                "Mean IoU Rank",
            ]
        ]

    )

    logger.info("")
    
    
def print_best_models(overall_table):
    """
    Print best model for each metric.
    """

    logger.info("")
    logger.info("=" * 70)
    logger.info("Best Model by Metric")
    logger.info("=" * 70)

    metrics = [

        "Precision",

        "Recall",

        "F1",

        "Mean IoU",

    ]

    for metric in metrics:

        best = overall_table.loc[
            overall_table[metric].idxmax()
        ]

        logger.info(

            f"{metric:<12} : "

            f"{best['Model']} "

            f"({best[metric]:.4f})"

        )

    logger.info("")
    
    
    
# =============================================================================
# Main
# =============================================================================

def main():
    """
    Compare all trained object detection models.
    """

    logger.info("")
    logger.info("=" * 80)
    logger.info("BDD100K OBJECT DETECTION MODEL COMPARISON")
    logger.info("=" * 80)

    # -------------------------------------------------------------------------
    # Prepare output directory
    # -------------------------------------------------------------------------

    create_output_directory()

    # -------------------------------------------------------------------------
    # Print models
    # -------------------------------------------------------------------------

    print_models()

    # -------------------------------------------------------------------------
    # Load evaluation results
    # -------------------------------------------------------------------------

    all_results = load_all_results()

    print_loaded_files(all_results)

    # -------------------------------------------------------------------------
    # Create comparison tables
    # -------------------------------------------------------------------------

    tables = create_all_tables(

        all_results

    )

    print_overall_table(

        tables["overall"]

    )

    # -------------------------------------------------------------------------
    # Save tables
    # -------------------------------------------------------------------------

    save_results(

        tables

    )

    # -------------------------------------------------------------------------
    # Create plots
    # -------------------------------------------------------------------------

    create_overall_plots(

        tables

    )

    create_condition_plots(

        tables

    )

    # -------------------------------------------------------------------------
    # Ranking
    # -------------------------------------------------------------------------

    ranking = create_model_ranking(

        tables["overall"]

    )

    print_model_ranking(

        ranking

    )

    print_best_models(

        tables["overall"]

    )

    # -------------------------------------------------------------------------
    # Finished
    # -------------------------------------------------------------------------

    logger.info("")
    logger.info("=" * 80)
    logger.info("MODEL COMPARISON FINISHED SUCCESSFULLY")
    logger.info("=" * 80)

    logger.info("")
    logger.info(f"Results saved to:")
    logger.info(f"{OUTPUT_DIR}")
    logger.info("")


# =============================================================================
# Entry Point
# =============================================================================

if __name__ == "__main__":

    main()