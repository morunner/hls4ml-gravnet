import argparse
import os
import logging

parser = argparse.ArgumentParser(description="Evaluate a trained QGravNet model.")
parser.add_argument("runs", nargs="+", type=str, help="The name of the training run to evaluate.")
parser.add_argument("--names", nargs="*", default=None,
                    help="Optional display names, in the same order as runs.")
parser.add_argument("--outfile", type=str, default="eval_plot.png", help="Filename for the output plot.")
parser.add_argument("--title", type=str, default=None, help="Optional title for the plots.")
args = parser.parse_args()

def validate_args(args):
    if len(args.runs) == 0: 
        parser.error("At least one run must be specified.")
    if args.names is not None and len(args.names) != len(args.runs): 
        parser.error(f"If --names is provided, it must have the same number of entries as runs. Expected {len(args.runs)} but got {len(args.names)}.")
    if args.names is None:
        args.names = args.runs
    if os.path.exists(args.outfile):
        parser.error(f"Output file {args.outfile} already exists. Please specify a different filename or remove the existing file.")
validate_args(args)

import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import roc_auc_score, roc_curve

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3" # error only

from hls4ml_gravnet.utils.data import load_processed, shuffle_vertices, truncate_or_pad_vertices
from hls4ml_gravnet.utils.evaluation import load_run, response_rmse
from hls4ml_gravnet.utils.files import get_project_root_dir

from qgravnet import QGravNetFactory

PROJECT_ROOT = get_project_root_dir("hls4ml-gravnet")

logger = logging.getLogger("eval")
logging.basicConfig(level=logging.INFO)

if __name__ == "__main__":
    test_energy_pred = {}
    test_response_rmse = {}
    test_auc = {}
    roc_curves = {}
    for run in args.runs:
        model_cfg, weights_path, history, datapath, n_vertices, is_shuffled = load_run(PROJECT_ROOT / "data/results/" / run)
        D = load_processed(datapath)
        trained_model = QGravNetFactory(**model_cfg).create_keras_model(n_vertices, 4)
        trained_model.load_weights(weights_path)

        if is_shuffled:
            D["X_hits_test"] = shuffle_vertices(D["X_hits_test"], seed=0)
        D["X_hits_test"] = truncate_or_pad_vertices(D["X_hits_test"], n_vertices)
        test_energy_pred[run], test_pid_pred = trained_model.predict(D["X_hits_test"])

        test_response_rmse[run] = response_rmse(D["y_energy_test"], test_energy_pred[run])
        test_auc[run] = roc_auc_score(D["y_pid_test"], test_pid_pred)
        roc_curves[run] = roc_curve(D["y_pid_test"], test_pid_pred) #fpr, tpr, thresholds

        logger.info("-"*50 + f"\nRun: {run}")
        logger.info(f"Test Response RMSE: {test_response_rmse[run]:.4f}")
        logger.info(f"Test PID AUC: {test_auc[run]:.4f}\n\n")

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5), constrained_layout=True)

    for run, name in zip(args.runs, args.names):
        fpr, tpr, thresholds = roc_curves[run]
        ax1.plot(fpr, tpr, label=name)
    ax1.set_xlabel("Pion False Positive Rate")
    ax1.set_ylabel("Pion True Positive Rate")
    ax1.set_xlim(0.0, 1.0)
    ax1.set_ylim(0.0, 1.0)

    for run, name in zip(args.runs, args.names):
        ax2.hist(
            test_energy_pred[run].flatten() / D["y_energy_test"],
            bins=50,
            # histtype="stepfilled",
            alpha=0.7,
            density=True,
            label=name,
        )
    ax2.axvline(1.0, color="k", linestyle="--", lw=1, alpha=0.5)
    ax2.set_xlabel("Predicted / True Energy")
    ax2.set_ylabel("Density")
    ax2.set_xlim(0.0, 4.0)

    handles, labels = ax1.get_legend_handles_labels()
    fig.legend(
        handles, labels,
        loc="center left",
        bbox_to_anchor=(.99, 0.9), 
        frameon=False
    )

    if args.title is not None:
        plt.suptitle(args.title)

    # plt.tight_layout()

    plt.savefig(f"/scratch/lasfour/hgcal-clustering/hls4ml-gravnet/data/plots/{args.outfile}", dpi=300, bbox_inches="tight")
    logger.info("Completed.\n")