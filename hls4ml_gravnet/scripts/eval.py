import argparse
import os
import logging

import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import roc_auc_score, roc_curve

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3" # error only

from hls4ml_gravnet.utils.data import load_processed, shuffle_vertices
from hls4ml_gravnet.utils.evaluation import load_run, response_rmse
from hls4ml_gravnet.utils.files import get_project_root_dir

from qgravnet import QGravNetFactory

PROJECT_ROOT = get_project_root_dir("hls4ml-gravnet")

logger = logging.getLogger("eval")
logging.basicConfig(level=logging.INFO)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate a trained QGravNet model.")
    parser.add_argument("run", type=str, help="The name of the training run to evaluate.")
    args = parser.parse_args()

    model_cfg, weights_path, history, datapath, n_vertices, is_shuffled = load_run(PROJECT_ROOT / "data/results/" / args.run)
    D = load_processed(datapath)
    trained_model = QGravNetFactory(**model_cfg).create_keras_model(n_vertices, 4)
    trained_model.load_weights(weights_path)
    # trained_model.summary()

    if is_shuffled:
        D["X_hits_test"] = shuffle_vertices(D["X_hits_test"], seed=0)
    D["X_hits_test"] = D["X_hits_test"][:, :n_vertices, :]
    test_energy_pred, test_pid_pred = trained_model.predict(D["X_hits_test"])

    test_response_rmse = response_rmse(D["y_energy_test"], test_energy_pred)
    test_auc = roc_auc_score(D["y_pid_test"], test_pid_pred)
    fpr, tpr, thresholds = roc_curve(D["y_pid_test"], test_pid_pred)

    logger.info(f"Test Response RMSE: {test_response_rmse:.4f}")
    logger.info(f"Test PID AUC: {test_auc:.4f}")

    plt.figure(figsize=(12, 5))

    plt.subplot(1, 2, 1)
    plt.plot(fpr, tpr)
    plt.xlabel("Pion False Positive Rate")
    plt.ylabel("Pion True Positive Rate")
    plt.xlim(0.0, 1.0)
    plt.ylim(0.0, 1.0)

    plt.subplot(1, 2, 2)
    plt.hist(
        test_energy_pred.flatten() / D["y_energy_test"],
        bins=50,
        # histtype="stepfilled",
        alpha=0.7,
        density=True,
    )
    plt.axvline(1.0, color="k", linestyle="--", lw=1, alpha=0.7)
    plt.xlabel("Predicted / True Energy")
    plt.ylabel("Density")
    plt.xlim(0.0, 4.0)

    spcr = " " * 5
    notes_dataset = "Trained on small Garnet dataset \n(1 file, 10k events)" if "mini" in datapath else "Trained on full Garnet dataset \n(50 files, à 10k events)"
    notes = f"{notes_dataset}\n\n{n_vertices} vertices\n\nNo large skip connections"
    descr = (
        "QGravNet Evaluation \n\n"
        + spcr
        + f"AUC: {test_auc:.3f} \n"
        + spcr
        + f"Response RMS: {test_response_rmse:.3f}"
        # + "\n\n\nModel Config (changes from default):\n\n"
        # + "".join([f"{spcr}{k}: {v}\n" for k, v in model_cfg.items()])
        + "\n\nNotes: \n\n"
        + notes
    )
    plt.text(
        1.05,
        1.0,
        descr,
        transform=plt.gca().transAxes,
        fontsize=11,
        verticalalignment="top",
        horizontalalignment="left",
    )

    plt.tight_layout()

    plt.savefig(f"/scratch/lasfour/hgcal-clustering/hls4ml-gravnet/data/results/{args.run}/eval_plot.png", dpi=300)
    plt.savefig(f"/scratch/lasfour/hgcal-clustering/hls4ml-gravnet/data/results/{args.run}/eval_plot.pdf")

    logger.info("Completed.\n")