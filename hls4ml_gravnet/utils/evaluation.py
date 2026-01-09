# Adapted from: https://github.com/lorenzo-as/fast-gnn-clustering

import json
import os
import pickle

import numpy as np
import seaborn as sns
from matplotlib import pyplot as plt
from sklearn.metrics import roc_auc_score, roc_curve

try:
    from keras import ops

    def response_rmse(y_true, y_pred):
        y_true = ops.reshape(y_true, (-1,))
        y_pred = ops.reshape(y_pred, (-1,))
        response = ops.divide_no_nan(y_pred, y_true)
        return ops.sqrt(ops.mean(ops.square(response - 1.0)))

except ImportError:
    import tensorflow as tf

    def response_rmse(y_true, y_pred):
        y_true = tf.reshape(y_true, (-1,))
        y_pred = tf.reshape(y_pred, (-1,))
        response = tf.math.divide_no_nan(y_pred, y_true / 100)
        return tf.sqrt(tf.reduce_mean(tf.square(response - 1.0)))


def load_run(train_dir):
    """
    Load a training run directory. Returns:
      model_cfg
      weights_path
      history
      datapath
    """
    train_dir = os.path.abspath(train_dir)

    cfg_path = os.path.join(train_dir, 'model_cfg.pkl')
    weights_path = os.path.join(train_dir, 'model.weights.h5')
    history_path = os.path.join(train_dir, 'history.json')
    info_path = os.path.join(train_dir, 'info.json')

    # --- existence checks ---
    for path, name in [
        (cfg_path, 'model_cfg.pkl'),
        (weights_path, 'model.weights.h5'),
        (history_path, 'history.json'),
        (info_path, 'info.json'),
    ]:
        if not os.path.isfile(path):
            raise FileNotFoundError(f'Missing {name} in {train_dir}')

    with open(cfg_path, 'rb') as f:
        model_cfg = pickle.load(f)

    with open(history_path, 'r') as f:
        history = json.load(f)

    with open(info_path, 'r') as f:
        info = json.load(f)
        datapath = info.get('datapath')

    return model_cfg, weights_path, history, datapath


def compare_keras_hls_predictions(
    test_energy_pred: np.ndarray,
    test_pid_pred: np.ndarray,
    test_energy_pred_hls: np.ndarray,
    test_pid_pred_hls: np.ndarray,
    test_energy_true: np.ndarray,
    test_pid_true: np.ndarray,
):
    test_response_rmse = response_rmse(test_energy_true, test_energy_pred)
    test_auc = roc_auc_score(test_pid_true, test_pid_pred)
    fpr, tpr, _ = roc_curve(test_pid_true, test_pid_pred)

    hls_response_rmse = response_rmse(test_energy_true, test_energy_pred_hls)
    hls_auc = roc_auc_score(test_pid_true, test_pid_pred_hls)
    hls_fpr, hls_tpr, _ = roc_curve(test_pid_true, test_pid_pred_hls)

    with sns.axes_style('darkgrid'):
        plt.figure(figsize=(8, 4))

        plt.subplot(1, 2, 1)
        plt.plot(1 - fpr, tpr, label=f'Keras (AUC: {test_auc:.3f})')
        plt.plot(1 - hls_fpr, hls_tpr, label=f'HLS (AUC: {hls_auc:.3f})')

        plt.xlabel('Pion False Positive Rate')
        plt.ylabel('Pion True Positive Rate')
        plt.xlim(0.7, 1.0)
        plt.ylim(0.7, 1.0)
        plt.title('Pion Identification ROC Curve')
        plt.legend(loc='lower right')

        plt.subplot(1, 2, 2)

        plt.hist(
            test_energy_pred.flatten() * 100 / test_energy_true,
            bins=50,
            histtype='stepfilled',
            alpha=0.5,
            density=True,
            label=f'Keras (RMSE: {test_response_rmse:.3f})',
        )

        plt.hist(
            test_energy_pred_hls.flatten() * 100 / test_energy_true,
            bins=50,
            histtype='stepfilled',
            alpha=0.5,
            density=True,
            label=f'HLS (RMSE: {hls_response_rmse:.3f})',
        )

        plt.axvline(1.0, color='k', linestyle='--', lw=1, alpha=0.7)
        plt.xlabel('Predicted / True Energy')
        plt.ylabel('Density')
        plt.xlim(0.0, 2.0)
        plt.title('Energy Response Distribution')
        plt.legend(loc='upper right')

        plt.tight_layout()
        plt.show()
