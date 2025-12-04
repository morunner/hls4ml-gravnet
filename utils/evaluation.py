# Adapted from: https://github.com/lorenzo-as/fast-gnn-clustering

import json
import os
import pickle

import numpy as np
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
        response = tf.math.divide_no_nan(y_pred, y_true)
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


def display_evaluation_results(test_energy_pred: np.ndarray, test_pid_pred: np.ndarray, D: np.ndarray, model_cfg: dict):
    test_response_rmse = response_rmse(D['y_energy_test'], test_energy_pred)
    test_auc = roc_auc_score(D['y_pid_test'], test_pid_pred)
    fpr, tpr, thresholds = roc_curve(D['y_pid_test'], test_pid_pred)

    print(f'Test Response RMSE: {test_response_rmse:.4f}')
    print(f'Test PID AUC: {test_auc:.4f}')

    plt.figure(figsize=(12, 5))

    plt.subplot(1, 2, 1)
    plt.plot(fpr, tpr)
    plt.xlabel('Pion False Positive Rate')
    plt.ylabel('Pion True Positive Rate')
    plt.xlim(0.0, 1.0)
    plt.ylim(0.0, 1.0)

    plt.subplot(1, 2, 2)
    plt.hist(
        test_energy_pred.flatten() / D['y_energy_test'],
        bins=50,
        histtype='stepfilled',
        alpha=0.7,
        density=True,
    )
    plt.axvline(1.0, color='k', linestyle='--', lw=1, alpha=0.7)
    plt.xlabel('Predicted / True Energy')
    plt.ylabel('Density')
    plt.xlim(0.0, 4.0)

    spcr = ' ' * 5
    notes = 'Trained on 2% of Garnet dataset \n(1 file, 10k events)'
    descr = (
        'QGravNet Evaluation \n\n'
        + spcr
        + f'AUC: {test_auc:.3f} \n'
        + spcr
        + f'Response RMS: {test_response_rmse:.3f}'
        + '\n\n\nModel Config (changes from default):\n\n'
        + ''.join([f'{spcr}{k}: {v}\n' for k, v in model_cfg.items()])
        + '\n\nNotes: \n\n'
        + notes
    )
    plt.text(
        1.05,
        1.0,
        descr,
        transform=plt.gca().transAxes,
        fontsize=11,
        verticalalignment='top',
        horizontalalignment='left',
    )

    plt.tight_layout()
    plt.show()
