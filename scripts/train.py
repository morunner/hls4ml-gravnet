import argparse
import json
import os
import pickle

import numpy as np
from keras.optimizers import Adam
from qgravnet import QGravNetFactory

from utils.config import keras_model_cfg
from utils.data import load_processed
from utils.evaluation import response_rmse
from utils.files import DATASET_PATH, RESULTS_PATH

try:
    import keras
except ImportError:
    from tensorflow import keras

DATA_FILE = DATASET_PATH / 'toy_calo_64_vert/toy_calo_processed.h5'

optimizer_cfg = {
    'optimizer': Adam(learning_rate=0.001),
    'loss': {'regression': response_rmse, 'classification': 'binary_crossentropy'},
    'loss_weights': {'regression': 0.9, 'classification': 0.1},
    'metrics': {'classification': ['accuracy']},
}

callbacks = [
    keras.callbacks.ReduceLROnPlateau(factor=0.2, patience=5, verbose=1),
    keras.callbacks.EarlyStopping(patience=10, verbose=1, restore_best_weights=True),
]

n_epochs = 100
batch_size = 32


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog='TrainGravNet',
        description='Train the GravNet model',
    )
    parser.add_argument('output_dir')

    return parser.parse_args()


def main():
    args = parse_args()
    train_dir = RESULTS_PATH / args.output_dir

    os.makedirs(train_dir, exist_ok=False)

    D = load_processed(DATA_FILE)

    model = QGravNetFactory(**keras_model_cfg).create_keras_model(n_vertices=64, n_features=4)
    model.compile(**optimizer_cfg)

    history = model.fit(
        x=D['X_hits_train'],
        y={
            'regression': D['y_energy_train'],
            'classification': D['y_pid_train'],
        },
        epochs=n_epochs,
        validation_split=0.25,
        batch_size=batch_size,
        callbacks=callbacks,
        shuffle=True,
        verbose=1,
    )

    model.save_weights(os.path.join(train_dir, 'model.weights.h5'))

    with open(os.path.join(train_dir, 'model_cfg.pkl'), 'wb') as f:
        pickle.dump(keras_model_cfg, f)

    with open(os.path.join(train_dir, 'history.json'), 'w') as f:
        json.dump(history.history, f, default=lambda o: o.item() if isinstance(o, np.generic) else o)

    with open(os.path.join(train_dir, 'info.json'), 'w') as f:
        info = {
            'datapath': str(DATA_FILE),
            'n_epochs': n_epochs,
            'batch_size': batch_size,
        }
        json.dump(info, f, indent=2)

    print('Training complete')


if __name__ == '__main__':
    main()
