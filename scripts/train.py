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

DATA_FILE = DATASET_PATH / 'toy_calo_y_train_scaled/toy_calo_processed.h5'
TRAIN_DIR = RESULTS_PATH / 'train_new_quantization_cfg_y_train_scaled'

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

if __name__ == '__main__':
    os.makedirs(TRAIN_DIR, exist_ok=False)

    D = load_processed(DATA_FILE)

    model = QGravNetFactory(**keras_model_cfg).create_keras_model(n_vertices=128, n_features=4)
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

    model.save_weights(os.path.join(TRAIN_DIR, 'model.weights.h5'))

    with open(os.path.join(TRAIN_DIR, 'model_cfg.pkl'), 'wb') as f:
        pickle.dump(keras_model_cfg, f)

    with open(os.path.join(TRAIN_DIR, 'history.json'), 'w') as f:
        json.dump(history.history, f, default=lambda o: o.item() if isinstance(o, np.generic) else o)

    with open(os.path.join(TRAIN_DIR, 'info.json'), 'w') as f:
        info = {
            'datapath': str(DATA_FILE),
            'n_epochs': n_epochs,
            'batch_size': batch_size,
        }
        json.dump(info, f, indent=2)

    print('Training complete')
