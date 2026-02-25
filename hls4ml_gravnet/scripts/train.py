import argparse
import json
import os
import pickle
from pprint import pformat

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog='TrainGravNet',
        description='Train the GravNet model',
    )
    parser.add_argument('output_dir')
    parser.add_argument('--mini', action='store_true', help='Use a smaller dataset for quick testing')
    parser.add_argument('--num-vertices', type=int, default=128, help='Number of vertices to use')
    parser.add_argument('--shuffle-vertices', action='store_true', help='Shuffle vertices before training')

    return parser.parse_args()
args = parse_args()

# Heavy imports, delayed until after argument parsing

import numpy as np
from keras.optimizers import AdamW

try:
    import keras
except ImportError:
    from tensorflow import keras

from qgravnet import QGravNetFactory
from qgravnet.selectors import BinnedSelector

from hls4ml_gravnet.utils.config import keras_model_cfg
from hls4ml_gravnet.utils.data import load_processed, shuffle_vertices, truncate_or_pad_vertices
from hls4ml_gravnet.utils.evaluation import response_rmse
from hls4ml_gravnet.utils.files import DATASET_PATH, RESULTS_PATH
from hls4ml_gravnet.utils.regularizers import add_overflow_regularization

optimizer_cfg = {
    'optimizer': AdamW(learning_rate=5e-4, weight_decay=1e-5),
    'loss': {'regression': response_rmse, 'classification': 'binary_crossentropy'},
    'loss_weights': {'regression': 0.8, 'classification': 0.2},
    'metrics': {'classification': ['accuracy']},
}
callbacks = [
    keras.callbacks.ReduceLROnPlateau(factor=0.2, patience=5, verbose=1, min_delta=1e-3),
    keras.callbacks.EarlyStopping(patience=25, verbose=1, restore_best_weights=True, min_delta=1e-4),
]
n_epochs = 500
batch_size = 32

def main():
    train_dir = RESULTS_PATH / args.output_dir

    os.makedirs(train_dir, exist_ok=False)
    D = load_processed(DATA_FILE)
    energy_target = D['y_energy_train']

    overflow_reg_cfg = keras_model_cfg.pop('overflow_regularization_cfg', None)
    model = QGravNetFactory(**keras_model_cfg).create_keras_model(n_vertices=args.num_vertices, n_features=4)

    if overflow_reg_cfg is not None:
        selector = BinnedSelector(**keras_model_cfg['selector_cfg'])
        add_overflow_regularization(
            model,
            selector,
            C=overflow_reg_cfg.get('max_per_bin', None),
            lambda_overflow=overflow_reg_cfg.get('overflow_lambda', 0),
            monitor=True,
        )
    model.compile(**optimizer_cfg)

    if args.shuffle_vertices:
        D['X_hits_train'] = shuffle_vertices(D['X_hits_train'], seed=0)
    D['X_hits_train'] = truncate_or_pad_vertices(D['X_hits_train'], args.num_vertices)
    history = model.fit(
        x=D['X_hits_train'],
        y={
            'regression': energy_target,
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
    model.save(os.path.join(train_dir, f'{args.output_dir}.keras'))

    with open(os.path.join(train_dir, 'model_cfg.pkl'), 'wb') as f:
        pickle.dump(keras_model_cfg, f)
    with open(os.path.join(train_dir, "model_cfg.txt"), "w") as f:
        f.write(pformat(keras_model_cfg, sort_dicts=False))

    with open(os.path.join(train_dir, 'history.json'), 'w') as f:
        json.dump(history.history, f, default=lambda o: o.item() if isinstance(o, np.generic) else o)

    with open(os.path.join(train_dir, 'info.json'), 'w') as f:
        info = {
            'datapath': str(DATA_FILE),
            'n_vertices': args.num_vertices,
            'n_epochs': n_epochs,
            'batch_size': batch_size,
            'shuffle_vertices': args.shuffle_vertices,
        }
        json.dump(info, f, indent=2)

    print('Training complete')


if __name__ == '__main__':
    main()
