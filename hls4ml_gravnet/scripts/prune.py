import argparse
import json
import os
import pickle
from math import ceil

import numpy as np
import tensorflow_model_optimization as tfmot
from keras.layers import Dense
from keras.models import clone_model
from qgravnet import QGravNetFactory
from train import DATA_FILE, batch_size, callbacks, n_epochs, optimizer_cfg
from utils.config import keras_model_cfg
from utils.data import load_processed
from utils.files import RESULTS_PATH


def apply_pruning(layer, end_step):
    if isinstance(layer, Dense):
        if layer.name not in ['regression', 'classification']:
            final_sparsity = 0.75
            pruning_params = {
                'pruning_schedule': tfmot.sparsity.keras.PolynomialDecay(
                    initial_sparsity=0.0,
                    final_sparsity=final_sparsity,
                    begin_step=0,
                    end_step=end_step,
                    frequency=500,
                )
            }
            print(f'Pruning layer {layer.name} with PolynomialDecay, final sparsity = {0.75}')
            return tfmot.sparsity.keras.prune_low_magnitude(layer, **pruning_params)

    return layer


def parse_args():
    parser = argparse.ArgumentParser(
        prog='PruneGravNet',
        description='Prune an already trained GravNet model',
    )
    parser.add_argument('-i', '--input_dir')
    parser.add_argument('-o', '--output_dir')

    return parser.parse_args()


def main():
    args = parse_args()

    output_dir = RESULTS_PATH / args.output_dir
    os.makedirs(output_dir, exist_ok=False)

    D = load_processed(DATA_FILE)

    pretrained_model = QGravNetFactory(**keras_model_cfg).create_keras_model(n_vertices=64, n_features=4)
    pretrained_model.load_weights(RESULTS_PATH / args.input_dir / 'model.weights.h5')

    steps_per_epoch = ceil(len(D['X_hits_train']) / batch_size)
    end_step = steps_per_epoch * n_epochs
    print(f'Pruning end_step set to {end_step} (steps_per_epoch {steps_per_epoch} * epochs {n_epochs})')

    model_for_pruning = clone_model(
        pretrained_model,
        clone_function=lambda layer: apply_pruning(layer, end_step),
    )
    model_for_pruning.compile(**optimizer_cfg)

    print('Pruning model...')
    model_for_pruning.fit(
        x=D['X_hits_train'],
        y={
            'regression': D['y_energy_train'],
            'classification': D['y_pid_train'],
        },
        epochs=30,
        validation_split=0.25,
        batch_size=batch_size,
        callbacks=[
            tfmot.sparsity.keras.UpdatePruningStep(),
        ],
        shuffle=True,
        verbose=1,
    )

    print('Retraining pruned model')
    model_for_pruning.compile(**optimizer_cfg)
    history = model_for_pruning.fit(
        x=D['X_hits_train'],
        y={
            'regression': D['y_energy_train'],
            'classification': D['y_pid_train'],
        },
        epochs=n_epochs,
        initial_epoch=30,
        validation_split=0.25,
        batch_size=batch_size,
        callbacks=callbacks,
        shuffle=True,
        verbose=1,
    )

    # Reload model and strip pruning
    final_model = tfmot.sparsity.keras.strip_pruning(model_for_pruning)
    final_model.save_weights(os.path.join(output_dir, 'model.weights.h5'))

    with open(os.path.join(output_dir, 'model_cfg.pkl'), 'wb') as f:
        pickle.dump(keras_model_cfg, f)

    with open(os.path.join(output_dir, 'history.json'), 'w') as f:
        json.dump(history.history, f, default=lambda o: o.item() if isinstance(o, np.generic) else o)

    with open(os.path.join(output_dir, 'info.json'), 'w') as f:
        info = {
            'datapath': str(DATA_FILE),
            'n_epochs': n_epochs,
            'batch_size': batch_size,
        }
        json.dump(info, f, indent=2)

    print('Training complete')


if __name__ == '__main__':
    main()
