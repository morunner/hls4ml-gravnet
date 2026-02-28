import argparse
import json
import os
import pickle
from math import ceil

import numpy as np
import tensorflow_model_optimization as tfmot
from keras.layers import Dense
from keras.models import clone_model
from qkeras.utils import get_model_sparsity
from qgravnet import QGravNetFactory, GravNetFactory

from train import batch_size, callbacks, n_epochs, optimizer_cfg
from hls4ml_gravnet.utils.data import load_processed, shuffle_vertices, truncate_or_pad_vertices
from hls4ml_gravnet.utils.evaluation import load_run
from hls4ml_gravnet.utils.files import RESULTS_PATH


def apply_pruning(layer, end_step, final_sparsity=0.75):
    if isinstance(layer, Dense):
        if layer.name not in ['regression', 'classification']:
            pruning_params = {
                'pruning_schedule': tfmot.sparsity.keras.PolynomialDecay(
                    initial_sparsity=0.0,
                    final_sparsity=final_sparsity,
                    begin_step=0,
                    end_step=end_step,
                    frequency=500,
                )
            }
            print(f'Pruning layer {layer.name} with PolynomialDecay, final sparsity = {final_sparsity}')
            return tfmot.sparsity.keras.prune_low_magnitude(layer, **pruning_params)

    return layer

def log_model_sparsity(model) -> str:
    total, per_layer = get_model_sparsity(model, per_layer=True)

    lines = [
        f"Model sparsity ({model.name})",
        f"total: {total:.4f}",
        "-" * 40,
        *[f"{name:<40s} {sp:.4f}" for name, sp in per_layer],
    ]
    summary = "\n".join(lines)

    return summary

def parse_args():
    parser = argparse.ArgumentParser(
        prog='PruneGravNet',
        description='Prune an already trained GravNet model',
    )
    parser.add_argument('-i', '--input_dir')
    parser.add_argument('-o', '--output_dir')
    parser.add_argument('--sparsity', type=float, default=0.75, help='Target sparsity')

    return parser.parse_args()

n_pruning_epochs = 30

def main():
    args = parse_args()

    input_dir = RESULTS_PATH / args.input_dir
    output_dir = RESULTS_PATH / args.output_dir

    train_dir = RESULTS_PATH / args.input_dir
    model_cfg, _, _, datapath = load_run(train_dir=train_dir)

    model_cfg, weights_path, history, datapath, n_vertices, is_shuffled = load_run(input_dir)
    info = json.load(open(input_dir / 'info.json'))
    D = load_processed(datapath)

    factory_cls = GravNetFactory if info.get('factory', 'QGravNetFactory') == 'GravNetFactory' else QGravNetFactory
    pretrained_model = factory_cls(**model_cfg).create_keras_model(n_vertices=n_vertices, n_features=4)
    pretrained_model.load_weights(weights_path)

    steps_per_epoch = ceil(len(D['X_hits_train']) / batch_size)
    end_step = steps_per_epoch * n_pruning_epochs
    print(f'Pruning end_step set to {end_step} (steps_per_epoch {steps_per_epoch} * epochs {n_pruning_epochs})')

    model_for_pruning = clone_model(
        pretrained_model,
        clone_function=lambda layer: apply_pruning(layer, end_step, final_sparsity=args.sparsity),
    )
    model_for_pruning.compile(**optimizer_cfg)

    if is_shuffled:
        D['X_hits_train'] = shuffle_vertices(D['X_hits_train'], seed=0)
    D['X_hits_train'] = truncate_or_pad_vertices(D['X_hits_train'], n_vertices)

    print('Pruning model...')
    prune_history = model_for_pruning.fit(
        x=D['X_hits_train'],
        y={
            'regression': D['y_energy_train'],
            'classification': D['y_pid_train'],
        },
        epochs=n_pruning_epochs,
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
    retrain_history = model_for_pruning.fit(
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

    os.makedirs(output_dir, exist_ok=False)

    # Reload model and strip pruning
    final_model = tfmot.sparsity.keras.strip_pruning(model_for_pruning)
    final_model.save_weights(os.path.join(output_dir, 'model.weights.h5'))
    final_model.save(os.path.join(output_dir, f'{args.output_dir}.keras'))

    with open(os.path.join(output_dir, 'model_cfg.pkl'), 'wb') as f:
        pickle.dump(model_cfg, f)

    history = {
        k: prune_history.history.get(k, []) + retrain_history.history.get(k, [])
        for k in set(prune_history.history) | set(retrain_history.history)
    }
    with open(os.path.join(output_dir, 'history.json'), 'w') as f:
        json.dump(history, f, default=lambda o: o.item() if isinstance(o, np.generic) else o)

    with open(os.path.join(output_dir, 'info.json'), 'w') as f:
        info = {
            'datapath': str(datapath),
            'n_epochs': n_epochs,
            'n_vertices': n_vertices,
            'pruning_epochs': n_pruning_epochs,
            'batch_size': batch_size,
            'sparsity': args.sparsity,
            'is_shuffled': is_shuffled,
            'factory': factory_cls.__name__,
        }
        json.dump(info, f, indent=2)

    sparsity_summary = log_model_sparsity(final_model)
    print(sparsity_summary)
    with open(os.path.join(output_dir, 'sparsity.txt'), 'w') as f:
        f.write(sparsity_summary)

    print('Training complete')


if __name__ == '__main__':
    main()
