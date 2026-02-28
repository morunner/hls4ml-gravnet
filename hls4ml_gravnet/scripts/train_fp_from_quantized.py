import argparse
import json
import os
import pickle
from pprint import pformat

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description='Train a full-precision GravNet model using the config from a quantized run',
    )
    parser.add_argument('run', type=str, help='The name of the quantized training run to use')
    parser.add_argument('--out', type=str, default=None, help='Output name (default: {run}_FP)')

    return parser.parse_args()
args = parse_args()

# Heavy imports, delayed until after argument parsing

import numpy as np
from keras.optimizers import AdamW

try:
    import keras
except ImportError:
    from tensorflow import keras

from qgravnet import GravNetFactory
from qgravnet.selectors import BinnedSelector

from hls4ml_gravnet.utils.config import keras_model_cfg
from hls4ml_gravnet.utils.data import load_processed, shuffle_vertices, truncate_or_pad_vertices
from hls4ml_gravnet.utils.evaluation import load_run, response_rmse
from hls4ml_gravnet.utils.files import RESULTS_PATH
from hls4ml_gravnet.utils.regularizers import add_overflow_regularization
from hls4ml_gravnet.utils.config import remove_quantization_from_config

from train import callbacks, optimizer_cfg


def main():
    model_cfg, weights_path, history, datapath, n_vertices, is_shuffled = load_run(RESULTS_PATH / args.run)
    info = json.load(open(RESULTS_PATH / args.run / 'info.json'))


    if 'gravnet_kwargs' in model_cfg: # handle legacy config key 
        model_cfg['gravnet_cfg'] = model_cfg.pop('gravnet_kwargs')
    
    output_name = args.out if args.out else f"{args.run}_FP"
    train_dir = RESULTS_PATH / output_name
    
    os.makedirs(train_dir, exist_ok=False)
    
    D = load_processed(datapath)
    energy_target = D['y_energy_train']
    
    overflow_reg_cfg = model_cfg.pop('overflow_regularization_cfg', None)
    model_cfg = remove_quantization_from_config(model_cfg)
    
    model = GravNetFactory(**model_cfg).create_keras_model(n_vertices=n_vertices, n_features=4)
    
    if overflow_reg_cfg is not None:
        selector = BinnedSelector(**model_cfg['selector_cfg'])
        add_overflow_regularization(
            model,
            selector,
            C=overflow_reg_cfg.get('max_per_bin', None),
            lambda_overflow=overflow_reg_cfg.get('overflow_lambda', 0),
            monitor=True,
        )
    model.compile(**optimizer_cfg)
    
    if is_shuffled:
        D['X_hits_train'] = shuffle_vertices(D['X_hits_train'], seed=0)
    D['X_hits_train'] = truncate_or_pad_vertices(D['X_hits_train'], n_vertices)
    

    history = model.fit(
        x=D['X_hits_train'],
        y={
            'regression': energy_target,
            'classification': D['y_pid_train'],
        },
        epochs=info['n_epochs'],
        validation_split=0.25,
        batch_size=info['batch_size'],
        callbacks=callbacks,
        shuffle=True,
        verbose=1,
    )
    
    model.save_weights(os.path.join(train_dir, 'model.weights.h5'))
    model.save(os.path.join(train_dir, f'{output_name}.keras'))
    
    with open(os.path.join(train_dir, 'model_cfg.pkl'), 'wb') as f:
        pickle.dump(model_cfg, f)
    with open(os.path.join(train_dir, "model_cfg.txt"), "w") as f:
        f.write(pformat(model_cfg, sort_dicts=False))
    
    with open(os.path.join(train_dir, 'history.json'), 'w') as f:
        json.dump(history.history, f, default=lambda o: o.item() if isinstance(o, np.generic) else o)
    
    with open(os.path.join(train_dir, 'info.json'), 'w') as f:
        info = {
            'datapath': str(datapath),
            'n_vertices': n_vertices,
            'n_epochs': info['n_epochs'],
            'batch_size': info['batch_size'],
            'is_shuffled': is_shuffled,
            'full_precision': True,
            'factory': GravNetFactory.__name__,
            'trained_from_quantized_run': args.run,
        }
        json.dump(info, f, indent=2)
    
    print('Training complete')


if __name__ == '__main__':
    main()
