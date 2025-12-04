from typing import Tuple

import numpy as np


def generate_train_data_random(n_vertices: int, n_features: int, n_samples: int) -> Tuple[np.ndarray, np.ndarray]:
    random_inputs = np.random.normal(size=(n_samples, n_vertices, n_features), scale=1)
    random_energies = np.random.normal(size=(n_samples,), scale=40)
    random_pids = np.random.randint(size=(n_samples,), low=0, high=2)
    return random_inputs, random_energies, random_pids
