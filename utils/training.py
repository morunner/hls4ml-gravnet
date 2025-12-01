from typing import Tuple

import numpy as np


def generate_train_data_random(n_vertices: int, n_features: int, n_samples: int) -> Tuple[np.ndarray, np.ndarray]:
    random_inputs = np.random.normal(size=(n_samples, n_vertices, n_features), scale=10)
    random_outputs = np.random.normal(size=(n_samples, n_vertices, 4 * n_features), scale=5)
    return random_inputs, random_outputs
