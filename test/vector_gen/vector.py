from dataclasses import dataclass
from typing import Union

import numpy as np
import tensorflow as tf


@dataclass
class TestVectorBase:
    name: str
    expected_result: Union[np.ndarray, tf.Tensor, None]

    def __post_init__(self):
        """Automatically convert all tf.Tensors to np.ndarray"""
        for field_name, value in self.__dict__.items():
            if hasattr(value, 'numpy'):
                setattr(self, field_name, value.numpy())


@dataclass
class GlobalExchangeTestVector(TestVectorBase):
    x: Union[np.ndarray, tf.Tensor]


@dataclass
class GravnetCoreTestVector(TestVectorBase):
    coords: Union[np.ndarray, tf.Tensor]
    feats: Union[np.ndarray, tf.Tensor]
