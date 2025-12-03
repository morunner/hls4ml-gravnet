# Minimal QGravNet model to accelerate HLS synthesis time when optimizing
# GravNetCore

import keras
from qgravnet.layers import GravNetCore
from qkeras import QDense, quantized_bits


class QGravNetMinimalFactory:
    def __init__(self, n_neighbours: int):
        self.n_neighbours = n_neighbours

    def create_keras_model(self, n_vertices: int, n_features: int) -> keras.Model:
        quantizer = quantized_bits(8, 0, alpha=1.0)

        inputs = keras.Input(shape=(n_vertices, n_features))

        coords = QDense(4, kernel_quantizer=quantizer, bias_quantizer=quantizer)(inputs)
        feats = QDense(8, kernel_quantizer=quantizer, bias_quantizer=quantizer)(inputs)
        x = GravNetCore(n_neighbours=self.n_neighbours, name='qgn_core')(coords, feats)
        x = keras.layers.GlobalAveragePooling1D(name='global_avg_pool')(x)
        energies = QDense(
            1,
            activation=None,
            kernel_quantizer=quantizer,
            bias_quantizer=quantizer,
            name='regression',
        )(x)
        classes = QDense(
            1,
            activation='sigmoid',
            kernel_quantizer=quantizer,
            bias_quantizer=quantizer,
            name='classification',
        )(x)

        return keras.Model(inputs=inputs, outputs=[energies, classes], name='qgravnet_model_dual')
