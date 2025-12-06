# Minimal QGravNet model to accelerate HLS synthesis time when optimizing
# GravNetCore

import keras
from qgravnet.layers import GlobalExchange, GravNetCore
from qkeras import QActivation, QBatchNormalization, QDense, quantized_bits, quantized_relu, quantized_sigmoid


class QGravNetMinimalFactory:
    def __init__(self, n_neighbours: int):
        self.n_neighbours = n_neighbours

    def create_keras_model(self, n_vertices: int, n_features: int) -> keras.Model:
        quantizer = quantized_bits(8, 0, alpha=1.0)

        inputs = keras.Input(shape=(n_vertices, n_features))

        x = GlobalExchange(name='input_gex')(inputs)
        coords = QDense(4, kernel_quantizer=quantizer, bias_quantizer=quantizer)(x)
        feats = QDense(8, kernel_quantizer=quantizer, bias_quantizer=quantizer)(x)
        x = GravNetCore(n_neighbours=self.n_neighbours, name='qgn_core')([coords, feats])
        x = QBatchNormalization(
            name='input_bn',
            beta_quantizer=quantized_bits(8, 0, alpha=1.0),
            gamma_quantizer=quantized_relu(8, 0),
            mean_quantizer=quantized_bits(8, 0, alpha=1.0),
            variance_quantizer=quantized_bits(8, 0, alpha=1.0),
        )(x)
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
        )(x)
        classes = QActivation(quantized_sigmoid(8, 0), name='classification')(classes)

        return keras.Model(inputs=inputs, outputs=[energies, classes], name='qgravnet_model_dual')
