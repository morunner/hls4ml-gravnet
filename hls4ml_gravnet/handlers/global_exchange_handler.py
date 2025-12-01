from collections.abc import Sequence
from typing import Any

import keras

from hls4ml.converters.keras_v3._base import KerasV3LayerHandler, register


@register
class GlobalExchangeHandler(KerasV3LayerHandler):
    handles = ('GlobalExchange',)

    def handle(
        self, layer: 'keras.Layer', in_tensors: Sequence['keras.KerasTensor'], out_tensors: Sequence['keras.KerasTensor']
    ) -> dict[str, Any] | tuple[dict[str, Any], ...]:
        output_shape = layer.output.shape[1:]  # First dimension of shape is batch size
        assert len(output_shape) == 2, 'GlobalExchange supports only 2D output tensors'
        return {'n_vertices': output_shape[0], 'n_out_features': output_shape[1]}
