from collections.abc import Sequence
from typing import Any

import keras

from hls4ml.converters.keras_v3._base import KerasV3LayerHandler, register


@register
class GravNetCoreHandler(KerasV3LayerHandler):
    handles = ('GravNetCore',)

    def handle(
        self, layer: 'keras.Layer', in_tensors: Sequence['keras.KerasTensor'], out_tensors: Sequence['keras.KerasTensor']
    ) -> dict[str, Any] | tuple[dict[str, Any], ...]:
        output_shape = layer.output.shape[1:]  # First dimension of shape is batch size
        assert len(output_shape) == 2, 'GravNetCore supports only 2D output tensors'

        n_neighbours = layer.get_config().get('n_neighbours')
        assert n_neighbours is not None
        return {
            'n_vertices': output_shape[0],
            'n_out_features': output_shape[1],
            'n_neighbours': n_neighbours,
            'exponential_table': {'ScaleFactor': 2, 'Resolution': 16},
        }
