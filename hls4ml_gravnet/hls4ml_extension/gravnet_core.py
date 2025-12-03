from math import ceil, log2

from hls4ml.model.attributes import Attribute, TypeAttribute
from hls4ml.model.layers import Layer
from hls4ml.model.types import FixedPrecisionType, IntegerPrecisionType


class GravNetCore(Layer):
    _expected_attributes = [
        Attribute('n_vertices'),
        Attribute('n_out_features'),
        Attribute('n_neighbours'),
        Attribute('exponential_table', value_type=dict, default={'ScaleFactor': 2, 'Resolution': 16}, configurable=True),
        Attribute('exp_table_size', value_type=int),
        Attribute('exp_table_indexing_shmt', value_type=int),
        TypeAttribute('knn_idx'),
        TypeAttribute('knn_dist', configurable=True),
        TypeAttribute('exp_table', default=FixedPrecisionType(width=16, integer=0, signed=False), configurable=True),
        TypeAttribute('exp_table_idx', default=IntegerPrecisionType(width=4, signed=False)),
        TypeAttribute('weighted_feature', configurable=True),
    ]

    def initialize(self):
        n_vertices = self.get_attr('n_vertices', None)
        assert n_vertices is not None
        n_out_features = self.get_attr('n_out_features', None)
        assert n_out_features is not None
        self.add_output_variable([n_vertices, n_out_features])

        exp_table_params = self.get_attr('exponential_table', None)
        scale_factor = exp_table_params['ScaleFactor']
        resolution = exp_table_params['Resolution']

        if not log2(resolution).is_integer():
            raise ValueError('Exponential table resolution must be a power of two')
        if not log2(scale_factor).is_integer():
            raise ValueError('Scale factor must be a power of two')

        exp_table_size = scale_factor * resolution
        exp_table_indexing_shmt = int(log2(resolution))
        exp_table_size_nbits = int(log2(exp_table_size))

        self.set_attr('exp_table_size', exp_table_size)
        self.set_attr('exp_table_indexing_shmt', exp_table_indexing_shmt)
        self.set_attr('exp_table_idx_t', IntegerPrecisionType(width=exp_table_size_nbits, signed=False))

        knn_idx_bits = ceil(log2(n_vertices))
        self.set_attr('knn_idx_t', IntegerPrecisionType(width=knn_idx_bits, signed=False))
        self._set_type_t('knn_dist')
        self._set_type_t('exp_table')
        self._set_type_t('weighted_feature')
