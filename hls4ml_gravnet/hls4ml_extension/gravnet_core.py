from math import ceil, log2

from hls4ml.model.attributes import Attribute, TypeAttribute
from hls4ml.model.layers import Layer
from hls4ml.model.types import FixedPrecisionType, IntegerPrecisionType


class HGravNetCore(Layer):
    _expected_attributes = [
        Attribute('V'),
        Attribute('S'),
        Attribute('F'),
        Attribute('n_neighbours'),
        Attribute('exponential_table', value_type=dict, default={'ScaleFactor': 2, 'Resolution': 16}, configurable=True),
        Attribute('exp_table_size', value_type=int),
        Attribute('exp_table_indexing_shmt', value_type=int),
        TypeAttribute('coords_diff', configurable=True),
        TypeAttribute('knn_idx'),
        TypeAttribute('knn_dist', configurable=True),
        TypeAttribute('exp_table', default=FixedPrecisionType(width=16, integer=1, signed=False), configurable=True),
        TypeAttribute('exp_table_idx'),
        TypeAttribute('weighted_feature', configurable=True),
        TypeAttribute('accum', configurable=True),
    ]

    def initialize(self):
        n_vertices = self.get_attr('V', None)
        n_in_features = self.get_attr('F', None)
        assert n_vertices is not None
        assert n_in_features is not None
        self.add_output_variable([n_vertices, 2 * n_in_features])

        # We assert ExponentialTable params already here instead of the template
        # because the exponential table index data type depends on them
        exp_table_params = self.get_attr('exponential_table', None)
        if exp_table_params is None:
            scale_factor = 2
            resolution = 16
            exp_table_params = {'ScaleFactor': scale_factor, 'Resolution': resolution}
            self.set_attr('exponential_table', exp_table_params)
        else:
            if exp_table_params['ScaleFactor'] is None or exp_table_params['Resolution'] is None:
                raise ValueError('Please set a value for both ScaleFactor and Resolution in ExponentialTable config dict')
            scale_factor = exp_table_params['ScaleFactor']
            resolution = exp_table_params['Resolution']

        if not log2(resolution).is_integer():
            raise ValueError('Exponential table resolution must be a power of two')

        exp_table_size = scale_factor * resolution
        exp_table_indexing_shmt = int(ceil(log2(resolution)))
        exp_table_size_nbits = int(ceil(log2(exp_table_size)))

        self.set_attr('exp_table_size', exp_table_size)
        self.set_attr('exp_table_indexing_shmt', exp_table_indexing_shmt)
        self.set_attr('exp_table_idx_t', IntegerPrecisionType(width=exp_table_size_nbits, signed=False))

        knn_idx_bits = ceil(log2(n_vertices))
        self.set_attr('knn_idx_t', IntegerPrecisionType(width=knn_idx_bits, signed=False))
        self._set_type_t('coords_diff')
        self._set_type_t('exp_table')
        self._set_type_t('knn_dist')
        self._set_type_t('weighted_feature')
        self._set_type_t('accum')
