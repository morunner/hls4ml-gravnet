from math import ceil, log2

from hls4ml.model.attributes import Attribute, TypeAttribute
from hls4ml.model.layers import Layer
from hls4ml.model.types import IntegerPrecisionType


class GravNetCore(Layer):
    _expected_attributes = [
        Attribute('n_vertices'),
        Attribute('n_out_features'),
        Attribute('n_neighbours'),
        TypeAttribute('knn_idx_t'),
        TypeAttribute('knn_dist_t', configurable=True),
        TypeAttribute('exp_t', configurable=True),
        TypeAttribute('weighted_feature_t', configurable=True),
    ]

    def initialize(self):
        n_vertices = self.get_attr('n_vertices', None)
        assert n_vertices is not None
        n_out_features = self.get_attr('n_out_features', None)
        assert n_out_features is not None
        self.add_output_variable([n_vertices, n_out_features])

        knn_idx_bits = ceil(log2(n_vertices))
        self.set_attr('knn_idx_t', IntegerPrecisionType(width=knn_idx_bits, signed=False))
        self._set_type_t('knn_dist')
        self._set_type_t('exp')
        self._set_type_t('weighted_feature')
