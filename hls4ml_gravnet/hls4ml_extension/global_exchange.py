from hls4ml.model.attributes import Attribute, TypeAttribute
from hls4ml.model.layers import Layer


class HGlobalExchange(Layer):
    _expected_attributes = [Attribute('n_vertices'), Attribute('n_out_features'), TypeAttribute('mean', configurable=True)]

    def initialize(self):
        n_vertices = self.get_attr('n_vertices', None)
        assert n_vertices is not None
        n_out_features = self.get_attr('n_out_features', None)
        assert n_out_features is not None
        self.add_output_variable([n_vertices, n_out_features])

        self._set_type_t('mean')
