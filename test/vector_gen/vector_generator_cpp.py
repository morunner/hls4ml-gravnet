import re
from collections import defaultdict
from dataclasses import fields

import numpy as np
from jinja2 import Template

from test.vector_gen.vector import TestVectorBase
from test.vector_gen.vector_gen_config import gen_config
from test.vector_gen.vector_template import template_str as vector_template_str


class VectorGeneratorCpp:
    def __init__(self):
        self.vectors = defaultdict(list)

    @staticmethod
    def get_fields(obj):
        return [f.name for f in fields(obj) if f.name != 'name' and getattr(obj, f.name) is not None]

    @staticmethod
    def is_array(obj, field_name):
        val = getattr(obj, field_name)
        return isinstance(val, np.ndarray)

    @staticmethod
    def get_attr(obj, field_name):
        return getattr(obj, field_name)

    @staticmethod
    def get_cpp_type(obj, field_name):
        val = getattr(obj, field_name)
        if isinstance(val, int):
            return 'int'
        if isinstance(val, float):
            return 'float'
        return 'auto'

    @staticmethod
    def format_cpp(value):
        if isinstance(value, np.ndarray):
            flat = value.flatten().tolist()
            if not flat:
                return '{ 0 }'
            inner = ', '.join(f'{x:.9g}' for x in flat)
            return f'{{ {inner} }}'
        if isinstance(value, (list, tuple)):
            return f"{{ {', '.join(str(x) for x in value)} }}"
        return str(value)

    def add(self, vector: TestVectorBase):
        cls_name = type(vector).__name__
        cls_name = re.sub(r'([a-z0-9])([A-Z])', r'\1_\2', cls_name)
        cls_name = cls_name.lower()
        self.vectors[cls_name].append(vector)

    def get_number_of_vectors(self, cls_name: str):
        return len(self.vectors[cls_name])

    def save_to_cpp(self, filename='test_vectors.h'):
        t = Template(vector_template_str)
        template_kwargs = {
            'B': gen_config.B,
            'V': gen_config.V,
            'F': gen_config.F,
            'S': gen_config.S,
            'n_neighbors': gen_config.n_neighbors,
            'cpp': self.format_cpp,
            'get_fields': self.get_fields,
            'is_array': self.is_array,
            'get_attr': self.get_attr,
            'get_cpp_type': self.get_cpp_type,
            'length': lambda x: x.size if isinstance(x, np.ndarray) else len(x),
            'nvectors': self.get_number_of_vectors,
        }

        with open(filename, 'w') as f:
            f.write(t.render(data=self.vectors, **template_kwargs))
        print(f'[Gen] Generated vectors for classes: {list(self.vectors.keys())}')
