from dataclasses import dataclass, field
from math import ceil, log2


@dataclass(frozen=True)
class GravNetConfig:
    B: int = 4
    V: int = 128
    F: int = 4
    S: int = 8
    n_neighbours: int = 4
    distance_metric: str = 'l2_squared'
    exp_table_scale_factor: int = 2
    exp_table_resolution: int = 16
    exp_table_size: int = field(init=False)
    exp_table_size_nbits: int = field(init=False)
    exp_table_indexing_shmt: int = field(init=False)

    def __post_init__(self):
        exp_table_size = self.exp_table_scale_factor * self.exp_table_resolution
        exp_table_size_nbits = int(ceil(log2(exp_table_size)))
        exp_table_indexing_shmt = int(ceil(log2(self.exp_table_resolution)))

        object.__setattr__(self, 'exp_table_size', exp_table_size)
        object.__setattr__(self, 'exp_table_size_nbits', exp_table_size_nbits)
        object.__setattr__(self, 'exp_table_indexing_shmt', exp_table_indexing_shmt)


gravnet_config = GravNetConfig()
