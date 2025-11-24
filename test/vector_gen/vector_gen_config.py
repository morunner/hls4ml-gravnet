from dataclasses import dataclass


@dataclass(frozen=True)
class GenerateTestVectorsConfig:
    B: int = 1
    V: int = 128
    F: int = 4
    S: int = 8
    n_neighbours: int = 4
    vector_file_path: str = 'test/hls/include/test_vectors.h'


@dataclass(frozen=True)
class QGravNetLayerConfig:
    n_neighbours: int = 8
    n_dimensions: int = 8
    n_filters: int = 4
    n_propagate: int = 8
    name: str = 'gravnetlayer'


gen_config = GenerateTestVectorsConfig()
qgravnetlayer_config = QGravNetLayerConfig()
