from dataclasses import asdict

import numpy as np
import pytest
import tensorflow as tf

from quantized_gravnet.qgravnet.layers import GlobalExchange, QGravNetLayer
from test.vector_gen.vector import CollectNeighboursTestVector, EuclideanSquaredTestVector, GlobalExchangeTestVector
from test.vector_gen.vector_gen_config import gen_config, qgravnetlayer_config
from test.vector_gen.vector_generator_cpp import VectorGeneratorCpp


@pytest.fixture(scope='session')
def recorder():
    rec = VectorGeneratorCpp()
    yield rec
    rec.save_to_cpp(gen_config.vector_file_path)


def test_gen_vector_global_exchange(recorder) -> np.ndarray:
    x = tf.random.normal(shape=(gen_config.B, gen_config.V, gen_config.F))
    gex = GlobalExchange()
    result = gex.call(x)
    recorder.add(GlobalExchangeTestVector(name='global_exchange', x=x, expected_result=result))


def test_gen_vector_euclidean_squared(recorder) -> np.ndarray:
    A_matrix = tf.random.normal(shape=(gen_config.B, gen_config.V, gen_config.F))
    B_matrix = tf.random.normal(shape=(gen_config.B, gen_config.V, gen_config.F))
    result = QGravNetLayer._euclidean_squared(A_matrix, B_matrix)
    recorder.add(EuclideanSquaredTestVector(name='euclidean_squared', A=A_matrix, B=B_matrix, expected_result=result))


def test_gen_vector_collect_neighbours(recorder) -> np.ndarray:
    coords = tf.random.normal(shape=(gen_config.B, gen_config.V, gen_config.F))
    feats = tf.random.normal(shape=(gen_config.B, gen_config.V, gen_config.F))
    result = QGravNetLayer(**asdict(qgravnetlayer_config)).collect_neighbours(coords, feats)
    recorder.add(CollectNeighboursTestVector(name='collect_neighbours', coords=coords, feats=feats, expected_result=result))
