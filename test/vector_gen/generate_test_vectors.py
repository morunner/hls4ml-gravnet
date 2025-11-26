import pytest
import tensorflow as tf
from qgravnet.layers import GlobalExchange, GravNetCore

from test.vector_gen.gravnet_config import gravnet_config
from test.vector_gen.vector import (
    GlobalExchangeTestVector,
    GravnetCoreTestVector,
)
from test.vector_gen.vector_gen_config import gen_config
from test.vector_gen.vector_generator_cpp import VectorGeneratorCpp


@pytest.fixture(scope='session')
def recorder():
    rec = VectorGeneratorCpp()
    yield rec
    rec.save_to_cpp(gen_config.vector_file_path)


def test_gen_vector_global_exchange(recorder):
    x = tf.random.normal(shape=(gravnet_config.B, gravnet_config.V, gravnet_config.F))
    gex = GlobalExchange()
    result = gex.call(x)
    recorder.add(GlobalExchangeTestVector(name='global_exchange', x=x, expected_result=result))


def test_gen_vector_gravnet_core(recorder):
    coords = tf.random.normal(shape=(gravnet_config.B, gravnet_config.V, gravnet_config.S), mean=10)
    feats = tf.random.normal(shape=(gravnet_config.B, gravnet_config.V, gravnet_config.F), mean=10)
    gravnet_core = GravNetCore(gravnet_config.n_neighbors)
    result = gravnet_core.call(coords, feats)
    recorder.add(GravnetCoreTestVector(name='collect_neighbors', coords=coords, feats=feats, expected_result=result))
