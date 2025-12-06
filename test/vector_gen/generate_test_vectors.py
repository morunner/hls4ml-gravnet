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

    x_np = x.numpy()
    result_np = result.numpy()

    for b in range(gravnet_config.B):
        recorder.add(GlobalExchangeTestVector(name=f'global_exchange_{b}', x=x_np[b], expected_result=result_np[b]))


def test_gen_vector_gravnet_core(recorder):
    coords = tf.random.normal(shape=(gravnet_config.B, gravnet_config.V, gravnet_config.S), stddev=0.5)
    feats = tf.random.normal(shape=(gravnet_config.B, gravnet_config.V, gravnet_config.F), stddev=0.5)
    gravnet_core = GravNetCore(gravnet_config.n_neighbours)
    result = gravnet_core.call([coords, feats])

    coords_np = coords.numpy()
    feats_np = feats.numpy()
    result_np = result.numpy()

    for b in range(gravnet_config.B):
        recorder.add(
            GravnetCoreTestVector(
                name=f'gravnet_core_{b}', coords=coords_np[b], feats=feats_np[b], expected_result=result_np[b]
            )
        )
