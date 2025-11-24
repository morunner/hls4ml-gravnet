import pytest
import tensorflow as tf
from qgravnet.layers import GlobalExchange, GravNetCore

from test.vector_gen.vector import (
    EuclideanSquaredKnnTestVector,
    EuclideanSquaredTestVector,
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
    x = tf.random.normal(shape=(gen_config.B, gen_config.V, gen_config.F))
    gex = GlobalExchange()
    result = gex.call(x)
    recorder.add(GlobalExchangeTestVector(name='global_exchange', x=x, expected_result=result))


def test_gen_vector_euclidean_squared(recorder):
    A_matrix = tf.random.normal(shape=(gen_config.B, gen_config.V, gen_config.S))
    gravnet_core = GravNetCore(gen_config.n_neighbours)
    result = gravnet_core._euclidean_squared(A_matrix, A_matrix)
    recorder.add(EuclideanSquaredTestVector(name='euclidean_squared', A=A_matrix, expected_result=result))


def test_gen_vector_euclidean_squared_knn(recorder):
    A_matrix = tf.random.normal(shape=(gen_config.B, gen_config.V, gen_config.S))

    # Calculate knn from euclidean squared distances at the same time
    gravnet_core = GravNetCore(gen_config.n_neighbours)
    dist = gravnet_core._euclidean_squared(A_matrix, A_matrix)
    dist = dist + tf.eye(gen_config.V, batch_shape=[gen_config.B]) * 1e9  # mask diagonal
    ranked_distances, ranked_indixes = tf.nn.top_k(-dist, k=gen_config.n_neighbours)
    ranked_distances = -ranked_distances

    recorder.add(
        EuclideanSquaredKnnTestVector(
            name='euclidean_squared_knn',
            A=A_matrix,
            expected_ranked_distances=ranked_distances,
            expected_ranked_indices=ranked_indixes,
            expected_result=None,  # not used here
        )
    )


def test_gen_vector_gravnet_core(recorder):
    coords = tf.random.normal(shape=(gen_config.B, gen_config.V, gen_config.S))
    feats = tf.random.normal(shape=(gen_config.B, gen_config.V, gen_config.F))
    gravnet_core = GravNetCore(gen_config.n_neighbours)
    result = gravnet_core.call(coords, feats)
    recorder.add(GravnetCoreTestVector(name='collect_neighbours', coords=coords, feats=feats, expected_result=result))
