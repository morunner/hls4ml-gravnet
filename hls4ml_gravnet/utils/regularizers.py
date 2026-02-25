import tensorflow as tf

from qgravnet.selectors import BinnedSelector
from .evaluation import get_model_coord_layers


def add_overflow_regularization(
    model: tf.keras.Model,
    selector: BinnedSelector,
    C=2,
    lambda_overflow=1e-4,
    monitor=False,
):

    for i, layer in enumerate(get_model_coord_layers(model)):
        coords = layer.output  # (B, V, S)

        bins = selector.compute_bins(coords)  # (B, V, S)
        B = selector.bins_per_axis

        S = bins.shape[-1] 
        total_bins = B ** S

        bins = tf.cast(bins, tf.int64)

        multipliers = tf.cast(
            tf.pow(
                tf.constant(B, dtype=tf.int64),
                tf.range(tf.shape(bins)[-1], dtype=tf.int64)
            ),
            tf.int64
        )

        flat = tf.reduce_sum(bins * multipliers, axis=-1)

        counts = tf.math.bincount(
            flat,
            minlength=total_bins,
            maxlength=total_bins,
            axis=-1,
            dtype=tf.float32,
        )

        max_counts = tf.reduce_max(counts, axis=-1)  # (B,)

        # ----- MONITOR METRIC -----
        if monitor:
            print(f"Adding overflow regularization monitoring for block {i} with C={C} and lambda={lambda_overflow}")
            avg_max = tf.reduce_mean(max_counts)
            model.add_metric(
                avg_max,
                name=f"block{i}_avg_max_bin",
                aggregation="mean",
            )

        # ----- OVERFLOW LOSS -----
        if lambda_overflow > 0 and C is not None:
            overflow = tf.nn.relu(max_counts - float(C))
            loss = tf.reduce_mean(tf.square(overflow))

            model.add_loss(lambda_overflow * loss)