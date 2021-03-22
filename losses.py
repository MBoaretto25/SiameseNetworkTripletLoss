import tensorflow as tf


def triplet_loss(alpha):
    def loss(y_true, y_pred):
        anchor, positive, negative = y_pred[0], y_pred[1], y_pred[2]
        positive_distance = tf.math.sqrt(tf.math.reduce_sum(tf.math.square(anchor - positive), axis=-1))
        negative_distance = tf.math.sqrt(tf.math.reduce_sum(tf.math.square(anchor - negative), axis=-1))
        loss = tf.math.maximum(0.0, positive_distance - negative_distance + alpha)
        return loss
    return loss
