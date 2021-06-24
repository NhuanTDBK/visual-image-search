import numpy as np
import tensorflow as tf
import tensorflow.keras.backend as K
import tensorflow_probability as tfp
from tensorflow.keras import layers


class CosineSimilarity(layers.Layer):
    """
    Cosine similarity with classwise weights
    """

    def __init__(self, num_classes, l2_wd=0.0, **kwargs):
        super().__init__(**kwargs)
        self.l2_wd = l2_wd
        self.num_classes = num_classes

    def build(self, input_shape):
        input_dim = input_shape[-1]
        self.W = self.add_weight(shape=(input_dim, self.num_classes),
                                 initializer='glorot_uniform',
                                 regularizer=tf.keras.regularizers.l2(self.l2_wd),
                                 name="cosine_w",
                                 trainable=True)

    def call(self, inputs, **kwargs):
        x = tf.nn.l2_normalize(inputs, axis=-1)  # (batch_size, ndim)
        w = tf.nn.l2_normalize(self.W, axis=0)  # (ndim, nclass)
        cos = tf.matmul(x, w)  # (batch_size, nclass)
        return cos

    def compute_output_shape(self, input_shape):
        return None, self.num_classes


class ArcFace(layers.Layer):
    """
    Implementation of https://arxiv.org/pdf/1801.07698.pdf
    """

    def __init__(self, num_classes, margin=0.5, scale=30, l2_wd = 1e-5, **kwargs):
        super().__init__(**kwargs)
        self.num_classes = num_classes
        self.margin = margin
        self.scale = scale
        self.cos_similarity = CosineSimilarity(num_classes,l2_wd)

    def call(self, inputs, training):
        # If not training (prediction), labels are ignored
        feature, labels = inputs
        cos = self.cos_similarity(feature)

        if training:
            theta = tf.acos(K.clip(cos, -1.0 + K.epsilon(), 1.0 - K.epsilon()))
            tf.summary.histogram("angles", theta)

            cos_add = tf.cos(theta + self.margin)
            mask = tf.cast(labels, dtype=cos_add.dtype)
            output = mask * cos_add + (1 - mask) * cos
            output = output * self.scale

            tf.summary.histogram("logits", output)
            return output
        else:
            return cos


class AdaCos(layers.Layer):
    def __init__(self, num_classes, margin=0.0, scale=0.0,l2_wd = 1e-5, **kwargs):
        super().__init__(**kwargs)
        self.num_classes = num_classes

        self.cos_similarity = CosineSimilarity(num_classes,l2_wd)
        self.scale = tf.Variable(tf.sqrt(2.0) * tf.math.log(num_classes - 1.0),
                                 trainable=False)

    def call(self, inputs, training):
        # In inference, labels are ignored
        feature = inputs
        cos = self.cos_similarity(feature)

        if training:
            labels = inputs[1]
            mask = tf.cast(labels, dtype=cos.dtype)

            # Collect cosine values at only false labels
            B = (1 - mask) * tf.exp(self.scale * cos)
            B_avg = tf.reduce_mean(tf.reduce_sum(B, axis=-1), axis=0)

            # theta = tf.acos(tf.clip_by_value(cos, -1, 1))
            theta = tf.acos(K.clip(cos, -1.0 + K.epsilon(), 1.0 - K.epsilon()))
            # Collect cosine at true labels
            theta_true = tf.reduce_sum(mask * theta, axis=-1)
            # get median (=50-percentile)
            theta_med = tfp.stats.percentile(theta_true, q=50)

            scale = tf.math.log(B_avg) / tf.cos(tf.minimum(np.pi / 4, theta_med))
            scale = tf.stop_gradient(scale)
            logits = scale * cos

            self.scale.assign(scale)
            return logits
        else:
            return cos

    def compute_output_shape(self, input_shape):
        return (None, self.num_classes)




metric_layer_dict = {
    "arcface": ArcFace,
    "adacos": AdaCos,
    "linear": layers.Dot

}

class MetricLearner(layers.Layer):
    def get_config(self):
        return {
            "n_classes": self.n_classes
        }

    def __init__(self, n_classes,
                 dropout=0.0,
                 metric="arcface",
                 s=30.0,
                 margin=0.5,
                 ls_eps=0.0,
                 theta_zero=0.85,
                 l2_wd = 0,
                 *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.n_classes = n_classes
        self.dropout = dropout

        self.s = s
        self.margin = margin
        self.ls_eps = ls_eps
        self.theta_zero = theta_zero

        self.metric_layer = metric_layer_dict[metric](num_classes=self.n_classes,
                                                      margin=self.margin,
                                                      scale=self.s,l2_wd=l2_wd)

    def call(self, inputs, training=None, mask=None):
        x, y = inputs
        x = self.metric_layer([x, y])
        x = layers.Softmax(dtype=x.dtype)(x)
        return x

    def compute_output_shape(self, input_shape):
        return None, self.n_classes
