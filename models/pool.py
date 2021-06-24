from enum import Enum

import numpy as np
import tensorflow as tf



class MAC(tf.keras.layers.Layer):
    """Global max pooling (MAC) layer.
     Maximum Activations of Convolutions (MAC) is simply constructed by
     max-pooling over all dimensions per feature map. See
     https://arxiv.org/abs/1511.05879 for a reference.
    """

    def call(self, x, axis=None):
        """Invokes the MAC pooling instance.
        Args:
          x: [B, H, W, D] A float32 Tensor.
          axis: Dimensions to reduce. By default, dimensions [1, 2] are reduced.
        Returns:
          output: [B, D] A float32 Tensor.
        """
        if axis is None:
            axis = [1, 2]
        return mac(x, axis=axis)


class SPoC(tf.keras.layers.Layer):
    """Average pooling (SPoC) layer.
    Sum-pooled convolutional features (SPoC) is based on the sum pooling of the
    deep features. See https://arxiv.org/pdf/1510.07493.pdf for a reference.
    """

    def call(self, x, axis=None):
        """Invokes the SPoC instance.
        Args:
          x: [B, H, W, D] A float32 Tensor.
          axis: Dimensions to reduce. By default, dimensions [1, 2] are reduced.
        Returns:
          output: [B, D] A float32 Tensor.
        """
        if axis is None:
            axis = [1, 2]
        return spoc(x, axis)


class GeM(tf.keras.layers.Layer):
    """Generalized mean pooling (GeM) layer.
    Generalized Mean Pooling (GeM) computes the generalized mean of each
    channel in a tensor. See https://arxiv.org/abs/1711.02512 for a reference.
    """

    def __init__(self, power=3.):
        """Initialization of the generalized mean pooling (GeM) layer.
        Args:
          power:  Float power > 0 is an inverse exponent parameter, used during the
            generalized mean pooling computation. Setting this exponent as power > 1
            increases the contrast of the pooled feature map and focuses on the
            salient features of the image. GeM is a generalization of the average
            pooling commonly used in classification networks (power = 1) and of
            spatial max-pooling layer (power = inf).
        """
        super(GeM, self).__init__()
        self.power = power
        self.eps = 1e-6

    def call(self, x, axis=None):
        """Invokes the GeM instance.
        Args:
          x: [B, H, W, D] A float32 Tensor.
          axis: Dimensions to reduce. By default, dimensions [1, 2] are reduced.
        Returns:
          output: [B, D] A float32 Tensor.
        """
        if axis is None:
            axis = [1, 2]
        return gem(x, power=self.power, eps=self.eps, axis=axis)


def mac(x, axis=None):
    """Performs global max pooling (MAC).
    Args:
      x: [B, H, W, D] A float32 Tensor.
      axis: Dimensions to reduce. By default, dimensions [1, 2] are reduced.
    Returns:
      output: [B, D] A float32 Tensor.
    """
    if axis is None:
        axis = [1, 2]
    return tf.reduce_max(x, axis=axis, keepdims=False)


def spoc(x, axis=None):
    """Performs average pooling (SPoC).
    Args:
      x: [B, H, W, D] A float32 Tensor.
      axis: Dimensions to reduce. By default, dimensions [1, 2] are reduced.
    Returns:
      output: [B, D] A float32 Tensor.
    """
    if axis is None:
        axis = [1, 2]
    return tf.reduce_mean(x, axis=axis, keepdims=False)


def gem(x, axis=None, power=3., eps=1e-6):
    """Performs generalized mean pooling (GeM).
    Args:
      x: [B, H, W, D] A float32 Tensor.
      axis: Dimensions to reduce. By default, dimensions [1, 2] are reduced.
      power: Float, power > 0 is an inverse exponent parameter (GeM power).
      eps: Float, parameter for numerical stability.
    Returns:
      output: [B, D] A float32 Tensor.
    """
    if axis is None:
        axis = [1, 2]
    tmp = tf.pow(tf.maximum(x, eps), power)
    out = tf.pow(tf.reduce_mean(tmp, axis=axis, keepdims=False), 1. / power)
    return out


pooling_dict = {
    "mac": MAC,
    "gem": GeM,
    "spoc": SPoC,

    "global_avg_1d": tf.keras.layers.GlobalAveragePooling1D,
    "global_max_1d": tf.keras.layers.GlobalMaxPool1D,
}


class LocalGlobalExtractor(tf.keras.layers.Layer):
    def __init__(self, pool, fc_dim=512, dropout_rate=0.5, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.fts = tf.keras.models.Sequential(
            [
                tf.keras.layers.Dropout(dropout_rate,name="dropout_lb"),
                tf.keras.layers.Dense(fc_dim,name="fc_lb"),
                tf.keras.layers.BatchNormalization(name="bn_lb")
            ])
        self.pool_layer = pooling_dict[pool]()

    def call(self, inputs, **kwargs):
        x = inputs
        x = self.pool_layer(x)
        x = self.fts(x)

        return x
