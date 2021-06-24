from typing import List

import tensorflow as tf


def extract_image_vector(model: tf.keras.models.Model, X: tf.TensorArray) -> List[float]:
    y = model(X)

    return y
