import tensorflow as tf

from app.search.index import FlatIndex
from models.object_detection import extract_patches
import tensorflow_hub as hub


class Engine(object):
    object_detection_model: tf.keras.models.Model
    image_model: tf.keras.models.Model

    image_vector_index: FlatIndex
    patch_vector_index: FlatIndex

    def find_similar_patch(self, img,n_neighbors=50):
        img = tf.image.resize(img,(384,384))
        object_patches = extract_patches(self.object_detection_model,img)

        for i in range(len(object_patches)):
            self.patch_vector_index.k_nearest_neighbors(X,n_neighbors)

