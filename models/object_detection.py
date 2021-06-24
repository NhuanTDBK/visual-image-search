from typing import Dict

import tensorflow as tf


# def load_img(path: str):
#    contents = tf.io.read_file(path)
#    h, w = tf.image.extract_jpeg_shape(contents)
#    img = tf.image.decode_jpeg(contents, channels=3)
#    img, im_height, im_width = load_img(path)
#    converted_img = tf.image.convert_image_dtype(img, tf.float32)[tf.newaxis, ...]
#    return img, h, w


def extract_patches(detector: tf.keras.models.Model,
                    img: tf.TensorArray,
                    min_score: float = 0.4,
                    max_boxes: int = 10) -> Dict:
    im_height, im_width = tf.shape(img)
    result = detector(img)

    result = {key: value.numpy() for key, value in result.items()}

    boxes = result["detection_boxes"]
    entities = result["detection_class_entities"]
    scores = result["detection_scores"]

    examples = []

    for i in range(min(len(boxes), max_boxes)):
        if scores[i] >= min_score:
            example = {}
            ymin, xmin, ymax, xmax = tuple(boxes[i])
            class_name = entities[i].decode("ascii")
            xmin, xmax, ymin, ymax = (xmin * im_width, xmax * im_width, ymin * im_height, ymax * im_height)
            tmp = tf.image.crop_to_bounding_box(img, ymin, xmin, ymax - ymin, xmax - xmin)

            example["class_name"] = class_name
            example["arr"] = tmp.numpy()

            example["xmin"] = xmin
            example["xmax"] = xmax
            example["ymin"] = ymin
            example["ymax"] = ymax

            examples.append(example)

    return {
        "results": examples,
        "height": im_height,
        "width": im_width
    }
