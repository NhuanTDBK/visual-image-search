import numpy as np
import tensorflow as tf


def _to_int32(a):
    return np.int32(np.ceil(a))


def extract_patches(detector: tf.keras.models.Model,
                    img: tf.TensorArray,
                    min_score: float = 0.4,
                    max_boxes: int = 10):
    shape = tf.shape(img)
    im_height, im_width = shape[0].numpy(), shape[1].numpy()
    result = detector(img[tf.newaxis, ...])

    result = {key: value.numpy() for key, value in result.items()}

    boxes = result["detection_boxes"][0]
    # entities = result["detection_class_entities"]
    scores = result["detection_scores"][0]

    examples = []

    for i in range(min(len(boxes), max_boxes)):
        if scores[i] >= min_score:
            example = {}
            ymin, xmin, ymax, xmax = tuple(boxes[i])
            # class_name = entities[i].decode("ascii")

            xmin, xmax, ymin, ymax = _to_int32(xmin * im_width), _to_int32(xmax * im_width), _to_int32(
                ymin * im_height), _to_int32(ymax * im_height)
            tmp = tf.image.crop_to_bounding_box(img, ymin, xmin, ymax - ymin, xmax - xmin)

            # example["class_name"] = class_name
            example["arr"] = tmp.numpy()
            example["score"] = scores[i]

            example["bounding_box"] = (xmin, xmax, ymin, ymax)

            examples.append(example)

    return {
        "results": examples,
        "height": im_height,
        "width": im_width
    }
