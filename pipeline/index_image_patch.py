import argparse

import efficientnet.tfkeras as efn
import numpy as np
import tensorflow as tf
import tensorflow_hub as hub


class Example(object):
    img_id: int
    example_id: int
    class_name: str
    arr: np.ndarray
    emb_arr: np.ndarray
    salient_colors: np.ndarray


def load_img(path):
    contents = tf.io.read_file(path)
    h, w = tf.image.extract_jpeg_shape(contents)
    img = tf.image.decode_jpeg(contents, channels=3)

    return img, h, w


def extract_patches(detector, path, min_score=0.4, max_boxes=10):
    img, im_height, im_width = load_img(path)

    converted_img = tf.image.convert_image_dtype(img, tf.float32)[tf.newaxis, ...]
    result = detector(converted_img)

    result = {key: value.numpy() for key, value in result.items()}

    boxes = result["detection_boxes"]
    entities = result["detection_class_entities"]
    scores = result["detection_scores"]

    examples = []

    for i in range(min(len(boxes), max_boxes)):
        if scores[i] >= min_score:
            ymin, xmin, ymax, xmax = tuple(boxes[i])
            class_name = entities[i].decode("ascii")
            xmin, xmax, ymin, ymax = (xmin * im_width, xmax * im_width, ymin * im_height, ymax * im_height)
            tmp = tf.image.crop_to_bounding_box(img, ymin, xmin, ymax - ymin, xmax - xmin)

            example = Example()
            example.class_name = class_name
            example.arr = tmp.numpy()

            examples.append(example)

    return examples


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("--detection_model_name", type=str,
                        default='https://tfhub.dev/google/faster_rcnn/openimages_v4/inception_resnet_v2/1')
    parser.add_argument("--emb_model_name", type=str, default='effb7')
    parser.add_argument("--epochs", type=int, default=25)
    parser.add_argument("--image_size", type=int, default=512)

    args = parser.parse_args()
    params = vars(args)

    return params


if __name__ == '__main__':
    image_extractor_mapper = {
        "b0": efn.EfficientNetB0,
        "b1": efn.EfficientNetB1,
        "b2": efn.EfficientNetB2,
        "b3": efn.EfficientNetB3,
        "b4": efn.EfficientNetB4,
        "b5": efn.EfficientNetB5,
        "b6": efn.EfficientNetB6,
        "b7": efn.EfficientNetB7
    }

    params = parse_args()

    detector = hub.load(params["detection_model_name"]).signatures['default']


    list_of_files = []
    track_id = 100

    for fpath in list_of_files:
        patches = extract_patches(detector, fpath, )
        for i in range(len(patches)):
            patches[i].img_id = track_id
            patches[i].example_id = track_id + i

            track_id += 1

    emb_model = image_extractor_mapper[params["emb_model_name"]](include_top=False, weights="imagenet", )

