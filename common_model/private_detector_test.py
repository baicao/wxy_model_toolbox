import os
from typing import List

import tensorflow as tf
from absl import logging as absl_logging
from private_detector.utils.preprocess import preprocess_for_evaluation


def read_image(filename: str) -> tf.Tensor:
    """
    Load and preprocess image for inference with the Private Detector

    Parameters
    ----------
    filename : str
        Filename of image

    Returns
    -------
    image : tf.Tensor
        Image ready for inference
    """
    image = tf.io.read_file(filename)
    image = tf.io.decode_jpeg(image, channels=3)

    image = preprocess_for_evaluation(image, 480, tf.float16)

    image = tf.reshape(image, -1)

    return image


def inference(model: str, image_paths: List[str]) -> None:
    """
    Get predictions with a Private Detector model

    Parameters
    ----------
    model : str
        Path to saved model
    image_paths : List[str]
        Path(s) to image to be predicted on
    """
    model = tf.saved_model.load(model)

    for image_path in image_paths:
        image = read_image(image_path)

        preds = model([image])

        print(
            f"Probability: {100 * tf.get_static_value(preds[0])[0]:.2f}% - {image_path}"
        )


if __name__ == "__main__":
    tf.get_logger().setLevel("ERROR")
    absl_logging.set_verbosity(absl_logging.ERROR)

    DATA_DIR = "/dockerdata/gisellewang/porn/test_pics"
    image_paths = os.listdir(DATA_DIR)
    image_paths = [x for x in image_paths if x.endswith(".jpg") or x.endswith(".jpeg")]
    image_paths = [os.path.join(DATA_DIR, x) for x in image_paths]

    inference(model="./private_detector/saved_model", image_paths=image_paths)
