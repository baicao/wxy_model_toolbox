import os
import opennsfw2 as n2

if __name__ == "__main__":

    # To get the NSFW probability of a single image.
    image_path = "/dockerdata/gisellewang/porn/test_pics/0.jpeg"
    MODEL_PATH = "/dockerdata/gisellewang/porn/opennsfw2-main/opennsfw2/weights/open_nsfw_weights.h5"

    nsfw_probability = n2.predict_image(image_path, weights_path=MODEL_PATH)

    # To get the NSFW probabilities of a list of images.
    # This is better than looping with `predict_image` as the model will only be instantiated once
    # and batching is used during inference.

    DATA_DIR = "/dockerdata/gisellewang/porn/test_pics"
    image_paths = os.listdir(DATA_DIR)
    image_paths = [x for x in image_paths if x.endswith(".jpg") or x.endswith(".jpeg")]
    image_paths = [os.path.join(DATA_DIR, x) for x in image_paths]

    rs = n2.predict_images(image_paths, weights_path=MODEL_PATH)
    print("nsfw_probabilities", rs)
