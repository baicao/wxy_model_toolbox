#!/usr/bin/env python
import os
import sys
import time
from platform import system
from multiprocessing.pool import ThreadPool
import traceback
import requests
import numpy as np

import tensorflow.compat.v1 as tf  # pylint: disable=import-error

tf.disable_v2_behavior()

if system() == "Darwin":
    MODEL_DIR = "/Users/xiangyuwang/Software/"
elif system() == "Linux":
    MODEL_DIR = "/dockerdata/keywords_model"
else:
    sys.exit()
sys.path.append(MODEL_DIR)
# pylint: disable=wrong-import-position
from open_nsfw.model_tf2 import OpenNsfwModel, InputType
from open_nsfw.image_utils import create_tensorflow_image_loader
from open_nsfw.image_utils import create_yahoo_image_loader
from common.log_factory import logger

# pylint: enable=wrong-import-position

IMAGE_LOADER_TENSORFLOW = "tensorflow"
IMAGE_LOADER_YAHOO = "yahoo"


class NSFWPredicter(object):
    def __init__(self, input_type=None, image_loader=None) -> None:
        if input_type is not None and input_type in [
            InputType.TENSOR.name.lower(),
            InputType.BASE64_JPEG.name.lower(),
        ]:
            self.input_type = input_type
        else:
            self.input_type = InputType.TENSOR.name.lower()

        if image_loader is not None and image_loader in [
            IMAGE_LOADER_YAHOO,
            IMAGE_LOADER_TENSORFLOW,
        ]:
            self.image_loader = IMAGE_LOADER_YAHOO
        else:
            self.image_loader = IMAGE_LOADER_YAHOO
        self.package_name = "open_nsfw"
        self.logger = logger
        self.temp_dir = "temp_image"
        if not os.path.exists(self.temp_dir):
            os.mkdir(self.temp_dir)
        self.num_processes = 5
        self.package_dir = self._found_install_dir(self.package_name)
        print("package_dir", self.package_dir)
        model_weights = os.path.join(self.package_dir, "data/open_nsfw-weights.npy")
        try:
            # pylint: disable=import-outside-toplevel
            from common.gpu_manager import GPUManager

            gm = GPUManager()
            gpu_id = gm.auto_choice_gpuid()
            os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
            print("ini porn image predict")
            # pylint: enable=import-outside-toplevel

        except:  # pylint: disable=bare-except
            pass
        self.model = OpenNsfwModel()
        print("input_type", self.input_type)
        self.model.build(
            weights_path=model_weights, input_type=InputType[self.input_type.upper()]
        )

    def _found_install_dir(self, package_name):
        installed_dir = None
        for p in sys.path:
            installed_dir = os.path.join(p, package_name)
            if os.path.exists(installed_dir):
                self.logger.info("found installed dir -> %s", installed_dir)
                break
        return installed_dir

    # 多线程下载图片
    def download_imgs(self, image_url_list, name=""):
        def download(root_dir, img_url, name):
            print(f"img_url:{img_url}")
            resp = requests.get(img_url)
            if resp.status_code == 200:
                try:
                    resp = requests.get(img_url)
                    temp_file = os.path.join(root_dir, "{}.jpg".format(name))
                    with open(temp_file, "wb") as f:
                        f.write(resp.content)
                    return (img_url, temp_file)
                except:  # pylint: disable=bare-except
                    self.logger.error(
                        "下载失败: %s %s -> %s", name, img_url, traceback.format_exc()
                    )
            return (img_url, None)

        pool = ThreadPool(processes=self.num_processes)
        thread_list = []

        if name == "":
            name = str(int(time.time()))

        for i, image_url in enumerate(image_url_list):
            temp_name = f"{i}_{name}"
            out = pool.apply_async(
                func=download, args=(self.temp_dir, image_url, temp_name)
            )
            thread_list.append(out)

        pool.close()
        pool.join()
        # 获取输出结果
        image_list = []
        for p in thread_list:
            output = p.get()  # get会阻塞
            image_list.append(
                {
                    "url": output[0],
                    "file": output[1],
                }
            )

        return image_list

    # 多线程解析图片
    def parse_image_files(self, file_list):
        def _parse_image_file(index, input_file):
            input_type = InputType[self.input_type.upper()]

            fn_load_image = None

            if input_type == InputType.TENSOR:
                if self.image_loader == IMAGE_LOADER_TENSORFLOW:
                    fn_load_image = create_tensorflow_image_loader(
                        tf.Session(graph=tf.Graph())
                    )
                else:
                    fn_load_image = create_yahoo_image_loader()
            elif input_type == InputType.BASE64_JPEG:
                import base64  # pylint: disable=import-outside-toplevel

                def fn_load_image(filename):  # pylint: disable = function-redefined
                    return np.array(
                        [base64.urlsafe_b64encode(open(filename, "rb").read())]
                    )

            image = fn_load_image(input_file)
            return (index, image)

        pool = ThreadPool(processes=self.num_processes)
        thread_list = []

        for i, file in enumerate(file_list):
            if file["file"] is not None and os.path.exists(file["file"]):
                out = pool.apply_async(func=_parse_image_file, args=(i, file["file"]))
                thread_list.append(out)

        pool.close()
        pool.join()
        # 获取输出结果
        image_list = []
        index = []
        for p in thread_list:
            output = p.get()  # get会阻塞
            image_list.append(output[1][0])
            index.append(output[0])

        return image_list, index

    def predict_bulk(self, url_list, name=""):
        file_list = self.download_imgs(url_list, name)
        image_list, index_list = self.parse_image_files(file_list)
        input_image = np.array(image_list)
        with tf.Session() as sess:
            sess.run(tf.global_variables_initializer())

            predictions = sess.run(
                self.model.predictions, feed_dict={self.model.input: input_image}
            )
            # print(predictions)
        pred = np.argmax(predictions, axis=1)
        prob = np.max(predictions, axis=1)
        for i in range(predictions.shape[0]):
            index = index_list[i]
            file_list[index]["pred"] = pred[i]
            file_list[index]["prob"] = prob[i]
        # print("file_list", file_list)
        return file_list

    def parse_image_file(self, input_file):
        input_type = InputType[self.input_type.upper()]

        fn_load_image = None

        if input_type == InputType.TENSOR:
            if self.image_loader == IMAGE_LOADER_TENSORFLOW:
                fn_load_image = create_tensorflow_image_loader(
                    tf.Session(graph=tf.Graph())
                )
            else:
                fn_load_image = create_yahoo_image_loader()
        elif input_type == InputType.BASE64_JPEG:
            import base64  # pylint: disable=import-outside-toplevel

            def fn_load_image(filename):  # pylint: disable = function-redefined
                return np.array([base64.urlsafe_b64encode(open(filename, "rb").read())])

        image = fn_load_image(input_file)
        return image

    def predict(self, input_file):
        image = self.parse_image_file(input_file)
        with tf.Session() as sess:
            sess.run(tf.global_variables_initializer())

            predictions = sess.run(
                self.model.predictions, feed_dict={self.model.input: image}
            )
            print(predictions)


if __name__ == "__main__":
    FILE = "WechatIMG54369.jpeg"
    FILE2 = "WechatIMG54258.jpeg"
    predicter = NSFWPredicter()
    # predicter.predict(file, file2)
    image_url_list = [
        "http://q1.qlogo.cn/qhis/RCfVSK6LJ8BKklocEMCQ75qdvdny06yv1xkcA8UFgRGYhKR3OnILzjFJVMPxrnliaibfr5ibsibCdoI/640",
    ]
    # file_list = predicter.download_imgs(image_url_list, "test")
    # image_list, index = predicter.parse_image_files(file_list)
    rs = predicter.predict_bulk(image_url_list, name="test")
    print(rs)
