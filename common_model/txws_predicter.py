import sys
import os
from platform import system
from collections import deque
from typing import Any, Callable, Deque, List
import tensorflow as tf
import numpy as np

if system() == "Darwin":
    MODEL_DIR = "/Users/xiangyuwang/Software/"
elif system() == "Linux":
    MODEL_DIR = "/dockerdata/gisellewang"
else:
    sys.exit()
sys.path.insert(0, MODEL_DIR)
# pylint: disable=wrong-import-position
from txws_text_classification.utils.log_util import LogFactory
from txws_text_classification.utils.tokenizer import Tokenizer
from txws_text_classification.utils.token_seq_builder import HAN_Token2Matrix
from txws_text_classification.utils.base_util import read_token_map
from txws_text_classification.utils.const import (
    TC_CRF,
    TC_OTHER_NE,
    TC_NER_DL,
    TC_IP,
    TC_VIDEO,
    TC_PRODUCTION,
    TC_PER_W,
    TC_ORG_W,
    TC_LOC_W,
    TC_VIDEO_W,
    TC_RUL,
    TC_CUS,
    QQSEG_METHOD,
    JIEBA_METHOD,
    COARS_GRAINED_METHOD,
)
from txws_text_classification.utils.base_util import tf_version
from common.log_factory import logger
# pylint: enable=wrong-import-position


class TextInferenceBase(object):

    def __init__(
        self,
        model_dir: str = None,
        logger: LogFactory = None,
        tokenizer: Tokenizer = None,
        token_2_ids_builder: HAN_Token2Matrix = None,
        segment_method: int = QQSEG_METHOD,
    ) -> None:
        if logger is None:
            self.logger = LogFactory(
                log_dir="logs",
                log_prefix="text_inference.log",
                scope_name="text_inference",
                use_webhook=False,
            )
        self.logger = logger
        if not os.path.exists(model_dir):
            error_info = "model_dir -> {} did not exist!please check you param!".format(
                model_dir)
            self.logger.info(error_info)
            raise FileNotFoundError(error_info)

        if os.path.isfile(model_dir):
            error_info = "{} is a file,but we expoected a directory!".format(
                model_dir)
            self.logger.fatal(error_info)
            raise ValueError(error_info)

        self.model_dir = model_dir
        if tokenizer is None:
            error_info = "param tokenizer can not be None,please check...."
            self.logger.fatal(error_info)
            raise ValueError(error_info)
        self.tokenizer = tokenizer

        if token_2_ids_builder is None:
            error_info = "param token_2_ids_builder is None which is not expected,please check..."
            self.logger.info(error_info)
            raise ValueError(error_info)
        self.token_2_ids_builder = token_2_ids_builder
        self.segment_method = segment_method

        # set a flag to avoid reload tf model!
        self.model_already_load = False

    def _single_text_preprocess(self, text: str = None) -> List:  # pylint: disable=method-hidden
        """
        handle single text
        Args:
            text:str
        Returns:
            cleaned_text:str
        """
        cleaned_text = []
        text_split = text.split("|")
        for chat_content in text_split:
            chat_pair = chat_content.split(":", 1)
            if len(chat_pair) == 2:
                chat = chat_pair[1]
            elif len(chat_pair) == 1:
                chat = chat_pair[0]
            else:
                continue
            chat = self.tokenizer.encode_tiny_digit(chatmsg=chat)
            chat = self.tokenizer.drop_invalid_chars_with_single_text(
                text=chat)
            if chat == "":
                continue
            cleaned_text.append(chat)
        return cleaned_text

    def set_token_id_map(self, token_id_map: dict = None):
        self.token_2_ids_builder.token_map = token_id_map

    def set_empty_text_category(self, category: str) -> None:
        self.logger.debug(
            "set the empty text string with category -> {}".format(category))
        self.empty_text_category = category

    def set_empty_text_prob(self, prob: float = 1.0) -> None:
        self.logger.debug("set the empty text prob -> {}".format(prob))
        self.empty_text_prob = prob

    def set_id_2_label_dict(self, id_2_label_dict: dict = None) -> None:
        self.logger.debug("set id_2_label_dict -> {}".format(id_2_label_dict))
        self.id_2_label_dict = id_2_label_dict

    def registry_single_text_preprocess_function(self,
                                                 text_op: Callable = None
                                                 ) -> None:
        """
        registry you customed text preprocess function,then will use this function to
            handle our text
        Args:
            text_op:Callable,should be a function and accept one param at most!
        Raise:
            ValueError:if the text_op is not unexpected!
        """
        # check the specified op!\
        try:
            test_text = "举报人:我是花开富贵!"
            text_op(test_text)
        except Exception as e:
            error_info = "the text_op is not expected,will occured error -> {} while invoke!".format(
                str(e))
            self.logger.fatal(error_info)
            raise RuntimeError(error_info)
        self.logger.info("registry custome preporcess text function!")

        self._single_text_preprocess = text_op

    def build_input_ids(self, eval_text_list: List = None) -> np.ndarray:
        """
        build the input ids for specify text!
        Args:
            eval_text_list:a container of text!
        Returns:
            input_ids:array,maybe a 3-D array!
        """
        # container to store the chatmsg after split
        split_chatmsg_list = deque()
        # store the list size of each split chatmsg!
        split_chatmsg_size_list = deque()
        # some text maybe empty,we should track these data!
        valid_indicis_seq = deque()
        invalid_indicis_seq = deque()
        for indicis, text in enumerate(eval_text_list):
            chat_content_list = self._single_text_preprocess(text=text)
            if len(chat_content_list) == 0:
                invalid_indicis_seq.append(indicis)
                continue
            current_size = len(chat_content_list)
            split_chatmsg_list.extend(chat_content_list)
            split_chatmsg_size_list.append(current_size)
            valid_indicis_seq.append(indicis)

        split_chatmsg_list = list(split_chatmsg_list)
        split_chatmsg_size_list = list(split_chatmsg_size_list)
        # must convert deque -> array
        if self.segment_method == QQSEG_METHOD:
            segment_tokens = self.tokenizer.cut_words_with_qqseg(
                batch_text=split_chatmsg_list)
        elif self.segment_method == JIEBA_METHOD:
            segment_tokens = self.tokenizer.cut_words_with_jieba(
                batch_text=split_chatmsg_list)
        else:
            error_info = "not support for {},only support qqseg({}) / jieba({})".format(
                self.segment_method, QQSEG_METHOD, JIEBA_METHOD)
            self.logger.fatal(error_info)
            raise ValueError(error_info)
        valid_indicis_seq, segment_tokens = self.agg_segment_result(
            segment_tokens=segment_tokens,
            split_chatmsg_size_list=split_chatmsg_size_list,
            invalid_indicis_seq=invalid_indicis_seq,
            valid_indicis_seq=valid_indicis_seq,
        )
        input_ids = self.token_2_ids_builder.handle_batch_data(
            segment_tokens, convert_numpy=True)
        return valid_indicis_seq, invalid_indicis_seq, input_ids

    def inference(
        self,
        eval_text_list: List = None,
        return_word_attention: bool = False,
        return_sentence_attention: bool = False,
    ) -> tuple:
        """
        predict for eval_text_list
        Args:
            eval_text_list:List,contains the text
            return_word_weight:bool,if true,we will return the attention value of word,it is a 3-D tensor
            return_sentence_weight:bool,if ture,return the attention value of sentence,2-D tensor
        Returns:
            prob_values:np.ndarray
            word_attention_vlaues:np.ndarray
            sentence_attention_values:np.ndarray
        """
        raise NotImplementedError("not implement function -> {}".format(
            self.inference.__name__))

    def agg_segment_result(
        self,
        segment_tokens: List = None,
        split_chatmsg_size_list: List = None,
        invalid_indicis_seq: Deque = None,
        valid_indicis_seq: Deque = None,
    ) -> None:
        """
        agg the segment result
        Args:
            segment_tokens:List
            split_chatmsg_size:list,each value is the sizeof split chat content
            invalid_indicis_seq:sequence,contain the indicis of empty string value
            valid_indicis_seq:sequence,contains the indicis of not empty string value
        Returns:
            agg_segment_tokens:List
        """
        agg_valid_indicis_seq = deque()
        agg_segment_tokens = deque()
        start_indicis = 0
        for valid_indicis, split_chatmsg_size in zip(valid_indicis_seq,
                                                     split_chatmsg_size_list):
            current_text_tokens = []
            for _ in range(split_chatmsg_size):
                split_chatmsg_tokens = segment_tokens[start_indicis]
                start_indicis += 1
                split_chatmsg_tokens = self.tokenizer.filter_tokens(
                    split_chatmsg_tokens)
                if len(split_chatmsg_tokens) == 0:
                    continue
                current_text_tokens.append(split_chatmsg_tokens)
            if len(current_text_tokens) > 0:
                agg_segment_tokens.append(current_text_tokens)
                agg_valid_indicis_seq.append(valid_indicis)
            else:
                self.logger.debug("found empty token list after filtered!")
                invalid_indicis_seq.append(valid_indicis)
        # you'd better convert queue -> list!
        agg_valid_indicis_seq = list(agg_valid_indicis_seq)
        agg_segment_tokens = list(agg_segment_tokens)
        return (agg_valid_indicis_seq, agg_segment_tokens)

    def _create_same_shape_array_with_specified_array(
        self,
        specified_array: np.ndarray = None,
        sample_size: int = None,
        fill_value: Any = None,
    ) -> np.ndarray:
        """
        Args:
            if the array has shape d1,d2,d3,we will create new_array with shape
                sample_size,d2,d3 which has the same dtype with specified array!
            specified_array:np.ndarray
            sample_size:int
        """
        if fill_value is None:
            self.logger.debug(
                "did not specify any value to fill your array,we will use zero to fill!"
            )
            fill_value = 0
        shape = specified_array.shape
        new_shape = [sample_size]
        if len(shape) == 1:
            self.logger.debug("get a 1-D array")
        else:
            new_shape.extend(shape[1:])
        data_type = specified_array.dtype
        if data_type == np.str:
            data_type = np.object
        new_array = np.full(shape=new_shape,
                            fill_value=fill_value,
                            dtype=data_type)
        return new_array

    def _create_array_with_sepcified_meta_datas(
            self,
            data_shape: List = None,
            data_type: str = None,
            fill_value: Any = None) -> np.ndarray:
        if data_type == np.str:
            data_type = np.object
        return np.full(shape=data_shape,
                       fill_value=fill_value,
                       dtype=data_type)

    def shutdown(self):
        raise NotImplementedError("rewrite!")


class TextInferenceWithTf1(TextInferenceBase):

    def __init__(
        self,
        model_dir: str = None,
        logger: LogFactory = None,
        tokenizer: Tokenizer = None,
        token_2_ids_builder: HAN_Token2Matrix = None,
        segment_method: int = QQSEG_METHOD,
        pb_model_tags: List = None,
        pb_signature_def: str = "serving_default",
    ) -> None:
        """
        Args:
            pb_model_tags:the tags of exported model
            pb_signature_def:str,the key of the pb define!
        """
        super().__init__(
            model_dir=model_dir,
            logger=logger,
            tokenizer=tokenizer,
            token_2_ids_builder=token_2_ids_builder,
            segment_method=segment_method,
        )
        if pb_model_tags is None:
            pb_model_tags = ["serve"]
        self.pb_model_tags = pb_model_tags
        self.pb_signature_def = pb_signature_def

    def load_model(self) -> None:
        """
        load pb model
        Args:
            input_tensor_names:list,the input tensor keys you specified!
            output_tensor_names:list,the output tensor keys you specified!
        """
        if self.model_already_load:
            self.logger.info("the tensorflow model already load!")
        self.logger.info("load model using tensorflow 1.x api....")
        # pylint: disable=no-member
        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True
        self.sess = tf.Session(config=config)
        meta_graph = tf.saved_model.loader.load(sess=self.sess,
                                                tags=self.pb_model_tags,
                                                export_dir=self.model_dir)
        # pylint: enable=no-member
        signature_def = meta_graph.signature_def
        self.logger.debug(
            "the tensor infos as follows:\n{}".format(signature_def))

        tensor_infos = signature_def[self.pb_signature_def]
        # get the input_ids tensor name
        input_ids_tensor_name = tensor_infos.inputs["input_ids"].name

        # get the prob tensor name
        prob_tensor_name = tensor_infos.outputs["prob"].name
        # word attention tensor name
        word_attention_tensor_name = tensor_infos.outputs[
            "word_attention"].name
        # sentence attention tensor name
        sentence_attention_tensor_name = tensor_infos.outputs[
            "sentence_attention"].name

        # set dynamic attr
        self.input_ids_tensor = self.sess.graph.get_tensor_by_name(
            input_ids_tensor_name)
        self.softmax_tensor = self.sess.graph.get_tensor_by_name(
            prob_tensor_name)
        self.word_attention_tensor = self.sess.graph.get_tensor_by_name(
            word_attention_tensor_name)
        self.sentence_attention_tensor = self.sess.graph.get_tensor_by_name(
            sentence_attention_tensor_name)

        self.logger.info("Successfully load tensorflow model!")
        self.model_already_load = True

    def inference(
        self,
        eval_text_list: List = None,
        return_word_attention: bool = False,
        return_sentence_attention: bool = False,
        batch_size: int = 2000,
    ) -> tuple:
        total_predict_prob_array = None
        total_predict_category_array = None
        total_word_attention_array = None
        total_sentence_attention_array = None

        if not return_word_attention:
            self.logger.debug("word attention array will be None!")
        if not return_sentence_attention:
            self.logger.debug("sentence attention array will be None!")
        if len(eval_text_list) == 0:
            self.logger.debug("empty predict text!")
            return (
                total_predict_prob_array,
                total_predict_category_array,
                total_word_attention_array,
                total_sentence_attention_array,
            )

        if not self.model_already_load:
            self.logger.debug(
                "tf model did not initialized,please invoke function -> {}".
                format(self.load_model.__name__))
            return (
                total_predict_prob_array,
                total_predict_category_array,
                total_word_attention_array,
                total_sentence_attention_array,
            )
        total_size = len(eval_text_list)

        valid_indicis_seq, _, input_ids = self.build_input_ids(
            eval_text_list=eval_text_list)

        valid_size = len(input_ids)
        n_iters = valid_size // batch_size
        left, right = 0, batch_size
        score_list, word_attention_list, sentence_attention_list = [], [], []
        for i in range(n_iters):
            if i == (n_iters - 1):
                right = valid_size

            batch_input_ids = input_ids[left:right]
            feed_dict = {self.input_ids_tensor: batch_input_ids}
            fetches = [
                self.softmax_tensor,
                self.word_attention_tensor,
                self.sentence_attention_tensor,
            ]

            batch_score, batch_word_attention, batch_sentence_attention = self.sess.run(
                fetches=fetches, feed_dict=feed_dict)
            score_list.append(batch_score)
            word_attention_list.append(batch_word_attention)
            sentence_attention_list.append(batch_sentence_attention)
            left += batch_size
            right += batch_size

        # agg the batch result
        valid_score_array = np.vstack(score_list)

        # wrap for predict prob
        valid_prob_array = np.max(valid_score_array, axis=1)
        total_prob_array = self._create_same_shape_array_with_specified_array(
            specified_array=valid_prob_array,
            sample_size=total_size,
            fill_value=self.empty_text_prob,
        )
        total_prob_array[valid_indicis_seq] = valid_prob_array

        if return_word_attention:
            valid_word_attention_array = np.vstack(word_attention_list)
            total_word_attention_array = self._create_same_shape_array_with_specified_array(
                specified_array=valid_word_attention_array,
                sample_size=total_size,
                fill_value=0,
            )
            total_word_attention_array[
                valid_indicis_seq] = valid_word_attention_array
        if return_sentence_attention:
            valid_sentence_attention_array = np.vstack(sentence_attention_list)
            total_sentence_attention_array = self._create_same_shape_array_with_specified_array(
                specified_array=valid_sentence_attention_array,
                sample_size=total_size,
                fill_value=0,
            )
            total_sentence_attention_array[
                valid_indicis_seq] = valid_sentence_attention_array

        valid_predict_id_array = np.argmax(valid_score_array, axis=1)
        valid_label_array = np.array([
            self.id_2_label_dict[predict_id]
            for predict_id in valid_predict_id_array
        ])
        if len(valid_label_array) > 0:
            total_predict_category_array = self._create_same_shape_array_with_specified_array(
                specified_array=valid_label_array,
                sample_size=total_size,
                fill_value=self.empty_text_category,
            )
            total_predict_category_array[valid_indicis_seq] = valid_label_array
        else:
            total_predict_category_array = self._create_array_with_sepcified_meta_datas(
                data_shape=[total_size],
                data_type=np.str,
                fill_value=self.empty_text_category,
            )

        return (
            total_predict_prob_array,
            total_predict_category_array,
            total_word_attention_array,
            total_sentence_attention_array,
        )

    def shutdown(self):
        self.logger.info("shutdown all the sources!")
        self.input_ids_tensor = None
        self.softmax_tensor = None
        self.word_attention_tensor = None
        self.sentence_attention_tensor = None
        self.sess.close()
        self.tokenizer.shutdown()


class TextInferenceWithTf2(TextInferenceBase):

    def __init__(
        self,
        model_dir: str = None,
        logger: LogFactory = None,
        tokenizer: Tokenizer = None,
        token_2_ids_builder: HAN_Token2Matrix = None,
        segment_method: int = QQSEG_METHOD,
        pb_model_tags: List = None,
        pb_signature_def: str = "serving_default",
    ) -> None:
        super().__init__(
            model_dir=model_dir,
            logger=logger,
            tokenizer=tokenizer,
            token_2_ids_builder=token_2_ids_builder,
            segment_method=segment_method,
        )
        self.pb_model_tags = pb_model_tags
        self.logger.info("pb model tags -> {}".format(pb_model_tags))
        self.pb_signature_def = pb_signature_def
        self.logger.info("pb signature -> {}".format(pb_signature_def))

    def load_model(self):
        loaded = tf.saved_model.load(self.model_dir, tags=self.pb_model_tags)
        infer_obj = loaded.signatures[self.pb_signature_def]
        self.model = infer_obj
        self.logger.info(infer_obj.structured_outputs)
        self.logger.info("Successfully loaded model with tensorflow2.x api!")
        self.model_already_load = True

    def inference(
        self,
        eval_text_list: List = None,
        return_word_attention: bool = False,
        return_sentence_attention: bool = False,
        batch_size: int = 2000,
    ) -> tuple:
        total_predict_prob_array = None
        total_predict_category_array = None
        total_word_attention_array = None
        total_sentence_attention_array = None

        if not return_word_attention:
            self.logger.debug("word attention array will be None!")
        if not return_sentence_attention:
            self.logger.debug("sentence attention array will be None!")
        if len(eval_text_list) == 0:
            self.logger.debug("empty predict text!")
            return (
                total_predict_prob_array,
                total_predict_category_array,
                total_word_attention_array,
                total_sentence_attention_array,
            )

        if not self.model_already_load:
            self.logger.debug(
                "tf model did not initialized,please invoke function -> {}".
                format(self.load_model.__name__))
            return (
                total_predict_prob_array,
                total_predict_category_array,
                total_word_attention_array,
                total_sentence_attention_array,
            )
        total_size = len(eval_text_list)

        valid_indicis_seq, _, input_ids = self.build_input_ids(
            eval_text_list=eval_text_list)

        valid_size = len(input_ids)
        n_iters = max(1, valid_size // batch_size)
        left, right = 0, batch_size
        score_list, word_attention_list, sentence_attention_list = [], [], []
        for i in range(n_iters):
            if i == (n_iters - 1):
                right = valid_size
            batch_input_ids = input_ids[left:right]
            batch_input_ids_tensor = tf.constant(batch_input_ids)
            predict_array_map: dict = self.model(batch_input_ids_tensor)
            batch_score = predict_array_map["prob"]
            batch_word_attention = predict_array_map["word_attention"]
            batch_sentence_attention = predict_array_map["sentence_attention"]
            score_list.append(batch_score)
            word_attention_list.append(batch_word_attention)
            sentence_attention_list.append(batch_sentence_attention)
            left += batch_size
            right += batch_size

        # agg the batch result
        valid_score_array = np.vstack(score_list)

        # wrap for predict prob
        valid_prob_array = np.max(valid_score_array, axis=1)
        total_predict_prob_array = self._create_same_shape_array_with_specified_array(
            specified_array=valid_prob_array,
            sample_size=total_size,
            fill_value=self.empty_text_prob,
        )
        total_predict_prob_array[valid_indicis_seq] = valid_prob_array

        if return_word_attention:
            valid_word_attention_array = np.vstack(word_attention_list)
            total_word_attention_array = self._create_same_shape_array_with_specified_array(
                specified_array=valid_word_attention_array,
                sample_size=total_size,
                fill_value=0.0,
            )
            total_word_attention_array[
                valid_indicis_seq] = valid_word_attention_array
        if return_sentence_attention:
            valid_sentence_attention_array = np.vstack(sentence_attention_list)
            total_sentence_attention_array = self._create_same_shape_array_with_specified_array(
                specified_array=valid_sentence_attention_array,
                sample_size=total_size,
                fill_value=0.0,
            )
            total_sentence_attention_array[
                valid_indicis_seq] = valid_sentence_attention_array

        valid_predict_id_array = np.argmax(valid_score_array, axis=1)
        valid_label_array = np.array([
            self.id_2_label_dict[predict_id]
            for predict_id in valid_predict_id_array
        ])
        if len(valid_label_array) > 0:
            total_predict_category_array = self._create_same_shape_array_with_specified_array(
                specified_array=valid_label_array,
                sample_size=total_size,
                fill_value=self.empty_text_category,
            )
            total_predict_category_array[valid_indicis_seq] = valid_label_array
        else:
            total_predict_category_array = self._create_array_with_sepcified_meta_datas(
                data_shape=[total_size],
                data_type=np.str,
                fill_value=self.empty_text_category,
            )

        return (
            total_predict_prob_array,
            total_predict_category_array,
            total_word_attention_array,
            total_sentence_attention_array,
        )

    def shutdown(self):
        self.logger.info("clear tf session!")
        tf.keras.backend.clear_session()
        self.tokenizer.shutdown()


class TextInferenceWX:

    def __init__(
        self,
        model_dir: str = None,
        pb_model_tags: List = None,
        pb_signature_def: str = "serving_default",
        segment_method=JIEBA_METHOD,
        logger=None,
    ) -> None:
        self.logger = logger
        self.package_name = "txws_text_classification"
        self.segment_method = segment_method
        self._initialize_all_with_default()
        if pb_model_tags is None:
            pb_model_tags = ["serve"]
        self.pb_model_tags = pb_model_tags
        self.pb_signature_def = pb_signature_def
        self.model_dir = model_dir

        tf_major_verion = tf_version()
        if tf_major_verion == 1:
            self.logger.info("found tensorflow1.x...")
            runner = TextInferenceWithTf1(
                model_dir=model_dir,
                logger=self.logger,
                tokenizer=self.tokenizer,
                token_2_ids_builder=self.token_2_ids_builder,
                segment_method=segment_method,
                pb_model_tags=pb_model_tags,
                pb_signature_def=pb_signature_def,
            )
        elif tf_major_verion == 2:
            self.logger.info("found tensorflow 2.x...")
            runner = TextInferenceWithTf2(
                model_dir=model_dir,
                logger=self.logger,
                tokenizer=self.tokenizer,
                token_2_ids_builder=self.token_2_ids_builder,
                segment_method=segment_method,
                pb_model_tags=pb_model_tags,
                pb_signature_def=pb_signature_def,
            )
        else:
            error_info = "support tensorflow for 1.x/2.x,otherwise get -> {}".format(
                tf.__version__)
            self.logger.info(error_info)
            raise RuntimeError(error_info)

        self.runner = runner
        self.runner.set_empty_text_category("其它")
        self.runner.set_empty_text_prob(1.0)
        self.runner.set_id_2_label_dict(self.id_2_label_dict)

    def _initialize_all_with_default(self):
        if self.logger is None:
            self.logger = LogFactory(log_dir="logs",
                                     log_prefix="text_inference.log",
                                     scope_name="text_infer")
        self.logger.info("initialize with default source...")
        self.package_dir = self._found_install_dir(self.package_name)
        if self.package_dir is None:
            error_info = "can not found installed dir!"
            self.logger.fatal(error_info)
            raise Exception(error_info)
        self.sentence_size = 45
        self.word_size = 20
        self.max_features = 200000
        self.vocab_fp = os.path.join(self.package_dir, "assets/vocab_wx.txt")
        self.token_2_id_dict = read_token_map(
            fp=self.vocab_fp,
            max_feature_size=self.max_features,
            with_head=True)
        self.replace_pairs_fp = os.path.join(self.package_dir,
                                             "assets/replace_pairs.txt")
        self.replace_patterns_fp = os.path.join(self.package_dir,
                                                "assets/replace_patterns.txt")
        self.stopwords_fp = os.path.join(self.package_dir,
                                         "assets/stopwords.txt")
        qqseg_mode = (TC_CRF
                      | TC_OTHER_NE
                      | TC_NER_DL
                      | TC_IP
                      | TC_VIDEO
                      | TC_PRODUCTION
                      | TC_PER_W
                      | TC_ORG_W
                      | TC_LOC_W
                      | TC_VIDEO_W
                      | TC_RUL
                      | TC_CUS)
        self.qqseq_package_dir = self._found_install_dir(package_name="qqseg")
        self.qqseg_initialize_source_dir = os.path.join(
            self.qqseq_package_dir, "qqseg_data")
        self.logger.info("self.qqseg_initialize_source_dir:{}".format(
            self.qqseg_initialize_source_dir))

        logger.info("qqseg_initialize_source_dir:{}".format(
            self.qqseg_initialize_source_dir))
        if self.segment_method == JIEBA_METHOD:
            self.logger.info("init JIEBA_METHOD")
            tokenizer = Tokenizer(
                token_method=JIEBA_METHOD,
                logger=self.logger,
            )
        elif self.segment_method == QQSEG_METHOD:
            self.logger.info("init QQSEG_METHOD")
            tokenizer = Tokenizer(
                token_method=QQSEG_METHOD,
                logger=self.logger,
                qqseg_mode=qqseg_mode,
                qqseg_method=COARS_GRAINED_METHOD,
                qqseg_initialize_source_dir=self.qqseg_initialize_source_dir,
                qqseg_num_thread=1,
            )
        tokenizer.initialize_stopwords(self.stopwords_fp)
        tokenizer.initialize_replace_pairs(self.replace_pairs_fp)
        tokenizer.initialize_replace_patterns(self.replace_patterns_fp)
        tokenizer.initialize_opcc()
        self.tokenizer = tokenizer

        self.token_2_ids_builder = HAN_Token2Matrix(
            sentence_size=self.sentence_size,
            word_size=self.word_size,
            max_features=self.max_features,
            logger=self.logger,
        )
        self.token_2_ids_builder.set_token_map(self.token_2_id_dict)

        self.id_2_label_dict = {
            0: "传销民资",
            1: "赌博形式",
            2: "返利诈骗",
            3: "仿冒欺诈",
            4: "非法售卖",
            5: "黑灰产",
            6: "黑五类",
            7: "兼职欺诈",
            8: "荐股欺诈",
            9: "交友欺诈",
            10: "金融贷款欺诈",
            11: "可疑",
            12: "免费送",
            13: "其它",
            14: "色情低俗",
            15: "违禁品",
        }

    def _found_install_dir(self, package_name):
        installed_dir = None
        for p in sys.path:
            installed_dir = os.path.join(p, package_name)
            if os.path.exists(installed_dir):
                self.logger.info(
                    "found installed dir -> {}".format(installed_dir))
                break
        return installed_dir

    def set_sentence_size(self, sentence_size: int = 45) -> None:
        self.sentence_size = sentence_size

    def set_word_size(self, word_size: int = 20) -> None:
        self.word_size = word_size

    def set_max_features(self, max_features: int = 200000) -> None:
        self.max_features = max_features

    def update_token_2_ids_builder(self):
        self.token_2_ids_builder.token_max_features = self.max_features
        self.token_2_ids_builder.sentence_size = self.sentence_size
        self.token_2_ids_builder.word_size = self.word_size
        self.token_2_ids_builder.initialize_token_map(self.token_2_id_dict)

    def inference(
        self,
        eval_text_list: List = None,
        return_word_attention: bool = False,
        return_sentence_attention: bool = False,
        batch_size: int = 2000,
    ) -> None:
        """
        inference
        """
        if not self.runner.model_already_load:
            self.runner.load_model()
        res = self.runner.inference(
            eval_text_list=eval_text_list,
            return_word_attention=return_word_attention,
            return_sentence_attention=return_sentence_attention,
            batch_size=batch_size,
        )
        return res


class TextInferenceQQ:

    def __init__(
        self,
        model_dir: str = None,
        pb_model_tags: List = None,
        pb_signature_def: str = "serving_default",
        segment_method=JIEBA_METHOD,
        logger: LogFactory = None,
    ) -> None:
        self.logger = logger
        self.package_name = "txws_text_classification"
        self.segment_method = segment_method
        self._initialize_all_with_default()
        if pb_model_tags is None:
            pb_model_tags = ["serve"]
        self.pb_model_tags = pb_model_tags
        self.pb_signature_def = pb_signature_def
        self.model_dir = model_dir

        tf_major_verion = tf_version()
        if tf_major_verion == 1:
            self.logger.info("found tensorflow 1.x...")
            runner = TextInferenceWithTf1(
                model_dir=model_dir,
                logger=self.logger,
                tokenizer=self.tokenizer,
                token_2_ids_builder=self.token_2_ids_builder,
                segment_method=segment_method,
                pb_model_tags=pb_model_tags,
                pb_signature_def=pb_signature_def,
            )
        elif tf_major_verion == 2:
            self.logger.info("found tensorflow 2.x...")
            runner = TextInferenceWithTf2(
                model_dir=model_dir,
                logger=self.logger,
                tokenizer=self.tokenizer,
                token_2_ids_builder=self.token_2_ids_builder,
                segment_method=segment_method,
                pb_model_tags=pb_model_tags,
                pb_signature_def=pb_signature_def,
            )
        else:
            version = tf.__version__
            error_info = f"support tensorflow for 1.x/2.x,otherwise get -> {version}"
            self.logger.info(error_info)
            raise RuntimeError(error_info)

        self.runner = runner
        self.runner.set_empty_text_category("其它")
        self.runner.set_empty_text_prob(1.0)
        self.runner.set_id_2_label_dict(self.id_2_label_dict)

    def _initialize_all_with_default(self):
        if self.logger is None:
            self.logger = LogFactory(log_dir="logs",
                                     log_prefix="text_inference.log",
                                     scope_name="text_infer")
        self.logger.info("initialize with default source...")
        self.package_dir = self._found_install_dir(self.package_name)
        self.logger.info("package_dir:{}".format(self.package_dir))
        if self.package_dir is None:
            error_info = "can not found installed dir!"
            self.logger.fatal(error_info)
            raise Exception(error_info)
        self.sentence_size = 30
        self.word_size = 15
        self.max_features = 130000
        self.vocab_fp = os.path.join(self.package_dir, "assets/vocab_qq.txt")
        self.token_2_id_dict = read_token_map(
            fp=self.vocab_fp,
            max_feature_size=self.max_features,
            with_head=True)
        self.replace_pairs_fp = os.path.join(self.package_dir,
                                             "assets/replace_pairs.txt")
        self.replace_patterns_fp = os.path.join(self.package_dir,
                                                "assets/replace_patterns.txt")
        self.stopwords_fp = os.path.join(self.package_dir,
                                         "assets/stopwords.txt")
        qqseg_mode = (TC_CRF
                      | TC_OTHER_NE
                      | TC_NER_DL
                      | TC_IP
                      | TC_VIDEO
                      | TC_PRODUCTION
                      | TC_PER_W
                      | TC_ORG_W
                      | TC_LOC_W
                      | TC_VIDEO_W
                      | TC_RUL
                      | TC_CUS)
        self.qqseq_package_dir = self._found_install_dir(package_name="qqseg")
        self.qqseg_initialize_source_dir = os.path.join(
            self.qqseq_package_dir, "qqseg_data")
        self.logger.info("self.qqseg_initialize_source_dir:{}".format(
            self.qqseg_initialize_source_dir))

        logger.info("qqseg_initialize_source_dir:{}".format(
            self.qqseg_initialize_source_dir))
        if self.segment_method == JIEBA_METHOD:
            self.logger.info("init JIEBA_METHOD")
            tokenizer = Tokenizer(
                token_method=JIEBA_METHOD,
                logger=self.logger,
            )
        elif self.segment_method == QQSEG_METHOD:
            self.logger.info("init QQSEG_METHOD")
            tokenizer = Tokenizer(
                token_method=QQSEG_METHOD,
                logger=self.logger,
                qqseg_mode=qqseg_mode,
                qqseg_method=COARS_GRAINED_METHOD,
                qqseg_initialize_source_dir=self.qqseg_initialize_source_dir,
                qqseg_num_thread=1,
            )
        tokenizer.initialize_stopwords(self.stopwords_fp)
        tokenizer.initialize_replace_pairs(self.replace_pairs_fp)
        tokenizer.initialize_replace_patterns(self.replace_patterns_fp)
        tokenizer.initialize_opcc()
        self.tokenizer = tokenizer

        self.token_2_ids_builder = HAN_Token2Matrix(
            sentence_size=self.sentence_size,
            word_size=self.word_size,
            max_features=self.max_features,
            logger=self.logger,
        )
        self.token_2_ids_builder.set_token_map(self.token_2_id_dict)

        self.id_2_label_dict = {
            0: "盗号",
            1: "赌博形式",
            2: "返利诈骗",
            3: "仿冒欺诈",
            4: "黑灰产",
            5: "兼职欺诈",
            6: "荐股欺诈",
            7: "交易欺诈",
            8: "交友欺诈",
            9: "金融贷款欺诈",
            10: "免费送",
            11: "敲诈勒索",
            12: "色情",
            13: "其它",
        }

    def _found_install_dir(self, package_name):
        installed_dir = None
        for p in sys.path:
            installed_dir = os.path.join(p, package_name)
            if os.path.exists(installed_dir):
                self.logger.info(
                    "found installed dir -> {}".format(installed_dir))
                break
        return installed_dir

    def set_sentence_size(self, sentence_size: int = 30) -> None:
        self.sentence_size = sentence_size

    def set_word_size(self, word_size: int = 15) -> None:
        self.word_size = word_size

    def set_max_features(self, max_features: int = 130000) -> None:
        self.max_features = max_features

    def update_token_2_ids_builder(self):
        self.token_2_ids_builder.token_max_features = self.max_features
        self.token_2_ids_builder.sentence_size = self.sentence_size
        self.token_2_ids_builder.word_size = self.word_size
        self.token_2_ids_builder.initialize_token_map(self.token_2_id_dict)

    def inference(
        self,
        eval_text_list: List = None,
        return_word_attention: bool = False,
        return_sentence_attention: bool = False,
        batch_size: int = 2000,
    ) -> None:
        """
        inference
        """
        if not self.runner.model_already_load:
            self.runner.load_model()
        res = self.runner.inference(
            eval_text_list=eval_text_list,
            return_word_attention=return_word_attention,
            return_sentence_attention=return_sentence_attention,
            batch_size=batch_size,
        )
        return res


if __name__ == "__main__":
    PB_MODEL_DIR = "/Users/xiangyuwang/Software/txws_text_classification/saved_model"
    # pb_model_dir = "/dockerdata/sunyyao/tutorial/tf_model/saved_model"
    # wx场景
    WX_PB_MODEL_DIR = os.path.join(PB_MODEL_DIR, "wx")
    s = TextInferenceWX(model_dir=WX_PB_MODEL_DIR)

    # # qq场景
    # qq_pb_model_dir = os.path.join(pb_model_dir, "qq")
    # s = TextInferenceQQ(model_dir=qq_pb_model_dir, logger=logger)

    eval_datas = [
        "今日股票不错,请进群找老师领取分享资料|过夜500，到了发房间号。",
        "过夜500，到了发房间号。",
        "|其他人(858668462):这奶子是真的大|其他人(989640242):奶大不好|其他人(3322735714):我喜欢没毛的 馒头逼|其他人(858668462):直播|其他人(858668462):这女的可以\
        |被举报人:032j.tv|其他人(858668462):032b.tv|其他人(989640242):大家知道和几岁女孩做爱是犯法的|其他人(3602638725):14以下|其他人(989640242):十五岁逼好玩，还没毛\
        |其他人(3602638725):不好弄的哭哭啼啼的|其他人(3322735714):12岁就有毛了|其他人(3322735714):现在孩子都发育早",
    ]

    res = s.inference(
        eval_text_list=eval_datas,
        return_sentence_attention=False,
        return_word_attention=False,
    )
    print(res)
