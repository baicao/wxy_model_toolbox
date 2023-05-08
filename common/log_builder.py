# -*- coding:utf-8 -*-
import logging
import os
import time
from logging.handlers import TimedRotatingFileHandler


class Log(object):

    def __init__(self, logger="main", log_cat="search", log_type="both"):
        """
            指定保存日志的文件路径，日志级别，以及调用文件
            将日志存入到指定的文件中
        """
        # 创建一个logger
        self.logger = logging.getLogger(logger)
        self.logger_level = logging.INFO
        self.logger.setLevel(self.logger_level)
        # 创建一个handler，用于写入日志文件
        self.log_time = time.strftime("%Y_%m_%d")
        file_dir = os.getcwd() + "/logs"
        if not os.path.exists(file_dir):
            os.mkdir(file_dir)
        self.log_path = file_dir

        # 再创建一个handler，用于输出到控制台

        # 定义handler的输出格式
        formatter = logging.Formatter(
            "[%(asctime)s] %(filename)s->%(funcName)s line:%(lineno)d [%(levelname)s]%(message)s"
        )

        # 给logger添加handler
        if log_type in ["file", "both"]:
            self.log_name = self.log_path + "/" + log_cat + "." + self.log_time + ".log"
            print(f"prepare log file in {self.log_name}")
            file_handler = TimedRotatingFileHandler(
                self.log_name,
                when="D",
                encoding="utf-8",
                interval=1,
                backupCount=3)  # 这个是python3的
            file_handler.setFormatter(formatter)
            file_handler.setLevel(self.logger_level)
            self.logger.addHandler(file_handler)
        if log_type in ["console", "both"]:
            stream_handler = logging.StreamHandler()
            stream_handler.setLevel(self.logger_level)
            stream_handler.setFormatter(formatter)
            self.logger.addHandler(stream_handler)

        #  添加下面一句，在记录日志之后移除句柄
        # self.logger.removeHandler(ch)
        # self.logger.removeHandler(fh)
        # 关闭打开的文件
        # fh.close()
        # ch.close()

    def getlog(self):
        return self.logger
