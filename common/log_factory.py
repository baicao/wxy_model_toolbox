import os
import sys
import datetime
import logging
from logging.handlers import TimedRotatingFileHandler
from logging import FileHandler
import traceback


class ProcessSafeFileHandler(FileHandler):
    def __init__(
        self,
        filename: str,
        mode: str = "a",
        encoding: str = "utf-8",
        delay: bool = False,
        suffix="%Y-%m-%d",
    ) -> None:
        """
        Args:
            fileanme:str->the log file path
            mode:file open mod,has a,w..,default is append content to this file
            --- if we use append mod,this will be process safe,important
            encoding:the file encod style
            dealy:bool->default is false
            suffix:str->default,we will save the file use the date time as the suffix,split our log by different day
        """
        if mode != "a":
            print(
                "Waring:you use open file mode {},which maybe not process safe,we suggest you use a mode".format(
                    mode
                )
            )
        now = datetime.datetime.now()
        suffix_time = now.strftime(suffix)
        time_filename = "{}.{}".format(filename, suffix_time)
        FileHandler.__init__(
            self, time_filename, mode=mode, encoding=encoding, delay=delay
        )
        self.filename = os.fspath(filename)
        self.mode = mode
        self.encoding = encoding
        self.suffix = suffix
        self.suffix_time = suffix_time

    def emit(self, record):
        try:
            if self._check_base_filename():
                self._make_base_filename()
            FileHandler.emit(self, record)
        except (KeyboardInterrupt, SystemExit):
            raise
        except:  # pylint: disable=bare-except
            traceback.print_exc()
            self.handleError(record)

    def _check_base_filename(self) -> bool:
        _check_status: bool = False
        current_time = datetime.datetime.now().strftime(self.suffix)
        time_filename = "{}.{}".format(self.filename, self.suffix_time)
        if (current_time != self.suffix_time) or (not os.path.exists(time_filename)):
            _check_status = True
        else:
            _check_status = False
        return _check_status

    def _make_base_filename(self):
        if self.stream:
            self.stream.close()
            self.stream = None
        current_time = datetime.datetime.now().strftime(self.suffix)
        self.suffix_time = current_time
        # update the baseFilename
        self.baseFilename = "{}.{}".format(self.filename, self.suffix_time)
        if not self.delay:
            self.stream = open(self.baseFilename, self.mode, encoding=self.encoding)


class LogFactory(object):
    headers = {"Content-Type": "text/plain"}

    def __init__(
        self,
        log_dir: str = "sb",
        log_level: int = logging.INFO,
        log_prefix="xx.log",
        log_format=None,
        scope_name="xx",
        use_stream=True,
        file_handler_type="rolling",
        timeout=50,
        **kwargs
    ):
        """
        Args:
            log_dir:the directory to save log,default is logs which is on current directory!
            log_leve:int,can be warn,info,error,fatal....
            webhook_url:a url which push info
            use_stream:bool,whether show info to other stream
            file_handler_type:str,if rolling,set rolling log by day/normal:a generic a+ mode file
            scope_name:the scope name,to prevent that different loggers write the same content
            mentioned_list:the person list which you want to push info,default not @ anyone
            timeout:the timeout for net request
            kwargs:some optional params,like log file save number

        """
        self.log_dir = log_dir
        self.log_level = log_level
        self.use_stream = use_stream
        self.file_handler_type = file_handler_type
        self.timeout = timeout
        self.prefix = log_prefix
        self.format = log_format

        if not os.path.exists(self.log_dir):
            os.makedirs(self.log_dir)

        if not isinstance(self.log_level, int):
            self.log_level = int(self.log_level)
            # if you not specify,use default 5
        self.max_logfile_number = kwargs.get("logfile_number", 5)

        self._set_logger(
            prefix=self.prefix, log_format=self.format, scope_name=scope_name
        )

    def _set_logger(self, prefix: str, scope_name: str, log_format: str = None):
        """
        Args:
            prefix:the prefix of log file
        """
        # the basict log file path
        log_fp = os.path.join(self.log_dir, prefix)
        if self.file_handler_type == "rolling":
            file_handler = TimedRotatingFileHandler(
                filename=log_fp,
                when="midnight",
                interval=1,
                backupCount=self.max_logfile_number,  # hard code
                encoding="utf-8",
            )
        # normal log file
        elif self.file_handler_type == "normal":
            file_handler = FileHandler(filename=log_fp, mode="a", encoding="utf-8")
        elif self.file_handler_type == "process_safe":
            file_handler = ProcessSafeFileHandler(filename=log_fp, suffix="%Y-%m-%d")

        # default log format
        if log_format is None:
            log_format = (
                "%(asctime)s [%(levelname)s] %(filename)s %(name)s: %(message)s"
            )

        formatter = logging.Formatter(log_format)
        file_handler.setLevel(self.log_level)
        file_handler.setFormatter(formatter)

        _logger = logging.getLogger(scope_name)
        _logger.setLevel(self.log_level)
        _logger.addHandler(file_handler)

        # add to stream
        if self.use_stream:
            stream_handler = logging.StreamHandler(stream=sys.stdout)
            stream_handler.setLevel(self.log_level)
            stream_handler.setFormatter(formatter)
            _logger.addHandler(stream_handler)
        self.logger = _logger

    def get_logger(self):
        return self.logger

    def __str__(self):
        p_tr = hex(id(self))
        return "<object with log and push info at {}>".format(p_tr)


ROOT = os.getcwd()
LOG_ROOT = os.path.join(ROOT, "logs")

logger = LogFactory(
    log_dir=LOG_ROOT,
    log_prefix="main.log",
    log_level=logging.INFO,
    scope_name="main",
    use_stream=False,
).get_logger()


if __name__ == "__main__":
    my_logger = LogFactory(
        log_dir=".",
        log_level=logging.INFO,
        scope_name="test module",
        webhook_url="https://www.baidu.com",
    )
    print(my_logger)
