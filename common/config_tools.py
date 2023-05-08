import os
from configparser import ConfigParser
import traceback
from typing import Any, Callable


class ConfParserWithouLower(ConfigParser):

    def __init__(self, defaults=None):
        ConfigParser.__init__(self, defaults=defaults)

    def optionxform(self, optionstr):
        return optionstr


class SimpleParser(object):

    def __init__(self, config_file):
        if not os.path.exists(config_file):
            raise ValueError(f"{config_file} is not an valid file path")
        self.config_file = config_file
        self.parser = ConfParserWithouLower()
        self.parser.read(self.config_file)

    def get_field_value(self,
                        scope_name: str,
                        feature_key: str,
                        transfer_function: Callable = None) -> Any:
        if scope_name not in self.parser:
            raise ValueError(
                f"scope_name->{scope_name} not in your config file,please check..."
            )

        value = self.parser.get(scope_name, feature_key)
        if transfer_function is not None:
            try:
                value = transfer_function(value)
            except:  # pylint:disable=bare-except
                value_type = type(value)
                print(
                    f"failed to use function {transfer_function} to convert '{value}' which has type {value_type},\
                        please check....")
                traceback.print_exc()
        return value

    def get_field_value_safe(self, scope_name: str, feature_key: str,
                             default: Any):
        if scope_name not in self.parser or self.parser.has_option(
                scope_name, feature_key):
            return self.parser.get(scope_name, feature_key)
        return default

    @property
    def keys(self):
        keys_value: list = list(self.parser.keys())
        keys_value.remove("DEFAULT")
        return keys_value

    def get_scope_dict(self, scope_name: str) -> dict:
        if scope_name not in self.parser:
            raise ValueError(
                f"scope_name->{scope_name} not in your config file,please check"
            )
        scope_value = self.parser[scope_name]
        scope_value_dict = dict(scope_value)
        if len(scope_value_dict) == 0:
            print(
                f"Warning:scope_name {scope_name} return an empty value dict..."
            )
        return scope_value_dict


# test
def test():
    config_file = "conf/test.conf"
    parser = SimpleParser(config_file=config_file)
    sunflower_color = parser.get_field_value("flower",
                                             "sunflower",
                                             transfer_function=int)
    flower_color_map = parser.get_scope_dict("flower")
    print(sunflower_color)
    print(flower_color_map)


if __name__ == "__main__":
    CONF_FILE = os.path.join(os.getcwd(), "conf.ini")
    CONF_FILE = "../config.ini"
    if os.path.exists(CONF_FILE):
        conf = SimpleParser(CONF_FILE)
        rs = conf.get_field_value_safe("ES_WX_SOCIAL_REPORT", "password",
                                       "def")
