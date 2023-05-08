import os
import configparser
import traceback
from common.encrypt_decrypt import AesCrypt


def json_2_ini(json_config, cfg_file):
    cfg = configparser.ConfigParser()
    cfg.read(cfg_file, encoding="utf-8")
    cfg.add_section("default")
    for k in json_config:
        if isinstance(json_config[k], dict):
            cfg.add_section(k)
            for ik in json_config[k]:
                cfg.set(k, ik, str(json_config[k][ik]))
        else:
            cfg.set("default", k, str(json_config[k]))  # 修改db_port的值为69
    cfg.write(open(cfg_file, "w"))


def ini_2_json(cfg_file):
    aes_crypt = AesCrypt(key="kfdata_for170830")
    config = {}
    cfg = configparser.ConfigParser()
    cfg.read(cfg_file, encoding="utf-8")
    for sec in cfg.sections():
        if sec not in config:
            config[sec] = {}
        for option in cfg.options(sec):
            if option.startswith("main_") or option.startswith("backups_"):
                k, v = tuple(option.split("_"))
                if k not in config[sec]:
                    config[sec][k] = {}
                if v == "port":
                    config[sec][k][v] = cfg.getint(sec, option)
                else:
                    config[sec][k][v] = cfg.get(sec, option)
            elif option in ["connUseCount", "ticketMax"]:
                config[sec][option] = cfg.getint(sec, option)
            elif option == "port" and sec.startswith("MYSQL"):
                config[sec][option] = cfg.getint(sec, option)
            elif option.find("password") != -1:
                try:
                    config[sec][option] = aes_crypt.decrypt(
                        cfg.get(sec, option))
                except:  # pylint: disable=bare-except
                    traceback.format_exc()
                    config[sec][option] = cfg.get(sec, option)
            else:
                config[sec][option] = cfg.get(sec, option)
    return config


# file_root = os.path.abspath(__file__)
# root = os.path.abspath(os.path.join(file_root, ".."))
CONFIG = ini_2_json("/data/report/common/config.ini")
print("config")
# for i in range(3):
#     config_file = os.path.join(root, "config.ini")
#     if os.path.exists(config_file):
#         CONFIG = ini_2_json(config_file)
#         break
#     root = os.path.abspath(os.path.join(root, ".."))
