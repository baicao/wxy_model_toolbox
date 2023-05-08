class OpenccHandler:
    def __init__(self) -> None:
        try:
            import opencc
        except ImportError as e:
            raise e
        self.converter = opencc.OpenCC("s2t.json")

    def tc_2_sc(self, to_convert):
        return self.converter.conver(to_convert, "zh-cn")


class zhconvHandler:
    def __init__(self) -> None:
        try:
            from zhconv import convert
        except ImportError as e:
            raise e
        self.convert = convert

    def tc_2_sc(self, to_convert):
        return self.convert(to_convert, "zh-cn")


class ZhCharacterTrans:
    def __init__(self, handle_type="zhconv") -> None:
        self.handle_type = handle_type
        if handle_type == "opencc":
            self.handler = OpenccHandler()
        else:
            self.handler = zhconvHandler()

    def tc_2_sc(self, to_convert):
        return self.handler.tc_2_sc(to_convert)


if __name__ == "__main__":
    t = ZhCharacterTrans()
    rs = t.tc_2_sc("他說100就可以，後來慢慢的加，最後電話打不同，微信打不同無法聯繫，申請凍結對方收到的款，所有通話都是電話通話")
