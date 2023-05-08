import base64
import traceback

from Crypto.Cipher import AES


def pad(s, block_size):
    """补位
    :param s:
    :param block_size:
    :return:
    """
    return s + (
        (block_size - len(s) % block_size) * chr(block_size - len(s) % block_size)
    ).encode("utf8")


def un_pad(s):
    return s[: -ord(s[len(s) - 1 :])]


class AesCrypt(object):
    """
    加密函数，如果text不足16位，补足为16位，
    如果大于16但不是16的倍数，那就补足为16的倍数。
    补足方法：PKCS5
    """

    def __init__(self, key, decode_type="base64", padding="PKCS5"):
        self.key = key.encode("utf8")
        self.mode = AES.MODE_ECB
        self.decode_type = decode_type
        self.padding = padding

    @staticmethod
    def _pad(s, block_size):
        return s + (
            (block_size - len(s) % block_size) * chr(block_size - len(s) % block_size)
        ).encode("utf8")

    @staticmethod
    def _un_pad(s):
        return s[: -ord(s[len(s) - 1 :])]

    def encrypt(self, text):
        cipher = AES.new(self.key, self.mode)
        # 这里密钥key 长度必须为16（AES-128）,
        cipher_text = cipher.encrypt(self._pad(text.encode("utf8"), AES.block_size))

        # AES加密时候得到的字符串是二进制，需要转成字符串
        if self.decode_type == "hex":
            en_text = cipher_text.hex()
        else:  # 默认使用base64编码
            en_text = base64.b64encode(cipher_text).decode()

        return en_text

    def decrypt(self, text):
        cipher = AES.new(self.key, self.mode)

        if self.decode_type == "hex":
            byte_text = bytes.fromhex(text)
        else:  # 默认使用base64解码
            byte_text = base64.b64decode(text)
        plain_text = cipher.decrypt(byte_text)
        if self.padding == "PKCS5":
            plain_text = self._un_pad(plain_text)
        plain_text = plain_text.strip(b"\x00")
        return bytes.decode(plain_text)


class PrpCrypt(object):
    """
    加密函数，如果text不足16位，补足为16位，
    如果大于16但不是16的倍数，那就补足为16的倍数。
    补足方法：PKCS5
    """

    def __init__(self, key):
        self.key = key.encode("utf8")
        self.mode = AES.MODE_ECB

    def encrypt(self, text):
        cipher = AES.new(self.key, self.mode)
        # 这里密钥key 长度必须为16（AES-128）,
        cipher_text = cipher.encrypt(pad(text.encode("utf8"), AES.block_size))
        # 因为AES加密时候得到的字符串不一定是ascii字符集的，输出到终端或者保存时候可能存在问题
        # 所以这里将加密的字符串进行base64编码
        return base64.b64encode(cipher_text).decode()

    def decrypt(self, text):
        cipher = AES.new(self.key, self.mode)
        try:
            plain_text = cipher.decrypt(base64.b64decode(text))
            # return bytes.decode(un_pad(plain_text))
            decode_data = bytes.decode(plain_text)  # , 'unicode_escape'
            decode_data = un_pad(decode_data)
            return decode_data
        except:  # pylint: disable=bare-except
            print("decrypt error %s", traceback.format_exc())
            return text


if __name__ == "__main__":
    # aes_crypt = AesCrypt(key="kfdata_for170830")
    # encrypt_value = aes_crypt.encrypt("18695876670")
    # print(encrypt_value)
    aes_crypt = AesCrypt(key="tyz349768878tyq1")
    encrypt_value = aes_crypt.encrypt("wxid_ntzmxwzjgdir22")
    print(encrypt_value)
    TEST = "5j36rdeGRpb6/C7bdlZqbtbjGV9tP2xC6xh2Ur+7sAA="
    DECRYPT_VALUE = aes_crypt.decrypt(text=TEST)
    print(DECRYPT_VALUE)

    ac = AesCrypt(key="20200617kfywaqjb", padding="no")
    decode_content = ac.decrypt(
        "kW5R0nVotHjPVtOoC5Q4vvQl/hRG9F/gakuo/fCd23X6os8yM5ZJUMVIfH89e+kCKHh0a9QtBSmDjn8qNfE6BYo1R95BZQKyB0cyCPMkC17spWOYD92keiLh9SyQamoRP1Wxov2zIqceBLNWHajB1SexKAqcrH5Khh6zr92mWUY="
    )
