#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Author  : harveywang
# @Time    : 2022/11/9 17:26
# @Desc    :


from __future__ import division
import string
import math

CHARS = string.ascii_uppercase + string.ascii_lowercase


class RailFence:

    def __init__(self, mask):

        self.mask = mask
        self.row = len(self.mask)

        self.length = 0
        self.column = 0
        self.no_white_space = ''
        self.order = []

    def encrypt(self, src, is_drop_white_space=False):
        """
        :param src: 需要加密的字符串
        :param is_drop_white_space: 是否需要剔除空格
        :return:
        """
        if not isinstance(src, str):
            raise TypeError('Encryption src text is not string')

        if is_drop_white_space:

            for i in src:
                if i in string.whitespace:
                    continue

                self.no_white_space += i

        else:
            self.no_white_space = src

        self.length = len(self.no_white_space)
        self.column = int(math.ceil(self.length / self.row))
        # print("length: %d, column: %d" % (self.length, self.column))
        self.__check()

        # get mask order
        self.__get_order()

        grid = [[] for _ in range(self.row)]
        # print(grid)
        for c in range(self.column):
            end_index = (c + 1) * self.row
            # print(end_index)
            if end_index > self.length:
                end_index = self.length
            r = self.no_white_space[c * self.row: end_index]
            # print(r + "#")
            for i, j in enumerate(r):
                if self.mask and len(self.order) > 0:
                    grid[self.order[i]].append(j)
                else:
                    grid[i].append(j)
            # print(grid)
        return ''.join([''.join(l) for l in grid])

    def decrypt(self, dst):

        self.length = len(dst)
        self.column = int(math.ceil(self.length / self.row))
        try:
            self.__check()
        except Exception as msg:
            print(msg)
        # get mask order
        self.__get_order()

        grid = [[] for i in range(self.row)]
        space = self.row * self.column - self.length
        ns = self.row - space
        prev_e = 0
        for i in range(self.row):
            if self.mask:
                s = prev_e
                o = 0
                for x, y in enumerate(self.order):
                    if i == y:
                        o = x
                        break
                if o < ns:
                    e = s + self.column
                else:
                    e = s + (self.column - 1)
                r = dst[s: e]
                prev_e = e
                grid[o] = list(r)
            else:

                if i < self.row - space:
                    start_index = i * self.column
                    end_index = start_index + self.column
                else:
                    start_index = ns * self.column + (i - ns) * (self.column - 1)
                    end_index = start_index + (self.column - 1)
                r = dst[start_index:end_index]
                grid[i] = list(r)
        res = ''
        for c in range(self.column):
            for i in range(self.row):
                line = grid[i]
                if len(line) == c:
                    res += ' '
                else:
                    res += line[c]
        return res.strip()

    def __check(self):
        # The length of column must be equal or bigger than 2
        if self.column < 2:
            raise Exception('Unexpected column number')

        # The length of the mask must be equal to row
        if self.mask and len(self.mask) != self.row:
            raise ValueError('Mask length not match, must be equal to row')

    def __get_order(self):
        """
        获取密钥的ASCII码的顺序
        :return: 示例 [2, 1, 0, 3]
        """
        if self.mask:
            mask_order = []
            for i in self.mask:
                mask_order.append(CHARS.index(i))
            ordered = sorted(mask_order, reverse=False)
            for i in range(self.row):
                now = mask_order[i]
                for j, k in enumerate(ordered):
                    if k == now:
                        self.order.append(j)
                        break
            # (self.order)


def encrypt(src, mask=None, is_drop_white_space=False):
    rf = RailFence(mask)
    return rf.encrypt(src, is_drop_white_space)


def decrypt(dst, mask=None):
    rf = RailFence(mask)
    return rf.decrypt(dst)


def test():
    account = "12345678adb"
    # 123 456 78a bc
    pass_wd = "crm"  # ascii 顺序 021
    # 132 465 7a8 bc
    s_account = str(account)
    en_account = encrypt(s_account, pass_wd, True)
    print(s_account, en_account)

    de_account = decrypt(en_account, pass_wd)
    if s_account != de_account:
        print(s_account, en_account, de_account, "not equal")

    print(s_account, en_account, de_account)


if __name__ == '__main__':
    test()
