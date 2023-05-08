import os
import re
import cv2
import time
import numpy as np
import pandas as pd
import math
import logging
import uuid
import pytesseract
from pytesseract import Output
from paddleocr import PaddleOCR, paddleocr

paddleocr.logging.disable(logging.DEBUG)
paddleocr.logging.disable(logging.WARNING)
from horizontal_correction import HorizontalCorrection  # NOQA: E402


class HouseholdRegister:
    def __init__(self, data_dir) -> None:
        self.ocr = PaddleOCR()
        self.horizontal_corrector = HorizontalCorrection()
        self.data_dir = data_dir
        self.min_pixel = 10
        self.structure = [
            [
                {"type": "fix", "value": "姓名", "enum": ["姓名", "姓", "名"]},
                {"type": "ocr", "key": "姓名", "value": ""},
                {"type": "fix", "value": "户主或与户主关系"},
                {
                    "type": "ocr",
                    "key": "户主或与户主关系",
                    "value": "",
                    "enum": {
                        "非亲属": ["非亲属"],
                        "户主": ["户主", "产主"],
                        "夫": ["夫"],
                        "妻": ["妻"],
                    },
                },
            ],
            [
                {"type": "fix", "value": "曾用名"},
                {"type": "ocr", "key": "曾用名", "value": ""},
                {"type": "fix", "value": "性别"},
                {"type": "ocr", "key": "性别", "value": "", "enum": ["男", "女"]},
            ],
            [
                {"type": "fix", "value": "出生地"},
                {"type": "ocr", "key": "出生地", "value": ""},
                {"type": "fix", "value": "民族"},
                {"type": "ocr", "key": "民族", "value": "", "enum": ["汉族", "汉"]},
            ],
            [
                {"type": "fix", "value": "籍贯"},
                {"type": "ocr", "key": "籍贯", "value": ""},
                {"type": "fix", "value": "出生日期"},
                {"type": "ocr", "key": "出生日期", "value": "",},
            ],
            [
                {"type": "fix", "value": "本市（县）其他地址"},
                {"type": "ocr", "key": "本市（县）其他地址", "value": ""},
                {"type": "fix", "value": "宗教信仰"},
                {"type": "ocr", "key": "宗教信仰", "value": "",},
            ],
            [
                {"type": "fix", "value": "公民身份号码"},
                {"type": "ocr", "key": "公民身份号码", "value": ""},
                {"type": "fix", "value": "身高"},
                {"type": "ocr", "key": "身高", "value": "",},
                {"type": "fix", "value": "血型"},
                {
                    "type": "ocr",
                    "key": "血型",
                    "value": "",
                    "enum": ["O型", "A型", "B型", "AB型", "不明"],
                },
            ],
            [
                {"type": "fix", "value": "文化程度"},
                {"type": "ocr", "key": "文化程度", "value": ""},
                {"type": "fix", "value": "婚姻状况"},
                {"type": "ocr", "key": "婚姻状况", "value": "", "enum": ["未婚", "已婚"]},
                {"type": "fix", "value": "兵役状况"},
                {
                    "type": "ocr",
                    "key": "兵役状况",
                    "value": "",
                    "enum": {
                        "未服兵役": ["未服兵役"],
                        "服现役": ["服现役", "服现"],
                        "服役": ["服役"],
                        "退出现役": ["退出现役"],
                    },
                },
            ],
            [
                {"type": "fix", "value": "服务处所"},
                {"type": "ocr", "key": "服务处所", "value": ""},
                {"type": "fix", "value": "职业"},
                {
                    "type": "ocr",
                    "key": "职业",
                    "value": "",
                    "enum": {
                        "干部": ["千部", "干部"],
                        "人民警察": ["人民警察"],
                        "公安干警": ["公安干警"],
                        "职员": ["职员"],
                        "农民": ["农民"],
                        "主持人": ["主持人"],
                        "待业": ["待业"],
                        "工程师": ["工程师"],
                    },
                },
            ],
            [
                {"type": "fix", "value": "何时由何地迁来本市（县）"},
                {"type": "ocr", "key": "何时由何地迁来本市（县）", "value": ""},
            ],
            [
                {"type": "fix", "value": "何时由何地迁来本址"},
                {"type": "ocr", "key": "何时由何地迁来本址", "value": "",},
            ],
        ]

    def target_line_number(self, sort_bound, current_index, line_num):
        if current_index == 0:
            return 0
        _, _, y, _, h = tuple(sort_bound[current_index])
        _, _, last_y, _, _ = tuple(sort_bound[current_index - 1])
        if y <= last_y + h + self.min_pixel:
            return line_num + 1
        return line_num

    def rotate(self, image):
        rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        results = pytesseract.image_to_osd(
            rgb, lang="chi_sim+eng", output_type=Output.DICT
        )

        rotate = results["rotate"]
        if rotate > 0:
            dts = None
            if rotate == 90:
                dts = cv2.ROTATE_90_CLOCKWISE
            elif rotate == 180:
                dts = cv2.ROTATE_180
            elif rotate == 270:
                dts = cv2.ROTATE_90_COUNTERCLOCKWISE
            if dts is not None:
                image2 = cv2.rotate(image, dts)
                return image2
        return image

    def find_structure(self, image):
        # 灰度图片
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        binary = cv2.adaptiveThreshold(
            ~gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 35, -5
        )

        h, w = binary.shape
        hors_k = int(math.sqrt(w) * 1.2)
        vert_k = int(math.sqrt(h) * 1.2)
        # 自适应获取核值
        # 识别横线:
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (hors_k, 1))
        eroded = cv2.erode(binary, kernel, iterations=1)
        dilated_col = cv2.dilate(eroded, kernel, iterations=1)

        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, vert_k))
        eroded = cv2.erode(binary, kernel, iterations=1)
        dilated_row = cv2.dilate(eroded, kernel, iterations=1)

        # 将识别出来的横竖线合起来
        bitwise_and = cv2.bitwise_and(dilated_col, dilated_row)

        return dilated_col, dilated_row

    def ocr_text(self, image):
        text = self.ocr.ocr(image, cls=True)
        text = text[0]
        text = sorted(text, key=lambda x: x[0][0][0] * x[0][0][1])
        text2 = ""
        if len(text) > 0:
            for t in text:
                if len(t) == 0:
                    break
                if t[1][0] == "":
                    break
                text2 += t[1][0]
        return text2

    def log_image(self, temp_dir, name, image):
        filepath = os.path.join(temp_dir, name)
        cv2.imwrite(filepath, image)

    def find_counter(self, image, temp_dir=None):
        dilated_col, dilated_row = self.find_structure(image)

        # 标识表格轮廓
        merge = cv2.add(dilated_col, dilated_row)
        cv2.imwrite("merge.png", merge)

        # 获取轮廓，并筛选轮廓
        contours, hierarchy = cv2.findContours(
            merge, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE
        )
        hierarchy = hierarchy[0]
        drop_hierarchy = set()
        for i, contour in enumerate(contours):
            x, y, w, h = cv2.boundingRect(contour)
            if h <= self.min_pixel or w <= self.min_pixel:
                drop_hierarchy.add(i)
        sort_bound = []
        for i, contour in enumerate(contours):
            x, y, w, h = cv2.boundingRect(contour)

            im = image[y : y + h, x : x + w]
            current_hierarchy = hierarchy[i]
            # 子轮廓为空或者没有信息的轮廓 或者 或者没有父轮廓，则丢弃这个轮廓
            if (
                (
                    current_hierarchy[-2] != -1
                    and current_hierarchy[-2] not in drop_hierarchy
                )
                or current_hierarchy[-1] == -1
                or i in drop_hierarchy
            ):
                if temp_dir is not None:
                    self.log_image(temp_dir, f"drop_{i}_{current_hierarchy}.png", im)
                continue
            text = self.ocr_text(im)
            if temp_dir is not None:
                self.log_image(temp_dir, f"keep_{i}_{text}.png", im)
            sort_bound.append([i, x, y, w, h, text])
        return sort_bound

    # 根据轮廓左上角的y坐标对轮廓排序，相近的y为一行，同一行内再根据x坐标排序
    def line_up_by_location(self, sort_bound):
        groups = [[]]
        avg = []
        median_h = np.median([int(x[-2]) for x in np.squeeze(sort_bound)])
        bread_avg = [median_h]

        sort_bound.sort(key=lambda x: x[2])
        for _, value in enumerate(sort_bound):
            if len(groups[-1]) > 1:
                u = np.mean(avg)
                diff = abs(int(value[2]) - u)
                std = math.ceil(np.std(avg, ddof=1))
                print(f"diff:{diff}, std:{std}, avg:{avg}")

                if (
                    diff > self.min_pixel
                    and diff >= std * 3
                    or abs(diff - np.mean(bread_avg)) <= self.min_pixel
                ):
                    bread_avg.append(int(diff))
                    groups[-1] = sorted(groups[-1], key=lambda x: x[1])
                    groups.append([])
                    avg = []
            groups[-1].append(value)
            avg.append(int(value[2]))
        return groups

    def fix_ocr(self, structure, ocr):
        # 获取当前结构的枚举值，如果有枚举值就用枚举值，如果没有就用value
        if "enum" in structure and isinstance(structure["enum"], list):
            structure_enum = structure["enum"]
            pattern = re.compile("|".join(structure_enum))
            rs = re.findall(pattern, ocr)
            if len(rs) > 0:
                return True, rs[0]
        elif "enum" in structure and isinstance(structure["enum"], dict):
            for key in structure["enum"]:
                structure_enum = structure["enum"][key]
                pattern = re.compile("|".join(structure_enum))
                rs = re.findall(pattern, ocr)
                if len(rs) > 0:
                    return True, key
        else:
            structure_value = structure["value"]
            if structure_value in ocr:
                return True, structure_value

        return False, ""

    def counter_2_table(self, sort_bound):
        groups = self.line_up_by_location(sort_bound)
        user_info = {}
        for x, group in enumerate(groups):
            # 读取当前行的结构
            if x >= len(self.structure):
                break
            current_structure = self.structure[x]
            ocr_cursor = 0
            structure_cursor = 0
            match_structure_cursor = []
            # 遍历直到ocr识别的内容结束为止
            while ocr_cursor < len(group):
                current_ocr_text = group[ocr_cursor][-1]

                structure = current_structure[structure_cursor]
                find_ocr_rs, find_value = self.fix_ocr(structure, current_ocr_text)
                print(
                    f"find_ocr_rs:{find_ocr_rs}, find_value:{find_value}, current_ocr_text:{current_ocr_text}"
                )
                # 如果当前结构是固定结构，
                if current_structure[structure_cursor]["type"] == "fix" and find_ocr_rs:
                    ocr_cursor += 1
                    structure_cursor += 1
                    match_structure_cursor.append((ocr_cursor, structure_cursor))
                    if ocr_cursor >= len(group):
                        break
                    name = current_structure[structure_cursor]["key"]
                    ocr_structure = current_structure[structure_cursor]
                    current_ocr_text = group[ocr_cursor][-1]
                    if "enum" in ocr_structure:
                        find_ocr_rs, find_value = self.fix_ocr(
                            ocr_structure, current_ocr_text
                        )
                        print(
                            f"fix find_ocr_rs:{find_ocr_rs}, find_value:{find_value}, current_ocr_text:{current_ocr_text} structure:{ocr_structure}"
                        )
                        if find_ocr_rs:
                            current_ocr_text = find_value
                    user_info[name] = current_ocr_text
                    ocr_cursor += 1
                    structure_cursor += 1

                elif (
                    current_structure[structure_cursor]["type"] == "ocr" and find_ocr_rs
                ):
                    name = current_structure[structure_cursor]["key"]
                    current_ocr_text = group[ocr_cursor][-1]
                    user_info[name] = find_value
                    match_structure_cursor.append((ocr_cursor, structure_cursor))
                    ocr_cursor += 1
                    structure_cursor += 1
                else:
                    structure_cursor += 1

                if structure_cursor >= len(current_structure) and ocr_cursor < len(
                    group
                ):

                    # 如果已经存在匹配的，从匹配后开始遍历，否则从0开始
                    if len(match_structure_cursor) > 0:
                        structure_cursor = match_structure_cursor[-1][-1] + 1
                        if structure_cursor >= len(current_structure):
                            break
                    else:
                        structure_cursor = 0
                    ocr_cursor += 1
        return user_info

    def inference(self, image_file):
        id = str(uuid.uuid1())
        temp_dir = os.path.join(self.data_dir, id)
        os.mkdir(temp_dir)
        image = cv2.imread(image_file, 1)
        rotate_image = self.rotate(image)
        correct_image, _, _ = self.horizontal_corrector.process(rotate_image)
        # 找到每个单元格的轮廓
        sort_bound = self.find_counter(correct_image, temp_dir)
        # 根据结构解析户口本，并返回户口本信息
        user_info = self.counter_2_table(sort_bound)
        user_info["id"] = id
        return user_info


if __name__ == "__main__":
    data_root = "/Users/xiangyuwang/Desktop/户口图片"
    data_dir = "/Users/xiangyuwang/Desktop/户口图片temp"
    hr = HouseholdRegister(data_dir)

    # name_list = os.listdir(data_root)
    # name_list = [name for name in name_list if name.endswith(".png") or name.endswith(".jpg")]
    # user_info_list = []
    # for name in name_list:
    #     # if name.find("w202210261148271563.jpg") == -1:
    #     #     continue
    #     image_file = os.path.join(data_root, name)
    #     try:
    #         start = time.time()
    #         user_info = hr.inference(image_file)
    #         print(f"cost:{time.time()-start}")
    #     except:
    #         print(f"error {image_file}")
    #         continue
    #     user_info_list.append(user_info)
    #     user_info["image_file"] = image_file
    #     print(user_info)
    # df = pd.DataFrame(user_info_list)
    # df.to_excel("/Users/xiangyuwang/Desktop/user_info.xlsx", index=False)

    image_file = "/Users/xiangyuwang/Works/KFBusiness/teen_refund/teen_refund/pic_ocr/temp/0b3d81b4-a691-11ed-ad5b-c6e2a36e5339.jpg"
    rs = hr.inference(image_file)

    # image_file = os.path.join(data_root, "w202210261148271563.jpg")
    # image = cv2.imread(image_file, 1)
    # rotate_image = hr.rotate(image)
    # correct_image, _, _ = hr.horizontal_corrector.process(rotate_image)
    # cv2.imwrite("rotate_image.png", rotate_image)
    # cv2.imwrite("correct_image.png", correct_image)

    # sort_bound = hr.find_counter(correct_image, temp_dir="/Users/xiangyuwang/Desktop/test/temp/test")
    # groups = hr.line_up_by_location(sort_bound)
    # user_info = hr.counter_2_table(sort_bound)
    # print(user_info)
