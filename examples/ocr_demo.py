# -*- coding: utf-8 -*-
"""
@author:XuMing(xuming624@qq.com)
@description: Image ocr demo.
"""
import time
import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from imgocr import ImgOcr
from imgocr import draw_ocr_boxes


if __name__ == "__main__":
    m = ImgOcr(use_gpu=False)
    pwd_path = os.path.dirname(os.path.abspath(__file__))
    img_path = os.path.join(pwd_path, "data/11.jpg") # "data/11.jpg"
    s = time.time()
    result = m.ocr(img_path)
    e = time.time()
    print("total time: {:.4f} s".format(e - s))
    print("result:", result)
    for i in result:
        print(i['text'])

    # draw boxes
    save_img_path = os.path.join(pwd_path, '11_box.jpg')
    draw_ocr_boxes(img_path, result, save_img_path)
    print(f'Save result to {save_img_path}')
