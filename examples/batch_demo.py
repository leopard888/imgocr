# -*- coding: utf-8 -*-
"""
@author:XuMing(xuming624@qq.com)
@description: 
"""
import csv
import sys
import os
from tqdm import tqdm
from glob import glob
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from imgocr import ImgOcr, draw_ocr_boxes


if __name__ == '__main__':
    m = ImgOcr(use_gpu=False, is_efficiency_mode=False)
    image_dir = 'data'
    saved_dir = 'ocr_results'
    if not os.path.exists(saved_dir):
        os.makedirs(saved_dir)
    images = glob(image_dir + '/*.[jJpP][pPnN][gG]')
    print(f"Found {len(images)} images in {image_dir}")
    ocr_results = []
    for path in tqdm(images):
        res = m.ocr(path)
        res_list = [i['text'] for i in res if i]
        result = "\n".join(res_list)
        ocr_results.append(result)
        # Save ocr box img
        saved_img_path = os.path.join(saved_dir, os.path.basename(path))
        draw_ocr_boxes(path, res, saved_img_path)
    output_file = os.path.join(saved_dir, 'ocr_results.csv')
    with open(output_file, 'w', encoding='utf-8', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['images', 'ocr_results'])
        writer.writerows(zip(images, ocr_results))
    print(f"OCR results saved to {output_file}")
