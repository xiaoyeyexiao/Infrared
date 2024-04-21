import cv2
import glob
import os
import sys

# 定义一个自定义的排序函数，用于提取文件名中的数字部分并进行排序
def get_numeric_part(filename):
    return int(''.join(filter(str.isdigit, os.path.basename(filename))))

def extract_ROI(region, gray_directory, region_directory):
    
    # 获取所有的gray图片
    # 这里使用自定义的排序函数，否则会按照ASCII值排序，导致frame_10排在frame_2前面
    gray_paths = sorted(glob.glob(os.path.join(gray_directory, '*.jpg')), key = get_numeric_part)
    
    for i, gray_path in enumerate(gray_paths):
        origin_gray = cv2.imread(gray_path)
        origin_gray = origin_gray[region[0]:region[1], region[2]:region[3]]
         # 新图片的名字与原来的命名方式相同
        new_filename = f'frame_{i + 1}.jpg'
        # 构造完整的存放路径
        output_path = os.path.join(region_directory, new_filename)
        # 保存裁剪后的图片到目标文件夹
        cv2.imwrite(output_path, origin_gray)