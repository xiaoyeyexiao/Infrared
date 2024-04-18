import json
import cv2
import glob
import os
import sys

# 定义一个自定义的排序函数，用于提取文件名中的数字部分并进行排序
def get_numeric_part(filename):
    return int(''.join(filter(str.isdigit, os.path.basename(filename))))

def process(json_folder_path, image_folder_path):

    # 使用glob.glob()找到文件夹内所有的JSON文件
    # 这里使用自定义的排序函数，否则会按照ASCII值排序，导致frame_10排在frame_2前面
    json_files = sorted(glob.glob(os.path.join(json_folder_path, '*.json')), key = get_numeric_part)

    for json_file in json_files:
        # 读取JSON文件
        with open(json_file, 'r') as f:
            data = json.load(f)

        # 获取文件名字(包含扩展名)
        json_filename = os.path.basename(json_file)
        # 加上新的扩展名，得到新的文件名
        json_filename_without_ext, _ = os.path.splitext(json_filename)
        image_filename = json_filename_without_ext + ".jpg"
        # 获取图像路径
        image_path = os.path.join(image_folder_path, image_filename)
        # 加载图像
        image = cv2.imread(image_path)
        # 转换为灰度图
        gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        # 指定输出文件路径
        output_file_path = "/home/yechentao/Programs/Infrared/print_data.txt"
        # 对每个JSON文件中的每个shape进行操作
        for shape in data['shapes']:
            # 提取矩形对角点坐标
            rect_points = shape['points']
            pt1 = tuple(map(int, rect_points[0]))  # 转换为整数元组
            pt2 = tuple(map(int, rect_points[1]))

            # 定位矩形区域并计算平均值
            rect_region = gray_image[pt1[1]:pt2[1], pt1[0]:pt2[0]]
            mean_val = cv2.mean(rect_region)

            # print(f"图像：{image_filename}, 平均值: {mean_val[0]}")

            # 打开输出文件，以追加模式写入
            with open(output_file_path, "a") as output_file:
                output_file.write(f"图像：{image_filename}, 平均值: {mean_val[0]}\n")