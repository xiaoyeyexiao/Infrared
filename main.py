import vedio_to_image
import os
import numpy as np
import cv2
import extract_ROI
import identify_ROI

"""
用到的文件: main.py, vedio_to_image.py, identify_ROI.py, identify_gray_face.py, identify_color_face.py, 
           identify_color_feature.py, extract_ROI.py
"""

# 选择要识别对象
i = 8

"""将视频截取为图片"""
# 视频路径
color_vedio_path = f'/home/mengqingyi/code/yechentao/Infrared_vedio_image/vedio/color{i}.mp4'
gray_video_path = f'/home/mengqingyi/code/yechentao/Infrared_vedio_image/vedio/gray{i}.mp4'
# 图片保存路径
color_directory = f'/home/mengqingyi/code/yechentao/Infrared_vedio_image/image/color{i}'
gray_directory = f'/home/mengqingyi/code/yechentao/Infrared_vedio_image/image/gray{i}'
# 假如目录不存在，则创建对应目录
os.makedirs(color_directory, exist_ok=True)
os.makedirs(gray_directory, exist_ok=True)
# 每隔多少帧截取一次
frame_interval = 1
# 是否只截取一张图片
isoneimage = True
# 将视频截取为图片
# vedio_to_image.video_to_image(color_vedio_path, color_directory, frame_interval, isoneimage)
# vedio_to_image.video_to_image(gray_video_path, gray_directory, frame_interval, isoneimage)


"""识别ROI区域"""
# 要识别的图片路径
color_path = color_directory + '/frame_1.jpg'
gray_path = gray_directory + '/frame_1.jpg'
# 分别存储6个ROI区域的坐标
nose_region = []
left_forehead_region = []
right_forehead_region = []
left_cheek_region = []
right_cheek_region = []
jaw_region = []
# 识别
# nose_region, left_forehead_region, right_forehead_region, left_cheek_region, right_cheek_region, jaw_region = identify_ROI.identify_ROI(gray_path, color_path)


"""提取ROI区域"""
maps = {'nose': nose_region,
        'left_forehead': left_forehead_region,
        'right_forehead': right_forehead_region,
        'left_cheek': left_cheek_region,
        'right_cheek': right_cheek_region,
        'jaw': jaw_region}
# gray_directory = f'/home/mengqingyi/code/yechentao/Infrared_vedio_image/testimage/newgray{i}'
for key, value in maps.items():
    # ROI区域图片保存路径
    region_directory = '/home/mengqingyi/code/yechentao/Infrared_vedio_image/region/' + key + f'/gray{i}'
    # 假如目录不存在，则创建对应目录
    os.makedirs(region_directory, exist_ok=True)
    # 将所有灰度图的各个ROI截取出来，存放进相应的文件夹中
    # extract_ROI.extract_ROI(value, gray_directory, region_directory)