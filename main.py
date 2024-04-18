import vedio_to_image
import json
import numpy as np
import cv2
import process_data
import identify_gray_nose
import multi_ostu

"""将视频截取为图片"""
# # 视频路径
# video_path = "/home/yechentao/Programs/Infrared/vedio/color3.mp4"
# # 图片保存路径
# output_directory = "/home/yechentao/Programs/Infrared/image/color3"
# # 每隔多少帧截取一次
# frame_interval = 10
# # 是否只截取一张图片
# isoneimage = False
# # 将视频截取为图片
# vedio_to_image.video_to_image(video_path, output_directory, frame_interval, isoneimage)

"""识别鼻子"""
gray_path = '/home/mengqingyi/code/yechentao/Infrared_vedio_image/image/gray8/frame_1.jpg'
color_path = '/home/mengqingyi/code/yechentao/Infrared_vedio_image/image/color8/frame_1.jpg'
# 识别
identify_gray_nose.identify_nose(gray_path, color_path)

"""呼吸波形图"""
# # 指定包含JSON文件的文件夹路径
# json_folder_path = '/home/yechentao/Programs/Infrared/newJson'
# # 指定包含图像文件的文件夹路径
# image_folder_path = '/home/yechentao/Programs/Infrared/grayimage'
# # 计算鼻孔处的像素变化值
# process_data.process(json_folder_path, image_folder_path)