import cv2
import numpy as np
from matplotlib import pyplot as plt
import os
import sys
from scipy.signal import butter, filtfilt
import matplotlib.pyplot as plt
from scipy.fft import fft, fftfreq
from PyEMD import EMD, Visualisation
from scipy.signal import find_peaks
from scipy.fft import fft, fftfreq  # 傅里叶变化

import argparse
from filter import *
# from frequency_method import *

import math
                      
# signal_list = []
# video_path = 'nose.mp4' # frame_size:456,250
# cap = cv2.VideoCapture(video_path)
# width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))  #获取视频的宽度
# height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)) 

# # 读取图像并创建一个视频写入器
# fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # 设置编码格式为MP4
# out = cv2.VideoWriter('output_video.mp4', fourcc, 25.0, (width, height))


# # 逐帧处理视频
# while True:
#     # 读取一帧
#     ret, frame = cap.read() # 剪映裁剪后有三个通道 (1080, 1920, 3)  原视频每一帧的维度 (720, 1280, 3)
#     if not ret:
#         break  # 视频结束
    
#     # 图像预处理 转换为灰度
#     gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
#     # gray_frame = Histogram_Equalization(gray_frame)
#     signal_list.append(gray_frame)


def rom(original_signal):
    # ROM 的算法
    threhod = 125 
    signal_list = []
    
    for i in range(1,len(original_signal)):
        diff = original_signal[i] - original_signal[i-1]
        abs_diff = np.abs(diff)
        # abs_diff = cv2.absdiff(original_signal[i],original_signal[i-1])
        cv2.imwrite(os.path.join('rom1_diff', f"diff_frame_{i}.png"),abs_diff)
        _, binary_image= cv2.threshold(abs_diff, threhod, 255, cv2.THRESH_BINARY)
        
        cv2.imwrite(os.path.join('rom1_binary_imag', f"binary_frame_{i}.png"),binary_image)
        
        # # 填充空洞
        # binary_image = cv2.morphologyEx(binary_image, cv2.MORPH_CLOSE, np.ones((5, 5), np.uint8))
        
        # 开运算
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
        binary_image = cv2.morphologyEx(binary_image, cv2.MORPH_OPEN, kernel, iterations=1)
        cv2.imwrite(os.path.join('rom1_open', f"open_frame_{i}.png"),binary_image)
        
        # 连通域分析
        contours, _ = cv2.findContours(binary_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if contours:  # 确保存在轮廓
        # 假设最大的连通区域就是运动目标
        # 根据轮廓面积排序，选择最大的轮廓
            contour = max(contours, key=cv2.contourArea)
            # 创建一个全黑的图像
            R = np.zeros_like(binary_image)
            # 将最大的连通区域绘制到全黑图像上，白色代表运动目标
            cv2.drawContours(R, [contour], -1, (255, 255, 255), -1)
            cv2.imwrite(os.path.join('rom1_template', f"rom_template_{i}.png"),R)
            
            # 计算轮廓在原始图像中的位置
            x, y, w, h = cv2.boundingRect(contour)
            
            # 提取原始图像中的对应区域
            if i == 1:
                # 第1张图像使用前两张的差分模版进行裁剪
                cropped_image = original_signal[0].copy()[y:y+h, x:x+w]
                signal_list.append(cropped_image)
                cv2.imwrite(os.path.join('rom1_cropped_image', f"rom_cropped_image_0.png"),cropped_image)
                
                img = cv2.rectangle(original_signal[0].copy(),(x,y),(x+w,y+h),(0,255,0),2)
                cv2.imwrite(os.path.join('rom1_feature', f"rom_frame_0.png"),img)
                
                cropped_image = original_signal[i][y:y+h, x:x+w]
                cv2.imwrite(os.path.join('rom1_cropped_image', f"rom_cropped_image_1.png"),cropped_image)
                signal_list.append(cropped_image)
                
                img = cv2.rectangle(original_signal[1].copy(),(x,y),(x+w,y+h),(0,255,0),2)
                cv2.imwrite(os.path.join('rom1_feature', f"rom_frame_1.png"),img)
                
            else:
                cropped_image = original_signal[i][y:y+h, x:x+w]
                signal_list.append(cropped_image)
                cv2.imwrite(os.path.join('rom1_cropped_image', f"rom_cropped_image_{i}.png"),cropped_image)
                
                img = cv2.rectangle(original_signal[i].copy(),(x,y),(x+w,y+h),(0,255,0),2)
                cv2.imwrite(os.path.join('rom1_feature', f"rom_frame_{i}.png"),img)
        else:
            # 如果不存在连通域，保留原图为特征图
            if i==1:
                signal_list.append(original_signal[0])
                signal_list.append(original_signal[1])
                cv2.imwrite(os.path.join('rom1_cropped_image', f"rom_cropped_image_0.png"),cropped_image)
                cv2.imwrite(os.path.join('rom1_cropped_image', f"rom_cropped_image_1.png"),cropped_image)
            else:
                signal_list.append(original_signal[i])
                cv2.imwrite(os.path.join('rom1_cropped_image', f"rom_cropped_image_{i}.png"),cropped_image)
            
    # out.release()
    return signal_list
    

# 这个暂时有点问题，保留两个联通区域之后呢？
def rom2(original_signal):
    # 改进的ROM 的算法
    signal_list = []
    abs_diff_list = []
    group_each = 20
    group_num = len(original_signal)//group_each
    
    # 分成group_each个子列表
    signal = [original_signal[i:i + group_each] for i in range(0, len(original_signal), group_each)]
    
    #原始信号差分
    for i in range(1,len(original_signal)):
        diff = original_signal[i] - original_signal[i-1]
        abs_diff = np.abs(diff)
        abs_diff_list.append(abs_diff)
    
    # 20帧为一组累加
    for i in range(group_num):
        diff_group = abs_diff_list[i:i+20]
        diff_sum = np.add.reduce(diff_group) # 对位相加==>uint32
        
        diff_sum = np.array(diff_sum, dtype=np.uint8)
        cv2.imwrite(os.path.join('rom2_diff', f"diff_frame_{i}.png"),diff_sum)
          
        # _, binary_image= cv2.threshold(abs_diff, 125, 255, cv2.THRESH_BINARY)
        
        ## 使用大津阈值法计算阈值 cv2.THRESH_OTSU
        threshold_otsu, _ = cv2.threshold(diff_sum, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        _, binary_image= cv2.threshold(diff_sum, threshold_otsu, 255, cv2.THRESH_BINARY)
        
        cv2.imwrite(os.path.join('rom2_binary_image', f"binary_frame_{i}.png"),binary_image)
        
        # 创建一个核，用于开运算
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
        binary_image = cv2.morphologyEx(binary_image, cv2.MORPH_OPEN, kernel, iterations=2)
        cv2.imwrite(os.path.join('rom2_open', f"open_frame_{i}.png"),binary_image)
        
        # 填充空洞
        # binary_image = cv2.morphologyEx(binary_image, cv2.MORPH_CLOSE, np.ones((5, 5), np.uint8))
        contours, _ = cv2.findContours(binary_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        if contours: 
            # contours = sorted(contours, key=cv2.contourArea, reverse=True)
            contours = sorted(contours, key=lambda x: cv2.contourArea(x), reverse=True)
            # 创建一个全黑的图像
            R = np.zeros_like(binary_image)
            for j in range(2) if len(contours)>=2 else range(len(contours)):                
                if len(contours) >= 2:
                    cv2.drawContours(R, [contours[j]], -1, (255, 255, 255), -1)
                    cv2.imwrite(os.path.join('rom2_template', f"rom_template_{i}.png"),R)

                else:
                    cv2.drawContours(R, [contours[0]], -1, (255, 255, 255), -1)
                    cv2.imwrite(os.path.join('rom2_template', f"rom_template_{i}.png"),R)
                    
                x, y, w, h = cv2.boundingRect(contours[j])
                for k in range(len(signal[i])):
                    print('*************************:', str(i))
                    img = cv2.rectangle(signal[i][k],(x,y),(x+w,y+h),(0,255,0),2)
                    frame_index = i*20+k
                    print(frame_index)
                    cv2.imwrite(os.path.join('rom2_feature', f"rom_frame_{frame_index}.png"),img)
            
    return signal_list

# rom2(signal_list)