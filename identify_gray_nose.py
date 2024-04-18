import sys
import cv2
import os
import numpy as np
import identify_color_nose
import identify_color_body
import identify_gray_body

"""
在本代码中, 图片在水平方向上的长度称为宽, 在竖直方向上的长度称为宽
image.shape: (x, y), x表示第几行, y表示第几列
    原彩色图: (1520, 2688), 宽放大1.33倍后: (1520, 3575), 裁出人脸部分后: (901, 3575), 缩小后: (464, 1841)
    原灰度图: (720, 1280), 裁出人脸部分后: (403, 1280)
color_head: 依次为彩色图中头部的最上边、最左边、最右边
gray_head: 依次为灰度图中头部的最上边、最左边、最右边
color_nose: 依次为彩色图中鼻尖、左鼻孔、右鼻孔的坐标(y,x)
gray_nose: 依次为灰度图中鼻尖、左鼻孔、右鼻孔的坐标(y,x)
nose_dis: 依次为彩色图中鼻尖、左鼻孔、右鼻孔与头部最上边的距离、与头部最左边的距离
color_face_image: 经过放缩、去掉脖子以下区域,只留人脸的彩色图
color_face_image: 经过放缩、去掉脖子以下区域,只留人脸的灰度图
用到的文件: main.py, identify_gray_nose.py, identify_gray_body.py, identify_color_body.py, identify_color_nose.py
"""

def identify_nose(gray_path, color_path):
    
    # 读取图像
    origin_gray = cv2.imread(gray_path)
    color = cv2.imread(color_path)
    # origin_gray是三通道的，gray才是真正的灰度图
    gray = cv2.cvtColor(origin_gray, cv2.COLOR_BGR2GRAY)
    
    """得到彩色图的人脸位置和鼻子位置"""
    # 这里注意！！！鼻子坐标的存储形式为[y, x]
    # 先在原图上找到鼻孔,因为放缩图片后,可能找不到鼻孔的位置
    color_nose = identify_color_nose.identify_color_nose(color)
    
    # 将图片的宽放大到原来的1.33倍, 宽高比变为2.35
    color = cv2.resize(color, (0, 0), fx = 1.33, fy = 1)
    color_head = [[]]
    color_face_image, color_head= identify_color_body.identify_color_body(color_path)
    print("-----得到彩色图的人脸位置和鼻子位置-----")
    print("color_head(top_row, left_col, right_col): ", color_head)
    
    # 拉伸图片的长后,在原图上的鼻孔坐标的纵坐标y也要放缩(横坐标x不用)
    for i in range(3):
        color_nose[i][0] = int(color_nose[i][0] * 1.33)
    print("color_nose(y, x): ", color_nose)
    
    # 在彩色图上绘制鼻子区域
    cv2.circle(color_face_image, color_nose[0], 3, (0, 255, 0), -1)
    cv2.circle(color_face_image, color_nose[1], 3, (0, 255, 0), -1)
    cv2.circle(color_face_image, color_nose[2], 3, (0, 255, 0), -1)
    
    # """显示图片"""
    # print("-----show image-----")
    # # 创建窗口
    # cv2.namedWindow('Resizable Window', cv2.WINDOW_NORMAL)
    # # 显示图片
    # cv2.imshow('Resizable Window', color_face_image)
    # # 调整窗口大小
    # cv2.resizeWindow('Resizable Window', 800, 600)  # 设置窗口大小为800x600
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()
    # sys.exit()
    
    """得到灰度图的人脸位置"""
    print("-----得到灰度图的人脸位置-----")
    gray_head = [[]]
    gray_face_image, gray_head = identify_gray_body.identify_gray_body(gray_path)
    print("gray_head(top_row, left_col, right_col): ", gray_head)
    
    # """显示图片"""
    # print("-----show image-----")
    # # 创建窗口
    # cv2.namedWindow('Resizable Window', cv2.WINDOW_NORMAL)
    # # 显示图片
    # cv2.imshow('Resizable Window', gray_face_image)
    # # 调整窗口大小
    # cv2.resizeWindow('Resizable Window', 800, 600)  # 设置窗口大小为800x600
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()
    # sys.exit()
    
    """确定灰度图中鼻子的大致位置"""
    # 计算彩色图和灰度图中头的宽度
    color_head_width = color_head[2] - color_head[1]
    gray_head_width = gray_head[2] - gray_head[1]
    # 为了让彩色图中头的宽度与灰度图的相等，将彩色图等比例缩小
    ratio = gray_head_width / color_head_width
    color_face_image = cv2.resize(color_face_image, (0, 0), fx = ratio, fy = ratio)
    
    # 缩小图片后,原图上的鼻子坐标和人脸位置都要按相同的比例变换
    for i in range(3):
        for j in range(2):
         color_nose[i][j] = int(color_nose[i][j] * ratio)
    for i in range(3):
        color_head[i] = int(color_head[i] * ratio)
    print("-----确定灰度图中鼻子的大致位置-----")
    print("color_nose(y, x): ", color_nose)
    print("color_head(top_row, left_col, right_col): ", color_head)
    
    # """显示图片"""
    # print("-----show image-----")
    # # 创建窗口
    # cv2.namedWindow('Resizable Window', cv2.WINDOW_NORMAL)
    # # 显示图片
    # cv2.imshow('Resizable Window', color_face_image)
    # # 调整窗口大小
    # cv2.resizeWindow('Resizable Window', 800, 600)  # 设置窗口大小为800x600
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()
    # sys.exit()
    
    # 计算出彩色图中鼻子位置(y, x)距离左边的距离和上边的距离
    nose_dis = np.zeros((3, 2), dtype=int)
    for i in range(3):
        nose_dis[i][0] = color_nose[i][0] - color_head[1]
        nose_dis[i][1] = color_nose[i][1] - color_head[0]
    print("nose_dis(left, top): ", nose_dis)

    mid_col = (gray_head[2] + gray_head[1]) // 2
    # 确定灰度图中鼻子的大致位置(y, x)
    gray_nose = np.zeros((3, 2), dtype=int)
    for i in range(3):
        # gray_nose[i][0] = int(nose_dis[i][0] * left_ratio) + gray_head[1]
        gray_nose[i][0] = nose_dis[i][0] + gray_head[1]
        gray_nose[i][1] = nose_dis[i][1] + gray_head[0]
    move = mid_col - gray_nose[0][0]
    # print("mid_cal: ", mid_col)
    # print("move: ", move)
    # 将鼻子的位置调整到人脸中间
    for i in range(3):
        gray_nose[i][0] = gray_nose[i][0] + move
    # 在原来的灰度图上绘制鼻子区域
    cv2.circle(gray, gray_nose[0], 3, (0, 255, 0), -1)
    cv2.circle(gray, gray_nose[1], 3, (0, 255, 0), -1)
    cv2.circle(gray, gray_nose[2], 3, (0, 255, 0), -1)
    print("gray_nose(y, x): ", gray_nose)
    
    """显示图片"""
    print("-----show image-----")
    # 创建窗口
    cv2.namedWindow('Resizable Window', cv2.WINDOW_NORMAL)
    # 显示图片
    cv2.imshow('Resizable Window', gray)
    # 调整窗口大小
    cv2.resizeWindow('Resizable Window', 800, 600)  # 设置窗口大小为800x600
    cv2.waitKey(0)
    cv2.destroyAllWindows()
