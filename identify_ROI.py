import sys
import cv2
import os
import numpy as np
import identify_color_feature
import identify_color_face
import identify_gray_face

"""
1. 在本代码中, 图片在水平方向上的长度称为宽, 在竖直方向上的长度称为宽

2. image.shape: (x, y), x表示第几行, y表示第几列
    原彩色图: (1520, 2688), 宽放大1.33倍后: (1520, 3575), 裁出人脸部分后: (901, 3575), 缩小后: (464, 1841)
    原灰度图: (720, 1280), 裁出人脸部分后: (403, 1280)
    
3. 部分重要变量解释: 
    color_head: 依次为彩色图中头部的最上边、最左边、最右边
    gray_head: 依次为灰度图中头部的最上边、最左边、最右边
    color_nose: 依次为彩色图中鼻尖、左鼻孔、右鼻孔的坐标, 坐标形式为(y,x)
    gray_nose: 依次为灰度图中鼻尖、左鼻孔、右鼻孔的坐标, 坐标形式为(y,x)
    nose_dis: 依次为彩色图中鼻尖、左鼻孔、右鼻孔与头部最上边的距离、与头部最左边的距离
    color_face_image: 经过放缩、去掉脖子以下区域,只留人脸的彩色图
    gray_face_image: 经过放缩、去掉脖子以下区域,只留人脸的灰度图
    nose_region: 灰度图中的鼻子区域, 依次为上边、下边、左边、右边
"""

def identify_ROI(gray_path, color_path):
    
    # 读取图像
    color = cv2.imread(color_path)
    origin_gray = cv2.imread(gray_path)
    # origin_gray是三通道的，gray才是真正的灰度图
    gray = cv2.cvtColor(origin_gray, cv2.COLOR_BGR2GRAY)
    
    """得到彩色图的人脸坐标和各个特征点的坐标"""
    # 这里注意！！！特征点的存储形式为[y, x]
    # 先在原图上找到特征点,因为放缩图片后,可能找不到特征点的位置
    color_nose, color_left_eyebrow, color_right_eyebrow, color_jaw, L = identify_color_feature.identify_color_feature(color)
    
    # 将图片的宽放大到原来的1.33倍, 宽高比变为2.35
    color = cv2.resize(color, (0, 0), fx = 1.33, fy = 1)
    color_head = [[]]
    color_face_image, color_head= identify_color_face.identify_color_face(color_path)
    print("-----得到彩色图的人脸坐标和各个特征点的坐标-----")
    # print("color_head(top_row, left_col, right_col): ", color_head)
    
    # 拉伸图片的长后,在原图上的特征点的纵坐标y也要放缩(横坐标x不用)
    for i in range(3):
        color_nose[i][0] = int(color_nose[i][0] * 1.33)
        color_jaw[i][0] = int(color_jaw[i][0] * 1.33)
    for i in range(5):
        color_left_eyebrow[i][0] = int(color_left_eyebrow[i][0] * 1.33)
        color_right_eyebrow[i][0] = int(color_right_eyebrow[i][0] * 1.33)
    # print("color_nose(y, x): ", color_nose)
    # print("color_left_eyebrow(y, x): ", color_left_eyebrow)
    # print("color_right_eyebrow(y, x): ", color_right_eyebrow)
    # print("color_jaw(y, x): ", color_jaw)
    
    # 在彩色图上绘制特征点
    # 绘制鼻子
    for i in range(len(color_nose)):
        cv2.circle(color, color_nose[i], 3, (0, 255, 0), -1)
    # 绘制左眉毛
    for i in range(len(color_left_eyebrow)):
        cv2.circle(color, color_left_eyebrow[i], 3, (0, 255, 0), -1)
    # 绘制右眉毛
    for i in range(len(color_right_eyebrow)):
        cv2.circle(color, color_right_eyebrow[i], 3, (0, 255, 0), -1)
    # 绘制下巴
    for i in range(len(color_jaw)):
        cv2.circle(color, color_jaw[i], 3, (0, 255, 0), -1)
    
    """显示图片"""
    print("-----show image-----")
    # 创建窗口
    cv2.namedWindow('Resizable Window', cv2.WINDOW_NORMAL)
    # 显示图片
    cv2.imshow('Resizable Window', color)
    # 调整窗口大小
    cv2.resizeWindow('Resizable Window', 800, 600)  # 设置窗口大小为800x600
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    # sys.exit()
    
    """得到灰度图的人脸坐标"""
    print("-----得到灰度图的人脸坐标-----")
    gray_head = [[]]
    gray_face_image, gray_head = identify_gray_face.identify_gray_face(gray_path)
    # print("gray_head(top_row, left_col, right_col): ", gray_head)
    
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
    
    """确定灰度图中各个特征点的坐标"""
    # 计算彩色图和灰度图中头的宽度
    color_head_width = color_head[2] - color_head[1]
    gray_head_width = gray_head[2] - gray_head[1]
    # 为了让彩色图中头的宽度与灰度图的相等，将彩色图等比例缩小
    ratio = gray_head_width / color_head_width
    color_face_image = cv2.resize(color_face_image, (0, 0), fx = ratio, fy = ratio)
    
    # 缩小彩色图后,原彩色图上的特征点坐标(y, x)和人脸坐标都要按相同的比例变换
    for i in range(3):
        for j in range(2):
         color_nose[i][j] = int(color_nose[i][j] * ratio)
         color_jaw[i][j] = int(color_jaw[i][j] * ratio)
    for i in range(5):
        for j in range(2):
            color_left_eyebrow[i][j] = int(color_left_eyebrow[i][j] * ratio)
            color_right_eyebrow[i][j] = int(color_right_eyebrow[i][j] * ratio)
    for i in range(3):
        color_head[i] = int(color_head[i] * ratio)
    print("-----确定灰度图中各个特征点的坐标-----")
    # print("color_nose(y, x): ", color_nose)
    # print("color_left_eyebrow(y, x): ", color_left_eyebrow)
    # print("color_right_eyebrow(y, x): ", color_right_eyebrow)
    # print("color_jaw(y, x): ", color_jaw)
    # print("color_head(top_row, left_col, right_col): ", color_head)
    
    # """显示图片"""
    # print("-----show image-----")
    # # 创建窗口
    # cv2.namedWindow('Resizable Window', cv2.WINDOW_NORMAL)
    # # 显示图片
    # cv2.imshow('Resizable Window', color)
    # # 调整窗口大小
    # cv2.resizeWindow('Resizable Window', 800, 600)  # 设置窗口大小为800x600
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()
    # sys.exit()
    
    # 计算出彩色图中特征点坐标(y, x)距离头部左边的距离和上边的距离
    nose_dis = np.zeros((3, 2), dtype=int)
    left_eyebrow_dis = np.zeros((5, 2), dtype=int)
    right_eyebrow_dis = np.zeros((5, 2), dtype=int)
    jaw_dis = np.zeros((3, 2), dtype=int)
    for i in range(3):
        nose_dis[i][0] = color_nose[i][0] - color_head[1]
        nose_dis[i][1] = color_nose[i][1] - color_head[0]
        jaw_dis[i][0] = color_jaw[i][0] - color_head[1]
        jaw_dis[i][1] = color_jaw[i][1] - color_head[0]
    for i in range(5):
        left_eyebrow_dis[i][0] = color_left_eyebrow[i][0] - color_head[1]
        left_eyebrow_dis[i][1] = color_left_eyebrow[i][1] - color_head[0]
        right_eyebrow_dis[i][0] = color_right_eyebrow[i][0] - color_head[1]
        right_eyebrow_dis[i][1] = color_right_eyebrow[i][1] - color_head[0]
    # print("nose_dis(left, top): ", nose_dis)
    # print("left_eyebrow_dis(left, top): ", left_eyebrow_dis)
    # print("right_eyebrow_dis(left, top): ", right_eyebrow_dis)
    # print("jaw_dis(left, top): ", jaw_dis)

    # 人脸的中线
    mid_col = (gray_head[2] + gray_head[1]) // 2
    # 确定灰度图中特征点的坐标(y, x)
    gray_nose = np.zeros((3, 2), dtype=int)
    gray_left_eyebrow = np.zeros((5, 2), dtype=int)
    gray_right_eyebrow = np.zeros((5, 2), dtype=int)
    gray_jaw = np.zeros((3, 2), dtype=int)
    for i in range(3):
        gray_nose[i][0] = gray_head[1] + nose_dis[i][0]
        gray_nose[i][1] = gray_head[0] + nose_dis[i][1]
        gray_jaw[i][0] = gray_head[1] + jaw_dis[i][0]
        gray_jaw[i][1] = gray_head[0] + jaw_dis[i][1]
    for i in range(5):
        gray_left_eyebrow[i][0] = gray_head[1] + left_eyebrow_dis[i][0]
        gray_left_eyebrow[i][1] = gray_head[0] + left_eyebrow_dis[i][1]
        gray_right_eyebrow[i][0] = gray_head[1] + right_eyebrow_dis[i][0]
        gray_right_eyebrow[i][1] = gray_head[0] + right_eyebrow_dis[i][1]
    
    # 由于鼻子一般在人脸中间，所以这里计算鼻尖位置与人脸中线的偏差，调整各个特征点的位置
    move = mid_col - gray_nose[0][0]
    # 将鼻尖的位置调整到人脸中间，其他特征点按同样的距离调整
    for i in range(3):
        gray_nose[i][0] = gray_nose[i][0] + move
        gray_jaw[i][0] = gray_jaw[i][0] + move
    for i in range(5):
        gray_left_eyebrow[i][0] = gray_left_eyebrow[i][0] + move
        gray_right_eyebrow[i][0] = gray_right_eyebrow[i][0] + move
    # print("gray_nose(y, x): ", gray_nose)
    # print("gray_left_eyebrow(y, x): ", gray_left_eyebrow)
    # print("gray_right_eyebrow(y, x): ", gray_right_eyebrow)
    # print("gray_jaw(y, x): ", gray_jaw)
    
    # 在原来的灰度图上绘制各个特征点
    # 绘制鼻子
    for i in range(len(gray_nose)):
        cv2.circle(origin_gray, gray_nose[i], 3, (0, 255, 0), -1)
    # 绘制左眉毛
    for i in range(len(gray_left_eyebrow)):
        cv2.circle(origin_gray, gray_left_eyebrow[i], 3, (0, 255, 0), -1)
    # 绘制右眉毛
    for i in range(len(gray_right_eyebrow)):
        cv2.circle(origin_gray, gray_right_eyebrow[i], 3, (0, 255, 0), -1)
    # 绘制下巴
    for i in range(len(gray_jaw)):
        cv2.circle(origin_gray, gray_jaw[i], 3, (0, 255, 0), -1)
    
    # 得到6个ROI区域，其中L为第29个特征点和第30个特征点之间的距离，每个人的长度不一样
    # 鼻子区域，高度为1.3L，宽度为3L
    nose_top_row = gray_nose[1][1] - int(L * 0.65)
    nose_bottom_row = gray_nose[1][1] + int(L * 0.65)
    # nose_top_row = gray_nose[0][1]
    # nose_bottom_row = gray_nose[0][1] + int(L * 1.5)
    nose_left_col = gray_nose[0][0] - int(L * 1.5)
    nose_right_col = gray_nose[0][0] + int(L * 1.5)
    nose_region = [nose_top_row, nose_bottom_row, nose_left_col, nose_right_col]
    cv2.rectangle(origin_gray, (nose_region[2], nose_region[0]), (nose_region[3], nose_region[1]), (0, 255, 0), 2)
    # 额头左侧区域，高度为0.7L，宽度为2.5L
    left_forehead_top_row = gray_left_eyebrow[1][1] - int(L * 0.7)
    left_forehead_bottom_row = gray_left_eyebrow[1][1]
    left_forehead_left_col = gray_left_eyebrow[1][0]
    left_forehead_right_col = gray_left_eyebrow[1][0] + int(L * 2.5)
    left_forehead_region = [left_forehead_top_row, left_forehead_bottom_row, left_forehead_left_col, left_forehead_right_col]
    cv2.rectangle(origin_gray, (left_forehead_region[2], left_forehead_region[0]), (left_forehead_region[3], left_forehead_region[1]), (0, 255, 0), 2)
    # 额头右侧区域，高度为L，宽度为2.5L
    right_forehead_top_row = gray_right_eyebrow[3][1] - int(L * 0.7)
    right_forehead_bottom_row = gray_right_eyebrow[3][1]
    right_forehead_left_col = gray_right_eyebrow[3][0] - int(L * 2.5)
    right_forehead_right_col = gray_right_eyebrow[3][0]
    right_forehead_region = [right_forehead_top_row, right_forehead_bottom_row, right_forehead_left_col, right_forehead_right_col]
    cv2.rectangle(origin_gray, (right_forehead_region[2], right_forehead_region[0]), (right_forehead_region[3], right_forehead_region[1]), (0, 255, 0), 2)
    # 脸颊左侧区域，高度为0.7L，宽度为2L
    left_cheek_top_row = gray_nose[1][1] + int(L * 0.7) # 距离左鼻孔0.7L
    left_cheek_bottom_row = left_cheek_top_row + int(L * 0.7)
    left_cheek_right_col = gray_nose[1][0] - int(L * 0.5)
    left_cheek_left_col = left_cheek_right_col - int(L * 2)
    left_cheek_region = [left_cheek_top_row, left_cheek_bottom_row, left_cheek_left_col, left_cheek_right_col]
    cv2.rectangle(origin_gray, (left_cheek_region[2], left_cheek_region[0]), (left_cheek_region[3], left_cheek_region[1]), (0, 255, 0), 2)
    # 脸颊右侧区域，高度为0.7L，宽度为2L
    right_cheek_top_row = gray_nose[2][1] + int(L * 0.7) # 距离右鼻孔0.7L
    right_cheek_bottom_row = right_cheek_top_row + int(L * 0.7)
    right_cheek_left_col = gray_nose[2][0] + int(L * 0.5)
    right_cheek_right_col = right_cheek_left_col + int(L * 2)
    right_cheek_region = [right_cheek_top_row, right_cheek_bottom_row, right_cheek_left_col, right_cheek_right_col]
    cv2.rectangle(origin_gray, (right_cheek_region[2], right_cheek_region[0]), (right_cheek_region[3], right_cheek_region[1]), (0, 255, 0), 2)
    # 下巴区域，高度为0.7L，宽度为2.5L
    jaw_top_row = gray_jaw[0][1] - int(L * 0.7)
    jaw_bottom_row = gray_jaw[0][1]
    jaw_left_col = gray_jaw[1][0] - int(L * 1.25)
    jaw_right_col = gray_jaw[1][0] + int(L * 1.25)
    jaw_region = [jaw_top_row, jaw_bottom_row, jaw_left_col, jaw_right_col]
    cv2.rectangle(origin_gray, (jaw_region[2], jaw_region[0]), (jaw_region[3], jaw_region[1]), (0, 255, 0), 2)
    
    print("nose_region", nose_region)
    print("left_forehead_region", left_forehead_region)
    print("right_forehead_region", right_forehead_region)
    print("left_cheek_region", left_cheek_region)
    print("right_cheek_region", right_cheek_region)
    print("jaw_region", jaw_region)
    
    """显示图片"""
    print("-----show image-----")
    # 创建窗口
    cv2.namedWindow('Resizable Window', cv2.WINDOW_NORMAL)
    # 显示图片
    cv2.imshow('Resizable Window', origin_gray)
    # 调整窗口大小
    cv2.resizeWindow('Resizable Window', 800, 600)  # 设置窗口大小为800x600
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    
    return nose_region, left_forehead_region, right_forehead_region, left_cheek_region, right_cheek_region, jaw_region
