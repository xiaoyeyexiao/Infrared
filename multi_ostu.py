import cv2
import numpy as np
import matplotlib.pyplot as plt
import sys

def multi_ostu(image, classes):
    # 如果输入的图像类型不是uint8，则会引发AssertionError
    assert image.dtype == np.uint8, "Input image should be of type uint8"
    
    maxSum = 0
    # classes个类别，只要classes - 1个阈值就可以分割
    thresholds = [0] * (classes - 1)

    # 绘制直方图
    # [image]: 指定计算直方图的图像
    # [0]: 指定对图像的第一个通道(即灰度图中唯一的通道)进行直方图计算
    # None: 不适用掩码
    # [256]: 指定直方图的bin数量为256，表示256个小区间
    # [0, 256]: 指定灰度值的范围
    # .ravel(): 将计算得到的直方图数组展平为一维数组，方便后续使用
    histogram = cv2.calcHist([image], [0], None, [256], [0, 256]).ravel()

    # 累积和表 P: 存储累积像素点数量，例如P[100]表示 灰度值<=100 的像素点数量
    # 累积和表 S: 存储累积灰度值和，例如S[100]表示 灰度值<=100 的所有像素点的灰度值和
    P = np.zeros(len(histogram) + 1, dtype=np.int32)
    S = np.zeros(len(histogram) + 1, dtype=np.int32)
    # .cumsum(): 计算累积和
    P[1:] = np.cumsum(histogram)
    # np.arange(len(histogram))创建一个从0到255的数组，然后两个数组对应位置元素相乘(即求对应灰度级的灰度值和)
    S[1:] = np.cumsum(np.arange(len(histogram)) * histogram)

    # H表用于存储类间方差的值，其中每个元素对应于两个灰度级之间的类的类间方差
    H = np.zeros((len(histogram), len(histogram)), dtype=np.float32)
    # 将灰度级 u 和 v 之间的所有像素点视为一类，计算该类的类间方差，将结果存储到H表中
    for u in range(len(histogram)):
        for v in range(u + 1, len(histogram)):
            # 这里类间方差的计算公式为论文 A Fast Algorithm for Multilevel Thresholding 中的公式(29)
            # np.finfo(float).eps: 浮点数类型的最小精度，防止除零导致的运行时错误
            H[u, v] = ((S[v] - S[u]) ** 2) / (P[v] - P[u] + np.finfo(float).eps)

    # 用于存储每个阈值的索引，第一个索引初始化为0，最后一个索引初始化为255
    index = [0] * (classes + 1)
    index[0] = 0
    index[len(index) - 1] = len(histogram) - 1

    def lhOTSU_for_loop(maxSum, lhthres, H, u, vmax, level):
        # nonlocal：可以在内部函数中修改外部函数的局部变量
        nonlocal thresholds, index

        for i in range(u, vmax):
            # 在每次循环中，将当前级别的阈值索引存储到index数组中
            index[level] = i

            # 如果当前级别 level 大于或等于总类别数 classes，则说明已经找到了一个完整的阈值组合
            if level + 1 >= classes:
                # 计算类间方差
                sum_ = 0
                for c in range(classes):
                    u = index[c]
                    v = index[c + 1]
                    sum_ += H[u, v]

                # 若类间方差超过了之前的最大值，则更新最大值，并将当前阈值组合存储到 lhthres 中
                if maxSum[0] < sum_:
                    # Return calculated threshold.
                    lhthres[:] = index[1:classes]
                    maxSum[0] = sum_
            # 如果当前级别小于总类别数，说明还需要继续递归
            else:
                lhOTSU_for_loop(maxSum, lhthres, H, i + 1, vmax + 1, level + 1)

    lhOTSU_for_loop([maxSum], thresholds, H, 1, len(histogram) - classes + 1, 1)

    return thresholds

# if __name__ == "__main__":
#     image = cv2.imread("/home/yechentao/Programs/Infrared/image/gray4/frame_1.jpg")
#     gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
#     # cv2.imshow("src", gray_image)

#     classes = 4
#     thresholds = multi_ostu(gray_image, classes)
#     print("Thresholds:", thresholds)
    
#     # 将目标转为白色，背景转为黑色
#     mask = gray_image > thresholds[len(thresholds) - 1]
#     gray_image[mask] = 255
#     gray_image[~mask] = 0
    
#     cv2.imshow("Infrared",gray_image)
#     cv2.waitKey(0)
#     cv2.destroyAllWindows()
