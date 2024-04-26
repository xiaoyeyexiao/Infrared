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

def Histogram_Equalization(grey):
    # 计算灰度图像的直方图
    hist, bins = np.histogram(grey.ravel(), 256, [0, 256])
    cdf = hist.cumsum()
    cdf_normalized = (cdf - cdf.min()) * 255 / (cdf.max() - cdf.min())
    signal = cdf_normalized[grey].astype('uint8')
    return signal

def Butterworth(frame, l=0.5, h=2.5, fps=25):
    '''
        Input:
            orinal_signal: 二维灰度图 e.g. (1080, 1920)
        Output:
            signal : 一个灰度图进行滤波后，二维. e.g. (1080, 1920)
    '''
    # 巴特沃兹带通滤波器参数
    lowcut = l  # 低频截止频率
    highcut = h  # 高频截止频率
    nyq = fps/2  # 信号的最高频率，即采样率的一半
    b, a = butter(N=5, Wn=[lowcut / nyq, highcut / nyq], btype='bandpass')
    
    # 逐帧进行巴特沃兹带通滤波器之后计算均值
    signal = filtfilt(b, a, frame)
    
    return signal


class EMDDecomposition:
    def __init__(self, emd_instance=None):
        self.emd = emd_instance if emd_instance is not None else EMD()

    def get_imfs_res(self, signal, t):
        self.emd.emd(signal, t)
        imfs, res = self.emd.get_imfs_and_residue()
        return imfs, res
    def decompose(self, signal, t):
        """
        对信号进行EMD分解。
        :param signal: 输入信号
        :param t: 时间向量
        :return: 分解得到的固有模态函数列表
        """
        return self.emd.emd(signal, t)

    def get_imfs(self, signal, t):
        """
        获取信号的固有模态函数。
        :param signal: 输入信号
        :param t: 时间向量
        :return: 固有模态函数列表
        """
        return self.decompose(signal, t)

    def reconstruct(self, imfs):
        """
        根据固有模态函数重构信号。
        :param imfs: 固有模态函数列表
        :return: 重构信号
        """
        return np.sum(imfs, axis=0)

    def get_instantaneous_frequencies(self, imfs, t):
        """
        计算每个固有模态函数的瞬时频率。
        :param imfs: 固有模态函数列表
        :param t: 时间向量
        :return: 瞬时频率列表
        """
        instantaneous_frequencies = []
        for imf in imfs:
            instantaneous_frequency = self.emd.get_instantaneous_frequency(imf, t)
            instantaneous_frequencies.append(instantaneous_frequency)
        return instantaneous_frequencies
    
    def visionlization(self, imfs, res, t):
        vis = Visualisation()
        vis.plot_imfs(imfs=imfs, residue=res, t=t, include_residue=True)
        vis.plot_instant_freq(t, imfs=imfs)
        vis.show()
        
        
        