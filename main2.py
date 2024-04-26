import cv2
import numpy as np
from matplotlib import pyplot as plt
import os
import sys
import matplotlib.pyplot as plt
from PyEMD import EMD, Visualisation
from scipy.fft import fft, fftfreq  # 傅里叶变化
from scipy.signal import fftconvolve
from scipy.signal import argrelextrema
from scipy.signal import find_peaks, butter, lfilter, filtfilt
from scipy.interpolate import make_interp_spline
import argparse
from filter import *
from rom import *

parser = argparse.ArgumentParser(description='')
parser.add_argument('--fps', type=int, default=25, help='帧率')
parser.add_argument('--gray_mean', type=bool, default=True, help='直接使用每一帧的均值作为信号')
parser.add_argument('--he', type=bool, default=False, help='Histogram Equalization 直方图均衡化')
parser.add_argument('--butterworth', type=bool, default=True, help='巴特沃斯滤波器')
# parser.add_argument('--filter', type=str, default='butterworth',  help='巴特沃斯滤波器')
parser.add_argument('--emd', type=bool, default=False, help='经验模态分解EMD')
parser.add_argument('--frequency_method', type=str, default='fft',  help='频域转换方法')
parser.add_argument('--out_sigal', type=str, default='heart', help='心跳或者呼吸')
parser.add_argument('--rom', type=str, default='rom', help='rom提取')

def moving_average(interval, windowsize):
 
    window = np.ones(int(windowsize)) / float(windowsize)
    re = np.convolve(interval, window, 'same')
    return re

def plot_chart(x_values, y_values, title, x_label, y_label, grid=True):
    plt.figure()
    
    # 平滑曲线
    y_av = moving_average(y_values, 25)
    plt.plot(x_values, y_av, 'b')
    plt.xlabel(x_label)
    plt.ylabel(y_label)

    # plt.grid()网格线设置
    plt.xlim(left=0)
    if grid:
        plt.grid(True)
    plt.title(title)
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.show()
        
    # plt.plot(x_values, y_values)
    # plt.title(title)
    # plt.xlabel(x_label)
    # plt.ylabel(y_label)
    # plt.xlim(left=0)
    
    # if grid:
    #     plt.grid(True)
    # plt.show()
    

def main(args, signal_list):
    
    # 初始化参数
    fps = args.fps
    t = np.arange(len(signal_list))/fps # time/s
    
    # t = len(signal_list) # frame times
    
    # plot_chart(t, [np.mean(frame) for frame in signal_list], "original Signal", "time/s", "", grid=True)
    last_signal_list = signal_list
    
    if args.he is True:
        print('************begin Histogram Equalization************')
        original_signal_list = last_signal_list
        
        last_signal_list = []
        i = 0
        for frame in original_signal_list:
            if len(frame.shape) == 2:
                equalized_frame = Histogram_Equalization(frame)
                last_signal_list.append(equalized_frame)
                cv2.imwrite(os.path.join('equalized_frame', f"he_frame_{i}.png"),equalized_frame)
                i = i+1
            else:
                print('图片不是灰度图，无法进行直方图均衡化')
                break
        plot_chart(t, [np.mean(frame) for frame in last_signal_list], "he Signal", "time/s", "", grid=True)
    
    # rom获取signal
    # if args.rom == 'rom':
    #     print('************begin rom************')
    #     last_signal_list = rom(last_signal_list.copy())
    #     print('rom signal list:',len(last_signal_list))
    
        # plot_chart(t, [np.mean(frame) for frame in last_signal_list], "rom Signal", "time/s", "", grid=True)
        
        # if args.frequency_method == 'fft':
        #     print('************rom fft************')
        #     sig = [np.mean(frame) for frame in last_signal_list]
        #     signal_array = np.fft.fft(sig)
        #     freq = np.fft.fftfreq(len(signal_array))
        #     heart_rate_frequency = np.where((freq > 0.5) & (freq < 2.5))
        
        #     # 计算频率
        #     # 正常：在12-20次/分，心率一般在60-100次/分
        #     heart_rate_per_minute = np.mean(freq[heart_rate_frequency]) * 60
        #     print('频率(次/分)：', heart_rate_per_minute)
    
    
    if args.butterworth is True:
        i=0
        print('************begin butterworth filter************')
    # if args.filter == 'butterworth':
        original_signal_list = last_signal_list
        last_signal_list = []
        last_signal_list_heart = []
        last_signal_list_bearth = []
        for frame in original_signal_list:
            butterworth_frame = Butterworth(frame,l=0.2, h=1.67, fps=25)
            butterworth_frame_heart = Butterworth(frame,l=1, h=1.67, fps=25) # heart
            butterworth_frame_breath = Butterworth(frame,l=0.2, h=0.33, fps=25) 
            # butterworth_frame = Butterworth(frame,l=0.2, h=0.33, fps=25) # breath
            # butterworth_frame = Butterworth(frame,l=0.2, h=1.67, fps=25) # both
            last_signal_list.append(abs(butterworth_frame))
            last_signal_list_heart.append(abs(butterworth_frame_heart))
            last_signal_list_bearth.append(abs(butterworth_frame_breath))
            
            cv2.imwrite(os.path.join('butter_frame', f"he_frame_{i}.png"),butterworth_frame)
            i= i+1
            
        plot_chart(t, [np.mean(frame) for frame in last_signal_list_heart], "heart-butterworth Signal", "time/s", "", grid=True)
        plot_chart(t, [np.mean(frame) for frame in last_signal_list_bearth], "breath-butterworth Signal", "time/s", "", grid=True)
            
    if args.emd is True:
        print('************emd************')
        original_signal_list = last_signal_list
        last_signal_list = []
                
        # 逐帧均值计算
        average_signal_list = []
        for frame in original_signal_list:
            average_signal_list.append(np.mean(frame))
            
        emd = EMDDecomposition()
        imfs, res = emd.get_imfs_res(np.array(average_signal_list),t)
        # 信号绘制
        emd.visionlization(imfs, res, t)
        
        # END 信号重构
        print(imfs.shape[0])
        # IMFs合并
        # selected_imfs = imfs[imfs.shape[0]-4:imfs.shape[0]-1, :]
        selected_imfs = imfs[2:, :] 
        last_signal_list = np.sum(selected_imfs, axis=0)
        
        # # 重构信号绘制
        plot_chart(t, last_signal_list, "emd Rreconstructed Signal", "time/s", "", grid=True)
        
    if args.frequency_method == 'fft':
        print('************fft************')
        if args.gray_mean is True:
            original_signal_list = last_signal_list
            last_signal_list = []
            for frame in original_signal_list:
                last_signal_list.append(np.mean(frame))
                #  last_signal_list.append(np.max(frame) - np.min(frame))
            last_signal_list = np.array(last_signal_list)
            
            # 傅里叶变换
            fft_signal = np.fft.fft(last_signal_list)
            fft_signal = np.abs(fft_signal)
            # 计算频率
            # freqs = np.fft.fftfreq(len(fft_signal)) * fps
            freqs = np.fft.fftfreq(len(fft_signal),1/fps) 
            
            # 呼吸频率
            min_freq = 0.2
            max_freq = 0.33 
            #限制频率范围
            idx = np.logical_and(freqs >= min_freq, freqs <= max_freq)
            # 在限制的频率范围内找到最大的频谱值
            peak_freq = freqs[idx][np.argmax(fft_signal[idx])]
            # 频率从赫兹转换为每分钟的次数
            rate = peak_freq * 60
            print('呼吸频率(次/分)：', rate)
            # # 频域转换成时域
            # time_signal = np.fft.ifft(fft_signal)
            # plot_chart(t,time_signal, "Frequency Spectrum of Breathing Signal", "time", "Frequency (Hz)", grid=True)
            # plot_chart(freqs,fft_signal, "Frequency Spectrum of Breathing Signal", "Frequency (Hz)", "Magnitude", grid=True)
            
            
            # 心跳频率
            # 信号频率范围
            min_freq = 1
            max_freq = 1.67
            #限制频率范围
            idx = np.logical_and(freqs >= min_freq, freqs <= max_freq)
            # 在限制的频率范围内找到最大的频谱值
            peak_freq = freqs[idx][np.argmax(fft_signal[idx])]
            # 频率从赫兹转换为每分钟的次数
            rate = peak_freq * 60
            print('心跳频率(次/分)：', rate)
        
        
        # xf, max_frequency_index = fft_change(last_signal,fps)

        # # 计算心跳
        # 1 次/分钟 = 1/60 次/秒 = 0.0167 Hz
        # # 呼吸在12-20次/分: 0.2 - 0.33 Hz ，心率一般在60-100次/分 1 - 1.67 Hz
        # heart_rate_per_minute_emd = xf[max_frequency_index] * 60
        # print('频率(次/分)：', heart_rate_per_minute_emd)
    

if __name__ == '__main__':
    args = parser.parse_args()
    
    signal_list = []
    
    # video_path = 'gray4.mp4' # frame_size:456,250
    # cap = cv2.VideoCapture(video_path) 
    # # 逐帧处理视频
    # i = 0
    # while True:
    #     # 读取一帧
    #     ret, frame = cap.read() # 剪映裁剪后有三个通道 (1080, 1920, 3)  原视频每一帧的维度 (720, 1280, 3)
    #     if not ret:
    #         break  # 视频结束
        
    #     # 图像预处理 转换为灰度
    #     gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
    #     # cv2.imwrite(os.path.join('roi_sequence', f"frame_{i}.png"),gray_frame)
    #     i = i+1
    #     signal_list.append(gray_frame)
        
    
    file_pathname = '/home/mengqingyi/code/yechentao/Infrared_vedio_image/region/nose/gray8'
    for filename in os.listdir(file_pathname):
        img = cv2.imread(file_pathname+'/'+filename)
        gray_frame = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) # 灰度图
        # cv2.imwrite(os.path.join('roi_sequence', filename),gray_frame)
        signal_list.append(gray_frame)
        # signal_list.append(img)
        
    main(args, signal_list)