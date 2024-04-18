import cv2
import dlib

def identify_color_nose(image):
    # 加载Dlib的人脸检测器和面部关键点检测器
    detector = dlib.get_frontal_face_detector()
    predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # 使用人脸检测器检测图像中的人脸
    faces = detector(gray)

    color_nose = []
    for face in faces:
        # 检测人脸的面部特征点
        landmarks = predictor(gray, face)
        
        # 根据面部特征点的索引获取鼻子的坐标
        nose_tip = (landmarks.part(30).x, landmarks.part(30).y)
        nose_left = (landmarks.part(31).x, landmarks.part(31).y)
        nose_right = (landmarks.part(35).x, landmarks.part(35).y)
        
        # 调整鼻孔区域大小
        # 增大鼻孔区域
        nose_tip = [nose_tip[0], nose_tip[1] - 10]
        nose_left = [nose_left[0] - 5, nose_left[1]]
        nose_right = [nose_right[0] + 5, nose_right[1]]
        
        color_nose = [nose_tip, nose_left, nose_right]
        
    return color_nose

# color = cv2.imread('/home/yechentao/Programs/Infrared/image/color2/frame_1.jpg')

# # """显示图片"""
# # print("-----show image-----")
# # # 创建窗口
# # cv2.namedWindow('Resizable Window', cv2.WINDOW_NORMAL)
# # # 显示图片
# # cv2.imshow('Resizable Window', color)
# # # 调整窗口大小
# # cv2.resizeWindow('Resizable Window', 800, 600)  # 设置窗口大小为800x600
# # cv2.waitKey(0)
# # cv2.destroyAllWindows()
# color_nose = identify_color_nose(color)
# # 计算宽度和高度的缩放比例
# # scale_l = src_w / new_w
# # scale_w = src_h / new_h
# ratio = 1.33

# # 将图片长宽比变为2.35
# color = cv2.resize(color, (0, 0), fx = 1.33, fy = 1)
# # 使用缩放比例来计算原图上的坐标
# for i in range(3):
#     color_nose[i][0] = int(color_nose[i][0] * ratio)
# cv2.circle(color, color_nose[0], 3, (0, 255, 0), -1)
# cv2.circle(color, color_nose[1], 3, (0, 255, 0), -1)
# cv2.circle(color, color_nose[2], 3, (0, 255, 0), -1)
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