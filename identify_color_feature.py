import cv2
import dlib
import sys

def identify_color_feature(image):
    # 加载Dlib的人脸检测器和面部关键点检测器
    detector = dlib.get_frontal_face_detector()
    predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # 使用人脸检测器检测图像中的人脸
    faces = detector(gray)

    # # 对于每张检测到的人脸，检测并绘制关键点
    # for face in faces:
    #     # 使用面部关键点检测器检测关键点
    #     landmarks = predictor(gray, face)
        
    #     # 绘制关键点
    #     for n in range(0, 68):
    #         x = landmarks.part(n).x
    #         y = landmarks.part(n).y
    #         cv2.circle(image, (x, y), 2, (0, 255, 0), -1)
    
    color_nose = []
    color_left_eyebrow = []
    color_right_eyebrow = []
    color_jaw = []
    for face in faces:
        # 检测人脸的面部特征点
        landmarks = predictor(gray, face)
        
        # 取第29个特征点到第30个特征点之间的距离作为基础距离L
        L = landmarks.part(30).y - landmarks.part(29).y
        
        # 以下的坐标均为(y, x)
        # 获取鼻子的坐标
        nose_tip = [landmarks.part(30).x, landmarks.part(30).y]
        nose_left = [landmarks.part(31).x, landmarks.part(31).y]
        nose_right = [landmarks.part(35).x, landmarks.part(35).y]
        color_nose = [nose_tip, nose_left, nose_right]
        
        # 获取左眉毛的坐标
        for i in range(17, 22):
            eyebrow = [landmarks.part(i).x, landmarks.part(i).y]
            color_left_eyebrow.append(eyebrow)

        # 获取右眉毛的坐标
        for i in range(22, 27):
            eyebrow = [landmarks.part(i).x, landmarks.part(i).y]
            color_right_eyebrow.append(eyebrow)
            
        # 获取下巴的坐标
        for i in range(7, 10):
            jaw = [landmarks.part(i).x, landmarks.part(i).y]
            color_jaw.append(jaw)
        
    # # 绘制鼻子
    # for i in range(len(color_nose)):
    #     cv2.circle(image, color_nose[i], 3, (0, 255, 0), -1)
        
    # # 绘制左眉毛
    # for i in range(len(color_left_eyebrow)):
    #     cv2.circle(image, color_left_eyebrow[i], 3, (0, 255, 0), -1)
    
    # # 绘制右眉毛
    # for i in range(len(color_right_eyebrow)):
    #     cv2.circle(image, color_right_eyebrow[i], 3, (0, 255, 0), -1)
        
    # # 绘制下巴
    # for i in range(len(color_jaw)):
    #     cv2.circle(image, color_jaw[i], 3, (0, 255, 0), -1)
    
    return color_nose, color_left_eyebrow, color_right_eyebrow, color_jaw, L
    
#     """显示图片"""
#     print("-----show image-----")
#     # 创建窗口
#     cv2.namedWindow('Resizable Window', cv2.WINDOW_NORMAL)
#     # 显示图片
#     cv2.imshow('Resizable Window', image)
#     # 调整窗口大小
#     cv2.resizeWindow('Resizable Window', 800, 600)  # 设置窗口大小为800x600
#     cv2.waitKey(0)
#     cv2.destroyAllWindows()

# color_path = '/home/mengqingyi/code/yechentao/Infrared_vedio_image/image/color8/frame_1.jpg'
# color = cv2.imread(color_path)
# identify_color_nose(color)