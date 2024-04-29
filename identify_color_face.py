import torchvision.transforms as transforms
from torchvision import models
from PIL import Image
import numpy as np
import cv2
import torch

# 图像预处理
def preprocess_image(color_path):
    image = Image.open(color_path).convert("RGB")
    new_width = int(image.width * 1.33)  # 计算调整后的宽度
    # 调整图像大小
    image = image.resize((new_width, image.height))
    preprocess = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    return preprocess(image).unsqueeze(0)

def identify_color_face(color_path):
    # 加载预训练的 DeepLabv3 模型
    model = models.segmentation.deeplabv3_resnet50(pretrained=True)
    model.eval()
    # 加载图像
    input_image = preprocess_image(color_path)

    # 获取原始图像尺寸
    original_image = Image.open(color_path)
    new_width = int(original_image.width * 1.33)  # 计算调整后的宽度
    # 调整图像大小
    original_image = original_image.resize((new_width, original_image.height))
    original_width, original_height = original_image.size

    # 推理
    with torch.no_grad():
        output = model(input_image)['out'][0]
    output_predictions = output.argmax(0)

    # 将预测结果调整为与原始图像相同的大小
    output_predictions_resized = transforms.Resize((original_height, original_width))(output_predictions.unsqueeze(0))
    output_predictions_resized = output_predictions_resized.squeeze(0)

    # 定义阈值，将人像区域与其他区域分隔开来
    threshold = 0.5
    face_mask = (output_predictions_resized == 15)  # 在 COCO 数据集上，15 是人的类别标签
    face_mask = face_mask.float() * 255  # 将布尔值转换为 0 或 255

    # 将二值图像转换为 OpenCV 格式
    face_mask_cv = face_mask.byte().cpu().numpy()

    # 寻找轮廓
    contours, _ = cv2.findContours(face_mask_cv, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # 将轮廓绘制在原始图像上
    output_image = cv2.cvtColor(np.array(original_image), cv2.COLOR_RGB2BGR)
    cv2.drawContours(output_image, contours, -1, (0, 255, 0), 2)

    # 头的上边
    head_top_row = float('inf')
    # 头的左边和右边
    head_left_col = float('inf')
    head_right_col = float('-inf')

    # 遍历每个轮廓
    for contour in contours:
        for point in contour:
            x, y = point[0]
            if y < head_top_row:
                head_top_row = y

    # 截取头部位置
    mid_row = head_top_row + int((output_image.shape[0] -head_top_row) * 0.8)
    output_image = output_image[0:mid_row, :, :] 
    
    # 遍历每个轮廓
    for contour in contours:
        for point in contour:
            x, y = point[0]
            if x < head_left_col and y < mid_row:
                head_left_col = x
            if x > head_right_col and y < mid_row:
                head_right_col = x

    return output_image, [head_top_row, head_left_col, head_right_col]

    # """显示图片"""
    # print("-----show image-----")
    # # 创建窗口
    # cv2.namedWindow('Resizable Window', cv2.WINDOW_NORMAL)
    # # 显示图片
    # cv2.imshow('Resizable Window', output_image)
    # # 调整窗口大小
    # cv2.resizeWindow('Resizable Window', 800, 600)  # 设置窗口大小为800x600
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()
    
# identify_color_body('/home/yechentao/Programs/Infrared/image/color4/frame_1.jpg')