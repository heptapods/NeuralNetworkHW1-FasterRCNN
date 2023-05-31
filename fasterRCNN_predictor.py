import numpy as np
import torch
from PIL import Image, ImageDraw, ImageFont
from torchvision.transforms import transforms as T
from torchvision.models.detection import fasterrcnn_resnet50_fpn
from utils import transform, object_list
import math
from utils import calculate_intersection_ratio

model = fasterrcnn_resnet50_fpn(num_classes=21,trainable_backbone_layer=5)
model_path = 'results/my_fasterRCNN.pth'  # 模型文件的路径
model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))
model.eval()

image_dir = 'pictures/'
image_name = 'car.jpg'  # 输入图像的路径
image = Image.open(image_dir + image_name)
if image.mode == 'RGBA' or image.mode == 'LA':
    # 将图像转换为RGB模式
    image = image.convert('RGB')
# 定义数据预处理
width, height = image.size
image_tensor = transform(image)

if __name__ == '__main__':
    output = model([image_tensor])
    print(output)
    # proposals = output[0]['proposals'].bbox.tolist()
    # print(proposals)
    # 获取预测框的坐标信息
    boxes = output[0]['boxes'].tolist()
    labels = output[0]['labels'].tolist()
    scores = output[0]['scores'].tolist()
    image_draw = ImageDraw.Draw(image)
    thresh = scores[0] * 0.9
    selected_boxes = []

    # 将图像和预测框的坐标转换为可绘制的格式
    for box, score, label in zip(boxes, scores, labels):
        if score >= thresh:
            select = False
            if len(selected_boxes) > 0:
                for selected_box, selected_label in selected_boxes:
                    if calculate_intersection_ratio(selected_box, box) > 0.5 and label == selected_label:
                        select = True
                if select:
                    continue
            x1, y1, x2, y2 = box
            x1 *= width/800
            x2 *= width / 800
            y1 *= height / 800
            y2 *= height / 800
            # 标出方框
            image_draw.rectangle([(x1, y1), (x2, y2)], outline="red", width=5)
            image_draw.text((x1 + 10, y1 + 5), f"{object_list[label]}, {score: .3f}", fill=255)
            selected_boxes.append((box, label))

    # 显示带有标注的图像
    image.show()
    image.save(image_dir + "processed_"+image_name)