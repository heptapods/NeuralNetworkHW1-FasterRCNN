from torchvision.transforms import transforms as T
import torch
# 确定类别
object_dict = {
    'aeroplane': 1,
    'bicycle': 2,
    'bird': 3,
    'boat': 4,
    'bottle': 5,
    'bus': 6,
    'car': 7,
    'cat': 8,
    'chair': 9,
    'cow': 10,
    'diningtable': 11,
    'dog': 12,
    'horse': 13,
    'motorbike':14,
    'person': 15,
    'pottedplant': 16,
    'sheep': 17,
    'sofa': 18,
    'train': 19,
    'tvmonitor': 20
}

object_list = ['background']
object_list.extend(object_dict.keys())

# 定义数据预处理
transform = T.Compose([
    T.Resize((800, 800)),
    T.ToTensor(),
    T.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
])


def collate_fn(batch):
    transformed_images = []
    targets = []
    for img, target in batch:
        # 将 "bndbox" 转换为 "boxes"
        boxes = []
        labels = []
        width, height = img.size
        for obj in target['annotation']['object']:
            box = [
                float(obj['bndbox']['xmin']) * 800/width,
                float(obj['bndbox']['ymin']) * 800/height,
                float(obj['bndbox']['xmax']) * 800/width,
                float(obj['bndbox']['ymax']) * 800/height,
            ]
            boxes.append(box)
            labels.append(object_dict[obj['name']])
        target['boxes'] = torch.tensor(boxes, dtype=torch.float)
        target['labels'] = torch.tensor(labels, dtype=torch.int64)
        transformed_images.append(transform(img))
        targets.append(target)
    return transformed_images, targets