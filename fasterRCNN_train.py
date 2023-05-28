import torch
import torchvision
from torchvision.models.detection import fasterrcnn_resnet50_fpn
from torchvision.datasets import VOCDetection
from torch.utils.data import DataLoader
import json
from utils import object_dict, transform, collate_fn
from caculate_mAP import calculate_mAP

# 定义数据加载器 train_loader 和 test_loader
my_device = 'cuda'

# 数据集路径
root_dir = '/Users/yanxinyu/pythonProject/HW2_FasterRCNN'
# 数据集使用的年份
year = '2012'
# 训练/测试集
train_set = VOCDetection(root=root_dir, year=year, image_set='train', download=False)
val_set = VOCDetection(root=root_dir, year=year, image_set='val', download=False)

# 将数据集包装为 DataLoader，batch_size可以根据计算资源调整
train_loader = DataLoader(train_set, batch_size=16, shuffle=True, num_workers=4, collate_fn=collate_fn)
test_loader = DataLoader(val_set, batch_size=16, shuffle=False, num_workers=4, collate_fn=collate_fn)

# 创建 Faster R-CNN 模型
model = fasterrcnn_resnet50_fpn(num_classes=21, weights_backbone=torchvision.models.ResNet50_Weights.IMAGENET1K_V1)


# 定义优化器
optimizer = torch.optim.SGD(model.parameters(), lr=0.01, momentum=0.1)
lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=0.8)

def estimate_loss(model, data_loader, sample_ratio=0.05):
    sample_size = int(len(data_loader.dataset) * sample_ratio)
    sampled_data_loader = torch.utils.data.DataLoader(data_loader.dataset, batch_size=data_loader.batch_size, sampler=torch.utils.data.RandomSampler(data_loader.dataset, replacement=True, num_samples=sample_size), collate_fn=collate_fn)
    total_loss = 0
    with torch.no_grad():
        for images, targets in sampled_data_loader:
            images = list(image.to(my_device) for image in images)
            targets = [{k: v.to(my_device) if torch.is_tensor(v) else v for k, v in t.items()} for t in targets]
            model.to(my_device)
            loss_dict = model(images, targets)
            total_loss += sum(loss for loss in loss_dict.values()).item()
    estimated_loss = total_loss / sample_size * len(data_loader.dataset)
    return estimated_loss

def estimate_mAP(model, data_loader, iou_threshold=0.5, conf_threshold=0.05, device='cpu', sample_ratio=0.05):
    sample_size = int(len(data_loader.dataset) * sample_ratio)
    sampled_data_loader = torch.utils.data.DataLoader(data_loader.dataset, batch_size=data_loader.batch_size, sampler=torch.utils.data.RandomSampler(data_loader.dataset, replacement=True, num_samples=sample_size), collate_fn=collate_fn)
    return calculate_mAP(model, sampled_data_loader, iou_threshold, conf_threshold, device)

if __name__ == '__main__':
# 迭代训练
    num_epochs = 20
    loss_and_mAP = {
            'train_loss':[],
            'test_loss':[],
            'mAP':[]
        }

    # 计算初始值的loss
    train_loss = estimate_loss(model, train_loader, sample_ratio=0.2)
    test_loss = estimate_loss(model, test_loader, sample_ratio=0.2)
    print(f"epoch={0}, round={0}, train_loss={train_loss}")
    print(f"epoch={0}, round={0}, test_loss={test_loss}")
    print('lr=',optimizer.param_groups[0]['lr'])
    loss_and_mAP['train_loss'].append(train_loss)
    loss_and_mAP['test_loss'].append(test_loss)

    for epoch in range(num_epochs):
        num = 0
        for images, targets in train_loader:
            # 将图像和目标数据加载到设备
            images = [image.to(my_device) for image in images]
            targets = [{k: v.to(my_device) if torch.is_tensor(v) else v for k, v in t.items()} for t in targets]
            model.to(my_device)

            # 设置模型为训练模式
            model.train()

            # 清零优化器的梯度
            optimizer.zero_grad()

            # 获取模型的预测结果和损失
            loss_dict = model(images, targets)
            print(loss_dict)

            # 计算总损失
            loss = sum(loss for loss in loss_dict.values())

            # 反向传播和梯度更新
            loss.backward()
            optimizer.step()
            # 输出当前迭代的损失函数值
            num += 1
            print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {loss_dict}")
            print(f"Round [{num}], Loss: {loss}")

            if num%10==0:
                train_loss = estimate_loss(model, train_loader, sample_ratio=0.2)
                test_loss = estimate_loss(model, test_loader, sample_ratio=0.2)
                print(f"epoch={epoch}, round={num}, train_loss={train_loss}")
                print(f"epoch={epoch}, round={num}, test_loss={test_loss}")
                print('lr=',optimizer.param_groups[0]['lr'])
                loss_and_mAP['train_loss'].append(train_loss)
                loss_and_mAP['test_loss'].append(test_loss)


                # 打印结果
                print(f'Epoch {epoch + 1}/{num_epochs}, round {num}'
                      f'Train Loss: {train_loss:.3f}, '
                      f'Test Loss: {test_loss:.3f}, '
                )
                with open('loss_and_mAP.json', 'w') as file:
                    json.dump(loss_and_mAP, file)

        lr_scheduler.step()
        mAP = estimate_mAP(model, test_loader, sample_ratio=0.10)
        loss_and_mAP['mAP'].append(mAP)
        torch.save(model.state_dict(), f'new_model_epoch{epoch+1}.pth')