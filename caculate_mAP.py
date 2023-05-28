import torch
import numpy as np

def calculate_iou(box1, box2):
    # 计算交集的左上角和右下角坐标
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])

    # 计算交集面积和并集面积
    intersection = max(0, x2 - x1 + 1) * max(0, y2 - y1 + 1)
    area_box1 = (box1[2] - box1[0] + 1) * (box1[3] - box1[1] + 1)
    area_box2 = (box2[2] - box2[0] + 1) * (box2[3] - box2[1] + 1)
    union = area_box1 + area_box2 - intersection

    # 计算IoU
    iou = intersection / union if union > 0 else 0
    return iou

# 计算ap
def calculate_ap(precision, recall):
    # 将 precision 和 recall 转换为 numpy 数组
    precision = np.array(precision)
    recall = np.array(recall)


    interpolated_precision = np.maximum.accumulate(precision)  # 按照递减的顺序计算每个位置的最大精确度
    recall_range = np.linspace(0, 1, num=101)  # 在 [0, 1] 范围内取 101 个点
    interpolated_precision = np.interp(recall_range, recall, interpolated_precision)  # 插值计算精确度

    # 计算 Average Precision (AP)
    ap = np.mean(interpolated_precision)


    return ap

def evaluate(pred_boxes, pred_labels, pred_scores, target_boxes, target_labels, conf_threshold=0.05,
             iou_threshold=0.5):
    """
    计算每一类的ap
    """
    average_precisions = dict()
    flattened_target_labels = [item for sublist in target_labels for item in sublist]  # 展开列表
    for c in np.unique(flattened_target_labels):
        tp = []
        fp = []
        tp_fp_scores = []
#         对于每一类别，查看每一张图对应类别的候选框，并计算iou
        for pred_boxes_batch, pred_labels_batch, pred_scores_batch, target_boxes_batch, target_labels_batch in zip(
            pred_boxes, pred_labels, pred_scores, target_boxes, target_labels
        ):
            pred_boxes_for_this_label = pred_boxes_batch[(pred_labels_batch==c) & (pred_scores_batch>=conf_threshold)]
            pred_scores_for_this_label = pred_scores_batch[(pred_labels_batch==c) & (pred_scores_batch>=conf_threshold)]
            target_boxes_for_this_label = target_boxes_batch[target_labels_batch == c]
            if len(pred_boxes_for_this_label) == 0 or len(target_boxes_for_this_label)==0:
                continue

            target_box_detected = [] #防止同一个target_box被多个pred_box重合
            pred_boxes_ious_and_ground_truth = []
            for pred_box, pred_score in zip(pred_boxes_for_this_label, pred_scores_for_this_label):
                ious = [calculate_iou(pred_box, target_box) for target_box in target_boxes_for_this_label]
                max_iou = np.max(ious)
                max_idx = np.argmax(ious)
                pred_boxes_ious_and_ground_truth.append((max_iou, max_idx, pred_score))

            # 计算哪些是tp，哪些是fp,
            pred_boxes_ious_and_ground_truth = sorted(pred_boxes_ious_and_ground_truth, reverse=True)
            for max_iou, max_idx, pred_score in pred_boxes_ious_and_ground_truth:
                if max_iou >= iou_threshold and max_idx not in target_box_detected:
                    tp_fp_scores.append(('tp', pred_scores))
                    target_box_detected.append(max_idx)
                else:
                    tp_fp_scores.append(('fp', pred_scores))
        #填入tp与fp中
        for tp_fp, score in sorted(tp_fp_scores, key=lambda x:x[1], reverse=True):
            if tp_fp == 'tp':
                tp.append(1)
                fp.append(0)
            else:
                tp.append(0)
                fp.append(1)

        if len(tp) == 0:
            continue
        # 计算每个预测框的精确率和召回率
        tp_cumsum = np.cumsum(tp)
        fp_cumsum = np.cumsum(fp)
        recall = tp_cumsum / len(tp)
        precision = tp_cumsum / (tp_cumsum + fp_cumsum)
        print(f"type {c}, precision {precision}, recall {recall}")
        ap = calculate_ap(precision, recall)
        average_precisions[c] = ap
    return average_precisions



def calculate_mAP(model, test_loader, iou_threshold=0.5, conf_threshold=0.05, device='cpu'):
    model.eval()
    pred_boxes = []
    pred_labels = []
    pred_scores = []
    target_boxes = []
    target_labels = []
    with torch.no_grad():
        for images, targets in test_loader:
            # 将数据传输到设备上
            images = list(image.to(device) for image in images)
            targets = [{k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in t.items()} for t in targets]

            # 获取模型的预测结果
            outputs = model(images)
            # 遍历每个输出结果
            for i, output in enumerate(outputs):
                # 将预测结果和目标结果转换为numpy数组
                pred_boxes_batch = output['boxes'].detach().cpu().numpy()
                pred_labels_batch = output['labels'].detach().cpu().numpy()
                pred_scores_batch = output['scores'].detach().cpu().numpy()
                target_boxes_batch = targets[i]['boxes'].detach().cpu().numpy()
                target_labels_batch = targets[i]['labels'].detach().cpu().numpy()

                pred_boxes.append(pred_boxes_batch)
                pred_labels.append(pred_labels_batch)
                pred_scores.append(pred_scores_batch)
                target_boxes.append(target_boxes_batch)
                target_labels.append(target_labels_batch)

        ap = evaluate(pred_boxes, pred_labels, pred_scores, target_boxes, target_labels)
        print(ap)
        mAP = sum(ap.values())/len(ap) if len(ap)>0 else 0
        return mAP
