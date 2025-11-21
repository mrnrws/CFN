import torch.nn as nn
import torch
import numpy as np
from scipy.spatial import cKDTree


def com_metrics(outputs,labels,threshold=0.5):
    if isinstance(outputs, torch.Tensor):
        outputs = outputs.detach().cpu()
    if isinstance(labels, torch.Tensor):
        labels = labels.detach().cpu()

    gt = labels.float().numpy()[0, 0]
    pre_binary = (outputs > 0.5).float().numpy()[0, 0]
    # pos_weight = (gt == 0).sum().float() / (gt == 1).sum().float()
    # neg_weight = (gt == 1).sum().float() / (gt == 0).sum().float()
    # # print('pos_weight: {:.4f}, neg_weight: {:.4f}'.format(pos_weight, neg_weight))
    
    TP = (pre_binary * gt).sum()  # True Positive
    FP = (pre_binary * (1 - gt)).sum()  # False Positive
    TN = ((1 - pre_binary) * (1 - gt)).sum()  # True Negative
    FN = ((1 - pre_binary) * gt).sum()  # False Negative
    # print('TP: {}, FP: {}, TN: {}, FN: {}'.format(TP, FP, TN, FN))
    
    precision = TP / (TP + FP+ 1e-7)
    recall = TP / (TP + FN + 1e-7)
    specificity = TN/(TN+FP+1e-7)
    accuracy = (TP + TN) / (TP + TN + FP + FN + 1e-7)
    f1 = 2 * (precision * recall) / (precision + recall + 1e-7)
        
    iou = TP / (TP + FP + FN + 1e-7)
    miou = 0.5* (TP / (TP + FP + FN + 1e-7) + TN / (TN + FP + FN + 1e-7) )
    dice = 2 * TP / (2 * TP + FP + FN + 1e-7)


    def voxel_to_points(voxel_data, threshold=0.5):
        """ 将体素数据转换为点云（坐标列表） """
        return np.argwhere(voxel_data > threshold) 

    def downsample_points(points, max_pts=10000):
        if len(points) > max_pts:
            idx = np.random.choice(len(points), max_pts, replace=False)
            return points[idx]
        return points

    def hausdorff_dist(pred_points,gt_points):
        if len(pred_points) == 0 or len(gt_points) == 0:
            print("Warning: empty points in hausdorff_dist")
            return 0.0, 0.0
        if len(pred_points) > 0 and len(gt_points) > 0:
            tree_pred,tree_gt = cKDTree(pred_points),cKDTree(gt_points)
            dists1 = tree_pred.query(gt_points, k=1)[0]  # 计算 set2 到 set1 的最小距离
            dists2 = tree_gt.query(pred_points, k=1)[0]  
            hd = max(dists1.max(),dists2.max())  # hausdorff_dist
            hd_95 = np.percentile(np.concatenate([dists1, dists2]), 95)  # hausdorff_95
            return hd, hd_95

    def compute_distance(pred, gt):
        pred_points = downsample_points(voxel_to_points(pred))  # 转换为点云
        gt_points = downsample_points(voxel_to_points(gt)) 

        hd, hd_95 = hausdorff_dist(pred_points, gt_points)

        return hd, hd_95

    hd, hd_95 = compute_distance(pre_binary, gt)

    def to_float(x):
        return x.item() if isinstance(x, torch.Tensor) else x

    return {
        "Precision": precision.item(),
        "Recall": recall.item(),
        "F1-score": f1.item(),
        "IoU": iou.item(),
        "mIoU": miou.item(),
        "HD": hd,
        "HD95": to_float(hd_95)
    }



'没有用到'
def con_matrix(outputs, labels, args):
    y_pred = outputs.detach().cpu().numpy()
    y_true = labels.detach().cpu().numpy()

    y_pred = y_pred.argmax(axis=1).flatten()
    y_true = y_true.flatten()

    num_class = args.out_channels
    current = confusion_matrix(y_true, y_pred, labels=range(num_class))  # confusion_matrix混淆矩阵，计算把xxx预测成xxx的次数

    # compute mean iou
    intersection = np.diag(current)
    # 一维数组的形式返回混淆矩阵的对角线元素
    ground_truth_set = current.sum(axis=1)
    # 按行求和
    predicted_set = current.sum(axis=0)
    # 按列求和
    union = ground_truth_set + predicted_set - intersection + 1e-7
    IoU = intersection / union.astype(np.float32)
    union_dice = ground_truth_set + predicted_set + 1e-7
    DICE = 2 * intersection / union_dice.astype(np.float32)

    return np.mean(IoU), np.mean(DICE)




