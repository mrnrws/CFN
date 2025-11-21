import torch
import numpy as np
from scipy.spatial import cKDTree


def com_metrics(outputs,labels):
    if isinstance(outputs, torch.Tensor):
        outputs = outputs.detach().cpu()
    if isinstance(labels, torch.Tensor):
        labels = labels.detach().cpu()

    gt = labels.float().numpy()[0, 0]
    pre_binary = (outputs > 0.5).float().numpy()[0, 0]
    
    TP = (pre_binary * gt).sum()  
    FP = (pre_binary * (1 - gt)).sum()  
    TN = ((1 - pre_binary) * (1 - gt)).sum()  
    FN = ((1 - pre_binary) * gt).sum()  
    
    precision = TP / (TP + FP+ 1e-7)
    recall = TP / (TP + FN + 1e-7)
    specificity = TN/(TN+FP+1e-7)
    accuracy = (TP + TN) / (TP + TN + FP + FN + 1e-7)
    f1 = 2 * (precision * recall) / (precision + recall + 1e-7)
        
    iou = TP / (TP + FP + FN + 1e-7)
    miou = 0.5* (TP / (TP + FP + FN + 1e-7) + TN / (TN + FP + FN + 1e-7) )
    dice = 2 * TP / (2 * TP + FP + FN + 1e-7)


    def voxel_to_points(voxel_data, threshold=0.5):
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
            dists1 = tree_pred.query(gt_points, k=1)[0]  
            dists2 = tree_gt.query(pred_points, k=1)[0]  
            hd = max(dists1.max(),dists2.max())  
            hd_95 = np.percentile(np.concatenate([dists1, dists2]), 95)  
            return hd, hd_95

    def compute_distance(pred, gt):
        pred_points = downsample_points(voxel_to_points(pred))  
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


