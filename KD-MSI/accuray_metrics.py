import os
import cv2
import numpy as np
from sklearn.metrics import confusion_matrix
import argparse


class IOUMetric:
    """
    Class to calculate mean-iou using fast_hist method
    """

    def __init__(self, num_classes):
        self.num_classes = num_classes
        self.hist = np.zeros((num_classes, num_classes))  # 混淆矩阵

    def _fast_hist(self, label_pred, label_true):
        # 找出标签中需要计算的类别,去掉了背景
        mask = (label_true >= 0) & (label_true < self.num_classes)
        # # np.bincount计算了从0到n**2-1这n**2个数中每个数出现的次数，返回值形状(n, n)
        hist = np.bincount(self.num_classes * label_true[mask].astype(int) +label_pred[mask], minlength=self.num_classes ** 2).reshape(self.num_classes, self.num_classes)
        return hist

    def evaluate(self, predictions, gts):
        for lp, lt in zip(predictions, gts):
            assert len(lp.flatten()) == len(lt.flatten())
            self.hist += self._fast_hist(lp.flatten(), lt.flatten())
        iou = np.diag(self.hist) / (self.hist.sum(axis=1) + self.hist.sum(axis=0) - np.diag(self.hist))
        miou = np.nanmean(iou)
        acc = np.diag(self.hist).sum() / self.hist.sum()
        acc_cls = np.nanmean(np.diag(self.hist) / self.hist.sum(axis=1))
        freq = self.hist.sum(axis=1) / self.hist.sum()
        fwavacc = (freq[freq > 0] * iou[freq > 0]).sum()
        return acc, acc_cls, iou, miou, fwavacc



def calculate_metrics(label_path, predict_path, num_images=None):
    '''
    计算真实mask文件夹和预测mask文件夹之间的九项指标
    Args:
        label_path: 真实mask文件夹路径
        predict_path: 预测mask文件夹路径
        num_images: 计算前n张图像,默认为None表示计算所有图像
    Returns:
        dict: 包含9个评价指标的字典
    '''

    # 获取预测文件列表和标签文件列表
    predict_files = sorted([f for f in os.listdir(predict_path) if f.endswith('.png')])
    label_files = sorted([f for f in os.listdir(label_path) if f.endswith('.png')])

    # 获取两个文件夹中共有的文件
    common_files = sorted(list(set(predict_files) & set(label_files)))
    if num_images is not None:
        common_files = common_files[:num_images]

    # 读取mask并转换格式
    labels = []
    predicts = []
    for im in common_files:
        lab_path = os.path.join(label_path, im)
        pre_path = os.path.join(predict_path, im)

        # 添加文件存在性检查
        if not os.path.exists(lab_path):
            print(f"标签文件不存在: {lab_path}")
            continue
        if not os.path.exists(pre_path):
            print(f"预测文件不存在: {pre_path}")
            continue

        # 添加图像读取检查
        label = cv2.imread(lab_path, 0)
        pre = cv2.imread(pre_path, 0)

        if label is None:
            print(f"无法读取标签图像: {lab_path}")
            continue
        if pre is None:
            print(f"无法读取预测图像: {pre_path}")
            continue

        # 继续处理
        label = label/255
        pre = pre/255
        pre = np.uint8(pre)
        label = np.uint8(label)
        labels.append(label)
        predicts.append(pre)

    # 添加数据有效性检查
    if not labels or not predicts:
        raise ValueError("没有成功读取到任何有效的图像对")

    # 计算IOU相关指标
    el = IOUMetric(2)
    acc, acc_cls, iou, miou, fwavacc = el.evaluate(predicts, labels)

    # 计算混淆矩阵相关指标
    init = np.zeros((2, 2))
    for im in common_files:
        lb_path = os.path.join(label_path, im)
        pre_path = os.path.join(predict_path, im)
        lb = cv2.imread(lb_path, 0)/255
        lb = np.uint8(lb)
        pre = cv2.imread(pre_path, 0)/255
        pre = np.uint8(pre)
        lb = lb.flatten()
        pre = pre.flatten()
        confuse = confusion_matrix(lb, pre,labels=[0, 1])
        init += confuse

    precision = init[1][1] / (init[0][1] + init[1][1])
    recall = init[1][1] / (init[1][0] + init[1][1])
    accuracy = (init[0][0] + init[1][1]) / init.sum()
    f1_score = 2 * precision * recall / (precision + recall)
    metrics_dict = {
        'acc': acc,                  # 整体准确率
        'acc_cls': acc_cls,          # 各类别准确率的平均值
        'iou': iou,                  # 各类别的IoU
        'miou': miou,                # 各类别IoU的平均值
        'fwavacc': fwavacc,          # 频权平均准确率
        'class_accuracy': precision,  # 类别准确率
        'class_recall': recall,       # 类别召回率
        'accuracy': accuracy,         # 准确率
        'f1_score': f1_score         # F1分数
    }

    return metrics_dict


def main(args):
    label_folder = args.label_folder
    predict_folder = args.predict_folder
    num_images = args.num_images  # None 表示处理全部

    # Check if folders exist
    if not os.path.isdir(label_folder):
        print(f"Error: Label folder does not exist: {label_folder}")
        return

    if not os.path.isdir(predict_folder):
        print(f"Error: Predict folder does not exist: {predict_folder}")
        return

    try:
        # Calculate metrics
        metrics = calculate_metrics(label_folder, predict_folder, num_images)

        # Print results
        print("=" * 50)
        print("Accuracy Metrics Results:")
        print("=" * 50)
        print(f"Overall Accuracy:            {metrics['acc']:.4f}")
        print(f"Class Average Accuracy:      {metrics['acc_cls']:.4f}")
        print(f"IoU per class:               {metrics['iou']}")
        print(f"Mean IoU:                    {metrics['miou']:.4f}")
        print(f"Frequency Weighted Accuracy: {metrics['fwavacc']:.4f}")
        print(f"Class Precision:             {metrics['class_accuracy']:.4f}")
        print(f"Class Recall:                {metrics['class_recall']:.4f}")
        print(f"Accuracy:                    {metrics['accuracy']:.4f}")
        print(f"F1 Score:                    {metrics['f1_score']:.4f}")
        print("=" * 50)

    except Exception as e:
        print(f"Error calculating metrics: {str(e)}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Evaluate change detection results with multiple accuracy metrics"
    )

    parser.add_argument(
        "--label_folder",
        type=str,
        required=True,
        help="Path to ground-truth label folder"
    )

    parser.add_argument(
        "--predict_folder",
        type=str,
        required=True,
        help="Path to prediction result folder"
    )

    parser.add_argument(
        "--num_images",
        type=int,
        default=None,
        help="Number of images to process (default: all images)"
    )

    args = parser.parse_args()
    main(args)