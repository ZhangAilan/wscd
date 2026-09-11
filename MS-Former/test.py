import sys

sys.path.insert(0, '.')
import torch
import torchvision.transforms.functional as ttf
import scipy.io as scio
import torch.backends.cudnn as cudnn
from torch.nn.parallel import gather
import torch.optim.lr_scheduler
import datasets.dataset as myDataLoader
import datasets.Transforms as myTransforms
from utils.metric_tool import ConfuseMatrixMeter
from PIL import Image
import os, time
import numpy as np
from argparse import ArgumentParser
from models.model import get_model


def calculate_binary_metrics(label_dir, prediction_dir, file_names):
    """Calculate dataset-level metrics from saved binary predictions and labels."""
    confusion_matrix = np.zeros((2, 2), dtype=np.int64)
    evaluated = 0
    skipped = []

    for file_name in file_names:
        label_path = os.path.join(label_dir, file_name)
        prediction_path = os.path.join(prediction_dir, file_name)
        if not os.path.isfile(label_path) or not os.path.isfile(prediction_path):
            skipped.append(file_name)
            continue

        label = np.asarray(Image.open(label_path).convert('L')) > 0
        prediction = np.asarray(Image.open(prediction_path).convert('L')) > 0
        if label.shape != prediction.shape:
            raise ValueError(
                f"Prediction/label size mismatch for {file_name}: "
                f"{prediction.shape} versus {label.shape}."
            )

        encoded = 2 * label.astype(np.uint8).ravel() + prediction.astype(np.uint8).ravel()
        confusion_matrix += np.bincount(encoded, minlength=4).reshape(2, 2)
        evaluated += 1

    if evaluated == 0:
        raise RuntimeError("No matching binary prediction and label image pairs were found.")

    tn, fp = confusion_matrix[0]
    fn, tp = confusion_matrix[1]
    eps = np.finfo(np.float64).eps
    total = confusion_matrix.sum()
    overall_accuracy = (tp + tn) / (total + eps)
    class_accuracy = np.diag(confusion_matrix) / (confusion_matrix.sum(axis=1) + eps)
    iou = np.diag(confusion_matrix) / (
        confusion_matrix.sum(axis=1) + confusion_matrix.sum(axis=0) - np.diag(confusion_matrix) + eps
    )
    frequency = confusion_matrix.sum(axis=1) / (total + eps)
    precision = tp / (tp + fp + eps)
    recall = tp / (tp + fn + eps)
    f1 = 2 * precision * recall / (precision + recall + eps)
    miou = iou.mean()
    fwavacc = (frequency * iou).sum()
    expected_accuracy = (
        (tp + fn) * (tp + fp) + (tn + fp) * (tn + fn)
    ) / ((total + eps) ** 2)
    kappa = (overall_accuracy - expected_accuracy) / (1 - expected_accuracy + eps)

    lines = [
        "=" * 50,
        "Binary Change Detection Metrics",
        "=" * 50,
        f"Evaluated image pairs:        {evaluated}",
        f"Skipped image pairs:          {len(skipped)}",
        f"Confusion matrix [[TN, FP], [FN, TP]]: {confusion_matrix.tolist()}",
        f"Overall Accuracy:             {overall_accuracy:.4f}",
        f"Class Average Accuracy:       {class_accuracy.mean():.4f}",
        f"IoU per class [unchanged, changed]: {iou.tolist()}",
        f"Mean IoU:                     {miou:.4f}",
        f"Frequency Weighted Accuracy:  {fwavacc:.4f}",
        f"Change Precision:             {precision:.4f}",
        f"Change Recall:                {recall:.4f}",
        f"Change F1:                    {f1:.4f}",
        f"Kappa:                        {kappa:.4f}",
        "=" * 50,
    ]
    return "\n".join(lines), skipped


@torch.no_grad()
def val(args, val_loader, model, vis_dir):
    model.eval()

    cd_evaluation = ConfuseMatrixMeter(n_class=2)

    total_batches = len(val_loader)
    print(len(val_loader))

    for batch_idx, batched_inputs in enumerate(val_loader):

        img, patch_target, target = batched_inputs
        B = img.size(0)  # batch size
        start_time = time.time()

        if args.onGPU == True:
            img = img.cuda()
            patch_target = patch_target.cuda()
            target = target.cuda()

        img_var = torch.autograd.Variable(img).float()
        patch_target_var = torch.autograd.Variable(patch_target).float()
        target_var = torch.autograd.Variable(target).float()  # only used for evaluation

        # run the model
        _, C, H, W = img_var.size()
        change_mask = torch.zeros(B, 1, H, W).cuda()
        for patch_h in range(0, H, 256):
            for patch_w in range(0, W, 256):
                h_end = min(patch_h + 256, H)
                w_end = min(patch_w + 256, W)
                patch_mask = model(img_var[:, :, patch_h: h_end, patch_w: w_end])
                if patch_mask.size(2) != (h_end - patch_h) or patch_mask.size(3) != (w_end - patch_w):
                    patch_mask = torch.nn.functional.interpolate(
                        patch_mask, size=(h_end - patch_h, w_end - patch_w), mode='bilinear', align_corners=False)
                change_mask[:, :, patch_h: h_end, patch_w: w_end] = patch_mask

        pred = torch.where(change_mask > 0.5, torch.ones_like(change_mask), torch.zeros_like(change_mask)).long()

        # torch.cuda.synchronize()
        time_taken = time.time() - start_time

        # compute the confusion matrix
        if args.onGPU and torch.cuda.device_count() > 1:
            pred = gather(pred, 0, dim=0)

        # save change maps for each image in the batch
        for i in range(B):
            img_name = val_loader.sampler.data_source.file_list[batch_idx * B + i]
            
            pr = pred[i, 0].cpu().numpy()
            gt = target_var[i, 0].cpu().numpy()
            index_tp = np.where(np.logical_and(pr == 1, gt == 1))
            index_fp = np.where(np.logical_and(pr == 1, gt == 0))
            index_tn = np.where(np.logical_and(pr == 0, gt == 0))
            index_fn = np.where(np.logical_and(pr == 0, gt == 1))
            #
            map = np.zeros([gt.shape[0], gt.shape[1], 3])
            map[index_tp] = [255, 255, 255]  # white
            map[index_fp] = [255, 0, 0]  # red
            map[index_tn] = [0, 0, 0]  # black
            map[index_fn] = [0, 255, 0]  # green

            change_map = Image.fromarray(np.array(map, dtype=np.uint8))
            change_map.save(vis_dir + img_name)

            # 保存二值预测图到单独文件夹
            binary_map = Image.fromarray((pr * 255).astype(np.uint8))
            binary_map.save(args.binary_vis_dir + img_name)

            f1 = cd_evaluation.update_cm(pr, gt)

        if batch_idx % 5 == 0:
            print('\r[%d/%d] F1: %3f time: %.3f' % (batch_idx, total_batches, f1, time_taken),
                  end='')

    scores = cd_evaluation.get_scores()

    return scores


def val_change_detection(args):
    torch.backends.cudnn.benchmark = True
    SEED = 2023
    torch.manual_seed(SEED)
    torch.cuda.manual_seed(SEED)

    model = get_model(args.patch_size, args.memory_length, args.depth, args.dino_ckpt_path)

    args.save_dir = args.save_dir + '_iter_' + str(args.max_steps) + '_lr_' + str(
        args.lr) + '_p_' + str(args.patch_size) + '_m_' + str(args.memory_length) + '_d_' + str(args.depth) + '/'

    args.vis_dir = './predict/' + '_patch_' + str(args.patch_size) + '/'
    args.binary_vis_dir = './predict_binary/' + '_patch_' + str(args.patch_size) + '/'


    if not os.path.exists(args.save_dir):
        os.makedirs(args.save_dir)

    if not os.path.exists(args.vis_dir):
        os.makedirs(args.vis_dir)
    
    if not os.path.exists(args.binary_vis_dir):
        os.makedirs(args.binary_vis_dir)

    if args.onGPU:
        model = model.cuda()

    total_params = sum([np.prod(p.size()) for p in model.parameters()])
    print('Total network parameters (excluding idr): ' + str(total_params))

    mean = [0.406, 0.456, 0.485, 0.406, 0.456, 0.485]
    std = [0.225, 0.224, 0.229, 0.225, 0.224, 0.229]

    # compose the data with transforms
    valDataset = myTransforms.Compose([
        myTransforms.Normalize(mean=mean, std=std),
        myTransforms.Scale(args.inWidth, args.inHeight),
        myTransforms.ToTensor()
    ])

    test_data = myDataLoader.Dataset("test", file_root=args.test_data_root, transform=valDataset, list_file=args.test_list_file)
    testLoader = torch.utils.data.DataLoader(
        test_data, shuffle=False,
        batch_size=args.batch_size, num_workers=args.num_workers, pin_memory=False)

    if args.onGPU:
        cudnn.benchmark = True

    logFileLoc = args.save_dir + args.logFile
    if os.path.isfile(logFileLoc):
        logger = open(logFileLoc, 'a')
    else:
        logger = open(logFileLoc, 'w')
        logger.write("Parameters: %s" % (str(total_params)))
        logger.write(
            "\n%s\t%s\t%s\t%s\t%s\t%s" % ('Epoch', 'Kappa', 'IoU', 'F1', 'R', 'P'))
    logger.flush()

    # load the model
    model_file_name = args.save_dir + 'best_model.pth'
    state_dict = torch.load(model_file_name)
    model.load_state_dict(state_dict)

    score_test = val(args, testLoader, model, args.vis_dir)
    report, skipped = calculate_binary_metrics(
        label_dir=os.path.join(args.test_data_root, 'label'),
        prediction_dir=args.binary_vis_dir,
        file_names=test_data.file_list,
    )
    metrics_path = os.path.join(args.binary_vis_dir, 'binary_metrics.txt')
    with open(metrics_path, 'w', encoding='utf-8') as metrics_file:
        metrics_file.write(report + '\n')
        if skipped:
            metrics_file.write('Skipped files:\n' + '\n'.join(skipped) + '\n')
    print('\n' + report)
    print(f'Binary metrics saved to: {metrics_path}')
    torch.cuda.empty_cache()
    print("\nLEVIR_Test :\t Kappa (te) = %.4f\t IoU (te) = %.4f\t F1 (te) = %.4f\t R (te) = %.4f\t P (te) = %.4f" \
          % (score_test['Kappa'], score_test['IoU'], score_test['F1'], score_test['recall'], score_test['precision']))
    logger.write("\n%s\t\t%.4f\t\t%.4f\t\t%.4f\t\t%.4f\t\t%.4f" % ('LEVIR_Test',
                                                                   score_test['Kappa'],
                                                                   score_test['IoU'],
                                                                   score_test['F1'],
                                                                   score_test['recall'],
                                                                   score_test['precision']))
    logger.flush()
    scio.savemat(args.vis_dir + 'results.mat', score_test)

    logger.close()


if __name__ == '__main__':
    dataset_root = r'E:\weakly_CD_dataset\dataset\whu_CDC_dataset\whu_CDC_dataset_converted'
    dino_ckpt_path = r'E:\zyh-dinov3-wcd\dino\dinov3_vith16plus_pretrain_lvd1689m-7c1da9a5.pth'
    parser = ArgumentParser()
    parser.add_argument('--test_data_root', type=str, default=dataset_root, help='Testing data directory')
    parser.add_argument('--test_list_file', type=str, default=os.path.join(dataset_root, 'list', 'test.txt'),
                        help='Testing list file path')
    parser.add_argument('--inWidth', type=int, default=256, help='Width of RGB image')
    parser.add_argument('--inHeight', type=int, default=256, help='Height of RGB image')
    parser.add_argument('--patch_size', type=int, default=16, help='size of label patch')
    parser.add_argument('--memory_length', type=int, default=128, help='size of label patch')
    parser.add_argument('--depth', type=int, default=2, help='size of label patch')
    parser.add_argument('--max_steps', type=int, default=20000, help='Max. number of iterations')
    parser.add_argument('--num_workers', type=int, default=4, help='No. of parallel threads')
    parser.add_argument('--batch_size', type=int, default=1, help='Batch size')
    parser.add_argument('--lr', type=float, default=4e-5, help='Initial learning rate')
    parser.add_argument('--lr_mode', default='poly', help='Learning rate policy')
    parser.add_argument('--save_dir', default='./weights/whu/', help='Directory to save the results')
    parser.add_argument('--logFile', default='trainValLog.txt',
                        help='File that stores the training and validation logs')
    parser.add_argument('--onGPU', default=True, type=lambda x: (str(x).lower() == 'true'),
                        help='Run on CPU or GPU. If TRUE, then GPU.')
    parser.add_argument('--weight', default='', type=str, help='pretrained weight, can be a non-strict copy')
    parser.add_argument('--dino_ckpt_path', default=dino_ckpt_path, type=str,
                        help='DINOv3 ViT-H+/16 checkpoint; overrides DINO_CKPT_PATH and the project default')
    parser.add_argument('--ms', type=int, default=0, help='apply multi-scale training, default False')

    args = parser.parse_args()
    print('Called with args:')
    print(args)

    val_change_detection(args)
