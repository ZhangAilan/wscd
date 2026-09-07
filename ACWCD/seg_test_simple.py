"""Fast, single-pass evaluation for ACWCD.

This variant intentionally trades a little accuracy for speed: it uses one
resized image pair per sample and does not use test-time flip or multi-scale
ensembling.  Prediction saving and metric calculation match ``seg_test.py``.
"""

import argparse
import os
from collections import OrderedDict

import numpy as np
import torch
import torch.nn.functional as F
from omegaconf import OmegaConf
from PIL import Image
from tqdm import tqdm

from datasets import weaklyCD
from models.model_ACWCD import ACWCD


parser = argparse.ArgumentParser()
parser.add_argument("--config", default="configs/LEVIR.yaml", type=str, help="config")
parser.add_argument("--save_dir", default="./results/LEVIR", type=str, help="save_dir")
parser.add_argument("--eval_set", default="test", type=str, help="eval_set")
parser.add_argument("--model_path", required=True, type=str, help="model_path")
parser.add_argument("--pooling", default="gmp", type=str, help="pooling method")
parser.add_argument("--bkg_score", default=0.50, type=float, help="bkg_score")
parser.add_argument("--resize_long", default=256, type=int, help="resize the long side")


def resize_long_side(inputs, resize_long):
    """Resize a batch while preserving its aspect ratio."""
    _, _, height, width = inputs.shape
    ratio = resize_long / max(height, width)
    target_height = max(1, int(height * ratio))
    target_width = max(1, int(width * ratio))
    return F.interpolate(
        inputs,
        size=(target_height, target_width),
        mode="bilinear",
        align_corners=False,
    )


def test(model, dataset, device):
    """Run exactly one forward pass per image pair and save binary PNG masks."""
    data_loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=cfg.test.batch_size,
        shuffle=False,
        num_workers=2,
        pin_memory=False,
    )

    model.to(device)
    with torch.no_grad():
        for name, inputs_a, inputs_b, labels, _ in tqdm(data_loader):
            inputs_a = inputs_a.to(device)
            inputs_b = inputs_b.to(device)

            # The original script averages flipped predictions at three scales.
            # Use only this base-scale pass in the simplified evaluator.
            inputs_a = resize_long_side(inputs_a, args.resize_long)
            inputs_b = resize_long_side(inputs_b, args.resize_long)
            _, logits, _ = model(inputs_a, inputs_b)

            logits = F.interpolate(
                logits,
                size=labels.shape[1:],
                mode="bilinear",
                align_corners=False,
            )
            predictions = torch.argmax(logits, dim=1)

            output_name = os.path.splitext(os.path.basename(name[0]))[0] + ".png"
            prediction_path = os.path.join(args.save_dir, "seg-prediction", output_name)
            Image.fromarray(
                (predictions.squeeze().cpu().numpy() * 255).astype(np.uint8)
            ).save(prediction_path)

            # Keep the reference script's false-negative/false-positive preview.
            prediction_np = predictions.squeeze().cpu().numpy()
            label_np = labels.squeeze().cpu().numpy()
            color_preview = np.zeros((*label_np.shape, 3), dtype=np.uint8)
            color_preview[label_np == 1] = [255, 255, 255]
            color_preview[np.logical_and(prediction_np == 0, label_np == 1)] = [0, 0, 255]
            color_preview[np.logical_and(prediction_np == 1, label_np == 0)] = [255, 0, 0]
            color_path = os.path.join(args.save_dir, "seg-prediction-color", output_name)
            Image.fromarray(color_preview).save(color_path)


def _load_single_channel_label(image_path):
    """Read binary label images while rejecting ambiguous multi-channel files."""
    label = np.asarray(Image.open(image_path))
    if label.ndim == 3:
        if not np.all(label == label[..., :1]):
            raise ValueError(f"Expected a grayscale binary label image: {image_path}")
        label = label[..., 0]
    if label.ndim != 2:
        raise ValueError(f"Expected a 2D label image: {image_path}")
    return label


def load_saved_predictions_and_labels(dataset, prediction_dir):
    """Load the exact PNG predictions saved by this script and matching labels."""
    predictions, gts = [], []
    missing_predictions = []

    for name in dataset.name_list:
        filename = os.path.splitext(os.path.basename(str(name)))[0] + ".png"
        prediction_path = os.path.join(prediction_dir, filename)
        label_path = os.path.join(dataset.label_dir, str(name))

        if not os.path.isfile(prediction_path):
            missing_predictions.append(prediction_path)
            continue
        if not os.path.isfile(label_path):
            raise FileNotFoundError(f"Label file not found: {label_path}")

        prediction = _load_single_channel_label(prediction_path)
        label = _load_single_channel_label(label_path)
        if prediction.shape != label.shape:
            raise ValueError(
                f"Prediction/label size mismatch for {filename}: "
                f"{prediction.shape} vs {label.shape}"
            )

        predictions.append((prediction > 0).astype(np.int16))
        gts.append((label > 0).astype(np.int16))

    if missing_predictions:
        examples = "\n".join(missing_predictions[:5])
        raise FileNotFoundError(
            f"Missing {len(missing_predictions)} prediction files in {prediction_dir}.\n{examples}"
        )
    if not predictions:
        raise RuntimeError(f"No prediction files found in {prediction_dir}")
    return predictions, gts


def calculate_metrics(predictions, gts, num_classes=2):
    """Calculate the same nine metrics used by the reference evaluation script."""
    hist = np.zeros((num_classes, num_classes), dtype=np.float64)

    for prediction, gt in zip(predictions, gts):
        prediction = prediction.flatten()
        gt = gt.flatten()
        valid = (gt >= 0) & (gt < num_classes)
        hist += np.bincount(
            num_classes * gt[valid].astype(int) + prediction[valid].astype(int),
            minlength=num_classes ** 2,
        ).reshape(num_classes, num_classes)

    with np.errstate(divide="ignore", invalid="ignore"):
        iou = np.diag(hist) / (hist.sum(axis=1) + hist.sum(axis=0) - np.diag(hist))
        acc = np.diag(hist).sum() / hist.sum()
        acc_cls = np.nanmean(np.diag(hist) / hist.sum(axis=1))
        freq = hist.sum(axis=1) / hist.sum()
        fwavacc = (freq[freq > 0] * iou[freq > 0]).sum()
        precision = hist[1, 1] / (hist[0, 1] + hist[1, 1])
        recall = hist[1, 1] / (hist[1, 0] + hist[1, 1])
        accuracy = (hist[0, 0] + hist[1, 1]) / hist.sum()
        f1_score = 2 * precision * recall / (precision + recall)

    return {
        "acc": acc,
        "acc_cls": acc_cls,
        "iou": iou,
        "miou": np.nanmean(iou),
        "fwavacc": fwavacc,
        "class_accuracy": precision,
        "class_recall": recall,
        "accuracy": accuracy,
        "f1_score": f1_score,
    }


def save_metrics(metrics, output_path):
    lines = [
        "=" * 50,
        "Accuracy Metrics Results:",
        "=" * 50,
        f"Overall Accuracy:            {metrics['acc']:.4f}",
        f"Class Average Accuracy:      {metrics['acc_cls']:.4f}",
        f"IoU per class:               {metrics['iou']}",
        f"Mean IoU:                    {metrics['miou']:.4f}",
        f"Frequency Weighted Accuracy: {metrics['fwavacc']:.4f}",
        f"Class Precision:             {metrics['class_accuracy']:.4f}",
        f"Class Recall:                {metrics['class_recall']:.4f}",
        f"Accuracy:                    {metrics['accuracy']:.4f}",
        f"F1 Score:                    {metrics['f1_score']:.4f}",
        "=" * 50,
    ]
    result_text = "\n".join(lines)
    print(result_text)
    with open(output_path, "w", encoding="utf-8") as file:
        file.write(result_text + "\n")


def main(config):
    test_dataset = weaklyCD.CDDataset(
        root_dir=config.dataset.root_dir,
        name_list_dir=config.dataset.name_list_dir,
        split=args.eval_set,
        stage="test",
        aug=False,
        num_classes=config.dataset.num_classes,
    )
    model = ACWCD(
        backbone=config.backbone.config,
        stride=config.backbone.stride,
        num_classes=config.dataset.num_classes,
        embedding_dim=256,
        pretrained=True,
        pooling=args.pooling,
    )

    trained_state_dict = torch.load(args.model_path, map_location="cpu")
    state_dict = OrderedDict()
    for key, value in trained_state_dict.items():
        key = key.replace("diff.0.bias", "diff.bias")
        key = key.replace("diff.0.weight", "diff.weight")
        state_dict[key] = value
    model.load_state_dict(state_dict=state_dict, strict=True)
    model.eval()

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    test(model=model, dataset=test_dataset, device=device)
    if device.type == "cuda":
        torch.cuda.empty_cache()

    prediction_dir = os.path.join(args.save_dir, "seg-prediction")
    predictions, gts = load_saved_predictions_and_labels(test_dataset, prediction_dir)
    metrics_path = os.path.join(args.save_dir, "metrics.txt")
    save_metrics(calculate_metrics(predictions, gts), metrics_path)
    print(f"Metrics saved to: {metrics_path}")
    return True


if __name__ == "__main__":
    args = parser.parse_args()
    cfg = OmegaConf.load(args.config)
    cfg.cam.bkg_score = args.bkg_score
    print(cfg)
    print(args)

    args.save_dir = os.path.join(args.save_dir, args.eval_set)
    os.makedirs(os.path.join(args.save_dir, "seg-prediction"), exist_ok=True)
    os.makedirs(os.path.join(args.save_dir, "seg-prediction-color"), exist_ok=True)
    main(config=cfg)
