"""Standalone accuracy evaluation for saved binary change-detection masks.

Predictions are read from a folder of PNG masks. Ground-truth labels come from
the dataset section of the project config (``dataset.root_dir``/``label``, or
``dataset.label_dir`` when present); ``--label_dir`` overrides both.

Prediction and label masks are both treated as binary, so bool, 0/1 and 0/255
encodings give the same result.

Example:
    conda run -n zyh_wscd python accuray.py ^
        --config configs/WHU.yaml ^
        --prediction_dir results/WHU/test/prediction
"""

import argparse
import os

import numpy as np
from omegaconf import OmegaConf
from PIL import Image

MASK_SUFFIX = ".png"
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))


def as_binary(mask):
    """Reduce a bool, 0/1 or 0/255 mask to a uint8 {0, 1} array."""
    mask = np.asarray(mask)
    if mask.ndim == 3:
        mask = mask[..., 0]
    return (mask > 0).astype(np.uint8)


def read_mask(path):
    """Read one mask image as a binary uint8 array."""
    return as_binary(np.asarray(Image.open(path).convert("L")))


def align_mask(mask, target_shape):
    """Nearest-neighbour resize a mask onto ``target_shape`` when needed."""
    if mask.shape == target_shape:
        return mask, False
    height, width = target_shape
    rows = np.minimum((np.arange(height) * mask.shape[0]) // height, mask.shape[0] - 1)
    cols = np.minimum((np.arange(width) * mask.shape[1]) // width, mask.shape[1] - 1)
    return mask[rows][:, cols], True


def mask_key(name):
    """Folder-independent key used to match prediction and label files.

    Strips any repeated ".png" suffix, so "test_000024.png.png" and
    "test_000024.png" share the key "test_000024".
    """
    key = os.path.basename(name).lower()
    while key.endswith(".png"):
        key = key[:-4]
    return key


def list_masks(directory):
    return sorted(
        name for name in os.listdir(directory) if name.lower().endswith(MASK_SUFFIX)
    )


def resolve_split_names(name_list_dir, split):
    """Return the mask keys listed for ``split``, or None when unavailable."""
    if not name_list_dir or not split:
        return None
    candidates = [name_list_dir]
    if not os.path.isabs(name_list_dir):
        candidates.append(os.path.join(SCRIPT_DIR, name_list_dir))
    for base in candidates:
        list_path = os.path.join(base, "{}.txt".format(split))
        if os.path.isfile(list_path):
            names = np.atleast_1d(np.loadtxt(list_path, dtype=str))
            if names.ndim == 2:
                names = names[:, 0]
            return [mask_key(name) for name in names]
    return None


def collect_mask_pairs(prediction_dir, label_dir, names=None):
    """Load same-named prediction/label pairs.

    Returns ``(predictions, labels, matched_keys, info)`` where ``info`` records
    missing and resized masks so the report can explain the sample count.
    """
    prediction_files = {mask_key(name): name for name in list_masks(prediction_dir)}
    label_files = {mask_key(name): name for name in list_masks(label_dir)}

    if names is None:
        names = sorted(set(prediction_files) & set(label_files))

    predictions, labels, matched = [], [], []
    missing_labels, missing_predictions, resized = [], [], []
    for name in names:
        key = mask_key(name)
        if key not in label_files:
            missing_labels.append(key)
            continue
        if key not in prediction_files:
            missing_predictions.append(key)
            continue
        label = read_mask(os.path.join(label_dir, label_files[key]))
        prediction, was_resized = align_mask(
            read_mask(os.path.join(prediction_dir, prediction_files[key])), label.shape
        )
        predictions.append(prediction)
        labels.append(label)
        matched.append(key)
        if was_resized:
            resized.append(key)

    if not matched:
        raise RuntimeError(
            "No matching prediction/label pairs were found.\n"
            "  predictions: {}\n  labels:      {}".format(prediction_dir, label_dir)
        )
    info = {
        "missing_labels": missing_labels,
        "missing_predictions": missing_predictions,
        "resized": resized,
    }
    return predictions, labels, matched, info


def safe_divide(numerator, denominator):
    numerator = np.asarray(numerator, dtype=np.float64)
    denominator = np.asarray(denominator, dtype=np.float64)
    with np.errstate(divide="ignore", invalid="ignore"):
        return numerator / denominator


def confusion_matrix(predictions, labels, num_classes=2):
    hist = np.zeros((num_classes, num_classes), dtype=np.float64)
    for prediction, label in zip(predictions, labels):
        prediction = as_binary(prediction).reshape(-1)
        label = as_binary(label).reshape(-1)
        if prediction.shape != label.shape:
            raise ValueError(
                "Mask shape mismatch: prediction={}, label={}".format(
                    prediction.shape, label.shape
                )
            )
        valid = label < num_classes
        hist += np.bincount(
            num_classes * label[valid].astype(np.int64)
            + prediction[valid].astype(np.int64),
            minlength=num_classes ** 2,
        ).reshape(num_classes, num_classes)
    return hist


def calculate_metrics(predictions, labels, num_classes=2):
    """Reference nine metrics plus Cohen's kappa, from one confusion matrix."""
    hist = confusion_matrix(predictions, labels, num_classes)
    total = hist.sum()
    if not total:
        raise ValueError("No valid pixels were available for metric calculation.")

    eps = np.finfo(np.float64).eps
    true_negative, false_positive = hist[0]
    false_negative, true_positive = hist[1]

    iou = safe_divide(
        np.diag(hist), hist.sum(axis=1) + hist.sum(axis=0) - np.diag(hist)
    )
    acc = np.diag(hist).sum() / total
    acc_cls = np.nanmean(safe_divide(np.diag(hist), hist.sum(axis=1)))
    frequency = hist.sum(axis=1) / total
    fwavacc = float((frequency * np.nan_to_num(iou)).sum())

    precision = true_positive / (true_positive + false_positive + eps)
    recall = true_positive / (true_positive + false_negative + eps)
    f1_score = 2 * precision * recall / (precision + recall + eps)
    accuracy = (true_negative + true_positive) / total
    expected = (hist.sum(axis=0) * hist.sum(axis=1)).sum() / (total ** 2 + eps)
    kappa = (acc - expected) / (1 - expected + eps)

    return {
        "confusion_matrix": hist,
        "pixels": total,
        "ground_truth_foreground": hist[1].sum(),
        "prediction_foreground": hist[:, 1].sum(),
        "acc": acc,
        "acc_cls": acc_cls,
        "iou": iou,
        "miou": np.nanmean(iou),
        "fwavacc": fwavacc,
        "class_accuracy": precision,
        "class_recall": recall,
        "accuracy": accuracy,
        "f1_score": f1_score,
        "kappa": kappa,
    }


def _format_array(values):
    return "[{}]".format(", ".join("{:.4f}".format(value) for value in values))


def _format_count(value):
    return "{:,}".format(int(value))


def format_report(metrics, info=None):
    """Build the console/file report for a metrics dictionary."""
    lines = [
        "=" * 62,
        "Binary Change Detection Accuracy",
        "=" * 62,
    ]
    for label, value in info or []:
        lines.append("{:<30}{}".format(label + ":", value))
    lines.append("-" * 62)
    lines.append("Confusion matrix [ground truth rows, prediction columns]:")
    lines.append(
        np.array2string(metrics["confusion_matrix"], precision=0, suppress_small=True)
    )
    lines.append(
        "Pixels evaluated:             {}".format(_format_count(metrics["pixels"]))
    )
    lines.append(
        "Ground-truth foreground:      {}".format(
            _format_count(metrics["ground_truth_foreground"])
        )
    )
    lines.append(
        "Prediction foreground:        {}".format(
            _format_count(metrics["prediction_foreground"])
        )
    )
    lines.append("-" * 62)
    lines.append("Overall Accuracy:             {:.4f}".format(metrics["acc"]))
    lines.append("Class Average Accuracy:       {:.4f}".format(metrics["acc_cls"]))
    lines.append("IoU per class:                {}".format(_format_array(metrics["iou"])))
    lines.append("Mean IoU:                     {:.4f}".format(metrics["miou"]))
    lines.append("Frequency Weighted Accuracy:  {:.4f}".format(metrics["fwavacc"]))
    lines.append(
        "Class Precision:              {:.4f}".format(metrics["class_accuracy"])
    )
    lines.append("Class Recall:                 {:.4f}".format(metrics["class_recall"]))
    lines.append("Accuracy:                     {:.4f}".format(metrics["accuracy"]))
    lines.append("F1 Score:                     {:.4f}".format(metrics["f1_score"]))
    lines.append("Kappa:                        {:.4f}".format(metrics["kappa"]))
    lines.append("=" * 62)

    if not metrics["ground_truth_foreground"]:
        lines.append(
            "WARNING: the ground-truth masks contain no foreground pixels, "
            "so recall and F1 are undefined."
        )
    if not metrics["prediction_foreground"]:
        lines.append(
            "WARNING: every prediction mask is empty, so the model produced no "
            "changed pixels."
        )
    return "\n".join(lines)


def save_metrics(metrics, output_path, info=None):
    """Print the report and store the same text as UTF-8."""
    report = format_report(metrics, info=info)
    print(report, flush=True)
    with open(output_path, "w", encoding="utf-8") as file:
        file.write(report + "\n")
    return report


def resolve_label_dir(cfg, override=None):
    if override:
        return override
    dataset_cfg = cfg.get("dataset", {}) if hasattr(cfg, "get") else {}
    label_dir = dataset_cfg.get("label_dir")
    if label_dir:
        return str(label_dir)
    root_dir = dataset_cfg.get("root_dir")
    if root_dir:
        return os.path.join(str(root_dir), "label")
    raise SystemExit(
        "Could not resolve the label folder: pass --label_dir or add "
        "dataset.root_dir to the config."
    )


def build_parser():
    parser = argparse.ArgumentParser(
        description="Compute change-detection accuracy from saved prediction masks"
    )
    parser.add_argument(
        "--config", default="configs/WHU.yaml", type=str,
        help="project config used to locate the ground-truth labels",
    )
    parser.add_argument(
        "--prediction_dir", "--predict_folder", dest="prediction_dir",
        default="results/WHU/test/prediction", type=str,
        help="folder containing the predicted binary PNG masks",
    )
    parser.add_argument(
        "--label_dir", "--label_folder", dest="label_dir", default=None, type=str,
        help="ground-truth label folder; defaults to dataset.root_dir/label",
    )
    parser.add_argument(
        "--eval_set", default="test", type=str,
        help="split listed in dataset.name_list_dir; empty string evaluates every "
             "matching mask",
    )
    parser.add_argument(
        "--num_images", default=None, type=int,
        help="evaluate only the first N masks (default: all)",
    )
    parser.add_argument(
        "--output", default=None, type=str,
        help="txt report path; defaults to <prediction_dir>/../metrics.txt",
    )
    return parser


def main(argv=None):
    args = build_parser().parse_args(argv)
    cfg = OmegaConf.load(args.config)

    prediction_dir = os.path.abspath(args.prediction_dir)
    label_dir = os.path.abspath(resolve_label_dir(cfg, args.label_dir))
    if not os.path.isdir(prediction_dir):
        raise SystemExit("Prediction folder not found: {}".format(prediction_dir))
    if not os.path.isdir(label_dir):
        raise SystemExit(
            "Label folder not found: {}\n"
            "Pass --label_dir to point at the converted dataset labels.".format(
                label_dir
            )
        )

    dataset_cfg = cfg.get("dataset", {}) if hasattr(cfg, "get") else {}
    names = resolve_split_names(dataset_cfg.get("name_list_dir"), args.eval_set)
    if names is None:
        print(
            "Name list for split '{}' was not found; evaluating every matching "
            "mask instead.".format(args.eval_set),
            flush=True,
        )
    else:
        names = names[: args.num_images] if args.num_images else names

    predictions, labels, matched, pair_info = collect_mask_pairs(
        prediction_dir, label_dir, names
    )
    if args.num_images and names is None:
        predictions = predictions[: args.num_images]
        labels = labels[: args.num_images]
        matched = matched[: args.num_images]

    info = [
        ("Config", args.config),
        ("Predictions", prediction_dir),
        ("Labels", label_dir),
        ("Split", args.eval_set or "all masks"),
        ("Matched images", len(matched)),
    ]
    if args.num_images:
        info.append(("Image limit", args.num_images))
    if pair_info["missing_labels"]:
        info.append(("Missing labels", len(pair_info["missing_labels"])))
    if pair_info["missing_predictions"]:
        info.append(("Missing predictions", len(pair_info["missing_predictions"])))
    if pair_info["resized"]:
        info.append(("Resized predictions", len(pair_info["resized"])))
    for label, key in (
        ("Missing label examples", "missing_labels"),
        ("Missing prediction examples", "missing_predictions"),
    ):
        examples = pair_info[key][:5]
        if examples:
            info.append((label, ", ".join(examples)))

    metrics = calculate_metrics(predictions, labels)

    output_path = args.output or os.path.join(
        os.path.dirname(prediction_dir), "metrics.txt"
    )
    save_metrics(metrics, output_path, info=info)
    print("Metrics saved to: {}".format(output_path), flush=True)
    return metrics


if __name__ == "__main__":
    main()
