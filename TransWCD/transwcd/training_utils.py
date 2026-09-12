"""Standalone helpers used by TransWCD training and evaluation scripts."""

import numpy as np
import torch
import torch.nn.functional as F
import torchvision
import matplotlib.pyplot as plt


class AverageMeter:
    def __init__(self, *keys):
        self.__data = {key: [0.0, 0] for key in keys}

    def add(self, values):
        for key, value in values.items():
            if key not in self.__data:
                self.__data[key] = [0.0, 0]
            self.__data[key][0] += value
            self.__data[key][1] += 1

    def get(self, *keys):
        if len(keys) == 1:
            return self.__data[keys[0]][0] / self.__data[keys[0]][1]
        return tuple(self.__data[key][0] / self.__data[key][1] for key in keys)

    def pop(self, key=None):
        if key is None:
            for value in self.__data.values():
                value[:] = [0.0, 0]
            return None
        value = self.get(key)
        self.__data[key] = [0.0, 0]
        return value


class PolyWarmupAdamW(torch.optim.AdamW):
    def __init__(self, params, lr, weight_decay, betas, warmup_iter=None,
                 max_iter=None, warmup_ratio=None, power=None):
        super().__init__(params, lr=lr, betas=betas, weight_decay=weight_decay, eps=1e-8)
        self.global_step = 0
        self.warmup_iter = warmup_iter
        self.warmup_ratio = warmup_ratio
        self.max_iter = max_iter
        self.power = power
        self.__init_lr = [group["lr"] for group in self.param_groups]

    def step(self, closure=None):
        if self.global_step < self.warmup_iter:
            lr_mult = 1 - (1 - self.global_step / self.warmup_iter) * (1 - self.warmup_ratio)
        elif self.global_step < self.max_iter:
            lr_mult = (1 - self.global_step / self.max_iter) ** self.power
        else:
            lr_mult = None

        if lr_mult is not None:
            for index, group in enumerate(self.param_groups):
                group["lr"] = self.__init_lr[index] * lr_mult
        super().step(closure)
        self.global_step += 1


def cam_to_label(cam, cls_label, img_box=None, ignore_mid=False, cfg=None):
    del ignore_mid
    cam_value, pseudo_label = cam.max(dim=1, keepdim=False)
    pseudo_label += 1
    pseudo_label[cam_value <= cfg.cam.bkg_score] = 0
    if img_box is None:
        return pseudo_label

    result = torch.ones_like(pseudo_label)
    for index, coord in enumerate(img_box):
        result[index, coord[0]:coord[1], coord[2]:coord[3]] = pseudo_label[
            index, coord[0]:coord[1], coord[2]:coord[3]
        ]
    return cam, result


def multi_scale_cam(model, inputs_A, inputs_B, scales):
    batch, _, height, width = inputs_A.shape
    with torch.no_grad():
        def run_cam(image_a, image_b):
            image_a = torch.cat([image_a, image_a.flip(-1)], dim=0)
            image_b = torch.cat([image_b, image_b.flip(-1)], dim=0)
            cam = model(image_a, image_b, cam_only=True)
            cam = F.interpolate(cam, size=(height, width), mode="bilinear", align_corners=False)
            return torch.max(cam[:batch], cam[batch:].flip(-1))

        cams = [F.relu(run_cam(inputs_A, inputs_B))]
        for scale in scales:
            if scale != 1.0:
                size = (int(scale * height), int(scale * width))
                cams.append(F.relu(run_cam(
                    F.interpolate(inputs_A, size=size, mode="bilinear", align_corners=False),
                    F.interpolate(inputs_B, size=size, mode="bilinear", align_corners=False),
                )))
        cam = torch.stack(cams, dim=0).sum(dim=0)
        cam = cam + F.adaptive_max_pool2d(-cam, (1, 1))
        return cam / (F.adaptive_max_pool2d(cam, (1, 1)) + 1e-5)


def scores(label_trues, label_preds, num_classes=2):
    hist = np.zeros((num_classes, num_classes))
    for label_true, label_pred in zip(label_trues, label_preds):
        mask = (label_true >= 0) & (label_true < num_classes)
        hist += np.bincount(
            num_classes * label_true[mask].astype(int) + label_pred[mask],
            minlength=num_classes ** 2,
        ).reshape(num_classes, num_classes)
    eps = np.finfo(np.float32).eps
    recall = np.diag(hist) / (hist.sum(axis=1) + eps)
    precision = np.diag(hist) / (hist.sum(axis=0) + eps)
    f1 = 2 * recall * precision / (recall + precision + eps)
    iou = np.diag(hist) / (hist.sum(axis=1) + hist.sum(axis=0) - np.diag(hist) + eps)
    return {
        "OA": np.diag(hist).sum() / (hist.sum() + eps),
        "f1": dict(zip(range(num_classes), f1)),
        "precision": dict(zip(range(num_classes), precision)),
        "iou": dict(zip(range(num_classes), iou)),
        "recall": dict(zip(range(num_classes), recall)),
    }


def _denormalize_img(images, mean=(123.675, 116.28, 103.53), std=(58.395, 57.12, 57.375)):
    result = torch.zeros_like(images)
    for channel, (channel_mean, channel_std) in enumerate(zip(mean, std)):
        result[:, channel] = images[:, channel] * channel_std + channel_mean
    return result.type(torch.uint8)


def tensorboard_image(imgs, cam):
    images = _denormalize_img(imgs)
    grid_imgs = torchvision.utils.make_grid(images, nrow=4)
    cam = F.interpolate(cam, size=images.shape[2:], mode="bilinear", align_corners=False).cpu()
    heatmap = plt.get_cmap("jet")(cam.max(dim=1)[0].numpy())[:, :, :, :3] * 255
    cam_image = torch.from_numpy(heatmap).permute(0, 3, 1, 2) * 0.5 + images.cpu() * 0.5
    return grid_imgs, torchvision.utils.make_grid(cam_image.type(torch.uint8), nrow=4)


def tensorboard_label(labels):
    labels = np.squeeze(labels)
    cmap = np.zeros((256, 3), dtype=np.uint8)
    for index in range(256):
        value = index
        for bit in range(8):
            cmap[index, 0] |= ((value & 1) != 0) << (7 - bit)
            cmap[index, 1] |= ((value & 2) != 0) << (7 - bit)
            cmap[index, 2] |= ((value & 4) != 0) << (7 - bit)
            value >>= 3
    colored = torch.from_numpy(cmap[labels.astype(np.int16)]).permute(0, 3, 1, 2)
    return torchvision.utils.make_grid(colored, nrow=4)
