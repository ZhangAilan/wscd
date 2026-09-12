"""DINOv3 backbone adapter for the original TransWCD model."""

import os
from pathlib import Path

import torch
import torch.nn as nn


DEFAULT_DINO_CKPT_PATH = r"E:\zyh-dinov3-wcd\dino\dinov3_vith16plus_pretrain_lvd1689m-7c1da9a5.pth"


def load_dinov3_model(ckpt_path=None):
    """Load the local DINOv3 ViT-H+/16 checkpoint."""
    repo_dir = Path(__file__).resolve().parents[3] / "dinov3"
    ckpt_path = ckpt_path or os.environ.get("DINO_CKPT_PATH", DEFAULT_DINO_CKPT_PATH)
    if not repo_dir.is_dir():
        raise FileNotFoundError(f"DINOv3 source directory not found: {repo_dir}")
    if not os.path.isfile(ckpt_path):
        raise FileNotFoundError(
            f"DINOv3 checkpoint not found: {ckpt_path}. "
            "Set DINO_CKPT_PATH or pass dino_ckpt_path."
        )
    return torch.hub.load(
        str(repo_dir), "dinov3_vith16plus", source="local", weights=ckpt_path
    )


class DINOv3Backbone(nn.Module):
    """Expose DINO patch features through TransWCD's four-stage interface."""

    def __init__(self, ckpt_path=None, block_indices=(7, 15, 23, 31), freeze=True):
        super().__init__()
        self.dino = load_dinov3_model(ckpt_path)
        self.patch_size = getattr(self.dino, "patch_size", 16)
        self.block_indices = tuple(block_indices)
        self.embed_dims = [self.dino.embed_dim] * 4
        self.strides = [self.patch_size] * 4
        self.freeze = freeze

        if freeze:
            for parameter in self.dino.parameters():
                parameter.requires_grad = False
            self.dino.eval()
            self.register_state_dict_post_hook(self._remove_dino_weights)
            self.register_load_state_dict_pre_hook(self._restore_dino_weights)

    @staticmethod
    def _remove_dino_weights(module, state_dict, prefix, local_metadata):
        dino_prefix = prefix + "dino."
        for key in [key for key in state_dict if key.startswith(dino_prefix)]:
            state_dict.pop(key)

    @staticmethod
    def _restore_dino_weights(module, state_dict, prefix, local_metadata,
                               strict, missing_keys, unexpected_keys, error_msgs):
        for key, value in module.dino.state_dict().items():
            state_dict.setdefault(prefix + "dino." + key, value)

    def train(self, mode=True):
        super().train(mode)
        if self.freeze:
            self.dino.eval()
        return self

    def forward(self, x):
        height, width = x.shape[-2:]
        if height % self.patch_size or width % self.patch_size:
            raise ValueError(
                f"DINOv3 ViT-H+/16 requires dimensions divisible by {self.patch_size}, "
                f"got {(height, width)}"
            )
        context = torch.no_grad() if self.freeze else torch.enable_grad()
        with context:
            features = self.dino.get_intermediate_layers(
                x, n=self.block_indices, reshape=True, norm=True
            )
        return list(features)
