import torch
import torch.nn as nn
import torch.nn.functional as F
import os
from . import mix_transformer
from .seg_head import SimpleSeg
import numpy as np


DEFAULT_DINO_CKPT_PATH = r"E:\zyh-dinov3-wcd\dino\dinov3_vith16plus_pretrain_lvd1689m-7c1da9a5.pth"


def load_dinov3_model(device, ckpt_path=None):
    """Load the local DINOv3 ViT-H+/16 backbone using the reference project's API."""
    repo_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "dinov3"))
    ckpt_path = ckpt_path or os.environ.get("DINO_CKPT_PATH", DEFAULT_DINO_CKPT_PATH)
    if not os.path.isfile(ckpt_path):
        raise FileNotFoundError(
            f"DINOv3 checkpoint not found: {ckpt_path}. "
            "Set DINO_CKPT_PATH or pass dino_ckpt_path explicitly."
        )

    model = torch.hub.load(
        repo_dir,
        "dinov3_vith16plus",
        source="local",
        weights=ckpt_path,
    )
    return model.to(device).eval()


class DINOv3Encoder(nn.Module):
    """Adapt DINOv3 patch tokens to the feature/attention interface used by ACWCD."""

    def __init__(self, device, dino_model=None, dino_ckpt_path=None, out_channels=512):
        super().__init__()
        self.dino = dino_model if dino_model is not None else load_dinov3_model(device, dino_ckpt_path)
        self.embed_dim = self.dino.embed_dim
        self.patch_size = getattr(self.dino, "patch_size", 16)
        self.proj = nn.Conv2d(self.embed_dim, out_channels, kernel_size=1, bias=False)
        self.embed_dims = [out_channels] * 4

        # DINOv3 is used as a frozen feature extractor, as in the reference project.
        for parameter in self.dino.parameters():
            parameter.requires_grad = False

    def forward(self, x):
        with torch.no_grad():
            features = self.dino.forward_features(x)
            patch_tokens = features["x_norm_patchtokens"]

        batch_size, token_count, _ = patch_tokens.shape
        height = x.shape[-2] // self.patch_size
        width = x.shape[-1] // self.patch_size
        if height * width != token_count:
            raise ValueError(
                f"DINOv3 returned {token_count} patch tokens for input {tuple(x.shape[-2:])}; "
                f"expected a {height}x{width} patch grid."
            )

        patch_map = patch_tokens.transpose(1, 2).contiguous().view(
            batch_size, self.embed_dim, height, width
        ).to(dtype=self.proj.weight.dtype)
        c4 = self.proj(patch_map)

        # Preserve ACWCD's expected [B, 16, N, N] attention input (two 8-head maps).
        normalized_tokens = F.normalize(patch_tokens.float(), dim=-1)
        attention = torch.softmax(normalized_tokens @ normalized_tokens.transpose(1, 2), dim=-1)
        attention = attention.unsqueeze(1).expand(-1, 8, -1, -1).contiguous()
        return [c4, c4, c4, c4], [attention, attention]


class ACWCD(nn.Module):
    def __init__(self, backbone, num_classes=None, embedding_dim=256, stride=None, pretrained=None,
                 pooling=None, dino_ckpt_path=None, dino_model=None):
        super().__init__()
        self.num_classes = num_classes
        self.embedding_dim = embedding_dim
        self.feature_strides = [4, 8, 16, 32]
        self.stride = stride
        if backbone in {"dinov3_vith16plus", "dinovith+", "dinov3_vith+"}:
            self.encoder = DINOv3Encoder(
                device=torch.device("cuda" if torch.cuda.is_available() else "cpu"),
                dino_model=dino_model,
                dino_ckpt_path=dino_ckpt_path,
                out_channels=512,
            )
        else:
            self.encoder = getattr(mix_transformer, backbone)(stride=self.stride)
        self.in_channels = self.encoder.embed_dims
        c1_in_channels, c2_in_channels, c3_in_channels, c4_in_channels = self.in_channels

        if pretrained and not isinstance(self.encoder, DINOv3Encoder):
            state_dict = torch.load('pretrained/' + backbone + '.pth')
            state_dict.pop('head.weight')
            state_dict.pop('head.bias')
            self.encoder.load_state_dict(state_dict, )

        if pooling == "gmp":
            self.pooling = F.adaptive_max_pool2d
        elif pooling == "gap":
            self.pooling = F.adaptive_avg_pool2d

        self.classifier = nn.Conv2d(in_channels=self.in_channels[3], out_channels=self.num_classes-1, kernel_size=1,
                                    bias=False)

        self.dropout = nn.Dropout2d(0.1)
        self.linear_pred = nn.Conv2d(self.in_channels[3], self.num_classes, kernel_size=1)

        self.decoder = SimpleSeg(feature_strides=self.feature_strides, in_channels=self.in_channels, embedding_dim=self.embedding_dim, num_classes=self.num_classes)

        self.attn_proj = nn.Conv2d(in_channels=16, out_channels=1, kernel_size=1, bias=True)
        nn.init.kaiming_normal_(self.attn_proj.weight, a=np.sqrt(5), mode="fan_out")

        self.diff_c4 = conv_diff_d(in_channels=2 * c4_in_channels, out_channels=c4_in_channels)
        self.diff_at = conv_diff_d(in_channels=32, out_channels=16)

    def get_param_groups(self):

        param_groups = [[], [], [], []]  # backbone; backbone_norm; cls_head; seg_head;

        for name, param in list(self.encoder.named_parameters()):

            if "norm" in name:
                param_groups[1].append(param)
            else:
                param_groups[0].append(param)

        param_groups[2].append(self.classifier.weight)
        param_groups[2].append(self.attn_proj.weight)
        param_groups[2].append(self.attn_proj.bias)

        for param in list(self.decoder.parameters()):
            param_groups[3].append(param)

        return param_groups

    def forward(self, x1, x2, cam_only=False, seg_detach=True,):
        _x1, _attns1 = self.encoder(x1)
        _x2, _attns2 = self.encoder(x2)

        _, _, _, _c4_1 = _x1
        _, _, _, _c4_2 = _x2

        ### integration ###

        _c4 = torch.absolute(_c4_1 - _c4_2)

        seg = self.decoder(_c4)

        attn_cat1 = torch.cat(_attns1[-2:], dim=1)  # .detach()
        attn_cat2 = torch.cat(_attns2[-2:], dim=1)  # .detach()

        _attns = torch.absolute(attn_cat1 - attn_cat2)

        attn_cat = _attns + _attns.permute(0, 1, 3, 2)
        change_attn = self.attn_proj(attn_cat)
        change_attn = torch.sigmoid(change_attn)[:, 0, ...]

        if cam_only:
            cam_s4 = F.conv2d(_c4, self.classifier.weight).detach()
            return cam_s4, change_attn

        cls = self.pooling(_c4, (1, 1))
        cls = self.classifier(cls)
        cls = cls.view(-1, self.num_classes-1)

        return cls, seg, change_attn

if __name__ == "__main__":
    pretrained_weights = torch.load('pretrained/mit_b1.pth')
    acwcd = ACWCD('mit_b1', num_classes=2, embedding_dim=256, pretrained=True)
    acwcd._param_groups()
    dummy_input = torch.rand(2, 3, 256, 256)
    acwcd(dummy_input)

def conv_diff_d(in_channels, out_channels):
    return nn.Sequential(
        nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
        nn.ReLU(),
    )

