import torch
import torch.nn as nn
import torch.nn.functional as F
from .decoder import FPN
from .former import Block
import os


DEFAULT_DINO_CKPT_PATH = r"E:\zyh-dinov3-wcd\dino\dinov3_vith16plus_pretrain_lvd1689m-7c1da9a5.pth"


def load_dinov3_model(ckpt_path=None):
    """Load the local DINOv3 ViT-H+/16 model used by ACWCD."""
    dino_repo = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "dinov3"))
    ckpt_path = ckpt_path or os.environ.get("DINO_CKPT_PATH", DEFAULT_DINO_CKPT_PATH)
    if not os.path.isdir(dino_repo):
        raise FileNotFoundError(f"DINOv3 source directory not found: {dino_repo}")
    if not os.path.isfile(ckpt_path):
        raise FileNotFoundError(
            f"DINOv3 checkpoint not found: {ckpt_path}. "
            "Set DINO_CKPT_PATH or pass --dino_ckpt_path."
        )

    return torch.hub.load(
        dino_repo,
        "dinov3_vith16plus",
        source="local",
        weights=ckpt_path,
    )


class DINOv3FeaturePyramid(nn.Module):
    """Expose four DINOv3 ViT-H+/16 block outputs through the MS-Former FPN API."""

    def __init__(self, ckpt_path=None, out_channels=128, block_indices=(7, 15, 23, 31)):
        super().__init__()
        self.dino = load_dinov3_model(ckpt_path)
        self.patch_size = getattr(self.dino, "patch_size", 16)
        self.block_indices = tuple(block_indices)
        self.projections = nn.ModuleList(
            nn.Sequential(
                nn.Conv2d(self.dino.embed_dim, out_channels, kernel_size=1, bias=False),
                nn.BatchNorm2d(out_channels),
                nn.ReLU(inplace=True),
            )
            for _ in self.block_indices
        )

        # Keep the pretrained DINOv3 representation fixed, matching ACWCD.
        for parameter in self.dino.parameters():
            parameter.requires_grad = False
        self.dino.eval()
        self.register_state_dict_post_hook(self._remove_frozen_dino_weights)
        self.register_load_state_dict_pre_hook(self._restore_frozen_dino_weights)

    @staticmethod
    def _remove_frozen_dino_weights(module, state_dict, prefix, local_metadata):
        """Avoid embedding the fixed ViT-H+ checkpoint in every training snapshot."""
        dino_prefix = prefix + "dino."
        for key in [key for key in state_dict if key.startswith(dino_prefix)]:
            state_dict.pop(key)

    @staticmethod
    def _restore_frozen_dino_weights(module, state_dict, prefix, local_metadata,
                                     strict, missing_keys, unexpected_keys, error_msgs):
        """Strict checkpoint loading uses the DINO weights loaded at construction time."""
        for key, value in module.dino.state_dict().items():
            state_dict.setdefault(prefix + "dino." + key, value)

    def train(self, mode=True):
        super().train(mode)
        # WCDNet.train() propagates to children; DINO must remain an inference encoder.
        self.dino.eval()
        return self

    def forward(self, x):
        height, width = x.shape[-2:]
        if height % self.patch_size or width % self.patch_size:
            raise ValueError(
                f"DINOv3 ViT-H+/16 requires dimensions divisible by {self.patch_size}, "
                f"got {(height, width)}."
            )

        with torch.no_grad():
            features = self.dino.get_intermediate_layers(
                x, n=self.block_indices, reshape=True, norm=True
            )
        return [projection(feature.to(dtype=projection[0].weight.dtype))
                for feature, projection in zip(features, self.projections)]


class WCDNet(nn.Module):
    def __init__(self, patch_size=32, memory_length=128, depth=3, dino_ckpt_path=None):
        super(WCDNet, self).__init__()
        self.patch_size = patch_size
        self.decoder_channel = 128
        self.embedding_channel = 128
        self.memory_length = memory_length
        self.depth = depth
        # DINOv3's four selected transformer stages replace the ResNet-18 feature extractor.
        channels = [128, 128, 128, 128]
        self.context_encoder = DINOv3FeaturePyramid(
            ckpt_path=dino_ckpt_path,
            out_channels=channels[0],
        )
        self.fpn_net = FPN(channels, self.decoder_channel)
        # attention
        self.memory_tokens = nn.Embedding(self.memory_length, self.embedding_channel)
        self.pixel_feature_tokens = nn.Conv2d(self.decoder_channel, self.embedding_channel, kernel_size=1)
        self.attention = nn.ModuleList(
            [Block(dim=self.embedding_channel, num_heads=2, mlp_ratio=4) for i in range(self.depth)]
        )
        # mask
        self.mask_generation = nn.Conv2d(self.embedding_channel, 1, kernel_size=1)
        self.region_mask_generation = nn.ModuleList(
            [nn.Conv2d(self.embedding_channel, 1, kernel_size=1) for i in range(self.depth)]
        )

    def forward(self, x, gt=None):
        test_mode = gt is None
        size = x.size()[2:]
        # temporal difference information extraction
        t1 = x[:, 0:3, :, :]
        t2 = x[:, 3:6, :, :]
        t1_c2, t1_c3, t1_c4, t1_c5 = self.context_encoder(t1)
        t2_c2, t2_c3, t2_c4, t2_c5 = self.context_encoder(t2)
        #
        c5 = torch.abs(t1_c5 - t2_c5)
        c4 = torch.abs(t1_c4 - t2_c4)
        c3 = torch.abs(t1_c3 - t2_c3)
        c2 = torch.abs(t1_c2 - t2_c2)
        p_out = self.fpn_net(c2, c3, c4, c5)
        # attention
        pixel_feature_tokens = self.pixel_feature_tokens(p_out)
        B, C, H, W = pixel_feature_tokens.size()
        memory_tokens = self.memory_tokens.weight.unsqueeze(0).repeat(B, 1, 1)
        pixel_feature_tokens = pixel_feature_tokens.flatten(2).transpose(1, 2)
        region_mask = []
        for idx, a_block in enumerate(self.attention):
            pixel_feature_tokens, memory_tokens, mp = \
                a_block(pixel_feature_tokens, memory_tokens, H, W, self.patch_size)
            region_mask.append(self.region_mask_generation[idx](mp))

        pixel_feature_tokens = pixel_feature_tokens.reshape(B, H, W, -1).permute(0, 3, 1, 2)
        # mask
        mask = self.mask_generation(pixel_feature_tokens)
        change_mask = F.interpolate(mask, size=size, mode='bilinear', align_corners=True)
        change_mask = torch.sigmoid(change_mask)
        change_mask_aux = F.adaptive_max_pool2d(mask, (size[0] // self.patch_size, size[1] // self.patch_size))
        change_mask_aux = torch.sigmoid(change_mask_aux)
        region_mask = torch.cat(region_mask, dim=1)
        region_mask = torch.sigmoid(region_mask)

        if test_mode:
            return change_mask

        return change_mask, change_mask_aux, region_mask


def get_model(patch_size, memory_length, depth, dino_ckpt_path=None):
    model = WCDNet(patch_size, memory_length, depth, dino_ckpt_path=dino_ckpt_path)

    return model
