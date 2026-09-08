"""Report ACWCD parameter counts without requiring pretrained checkpoint files.

Examples:
    conda run -n zyh_wscd python count_trainable_parameters.py
    conda run -n zyh_wscd python count_trainable_parameters.py --config configs/BCD.yaml
"""

import argparse
from pathlib import Path

import torch
import yaml

from models.model_ACWCD import ACWCD


PROJECT_ROOT = Path(__file__).resolve().parent
DINO_BACKBONES = {"dinov3_vith16plus", "dinovith+", "dinov3_vith+"}


def parameter_count(parameters):
    return sum(parameter.numel() for parameter in parameters)


def print_count(label, parameters):
    count = parameter_count(parameters)
    print(f"{label}: {count:,} ({count / 1_000_000:.4f} M)")
    return count


def build_model(config):
    backbone = config["backbone"]["config"]
    dino_model = None
    if backbone in DINO_BACKBONES:
        dino_repo = PROJECT_ROOT.parent / "dinov3"
        if not dino_repo.is_dir():
            raise FileNotFoundError(f"DINOv3 source directory not found: {dino_repo}")
        # Parameter shapes do not depend on checkpoint values.  This also avoids a download.
        dino_model = torch.hub.load(
            str(dino_repo), "dinov3_vith16plus", source="local", pretrained=False
        )

    return ACWCD(
        backbone=backbone,
        stride=config["backbone"].get("stride"),
        num_classes=config["dataset"]["num_classes"],
        embedding_dim=256,
        pretrained=False,
        pooling="gap",
        dino_model=dino_model,
    )


def main():
    parser = argparse.ArgumentParser(description="Count ACWCD trainable parameters.")
    parser.add_argument("--config", default="configs/LEVIR.yaml", help="Path relative to project root.")
    args = parser.parse_args()

    config_path = (PROJECT_ROOT / args.config).resolve()
    with config_path.open(encoding="utf-8") as config_file:
        config = yaml.safe_load(config_file)

    model = build_model(config)
    print(f"Config: {config_path}")
    print(f"Backbone: {config['backbone']['config']}")
    print_count("Total parameters", model.parameters())
    print_count("Trainable parameters (requires_grad=True)", (p for p in model.parameters() if p.requires_grad))
    print_count("Frozen parameters", (p for p in model.parameters() if not p.requires_grad))

    optimizer_parameters = [p for group in model.get_param_groups() for p in group]
    print_count("Parameters returned by get_param_groups()", optimizer_parameters)


if __name__ == "__main__":
    main()
