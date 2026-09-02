import torch
from torchvision import models

def count_params(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

models_list = [
    ("SqueezeNet1_1", models.squeezenet1_1(pretrained=False)),
    ("ResNet18", models.resnet18(pretrained=False)),
    ("DenseNet161", models.densenet161(pretrained=False)),
]

print("Model的可训练参数数量 (单位: M)")
print("-" * 40)
for name, model in models_list:
    params_m = count_params(model) / 1e6
    print(f"{name}: {params_m:.2f} M")
