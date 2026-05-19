import torch
import torch.nn as nn
from torchvision.models import mobilenet_v2, MobileNet_V2_Weights


class MobileNetV2_SurFake(nn.Module):
    def __init__(self, num_classes=2, pretrained=True):
        super().__init__()

        weights = MobileNet_V2_Weights.IMAGENET1K_V1 if pretrained else None
        self.model = mobilenet_v2(weights=weights)

        old_conv = self.model.features[0][0]

        new_conv = nn.Conv2d(
            in_channels=6,
            out_channels=old_conv.out_channels,
            kernel_size=old_conv.kernel_size,
            stride=old_conv.stride,
            padding=old_conv.padding,
            bias=False
        )

        with torch.no_grad():
            
            new_conv.weight[:, 0:3, :, :] = old_conv.weight

            
            mean_weight = old_conv.weight.mean(dim=1, keepdim=True)
            new_conv.weight[:, 3:6, :, :] = mean_weight.repeat(1, 3, 1, 1)

        self.model.features[0][0] = new_conv

        in_features = self.model.classifier[1].in_features
        self.model.classifier[1] = nn.Linear(in_features, num_classes)

    def forward(self, x):
        return self.model(x)