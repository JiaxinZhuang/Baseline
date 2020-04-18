"""ResNet.
"""

import torch
import torch.nn as nn
import torchvision
import torch.nn.functional as F


class ResNet50(nn.Module):
    """ResNet50.
        Modify renset50 here.
    """
    def __init__(self, num_classes=7, input_channel=3,
                 pretrained=True, bilinear=False):
        super(ResNet50, self).__init__()
        resnet50 = torchvision.models.resnet50(pretrained=pretrained)
        self.features = nn.Sequential(
            *list(resnet50.children())[:-2]
        )
        self.avgpool = resnet50.avgpool
        self.bilinear = bilinear
        if self.bilinear:
            self.classifier = nn.Linear(resnet50.inplanes ** 2, num_classes)
        else:
            self.classifier = nn.Linear(resnet50.inplanes, num_classes)

    def forward(self, inputs):
        out = self.features(inputs)
        out = self.avgpool(out)

        if self.bilinear:
            out = self.bilinear_layer(out)
        else:
            out = out.view(out.size(0), -1)
        out = self.classifier(out)
        return out

    def bilinear_layer(self, inputs):
        """Bilinear operation.
        Args:
            Inputs: [batch_size, channels, height, width]
        Returns:
            out: [batch_size, channels * channels]
        """
        inputs = F.adaptive_avg_pool2d(inputs, (1, 1))
        batch_size, channels, height, width = inputs.size()
        inputs = inputs.view(batch_size, channels, height * width)
        inputs_t = torch.transpose(inputs, 1, 2)
        out = torch.bmm(inputs, inputs_t) / (height * width)
        out = out.view(batch_size, -1)
        return out
