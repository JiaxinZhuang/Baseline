"""Network.

    Jiaxin Zhuang, lincolnz9511@gmail.com
"""
# -*- coding: utf-8 -*-

import sys
import torch.nn as nn
import torchvision
from torchsummary import summary


class Network(nn.Module):
    """Network interface.
    """
    def __init__(self, backbone="ResNet50", num_classes=200, pretrained=True,
                 _print=None):
        super(Network, self).__init__()
        self._print = _print if _print is not None else print
        if backbone == "ResNet50":
            self._print("Backbone: {}, with pretrained {}".format(backbone,
                                                                  pretrained))
            self.model = ResNet50(num_classes=num_classes,
                                  pretrained=pretrained)
        else:
            self._print("Need valid model backbone.")
            sys.exit(-1)

    def forward(self, inputs):
        return self.model(inputs)

    def print_model(self):
        """Print model here. Default input is 224 x 224 x 3
        """
        input_size = (3, 224, 224)
        summary(self.model, input_size)


class ResNet50(nn.Module):
    """ResNet50.
        Modify renset50 here.
    """
    def __init__(self, num_classes, pretrained):
        super(ResNet50, self).__init__()
        model = torchvision.models.resnet50(pretrained=pretrained)
        self.features = nn.Sequential(
            *list(model.children())[:-2]
        )
        self.avgpool = model.avgpool
        self.classifier = nn.Linear(model.inplanes, num_classes)

    def forward(self, inputs):
        out = self.features(inputs)
        out = self.avgpool(out)
        out = out.view(out.size(0), -1)
        out = self.classifier(out)
        return out


if __name__ == "__main__":
    model = Network(backbone="ResNet50", num_classes=200, pretrained=True)
    model.print_model()
