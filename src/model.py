"""Network.

    Jiaxin Zhuang, lincolnz9511@gmail.com
"""
# -*- coding: utf-8 -*-

import sys
import torch.nn as nn
from torchsummary import summary

from models.NIN import NIN
from models.ResNet import ResNet50


class Network(nn.Module):
    """Network interface.
    """
    def __init__(self, backbone="ResNet50", input_channel=3,
                 num_classes=200, pretrained=True, bilinear=False,
                 _print=None):
        super(Network, self).__init__()
        self._print = _print if _print is not None else print
        self.input_channel = input_channel
        if backbone == "ResNet50":
            self._print("Backbone: {}, with pretrained {}".format(backbone,
                                                                  pretrained))
            self.model = ResNet50(num_classes=num_classes,
                                  input_channel=input_channel,
                                  pretrained=pretrained, bilinear=bilinear)
        elif backbone == "NIN":
            self._print("Backbone: {}, with pretrained {}".format(backbone,
                                                                  pretrained))
            self.model = NIN(num_classes=num_classes,
                             input_channel=input_channel,
                             bilinear=bilinear)
        # elif backbone == "vgg16":
        #     self._print("Backbone: {}, with pretrained {}".format(backbone,
        #                                                           pretrained))
        #     self.model = torchvision.models.vgg16(pretrained=pretrained)
        #     self.model = (num_classes=num_classes,
        #                           input_channel=input_channel,
        #                           pretrained=pretrained, bilinear=bilinear)
        else:
            self._print("Need valid model backbone.")
            sys.exit(-1)

    def forward(self, inputs):
        return self.model(inputs)

    def print_model(self):
        """Print model here. Default input is 224 x 224 x 3
        """
        input_size = (self.input_channel, 224, 224)
        summary(self.model, input_size)


if __name__ == "__main__":
    model = Network(backbone="ResNet50", num_classes=200,
                    pretrained=True).cuda()
    model.print_model()
    model = Network(backbone="NIN", num_classes=10).cuda()
    model.print_model()
