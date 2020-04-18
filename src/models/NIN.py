"""Network in network.

    Jiaxin Zhuang, lincolnz9511@gmail.com
"""
# -*- coding: utf-8 -*-

import torch.nn as nn
from torchsummary import summary


class NIN(nn.Module):
    """Network in Network.
    """
    def __init__(self, num_classes=10, input_channel=3, bilinear=False):
        super(NIN, self).__init__()
        self.num_classes = num_classes
        self.model = nn.Sequential(
            nn.Conv2d(input_channel, 192, kernel_size=5, stride=1, padding=2),
            nn.ReLU(inplace=True),
            nn.Conv2d(192, 160, kernel_size=1, stride=1, padding=0),
            nn.ReLU(inplace=True),
            nn.Conv2d(160, 96, kernel_size=1, stride=1, padding=0),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
            nn.Dropout(0.5),

            nn.Conv2d(96, 192, kernel_size=5, stride=1, padding=2),
            nn.ReLU(inplace=True),
            nn.Conv2d(192, 192, kernel_size=1, stride=1, padding=0),
            nn.ReLU(inplace=True),
            nn.Conv2d(192, 192, kernel_size=1, stride=1, padding=0),
            nn.ReLU(inplace=True),
            nn.AvgPool2d(kernel_size=3, stride=2, padding=1),
            nn.Dropout(0.5),

            nn.Conv2d(192, 192, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(192, 192, kernel_size=1, stride=1, padding=0),
            nn.ReLU(inplace=True),
            nn.Conv2d(192, self.num_classes, kernel_size=1, stride=1,
                      padding=0),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d(output_size=1)
        )
        self._init()

    def forward(self, inputs):
        out = self.model(inputs)
        out = out.view(out.size(0), self.num_classes)
        return out

    def _init(self):
        print("Initialization->>>>")
        for m in self.model.modules():
            if isinstance(m, nn.Conv2d):
                m.weight.data.normal_(0, 0.05)
                m.bias.data.zero_()

    def parameters(self):
        """Override.
            Adjust Learning rate
        """
        base_lr = 1e-1
        params = []

        for key, value in self.named_parameters():
            if key == "model.20.weight":
                params += [{"params": [value], "lr": 0.1 * base_lr,
                            "momentum": 0.95, "weight_decay": 1e-4}]
            elif key == "model.20.bias":
                params += [{"params": [value], "lr": 0.1 * base_lr,
                            "momentum": 0.95, "weight_decay": 0.0}]
            elif "weight" in key:
                params += [{"params": [value], "lr": 1.0 * base_lr,
                            "momentum": 0.95, "weight_decay": 1e-4}]
            else:
                params += [{"params": [value], "lr": 2.0 * base_lr,
                            "momentum": 0.95, "weight_decay": 0.0}]
        return params


if __name__ == "__main__":
    num_classes = 10
    pretrained = False
    bilinear = False
    net = NIN(num_classes=num_classes, bilinear=bilinear).cuda()
    # for name, param in net.named_parameters():
    #     print(name, param.size())
    # print(net.parameters())
    summary(net, (3, 32, 32))
