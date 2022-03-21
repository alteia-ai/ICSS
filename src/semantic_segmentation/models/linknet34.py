import torch.nn as nn
import torch
from src.semantic_segmentation.models.blocks import Encoder34, Decoder, pretrain_link34

class LinkNet34(nn.Module):
    """
    Generate model architecture. LinkNet slightly modified : Encoder ResNet34 instead of ResNet18.
    """

    @staticmethod
    def weight_init(m):
        if isinstance(m, nn.Linear):
            torch.nn.init.kaiming_normal(m.weight.data)

    def __init__(self, in_channels, n_classes, pretrain, lf=False, drop=False):
        """
        Model initialization
        """
        super(LinkNet34, self).__init__()

        self.conv1 = nn.Conv2d(in_channels, 64, 7, 2, 3, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(3, 2, 1)

        self.encoder1 = Encoder34(1, 64, 64, 3, 1, 1)
        self.encoder2 = Encoder34(2, 64, 128, 3, 2, 1)
        self.encoder3 = Encoder34(3, 128, 256, 3, 2, 1, drop=drop)
        self.encoder4 = Encoder34(4, 256, 512, 3, 2, 1, drop=drop)

        self.decoder1 = Decoder(64, 64, 3, 1, 1, 0, drop=drop)
        self.decoder2 = Decoder(128, 64, 3, 2, 1, 1, drop=drop)
        self.decoder3 = Decoder(256, 128, 3, 2, 1, 1)
        self.decoder4 = Decoder(512, 256, 3, 2, 1, 1)

        # Classifier
        self.tp_conv1 = nn.Sequential(nn.ConvTranspose2d(64, 32, 3, 2, 1, 1),
                                      nn.BatchNorm2d(32),
                                      nn.ReLU(inplace=True), )
        self.conv2 = nn.Sequential(nn.Conv2d(32, 32, 3, 1, 1),
                                   nn.BatchNorm2d(32),
                                   nn.ReLU(inplace=True), )
        self.tp_conv2 = nn.ConvTranspose2d(32, n_classes, 2, 2, 0)
        # self.tp_conv2 = nn.ConvTranspose2d(32, 32, 2, 2, 0)
        # self.final_conv = nn.Conv2d(32, n_classes, 1)
        self.lsm = nn.LogSoftmax(dim=1)
        self.net_name = "LinkNet34"
        self.apply(self.weight_init)
        if pretrain and in_channels == 3:
            pretrain_link34(self)
        self.lf = lf

    def forward(self, x):
        # Initial block
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        # Encoder blocks
        e1 = self.encoder1(x)
        e2 = self.encoder2(e1)
        e3 = self.encoder3(e2)
        e4 = self.encoder4(e3)

        # Decoder blocks

        d4 = e3 + self.decoder4(e4)
        d3 = e2 + self.decoder3(d4)
        d2 = e1 + self.decoder2(d3)
        d1 = x + self.decoder1(d2)

        # Classifier
        y = self.tp_conv1(d1)
        y = self.conv2(y)
        y = self.tp_conv2(y)
        # y = self.final_conv(y)
        final = self.lsm(y)
        return final, e4, y
