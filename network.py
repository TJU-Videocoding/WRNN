import torch.nn as nn
from utils.wavlet import *
from utils.base import ResBlock
import torch


class MFCNN_C(nn.Module):
    def __init__(self, nf, nb):
        super(MFCNN_C, self).__init__()

        layers = [nn.Conv2d(in_channels=3, out_channels=nf, kernel_size=3, padding=1, bias=True),
                  nn.ReLU()]
        for _ in range(nb):
            layers.append(ResBlock(in_channel=nf, out_channel=nf))
        layers.append(DWT_CNN(nf))
        for _ in range(nb):
            layers.append(ResBlock(in_channel=nf, out_channel=nf))
        layers.append(DWT_CNN(nf))
        self.layers = nn.Sequential(*layers)

        branch_C = [
            ResBlock(in_channel=nf, out_channel=nf),
            ResBlock(in_channel=nf, out_channel=nf),
            DWT_CNN(nf), nn.Flatten(),
            nn.Linear(8192, 128), nn.ReLU(),
            nn.Linear(128, 64), nn.ReLU(),
            nn.Linear(64, 1)
        ]
        self.branch_C = nn.Sequential(*branch_C)

    def forward(self, x):
        content_feature = self.layers(x)
        out_C = self.branch_C(content_feature)
        return out_C


class CLCNN_large(nn.Module):
    def __init__(self, nf, nb):
        super(CLCNN, self).__init__()

        layers = [nn.Conv2d(in_channels=3, out_channels=nf, kernel_size=3, padding=1, bias=True),
                  nn.ReLU()]
        for _ in range(nb):
            layers.append(ResBlock(in_channel=nf, out_channel=nf))
        layers.append(DWT_CNN2(nf))
        for _ in range(nb):
            layers.append(ResBlock(in_channel=2*nf, out_channel=2*nf))
        layers.append(DWT_CNN2(2*nf))
        layers.append(ResBlock(in_channels=4 * nf, out_channels=4 * nf))
        layers.append(ResBlock(in_channels=4 * nf, out_channels=4 * nf))
        layers.append(DWT_CNN2(4 * nf))
        self.layers = nn.Sequential(*layers)

        regression = [
            nn.Linear(512, 128), nn.ReLU(),
            nn.Linear(128, 64), nn.ReLU(),
            nn.Linear(64, 1)
        ]
        self.regression = nn.Sequential(*regression)

    def forward(self, x):
        content_feature = self.layers(x)
        flat_feature = torch.squeeze(nn.functional.adaptive_avg_pool2d(content_feature, (1, 1)))
        out = self.regression(flat_feature)
        return out

class CLCNN(nn.Module):
    def __init__(self, nf, nb):
        super(CLCNN, self).__init__()

        layers = [nn.Conv2d(in_channels=3, out_channels=nf, kernel_size=3, padding=1, bias=True),
                  nn.ReLU()]
        for _ in range(nb):
            layers.append(ResBlock(in_channel=nf, out_channel=nf))
        layers.append(DWT_CNN(nf))
        for _ in range(nb):
            layers.append(ResBlock(in_channel=nf, out_channel=nf))
        layers.append(DWT_CNN(nf))
        layers.append(ResBlock(in_channels=nf, out_channels=nf))
        layers.append(ResBlock(in_channels=nf, out_channels=nf))
        layers.append(DWT_CNN(nf))
        self.layers = nn.Sequential(*layers)

        regression = [
            nn.Linear(128, 128), nn.ReLU(),
            nn.Linear(128, 64), nn.ReLU(),
            nn.Linear(64, 1)
        ]
        self.regression = nn.Sequential(*regression)

    def forward(self, x):
        content_feature = self.layers(x)
        flat_feature = torch.squeeze(nn.functional.adaptive_avg_pool2d(content_feature, (1, 1)))
        out = self.regression(flat_feature)
        return out