#####################################################
# Multi_column cnn 的 baseline
#####################################################

import torch.nn as nn
import torch
from torchvision import models
import sys
import math
import torch.nn.functional as F


class net(nn.Module):
    def __init__(self, bn=False):
        super(net, self).__init__()

        self.branch1 = nn.Sequential(nn.Conv2d(3, 16, kernel_size=9, padding=4),
                                     nn.ReLU(inplace=True),
                                     nn.MaxPool2d(2),
                                     nn.Conv2d(16, 32, kernel_size=7, padding=3),
                                     nn.ReLU(inplace=True),
                                     nn.MaxPool2d(2),
                                     nn.Conv2d(32, 16, kernel_size=7, padding=3),
                                     nn.ReLU(inplace=True),
                                     nn.Conv2d(16, 8, kernel_size=7, padding=3),
                                     nn.ReLU(inplace=True))
        self.branch2 = nn.Sequential(nn.Conv2d(3, 20, kernel_size=7, padding=3),
                                     nn.ReLU(inplace=True),
                                     nn.MaxPool2d(2),
                                     nn.Conv2d(20, 40, kernel_size=5, padding=2),
                                     nn.ReLU(inplace=True),
                                     nn.MaxPool2d(2),
                                     nn.Conv2d(40, 20, kernel_size=5, padding=2),
                                     nn.ReLU(inplace=True),
                                     nn.Conv2d(20, 10, kernel_size=5, padding=2),
                                     nn.ReLU(inplace=True))
        self.branch3 = nn.Sequential(nn.Conv2d(3, 24, kernel_size=5, padding=2),
                                     nn.ReLU(inplace=True),
                                     nn.MaxPool2d(2),
                                     nn.Conv2d(24, 48, kernel_size=3, padding=1),
                                     nn.ReLU(inplace=True),
                                     nn.MaxPool2d(2),
                                     nn.Conv2d(48, 24, kernel_size=3, padding=1),
                                     nn.ReLU(inplace=True),
                                     nn.Conv2d(24, 12, kernel_size=3, padding=1),
                                     nn.ReLU(inplace=True))

        self.fuse = nn.Conv2d(30, 1, kernel_size=1, padding=0)

        self.refine_encoder = nn.Sequential(nn.Conv2d(1, 64, kernel_size=3, padding=1),
                                            nn.ReLU(inplace=True),
                                            nn.MaxPool2d(2),
                                            nn.Conv2d(64, 128, kernel_size=3, padding=1),
                                            nn.ReLU(inplace=True),
                                            nn.MaxPool2d(2),
                                            )
        self.refine_decoder = nn.Sequential(nn.ConvTranspose2d(128, 64, kernel_size=4, padding=1, stride=2),
                                            nn.ReLU(inplace=True),
                                            nn.ConvTranspose2d(64, 64, kernel_size=4, padding=1, stride=2),
                                            nn.ReLU(inplace=True))
        self.refine_output_layer = nn.Conv2d(64, 1, kernel_size=3, padding=1)

    def forward(self, image, raw_density_map):
        branch1_feature = self.branch1(image)
        branch2_feature = self.branch2(image)
        branch3_feature = self.branch3(image)
        x = torch.cat((branch1_feature, branch2_feature, branch3_feature), 1)
        predicted_density_map = self.fuse(x)

        refined_encoder_feature = self.refine_encoder(raw_density_map)
        refined_decoder_feature = self.refine_decoder(refined_encoder_feature)
        refined_density_map = self.refine_output_layer(refined_decoder_feature)

        return predicted_density_map, refined_density_map


def weights_normal_init(model, dev=0.01):
    if isinstance(model, list):
        for m in model:
            weights_normal_init(m, dev)
    else:
        for m in model.modules():
            if isinstance(m, nn.Conv2d):
                # print torch.sum(m.weight)
                m.weight.data.normal_(0.0, dev)
                if m.bias is not None:
                    m.bias.data.fill_(0.0)
            elif isinstance(m, nn.Linear):
                m.weight.data.normal_(0.0, dev)


if __name__ == '__main__':
    net = mcnn_baseline()

