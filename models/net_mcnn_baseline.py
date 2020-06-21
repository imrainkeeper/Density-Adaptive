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

        self.fuse = nn.Conv2d(30, 1, kernel_size=1, padding=0)       # 这里最后一层不用relu

    def forward(self, image):
        branch1_feature = self.branch1(image)
        branch2_feature = self.branch2(image)
        branch3_feature = self.branch3(image)
        concatenated_feature = torch.cat((branch1_feature, branch2_feature, branch3_feature), 1)
        predict_density_map = self.fuse(concatenated_feature)
        return predict_density_map

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.normal_(m.weight, std=0.01)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)

# def weights_normal_init(model, dev=0.01):
#     if isinstance(model, list):
#         for m in model:
#             weights_normal_init(m, dev)
#     else:
#         for m in model.modules():
#             if isinstance(m, nn.Conv2d):
#                 nn.init.normal_(m.weight, std=0.01)
#                 if m.bias is not None:
#                     nn.init.constant_(m.bias, 0)
#             elif isinstance(m, nn.BatchNorm2d):
#                 nn.init.constant_(m.weight, 1)
#                 nn.init.constant_(m.bias, 0)
#             elif isinstance(m, nn.Linear):
#                 nn.init.normal_(m.weight, 0, 0.01)
#                 nn.init.constant_(m.bias, 0)


    # def _initialize_weights(self):
    #     for m in self.modules():
    #         if isinstance(m, nn.Conv2d):
    #             nn.init.normal_(m.weight, std=0.01)
    #             if m.bias is not None:
    #                 nn.init.constant_(m.bias, 0)
    #         elif isinstance(m, nn.BatchNorm2d):
    #             nn.init.constant_(m.weight, 1)
    #             nn.init.constant_(m.bias, 0)
    #         elif isinstance(m, nn.Linear):
    #             nn.init.normal_(m.weight, 0, 0.01)
    #             nn.init.constant_(m.bias, 0)


if __name__ == '__main__':
    net = mcnn_baseline()

