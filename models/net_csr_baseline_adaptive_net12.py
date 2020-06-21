#####################################################

#####################################################

import torch.nn as nn
import torch
from torchvision import models
import sys
import math
import torch.nn.functional as F

class net(nn.Module):
    def __init__(self, load_weights=True):
        super(net, self).__init__()
        self.frontend_feat = [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512]
        self.frontend = make_layers(self.frontend_feat, dilation=False)
        self.backend_feat = [512, 512, 512, 256, 128, 64]
        self.backend = make_layers(self.backend_feat, in_channels=512, dilation=True)
        self.output_layer = nn.Conv2d(64, 1, kernel_size=1)

        self.e1 = nn.Sequential(nn.Conv2d(1, 128, kernel_size=3, padding=1),
                                nn.ReLU(inplace=True))
        self.e2 = nn.Sequential(nn.Conv2d(128, 256, kernel_size=3, padding=1),
                                nn.ReLU(inplace=True))
        self.e3 = nn.Sequential(nn.Conv2d(256, 256, kernel_size=3, padding=1),
                                nn.ReLU(inplace=True))
        self.e4 = nn.Sequential(nn.Conv2d(256, 256, kernel_size=3, padding=1),
                                nn.ReLU(inplace=True))
        self.e5 = nn.Sequential(nn.Conv2d(256, 256, kernel_size=3, padding=1),
                                nn.ReLU(inplace=True))
        self.e6 = nn.Sequential(nn.Conv2d(256, 256, kernel_size=3, padding=1),
                                nn.ReLU(inplace=True))
        self.t1 = nn.Sequential(nn.ConvTranspose2d(256, 256, kernel_size=2, stride=2),
                                nn.ReLU(inplace=True))
        self.d1 = nn.Sequential(nn.Conv2d(512, 256, kernel_size=3, padding=1),
                                nn.ReLU(inplace=True))
        self.t2 = nn.Sequential(nn.ConvTranspose2d(256, 256, kernel_size=2, stride=2),
                                nn.ReLU(inplace=True))
        self.d2 = nn.Sequential(nn.Conv2d(512, 256, kernel_size=3, padding=1),
                                nn.ReLU(inplace=True))
        self.t3 = nn.Sequential(nn.ConvTranspose2d(256, 256, kernel_size=2, stride=2),
                                nn.ReLU(inplace=True))
        self.d3 = nn.Sequential(nn.Conv2d(512, 256, kernel_size=3, padding=1),
                                nn.ReLU(inplace=True))
        self.t4 = nn.Sequential(nn.ConvTranspose2d(256, 256, kernel_size=2, stride=2),
                                nn.ReLU(inplace=True))
        self.d4 = nn.Sequential(nn.Conv2d(512, 256, kernel_size=3, padding=1),
                                nn.ReLU(inplace=True))
        self.t5 = nn.Sequential(nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2),
                                nn.ReLU(inplace=True))
        self.d5 = nn.Sequential(nn.Conv2d(256, 64, kernel_size=3, padding=1),
                                nn.ReLU(inplace=True))

        self.adaptive_output_layer = nn.Conv2d(64, 1, kernel_size=3, padding=1)

        if load_weights:
            self._initialize_weights()
            mod = models.vgg16(pretrained=True)
            for i in range(len(self.frontend.state_dict().items())):
                list(self.frontend.state_dict().items())[i][1].data[:] = list(mod.state_dict().items())[i][1].data[:]
            for k in self.frontend.children():
                for param in k.parameters():
                    param.requires_grad = False
        else:
            self._initialize_weights()

    def forward(self, image, mini_image, point_map):
        frontend_feature = self.frontend(image)
        backend_feature = self.backend(frontend_feature)
        predict_density_map = self.output_layer(backend_feature)

        e1_feature = self.e1(point_map)
        e2_feature = self.e2(F.max_pool2d(e1_feature, 2))
        e3_feature = self.e3(F.max_pool2d(e2_feature, 2))
        e4_feature = self.e4(F.max_pool2d(e3_feature, 2))
        e5_feature = self.e5(F.max_pool2d(e4_feature, 2))
        e6_feature = self.e6(F.max_pool2d(e5_feature, 2))
        t1_feature = self.t1(e6_feature)
        t1_concat = torch.cat((e5_feature, t1_feature), 1)
        d1_feature = self.d1(t1_concat)
        t2_feature = self.t2(d1_feature)
        t2_concat = torch.cat((e4_feature, t2_feature), 1)
        d2_feature = self.d2(t2_concat)
        t3_feature = self.t3(d2_feature)
        t3_concat = torch.cat((e3_feature, t3_feature), 1)
        d3_feature = self.d3(t3_concat)
        t4_feature = self.t4(d3_feature)
        t4_concat = torch.cat((e2_feature, t4_feature), 1)
        d4_feature = self.d4(t4_concat)
        t5_feature = self.t5(d4_feature)
        t5_concat = torch.cat((e1_feature, t5_feature), 1)
        d5_feature = self.d5(t5_concat)

        adaptive_density_map = self.adaptive_output_layer(d5_feature)

        return predict_density_map, adaptive_density_map

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


def make_layers(cfg, in_channels=3, batch_norm=False, dilation=False):
    if dilation:
        d_rate = 2
    else:
        d_rate = 1
    layers = []
    for v in cfg:
        if v == 'M':
            layers += [nn.MaxPool2d(kernel_size=2, stride=2, ceil_mode=True)]
        else:
            conv2d = nn.Conv2d(in_channels, v, kernel_size=3, padding=d_rate, dilation=d_rate)
            if batch_norm:
                layers += [conv2d, nn.BatchNorm2d(v), nn.ReLU(inplace=True)]
            else:
                layers += [conv2d, nn.ReLU(inplace=True)]
            in_channels = v
    return nn.Sequential(*layers)


if __name__ == '__main__':
    net = csr_baseline_refine()