#####################################################
#
#####################################################


import torch.nn as nn
import torch
from torchvision import models
import sys
import math
import torch.nn.functional as F

class net6_adaptive(nn.Module):
    def __init__(self, load_weights=True):
        super(net6_adaptive, self).__init__()
        self.frontend_feat = [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512]
        self.frontend = make_layers(self.frontend_feat, dilation=False)
        self.backend_feat = [512, 512, 512, 256, 128, 64]
        self.backend = make_layers(self.backend_feat, in_channels=512, dilation=True)
        self.output_layer = nn.Conv2d(64, 1, kernel_size=1)

        self.refine_net = nn.Sequential(nn.Conv2d(8, 512, kernel_size=3, padding=1, stride=1),
                                        nn.ReLU(inplace=True),
                                        nn.Conv2d(512, 512, kernel_size=3, padding=1, stride=1),
                                        nn.ReLU(inplace=True),
                                        nn.Conv2d(512, 256, kernel_size=3, padding=1, stride=1),
                                        nn.ReLU(inplace=True),
                                        nn.Conv2d(256, 128, kernel_size=3, padding=1, stride=1),
                                        nn.ReLU(inplace=True),
                                        nn.Conv2d(128, 64, kernel_size=3, padding=1, stride=1),
                                        nn.ReLU(inplace=True),
                                        nn.Conv2d(64, 1, kernel_size=3, padding=1, stride=1)
                                        )

        if load_weights:
            mod = models.vgg16(pretrained=True)
            self._initialize_weights()
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

        point_map = point_map.repeat(1, 5, 1, 1)  # point map复制5遍
        fused_image = torch.cat((mini_image, point_map), 1)
        refined_density_map = self.refine_net(fused_image)
        # refined_density_map = self.refine_net(point_map)

        return predict_density_map, refined_density_map

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
    net = net0()
