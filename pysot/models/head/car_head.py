import torch
from torch import nn
import math

from pysot.models.head.head_utils import ConvMixerBlock
from pysot.models.hspa.hspa2 import HSPA


class AdaptiveWeightBlock(nn.Module):
    def __init__(self, in_channels, reduction=4):
        super().__init__()
        # 通道注意力分支
        self.channel_att = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(in_channels, in_channels // reduction, 1),
            nn.Conv2d(in_channels // reduction, in_channels, 1),
            nn.GELU(),
            nn.Sigmoid()  # 使用 Sigmoid 函数将通道注意力输出归一化到 [0, 1] 范围
        )
        # 空间注意力分支
        self.spatial_att = nn.Sequential(
            nn.Conv2d(in_channels, 1, kernel_size=3, padding=1),
            nn.GELU(),
            nn.Sigmoid()  # 使用 Sigmoid 函数将空间注意力输出归一化到 [0, 1] 范围
        )

    def forward(self, x):
        # 计算通道注意力权重
        channel_weight = self.channel_att(x)
        # 扩展通道注意力权重的维度以匹配输入特征图
        channel_weight = channel_weight.expand_as(x)

        # 计算空间注意力权重
        spatial_weight = self.spatial_att(x)
        # 扩展空间注意力权重的维度以匹配输入特征图
        spatial_weight = spatial_weight.expand_as(x)

        # 综合权重
        weighted_x = x * channel_weight * spatial_weight
        return weighted_x

class CARHead(torch.nn.Module):
    def __init__(self, cfg, in_channels):
        """
        Arguments:
            in_channels (int): number of channels of the input feature
        """
        super(CARHead, self).__init__()
        # TODO: Implement the sigmoid version first.
        num_classes = cfg.TRAIN.NUM_CLASSES

        cls_tower = []
        bbox_tower = []
        # 插入自适应权重块
        self.pr_cls_tower = AdaptiveWeightBlock(in_channels)
        self.pr_bbox_tower = AdaptiveWeightBlock(in_channels)
        for i in range(cfg.TRAIN.NUM_CONVS):
            cls_tower.append(
                nn.Conv2d(
                    in_channels,
                    in_channels,
                    kernel_size=3,
                    stride=1,
                    padding=1
                )
            )
            cls_tower.append(nn.GroupNorm(32, in_channels))
            cls_tower.append(nn.ReLU())
            bbox_tower.append(
                nn.Conv2d(
                    in_channels,
                    in_channels,
                    kernel_size=3,
                    stride=1,
                    padding=1
                )
            )
            bbox_tower.append(nn.GroupNorm(32, in_channels))
            bbox_tower.append(nn.ReLU())

        # 新增HSPA模块
        # self.hspa_cls = HSPA(channel=in_channels, reduction=4)  # 分类特征增强
        # self.hspa_bbox = HSPA(channel=in_channels, reduction=4) # 回归特征增强

        self.add_module('cls_tower', nn.Sequential(*cls_tower))
        self.add_module('bbox_tower', nn.Sequential(*bbox_tower))
        self.cls_logits = nn.Conv2d(
            in_channels, num_classes, kernel_size=3, stride=1,
            padding=1
        )
        self.bbox_pred = nn.Conv2d(
            in_channels, 4, kernel_size=3, stride=1,
            padding=1
        )
        self.centerness = nn.Conv2d(
            in_channels, 1, kernel_size=3, stride=1,
            padding=1
        )

        # initialization  self.hspa_cls,self.hspa_bbox,
        for modules in [self.cls_tower, self.bbox_tower,self.pr_cls_tower,self.pr_bbox_tower,
                        self.cls_logits, self.bbox_pred,
                        self.centerness]:
            for l in modules.modules():
                if isinstance(l, nn.Conv2d):
                    torch.nn.init.normal_(l.weight, std=0.01)
                    torch.nn.init.constant_(l.bias, 0)

        # initialize the bias for focal loss
        prior_prob = cfg.TRAIN.PRIOR_PROB
        bias_value = -math.log((1 - prior_prob) / prior_prob)
        torch.nn.init.constant_(self.cls_logits.bias, bias_value)

    def forward(self, x):
        # 分类分支
        x_cls = self.pr_cls_tower(x)
        cls_feat = self.cls_tower(x_cls)
        # cls_feat = self.cls_tower(x)
        # cls_feat = self.hspa_cls(cls_feat).contiguous()  # 确保内存连续
        logits = self.cls_logits(cls_feat)
        # centerness = self.centerness(cls_feat)

        # 回归分支
        x_box = self.pr_bbox_tower(x)
        bbox_feat = self.bbox_tower(x_box)
        # bbox_feat = self.bbox_tower(x)
        # bbox_feat = self.hspa_bbox(bbox_feat).contiguous()
        bbox_reg = torch.exp(self.bbox_pred(bbox_feat))
        centerness = self.centerness(bbox_feat)

        return logits, bbox_reg, centerness


class CARHead_CT(torch.nn.Module):
    def __init__(self, cfg, kernel=3, spatial_num=2):
        """
        Arguments:
            in_channels (int): number of channels of the input feature
        """
        super(CARHead_CT, self).__init__()
        # TODO: Implement the sigmoid version first.
        num_classes = cfg.TRAIN.NUM_CLASSES
        in_channels = cfg.TRAIN.CHANNEL_NUM

        cls_tower = []
        bbox_tower = []

        self.cls_spatial = ConvMixerBlock(in_channels, 3, depth=spatial_num)
        self.bbox_spatial = ConvMixerBlock(in_channels, 3, depth=spatial_num)

        for i in range(cfg.TRAIN.NUM_CONVS):
            cls_tower.append(
                nn.Conv2d(
                    in_channels,
                    in_channels,
                    kernel_size=kernel,
                    stride=1,
                    padding=kernel // 2,
                )
            )
            cls_tower.append(nn.GroupNorm(32, in_channels))
            cls_tower.append(nn.ReLU())
            bbox_tower.append(
                nn.Conv2d(
                    in_channels,
                    in_channels,
                    kernel_size=kernel,
                    stride=1,
                    padding=kernel // 2,
                )
            )
            bbox_tower.append(nn.GroupNorm(32, in_channels))
            bbox_tower.append(nn.ReLU())

        self.add_module('cls_tower', nn.Sequential(*cls_tower))
        self.add_module('bbox_tower', nn.Sequential(*bbox_tower))
        self.cls_logits = nn.Conv2d(
            in_channels, num_classes, kernel_size=kernel, stride=kernel // 2,
            padding=1
        )
        self.bbox_pred = nn.Conv2d(
            in_channels, 4, kernel_size=kernel, stride=kernel // 2,
            padding=1
        )
        self.centerness = nn.Conv2d(
            in_channels, 1, kernel_size=kernel, stride=kernel // 2,
            padding=1
        )

        # initialization
        for modules in [self.cls_tower, self.bbox_tower,
                        self.cls_logits, self.bbox_pred,
                        self.centerness]:
            for l in modules.modules():
                if isinstance(l, nn.Conv2d):
                    torch.nn.init.normal_(l.weight, std=0.01)
                    torch.nn.init.constant_(l.bias, 0)

        # initialize the bias for focal loss
        prior_prob = cfg.TRAIN.PRIOR_PROB
        bias_value = -math.log((1 - prior_prob) / prior_prob)
        torch.nn.init.constant_(self.cls_logits.bias, bias_value)

    def forward(self, x):
        cls_x = self.cls_spatial(x)
        bbox_x = self.bbox_spatial(x)

        cls_tower = self.cls_tower(cls_x)
        bbox_tower = self.bbox_tower(bbox_x)

        logits = self.cls_logits(cls_tower)
        centerness = self.centerness(cls_tower)
        bbox_reg = torch.exp(self.bbox_pred(bbox_tower))

        return logits, bbox_reg, centerness




class Scale(nn.Module):
    def __init__(self, init_value=1.0):
        super(Scale, self).__init__()
        self.scale = nn.Parameter(torch.FloatTensor([init_value]))

    def forward(self, input):
        return input * self.scale

