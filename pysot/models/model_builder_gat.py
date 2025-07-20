# Copyright (c) SenseTime. All Rights Reserved.

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import torch
import torch.nn as nn
import torch.nn.functional as F

from pysot.core.config import cfg
from pysot.models.loss_car import make_siamcar_loss_evaluator
from pysot.models.backbone import get_backbone
from pysot.models.neck import get_neck
from pysot.models.head.car_head import CARHead
from pysot.models.hspa.hspa import TrackingHSPA,SoftThresholdingOperation
from pysot.models.hspa.hspa4 import GAT_HSPA_Cascade
# from pysot.models.hspa.ar9 import Graph_Attention_Union_ar
from pysot.models.hspa.ar4 import Graph_Attention_Union_ar
from pysot.models.hspa.ar6 import SENetV2FeatureFusionModule
# from pysot.models.hspa.ar8 import Graph_Attention_Union_ar
from ..utils.location_grid import compute_locations


class Graph_Attention_Union(nn.Module):
    def __init__(self, in_channel, out_channel):
        super(Graph_Attention_Union, self).__init__()

        # search region nodes linear transformation
        self.support = nn.Conv2d(in_channel, in_channel, 1, 1)

        # target template nodes linear transformation
        self.query = nn.Conv2d(in_channel, in_channel, 1, 1)

        # linear transformation for message passing
        self.g = nn.Sequential(
            nn.Conv2d(in_channel, in_channel, 1, 1),
            nn.BatchNorm2d(in_channel),
            nn.ReLU(inplace=True),
        )

        # aggregated feature
        self.fi = nn.Sequential(
            nn.Conv2d(in_channel*2, out_channel, 1, 1),
            nn.BatchNorm2d(out_channel),
            nn.ReLU(inplace=True),
        )

    def forward(self, zf, xf):
        # linear transformation
        xf_trans = self.query(xf)
        zf_trans = self.support(zf)

        # linear transformation for message passing
        xf_g = self.g(xf)
        zf_g = self.g(zf)

        # calculate similarity
        shape_x = xf_trans.shape
        shape_z = zf_trans.shape

        zf_trans_plain = zf_trans.view(-1, shape_z[1], shape_z[2] * shape_z[3])
        zf_g_plain = zf_g.view(-1, shape_z[1], shape_z[2] * shape_z[3]).permute(0, 2, 1)
        xf_trans_plain = xf_trans.view(-1, shape_x[1], shape_x[2] * shape_x[3]).permute(0, 2, 1)

        similar = torch.matmul(xf_trans_plain, zf_trans_plain)
        similar = F.softmax(similar, dim=2)

        embedding = torch.matmul(similar, zf_g_plain).permute(0, 2, 1)
        embedding = embedding.view(-1, shape_x[1], shape_x[2], shape_x[3])

        # aggregated feature
        output = torch.cat([embedding, xf_g], 1)
        output = self.fi(output)
        return output

class Graph_Attention_Union2(nn.Module):
    def __init__(self, in_channel):
        super(Graph_Attention_Union2, self).__init__()

        # search region nodes linear transformation
        self.support = nn.Conv2d(in_channel, in_channel, 1, 1)

        # target template nodes linear transformation
        self.query = nn.Conv2d(in_channel, in_channel, 1, 1)
        self.conv_c1 = nn.Sequential(nn.Conv2d(2 * in_channel, in_channel, 3, 1, 1), nn.BatchNorm2d(in_channel), nn.ReLU())
        self.conv_c2 = nn.Sequential(nn.Conv2d(in_channel, 2, 3, 1, 1), nn.BatchNorm2d(2), nn.ReLU())
        self.zf_transpose_conv = nn.ConvTranspose2d(in_channels=in_channel, out_channels=in_channel,
                                                    kernel_size=3, stride=2, padding=1, output_padding=0)#将zf特征图从13*13变为25*25

        # linear transformation for message passing
        self.g = nn.Sequential(
            nn.Conv2d(in_channel, in_channel, 1, 1),
            nn.BatchNorm2d(in_channel),
            nn.ReLU(inplace=True),
        )

        # aggregated feature
    def fusion(self, f1, f2, f_vec):
        w1 = f_vec[:, 0, :, :].unsqueeze(1)
        w2 = f_vec[:, 1, :, :].unsqueeze(1)
        out1 = (w1 * f1) + (w2 * f2)
        out2 = (w1 * f1) * (w2 * f2)
        return out1 + out2

    def forward(self, zf, xf):
        # linear transformation
        xf_trans = self.query(xf)
        zf_trans = self.zf_transpose_conv(zf)#记得把这个放入参数更新中（反向传播）！！！
        # zf_trans = F.interpolate(zf, size=(25, 25), mode='bilinear', align_corners=False)#或者使用双线性插值试一下
        # zf_trans = F.interpolate(zf, size=(25, 25), mode='nearest')#或者最近邻算法
        zf_trans = self.support(zf_trans)
        # linear transformation for message passing
        xf_g = self.g(xf)
        zf_g = self.g(zf_trans)

        # calculate similarity
        shape_x = xf_trans.shape
        shape_z = zf_trans.shape

        zf_trans_plain = zf_trans.view(-1, shape_z[1], shape_z[2] * shape_z[3])
        zf_g_plain = zf_g.view(-1, shape_z[1], shape_z[2] * shape_z[3]).permute(0, 2, 1)
        xf_trans_plain = xf_trans.view(-1, shape_x[1], shape_x[2] * shape_x[3]).permute(0, 2, 1)

        similar = torch.matmul(xf_trans_plain, zf_trans_plain)
        similar = F.softmax(similar, dim=2)

        embedding = torch.matmul(similar, zf_g_plain).permute(0, 2, 1)
        embedding = embedding.view(-1, shape_x[1], shape_x[2], shape_x[3])

        # aggregated feature
        output = torch.cat([embedding, xf_g], 1)
        output = self.conv_c1(output)
        output = self.conv_c2(output)
        output = self.fusion(zf_trans,xf_trans,output)
        return output

class Graph_Attention_Union3(nn.Module):
    def __init__(self, in_channel, out_channel):
        super(Graph_Attention_Union3, self).__init__()

        # search region nodes linear transformation
        self.support = nn.Conv2d(in_channel, in_channel, 1, 1)

        # target template nodes linear transformation
        self.query = nn.Conv2d(in_channel, in_channel, 1, 1)

        # linear transformation for message passing
        self.g = nn.Sequential(
            nn.Conv2d(in_channel, in_channel, 1, 1),
            nn.BatchNorm2d(in_channel),
            nn.ReLU(inplace=True),
        )

        self.st = SoftThresholdingOperation(dim=2,topk=256)

        # aggregated feature
        self.fi = nn.Sequential(
            nn.Conv2d(in_channel*2, out_channel, 1, 1),
            nn.BatchNorm2d(out_channel),
            nn.ReLU(inplace=True),
        )

    def forward(self, zf, xf):
        # linear transformation
        xf_trans = self.query(xf)
        zf_trans = self.support(zf)

        # linear transformation for message passing
        xf_g = self.g(xf)
        zf_g = self.g(zf)

        # calculate similarity
        shape_x = xf_trans.shape
        shape_z = zf_trans.shape

        zf_trans_plain = zf_trans.view(-1, shape_z[1], shape_z[2] * shape_z[3])
        zf_g_plain = zf_g.view(-1, shape_z[1], shape_z[2] * shape_z[3]).permute(0, 2, 1)
        xf_trans_plain = xf_trans.view(-1, shape_x[1], shape_x[2] * shape_x[3]).permute(0, 2, 1)

        similar = torch.matmul(xf_trans_plain, zf_trans_plain)
        # similar = F.softmax(similar, dim=2)
        hspa_score = self.st(similar)

        embedding = torch.matmul(hspa_score, zf_g_plain).permute(0, 2, 1)
        # embedding = self.st(embedding)
        embedding = embedding.view(-1, shape_x[1], shape_x[2], shape_x[3])

        # aggregated feature
        output = torch.cat([embedding, xf_g], 1)
        output = self.fi(output)
        return output

class ModelBuilder(nn.Module):
    def __init__(self):
        super(ModelBuilder, self).__init__()

        # build backbone
        self.backbone = get_backbone(cfg.BACKBONE.TYPE,
                                     **cfg.BACKBONE.KWARGS)

        #build backbone
        if cfg.ADJUST.ADJUST:
            self.neck = get_neck(cfg.ADJUST.TYPE,**cfg.ADJUST.KWARGS)

        # build car head
        self.car_head = CARHead(cfg, 256)

        # build response map
        # self.attention = Graph_Attention_Union(256,256)
        # self.SENetV2_1 = SENetV2FeatureFusionModule(26,4)
        # self.SENetV2_2 = SENetV2FeatureFusionModule(50,4)
        # self.fusion = nn.Sequential(
        #     nn.Conv2d(256 * 2, 256, 1, 1),
        #     nn.BatchNorm2d(256),
        #     nn.ReLU(inplace=True),
        # )
        self.attention = Graph_Attention_Union_ar(256,256)
        # self.attention = TrackAgentAttention(in_dim=256, agent_num=64, head=4)


        # build loss
        self.loss_evaluator = make_siamcar_loss_evaluator(cfg)

    def template(self, z, roi):
        zf = self.backbone(z, roi)
        self.zf = zf
        # self.wt_z = wt_z
        # self.zf_cat = torch.cat([self.zf, self.wt_z], dim=1)
        # self.new_zf = self.fusion(self.zf_cat)

    def track(self, x):
        xf = self.backbone(x)
        # xf_cat = torch.cat([xf,wt_x], dim=1)
        # new_xf = self.fusion(xf_cat)
        features = self.attention(self.zf, xf)
        # features = self.attention(self.zf, xf)

        cls, loc, cen = self.car_head(features)
        return {
                'cls': cls,
                'loc': loc,
                'cen': cen
               }

    def log_softmax(self, cls):
        b, a2, h, w = cls.size()
        cls = cls.view(b, 2, a2//2, h, w)
        cls = cls.permute(0, 2, 3, 4, 1).contiguous()
        cls = F.log_softmax(cls, dim=4)
        return cls

    def forward(self, data):
        """ only used in training
        """
        template = data['template'].cuda()
        search = data['search'].cuda()
        label_cls = data['label_cls'].cuda()#特征图大小（25*25）的全0矩阵
        label_loc = data['bbox'].cuda()#x图片（287*287）对应的bbox
        target_box = data['target_box'].cuda()#z图片（127*127）对应的bbox
        neg = data['neg'].cuda()

        # get feature
        zf = self.backbone(template, target_box)
        xf = self.backbone(search)
        # zf_cat = torch.cat([zf,wt_z], dim=1)
        # xf_cat = torch.cat([xf,wt_x], dim=1)
        # new_zf = self.fusion(zf_cat)
        # new_xf = self.fusion(xf_cat)

        features = self.attention(zf, xf)

        cls, loc, cen = self.car_head(features)
        # cls = cls * cen
        locations = compute_locations(cls, cfg.TRACK.STRIDE, cfg.TRACK.OFFSET)
        cls = self.log_softmax(cls)
        cls_loss, loc_loss, cen_loss = self.loss_evaluator(
            locations,
            cls,
            loc,
            cen, label_cls, label_loc, neg
        )

        # get loss
        outputs = {}
        outputs['total_loss'] = cfg.TRAIN.CLS_WEIGHT * cls_loss + \
            cfg.TRAIN.LOC_WEIGHT * loc_loss + cfg.TRAIN.CEN_WEIGHT * cen_loss
        outputs['cls_loss'] = cls_loss
        outputs['loc_loss'] = loc_loss
        outputs['cen_loss'] = cen_loss
        return outputs
