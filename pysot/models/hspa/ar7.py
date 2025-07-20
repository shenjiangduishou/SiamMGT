import pywt
import pywt.data
import torch
from torch import nn
from functools import partial
import torch.nn.functional as F

# 论文地址 https://arxiv.org/pdf/2407.05848
# 微信公众号：AI缝合术

class SENetV2FeatureFusionModule(nn.Module):
    def __init__(self, in_channels, reduction_ratio = 4):
        super(SENetV2FeatureFusionModule, self).__init__()
        self.in_channels = in_channels
        self.reduction_ratio = reduction_ratio

        # 全局平均池化，沿高度方向
        self.avg_pool_h = nn.AdaptiveAvgPool2d((1, None))
        # 全局平均池化，沿宽度方向
        self.avg_pool_w = nn.AdaptiveAvgPool2d((None, 1))

        # 拼接后的全连接层操作
        self.fc_list = nn.ModuleList()
        for i in range(self.reduction_ratio):
            self.fc_list.append(nn.Linear(in_channels, in_channels // reduction_ratio))

        # 另一组全连接层操作
        self.fc5 = nn.Linear((in_channels // reduction_ratio)*reduction_ratio, in_channels)
        self.fc6 = nn.Linear(in_channels, in_channels)

    def forward(self, x):
        b, c, h, w = x.size()

        # 沿高度方向全局平均池化
        avg_pool_h = self.avg_pool_h(x).squeeze(2)
        # 沿宽度方向全局平均池化
        avg_pool_w = self.avg_pool_w(x).squeeze(3)

        # 拼接操作
        concat = torch.cat([avg_pool_h, avg_pool_w], dim=2)

        # 经过第一组全连接层
        fc_list = []
        for fc in self.fc_list:
            output = fc(concat)
            fc_list.append(output)
        # fc1 = self.fc1(concat)
        # fc2 = self.fc2(concat)
        # fc3 = self.fc3(concat)
        # fc4 = self.fc4(concat)

        # 组合操作
        concat = torch.cat(fc_list, dim=2)

        # 经过第二组全连接层
        fc5 = self.fc5(concat)
        fc6 = self.fc6(fc5)

        # 沿高度和宽度方向拆分
        split_h = fc6[:, :, :h]
        split_w = fc6[:, :, h:]

        # 与原始输入相乘进行特征融合
        result = x +x * split_h.unsqueeze(3) * split_w.unsqueeze(2)
        return result

class Graph_Attention_Union_ar(nn.Module):
    def __init__(self, in_channel, out_channel):
        super(Graph_Attention_Union_ar, self).__init__()

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

        self.fi2 = SENetV2FeatureFusionModule(50,4)

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
        # hspa_score = self.st(similar)

        embedding = torch.matmul(similar, zf_g_plain).permute(0, 2, 1)
        # embedding = self.st(embedding)
        embedding = embedding.view(-1, shape_x[1], shape_x[2], shape_x[3])

        # aggregated feature
        output = torch.cat([embedding, xf_g], 1)
        output = self.fi2(output)
        output = self.fi(output)

        return output

if __name__ == '__main__':
    channel = 256
    input = torch.randn(3, channel, 13, 13)  # b c h w输入
    input2 = torch.randn(3, channel, 25, 25)  # b c h w输入
    wtconv = Graph_Attention_Union_ar(in_channel=256, out_channel=256)
    output = wtconv(input,input2)
    print(input.size(),input2.size(),output.size())