import torch
import torch.nn as nn


class SENetV2FeatureFusionModule(nn.Module):
    def __init__(self, in_channels, reduction_ratio = 2):
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

if __name__ == '__main__':
    # 假设输入特征图的通道数为64，缩减比例为16
    in_channels = 64
    reduction_ratio = 4
    # 创建模块实例
    module = SENetV2FeatureFusionModule(26, reduction_ratio)
    # 假设输入特征图的batch size为4，高度和宽度均为32
    input_tensor = torch.randn(76, in_channels, 13, 13)
    output = module(input_tensor)
    print(output.shape)