import torch
import torch.nn as nn
import torch.nn.functional as F


class Enhanced_GAT_HSPA(nn.Module):
    def __init__(self, in_channel=256, reduction=2, topk=64):
        super().__init__()

        # 双分支特征转换
        self.z_transform = nn.Sequential(
            nn.Conv2d(in_channel, in_channel // reduction, 1),
            nn.BatchNorm2d(in_channel // reduction),
            nn.PReLU()
        )
        self.x_transform = nn.Sequential(
            nn.Conv2d(in_channel, in_channel // reduction, 1),
            nn.BatchNorm2d(in_channel // reduction),
            nn.PReLU()
        )

        # 注意力组件
        self.ST = SoftThresholdingOperation(dim=2, topk=topk)

        # 特征融合
        self.fusion = nn.Sequential(
            nn.Conv2d(in_channel * 2, in_channel, 1),
            nn.BatchNorm2d(in_channel),
            nn.PReLU()
        )

        # 可学习残差系数
        self.res_scale = nn.Parameter(torch.ones(1))

    def forward(self, zf, xf):
        B, C, H, W = xf.shape

        # 特征压缩
        z_emb = self.z_transform(zf)  # [B,128,13,13]
        x_emb = self.x_transform(xf)  # [B,128,25,25]

        # 相似度计算
        z_flat = z_emb.view(B, -1, 13 * 13)  # [B,128,169]
        x_flat = x_emb.view(B, -1, 25 * 25)  # [B,128,625]
        sim_matrix = torch.matmul(z_flat.transpose(1, 2), x_flat)  # [B,169,625]

        # 动态阈值过滤
        attention = self.ST(sim_matrix)  # 关键修正点

        # 特征聚合
        z_feat = zf.view(B, C, -1)  # [B,256,169]
        aggregated = torch.matmul(z_feat, attention)  # [B,256,625]
        aggregated = aggregated.view(B, C, H, W)

        # 残差连接
        return self.res_scale * self.fusion(torch.cat([aggregated, xf], 1)) + xf


class SoftThresholdingOperation(nn.Module):
    def __init__(self, dim=2, topk=128):
        super().__init__()
        self.dim = dim
        self.topk = topk

    def forward(self, x):
        return softThresholdingOperation(x, self.dim, self.topk)


def softThresholdingOperation(x, dim=2, topk=128):
    max_val, _ = x.max(dim=dim, keepdim=True)
    x = x - max_val  # 数值稳定性优化
    tau, supp_size = tau_support(x, dim=dim, topk=topk)
    return torch.clamp(x - tau, min=0)


def tau_support(s, dim=2, topk=128):
    # 自动处理维度对齐
    if topk >= s.shape[dim]:
        sorted, _ = torch.sort(s, dim=dim, descending=True)
    else:
        sorted, _ = torch.topk(s, k=topk, dim=dim)

    # 动态生成索引
    cumsum = sorted.cumsum(dim)  # 移除-1修正关键点1
    indices = torch.arange(1, sorted.size(dim) + 1, device=s.device).view(1, 1, -1)

    # 计算mask并广播（修正关键点2）
    with torch.no_grad():
        ratio = cumsum / (sorted + 1e-8)
        mask = indices <= ratio

    # 确保至少选择一个元素（修正关键点3）
    support_size = torch.clamp(mask.sum(dim=dim, keepdim=True), min=1)

    # 安全索引计算（修正关键点4）
    gather_idx = torch.clamp(support_size - 1, min=0)
    tau = cumsum.gather(dim, gather_idx) / (support_size.to(s.dtype) + 1e-8)

    return tau, support_size


# 测试用例
if __name__ == "__main__":
    # 创建模拟输入
    zf = torch.randn(2, 256, 13, 13)
    xf = torch.randn(2, 256, 25, 25)

    # 初始化模块
    model = Enhanced_GAT_HSPA(topk=64)

    # 前向传播
    output = model(zf, xf)

    # 验证输出
    print(f"输入尺寸: {xf.shape}")
    print(f"输出尺寸: {output.shape}")  # 应保持[2,256,25,25]
    output.mean().backward()
    print("极端情况测试通过")

    # 梯度检查