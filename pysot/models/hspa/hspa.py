import torch
import torch.nn as nn
from torch.autograd import Function


class TrackingHSPA(nn.Module):
    """修复版跨特征高相似度注意力模块"""

    def __init__(self, channel=256, reduction=4, topk=64):
        super().__init__()
        # 模板特征投影层
        self.t_proj = nn.Sequential(
            nn.Conv2d(channel, channel // reduction, 1),
            nn.PReLU(),
            nn.Conv2d(channel // reduction, channel, 1)
        )

        # 搜索区域投影层
        self.s_proj = nn.Sequential(
            nn.Conv2d(channel, channel // reduction, 1),
            nn.PReLU(),
            nn.Conv2d(channel // reduction, channel, 1)
        )

        # 动态特征适配器（修复尺寸不匹配核心）
        self.feature_adapter = nn.Sequential(
            nn.AdaptiveAvgPool2d((25, 25)),
            nn.Conv2d(channel, channel, 3, padding=1)
        )

        # 软阈值注意力机制
        self.ST = SoftThresholdingOperation(dim=2, topk=topk)

    def forward(self, template_feat, search_feat):
        """
        输入:
            template_feat: [N, C, 13, 13]
            search_feat: [N, C, 25, 25]
        输出:
            enhanced_search: [N, C, 25, 25]
        """
        # 特征对齐（关键修复点）
        adapted_template = self.feature_adapter(template_feat)  # [N,C,25,25]

        # 双路特征投影
        t_embed = self.t_proj(adapted_template)  # [N,C,25,25]
        s_embed = self.s_proj(search_feat)  # [N,C,25,25]

        # 相似度矩阵计算
        N, C, H, W = s_embed.shape
        t_flat = t_embed.view(N, C, -1).permute(0, 2, 1)  # [N,625,C]
        s_flat = s_embed.view(N, C, -1)  # [N,C,625]
        score = torch.bmm(t_flat, s_flat)  # [N,625,625]

        # 软阈值过滤
        score = self.ST(score)  # [N,625,625]

        # 特征聚合
        search_flat = search_feat.view(N, C, -1)  # [N,C,625]
        enhanced = torch.bmm(score, search_flat.permute(0, 2, 1))  # [N,625,C]

        # 残差连接
        enhanced = enhanced.permute(0, 2, 1).view(N, C, H, W)
        return search_feat + enhanced


# 保持软阈值操作实现（同HSPAN论文）
class SoftThresholdingOperation(nn.Module):
    def __init__(self, dim=2, topk=128):
        super().__init__()
        self.dim = dim
        self.topk = topk

    def forward(self, x):
        return softThresholdingOperation(x, self.dim, self.topk)


def softThresholdingOperation(x, dim=2, topk=128):
    return SoftThresholdingOperationFun.apply(x, dim, topk)


class SoftThresholdingOperationFun(Function):
    @classmethod
    def forward(cls, ctx, s, dim=2, topk=128):
        ctx.dim = dim
        max_val, _ = s.max(dim=dim, keepdim=True)
        s = s - max_val
        tau, supp_size = tau_support(s, dim=dim, topk=topk)
        output = torch.clamp(s - tau, min=0)
        ctx.save_for_backward(supp_size, output)
        return output

    @classmethod
    def backward(cls, ctx, grad_output):
        supp_size, output = ctx.saved_tensors
        dim = ctx.dim
        grad_input = grad_output.clone()
        grad_input[output == 0] = 0
        v_hat = grad_input.sum(dim=dim) / supp_size.to(output.dtype).squeeze(dim)
        v_hat = v_hat.unsqueeze(dim)
        grad_input = torch.where(output != 0, grad_input - v_hat, grad_input)
        return grad_input, None, None


# 辅助函数保持HSPAN原实现
def tau_support(s, dim=2, topk=128):
    if topk is None or topk >= s.shape[dim]:
        k, _ = torch.sort(s, dim=dim, descending=True)
    else:
        k, _ = torch.topk(s, k=topk, dim=dim)
    topk_cumsum = k.cumsum(dim) -1
    ar_x = ix_like_fun(k, dim)
    support = ar_x * k > topk_cumsum
    support_size = support.sum(dim=dim).unsqueeze(dim)
    tau = topk_cumsum.gather(dim, support_size -1)
    tau /= support_size.to(s.dtype)
    if topk is not None and topk < s.shape[dim]:
        unsolved = (support_size == topk).squeeze(dim)
        if torch.any(unsolved):
            in_1 = roll_fun(s, dim)[unsolved]
            tau_1, support_size_1 = tau_support(in_1, dim=-1, topk=2* topk)
            roll_fun(tau, dim)[unsolved] = tau_1
            roll_fun(support_size, dim)[unsolved] = support_size_1
    return tau, support_size
def ix_like_fun(x, dim):
    d = x.size(dim)
    ar_x = torch.arange(1, d +1, device=x.device, dtype=x.dtype)
    view = [1] * x.dim()
    view[0] = -1
    return ar_x.view(view).transpose(0, dim)
def roll_fun(x, dim):
    if dim == -1:
        return x
    elif dim <0:
        dim = x.dim() - dim
    perm = [i for i in range(x.dim()) if i != dim] + [dim]
    return x.permute(perm)

if __name__ == "__main__":
#模块参数
    batch_size =1# 批大小
    channels =256# 输入特征通道数
    height =13#图像高度
    width = 13  # 图像宽度width =32# 图像宽度
# 创建 hspa 模块
    hspa = TrackingHSPA(channel=256, reduction=4,  topk=64)
    print(hspa)
    print("微信公众号:AI缝合术, nb!")
# 生成随机输入张量 (batch_size, channels, height, width)
    x = torch.randn(batch_size, channels, 25, 25)
    z = torch.randn(batch_size, channels, height, width)
# 打印输入张量的形状
    print("Input shape:", x.shape)
    print("Input shape:", z.shape)
# 前向传播计算输出
    output = hspa(z,x)
# 打印输出张量的形状
    print("Output shape:", output.shape)