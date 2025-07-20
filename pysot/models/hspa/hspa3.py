import torch
import torch.nn as nn
from torch.autograd import Function

#在backbone里面进行添加的


def softThresholdingOperation(x, dim=2, topk=128):
    return SoftThresholdingOperationFun.apply(x, dim, topk)
class SoftThresholdingOperationFun(Function):
    @classmethod
    def forward(cls, ctx, s, dim=2, topk=128):
        ctx.dim = dim
        max, _ = s.max(dim=dim, keepdim=True)
        s = s - max
        tau, supp_size = tau_support(s, dim=dim, topk=topk)
        output = torch.clamp(s - tau,min=0)
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

class TrackingHSPA(nn.Module):
    """适配跟踪任务的改进版HSPA模块"""

    def __init__(self, channel=256, reduction=4, res_scale=0.1):
        super(TrackingHSPA, self).__init__()
        self.res_scale = res_scale

        # 特征压缩网络（参数共享）
        self.conv_match = nn.Sequential(
            nn.Conv2d(channel, channel // reduction, 1),
            nn.BatchNorm2d(channel // reduction),
            nn.PReLU()
        )

        # 动态topk计算模块
        self.topk_predict = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(channel, channel // 4, 1),
            nn.ReLU(),
            nn.Conv2d(channel // 4, 1, 1),
            nn.Sigmoid()  # 输出0~1比例
        )

        # 特征重组网络（参数共享）
        self.conv_assembly = nn.Sequential(
            nn.Conv2d(channel, channel, 3, padding=1),
            nn.BatchNorm2d(channel),
            nn.PReLU()
        )

    def forward(self, input):
        # 动态计算topk值
        B, C, H, W = input.shape
        topk_ratio = self.topk_predict(input.mean(dim=(2, 3), keepdim=True))
        topk = max(1, int(H * W * topk_ratio.mean().item()))

        # 特征压缩
        x_embed = self.conv_match(input)  # [B, C/r, H, W]

        # 相似度计算（适应不同尺寸特征图）
        x_embed_flat = x_embed.view(B, -1, H * W)  # [B, C/r, N]
        similarity = torch.bmm(x_embed_flat.transpose(1, 2), x_embed_flat)  # [B, N, N]

        # 动态阈值处理
        similarity = softThresholdingOperation(similarity, dim=2, topk=topk)

        # 特征重组
        x_assembly = self.conv_assembly(input).view(B, C, H * W)  # [B, C, N]

        # 注意力聚合
        out = torch.bmm(x_assembly, similarity)  # [B, C, N]
        out = out.view(B, C, H, W)

        return input + self.res_scale * out  # 残差连接


if __name__ == "__main__":
#模块参数
    batch_size =76# 批大小
    channels =256# 输入特征通道数
    height =32#图像高度
    width = 32  # 图像宽度width =32# 图像宽度
# 创建 hspa 模块
    hspa = TrackingHSPA(channel=256, reduction=2)
    print(hspa)
# 生成随机输入张量 (batch_size, channels, height, width)
    x = torch.randn(batch_size, channels, height, width)
# 打印输入张量的形状
    print("Input shape:", x.shape)
# 前向传播计算输出
    output = hspa(x)
# 打印输出张量的形状
    print("Output shape:", output.shape)