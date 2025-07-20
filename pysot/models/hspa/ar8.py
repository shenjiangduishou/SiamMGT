import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Function

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
        self.fc7 = nn.Linear(512, 512//2)
        self.channel_reduce = nn.Sequential(
            nn.Conv2d(512, 256, 1),
            nn.BatchNorm2d(256, eps=0.001),
        )

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
        fc6 = self.fc6(fc5).permute(0,2,1)
        fc7 = self.fc7(fc6).permute(0,2,1)
        x = self.channel_reduce(x)

        # 沿高度和宽度方向拆分
        split_h = fc7[:, :, :h]
        split_w = fc7[:, :, h:]

        # 与原始输入相乘进行特征融合
        result = x +x * split_h.unsqueeze(3) * split_w.unsqueeze(2)
        return result

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

    def __init__(self, channel=256, reduction=4):
        super(TrackingHSPA, self).__init__()

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

        return out

class Graph_Attention_Union_ar(nn.Module):
    def __init__(self, in_channel, out_channel):
        super(Graph_Attention_Union_ar, self).__init__()

        # search region nodes linear transformation
        self.support = nn.Conv2d(in_channel, in_channel, 1, 1)

        # target template nodes linear transformation
        self.query = nn.Conv2d(in_channel, in_channel, 1, 1)

        # linear transformation for message passing
        # self.g = nn.Sequential(
        #     nn.Conv2d(in_channel, in_channel, 1, 1),
        #     nn.BatchNorm2d(in_channel),
        #     nn.ReLU(inplace=True),
        # )

        self.g = nn.Sequential(TrackingHSPA(in_channel, reduction=4),
            nn.BatchNorm2d(in_channel),
            nn.ReLU(inplace=True),
        )

        # aggregated feature
        # self.fi = nn.Sequential(
        #     nn.Conv2d(in_channel*2, out_channel, 1, 1),
        #     nn.BatchNorm2d(out_channel),
        #     nn.ReLU(inplace=True),
        # )

        self.fi2 = nn.Sequential(SENetV2FeatureFusionModule(50,4),
            nn.BatchNorm2d(out_channel),
            nn.GELU(),)

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
        # output = self.hspa(output)
        # output = self.fi(output)
        output = self.fi2(output)

        return output

if __name__ == '__main__':
    channel = 256
    input = torch.randn(3, channel, 13, 13)  # b c h w输入
    input2 = torch.randn(3, channel, 25, 25)  # b c h w输入
    wtconv = Graph_Attention_Union_ar(in_channel=256, out_channel=256)
    output = wtconv(input,input2)
    print(input.size(),input2.size(),output.size())

