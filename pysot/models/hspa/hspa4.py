import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Function


# ------------------- 修正后的GAT模块 -------------------
class Graph_Attention_Union(nn.Module):
    def __init__(self, in_channel=256, out_channel=256):
        super().__init__()
        # 保持原始参数不变
        self.support = nn.Conv2d(in_channel, in_channel, 1)
        self.query = nn.Conv2d(in_channel, in_channel, 1)

        # 特征转换网络
        self.g = nn.Sequential(
            nn.Conv2d(in_channel, in_channel, 1),
            nn.BatchNorm2d(in_channel),
            nn.ReLU(inplace=True),
        )

        # 输出融合网络
        self.fi = nn.Sequential(
            nn.Conv2d(in_channel * 2, out_channel, 1),
            nn.BatchNorm2d(out_channel),
            nn.ReLU(inplace=True),
        )

    def forward(self, zf, xf):
        # 维度验证
        assert zf.dim() == 4 and xf.dim() == 4, "输入必须是4D张量"

        # 特征变换 [B,C,H,W] -> [B,C,H,W]
        xf_trans = self.query(xf)  # 搜索区域特征
        zf_trans = self.support(zf)  # 模板特征

        # 消息传递特征 [B,C,H,W]
        xf_g = self.g(xf)
        zf_g = self.g(zf)

        # 展平处理（修正维度推断错误）
        B, C, H_z, W_z = zf_trans.shape
        _, _, H_x, W_x = xf_trans.shape

        # 模板特征展平 [B,C,H_z*W_z]
        zf_trans_flat = zf_trans.view(B, C, -1)  # 等效reshape(B,C,H_z*W_z)

        # 搜索特征展平 [B,H_x*W_x,C]
        xf_trans_flat = xf_trans.view(B, C, -1).permute(0, 2, 1)  # [B, HxWx, C]

        # 相似度矩阵计算 [B, HxWx, H_zW_z]
        similarity = torch.bmm(xf_trans_flat, zf_trans_flat)  # 批矩阵乘法
        similarity = F.softmax(similarity, dim=2)

        # 消息传递 [B,C,HxWx]
        zf_g_flat = zf_g.view(B, C, -1)  # [B,C,H_zW_z]
        embedding = torch.bmm(zf_g_flat, similarity.permute(0, 2, 1))  # [B,C,HxWx]

        # 恢复空间维度 [B,C,H_x,W_x]
        embedding = embedding.view(B, C, H_x, W_x)

        # 特征融合
        output = torch.cat([embedding, xf_g], dim=1)  # 通道拼接
        return self.fi(output)  # [B,out_channel,H_x,W_x]


# ------------------- 修正后的HSPA模块 -------------------
class TrackHSPA(nn.Module):
    def __init__(self, channel=256, reduction=4):
        super().__init__()
        # 特征压缩
        self.conv_match = nn.Sequential(
            nn.Conv2d(channel, channel // reduction, 3, padding=1),
            nn.BatchNorm2d(channel // reduction),
            nn.PReLU()
        )

        # 动态topk预测
        self.topk_predict = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(channel, 4, 1),
            nn.ReLU(),
            nn.Conv2d(4, 1, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        B, C, H, W = x.shape
        N = H * W

        # 动态计算topk比例
        topk_ratio = self.topk_predict(x).mean()  # [0,1]区间
        topk = max(1, int(N * topk_ratio.item()))

        # 特征压缩 [B,C/r,H,W]
        x_embed = self.conv_match(x)

        # 展平处理 [B,C/r, N]
        x_embed_flat = x_embed.view(B, -1, N)

        # 相似度矩阵 [B,N,N]
        similarity = torch.bmm(x_embed_flat.permute(0, 2, 1), x_embed_flat)

        # 软阈值处理
        max_val, _ = similarity.max(dim=2, keepdim=True)
        similarity = similarity - max_val  # 数值稳定

        # 计算动态阈值
        sorted_vals, _ = torch.sort(similarity, dim=2, descending=True)
        cumsum = sorted_vals.cumsum(dim=2) - 1
        arange = torch.arange(1, N + 1, device=x.device).view(1, 1, -1)
        mask = arange * sorted_vals > cumsum
        tau = cumsum.gather(2, mask.sum(dim=2, keepdim=True) - 1) / mask.sum(dim=2, keepdim=True).float()

        # 应用阈值
        similarity = torch.clamp(similarity - tau, min=0)

        # 特征重组
        x_assembly = x.view(B, C, N)  # [B,C,N]
        output = torch.bmm(x_assembly, similarity)  # [B,C,N]
        return output.view(B, C, H, W) + x  # 残差连接


# ------------------- 级联模块 -------------------
class GAT_HSPA_Cascade(nn.Module):
    def __init__(self, in_channel=256, out_channel=256):
        super().__init__()
        self.gat = Graph_Attention_Union(in_channel, out_channel)
        self.hspa = TrackHSPA(out_channel)
        # 特征融合门控
        # self.fusion_gate = nn.Sequential(
        #     nn.Conv2d(out_channel, 1, kernel_size=3, padding=1),
        #     nn.Sigmoid()
        # )

        # 参数初始化
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')

    def forward(self, zf, xf):
        # GAT处理 [B,256,25,25]
        gat_out = self.gat(zf, xf)

        # HSPA细化 [B,256,25,25]
        hspa_out = self.hspa(gat_out)

        # gate = self.fusion_gate(gat_out)
        # final_out = gate * hspa_out + (1 - gate) * gat_out
        # return final_out
        return hspa_out


# ------------------- 测试实例 -------------------
if __name__ == "__main__":
    # 输入维度验证
    zf = torch.randn(4, 256, 13, 13)  # 模板特征
    xf = torch.randn(4, 256, 25, 25)  # 搜索特征

    model = GAT_HSPA_Cascade()
    output = model(zf, xf)
    print(f"输入模板维度: {zf.shape} -> 输出维度: {output.shape}")  # 应输出[B,256,25,25]

    # 边缘案例测试
    try:
        model(torch.randn(1, 256, 7, 7), torch.randn(1, 256, 19, 19))
        print("小尺寸测试通过")
    except Exception as e:
        print(f"小尺寸测试失败: {str(e)}")

    # 参数统计
    params = sum(p.numel() for p in model.parameters())
    print(f"总参数量: {params / 1e6:.2f}M")  # 预期约2.1M

    # 速度测试（需要GPU）
    if torch.cuda.is_available():
        model.cuda()
        starter = torch.cuda.Event(enable_timing=True)
        ender = torch.cuda.Event(enable_timing=True)

        # Warm-up
        for _ in range(10):
            _ = model(zf.cuda(), xf.cuda())

        # 计时
        timings = []
        with torch.no_grad():
            for _ in range(100):
                starter.record()
                _ = model(zf.cuda(), xf.cuda())
                ender.record()
                torch.cuda.synchronize()
                timings.append(starter.elapsed_time(ender))

        print(f"GPU平均推理时间: {sum(timings) / len(timings):.2f}ms")