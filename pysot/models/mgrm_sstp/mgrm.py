import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Function
import cv2
import numpy as np


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

def keep_largest_connected_component(mask, connectivity=8):
    """
    保留掩码中最大的连通域并填充内部孔洞

    参数:
    - mask: 输入掩码张量，形状为(B, 1, H, W)，值为0或1
    - connectivity: 连通性，4或8

    返回:
    - processed_mask: 处理后的掩码，形状同上
    """
    batch_size, _, height, width = mask.shape
    processed_mask = torch.zeros_like(mask)

    # 对每个batch分别处理
    for b in range(batch_size):
        # 转换为numpy数组以便使用scipy的连通域分析
        mask_np = mask[b, 0].cpu().numpy().astype(np.uint8)

        # 使用scipy的连通域分析
        from scipy import ndimage
        labeled_array, num_features = ndimage.label(mask_np,
                                                    structure=np.ones((3, 3)) if connectivity == 8 else np.array(
                                                        [[0, 1, 0], [1, 1, 1], [0, 1, 0]]))

        # 如果没有连通域，直接返回
        if num_features == 0:
            continue

        # 找出最大的连通域
        largest_component = 0
        max_size = 0
        for i in range(1, num_features + 1):
            component_size = np.sum(labeled_array == i)
            if component_size > max_size:
                max_size = component_size
                largest_component = i

        # 创建只包含最大连通域的掩码
        largest_mask = (labeled_array == largest_component).astype(np.uint8)
        # structuring_element = ndimage.generate_binary_structure(2, 1)  # 可以调整结构元素大小
        # largest_mask = ndimage.binary_closing(largest_mask, structure=structuring_element)#闭运算填充内部

        # 填充最大连通域内的孔洞
        filled_mask = ndimage.binary_fill_holes(largest_mask).astype(np.uint8)

        # 转回PyTorch张量
        processed_mask[b, 0] = torch.from_numpy(filled_mask).to(mask.device)

    return processed_mask

class Mask_Guided_Region_Module(nn.Module):
    def __init__(self, in_channel, out_channel):
        super(Mask_Guided_Region_Module, self).__init__()

        # search region nodes linear transformation
        self.support = nn.Sequential(
            nn.Conv2d(in_channel, in_channel, 1),
#             nn.Conv2d(in_channel, in_channel, 3, 1,1),
            nn.BatchNorm2d(in_channel ),
            nn.GELU()
        )

        # target template nodes linear transformation
        self.query = nn.Sequential(
            nn.Conv2d(in_channel, in_channel , 1),
#             nn.Conv2d(in_channel, in_channel, 3, 1,1),
            nn.BatchNorm2d(in_channel),
            nn.GELU()
        )

        # linear transformation for message passing
        self.g = nn.Sequential(
            nn.Conv2d(in_channel, in_channel, 3, 1,1),
            nn.Conv2d(in_channel, in_channel, 1, 1),
            nn.BatchNorm2d(in_channel),
            nn.GELU(),
        )


        # aggregated feature
        self.fi = nn.Sequential(
            nn.Conv2d(169, out_channel, 1, 1),
            nn.BatchNorm2d(out_channel),
            nn.GELU(),
        )

        self.fi1 = nn.Sequential(
            nn.Conv2d(in_channel * 2, out_channel, 5, 1,2),
            nn.BatchNorm2d(out_channel),
            nn.GELU(),
        )

        # self.fi2 = nn.Sequential(SENetV2FeatureFusionModule(50,4),
        #     nn.BatchNorm2d(out_channel),
        #     nn.GELU(),)

    def forward(self, zf, xf):
        # linear transformation
        xf_shape = xf.shape
        zf_shape = zf.shape
        xf_trans = self.query(xf)
        zf_trans = self.support(zf)
        xf_g = self.g(xf)

        # linear transformation for message passing
        # xf_g = self.g(xf)
        # zf_g = self.g(zf)

        # calculate similarity
        shape_x = xf_trans.shape
        shape_z = zf_trans.shape

        zf_trans_plain = zf_trans.view(-1, shape_z[1], shape_z[2] * shape_z[3])#B,C,N_Z
        # zf_g_plain = zf_g.view(-1, zf_shape[1], shape_z[2] * shape_z[3])
        xf_trans_plain = xf_trans.view(-1, shape_x[1], shape_x[2] * shape_x[3])

        zf_plain = zf.view(-1, zf_shape[1], zf_shape[2] * zf_shape[3])#B,C,N_Z
        xf_plain = xf.view(-1, xf_shape[1], xf_shape[2] * xf_shape[3])
        zf_s = softThresholdingOperation(torch.matmul(zf_trans_plain.permute(0, 2, 1),zf_trans_plain),topk=128) #B,N_Z,N_Z(169)
        # xf_s = softThresholdingOperation(torch.matmul(xf_trans_plain,xf_trans_plain.permute(0, 2, 1)),topk=256)
        zf_self = torch.matmul(zf_plain, zf_s)#B,C,N_Z
        # xf_self = torch.matmul(xf_plain, xf_s)#B,C,N_X
        zf_plain = zf_plain + zf_self #B,C,N_Z
        # xf_plain = xf_plain + xf_self
        si = softThresholdingOperation(torch.matmul(zf_plain.permute(0, 2, 1), xf_plain),topk=512)#B,N_Z,N_X
        si2 = softThresholdingOperation(torch.matmul(zf_plain.permute(0, 2, 1),xf_trans_plain),topk=512)#B,N_Z,N_X
        si = si + si2 #B,N_Z,N_X
        # mask1 = (torch.mean(si,dim=1) !=0)#B,1,N_X
        mask2 = (torch.mean(si2,dim=1) !=0)#B,1,N_X
        # mask = (mask1 & mask2).float().unsqueeze(1).view(-1,1, xf_shape[2] , xf_shape[3])#B,1,H_X,W_X
        mask = mask2.unsqueeze(1).float().unsqueeze(1).view(-1,1, xf_shape[2] , xf_shape[3])
        mask = keep_largest_connected_component(mask)
#         mask = (mask != 0).float().unsqueeze(1).view(-1,1, xf_shape[2] , xf_shape[3])
#         output = xf * (1-mask) + xf_g * mask
        si = si.view(-1, zf_shape[2] * zf_shape[3], xf_shape[2] , xf_shape[3])#B,N_Z,H_X,W_X
        concat = torch.concat([self.fi(si) , xf_g * mask],dim=1)#B,2C,H_X,W_X
        output = self.fi1(concat)
        # si = torch.matmul(zf_trans_plain.permute(0, 2, 1), xf_trans_plain)#B,N_Z,N_X
        # si = si.view(-1, zf_shape[2] * zf_shape[3], xf_shape[2] , xf_shape[3])#B,N_Z,H_X,W_X
#         output = torch.concat([xf,xf_g*mask],dim=1)
#         output = self.fi(output)

        return output

if __name__ == '__main__':
    channel = 256
    input = torch.randn(3, channel, 13, 13)  # b c h w输入
    input2 = torch.randn(3, channel, 25, 25)  # b c h w输入
    wtconv = Mask_Guided_Region_Module(in_channel=256, out_channel=256)
    output = wtconv(input,input2)
    print(input.size(),input2.size(),output.size())

