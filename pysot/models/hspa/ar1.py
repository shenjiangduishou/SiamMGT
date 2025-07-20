import pywt
import pywt.data
import torch
from torch import nn
from functools import partial
import torch.nn.functional as F
import cv2
import numpy as np

# 论文地址 https://arxiv.org/pdf/2407.05848
# 微信公众号：AI缝合术

def create_wavelet_filter(wave, in_size, out_size, type=torch.float):
    w = pywt.Wavelet(wave)
    dec_hi = torch.tensor(w.dec_hi[::-1], dtype=type)
    dec_lo = torch.tensor(w.dec_lo[::-1], dtype=type)
    dec_filters = torch.stack([dec_lo.unsqueeze(0) * dec_lo.unsqueeze(1),
                               dec_lo.unsqueeze(0) * dec_hi.unsqueeze(1),
                               dec_hi.unsqueeze(0) * dec_lo.unsqueeze(1),
                               dec_hi.unsqueeze(0) * dec_hi.unsqueeze(1)], dim=0)

    dec_filters = dec_filters[:, None].repeat(in_size, 1, 1, 1)

    rec_hi = torch.tensor(w.rec_hi[::-1], dtype=type).flip(dims=[0])
    rec_lo = torch.tensor(w.rec_lo[::-1], dtype=type).flip(dims=[0])
    rec_filters = torch.stack([rec_lo.unsqueeze(0) * rec_lo.unsqueeze(1),
                               rec_lo.unsqueeze(0) * rec_hi.unsqueeze(1),
                               rec_hi.unsqueeze(0) * rec_lo.unsqueeze(1),
                               rec_hi.unsqueeze(0) * rec_hi.unsqueeze(1)], dim=0)

    rec_filters = rec_filters[:, None].repeat(out_size, 1, 1, 1)

    return dec_filters, rec_filters


def wavelet_transform(x, filters):
    b, c, h, w = x.shape
    pad = (filters.shape[2] // 2 - 1, filters.shape[3] // 2 - 1)
    device = x.device
    filters = filters.to(device)
    x = F.conv2d(x, filters, stride=2, groups=c, padding=pad)
    x = x.reshape(b, c, 4, h // 2, w // 2)
    return x


def inverse_wavelet_transform(x, filters):
    b, c, _, h_half, w_half = x.shape
    pad = (filters.shape[2] // 2 - 1, filters.shape[3] // 2 - 1)
    device = x.device
    filters = filters.to(device)
    x = x.reshape(b, c * 4, h_half, w_half)
    x = F.conv_transpose2d(x, filters, stride=2, groups=c, padding=pad)
    return x


# Wavelet Transform Conv(WTConv2d)
class WTConv2d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=5, stride=1, bias=True, wt_levels=1, wt_type='db1',ex_downsampl=False):
        super(WTConv2d, self).__init__()

        assert in_channels == out_channels

        self.in_channels = in_channels
        self.wt_levels = wt_levels
        self.stride = stride
        self.dilation = 1

        self.wt_filter, self.iwt_filter = create_wavelet_filter(wt_type, in_channels, in_channels, torch.float)
        self.wt_filter = nn.Parameter(self.wt_filter, requires_grad=False)
        self.iwt_filter = nn.Parameter(self.iwt_filter, requires_grad=False)

        self.wt_function = partial(wavelet_transform, filters=self.wt_filter)
        self.iwt_function = partial(inverse_wavelet_transform, filters=self.iwt_filter)

        self.base_conv = nn.Conv2d(in_channels, in_channels, kernel_size, padding='same', stride=1, dilation=1,
                                   groups=in_channels, bias=bias)
        self.base_scale = _ScaleModule([1, in_channels, 1, 1])

        self.wavelet_convs = nn.ModuleList(
            [nn.Conv2d(in_channels * 4, in_channels * 4, kernel_size, padding='same', stride=1, dilation=1,
                       groups=in_channels * 4, bias=False) for _ in range(self.wt_levels)]
        )
        self.wavelet_scale = nn.ModuleList(
            [_ScaleModule([1, in_channels * 4, 1, 1], init_scale=0.1) for _ in range(self.wt_levels)]
        )

        if self.stride > 1:
            # 注册为buffer而非Parameter
            self.register_buffer('stride_filter',
                                 torch.ones(in_channels, 1, 1, 1))
            # 动态设备绑定
            self.do_stride = self._create_stride_conv
        else:
            self.do_stride = None

        if ex_downsampl:
            self.final_down = nn.Sequential(
                nn.Conv2d(in_channels, in_channels,
                          kernel_size=2, stride=1, padding=0,  # 核心参数调整
                          groups=in_channels),
                _ScaleModule([1, in_channels, 1, 1])
            )
        else:
            self.final_down = None

    def _create_stride_conv(self, x_in):
        # 每次前向传播时同步设备
        device = x_in.device
        return F.conv2d(x_in,
                        self.stride_filter.to(device),
                        bias=None,
                        stride=self.stride,
                        groups=self.in_channels)

    def forward(self, x):
        # assert x.device == self.stride_filter.device, \
        #     f"Device mismatch: {x.device} vs {self.stride_filter.device}"
        # x = self.do_stride(x)
        if len(x) == 3:
            x = x.unsqueeze(0)

        x_ll_in_levels = []
        x_h_in_levels = []
        shapes_in_levels = []

        curr_x_ll = x

        for i in range(self.wt_levels):
            curr_shape = curr_x_ll.shape
            shapes_in_levels.append(curr_shape)
            if (curr_shape[2] % 2 > 0) or (curr_shape[3] % 2 > 0):
                curr_pads = (0, curr_shape[3] % 2, 0, curr_shape[2] % 2)
                curr_x_ll = F.pad(curr_x_ll, curr_pads)

            curr_x = self.wt_function(curr_x_ll)
            curr_x_ll = curr_x[:, :, 0, :, :]

            shape_x = curr_x.shape
            curr_x_tag = curr_x.reshape(shape_x[0], shape_x[1] * 4, shape_x[3], shape_x[4])
            curr_x_tag = self.wavelet_scale[i](self.wavelet_convs[i](curr_x_tag))
            curr_x_tag = curr_x_tag.reshape(shape_x)

            x_ll_in_levels.append(curr_x_tag[:, :, 0, :, :])
            x_h_in_levels.append(curr_x_tag[:, :, 1:4, :, :])

        next_x_ll = 0

        for i in range(self.wt_levels - 1, -1, -1):
            curr_x_ll = x_ll_in_levels.pop()
            curr_x_h = x_h_in_levels.pop()
            curr_shape = shapes_in_levels.pop()

            curr_x_ll = curr_x_ll + next_x_ll

            curr_x = torch.cat([curr_x_ll.unsqueeze(2), curr_x_h], dim=2)
            next_x_ll = self.iwt_function(curr_x)

            next_x_ll = next_x_ll[:, :, :curr_shape[2], :curr_shape[3]]

        x_tag = next_x_ll
        assert len(x_ll_in_levels) == 0

        x = self.base_scale(self.base_conv(x))
        x = x + x_tag

        if self.do_stride is not None:
            if x.device != self.stride_filter.device:
                self.stride_filter = self.stride_filter.to(x.device)
            x = self.do_stride(x)

        if self.final_down is not None:
            x = self.final_down(x)
        return x


class _ScaleModule(nn.Module):
    def __init__(self, dims, init_scale=1.0, init_bias=0):
        super(_ScaleModule, self).__init__()
        self.dims = dims
        self.weight = nn.Parameter(torch.ones(*dims) * init_scale)
        self.bias = None

    def forward(self, x):
        return torch.mul(self.weight, x)


class DepthwiseSeparableConvWithWTConv2d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3,stride=1,ex_downsampl=False):
        super(DepthwiseSeparableConvWithWTConv2d, self).__init__()

        # 深度卷积：使用 WTConv2d 替换 3x3 卷积
        self.depthwise = WTConv2d(in_channels, in_channels, kernel_size=kernel_size,stride=stride,ex_downsampl=ex_downsampl)

        # 逐点卷积：使用 1x1 卷积
        self.pointwise = nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1, padding=0, bias=False)

    def forward(self, x):
        x = self.depthwise(x)
        x = self.pointwise(x)
        return x

if __name__ == '__main__':
    batch_size =76# 批大小
    channels =256# 输入特征通道数
    height =13#图像高度
    width = 13 # 图像宽度width =32# 图像宽度
    # input = cv2.imread(r'G:\pycharm_progress\SiamGAT-main\img\1.jpg')  # b c h w输入
    # t_input = torch.from_numpy(input.astype(np.float32)).permute(2, 0, 1)  # 转换为 PyTorch 张量并调整维度顺序
    # c,h,w = t_input.shape
    input = torch.randn(batch_size, channels, height, width)
    b,c, h, w = input.shape
    wtconv = DepthwiseSeparableConvWithWTConv2d(in_channels=c, out_channels=c,kernel_size=3,stride=1,ex_downsampl=False)
    # output = wtconv(t_input)
    # output_numpy = output.detach().cpu().numpy()  # 将 torch.Tensor 转换为 numpy.ndarray
    # output_numpy = np.transpose(output_numpy.squeeze(axis=0), (1, 2, 0))
    # while True:
    # cv2.imshow('input', input)
    # cv2.imshow('output0',output_numpy[:,:,0])
    # cv2.imshow('output1', output_numpy[:, :, 1])
    # cv2.imshow('output2', output_numpy[:, :, 2])
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()

    output = wtconv(input)
    print(input.size())
    print(output.size())