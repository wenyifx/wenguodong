import torch
import torch.nn as nn
import torch.nn.functional as F

__all__ = ["DepthSepConvV1", "DepthSepConvV2", "DepthSepConvV4"]


class SeBlock(nn.Module):
    def __init__(self, in_channel, reduction=4):
        super().__init__()
        self.Squeeze = nn.AdaptiveAvgPool2d(1)

        self.Excitation = nn.Sequential()
        self.Excitation.add_module(
            "FC1", nn.Conv2d(in_channel, in_channel // reduction, kernel_size=1)
        )
        self.Excitation.add_module("ReLU", nn.ReLU())
        self.Excitation.add_module(
            "FC2", nn.Conv2d(in_channel // reduction, in_channel, kernel_size=1)
        )
        self.Excitation.add_module("Sigmoid", nn.Sigmoid())

    def forward(self, x):
        y = self.Squeeze(x)
        ouput = self.Excitation(y)
        return x * (ouput.expand_as(x))


# 原始DepthSepConv
class DepthSepConvV1(nn.Module):
    def __init__(self, inp, oup, dw_size, stride, use_se):
        super(DepthSepConv, self).__init__()
        self.stride = stride
        self.inp = inp
        self.oup = oup
        self.dw_size = dw_size
        self.dw_sp = nn.Sequential(
            nn.Conv2d(
                self.inp,
                self.inp,
                kernel_size=self.dw_size,
                stride=self.stride,
                padding=(dw_size - 1) // 2,
                groups=self.inp,
                bias=False,
            ),
            nn.BatchNorm2d(self.inp),
            nn.Hardswish(),
            SeBlock(self.inp, reduction=16) if use_se else nn.Sequential(),
            nn.Conv2d(
                self.inp, self.oup, kernel_size=1, stride=1, padding=0, bias=False
            ),
            nn.BatchNorm2d(self.oup),
            nn.Hardswish(),
        )

    def forward(self, x):
        y = self.dw_sp(x)
        return y


# 本文
class DepthSepConvV2(nn.Module):
    def __init__(self, inp, oup, dw_size, stride, use_se):
        super(DepthSepConvV2, self).__init__()
        self.stride = stride
        self.use_res = stride == 1 and inp == oup

        self.dw_sp = nn.Sequential(
            nn.Conv2d(
                inp, inp, kernel_size=dw_size, stride=stride,
                padding=(dw_size - 1) // 2, groups=inp, bias=False,
            ),
            nn.BatchNorm2d(inp),
            nn.Hardswish(),
            SeBlock(inp, reduction=8 if use_se else 16) if use_se else nn.Sequential(),  # 降低reduction
            nn.Conv2d(inp, oup, kernel_size=1, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(oup),
        )
        self.act = nn.Hardswish()

    def forward(self, x):
        y = self.dw_sp(x)
        if self.use_res:
            y = y + x
        return self.act(y)


# 版本4：自适应卷积核（根据特征自动调整）
class DepthSepConvV4(nn.Module):
    def __init__(self, inp, oup, dw_size, stride, use_se):
        super(DepthSepConvV4, self).__init__()
        self.stride = stride

        # 动态选择卷积核大小
        if dw_size == 3:
            # 3x3卷积
            self.dw_conv = nn.Conv2d(inp, inp, kernel_size=3, stride=stride,
                                     padding=1, groups=inp, bias=False)
        else:
            # 5x5用两个3x3代替（更高效）
            self.dw_conv = nn.Sequential(
                nn.Conv2d(inp, inp, kernel_size=3, stride=stride,
                          padding=1, groups=inp, bias=False),
                nn.BatchNorm2d(inp),
                nn.Conv2d(inp, inp, kernel_size=3, stride=1,
                          padding=1, groups=inp, bias=False),
            )

        self.norm = nn.BatchNorm2d(inp)
        self.act1 = nn.Hardswish()

        # 增强的SE模块（添加局部注意力）
        if use_se:
            self.se = nn.Sequential(
                SeBlock(inp, reduction=8),
                nn.Conv2d(inp, inp, kernel_size=3, padding=1, groups=inp, bias=False),
                nn.BatchNorm2d(inp),
                nn.Sigmoid()
            )
        else:
            self.se = SeBlock(inp, reduction=16) if use_se else nn.Sequential()

        # 输出
        self.output = nn.Sequential(
            nn.Conv2d(inp, oup, kernel_size=1, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(oup),
            nn.Hardswish(),
        )

    def forward(self, x):
        y = self.dw_conv(x)
        y = self.norm(y)
        y = self.act1(y)
        if hasattr(self, 'se') and len(self.se) > 0:
            y = y * self.se(y)
        return self.output(y)