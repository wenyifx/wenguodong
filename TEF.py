import torch
import torch.nn as nn
import torch.nn.functional as F

__all__ = ['Zoom_cat_Pro']


class Zoom_cat_Pro(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        l, m, s = x[0], x[1], x[2]
        tgt_size = m.shape[2:]
        
        # 使用inplace操作减少内存分配
        l_down = F.adaptive_max_pool2d(l, tgt_size)
        l_down += F.adaptive_avg_pool2d(l, tgt_size)  # inplace相加
        
        s_up = F.interpolate(s, tgt_size, mode='nearest')
        
        lms = torch.cat([l_down, m, s_up], dim=1)
        return lms

       