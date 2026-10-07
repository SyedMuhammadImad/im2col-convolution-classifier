"""project implementation retained from author/assisted source; see ATTRIBUTION.md."""
import math
import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
NUM_CLASSES=6
def im2col(X: torch.Tensor, kH: int, kW: int, stride: int=1, pad: int=0):
    """
    2-D im2col.
    X      : (N, C, H, W)
    return : (N, C*kH*kW, out_H*out_W)
    """
    if X.ndim != 4 or min(X.shape) < 1: raise ValueError("Invalid input shape")
    if any(not isinstance(v, int) or v < 1 for v in (kH, kW, stride)) or not isinstance(pad, int) or pad < 0:
        raise ValueError("Invalid kernel, stride or padding")
    N, C, H, W = X.shape
    if H+2*pad < kH or W+2*pad < kW: raise ValueError("Kernel exceeds padded input")
    out_H = (H + 2 * pad - kH) // stride + 1
    out_W = (W + 2 * pad - kW) // stride + 1
    X_pad = F.pad(X, (pad, pad, pad, pad)) if pad > 0 else X
    cols = []
    for i in range(out_H):
        for j in range(out_W):
            patch = X_pad[:, :, i * stride:i * stride + kH, j * stride:j * stride + kW]
            cols.append(patch.reshape(N, -1))
    return torch.stack(cols, dim=-1)

def col2im(col: torch.Tensor, X_shape: tuple, kH: int, kW: int, stride: int=1, pad: int=0):
    """
    2-D col2im (gradient accumulation).
    col    : (N, C*kH*kW, out_H*out_W)
    X_shape: (N, C, H, W)
    return : (N, C, H, W)
    """
    if len(X_shape) != 4 or any(not isinstance(v, int) or v < 1 for v in X_shape): raise ValueError("Invalid output shape")
    if any(not isinstance(v, int) or v < 1 for v in (kH, kW, stride)) or not isinstance(pad, int) or pad < 0:
        raise ValueError("Invalid kernel, stride or padding")
    N, C, H, W = X_shape
    if H+2*pad < kH or W+2*pad < kW: raise ValueError("Kernel exceeds padded input")
    out_H = (H + 2 * pad - kH) // stride + 1
    out_W = (W + 2 * pad - kW) // stride + 1
    if tuple(col.shape) != (N, C*kH*kW, out_H*out_W): raise ValueError("Column shape mismatch")
    X_pad = torch.zeros((N, C, H + 2 * pad, W + 2 * pad), device=col.device, dtype=col.dtype)
    for i in range(out_H):
        for j in range(out_W):
            patch = col[:, :, i * out_W + j].reshape(N, C, kH, kW)
            X_pad[:, :, i * stride:i * stride + kH, j * stride:j * stride + kW] += patch
    if pad > 0:
        return X_pad[:, :, pad:-pad, pad:-pad]
    return X_pad

class Conv2d(nn.Module):
    """
    2-D convolution via im2col + matmul. Autograd handles backward.
    """

    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, bias=True):
        super().__init__()
        if any(not isinstance(v, int) or v < 1 for v in (in_channels, out_channels, kernel_size, stride)) or not isinstance(padding, int) or padding < 0:
            raise ValueError("Invalid convolution dimensions")
        self.in_ch = in_channels
        self.out_ch = out_channels
        self.kH = self.kW = kernel_size
        self.stride = stride
        self.pad = padding
        self.weight = nn.Parameter(torch.empty(out_channels, in_channels, self.kH, self.kW))
        self.bias_param = nn.Parameter(torch.empty(out_channels)) if bias else None
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        if self.bias_param is not None:
            fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weight)
            bound = 1 / math.sqrt(fan_in) if fan_in > 0 else 0
            nn.init.uniform_(self.bias_param, -bound, bound)

    def forward(self, x):
        if x.ndim != 4 or x.shape[1] != self.in_ch: raise ValueError("Convolution channel mismatch")
        N, C, H, W = x.shape
        out_H = (H + 2 * self.pad - self.kH) // self.stride + 1
        out_W = (W + 2 * self.pad - self.kW) // self.stride + 1
        col = im2col(x, self.kH, self.kW, self.stride, self.pad)
        w_flat = self.weight.view(self.out_ch, -1)
        out = (w_flat @ col).reshape(N, self.out_ch, out_H, out_W)
        if self.bias_param is not None:
            out = out + self.bias_param.view(1, self.out_ch, 1, 1)
        return out

class MaxPool2d(nn.Module):
    """
    2-D max pooling with non-overlapping windows (stride == kernel_size).
    """

    def __init__(self, kernel_size):
        super().__init__()
        if not isinstance(kernel_size, int) or kernel_size < 1: raise ValueError("Invalid pool size")
        self.k = kernel_size

    def forward(self, x):
        N, C, H, W = x.shape
        out_H, out_W = (H // self.k, W // self.k)
        col = im2col(x, self.k, self.k, stride=self.k, pad=0)
        col = col.reshape(N, C, self.k * self.k, out_H * out_W)
        out = col.max(dim=2).values
        return out.reshape(N, C, out_H, out_W)

class MyCNN(nn.Module):

    def __init__(self, num_classes=NUM_CLASSES):
        super().__init__()
        self.conv1 = Conv2d(1, 32, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(32)
        self.pool = MaxPool2d(2)
        self.conv2 = Conv2d(32, 64, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(64)
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(64 * 3 * 3, 384)
        self.bn_fc1 = nn.BatchNorm1d(384)
        self.drop1 = nn.Dropout(0.25)
        self.fc2 = nn.Linear(384, 192)
        self.bn_fc2 = nn.BatchNorm1d(192)
        self.drop2 = nn.Dropout(0.25)
        self.fc3 = nn.Linear(192, num_classes)

    def forward(self, x):
        x = self.pool(F.relu(self.bn1(self.conv1(x))))
        x = F.relu(self.bn2(self.conv2(x)))
        x = self.flatten(x)
        x = self.drop1(F.relu(self.bn_fc1(self.fc1(x))))
        x = self.drop2(F.relu(self.bn_fc2(self.fc2(x))))
        return self.fc3(x)
