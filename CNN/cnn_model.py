import torch
import torch.nn as nn
import torch.nn.functional as F


class SEBlock(nn.Module):
    """
    SE-Block: 让模型自动学习“强调”哪些通道（比如自动发现电压通道重要）
    """

    def __init__(self, channel, reduction=16):
        super(SEBlock, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool1d(1)
        self.fc = nn.Sequential(
            nn.Linear(channel, channel // reduction, bias=False),
            nn.GELU(),
            nn.Linear(channel // reduction, channel, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1)
        return x * y.expand_as(x)


class ResBlock1D(nn.Module):
    def __init__(self, in_channels, out_channels, dilation=1):
        super(ResBlock1D, self).__init__()
        self.conv1 = nn.Conv1d(in_channels, out_channels, kernel_size=3, padding=dilation, dilation=dilation)
        self.bn1 = nn.BatchNorm1d(out_channels)
        self.conv2 = nn.Conv1d(out_channels, out_channels, kernel_size=3, padding=1, dilation=1)
        self.bn2 = nn.BatchNorm1d(out_channels)
        self.se = SEBlock(out_channels)
        self.shortcut = nn.Sequential()
        if in_channels != out_channels:
            self.shortcut = nn.Conv1d(in_channels, out_channels, kernel_size=1)

    def forward(self, x):
        out = F.gelu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out = self.se(out)
        out += self.shortcut(x)
        return F.gelu(out)


class GridCNN(nn.Module):
    def __init__(self, in_channels, hidden_dim=256):
        super(GridCNN, self).__init__()

        # 🔥【Idea 2 实现】
        # 输入维度 = 原始物理通道(5) + 削减后的PE(4) + 动态全局特征(3) = 12
        # 我们在 forward 里拼接，所以这里的 entry 维度要预留好

        # 这里的 +3 代表: Mean_P_load, Mean_Q_load, Mean_P_cap
        self.entry = nn.Sequential(
            nn.Conv1d(in_channels + 3, hidden_dim, kernel_size=1),
            nn.BatchNorm1d(hidden_dim),
            nn.GELU()
        )

        self.layer1 = ResBlock1D(hidden_dim, hidden_dim, dilation=1)
        self.layer2 = ResBlock1D(hidden_dim, hidden_dim, dilation=2)
        self.layer3 = ResBlock1D(hidden_dim, hidden_dim, dilation=4)
        self.layer4 = ResBlock1D(hidden_dim, hidden_dim, dilation=8)

        self.decoder = nn.Sequential(
            nn.Conv1d(hidden_dim, hidden_dim // 2, kernel_size=1),
            nn.GELU(),
            nn.Conv1d(hidden_dim // 2, 1, kernel_size=1),
            nn.Hardsigmoid()
        )

    def forward(self, x):
        # x shape: [Batch, In_Channels, Nodes]
        # In_Channels 包含: 物理特征(5) + PE(4)

        # --- 🔥【Idea 2: 动态计算全局特征】---
        # 假设物理特征在前 5 位:
        # 0: Load P, 1: Load Q, 2: P_max, 3: V_mag, 4: V_angle

        # 1. 全网平均有功负荷 (Mean Load P)
        mean_p_load = x[:, 0:1, :].mean(dim=2, keepdim=True)
        # 2. 全网平均无功负荷 (Mean Load Q)
        mean_q_load = x[:, 1:2, :].mean(dim=2, keepdim=True)
        # 3. 全网平均发电容量 (Mean Gen Cap) - 反映电网规模/光照强度
        mean_p_cap = x[:, 2:3, :].mean(dim=2, keepdim=True)

        # 广播拼接
        N = x.shape[2]
        global_feats = torch.cat([mean_p_load, mean_q_load, mean_p_cap], dim=1)  # [B, 3, 1]
        global_feats_exp = global_feats.expand(-1, -1, N)  # [B, 3, N]

        # 拼接到原始输入
        x_input = torch.cat([x, global_feats_exp], dim=1)

        # --- 进网络 ---
        x = self.entry(x_input)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        out = self.decoder(x)
        return out