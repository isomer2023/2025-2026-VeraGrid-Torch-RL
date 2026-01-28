import math
import torch
import torch.nn as nn

def process_batch_features(x_raw, x_mean, x_std):
    """
    输入: (B, 5, N) -> P, Q, PV, V_mag, V_ang
    输出: (B, N, 6) -> P, Q, PV, V_phys, sin, cos
    """
    # 前3个通道 (P, Q, PV) 使用统计标准化
    x_norm_part = (x_raw[:, :3, :] - x_mean[:, :3, :]) / (x_std[:, :3, :] + 1e-6)

    # V_mag (通道3) 使用物理归一化 (V-1.0)/0.05
    v_raw = x_raw[:, 3, :]
    v_phys = (v_raw - 1.0) / 0.05

    # V_ang (通道4) 使用 sin/cos 嵌入
    ang = x_raw[:, 4, :]
    s = torch.sin(ang)
    c = torch.cos(ang)

    # 拼接
    x_feat = torch.cat([
        x_norm_part,  # (B, 3, N)
        v_phys.unsqueeze(1),  # (B, 1, N)
        s.unsqueeze(1),  # (B, 1, N)
        c.unsqueeze(1)  # (B, 1, N)
    ], dim=1)  # -> (B, 6, N)

    return x_feat.transpose(1, 2).contiguous()  # -> (B, N, 6)

class NativeGCNLayer(nn.Module):
    """ 原生实现的 GCN 层: X' = A * X * W """

    def __init__(self, in_features, out_features, dropout=0.0):
        super().__init__()
        self.linear = nn.Linear(in_features, out_features, bias=False)
        self.act = nn.GELU()
        self.drop = nn.Dropout(dropout)

    def forward(self, x, adj):
        # x: (B, N, Fin), adj: (N, N)
        out = self.linear(x)  # (B, N, Fout)
        # 矩阵乘法: (N,N) x (B,N,F) -> (B,N,F)
        # 既然 B 在前，可以用 einsum 或者把 B 移到后面
        out = torch.einsum('nm, bmf -> bnf', adj, out)
        return self.drop(self.act(out))

class HybridMultiHeadAttention(nn.Module):
    """ 混合注意力: 一半头看物理距离，一半头自由学习 """

    def __init__(self, d_model, n_heads, dropout=0.0):
        super().__init__()
        assert d_model % n_heads == 0
        self.n_heads = n_heads
        self.d_head = d_model // n_heads
        self.n_phys = n_heads // 2  # 物理头数量

        self.qkv = nn.Linear(d_model, 3 * d_model, bias=False)
        self.proj = nn.Linear(d_model, d_model, bias=False)
        self.drop = nn.Dropout(dropout)

        # 物理头的权重系数
        self.beta = nn.Parameter(torch.ones(1, self.n_phys, 1, 1))

    def forward(self, x, attn_bias):
        B, N, D = x.shape
        qkv = self.qkv(x)
        q, k, v = torch.chunk(qkv, 3, dim=-1)

        # (B, H, N, d)
        q = q.view(B, N, self.n_heads, self.d_head).transpose(1, 2)
        k = k.view(B, N, self.n_heads, self.d_head).transpose(1, 2)
        v = v.view(B, N, self.n_heads, self.d_head).transpose(1, 2)

        scale = 1.0 / math.sqrt(self.d_head)
        logits = (q @ k.transpose(-2, -1)) * scale  # (B, H, N, N)

        # --- 混合处理 ---
        phys_logits = logits[:, :self.n_phys, :, :]
        free_logits = logits[:, self.n_phys:, :, :]

        # 物理头加上 Bias
        phys_logits = phys_logits + (self.beta * attn_bias)

        # 合并
        logits = torch.cat([phys_logits, free_logits], dim=1)

        attn = torch.softmax(logits, dim=-1)
        attn = self.drop(attn)

        out = attn @ v
        out = out.transpose(1, 2).contiguous().view(B, N, D)
        return self.drop(self.proj(out))

class HybridGridTransformer(nn.Module):
    def __init__(self, in_dim=6, d_model=128, n_heads=8, n_layers=6, d_ff=256, dropout=0.1):
        super().__init__()

        self.embedding = nn.Linear(in_dim, d_model)

        # GCN 分支 (Local)
        self.gcn1 = NativeGCNLayer(d_model, d_model, dropout)
        self.gcn2 = NativeGCNLayer(d_model, d_model, dropout)
        self.gcn_norm = nn.LayerNorm(d_model)

        # Transformer 分支 (Global)
        self.layers = nn.ModuleList([
            nn.ModuleDict({
                'norm1': nn.LayerNorm(d_model),
                'attn': HybridMultiHeadAttention(d_model, n_heads, dropout),
                'norm2': nn.LayerNorm(d_model),
                'ff': nn.Sequential(
                    nn.Linear(d_model, d_ff), nn.GELU(), nn.Dropout(dropout),
                    nn.Linear(d_ff, d_model), nn.Dropout(dropout)
                )
            }) for _ in range(n_layers)
        ])

        # Head
        self.head = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, d_model // 2),
            nn.GELU(),
            nn.Linear(d_model // 2, 1)
        )

    def forward(self, x, adj, attn_bias):
        # x: (B, N, 6)
        h = self.embedding(x)

        # GCN 提取局部特征并残差连接
        h_local = self.gcn1(h, adj)
        h_local = self.gcn2(h_local, adj)
        h = self.gcn_norm(h + h_local)

        # Transformer 提取全局特征
        for layer in self.layers:
            h_norm = layer['norm1'](h)
            h = h + layer['attn'](h_norm, attn_bias)

            h_norm = layer['norm2'](h)
            h = h + layer['ff'](h_norm)

        out = self.head(h).squeeze(-1)  # (B, N)
        return torch.sigmoid(out)

class BaselineGNN(nn.Module):
    def __init__(self, in_dim=6, d_model=128):
        super().__init__()
        self.proj = nn.Linear(in_dim, d_model)
        self.gcn1 = NativeGCNLayer(d_model, d_model)
        self.gcn2 = NativeGCNLayer(d_model, d_model)
        self.gcn3 = NativeGCNLayer(d_model, d_model)
        self.head = nn.Linear(d_model, 1)

    def forward(self, x, adj):
        h = self.proj(x)
        h = h + self.gcn1(h, adj)  # Residuals for better training
        h = h + self.gcn2(h, adj)
        h = h + self.gcn3(h, adj)
        return torch.sigmoid(self.head(h)).squeeze(-1)

class BaselineMLP(nn.Module):
    def __init__(self, in_dim=6, d_model=128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, d_model),
            nn.LayerNorm(d_model),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(d_model, d_model),
            nn.LayerNorm(d_model),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(d_model, 1)
        )

    def forward(self, x, adj=None):
        # adj is ignored for MLP
        out = self.net(x)
        return torch.sigmoid(out).squeeze(-1)
