import math
import torch
import torch.nn as nn


def _index_add(dst: torch.Tensor, index: torch.Tensor, src: torch.Tensor) -> torch.Tensor:
    dst.index_add_(0, index, src)
    return dst


def _ensure_edge_index(edge_index: torch.Tensor) -> torch.Tensor:
    if edge_index.dim() != 2:
        raise ValueError(f"edge_index must be 2D, got shape={tuple(edge_index.shape)}")
    if edge_index.shape[0] == 2:
        return edge_index
    if edge_index.shape[1] == 2:
        return edge_index.t().contiguous()
    raise ValueError(f"edge_index must be (2,E) or (E,2), got shape={tuple(edge_index.shape)}")


class EdgeMPNNLayer(nn.Module):
    """
    edge_attr = [r_pu, x_pu]  -> edge_dim=2
    """

    def __init__(self, d_model: int, edge_dim: int, dropout: float = 0.0):
        super().__init__()
        self.msg = nn.Sequential(
            nn.Linear(d_model + edge_dim, d_model),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model, d_model),
            nn.Dropout(dropout),
        )
        self.upd = nn.Sequential(
            nn.Linear(2 * d_model, d_model),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model, d_model),
            nn.Dropout(dropout),
        )
        self.norm = nn.LayerNorm(d_model)

    def forward(self, h: torch.Tensor, edge_index: torch.Tensor, edge_attr: torch.Tensor) -> torch.Tensor:
        B, N, D = h.shape
        h_flat = h.reshape(B * N, D)

        if edge_index.dim() == 2:
            ei = _ensure_edge_index(edge_index)
            src = ei[0]
            dst = ei[1]

            if edge_attr.dim() != 2:
                raise ValueError(f"edge_attr must be (E,Fe) when edge_index is shared, got {tuple(edge_attr.shape)}")

            offsets = (torch.arange(B, device=h.device, dtype=torch.long) * N).view(B, 1)
            global_src = (src.view(1, -1) + offsets).reshape(-1)
            global_dst = (dst.view(1, -1) + offsets).reshape(-1)
            ea_rep = edge_attr.unsqueeze(0).expand(B, -1, -1).reshape(-1, edge_attr.shape[-1])

        elif edge_index.dim() == 3:
            if edge_index.shape[0] != B or edge_index.shape[1] != 2:
                raise ValueError(f"edge_index must be (B,2,E), got {tuple(edge_index.shape)}")
            if edge_attr.dim() != 3 or edge_attr.shape[0] != B:
                raise ValueError(f"edge_attr must be (B,E,Fe), got {tuple(edge_attr.shape)}")

            src = edge_index[:, 0, :]
            dst = edge_index[:, 1, :]
            offsets = (torch.arange(B, device=h.device, dtype=torch.long) * N).view(B, 1)
            global_src = (src + offsets).reshape(-1)
            global_dst = (dst + offsets).reshape(-1)
            ea_rep = edge_attr.reshape(-1, edge_attr.shape[-1])

        else:
            raise ValueError(f"edge_index must be 2D or 3D, got dim={edge_index.dim()}")

        h_src = h_flat[global_src]
        m_in = torch.cat([h_src, ea_rep], dim=-1)
        m = self.msg(m_in)

        agg = torch.zeros(B * N, D, device=h.device, dtype=h.dtype)
        _index_add(agg, global_dst, m)
        agg = agg.reshape(B, N, D)

        u_in = torch.cat([h, agg], dim=-1)
        dh = self.upd(u_in)
        return self.norm(h + dh)


class BiasMultiHeadAttention(nn.Module):
    def __init__(self, d_model: int, n_heads: int, dropout: float = 0.0, n_phys: int | None = None):
        super().__init__()
        assert d_model % n_heads == 0
        self.n_heads = n_heads
        self.d_head = d_model // n_heads
        self.n_phys = n_phys if n_phys is not None else (n_heads // 2)

        self.qkv = nn.Linear(d_model, 3 * d_model, bias=False)
        self.proj = nn.Linear(d_model, d_model, bias=False)
        self.drop = nn.Dropout(dropout)
        self.beta = nn.Parameter(torch.ones(1, self.n_phys, 1, 1))

    def forward(self, x: torch.Tensor, attn_bias: torch.Tensor | None) -> torch.Tensor:
        B, N, D = x.shape
        qkv = self.qkv(x)
        q, k, v = torch.chunk(qkv, 3, dim=-1)

        q = q.view(B, N, self.n_heads, self.d_head).transpose(1, 2)
        k = k.view(B, N, self.n_heads, self.d_head).transpose(1, 2)
        v = v.view(B, N, self.n_heads, self.d_head).transpose(1, 2)

        scale = 1.0 / math.sqrt(self.d_head)
        logits = (q @ k.transpose(-2, -1)) * scale

        if attn_bias is not None:
            phys_logits = logits[:, :self.n_phys] + (self.beta * attn_bias)
            free_logits = logits[:, self.n_phys:]
            logits = torch.cat([phys_logits, free_logits], dim=1)

        attn = torch.softmax(logits, dim=-1)
        out = self.drop(attn) @ v
        out = out.transpose(1, 2).contiguous().view(B, N, D)
        return self.drop(self.proj(out))


class TransformerBlock(nn.Module):
    def __init__(self, d_model: int, n_heads: int, d_ff: int, dropout: float, n_phys: int | None = None):
        super().__init__()
        self.norm1 = nn.LayerNorm(d_model)
        self.attn = BiasMultiHeadAttention(d_model, n_heads, dropout=dropout, n_phys=n_phys)
        self.norm2 = nn.LayerNorm(d_model)
        self.ff = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_ff, d_model),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor, attn_bias: torch.Tensor | None) -> torch.Tensor:
        x = x + self.attn(self.norm1(x), attn_bias)
        x = x + self.ff(self.norm2(x))
        return x


class HybridGridTransformer(nn.Module):
    """
    Output: (B,N,4) = [alpha, vm_resid, sin(Va), cos(Va)]
    """

    def __init__(
        self,
        in_dim: int = 4,
        d_model: int = 128,
        n_heads: int = 8,
        n_layers: int = 4,
        d_ff: int = 256,
        dropout: float = 0.0,
        edge_dim: int = 2,          # <-- fixed to [r_pu, x_pu]
        mpnn_steps: int = 2,
        use_mpnn: bool = True,
        use_transformer: bool = True,
        n_phys_heads: int | None = None,
        branch_mlp_depth: int = 2,
        normalize_angle: bool = True,
    ):
        super().__init__()
        self.use_mpnn = use_mpnn
        self.use_transformer = use_transformer
        self.normalize_angle = normalize_angle
        self.mpnn_steps = mpnn_steps

        self.embed = nn.Sequential(
            nn.Linear(in_dim, d_model),
            nn.LayerNorm(d_model),
        )

        self.mpnn = nn.ModuleList([
            EdgeMPNNLayer(d_model=d_model, edge_dim=edge_dim, dropout=dropout)
            for _ in range(mpnn_steps)
        ])

        # fallback if edges not provided
        self.adj_fallback = nn.Sequential(
            nn.Linear(d_model, d_model, bias=False),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        self.adj_norm = nn.LayerNorm(d_model)

        self.blocks = nn.ModuleList([
            TransformerBlock(d_model=d_model, n_heads=n_heads, d_ff=d_ff, dropout=dropout, n_phys=n_phys_heads)
            for _ in range(n_layers)
        ])

        def _make_mlp(out_dim: int):
            layers = []
            dim = d_model
            for _ in range(max(1, branch_mlp_depth - 1)):
                layers += [nn.LayerNorm(dim), nn.Linear(dim, dim), nn.GELU(), nn.Dropout(dropout)]
            layers += [nn.LayerNorm(dim), nn.Linear(dim, out_dim)]
            return nn.Sequential(*layers)

        self.head_alpha = _make_mlp(1)
        self.head_phys = _make_mlp(3)  # [vm_resid, sin, cos]

    def forward(
        self,
        x: torch.Tensor,
        adj: torch.Tensor | None,
        attn_bias: torch.Tensor | None,
        edge_index: torch.Tensor | None = None,
        edge_attr: torch.Tensor | None = None,
    ) -> torch.Tensor:
        h = self.embed(x)

        if self.use_mpnn and (edge_index is not None) and (edge_attr is not None):
            for layer in self.mpnn:
                h = layer(h, edge_index=edge_index, edge_attr=edge_attr)
        else:
            if adj is None:
                raise ValueError("adj is required when edge_index/edge_attr are not provided.")
            dh = self.adj_fallback(h)
            h2 = torch.bmm(adj, dh)
            h = self.adj_norm(h + h2)

        if self.use_transformer:
            for blk in self.blocks:
                h = blk(h, attn_bias)

        alpha = torch.sigmoid(self.head_alpha(h).squeeze(-1))  # (B,N)

        phys = self.head_phys(h)  # (B,N,3)
        vm_resid = phys[..., 0]
        s = phys[..., 1]
        c = phys[..., 2]

        if self.normalize_angle:
            norm = torch.sqrt(s * s + c * c + 1e-8)
            s = s / norm
            c = c / norm

        return torch.stack([alpha, vm_resid, s, c], dim=-1)
