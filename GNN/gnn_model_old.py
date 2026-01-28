import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import TransformerConv, BatchNorm


class GridGNN(torch.nn.Module):
    def __init__(self,
                 num_node_features,
                 num_edge_features,
                 hidden_dim=128,
                 heads=4):
        super(GridGNN, self).__init__()

        # -------------------------------------------------------
        # 1. 特征编码层 (Embedding)
        # -------------------------------------------------------
        self.node_enc = nn.Sequential(
            nn.Linear(num_node_features, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.PReLU()
        )

        self.edge_enc = nn.Sequential(
            nn.Linear(num_edge_features, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.PReLU()
        )

        # -------------------------------------------------------
        # 2. Graph Transformer 层
        # -------------------------------------------------------
        # TransformerConv 的特点：
        # 它可以捕捉节点特征之间的深层交互 (Self-Attention)
        # edge_dim 参数允许我们把线路阻抗(R, X)作为注意力计算的辅助信息

        # 第一层
        self.conv1 = TransformerConv(
            in_channels=hidden_dim,
            out_channels=hidden_dim // heads,  # TransformerConv 输出是 heads * out_channels
            heads=heads,
            edge_dim=hidden_dim,
            dropout=0.2,
            beta=True  # 开启门控机制，让模型决定保留多少原信息
        )
        self.bn1 = BatchNorm(hidden_dim)

        # 第二层
        self.conv2 = TransformerConv(
            in_channels=hidden_dim,
            out_channels=hidden_dim // heads,
            heads=heads,
            edge_dim=hidden_dim,
            dropout=0.2,
            beta=True
        )
        self.bn2 = BatchNorm(hidden_dim)

        # 第三层
        self.conv3 = TransformerConv(
            in_channels=hidden_dim,
            out_channels=hidden_dim // heads,
            heads=heads,
            edge_dim=hidden_dim,
            dropout=0.2,
            beta=True
        )
        self.bn3 = BatchNorm(hidden_dim)

        # -------------------------------------------------------
        # 3. 解码输出层
        # -------------------------------------------------------
        self.decoder = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, 1),
            nn.Hardsigmoid()
        )

    def forward(self, data):
        x, edge_index, edge_attr = data.x, data.edge_index, data.edge_attr

        # --- 1. 编码 ---
        x = self.node_enc(x)
        edge_attr = self.edge_enc(edge_attr)

        # --- 2. Transformer 交互 ---

        # Layer 1
        x_in = x
        x = self.conv1(x, edge_index, edge_attr=edge_attr)
        x = self.bn1(x)
        x = F.relu(x)
        x = x + x_in  # 残差连接

        # Layer 2
        x_in = x
        x = self.conv2(x, edge_index, edge_attr=edge_attr)
        x = self.bn2(x)
        x = F.relu(x)
        x = x + x_in

        # Layer 3
        x_in = x
        x = self.conv3(x, edge_index, edge_attr=edge_attr)
        x = self.bn3(x)
        x = F.relu(x)
        x = x + x_in

        # --- 3. 输出 ---
        out = self.decoder(x)

        return out