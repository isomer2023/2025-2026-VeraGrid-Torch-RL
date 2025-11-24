import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GATv2Conv, BatchNorm


class GridGNN(torch.nn.Module):
    def __init__(self, num_node_features, num_edge_features, hidden_dim=128, heads=4):
        super(GridGNN, self).__init__()

        # -------------------------------------------------------
        # 1. 特征编码层 (Embedding)
        # -------------------------------------------------------
        # 将物理特征映射到高维空间
        self.node_enc = nn.Sequential(
            nn.Linear(num_node_features, hidden_dim),
            nn.BatchNorm1d(hidden_dim),  # 归一化
            nn.PReLU()  # PReLU 比 ReLU 更不容易"死"掉
        )

        self.edge_enc = nn.Sequential(
            nn.Linear(num_edge_features, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.PReLU()
        )

        # -------------------------------------------------------
        # 2. 图神经网络层 (GNN Layers)
        # -------------------------------------------------------
        # 第一层 GNN
        self.conv1 = GATv2Conv(hidden_dim, hidden_dim, heads=heads, concat=False, edge_dim=hidden_dim)
        self.bn1 = BatchNorm(hidden_dim)

        # 第二层 GNN
        self.conv2 = GATv2Conv(hidden_dim, hidden_dim, heads=heads, concat=False, edge_dim=hidden_dim)
        self.bn2 = BatchNorm(hidden_dim)

        # 第三层 GNN (更深一点)
        self.conv3 = GATv2Conv(hidden_dim, hidden_dim, heads=heads, concat=False, edge_dim=hidden_dim)
        self.bn3 = BatchNorm(hidden_dim)

        # -------------------------------------------------------
        # 3. 解码输出层 (Decoder)
        # -------------------------------------------------------
        self.decoder = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, 1),
            nn.Hardsigmoid()  # 强制输出 [0, 1]
        )

    def forward(self, data):
        x, edge_index, edge_attr = data.x, data.edge_index, data.edge_attr

        # --- Step 1: 编码 ---
        x = self.node_enc(x)
        edge_attr = self.edge_enc(edge_attr)

        # --- Step 2: 消息传递 (带残差连接) ---

        # Layer 1
        x_in = x  # 保存输入用于残差
        x = self.conv1(x, edge_index, edge_attr=edge_attr)
        x = self.bn1(x)
        x = F.relu(x)
        x = x + x_in  # <--- 残差连接 (ResNet)

        # Layer 2
        x_in = x
        x = self.conv2(x, edge_index, edge_attr=edge_attr)
        x = self.bn2(x)
        x = F.relu(x)
        x = x + x_in  # <--- 残差连接

        # Layer 3
        x_in = x
        x = self.conv3(x, edge_index, edge_attr=edge_attr)
        x = self.bn3(x)
        x = F.relu(x)
        x = x + x_in  # <--- 残差连接

        # --- Step 3: 输出 ---
        out = self.decoder(x)

        return out