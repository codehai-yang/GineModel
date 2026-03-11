import torch
import torch.nn as nn
from torch_geometric.nn import GINEConv

class CostModelV2(nn.Module):
    """
    实验版本：使用175×176矩阵作为节点特征
    前175列：该节点到其他所有节点的回路平均单价
    第176列：该节点所有回路的湿区成本总和
    """

    def __init__(
            self,
            node_feat_dim = 176,    # 节点特征维度：175个单价 + 1个湿区成本
            edge_feat_dim = 4,      # 边特征维度：通断(3维) + 分支长度(1维)
            hidden_dim    = 64,     # 隐藏层维度
            num_layers    = 4       # GINE层数
    ):
        super(CostModelV2, self).__init__()

        # 输入投影层：176维节点特征降维到hidden_dim
        self.input_proj = nn.Linear(node_feat_dim, hidden_dim)

        #TODO 边特征BatchNorm：只对分支长度归一化,长度会出现为0的情况
        # self.edge_bn = nn.BatchNorm1d(1)

        # GINE层和LayerNorm层
        self.convs = nn.ModuleList()
        self.norms = nn.ModuleList()

        for _ in range(num_layers):
            # 每层GINE内部的MLP
            mlp = nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim * 2),
                nn.ReLU(),
                nn.Linear(hidden_dim * 2, hidden_dim)
            )
            self.convs.append(GINEConv(mlp, edge_dim=edge_feat_dim))
            self.norms.append(nn.LayerNorm(hidden_dim))

        # 全局池化后的MLP：直接输出总成本
        # global_add_pool后维度是hidden_dim
        self.mlp_reg = nn.Sequential(
            nn.Linear(hidden_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 1)    # 输出总成本，不加Sigmoid
        )

    def forward(self, x, edge_index, edge_attr, batch=None):
        """
        前向传播。

        参数：
            x:          节点特征 [175, 176]  ← 比标准版本大
            edge_index: 边索引   [2, 211]
            edge_attr:  边特征   [211, 4]
            batch:      batch向量，单图推理时传None

        返回：
            total_cost: 预测总成本（标量）
        """
        from torch_geometric.nn import global_add_pool

        # ===== 边特征BatchNorm =====
        onehot    = edge_attr[:, :3]        # 通断one-hot，不归一化
        length    = edge_attr[:, 3:4]       # 分支长度，归一化
        # length    = self.edge_bn(length)    # BatchNorm
        edge_attr = torch.cat([onehot, length], dim=1)  # [211, 4]

        # ===== 节点特征投影 =====
        # [175, 176] → [175, hidden_dim]
        h = self.input_proj(x)

        # ===== GINE消息传递 =====
        for conv, norm in zip(self.convs, self.norms):
            h = conv(h, edge_index, edge_attr)  # [175, hidden_dim]
            h = norm(h)                          # LayerNorm
            h = torch.relu(h)                    # 激活

        # ===== 全局池化 =====
        # 把175个节点嵌入加和，得到图级别的嵌入
        # batch=None时所有节点属于同一张图
        if batch is None:
            batch = torch.zeros(
                h.size(0), dtype=torch.long,device=h.device
            )                                   # 所有节点标记为图0
        graph_emb = global_add_pool(h, batch)   # [175, hidden_dim] → [1, hidden_dim]

        # ===== MLP输出总成本 =====
        total_cost = self.mlp_reg(graph_emb)    # [1, hidden_dim] → [1, 1]

        return total_cost.squeeze()             # 去掉多余维度，变成标量