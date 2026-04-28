import torch
import torch.nn as nn
from torch.nn.functional import mse_loss
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import random_split
from torch_geometric.data import Batch, Data
from torch_geometric.loader import DataLoader
from torch_geometric.datasets import QM9
import torch_geometric.nn as tgnn

from torch_scatter import scatter

import math
import os

from models.energy import energy_val


# --- EGNN 预测层 (保持不变) ---
import torch
import torch.nn as nn
from torch_scatter import scatter


# [!!! 这是一个新的、真正的“等变”层，用来替换您的 EGNN_Predict_Layer !!!]
class EquivariantLayer(nn.Module):
    """
    一个简单的 E(3)-等变图层（用于预测）
    它通过在消息传递中显式地使用 3D 向量来更新节点特征 h。
    这会创建一个对 3D 坐标高度敏感的“势能面”。
    """

    def __init__(self, hidden_channels, dropout):
        super().__init__()
        self.hidden_channels = hidden_channels

        # 1. 标量消息函数 (与之前类似)
        # (h_i, h_j, ||x_i - x_j||^2) -> m_ij
        self.phi_e = nn.Sequential(
            nn.Linear(2 * hidden_channels + 1, hidden_channels),
            nn.SiLU(),
            nn.Linear(hidden_channels, hidden_channels),
            nn.SiLU()
        )

        # 2. 节点更新函数 (现在也接收 3D 向量信息)
        # (h_i, m_i_agg, ||v_i||) -> h_i'
        self.phi_h = nn.Sequential(
            nn.Linear(hidden_channels * 2 + 1, hidden_channels),  # +1 是为了 ||v_i|| 的模长
            nn.SiLU(),
            nn.Linear(hidden_channels, hidden_channels),
        )

        # 3. 矢量“门控” (用于缩放 3D 向量)
        self.phi_x = nn.Sequential(
            nn.Linear(hidden_channels, 1),
        )

        self.dropout = nn.Dropout(dropout)

    def forward(self, h, pos, edge_index):
        row, col = edge_index

        # --- 几何计算 (不变) ---
        rel_pos = pos[row] - pos[col]  # (E, 3) 矢量
        dist_sq = (rel_pos ** 2).sum(dim=-1, keepdim=True)  # (E, 1) 标量

        # --- 1. 标量消息传递 (Scalar Message Passing) ---
        msg_input_scalar = torch.cat([h[row], h[col], dist_sq], dim=-1)
        m_ij = self.phi_e(msg_input_scalar)

        # a. 计算一个缩放因子 (E, 1)
        x_factor = self.phi_x(m_ij)

        # b. 缩放 3D 相对位置向量 (E, 3)
        m_x_ij = rel_pos * x_factor

        # --- 3. 聚合 (Aggregation) ---

        # a. 聚合标量消息 (N, C)
        m_i = scatter(m_ij, row, dim=0, dim_size=h.size(0), reduce='sum')

        # b. 聚合矢量消息 (N, 3)
        v_i = scatter(m_x_ij, row, dim=0, dim_size=h.size(0), reduce='sum')

        # c. [!!! 核心 !!!]
        # 获取矢量消息的“模长”(范数)，这是一个对旋转不变的标量！
        v_i_norm = v_i.norm(dim=-1, keepdim=True)  # (N, 1)

        # --- 4. 节点更新 (Node Update) ---
        # 我们将 h, 聚合的标量消息, 聚合的矢量模长 拼接在一起

        h_agg_input = torch.cat([h, m_i, v_i_norm], dim=-1)

        h_new = self.phi_h(h_agg_input)
        h_new = self.dropout(h_new)

        # EGNN 通常会有一个残差连接
        return h + h_new


# [!!! 修复 1：修改模型，移除时间 t !!!]
class EGNN(nn.Module):  # <-- 重命名为 EGNN
    def __init__(self, hidden_channels=128, n_layers=5, dropout=0.1):
        super(EGNN, self).__init__()  # <-- 修改了 super
        self.hidden_channels = hidden_channels
        self.embedding = nn.Embedding(100, hidden_channels)

        # [!!! 移除了 time_embedding !!!]

        self.layers = nn.ModuleList([
            EquivariantLayer(hidden_channels, dropout) for _ in range(n_layers)
        ])
        self.regression_head = nn.Sequential(
            nn.Linear(hidden_channels, hidden_channels),
            nn.SiLU(),
            nn.Linear(hidden_channels, 1)
        )
        self.projection_head = nn.Sequential(
            nn.Linear(hidden_channels, hidden_channels),
            nn.SiLU(),
            nn.Linear(hidden_channels, hidden_channels)
        )

    # [!!! 修复 2：修改 forward 签名，移除 t !!!]
    def forward(self, z, pos, batch):
        h = self.embedding(z)

        # [!!! 移除了 t_emb 和 h = h + t_emb !!!]

        edge_index = tgnn.radius_graph(pos, r=4.0, batch=batch)

        for layer in self.layers:
            h = layer(h, pos, edge_index)

        h_mol = tgnn.global_add_pool(h, batch)

        E_pred = self.regression_head(h_mol).squeeze(1)
        Z_proj = self.projection_head(h_mol)

        return E_pred, Z_proj


# --- 模型初始化 ---
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(f"使用设备: {device}")
energy_model = EGNN(hidden_channels=128).to(device)
print("已初始化 EGNN (目标：训练干净能量模型)。")
root_path = './data/QM9'
dataset = QM9(root=root_path)

train_ratio = 0.8
train_size = int(train_ratio * len(dataset))
test_size = len(dataset) - train_size
torch.manual_seed(42)
train_dataset, test_dataset = random_split(
    dataset, [train_size, test_size], generator=torch.Generator().manual_seed(42))
val_ratio = 0.1
val_size = int(val_ratio * len(train_dataset))
train_size = len(train_dataset) - val_size
train_dataset, val_dataset = random_split(
    train_dataset, [train_size, val_size], generator=torch.Generator().manual_seed(42))

print(f"\n数据集划分完成:")
print(f"  划分后训练集样本数: {len(train_dataset)}")
print(f"  验证集样本数: {len(val_dataset)}")
print(f"  测试集样本数: {len(test_dataset)}")


# [!!! 修复 4：修改 evaluate 函数 (移除 t) !!!]
def evaluate(model, dataloader, device):
    model.eval()
    total_mse_loss = 0.0
    num_batches = 0
    loss_fn_mse = nn.MSELoss()

    with torch.no_grad():
        for batch in dataloader:
            batch = batch.to(device)

            try:
                target_energy = energy_val(batch)
            except Exception as e:
                print(f"评估时计算 energy_val 失败: {e}")
                continue

            # [!!! 移除了 t_expanded_dummy !!!]
            E_pred, _ = model(batch.z, batch.pos, batch.batch)

            total_mse_loss += loss_fn_mse(E_pred, target_energy).item()
            num_batches += 1

    avg_mse_loss = total_mse_loss / num_batches if num_batches > 0 else 0.0
    model.train()
    return avg_mse_loss


# [!!! 修复 5：修改 train 函数 (移除 t) !!!]
def train(model, train_loader, val_loader, test_loader, epochs=50, lr=1e-4):
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-5)
    scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.8, patience=5, min_lr=1e-6)

    loss_fn_mse = nn.MSELoss()

    model.train()
    best_val_loss = float('inf')

    print("开始训练 EGNN (代理任务：预测*干净*能量)...")  # <-- 修改了标题

    for epoch in range(epochs):
        train_total_mse = 0.0
        train_num_batches = 0

        for batch in train_loader:
            batch = batch.to(device)
            optimizer.zero_grad()

            try:
                target_energy = energy_val(batch)
            except Exception as e:
                print(f"训练时计算 energy_val 失败: {e}")
                continue

            # [!!! 移除了 t_expanded_dummy !!!]
            E_pred, _ = model(batch.z, batch.pos, batch.batch)

            loss_mse = loss_fn_mse(E_pred, target_energy.detach())
            total_loss = loss_mse

            total_loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            train_total_mse += loss_mse.item()
            train_num_batches += 1

        train_avg_mse = train_total_mse / train_num_batches if train_num_batches > 0 else 0.0

        val_avg_mse = evaluate(model, val_loader, device)
        val_total_loss = val_avg_mse
        scheduler.step(val_total_loss)
        current_lr = optimizer.param_groups[0]['lr']

        print(f"Epoch {epoch + 1:3d}/{epochs} | "
              f"Train MSE: {train_avg_mse:.6f} | "
              f"Val MSE: {val_avg_mse:.6f} | "
              f"LR: {current_lr:.6f}")

        if val_total_loss < best_val_loss:
            best_val_loss = val_total_loss

            # [!!! 修复 6：修正了文件名 !!!]
            torch.save(model.state_dict(), 'best_energy_schnet.pth')
            print(f"  → 保存最佳模型（验证集 MSE: {best_val_loss:.6f}）")

    # 训练结束评估
    print("\n" + "-" * 50)
    print("训练完成！开始测试集最终评估...")
    # [!!! 修复 7：修正了文件名 !!!]
    model.load_state_dict(torch.load('best_energy_schnet.pth'))
    test_avg_mse = evaluate(model, test_loader, device)
    print(f"测试集最终 MSE(Energy) 损失: {test_avg_mse:.6f}")
    print("-" * 50)


# --- 构建数据加载器 (保持不变) ---
batch_size = 32
train_loader = DataLoader(
    train_dataset, batch_size=batch_size, shuffle=True, num_workers=0, drop_last=True)
val_loader = DataLoader(
    val_dataset, batch_size=batch_size, shuffle=False, num_workers=0)
test_loader = DataLoader(
    test_dataset, batch_size=batch_size, shuffle=False, num_workers=0)

# --- 启动训练与评估 (保持不变) ---
if __name__ == "__main__":
    train(
        model=energy_model,
        train_loader=train_loader,
        val_loader=val_loader,
        test_loader=test_loader,
        epochs=50,
        lr=1e-4,
    )
    # [!!! 修复 8：修正了文件名 !!!]
    print(f"\n最优模型已保存为 'best_energy_schnet.pth' ")