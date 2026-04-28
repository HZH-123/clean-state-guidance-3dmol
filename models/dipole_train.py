import os
import numpy as np
import torch
import torch.nn.functional as F
from torch_geometric.datasets import QM9
from torch_geometric.loader import DataLoader
from torch_geometric.nn import SchNet
from torch.optim.lr_scheduler import ReduceLROnPlateau
import argparse

def main():
    # ----------------------------
    # 配置
    # ----------------------------
    parser = argparse.ArgumentParser()
    parser.add_argument('--target', type=int, default=0,  # μ
                    help='QM9 target index (0-11). 0 = μ (dipole)')
    parser.add_argument('--epochs', type=int, default=300)
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--patience', type=int, default=20, help='Early stopping patience')
    parser.add_argument('--save_path', type=str, default='schnet_qm9_mu.pth')
    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    root_path = './data/QM9'
    print(f"Loading QM9 dataset from: {os.path.abspath(root_path)}")
    dataset = QM9(root=root_path)
    train_dataset = dataset[:100000]
    val_dataset = dataset[100000:110000]

    # 计算训练集目标均值和标准差（用于标准化）
    target_id = args.target
    train_targets = [dataset[i].y[0, target_id].item() for i in range(100000)]
    train_targets = torch.tensor(train_targets)
    mean = float(train_targets.mean())
    std = float(train_targets.std())
    print(f"Safe stats -> Mean: {mean:.4f}, Std: {std:.4f}")

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=0)
    print(f"Train: {len(train_dataset)}, Val: {len(val_dataset)}")

    # target 名称
    target_names = {
        0: "μ (dipole)",
        1: "α (isotropic polarizability)",
        2: "ε_HOMO",
        3: "ε_LUMO",
        4: "Gap (HOMO-LUMO)",
        5: "⟨R²⟩",
        6: "ZPVE",
        7: "U₀ (internal energy)",
        8: "U (internal energy at 298K)",
        9: "H (enthalpy)",
        10: "G (free energy)",
        11: "Cv (heat capacity)"
    }
    target_name = target_names.get(target_id, f"Unknown target {target_id}")
    print(f"Target: {target_id} → {target_name}")

    # ----------------------------
    # 模型初始化
    # ----------------------------
    model = SchNet(
        hidden_channels=128,
        num_filters=128,
        num_interactions=6,
        num_gaussians=50,
        cutoff=10.0,
        readout='add',
    ).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.8, patience=10, min_lr=1e-6)
    best_val_loss = float('inf')
    patience_counter = 0

    # ----------------------------
    # 训练循环
    # ----------------------------
    for epoch in range(1, args.epochs + 1):
        # -------- 训练 --------
        model.train()
        train_loss = 0
        for data in train_loader:
            data = data.to(device)
            optimizer.zero_grad()
            pred = model(data.z, data.pos, data.batch)

            # 标准化目标值
            target = (data.y[:, target_id] - mean) / std
            target = target.unsqueeze(1).to(device)

            loss = F.l1_loss(pred, target)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            train_loss += loss.item() * data.num_graphs
        train_loss /= len(train_dataset)

        # -------- 验证 --------
        model.eval()
        val_loss = 0
        with torch.no_grad():
            for data in val_loader:
                data = data.to(device)
                pred = model(data.z, data.pos, data.batch)
                target = (data.y[:, target_id] - mean) / std
                target = target.unsqueeze(1).to(device)
                loss = F.l1_loss(pred, target)
                val_loss += loss.item() * data.num_graphs
        val_loss /= len(val_dataset)

        scheduler.step(val_loss)

        # 早停 & 保存
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            torch.save(model.state_dict(), args.save_path)
            print(f" Epoch {epoch:03d} | Train Loss: {train_loss:.6f} | Val Loss: {val_loss:.6f} | Saved!")
        else:
            patience_counter += 1
            print(f"Epoch {epoch:03d} | Train Loss: {train_loss:.6f} | Val Loss: {val_loss:.6f}")

        if patience_counter >= args.patience:
            print(f"Early stopping at epoch {epoch}")
            break

    print(f"Pretraining finished. Best val loss: {best_val_loss:.6f}")
    print(f"Model saved to: {os.path.abspath(args.save_path)}")

if __name__ == '__main__':
    main()