import pandas as pd
import os
import torch
import torch.nn.functional as F
from torch_geometric.data import Data, Batch
from torch_geometric.loader import DataLoader
from torch_geometric.nn import SchNet
from torch.optim import Adam
from torch.optim.lr_scheduler import ReduceLROnPlateau
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import argparse
import warnings
import logging
import numpy as np
from tqdm import tqdm
from rdkit import Chem
from rdkit.Chem import AllChem

# (配置日志...)
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s',
                    handlers=[logging.StreamHandler()])
logger = logging.getLogger(__name__)
warnings.filterwarnings("ignore")


# ==============================================================================
# 节 1: Mpro 数据集加载器 (V6 - 直接读取 'ligand.mol2')
# ==============================================================================

class MproAffinityDataset(torch.utils.data.Dataset):
    """
    加载 Mpro 亲和力数据集 (直接从 mpro/[PDB_ID]/ligand.mol2 加载)。
    """

    def __init__(self, data_root, mpro_dir, split='train', train_ratio=0.7, val_ratio=0.15):
        self.root = data_root
        self.mpro_dir = mpro_dir
        self.split = split

        # 索引文件
        csv_path = os.path.join(self.root, "../mpro_3d_affinity_training_data_final.csv")  # 假设 CSV 叫这个
        if not os.path.exists(csv_path):
            # 备用，使用您在V4中生成的CSV
            csv_path = os.path.join(self.root, "mpro_3d_affinity_training_data_final.csv")
            if not os.path.exists(csv_path):
                # 备用，使用 V4 预处理脚本中的 info.csv
                csv_path = os.path.join(self.mpro_dir, "info.csv")
                if not os.path.exists(csv_path):
                    raise FileNotFoundError(f"在 {self.root} 或 {self.mpro_dir} 中均未找到索引 CSV。")

        # 加载索引文件
        df = pd.read_csv(csv_path)
        # 重命名 (以防万一)
        if "pdbid" in df.columns:
            df = df.rename(columns={"pdbid": "pdb_id", "pKa_reliable": "pAffinity"})

        total_len = len(df)

        train_len = int(total_len * train_ratio)
        val_len = int(total_len * val_ratio)
        df = df.sample(frac=1, random_state=42).reset_index(drop=True)

        if split == 'train':
            self.data_df = df.iloc[:train_len]
        elif split == 'val':
            self.data_df = df.iloc[train_len:train_len + val_len]
        else:
            self.data_df = df.iloc[train_len + val_len:]

        logger.info(f"[{split}] Mpro(V6)数据集加载完成, 共 {len(self.data_df)} 个样本.")

    def __len__(self):
        return len(self.data_df)

    def __getitem__(self, idx):
        row = self.data_df.iloc[idx]

        pdb_id = row['pdb_id']  # 大写 PDB ID
        y = torch.tensor([row['pAffinity']], dtype=torch.float)

        try:
            # [!!!] V6 关键：使用您确认的 V4 路径逻辑
            folder_path = os.path.join(self.mpro_dir, pdb_id)

            ligand_file_mol2 = os.path.join(folder_path, "ligand.mol2")
            ligand_file_pdb = os.path.join(folder_path, "ligand.pdb")

            if os.path.isfile(ligand_file_mol2):
                mol = Chem.MolFromMol2File(ligand_file_mol2, removeHs=False)
            elif os.path.isfile(ligand_file_pdb):
                mol = Chem.MolFromPDBFile(ligand_file_pdb, removeHs=False)
            else:
                raise FileNotFoundError(f"在 {folder_path} 中未找到 ligand.mol2 或 ligand.pdb")

            if mol is None:
                raise ValueError("RDKit 无法从文件加载分子")

            # 提取特征
            conf = mol.GetConformer()
            pos = torch.tensor(conf.GetPositions(), dtype=torch.float)
            z = torch.tensor([atom.GetAtomicNum() for atom in mol.GetAtoms()], dtype=torch.long)

            data = Data(z=z, pos=pos, y=y, pdb_id=pdb_id)
            return data

        except Exception as e:
            logger.error(f"加载样本失败: PDB ID={pdb_id}, 错误={e}")
            return Data(z=torch.empty((0,), dtype=torch.long),
                        pos=torch.empty((0, 3), dtype=torch.float),
                        y=torch.tensor([0.0], dtype=torch.float))


# ==============================================================================
# 节 2 & 3: 模型, 评估, 和 main()
# (这部分与 V5 训练脚本完全相同 - 它们是通用的)
# ==============================================================================

class SchNetForAffinity(torch.nn.Module):
    """用于亲和力预测（回归）的SchNet模型"""

    def __init__(self, pretrained_path=None, readout='mean'):
        super().__init__()

        # SchNet默认输出为 [batch_size, 1]
        self.schnet = SchNet(
            hidden_channels=128,
            num_filters=128,
            num_interactions=6,
            num_gaussians=50,
            cutoff=10.0,
            readout=readout  # 'add' 或 'mean'
        )

        if pretrained_path and os.path.exists(pretrained_path):
            try:
                # 加载 QM9 预训练权重
                qm9_state = torch.load(pretrained_path, map_location='cpu')

                # [!!! 关键修复 !!!]
                # 预训练的 QM9 模型通常只保存 SchNet 模块的 state_dict
                # 其键名是 'embedding.weight', 'interactions.0.lin.weight' 等。
                # 我们的模型将 SchNet 包装在 'self.schnet' 中,
                # 所以我们新模型的键名是 'schnet.embedding.weight', 'schnet.interactions.0.lin.weight'。
                # 我们需要手动添加 'schnet.' 前缀。

                pretrained_dict = {}
                if any(k.startswith('schnet.') for k in qm9_state.keys()):
                    # 如果预训练文件已经包含了 'schnet.' 前缀 (不太可能, 但保险起见)
                    pretrained_dict = qm9_state
                else:
                    # [!!!] 这是最可能的情况：为所有键添加 'schnet.' 前缀
                    for k, v in qm9_state.items():
                        pretrained_dict['schnet.' + k] = v

                model_dict = self.state_dict()

                # 过滤掉不匹配的层 (特别是最后的线性层 lin2)
                # SchNet 模块的最终回归层是 'lin2'
                matched = {k: v for k, v in pretrained_dict.items()
                           if k in model_dict and 'schnet.lin2' not in k}

                if matched:
                    self.load_state_dict(matched, strict=False)
                    logger.info(f"✅ 从预训练模型加载{len(matched)}个层: {pretrained_path} (跳过最终回归层 lin2)")
                else:
                    # 如果修复后仍然失败，说明预训练文件有问题
                    logger.warning("⚠️ 修复后仍未找到匹配的参数, 将从头训练")

            except Exception as e:
                logger.warning(f"⚠️ 加载预训练模型失败: {e}, 将从头训练")
        else:
            logger.info("初始化新模型, 从头训练")

    def forward(self, z, pos, batch):
        out = self.schnet(z, pos, batch)
        return out.view(-1)


def compute_metrics(pred, target):
    pred_np = pred.detach().cpu().numpy()
    target_np = target.detach().cpu().numpy()
    loss = F.mse_loss(pred, target)
    rmse = np.sqrt(mean_squared_error(target_np, pred_np))
    mae = mean_absolute_error(target_np, pred_np)
    try:
        r2 = r2_score(target_np, pred_np)
    except ValueError:
        r2 = 0.0  # 发生错误时（例如只有一个样本）
    return loss, rmse, mae, r2


def evaluate(model, loader, device):
    model.eval()
    all_preds = []
    all_targets = []
    valid_samples = 0
    total_loss = 0.0
    with torch.no_grad():
        for data in loader:
            data = data.to(device)
            if data.z.numel() == 0: continue
            logits = model(data.z, data.pos, data.batch)
            if logits.dim() == 0: logits = logits.unsqueeze(0)
            loss = F.mse_loss(logits, data.y)
            total_loss += loss.item() * logits.size(0)
            valid_samples += logits.size(0)
            all_preds.append(logits.cpu())
            all_targets.append(data.y.cpu())
    if valid_samples == 0: return 0.0, 999.0, 999.0, 0.0
    all_preds = torch.cat(all_preds)
    all_targets = torch.cat(all_targets)
    _, rmse, mae, r2 = compute_metrics(all_preds, all_targets)
    avg_loss = total_loss / valid_samples
    return avg_loss, rmse, mae, r2


def main():
    parser = argparse.ArgumentParser(description="Mpro Affinity Prediction with SchNet (V6)")
    parser.add_argument('--pretrained', type=str, default='schnet_pdbbind_pretrained.pth', help='预训练模型路径')
    #schnet_pdbbind_pretrained.pth
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--batch_size', type=int, default=8)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--save_path', type=str, default='schnet_mpro_affinity.pth')
    parser.add_argument('--data_root', type=str, default='.')
    parser.add_argument('--mpro_dir', type=str, default='../mpro')  # [!!!] V6 新增：mpro 文件夹路径
    parser.add_argument('--readout', type=str, default='mean', choices=['mean', 'add'])
    parser.add_argument('--patience', type=int, default=20)
    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.info(f"使用设备: {device}")

    args.data_root = os.path.abspath(os.path.normpath(args.data_root))
    args.mpro_dir = os.path.abspath(os.path.normpath(args.mpro_dir))

    train_dataset = MproAffinityDataset(data_root=args.data_root, mpro_dir=args.mpro_dir, split='train')
    val_dataset = MproAffinityDataset(data_root=args.data_root, mpro_dir=args.mpro_dir, split='val')
    test_dataset = MproAffinityDataset(data_root=args.data_root, mpro_dir=args.mpro_dir, split='test')

    if len(train_dataset) == 0:
        logger.error("❌ 错误: 训练集为空。")
        return

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=0,
                              collate_fn=Batch.from_data_list)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=0,
                            collate_fn=Batch.from_data_list)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, num_workers=0,
                             collate_fn=Batch.from_data_list)

    model = SchNetForAffinity(pretrained_path=args.pretrained, readout=args.readout).to(device)
    optimizer = Adam(model.parameters(), lr=args.lr, weight_decay=1e-5)
    scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.8, patience=10, min_lr=1e-6, verbose=True)

    best_val_rmse = np.inf
    early_stop_counter = 0

    for epoch in range(1, args.epochs + 1):
        model.train()
        train_preds, train_targets = [], []
        batch_processor = tqdm(train_loader, desc=f"训练 epoch {epoch}")

        for data in batch_processor:
            if data.z.numel() == 0: continue
            data = data.to(device)
            optimizer.zero_grad()
            logits = model(data.z, data.pos, data.batch)
            loss = F.mse_loss(logits, data.y)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            train_preds.append(logits.detach().cpu())
            train_targets.append(data.y.cpu())

        train_preds = torch.cat(train_preds)
        train_targets = torch.cat(train_targets)
        train_loss, train_rmse, train_mae, train_r2 = compute_metrics(train_preds, train_targets)

        val_loss, val_rmse, val_mae, val_r2 = evaluate(model, val_loader, device)
        scheduler.step(val_rmse)

        if val_rmse < best_val_rmse:
            best_val_rmse = val_rmse
            torch.save(model.state_dict(), args.save_path)
            early_stop_counter = 0
            logger.info(
                f"Epoch {epoch:02d} | Train RMSE: {train_rmse:.4f} | Val RMSE: {val_rmse:.4f} (R2: {val_r2:.4f}) | 🎉 保存最佳模型")
        else:
            early_stop_counter += 1
            logger.info(
                f"Epoch {epoch:02d} | Train RMSE: {train_rmse:.4f} | Val RMSE: {val_rmse:.4f} (R2: {val_r2:.4f}) | 早停计数: {early_stop_counter}/{args.patience}")

        if early_stop_counter >= args.patience:
            logger.info(f"早停触发: 连续{args.patience}个epoch未提升")
            break

    logger.info("\n开始测试集评估...")
    model.load_state_dict(torch.load(args.save_path))
    test_loss, test_rmse, test_mae, test_r2 = evaluate(model, test_loader, device)
    logger.info(f"测试集结果 | RMSE: {test_rmse:.4f} | MAE: {test_mae:.4f} | R2: {test_r2:.4f}")
    logger.info(f"\n✅ 训练完成. 最佳验证RMSE: {best_val_rmse:.4f}")


if __name__ == '__main__':
    main()