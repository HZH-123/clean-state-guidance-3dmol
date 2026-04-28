import math
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


def convert_to_paffinity(affinity_str):
    """
    [V4 最终修复版] 将 PDBbind 原始亲和力字符串 (e.g., 'Ki=10nM', 'Kd=1.3uM') 转换为 pAffinity (-log10(M)).
    """
    affinity_str = affinity_str.strip()

    # 1. 排除不等号
    if '<' in affinity_str or '>' in affinity_str:
        return None

    # 2. 确定类型 (Ki, Kd, 或 IC50) - 转换为大写以处理 ki/kd
    affinity_upper = affinity_str.upper()

    if 'KI=' in affinity_upper:
        val_unit = affinity_str.split('Ki=')[-1] if 'Ki=' in affinity_str else affinity_str.split('ki=')[-1]
    elif 'KD=' in affinity_upper:
        val_unit = affinity_str.split('Kd=')[-1] if 'Kd=' in affinity_str else affinity_str.split('kd=')[-1]
    elif 'IC50=' in affinity_upper:
        val_unit = affinity_str.split('IC50=')[-1] if 'IC50=' in affinity_str else affinity_str.split('ic50=')[-1]
    else:
        return None

    # 3. [!!! BUG FIX V4: 重新排序 !!!]
    # 必须先检查最具体的单位 (e.g., 'uM')，最后检查最通用的单位 ('M')
    if val_unit.endswith('mM'):
        unit_scale, val_str = 1e-3, val_unit[:-2]
    elif val_unit.endswith('uM'):
        unit_scale, val_str = 1e-6, val_unit[:-2]  # 'uM' 现在在 'M' 之前
    elif val_unit.endswith('nM'):
        unit_scale, val_str = 1e-9, val_unit[:-2]
    elif val_unit.endswith('pM'):
        unit_scale, val_str = 1e-12, val_unit[:-2]
    elif val_unit.endswith('M'):
        unit_scale, val_str = 1.0, val_unit[:-1]  # 'M' 现在在最后
    else:
        return None

    try:
        value = float(val_str)
        molar = value * unit_scale

        if molar <= 0: return None

        result = -math.log10(molar)
        return result
    except ValueError:
        return None
# ==============================================================================
# 节 1: PDBbind 数据集加载器 (用于预训练)
# ==============================================================================

# [!!! 替换您训练脚本中的 PDBbindDataset 类 !!!]
class PDBbindDataset(torch.utils.data.Dataset):
    def __init__(self, data_root, index_file_path, split='train', **kwargs): # 简化 __init__
        self.root = data_root
        self.index_file = index_file_path
        self.split = split

        # 加载已清洗的数据帧
        full_df = self._load_index()

        # 我们仍然需要划分训练/验证/测试集
        total_len = len(full_df)
        train_len = int(total_len * 0.8) # 假设 80/10/10 划分
        val_len = int(total_len * 0.1)
        full_df = full_df.sample(frac=1, random_state=42).reset_index(drop=True)

        if split == 'train':
            self.data_df = full_df.iloc[:train_len]
        elif split == 'val':
            self.data_df = full_df.iloc[train_len:train_len + val_len]
        else: # 'test'
            self.data_df = full_df.iloc[train_len + val_len:]

        logger.info(f"[{split}] PDBbind (CLEAN) 加载完成, 共 {len(self.data_df)} 个样本.")

    def _load_index(self):
        """
        [V13] 加载 *已清洗* 的索引文件 (格式: PDB_ID pAffinity)
        """
        data_list = []
        try:
            with open(self.index_file, 'r') as f:
                for line in f:
                    if line.startswith('#'):
                        continue
                    parts = line.split()
                    if len(parts) == 2:
                        pdb_id = parts[0]
                        pAffinity = float(parts[1])
                        data_list.append({'pdb_id': pdb_id, 'pAffinity': pAffinity})
        except FileNotFoundError:
            raise FileNotFoundError(f"PDBbind 清洁索引文件未找到: {self.index_file}")

        if not data_list:
            raise ValueError(f"未能从 {self.index_file} 加载任何数据。")

        return pd.DataFrame(data_list)

    def __len__(self):
        return len(self.data_df)

    def __getitem__(self, idx):
        # [!!! 关键: __getitem__ 现在可以 100% 信任数据 !!!]
        # 它不需要 try/except 来捕获 RDKit 错误

        row = self.data_df.iloc[idx]
        pdb_id = row['pdb_id']
        y = torch.tensor([row['pAffinity']], dtype=torch.float)

        ligand_mol2 = os.path.join(self.root, pdb_id, f"{pdb_id}_ligand.mol2")
        ligand_sdf = os.path.join(self.root, pdb_id, f"{pdb_id}_ligand.sdf")

        mol = None
        if os.path.isfile(ligand_mol2):
            mol = Chem.MolFromMol2File(ligand_mol2, removeHs=False)
        elif os.path.isfile(ligand_sdf):
            mol = Chem.MolFromMolFile(ligand_sdf, removeHs=False)

        if mol is None:
            if os.path.isfile(ligand_mol2):
                mol_with_h = Chem.MolFromMol2File(ligand_mol2, removeHs=True)
                if mol_with_h:
                    mol = Chem.AddHs(mol_with_h, addCoords=True)

        # 我们不需要再检查 mol is None，因为清洗脚本已经保证了它

        conf = mol.GetConformer()
        pos = torch.tensor(conf.GetPositions(), dtype=torch.float)
        z = torch.tensor([atom.GetAtomicNum() for atom in mol.GetAtoms()], dtype=torch.long)

        return Data(z=z, pos=pos, y=y, pdb_id=pdb_id)


# ==============================================================================
# 节 2 & 3: 模型, 评估, 和 main()
# (这部分与您的 V6 训练脚本完全相同)
# ==============================================================================

class SchNetForAffinity(torch.nn.Module):
    """用于亲和力预测（回归）的SchNet模型 (与 V6 脚本完全相同)"""

    def __init__(self, pretrained_path=None, readout='mean'):
        super().__init__()

        self.schnet = SchNet(
            hidden_channels=128,
            num_filters=128,
            num_interactions=6,
            num_gaussians=50,
            cutoff=10.0,
            readout=readout
        )

        if pretrained_path and os.path.exists(pretrained_path):
            try:

                logger.info(f"警告：正在加载预训练权重 {pretrained_path} 以继续训练。")

                qm9_state = torch.load(pretrained_path, map_location='cpu')
                pretrained_dict = {}

                if any(k.startswith('schnet.') for k in qm9_state.keys()):
                    pretrained_dict = qm9_state
                else:
                    for k, v in qm9_state.items():
                        pretrained_dict['schnet.' + k] = v

                model_dict = self.state_dict()
                matched = {k: v for k, v in pretrained_dict.items()
                           if k in model_dict and 'schnet.lin2' not in k}

                if matched:
                    self.load_state_dict(matched, strict=False)
                    logger.info(f"✅ 从预训练模型加载{len(matched)}个层: {pretrained_path} (跳过最终回归层 lin2)")
                else:
                    logger.warning("⚠️ 修复后仍未找到匹配的参数, 将从头训练")

            except Exception as e:
                logger.warning(f"⚠️ 加载预训练模型失败: {e}, 将从头训练")
        else:
            logger.info("初始化新模型, 从头训练 (用于 PDBbind 预训练)")

    def forward(self, z, pos, batch):
        out = self.schnet(z, pos, batch)
        return out.view(-1)


def compute_metrics(pred, target):
    """ (与 V6 脚本完全相同) """
    pred_np = pred.detach().cpu().numpy()
    target_np = target.detach().cpu().numpy()
    loss = F.mse_loss(pred, target)
    rmse = np.sqrt(mean_squared_error(target_np, pred_np))
    mae = mean_absolute_error(target_np, pred_np)
    try:
        r2 = r2_score(target_np, pred_np)
    except ValueError:
        r2 = 0.0
    return loss, rmse, mae, r2


def evaluate(model, loader, device):
    """ (与 V6 脚本完全相同) """
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
    parser = argparse.ArgumentParser(description="PDBbind Affinity Pre-Training with SchNet")

    # [!!!] 关键更改：修改了参数以适应 PDBbind
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--save_path', type=str, default='schnet_pdbbind_pretrained.pth')

    # [!!!] PDBbind 特定的路径
    parser.add_argument('--data_root', type=str, default='../P-L',
                        help='PDBbind v2020 已解压的结构文件夹路径 (e.g., v2020-other-PL)')
    parser.add_argument('--index_file', type=str, default='../index/INDEX_general_PL.2020R1.clean.lst',
                        help='PDBbind 索引文件路径 (e.g., ../INDEX_general_PL_data.2020)')

    # (与 V6 相同的参数)
    parser.add_argument('--readout', type=str, default='mean', choices=['mean', 'add'])
    parser.add_argument('--patience', type=int, default=10)  # 预训练任务更难, patience 可以设短一点

    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.info(f"使用设备: {device}")

    # 标准化路径
    args.data_root = os.path.abspath(os.path.normpath(args.data_root))
    args.index_file = os.path.abspath(os.path.normpath(args.index_file))

    # [!!!] 确保您已应用 V4 修复的 PDBbindDataset 类 [!!!]
    try:
        train_dataset = PDBbindDataset(data_root=args.data_root, index_file_path=args.index_file, split='train')
        val_dataset = PDBbindDataset(data_root=args.data_root, index_file_path=args.index_file, split='val')
        test_dataset = PDBbindDataset(data_root=args.data_root, index_file_path=args.index_file, split='test')
    except Exception as e:
        logger.error(f"❌ 错误: 加载 PDBbind 数据集失败: {e}")
        return

    if len(train_dataset) == 0:
        logger.error("❌ 错误: 训练集为空。请检查路径。")
        return

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=4,
                              pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=4,
                            pin_memory=True)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, num_workers=4,
                             pin_memory=True)

    # (模型、优化器和调度器不变)
    model = SchNetForAffinity(pretrained_path=None, readout=args.readout).to(device)
    optimizer = Adam(model.parameters(), lr=args.lr, weight_decay=1e-5)
    scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.8, patience=5, min_lr=1e-6, verbose=True)

    best_val_rmse = np.inf
    early_stop_counter = 0

    # (训练循环)
    for epoch in range(1, args.epochs + 1):
        model.train()
        train_preds, train_targets = [], []
        batch_processor = tqdm(train_loader, desc=f"预训练 epoch {epoch}")

        for data in batch_processor:

            # [!!! V7 双重保险检查 !!!]
            # 如果 collate_wrapper 返回了一个空批次 (num_graphs=0)，则跳过
            if data.num_graphs == 0:
                continue

            data = data.to(device)
            optimizer.zero_grad()
            logits = model(data.z, data.pos, data.batch)
            assert logits.shape == data.y.shape, f"logits {logits.shape} != y {data.y.shape}"
            # [!!!] 这里的 logits 和 data.y 现在大小保证一致 [!!!]
            loss = F.mse_loss(logits, data.y)

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            train_preds.append(logits.detach().cpu())
            train_targets.append(data.y.cpu())

        # (训练循环的其余部分不变)
        if not train_preds:
            logger.warning(f"Epoch {epoch:02d} 未处理任何有效数据。")
            continue

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
                f"Epoch {epoch:02d} | Train RMSE: {train_rmse:.4f} | Val RMSE: {val_rmse:.4f} (R2: {val_r2:.4f}) | 🎉 保存最佳预训练模型")
        else:
            early_stop_counter += 1
            logger.info(
                f"Epoch {epoch:02d} | Train RMSE: {train_rmse:.4f} | Val RMSE: {val_rmse:.4f} (R2: {val_r2:.4f}) | 早停计数: {early_stop_counter}/{args.patience}")

        if early_stop_counter >= args.patience:
            logger.info(f"早停触发: 连续{args.patience}个epoch未提升")
            break

    # (评估循环不变)
    logger.info("\n开始测试集评估...")
    try:
        model.load_state_dict(torch.load(args.save_path))
        test_loss, test_rmse, test_mae, test_r2 = evaluate(model, test_loader, device)
        logger.info(f"测试集结果 | RMSE: {test_rmse:.4f} | MAE: {test_mae:.4f} | R2: {test_r2:.4f}")
        logger.info(f"\n✅ 预训练完成. 最佳验证RMSE: {best_val_rmse:.4f}")
        logger.info(f"预训练模型已保存至: {args.save_path}")
    except FileNotFoundError:
        logger.error(f"❌ 错误: 未找到保存的模型文件 '{args.save_path}'。可能训练未成功保存任何模型。")
    except Exception as e:
        logger.error(f"❌ 评估失败: {e}")


if __name__ == '__main__':
    main()