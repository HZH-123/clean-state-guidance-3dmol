import pandas as pd
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.data import Data, Batch
from torch_geometric.loader import DataLoader
from torch_geometric.nn import SchNet
from torch.optim import Adam
from torch.optim.lr_scheduler import ReduceLROnPlateau
# [!!! 更改] 导入回归指标
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import math # 用于 RMSE
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


# [!!! 更改] 重命名并适配回归任务
class CSVRegressionDataset(torch.utils.data.Dataset):
    """
    加载 .csv 文件进行回归任务 (例如 pAffinity)。
    从 SMILES 实时生成 3D 构象。
    """

    def __init__(self, csv_path, split='train', train_ratio=0.7, val_ratio=0.15):
        self.csv_path = os.path.abspath(csv_path)
        self.split = split

        if not os.path.exists(self.csv_path):
            raise FileNotFoundError(f"未找到 CSV 文件: {self.csv_path}")

        df = pd.read_csv(self.csv_path)

        # [!!! 更改] 检查回归所需的列
        if 'SMILES' not in df.columns or 'pAffinity' not in df.columns:
            raise ValueError(f"CSV 文件必须包含 'SMILES' 和 'pAffinity' 列")

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

        logger.info(f"[{split}] CSVRegressionDataset 加载完成, 共 {len(self.data_df)} 个样本. (源: {self.csv_path})")

    def __len__(self):
        return len(self.data_df)

    def __getitem__(self, idx):
        row = self.data_df.iloc[idx]
        # [!!! 更改]
        smiles = row['SMILES']
        y = torch.tensor([row['pAffinity']], dtype=torch.float)

        try:
            mol = Chem.MolFromSmiles(smiles)
            if mol is None:
                raise ValueError("RDKit 无法解析 SMILES")

            mol = Chem.AddHs(mol)
            if mol is None:
                raise ValueError("AddHs 失败 (可能SMILES有误)")

            params = AllChem.ETKDGv3()
            params.randomSeed = 42
            status = AllChem.EmbedMolecule(mol, params)

            if status == -1:
                params.useBasicKnowledge = True
                status = AllChem.EmbedMolecule(mol, params)

            if status == -1:
                raise ValueError("ETKDGv3 构象生成失败")

            AllChem.UFFOptimizeMolecule(mol)

            conf = mol.GetConformer()
            pos = torch.tensor(conf.GetPositions(), dtype=torch.float)
            z = torch.tensor([atom.GetAtomicNum() for atom in mol.GetAtoms()], dtype=torch.long)

            data = Data(z=z, pos=pos, y=y, smiles=smiles)
            return data

        except Exception as e:
            logger.error(f"加载样本失败: SMILES='{smiles[:30]}...', 错误={e}")
            # [!!! 保持] 返回空数据但包含 smiles 键
            return Data(z=torch.empty((0,), dtype=torch.long),
                        pos=torch.empty((0, 3), dtype=torch.float),
                        y=y,
                        smiles=smiles)


# ==============================================================================
# 节 2: 模型 (适配为回归模型)
# ==============================================================================

# [!!! 更改] 重命名
class SchNetForRegression(torch.nn.Module):
    """用于回归任务的SchNet模型 (输出连续值)"""

    def __init__(self, pretrained_path=None, readout='mean'):
        super().__init__()

        # SchNet 默认的 out_features=1, 这正好是回归所需的
        self.schnet = SchNet(
            hidden_channels=128,
            num_filters=128,
            num_interactions=6,
            num_gaussians=50,
            cutoff=10.0,
            readout=readout
        )

        # [!!! 不变] 加载预训练权重的逻辑是完美的
        if pretrained_path and os.path.exists(pretrained_path):
            try:
                qm9_state = torch.load(pretrained_path, map_location='cpu')
                pretrained_dict = {}

                if any(k.startswith('schnet.') for k in qm9_state.keys()):
                    pretrained_dict = qm9_state
                else:
                    for k, v in qm9_state.items():
                        pretrained_dict['schnet.' + k] = v

                model_dict = self.state_dict()

                # 过滤掉不匹配的层 (特别是最终层 lin2)
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
            logger.info("初始化新模型, 从头训练")

    def forward(self, z, pos, batch):
        # 输出 [batch_size, 1] 的连续值
        out_value = self.schnet(z, pos, batch)
        # 展平为 [batch_size] 以匹配 y 的 [batch_size]
        return out_value.view(-1)


# ==============================================================================
# 节 3: 评估 (适配为回归指标)
# ==============================================================================

# [!!! 更改] 新的指标计算函数
def compute_regression_metrics(predictions, targets, loss_fn):
    """计算回归指标"""
    loss = loss_fn(predictions, targets)

    # 转换为 numpy
    targets_np = targets.detach().cpu().numpy()
    preds_np = predictions.detach().cpu().numpy()

    # 计算指标
    try:
        mse = mean_squared_error(targets_np, preds_np)
        rmse = math.sqrt(mse)
        mae = mean_absolute_error(targets_np, preds_np)
        r2 = r2_score(targets_np, preds_np)
    except ValueError as e:
        logger.warning(f"计算指标时出错 (可能批次为空): {e}")
        return loss, 999.0, 999.0, 0.0

    return loss, rmse, mae, r2


# [!!! 更改] 新的评估函数
def evaluate_regression(model, loader, device, loss_fn):
    model.eval()
    all_preds = []
    all_targets = []
    valid_samples = 0
    with torch.no_grad():
        for data in loader:
            data = data.to(device)
            if data.z.numel() == 0: continue  # 跳过加载失败的样本

            preds = model(data.z, data.pos, data.batch)

            target = data.y
            if preds.dim() == 0:
                preds = preds.unsqueeze(0)
            if target.dim() == 0:
                target = target.unsqueeze(0)

            valid_samples += preds.size(0)
            all_preds.append(preds.cpu())
            all_targets.append(target.cpu())

    if valid_samples == 0:
        logger.warning("验证集中没有有效样本。")
        return 999.0, 999.0, 999.0, 0.0

    all_preds = torch.cat(all_preds)
    all_targets = torch.cat(all_targets)

    # [!!! 更改] 使用辅助函数计算所有指标
    avg_loss, rmse, mae, r2 = compute_regression_metrics(all_preds, all_targets, loss_fn)

    return avg_loss.item(), rmse, mae, r2


# ==============================================================================
# 节 4: Main (适配为回归任务)
# ==============================================================================

def main():
    parser = argparse.ArgumentParser(description="PARP1 pAffinity Regression with SchNet")

    # [!!! 更改] 更新参数
    parser.add_argument('--csv_path', type=str, default='../parp4_quant_aggregated.csv',
                        help='指向包含 "SMILES" 和 "pAffinity" 的 .csv 文件路径')
    parser.add_argument('--pretrained', type=str, default='schnet_pdbbind_pretrained.pth',
                        help='预训练模型路径 (e.g., schnet_pdbbind_pretrained.pth)')
    parser.add_argument('--epochs', type=int, default=300)
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--save_path', type=str, default='schnet_parp4_regressor.pth')
    parser.add_argument('--readout', type=str, default='mean', choices=['mean', 'add'])
    parser.add_argument('--patience', type=int, default=20)
    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.info(f"使用设备: {device}")

    # [!!! 更改] 加载新数据集
    train_dataset = CSVRegressionDataset(csv_path=args.csv_path, split='train')
    val_dataset = CSVRegressionDataset(csv_path=args.csv_path, split='val')
    test_dataset = CSVRegressionDataset(csv_path=args.csv_path, split='test')

    if len(train_dataset) == 0:
        logger.error(f"❌ 错误: 训练集为空. 检查 CSV 路径: {args.csv_path}")
        return

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=0,
                              collate_fn=Batch.from_data_list)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=0,
                            collate_fn=Batch.from_data_list)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, num_workers=0,
                             collate_fn=Batch.from_data_list)

    # [!!! 更改] 加载新模型
    model = SchNetForRegression(pretrained_path=args.pretrained, readout=args.readout).to(device)
    optimizer = Adam(model.parameters(), lr=args.lr, weight_decay=1e-5)

    # [!!! 更改] 监控验证集损失 (Val Loss) - 保持不变
    scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.8, patience=10, min_lr=1e-6, verbose=True)

    # [!!! 更改] 损失函数
    loss_fn = nn.MSELoss()  # 均方误差损失

    best_val_loss = np.inf
    early_stop_counter = 0

    for epoch in range(1, args.epochs + 1):
        model.train()
        train_preds, train_targets = [], []
        batch_processor = tqdm(train_loader, desc=f"训练 epoch {epoch}")

        for data in batch_processor:
            if data.z.numel() == 0: continue
            data = data.to(device)
            optimizer.zero_grad()

            preds = model(data.z, data.pos, data.batch)
            target = data.y

            # [!!! 更改]
            loss = loss_fn(preds, target)

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            train_preds.append(preds.detach().cpu())
            train_targets.append(target.cpu())

        train_preds = torch.cat(train_preds)
        train_targets = torch.cat(train_targets)
        # [!!! 更改]
        train_loss, train_rmse, train_mae, train_r2 = compute_regression_metrics(train_preds, train_targets,
                                                                                    loss_fn)
        train_loss = train_loss.item() # 从张量中获取值

        # [!!! 更改]
        val_loss, val_rmse, val_mae, val_r2 = evaluate_regression(model, val_loader, device, loss_fn)
        scheduler.step(val_loss)  # 监控损失

        # [!!! 更改]
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), args.save_path)
            early_stop_counter = 0
            logger.info(
                f"Epoch {epoch:02d} | Train Loss: {train_loss:.4f} (RMSE: {train_rmse:.4f}) | Val Loss: {val_loss:.4f} (RMSE: {val_rmse:.4f}) | 🎉 保存最佳模型")
        else:
            early_stop_counter += 1
            logger.info(
                f"Epoch {epoch:02d} | Train Loss: {train_loss:.4f} (RMSE: {train_rmse:.4f}) | Val Loss: {val_loss:.4f} (RMSE: {val_rmse:.4f}) | 早停计数: {early_stop_counter}/{args.patience}")

        if early_stop_counter >= args.patience:
            logger.info(f"早停触发: 连续{args.patience}个epoch未提升")
            break

    logger.info("\n开始测试集评估...")
    model.load_state_dict(torch.load(args.save_path))
    # [!!! 更改]
    test_loss, test_rmse, test_mae, test_r2 = evaluate_regression(model, test_loader, device, loss_fn)
    logger.info(f"测试集结果 | Loss: {test_loss:.4f} | RMSE: {test_rmse:.4f} | MAE: {test_mae:.4f} | R2: {test_r2:.4f}")
    logger.info(f"\n✅ 训练完成. 最佳验证 Loss: {best_val_loss:.4f}")


if __name__ == '__main__':
    main()