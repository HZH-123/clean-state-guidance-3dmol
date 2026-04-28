import os
import torch
import torch.nn.functional as F
from torch_geometric.data import Data, Batch
from torch_geometric.loader import DataLoader
from torch_geometric.nn import SchNet
from torch.optim import Adam
from torch.optim.lr_scheduler import ReduceLROnPlateau
from sklearn.metrics import roc_auc_score
import argparse
import warnings
import logging

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler()]
)
logger = logging.getLogger(__name__)

warnings.filterwarnings("ignore")

from rdkit import Chem
from rdkit.Chem import AllChem
import deepchem as dc
from tqdm import tqdm


def generate_and_cache_tox21_3d_single(cache_dir="data/tox21_3d_single", max_atoms=50):
    """生成并缓存3D分子结构数据，增加详细日志"""
    cache_dir = ''.join(c for c in cache_dir if ord(c) >= 32)
    cache_dir = os.path.abspath(os.path.normpath(cache_dir))

    if all(os.path.exists(os.path.join(cache_dir, f"{split}.pt")) for split in ["train", "val", "test"]):
        logger.info(f"✅ 已找到缓存的单毒性Tox21数据: {cache_dir}")
        return

    os.makedirs(cache_dir, exist_ok=True)
    tasks, (train, valid, test), _ = dc.molnet.load_tox21(featurizer='Raw')
    splits = {"train": train, "val": valid, "test": test}
    total_success = 0
    total_failed = 0

    for split_name, dc_dataset in splits.items():
        data_list = []
        smiles_list = dc_dataset.ids
        labels = dc_dataset.y  # shape: [N, 12]
        split_success = 0
        split_failed = 0

        logger.info(f"开始处理{split_name}集，共{len(smiles_list)}个分子")
        for i in tqdm(range(len(smiles_list)), desc=f"处理{split_name}集"):
            try:
                smiles = smiles_list[i]
                label_12 = labels[i]
                mol = Chem.MolFromSmiles(smiles)
                if mol is None:
                    split_failed += 1
                    continue

                mol = Chem.AddHs(mol)
                if mol.GetNumAtoms() > max_atoms:
                    split_failed += 1
                    continue

                # 尝试生成3D构象
                success = False
                for _ in range(3):  # 最多尝试3次
                    if AllChem.EmbedMolecule(mol, AllChem.ETKDG()) == 0:
                        success = True
                        break
                if not success:
                    split_failed += 1
                    continue

                # 优化构象
                try:
                    AllChem.MMFFOptimizeMolecule(mol)
                except Exception as e:
                    logger.debug(f"构象优化失败 {smiles}: {e}")

                # 提取坐标和原子序数
                conf = mol.GetConformer()
                pos = torch.tensor([
                    [conf.GetAtomPosition(j).x, conf.GetAtomPosition(j).y, conf.GetAtomPosition(j).z]
                    for j in range(mol.GetNumAtoms())
                ], dtype=torch.float)
                z = torch.tensor([atom.GetAtomicNum() for atom in mol.GetAtoms()], dtype=torch.long)

                # 处理标签
                label_12 = torch.tensor(label_12, dtype=torch.float)
                label_12[torch.isnan(label_12)] = -1.0  # 标记缺失值

                valid_labels = label_12[label_12 != -1.0]
                if len(valid_labels) == 0:
                    split_failed += 1
                    continue

                # 单标签逻辑：只要有一个毒性标签为阳性则标记为1
                y_value = 1.0 if valid_labels.max() >= 0.5 else 0.0
                y = torch.tensor([y_value], dtype=torch.float)

                data = Data(z=z, pos=pos, y=y)
                data_list.append(data)
                split_success += 1

            except Exception as e:
                logger.debug(f"处理分子失败 {smiles}: {e}")
                split_failed += 1
                continue

        # 保存数据并更新统计
        save_path = os.path.join(cache_dir, f"{split_name}.pt")
        torch.save(data_list, save_path)
        logger.info(f"{split_name}集处理完成: 成功{split_success}个, 失败{split_failed}个, 保存至{save_path}")
        total_success += split_success
        total_failed += split_failed

    logger.info(f"所有数据集处理完成: 总成功{total_success}个, 总失败{total_failed}个")


class CachedTox21Single:
    """缓存的Tox21单标签数据集加载器"""

    def __init__(self, root="data/tox21_3d_single", split='train'):
        clean_root = ''.join(c for c in root if 32 <= ord(c) <= 126)
        clean_root = os.path.abspath(os.path.normpath(clean_root))
        self.root = clean_root
        self.split = split

        full_path = os.path.join(self.root, f"{split}.pt")
        if not os.path.exists(full_path):
            raise FileNotFoundError(f"未找到数据集文件: {full_path}")

        self.data_list = torch.load(full_path)
        if isinstance(self.data_list, Data):
            raise RuntimeError("❌ 错误: 加载到单个Data对象, 预期应为Data列表!")

        # 统计正负样本比例
        pos_count = sum(1 for data in self.data_list if data.y.item() == 1.0)
        neg_count = len(self.data_list) - pos_count
        logger.info(f"[{split}] 加载{len(self.data_list)}个分子, 正样本{pos_count}个, 负样本{neg_count}个")

    def __len__(self):
        return len(self.data_list)

    def __getitem__(self, idx):
        data = self.data_list[idx].clone()  # 防止数据共享导致的问题
        if hasattr(data, 'batch'):
            delattr(data, 'batch')

        # 确保标签格式正确
        if data.y.numel() != 1:
            raise ValueError(f"预期y包含1个元素, 实际有{data.y.numel()}个")
        if data.y.dim() == 0:
            data.y = data.y.unsqueeze(0)
        elif data.y.dim() > 1 or data.y.shape[0] != 1:
            data.y = data.y.view(-1)[:1]
        return data


class SchNetForSingleToxicity(torch.nn.Module):
    """用于单标签毒性预测的SchNet模型（使用默认输出层，兼容单标签）"""

    def __init__(self, pretrained_path=None, readout='add'):
        super().__init__()
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
                qm9_state = torch.load(pretrained_path, map_location='cpu')
                # 映射预训练参数（保留原始结构，SchNet默认输出为1维）
                pretrained_dict = {'schnet.' + k: v for k, v in qm9_state.items()}

                model_dict = self.state_dict()
                matched = {k: v for k, v in pretrained_dict.items() if k in model_dict}

                if matched:
                    self.load_state_dict(matched, strict=False)
                    logger.info(f"✅ 从预训练模型加载{len(matched)}个层: {pretrained_path}")
                else:
                    logger.warning("⚠️ 未找到匹配的参数, 将从头训练")
            except Exception as e:
                logger.warning(f"⚠️ 加载预训练模型失败: {e}, 将从头训练")
        else:
            logger.info("初始化新模型, 从头训练")

    def forward(self, z, pos, batch):
        # SchNet默认输出为 [batch_size, 1]，squeeze后为 [batch_size]
        return self.schnet(z, pos, batch).squeeze()


def compute_metrics(pred, target):
    """计算损失和AUC，增加对极端情况的处理"""
    target = target.float()
    valid_mask = (target != -1.0)
    valid_count = valid_mask.sum().item()

    if valid_count == 0:
        return torch.tensor(0.0, device=pred.device), 0.5

    # 提取有效样本
    pred_valid = pred[valid_mask]
    target_valid = target[valid_mask]

    # 计算损失
    loss = F.binary_cross_entropy_with_logits(pred_valid, target_valid)

    # 计算AUC（处理只有一类样本的情况）
    unique_labels = torch.unique(target_valid)
    if len(unique_labels) < 2:
        logger.debug(f"警告: 有效样本中仅包含{len(unique_labels)}类标签, AUC设为0.5")
        return loss, 0.5

    try:
        auc = roc_auc_score(
            target_valid.cpu().numpy(),
            torch.sigmoid(pred_valid).cpu().numpy()
        )
    except Exception as e:
        logger.debug(f"AUC计算失败: {e}, 设为0.5")
        auc = 0.5

    return loss, auc


def evaluate(model, loader, device):
    """评估函数，收集所有样本后统一计算AUC"""
    model.eval()
    all_preds = []
    all_targets = []

    with torch.no_grad():
        for data in loader:
            data = data.to(device)
            logits = model(data.z, data.pos, data.batch)
            all_preds.append(logits.cpu())
            all_targets.append(data.y.cpu())

    # 合并所有批次
    all_preds = torch.cat(all_preds)
    all_targets = torch.cat(all_targets)

    # 计算整体指标
    loss, auc = compute_metrics(all_preds, all_targets)
    return loss.item(), auc


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--pretrained', type=str, default=None,
                        help='QM9预训练模型路径 (schnet_qm9_pretrained.pth)')
    parser.add_argument('--epochs', type=int, default=50)
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--save_path', type=str, default='schnet_tox21_single_toxicity.pth')
    parser.add_argument('--data_root', type=str, default='data/tox21_3d_single')
    parser.add_argument('--readout', type=str, default='add', choices=['mean', 'add'])
    parser.add_argument('--patience', type=int, default=10, help='早停耐心值')
    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.info(f"使用设备: {device}")

    # 数据准备
    args.data_root = os.path.abspath(os.path.normpath(args.data_root))
    generate_and_cache_tox21_3d_single(cache_dir=args.data_root)

    train_dataset = CachedTox21Single(root=args.data_root, split='train')
    val_dataset = CachedTox21Single(root=args.data_root, split='val')
    test_dataset = CachedTox21Single(root=args.data_root, split='test')  # 加载测试集

    # 数据加载器（保持单线程）
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=0,  # 单线程
        collate_fn=Batch.from_data_list,
        drop_last=False
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=0,  # 单线程
        collate_fn=Batch.from_data_list
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=0,  # 单线程
        collate_fn=Batch.from_data_list
    )

    # 模型和优化器
    model = SchNetForSingleToxicity(pretrained_path=args.pretrained, readout=args.readout).to(device)
    optimizer = Adam(model.parameters(), lr=args.lr, weight_decay=1e-5)  # 增加权重衰减
    scheduler = ReduceLROnPlateau(optimizer, mode='max', factor=0.8, patience=5, min_lr=1e-6)

    # 训练过程
    best_val_auc = 0.0
    early_stop_counter = 0  # 早停计数器

    for epoch in range(1, args.epochs + 1):
        # 训练阶段
        model.train()
        train_preds = []
        train_targets = []

        for data in tqdm(train_loader, desc=f"训练 epoch {epoch}"):
            data = data.to(device)
            optimizer.zero_grad()

            logits = model(data.z, data.pos, data.batch)
            loss, _ = compute_metrics(logits, data.y)

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)  # 梯度裁剪
            optimizer.step()

            # 收集训练数据用于计算整体指标
            train_preds.append(logits.detach().cpu())
            train_targets.append(data.y.cpu())

        # 计算训练集整体指标
        train_preds = torch.cat(train_preds)
        train_targets = torch.cat(train_targets)
        train_loss, train_auc = compute_metrics(train_preds, train_targets)

        # 验证阶段
        val_loss, val_auc = evaluate(model, val_loader, device)

        # 学习率调度
        scheduler.step(val_auc)

        # 保存最佳模型和早停检查
        if val_auc > best_val_auc:
            best_val_auc = val_auc
            torch.save(model.state_dict(), args.save_path)
            early_stop_counter = 0  # 重置早停计数器
            logger.info(
                f"Epoch {epoch:02d} | "
                f"训练损失: {train_loss:.4f} | "
                f"训练AUC: {train_auc:.4f} | "
                f"验证AUC: {val_auc:.4f} | "
                "🎉 保存最佳模型"
            )
        else:
            early_stop_counter += 1
            logger.info(
                f"Epoch {epoch:02d} | "
                f"训练损失: {train_loss:.4f} | "
                f"训练AUC: {train_auc:.4f} | "
                f"验证AUC: {val_auc:.4f} | "
                f"早停计数: {early_stop_counter}/{args.patience}"
            )

        # 早停检查
        if early_stop_counter >= args.patience:
            logger.info(f"早停触发: 连续{args.patience}个epoch未提升")
            break

    # 测试集评估
    logger.info("\n开始测试集评估...")
    model.load_state_dict(torch.load(args.save_path))
    test_loss, test_auc = evaluate(model, test_loader, device)
    logger.info(f"测试集结果 | 损失: {test_loss:.4f} | AUC: {test_auc:.4f}")

    logger.info(f"\n✅ 训练完成. 最佳验证AUC: {best_val_auc:.4f}")
    logger.info(f"模型保存路径: {os.path.abspath(args.save_path)}")


if __name__ == '__main__':
    main()