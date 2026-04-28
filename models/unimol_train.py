#
# 这份代码假定您已经按照 PyG 官网指南
# 成功安装了 'torch_geometric' 及其所有依赖 (pyg_lib, torch_scatter等)
#

import os
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch.optim import Adam
from torch.optim.lr_scheduler import ReduceLROnPlateau
from sklearn.metrics import roc_auc_score
import argparse
import warnings
import logging
import numpy as np
import torch.hub  # <-- 新增：用于加载 PyG 预训练模型

# --- 导入 (来自 PyTorch Geometric) ---
try:
    # 动态加载 PyG 版本的 Graphormer。
    # torch.hub 会自动处理下载和缓存
    # 它返回一个 *PyG 模块*，而不是 HF Transformers 模块
    logger = logging.getLogger(__name__)  # 日志记录器需要先定义
    logger.info("正在尝试通过 torch.hub 加载 PyG Graphormer (pyg-team/graphormer)...")

    # 我们将在模型类中加载它，以获取预训练权重
    # from torch_geometric.nn import Graphormer (这是模型架构)

    logger.info("PyG Graphormer 依赖检查通过。")

except ImportError:
    print("❌ 错误: 未找到 `torch_geometric` 或 `pyg_lib`。")
    print("请确保您已经按照 PyG 官网指南成功安装了所有 PyTorch Geometric 依赖。")
    exit(1)
# ---------------------

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler()]
)
# logger 已在上面定义

warnings.filterwarnings("ignore")

from rdkit import Chem
from rdkit.Chem import AllChem
import deepchem as dc
from tqdm import tqdm


# ... (generate_and_cache... 函数被修改)
def generate_and_cache_tox21_3d_single(cache_dir="data/tox21_3d_graphormer", max_atoms=50):
    """
    生成并缓存3D分子结构数据 (Graphormer 格式)

    修改点:
    - 增加了 2D 邻接矩阵 (adj) 的生成和保存。
    """
    cache_dir = ''.join(c for c in cache_dir if ord(c) >= 32)
    cache_dir = os.path.abspath(os.path.normpath(cache_dir))

    if all(os.path.exists(os.path.join(cache_dir, f"{split}.pt")) for split in ["train", "val", "test"]):
        logger.info(f"✅ 已找到缓存的 Graphormer Tox21 数据: {cache_dir}")
        return

    os.makedirs(cache_dir, exist_ok=True)
    tasks, (train, valid, test), _ = dc.molnet.load_tox21(featurizer='Raw')
    splits = {"train": train, "val": valid, "test": test}
    total_success = 0
    total_failed = 0

    for split_name, dc_dataset in splits.items():
        data_list = []
        smiles_list = dc_dataset.ids
        labels = dc_dataset.y
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

                success = False
                for _ in range(3):
                    if AllChem.EmbedMolecule(mol, AllChem.ETKDG()) == 0:
                        success = True
                        break
                if not success:
                    split_failed += 1
                    continue

                try:
                    AllChem.MMFFOptimizeMolecule(mol)
                except Exception:
                    pass  # 忽略优化失败

                conf = mol.GetConformer()
                pos = torch.tensor(conf.GetPositions(), dtype=torch.float)
                z = torch.tensor([atom.GetAtomicNum() for atom in mol.GetAtoms()], dtype=torch.long)

                # --- 新增: 为 Graphormer 提取 2D 邻接矩阵 ---
                adj = torch.from_numpy(Chem.GetAdjacencyMatrix(mol)).float()
                # -----------------------------------------------

                label_12 = torch.tensor(label_12, dtype=torch.float)
                label_12[torch.isnan(label_12)] = -1.0
                valid_labels = label_12[label_12 != -1.0]
                if len(valid_labels) == 0:
                    split_failed += 1
                    continue

                y_value = 1.0 if valid_labels.max() >= 0.5 else 0.0
                y = torch.tensor([y_value], dtype=torch.float)

                # --- 修改: 保存所有 Graphormer 需要的数据 ---
                data_dict = {'atoms': z, 'coordinates': pos, 'adj': adj, 'label': y}
                data_list.append(data_dict)
                split_success += 1

            except Exception as e:
                logger.debug(f"处理分子失败 {smiles}: {e}")
                split_failed += 1
                continue

        save_path = os.path.join(cache_dir, f"{split_name}.pt")
        torch.save(data_list, save_path)
        logger.info(f"{split_name}集处理完成: 成功{split_success}个, 失败{split_failed}个, 保存至{save_path}")
        total_success += split_success
        total_failed += split_failed

    logger.info(f"所有数据集处理完成: 总成功{total_success}个, 总失败{total_failed}个")


# --- 新增: Graphormer 的数据整理器 (Collator) ---
# Uni-Mol 的 collator 依赖于特殊的 "dictionary"
# Graphormer 的 collator 只需要将 z, pos, adj 填充 (pad) 到批次中的最大尺寸
class GraphormerDataCollator:
    def __init__(self, max_atoms=256, label_name='label'):
        self.max_atoms = max_atoms
        self.label_name = label_name

    def __call__(self, batch):
        batch_size = len(batch)

        # 找到这个批次中最大的原子数
        max_nodes_in_batch = max(len(data['atoms']) for data in batch)
        # 确保不超过模型总限制
        max_nodes = min(max_nodes_in_batch, self.max_atoms)

        # 初始化用于填充的张量
        # (B, N)
        padded_atoms = torch.zeros(batch_size, max_nodes, dtype=torch.long)
        # (B, N, N)
        padded_adj = torch.zeros(batch_size, max_nodes, max_nodes, dtype=torch.float)
        # (B, N, 3)
        padded_coords = torch.zeros(batch_size, max_nodes, 3, dtype=torch.float)
        # (B, N)
        padding_mask = torch.ones(batch_size, max_nodes, dtype=torch.bool)  # True = "请 mask 掉"

        labels = []

        for i, data in enumerate(batch):
            num_nodes = len(data['atoms'])
            if num_nodes > max_nodes:
                # 如果数据超长，截断
                num_nodes = max_nodes

            # 填充数据
            padded_atoms[i, :num_nodes] = data['atoms'][:num_nodes]
            padded_adj[i, :num_nodes, :num_nodes] = data['adj'][:num_nodes, :num_nodes]
            padded_coords[i, :num_nodes] = data['coordinates'][:num_nodes]

            # padding_mask 中, True 是 mask, False 是真实原子
            padding_mask[i, :num_nodes] = False

            labels.append(data[self.label_name])

        return {
            'atoms': padded_atoms,  # 将作为 x (node features)
            'adj': padded_adj,  # 将作为 adj (adjacency matrix)
            'coordinates': padded_coords,  # 将作为 pos (3D coordinates)
            'padding_mask': padding_mask,  # 用于 Transformer 注意力
            'label': torch.stack(labels)
        }


# ---------------------------------------------------


# --- 修改: 数据集类 ---
# (功能相同，只是名字和路径变了)
class CachedTox21Graphormer(Dataset):
    """缓存的Tox21单标签数据集加载器 (用于 Graphormer)"""

    def __init__(self, root="data/tox21_3d_graphormer", split='train'):
        clean_root = ''.join(c for c in root if 32 <= ord(c) <= 126)
        clean_root = os.path.abspath(os.path.normpath(clean_root))
        self.root = clean_root
        self.split = split
        full_path = os.path.join(self.root, f"{split}.pt")
        self.data_list = torch.load(full_path)

        pos_count = sum(1 for data in self.data_list if data['label'].item() == 1.0)
        neg_count = len(self.data_list) - pos_count
        logger.info(f"[{split}] 加载{len(self.data_list)}个分子, 正样本{pos_count}个, 负样本{neg_count}个")

    def __len__(self):
        return len(self.data_list)

    def __getitem__(self, idx):
        data_dict = self.data_list[idx]
        y = data_dict['label']
        # ... (标签格式检查)
        if y.dim() == 0:
            y = y.unsqueeze(0)
        elif y.dim() > 1:
            y = y.view(-1)[:1]

        # 返回所有 Graphormer 需要的数据
        return {
            'atoms': data_dict['atoms'],
            'coordinates': data_dict['coordinates'],
            'adj': data_dict['adj'],
            'label': y
        }


# ---------------------------------------------------


# --- 修改: 模型类 ---
class GraphormerForSingleToxicity(torch.nn.Module):
    """
    用于单标签毒性预测的 Graphormer 模型 (可微的 nn.Module)
    """

    def __init__(self, pretrained_name='graphormer-base-pcqm4mv2', readout='cls'):
        super().__init__()

        if readout != 'cls':
            logger.warning("Graphormer 通常使用 'cls' (虚拟节点) 进行读出, 而不是 'mean'")

        try:
            # ... (在 GraphormerForSingleToxicity 的 __init__ 方法中)
            self.graphormer = torch.hub.load(
                'pyg-team/graphormer',  # GitHub repo
                pretrained_name,  # 模型名称
                pretrained=True,  # 告诉它加载权重
                trust_repo=True  # <-- 关键：添加此行以信任该仓库
            )
            logger.info(f"✅ 从 torch.hub 加载 PyG Graphormer: {pretrained_name}")

            # 获取预训练模型的隐藏层维度
            self.embed_dim = self.graphormer.embed_dim

        except Exception as e:
            logger.error(f"❌ 加载 PyG Graphormer 预训练模型失败: {e}")
            logger.error("请确保 'torch_geometric' 及其所有依赖已正确安装 (pyg_lib, torch_scatter...)")
            raise e

        # 初始化分类头 (与 UniMol 类似)
        self.head = torch.nn.Linear(self.embed_dim, 1)  # 输出 1 个 logit

    def forward(self, atoms, coordinates, adj, padding_mask, **kwargs):
        """ 这是一个可微的 forward pass """

        # PyG Graphormer 模型的 forward 签名是:
        # forward(self, x, adj, pos=None, padding_mask=None)

        # x: 节点特征 (我们使用原子序数 'atoms')
        # adj: 邻接矩阵 (我们使用 'adj')
        # pos: 3D 坐标 (我们使用 'coordinates')
        # padding_mask: 告诉模型哪些是真实原子 (我们使用 collator 生成的 'padding_mask')

        # 运行 Graphormer 主干
        outputs = self.graphormer(
            x=atoms,
            adj=adj,
            pos=coordinates,
            padding_mask=padding_mask
        )

        # Graphormer (pyg-team/graphormer 版本) 使用一个虚拟节点 (CLS token)
        # 它始终位于索引 0 的位置
        # outputs 的形状是 [BatchSize, NumAtoms, EmbedDim]
        mol_repr = outputs[:, 0, :]  # 提取 [CLS] token 的表示

        logits = self.head(mol_repr)

        return logits.squeeze()


# ---------------------------------------------------


def compute_metrics(pred, target):
    """计算损失和AUC (与之前相同, 无需修改)"""
    target = target.float().squeeze()
    valid_mask = (target != -1.0)
    valid_count = valid_mask.sum().item()
    if valid_count == 0: return torch.tensor(0.0, device=pred.device), 0.5

    pred_valid = pred[valid_mask]
    target_valid = target[valid_mask]
    loss = F.binary_cross_entropy_with_logits(pred_valid, target_valid)

    unique_labels = torch.unique(target_valid)
    if len(unique_labels) < 2: return loss, 0.5
    try:
        auc = roc_auc_score(target_valid.cpu().numpy(), torch.sigmoid(pred_valid).cpu().numpy())
    except Exception:
        auc = 0.5
    return loss, auc


def evaluate(model, loader, device):
    """评估函数 (与之前相同, 无需修改)"""
    model.eval()
    all_preds = []
    all_targets = []
    with torch.no_grad():  # 评估时禁用梯度
        for batch in loader:
            batch = {k: v.to(device) for k, v in batch.items() if isinstance(v, torch.Tensor)}
            labels = batch.pop('label')
            logits = model(**batch)
            all_preds.append(logits.cpu())
            all_targets.append(labels.cpu())
    all_preds = torch.cat(all_preds)
    all_targets = torch.cat(all_targets)
    loss, auc = compute_metrics(all_preds, all_targets)
    return loss.item(), auc


def main():
    parser = argparse.ArgumentParser()
    # --- 修改: 'pretrained' 参数不再是文件路径, 而是模型名称 ---
    parser.add_argument('--pretrained', type=str,
                        default='graphormer-base-pcqm4mv2',
                        help='PyG Graphormer 预训练模型名称 (来自 torch.hub)')
    parser.add_argument('--epochs', type=int, default=50)
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--save_path', type=str, default='graphormer_tox21_finetuned.pth')
    # --- 修改: 更改默认数据路径 ---
    parser.add_argument('--data_root', type=str, default='data/tox21_3d_graphormer')
    parser.add_argument('--readout', type=str, default='cls', choices=['cls', 'mean', 'add'],
                        help="Graphormer 通常使用 'cls' (虚拟节点)")
    parser.add_argument('--patience', type=int, default=10, help='早停耐心值')
    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.info(f"使用设备: {device}")

    # 数据准备
    args.data_root = os.path.abspath(os.path.normpath(args.data_root))
    generate_and_cache_tox21_3d_single(cache_dir=args.data_root)

    # --- 修改: 使用 Graphormer 的 Dataset ---
    train_dataset = CachedTox21Graphormer(root=args.data_root, split='train')
    val_dataset = CachedTox21Graphormer(root=args.data_root, split='val')
    test_dataset = CachedTox21Graphormer(root=args.data_root, split='test')

    # --- 修改: 使用 Graphormer 的 Collator ---
    collator = GraphormerDataCollator(
        max_atoms=256,
        label_name='label'
    )
    # ---------------------------------

    # --- 标准 DataLoader ---
    train_loader = DataLoader(
        train_dataset, batch_size=args.batch_size, shuffle=True,
        num_workers=0, collate_fn=collator
    )
    val_loader = DataLoader(
        val_dataset, batch_size=args.batch_size, shuffle=False,
        num_workers=0, collate_fn=collator
    )
    test_loader = DataLoader(
        test_dataset, batch_size=args.batch_size, shuffle=False,
        num_workers=0, collate_fn=collator
    )

    # --- 修改: 使用 Graphormer 的模型 ---
    model = GraphormerForSingleToxicity(pretrained_name=args.pretrained,readout = args.readout).to(device)

    # 训练循环的其余部分 (优化器, 调度器, 循环)
    # 与 Uni-Mol 脚本完全相同，因为我们已经将模型封装好
    optimizer = Adam(model.parameters(), lr=args.lr, weight_decay=1e-5)
    scheduler = ReduceLROnPlateau(optimizer, mode='max', factor=0.8, patience=5, min_lr=1e-6)

    best_val_auc = 0.0
    early_stop_counter = 0

    for epoch in range(1, args.epochs + 1):
        model.train()
        train_preds, train_targets = [], []

        for batch in tqdm(train_loader, desc=f"训练 epoch {epoch}"):
            batch = {k: v.to(device) for k, v in batch.items() if isinstance(v, torch.Tensor)}
            optimizer.zero_grad()

            labels = batch.pop('label')
            logits = model(**batch)

            loss, _ = compute_metrics(logits, labels)

            # --- 证明其可微性 ---
            loss.backward()
            # ---

            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            train_preds.append(logits.detach().cpu())
            train_targets.append(labels.cpu())

        # ... (验证、早停、测试逻辑完全不变) ...
        train_preds = torch.cat(train_preds)
        train_targets = torch.cat(train_targets)
        train_loss, train_auc = compute_metrics(train_preds, train_targets)

        val_loss, val_auc = evaluate(model, val_loader, device)
        scheduler.step(val_auc)

        if val_auc > best_val_auc:
            best_val_auc = val_auc
            torch.save(model.state_dict(), args.save_path)
            early_stop_counter = 0
            logger.info(
                f"Epoch {epoch:02d} | Train Loss: {train_loss:.4f} | Train AUC: {train_auc:.4f} | Val AUC: {val_auc:.4f} | 🎉 保存最佳模型")
        else:
            early_stop_counter += 1
            logger.info(
                f"Epoch {epoch:02d} | Train Loss: {train_loss:.4f} | Train AUC: {train_auc:.4f} | Val AUC: {val_auc:.4f} | 早停计数: {early_stop_counter}/{args.patience}")

        if early_stop_counter >= args.patience:
            logger.info(f"早停触发")
            break

    logger.info("\n开始测试集评估...")
    model.load_state_dict(torch.load(args.save_path))
    test_loss, test_auc = evaluate(model, test_loader, device)
    logger.info(f"测试集结果 | 损失: {test_loss:.4f} | AUC: {test_auc:.4f}")
    logger.info(f"\n✅ 训练完成. 最佳验证AUC: {best_val_auc:.4f}")


if __name__ == '__main__':
    main()