import pandas as pd
import torch
from rdkit import Chem
from torch_geometric.data import Data, Batch
from torch_geometric.datasets import QM9
from torch_geometric.loader import DataLoader
from torch_geometric.nn import SchNet  # 假设 SchNet 模块已导入
import os
import logging

from tqdm import tqdm

# (配置日志，以保持一致性)
logger = logging.getLogger(__name__)


class AffinityModelPredictor:

    def __init__(self, model_path="schnet_mpro_affinity.pth", device=None, readout="add"):
        """
        初始化模型和加载权重。

        Args:
            model_path (str): 微调后的模型权重路径。
            device (torch.device, optional): 运行推理的设备。
            readout (str): SchNet使用的 Readout 模式 ('add' 或 'mean')，应与训练时一致。
        """
        self.device = device if device else (
            torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
        )
        logger.info(f"亲和力预测器使用设备: {self.device}")

        # [!!! 关键：SchNet 架构参数与训练时保持一致 !!!]
        self.model = SchNet(
            hidden_channels=128,
            num_filters=128,
            num_interactions=6,
            num_gaussians=50,
            cutoff=10.0,
            readout=readout  # 使用训练时的 Readout 模式
        ).to(self.device)
        self._load_with_prefix(model_path)

        self.model.eval()
        logger.info(f"✅ 亲和力预测模型从 {model_path} 加载完成并设置为评估模式。")

    def _load_with_prefix(self, model_path):
        """
        加载包含 'schnet.' 前缀的权重文件。
        这是因为您的 V6 训练脚本保存的是完整的 model.state_dict()。
        """
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"未找到模型文件: {model_path}")

        state_dict = torch.load(model_path, map_location=self.device)

        # 检查加载的字典是否需要移除 'schnet.' 前缀
        if any(k.startswith('schnet.') for k in state_dict.keys()):
            # 如果加载的键带 'schnet.'，我们直接加载，因为 self.model 是一个 SchNetForAffinity 的实例
            # 但由于这里我们直接实例化了 SchNet，而不是 SchNetForAffinity，
            # 我们需要移除 'schnet.' 前缀来匹配 self.model (即 SchNet) 的内部键名。
            new_state_dict = {
                k.replace("schnet.", ""): v
                for k, v in state_dict.items()
                # 排除可能存在的 'lin2'，因为它在 V6 脚本中没有加载到 SchNetForAffinity 的顶层
                if not k.startswith("lin2")
            }

        else:
            # 如果键没有前缀 (例如，如果它是纯 SchNet 导出的)，直接使用
            new_state_dict = state_dict

        # 检查是否缺少必要的键
        missing_keys, unexpected_keys = self.model.load_state_dict(new_state_dict, strict=False)
        if missing_keys:
            logger.warning(f"⚠️ 亲和力预测器缺少键: {missing_keys}")
        if unexpected_keys:
            logger.warning(f"⚠️ 亲和力预测器意外键: {unexpected_keys}")

    def predict(self, data, differentiable=True, return_tensor=False):
        """
        预测亲和力 (pAffinity 值)。

        Args:
            data (Data | Batch): PyG格式的分子数据。
            differentiable (bool): 是否保持梯度连接。
            return_tensor (bool): 如果 differentiable=False，是否返回 tensor (否则返回 float 或 list)。

        Returns:
            float, list[float], 或 torch.Tensor: 预测的 pAffinity 值。
        """

        # 使用您提供的逻辑处理梯度上下文
        grad_context = torch.enable_grad() if differentiable else torch.no_grad()

        with grad_context:
            if isinstance(data, list):
                # 尽管文档说明不接受 list，但为了鲁棒性，仍然建议用户使用 Batch 封装
                raise TypeError("AffinityModelPredictor.predict 不支持 list 输入，请使用 Batch.")

            data = data.to(self.device)

            # 确保输入数据有效
            if data.z.numel() == 0:
                logger.warning("输入数据原子数量为零，返回 NaN。")
                return float('nan')

            # 提取特征
            z = data.z
            pos = data.pos

            # 确定 batch_vec
            if isinstance(data, Batch) or (hasattr(data, 'batch') and data.batch is not None):
                # 如果是 Batch 对象，或单个 Data 对象但具有 batch 属性（通常是 DataLoader的输出）
                batch_vec = data.batch
            else:
                # 如果是单个 Data 对象，手动创建 batch vector
                batch_vec = torch.zeros(z.size(0), dtype=torch.long, device=self.device)

            # [!!! 调用模型进行回归预测 !!!]
            # SchNet默认输出为 [batch_size, 1]
            pred = self.model(z, pos, batch_vec)

            # Squeeze: 将 [N, 1] 压缩为 [N]
            if pred.dim() == 2 and pred.shape[1] == 1:
                pred = pred.squeeze(1)

            if differentiable:
                # 允许梯度回传
                return pred
            else:
                # 禁用梯度，返回 CPU 值
                pred_cpu = pred.cpu()
                if return_tensor:
                    return pred_cpu
                else:
                    # 如果是单个 Data 且没有 batch 属性（即 Batch size = 1）
                    if pred_cpu.numel() == 1:
                        return pred_cpu.item()
                    else:
                        # 否则返回列表
                        return pred_cpu.tolist()

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

if __name__ == "__main__":

    # ----------------------------------------------------
    # [!!! 配置参数 - 根据您的实际路径调整 !!!]
    # ----------------------------------------------------
    DATA_ROOT = '.'  # Mpro数据集的索引文件所在路径 (例如, 包含 info.csv 的目录)
    MPRO_DIR = '../mpro'  # Mpro结构文件夹的根目录 (包含 PDB ID 文件夹的目录)
    MODEL_PATH = 'schnet_mpro_affinity.pth'  # 微调后模型的路径
    MAX_SAMPLES = 40  # 预测的样本数量 (使用较小的数字快速演示)
    # ----------------------------------------------------

    # 初始化预测器
    try:
        # 使用 AffinityModelPredictor 代替 ToxModelPredictor
        predictor = AffinityModelPredictor(
            model_path=MODEL_PATH,
            readout="mean"  # 保持与训练时一致
        )
    except Exception as e:
        print(f"[预测信息] 亲和力模型加载失败: {e}")
        # 如果模型加载失败，则无法继续
        exit()

    # 初始化数据集
    try:
        # 使用 MproAffinityDataset 代替 QM9
        # 加载 'test' 分割，以便评估模型在未见过数据上的表现
        dataset = MproAffinityDataset(data_root=DATA_ROOT, mpro_dir=MPRO_DIR, split='test')
    except Exception as e:
        print(f"[预测信息] Mpro数据集加载失败: {e}")
        exit()

    # 只取前 MAX_SAMPLES 个分子（如果总数超过）
    if len(dataset) > MAX_SAMPLES:
        subset = dataset[:MAX_SAMPLES]
    else:
        subset = dataset
        MAX_SAMPLES = len(dataset)

    loader = DataLoader(subset, batch_size=32, shuffle=False, collate_fn=Batch.from_data_list)

    print(f"[预测信息] 开始预测 Mpro 测试集亲和力（共{MAX_SAMPLES}个样本）...")

    affinity_results = {}
    global_idx = 0  # 全局分子索引

    # 使用 tqdm 跟踪进度
    for batch in tqdm(loader, desc="预测 Mpro 亲和力"):
        if batch.z.numel() == 0:
            continue

        # 使用 predictor.predict，设置 differentiable=False 返回数值
        # scores: list[float]
        scores = predictor.predict(batch, differentiable=False)

        # 记录结果和对应的 PDB ID (如果需要)
        for i, score in enumerate(scores):
            # 原始 Data 对象的 PDB ID 在 batch 中
            pdb_id = batch.pdb_id[i]

            # 真实标签（可选，用于比较）
            true_label = batch.y[i].item() if batch.y is not None else 'N/A'

            affinity_results[global_idx] = {
                'pdb_id': pdb_id,
                'pAffinity_pred': score,
                'pAffinity_true': true_label
            }
            global_idx += 1

    # 输出部分结果（前20个）
    print("\n[预测信息] Mpro PDB ID -> 亲和力 (pAffinity, 越高结合越紧密):")

    count = 0
    for result in affinity_results.values():
        if count >= 20: break

        pdb_id = result['pdb_id']
        pred_score = result['pAffinity_pred']
        true_score = result['pAffinity_true']

        print(f"  PDB ID: {pdb_id:8s} | 预测 pAffinity: {pred_score:.4f} | 真实 pAffinity: {true_score}")
        count += 1

    print(f"\n✅ 预测完成。共处理 {len(affinity_results)} 个有效样本。")
