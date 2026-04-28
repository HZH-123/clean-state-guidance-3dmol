import os
import torch
from torch_geometric.data import Batch, Data
from torch_geometric.datasets import QM9
from torch_geometric.loader import DataLoader
from torch_geometric.nn import SchNet
import warnings

warnings.filterwarnings("ignore")


class ToxModelPredictor:
    def __init__(self, model_path="schnet_tox21_single_toxicity.pth", device=None):
        self.device = device if device else (
            torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
        )
        self.model = SchNet(
            hidden_channels=128,
            num_filters=128,
            num_interactions=6,
            num_gaussians=50,
            cutoff=10.0,
            readout="add"
        ).to(self.device)
        self._load_without_prefix(model_path)
        self.model.eval()

    def _load_without_prefix(self, model_path):
        state_dict = torch.load(model_path, map_location=self.device)
        new_state_dict = {
            k.replace("schnet.", ""): v
            for k, v in state_dict.items()
        }
        self.model.load_state_dict(new_state_dict)

    def predict(self, data, differentiable=True, return_tensor=False):
        """
        预测能量 (不再接收 t)
        """
        grad_context = torch.enable_grad() if differentiable else torch.no_grad()

        with grad_context:
            if isinstance(data, list):
                raise TypeError("EnergyPredictor.predict 不支持 list 输入")

            elif isinstance(data, Batch):
                z = data.z
                batch_vec = data.batch
                pos = data.pos

                # [!!! 移除了 t 和 t_expanded 的所有逻辑 !!!]

                # [!!! 修复 6: 调用模型，不传入 t !!!]
                pred = self.model(z, pos, batch_vec)
                # --- 修改结束 ---

            elif isinstance(data, Data):
                if hasattr(data, 'batch') and data.batch is not None:
                    z = data.z
                    batch_vec = data.batch
                    pos = data.pos
                    # [!!! 移除了 t 和 t_expanded !!!]
                    pred = self.model(z, pos, batch_vec)
                else:
                    z = data.z
                    pos = data.pos
                    batch_vec = torch.zeros(z.size(0), dtype=torch.long, device=self.device)
                    # [!!! 移除了 t 和 t_expanded !!!]
                    pred = self.model(z, pos, batch_vec)

            else:
                raise TypeError("输入必须是 Data、Batch 或 list[Data]")

            if pred.dim() == 2 and pred.shape[1] == 1:
                pred = pred.squeeze(1)

            if differentiable:
                return pred
            else:
                pred_cpu = pred.cpu()
                if return_tensor:
                    return pred_cpu
                else:
                    if isinstance(data, Data) and not (hasattr(data, 'batch') and data.batch is not None):
                        return pred_cpu.item()
                    else:
                        return pred_cpu.tolist()


# 预测示例
if __name__ == "__main__":
    MAX_SAMPLES = 200  # 预测的QM9分子数量
    # 初始化预测器
    try:
        predictor = ToxModelPredictor()
    except Exception as e:
        print(f"[预测信息] 模型加载失败: {e}")
        exit()
    root_path = './data/QM9'
    dataset = QM9(root=root_path)
    # 只取前 MAX_SAMPLES 个分子
    subset = dataset[:MAX_SAMPLES]
    loader = DataLoader(subset, batch_size=32, shuffle=False)
    print(f"[预测信息] 开始预测QM9分子毒性（前{MAX_SAMPLES}个）...")
    toxicity_results = {}
    global_idx = 0  # 全局分子索引（0 到 MAX_SAMPLES-1）
    for batch in loader:
        # 使用已有的 predict_qm9_toxicity（只接受 batch）
        scores = predictor.predict(batch)  # list[float]
        for score in scores:
            toxicity_results[global_idx] = score
            global_idx += 1

    # 输出部分结果（前20个）
    print("\n[预测信息] QM9分子索引 -> 毒性分数（0-1，越高毒性可能性越大）:")
    for mol_idx in range(min(20, len(toxicity_results))):
        score = toxicity_results[mol_idx]
        print(f"  索引: {mol_idx:6d} | 毒性分数: {score:.4f}")
