import torch
from torch_geometric.data import Data, Batch
from torch_geometric.datasets import QM9
from torch_geometric.loader import DataLoader
from torch_geometric.nn import SchNet


class StrainModelPredictor:

    def __init__(self, model_path="schnet_qm9_E_strain.pth", device=None):
        self.device = device if device else (
            torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
        )

        # 1. 初始化 SchNet 模型架构
        self.model = SchNet(
            hidden_channels=128,
            num_filters=128,
            num_interactions=6,
            num_gaussians=50,
            cutoff=10.0,
            readout="add"
        ).to(self.device)
        try:
            state_dict = torch.load(model_path, map_location=self.device)
            self.model.load_state_dict(state_dict)
            print(f"StrainModelPredictor: 成功加载模型 {model_path}")
        except Exception as e:
            print(f"StrainModelPredictor: 加载模型 {model_path} 失败: {e}")
            print("请确保模型路径正确，并且模型架构参数与训练时一致。")

        self.model.eval()

    def predict(self, data, differentiable=True, return_tensor=False):
        """
        预测能量 (E_norm)。
        此 predict 方法的 API 与 ToxModelPredictor 完全相同。
        """
        grad_context = torch.enable_grad() if differentiable else torch.no_grad()

        with grad_context:
            if isinstance(data, list):
                raise TypeError("StrainModelPredictor.predict 不支持 list 输入")

            elif isinstance(data, Batch):
                z = data.z
                batch_vec = data.batch
                pos = data.pos
                pred = self.model(z, pos, batch_vec)

            elif isinstance(data, Data):
                if hasattr(data, 'batch') and data.batch is not None:
                    z = data.z
                    batch_vec = data.batch
                    pos = data.pos
                    pred = self.model(z, pos, batch_vec)
                else:
                    z = data.z
                    pos = data.pos
                    batch_vec = torch.zeros(z.size(0), dtype=torch.long, device=self.device)
                    pred = self.model(z, pos, batch_vec)

            else:
                raise TypeError("输入必须是 Data、Batch 或 list[Data]")

            # 统一输出形状
            if pred.dim() == 2 and pred.shape[1] == 1:
                pred = pred.squeeze(1)

            # 根据是否需要梯度返回
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


if __name__ == "__main__":
    MAX_SAMPLES = 200  # 预测的QM9分子数量
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # 1. 初始化预测器
    try:
        # model_path 必须指向您训练好的 .pth 文件
        predictor = StrainModelPredictor(model_path="schnet_qm9_E_strain.pth")
    except Exception as e:
        print(f"[预测信息] 模型加载失败: {e}")
        print("请确保 'schnet_qm9_E_strain.pth' 文件在同一目录下，或提供正确路径。")
        exit()

    # 2. 加载数据
    try:
        root_path = './data/QM9'
        dataset = QM9(root=root_path)
    except Exception as e:
        print(f"[预测信息] QM9 数据集加载失败: {e}")
        print("请确保 QM9 数据已下载到 './data/QM9' 目录下。")
        exit()
    dataset=dataset.to(device)
    # 只取前 MAX_SAMPLES 个分子
    subset = dataset[:MAX_SAMPLES]
    loader = DataLoader(subset, batch_size=32, shuffle=False)

    print(f"[预测信息] 开始预测QM9分子内部应变能（前{MAX_SAMPLES}个）...")

    # [修正] 变量名
    strain_results = {}
    global_idx = 0

    for batch in loader:
        # [修正] 调用 predict 时，明确设置 differentiable=False 来获取 list[float]
        scores = predictor.predict(batch, differentiable=False)

        for score in scores:
            strain_results[global_idx] = score
            global_idx += 1

    # 3. 输出结果
    print("\n[预测信息] QM9分子索引 -> 归一化内部应变能 (E_norm):")
    print("[提示] 这个值是归一化的能量，越低代表应变越小（越好）。")

    for mol_idx in range(min(20, len(strain_results))):
        score = strain_results[mol_idx]
        print(f"  索引: {mol_idx:6d} | E_norm: {score:.4f}")