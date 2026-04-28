import torch
from models.toxpredict_model import ToxModelPredictor

tox_predictor = ToxModelPredictor()

def energy_val(batch, w1=1.0, w2=1.0):
    """
    计算一个复合的多目标能量。
    """
    tox_scores = tox_predictor.predict_qm9_toxicity(batch)
    tox_scores = torch.tensor(tox_scores, dtype=torch.float32, device=batch.pos.device)
    tox_norm = (tox_scores - tox_scores.mean()) / (tox_scores.std() + 1e-6)
    energy = tox_norm*w1
    if energy.shape[0] != batch.num_graphs:
        raise ValueError(f"能量数量 {energy.shape[0]} ≠ 分子数 {batch.num_graphs}")
    return energy.detach()