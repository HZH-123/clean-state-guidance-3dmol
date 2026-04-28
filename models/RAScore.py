# sascore_wrapper.py
"""
智能 SAscore 封装：支持 SMILES 或 (z, pos) 3D 结构输入。
输出：0.0 ~ 1.0，值越大表示越容易合成（与 RAscore 语义一致）。
"""

import torch
import numpy as np
from typing import Union, List, Optional
from rdkit import Chem
from rdkit.Chem import AllChem
try:
    from rdkit.Chem import SAscore
    SASCORE_AVAILABLE = True
except ImportError:
    # Fallback: try rdkit.Chem.SA_Score (older RDKit versions)
    try:
        import rdkit.Chem.SA_Score as SAscore
        SASCORE_AVAILABLE = True
    except ImportError:
        SASCORE_AVAILABLE = False

if not SASCORE_AVAILABLE:
    raise ImportError("⚠️ SAscore not available. Please install RDKit with SAscore support.")


def _mol_from_z_pos(z: List[int], pos: Union[np.ndarray, torch.Tensor]) -> Optional[Chem.Mol]:
    """从原子序数和3D坐标构建RDKit分子"""
    try:
        if isinstance(pos, torch.Tensor):
            pos = pos.detach().cpu().numpy()
        if isinstance(z, torch.Tensor):
            z = z.detach().cpu().tolist()

        if len(z) != len(pos):
            return None

        # 创建分子
        mol = Chem.RWMol()
        for atomic_num in z:
            if atomic_num == 0:  # 跳过 padding
                continue
            mol.AddAtom(Chem.Atom(int(atomic_num)))

        if mol.GetNumAtoms() == 0:
            return None

        # 添加构象
        conf = Chem.Conformer(mol.GetNumAtoms())
        for i, coord in enumerate(pos[:mol.GetNumAtoms()]):
            conf.SetAtomPosition(i, coord.astype(float).tolist())
        mol.AddConformer(conf)

        # 推断化学键（基于距离）
        mol = Chem.AddHs(mol, addCoords=True)
        Chem.SanitizeMol(mol, sanitizeOps=Chem.SanitizeFlags.SANITIZE_ALL)
        return mol

    except Exception:
        return None


def sascore(
    smiles: Optional[str] = None,
    z: Optional[Union[List[int], torch.Tensor]] = None,
    pos: Optional[Union[np.ndarray, torch.Tensor]] = None
) -> float:
    """
    计算分子的 SAscore 可合成性分数，归一化到 [0, 1]。
    值越大表示越容易合成（与 RAscore 语义一致）。

    支持两种输入方式（二选一）：
      1. smiles: str
      2. z + pos: 原子序数列表 + 3D 坐标（QM9/SchNet 格式）

    Args:
        smiles: 分子 SMILES 字符串
        z: 原子序数列表，如 [6, 1, 1, 1, 1]（CH4）
        pos: 坐标矩阵，shape [N, 3]

    Returns:
        float: 归一化 SAscore ∈ [0, 1]。若输入无效，返回 0.0。
    """
    # 输入校验
    if (smiles is None) == (z is None or pos is None):
        raise ValueError("Please provide either 'smiles' OR ('z' and 'pos'), not both or neither.")

    # 情况1: 直接使用 SMILES
    if smiles is not None:
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return 0.0

    # 情况2: 从 z + pos 重建分子
    else:
        mol = _mol_from_z_pos(z, pos)
        if mol is None:
            return 0.0

    # 计算原始 SAscore（范围 ~1~10，越小越易合成）
    try:
        raw_score = SAscore.calculateScore(mol)
    except Exception:
        return 0.0

    # 归一化到 [0, 1]，越大越易合成
    # SAscore ∈ [1, 10] → normalized ∈ [0, 1]
    normalized = 1.0 - (raw_score - 1.0) / 9.0
    return float(np.clip(normalized, 0.0, 1.0))


# ======================
# 便捷别名
# ======================
get_synthesizability_score = sascore


# ======================
# 测试代码
# ======================
if __name__ == "__main__":
    print("🧪 Testing SAscore wrapper...")

    # 测试1: SMILES 输入
    print("From SMILES:")
    print("  Ethanol:", sascore(smiles="CCO"))
    print("  Benzene:", sascore(smiles="c1ccccc1"))

    # 测试2: z + pos 输入（模拟 QM9 数据）
    print("\nFrom z + pos (CH4):")
    z_ch4 = [6, 1, 1, 1, 1]
    pos_ch4 = np.array([
        [0.0000, 0.0000, 0.0000],  # C
        [0.6291, 0.6291, 0.6291],  # H
        [-0.6291, -0.6291, 0.6291],
        [-0.6291, 0.6291, -0.6291],
        [0.6291, -0.6291, -0.6291]
    ])
    print("  Methane:", sascore(z=z_ch4, pos=pos_ch4))

    print("\n✅ Done.")