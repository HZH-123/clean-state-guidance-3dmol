import os
import numpy as np
import torch
import torch.nn.functional as F
from torch_geometric.loader import DataLoader
from torch_geometric.nn import SchNet
from torch.optim.lr_scheduler import ReduceLROnPlateau
import argparse
import logging
import pickle  # <-- 用于加载 .smol (假设)

# --- RDKit Imports ---
from rdkit import Chem
from rdkit.Chem import AllChem

# --- PyG Imports ---
import torch_geometric.data as gdata
# [!!] 删除了 InMemoryDataset


# -----------------------------------------------------------------------------
# 0. SMOL 加载函数 (不变, 已验证)
# -----------------------------------------------------------------------------
def load_smol_file(smol_path: str):
    """
    加载 .smol 文件 (V7)
    """
    print(f"Loading .smol file (assuming pickle): {smol_path}")

    bond_type_map = {
        1.0: Chem.BondType.SINGLE,
        2.0: Chem.BondType.DOUBLE,
        3.0: Chem.BondType.TRIPLE,
        1.5: Chem.BondType.AROMATIC,
        1: Chem.BondType.SINGLE,
        2: Chem.BondType.DOUBLE,
        3: Chem.BondType.TRIPLE,
        4: Chem.BondType.AROMATIC,
        4.0: Chem.BondType.AROMATIC,
    }

    try:
        with open(smol_path, 'rb') as f:
            bytes_list = pickle.load(f)

        if not isinstance(bytes_list, list) or len(bytes_list) == 0:
            print("Error or Warning: .smol file is not a valid list or is empty.")
            return []
        if not isinstance(bytes_list[0], bytes):
            raise TypeError(f"Expected list to contain 'bytes' objects, but found {type(bytes_list[0])}")

        print(f"Successfully loaded {len(bytes_list)} serialized objects.")
        print("Now, deserializing dicts and constructing RDKit Mol objects...")

        mol_list = []
        corrupted_count = 0
        printed_bond_warning = False

        for i, b in enumerate(bytes_list):
            mol = None
            d = None
            try:
                d = pickle.loads(b)
                rw_mol = Chem.RWMol()

                atomics = d['atomics']
                charges = d['charges']
                for atomic_num, charge in zip(atomics, charges):
                    atom = Chem.Atom(int(atomic_num.item()))
                    atom.SetFormalCharge(int(charge.item()))
                    rw_mol.AddAtom(atom)

                bonds = d['bonds']
                for bond_info_tensor in bonds:
                    idx1 = int(bond_info_tensor[0].item())
                    idx2 = int(bond_info_tensor[1].item())
                    order = bond_info_tensor[2].item()
                    bond_type = bond_type_map.get(order)
                    if bond_type is None:
                        if not printed_bond_warning:
                            print(f"--- [ 调试: 未知键级 ] ---")
                            print(f"在 Object {i} 中发现未知键级: {order}。将其设置为 UNSPECIFIED。")
                            print(f"------------------------------")
                            printed_bond_warning = True
                        bond_type = Chem.BondType.UNSPECIFIED
                    rw_mol.AddBond(idx1, idx2, bond_type)

                mol = rw_mol.GetMol()
                if mol is None:
                    raise ValueError("rw_mol.GetMol() returned None")

                coords = d['coords']
                conf = Chem.Conformer(mol.GetNumAtoms())
                for atom_idx in range(mol.GetNumAtoms()):
                    pos = [float(c.item()) for c in coords[atom_idx]]
                    conf.SetAtomPosition(atom_idx, tuple(pos))
                mol.AddConformer(conf)

                Chem.SanitizeMol(mol)

                if mol is not None:
                    mol_list.append(mol)
                else:
                    corrupted_count += 1

            except Exception as e:
                corrupted_count += 1
                if i == 0:
                    print(f"--- [ 调试: 构造失败 ] ---")
                    print(f"在 Object 0 上构造 Mol 时出错: {e}")
                    if d and 'bonds' in d and len(d['bonds']) > 0:
                        print(f"Bonds[0] (实际格式): {d['bonds'][0]}")
                    print(f"-----------------------------")

        print(f"Successfully constructed {len(mol_list)} molecules.")
        if corrupted_count > 0:
            print(f"Warning: Skipped {corrupted_count} corrupted or invalid objects during construction.")
        return mol_list
    except Exception as e:
        print(f"\n--- [ 错误 ] ---")
        print(f"加载或反序列化 {smol_path} 时出错: {e}")
        print(f"------------------\n")
        raise


# -----------------------------------------------------------------------------
# 1. RDKit 应变能函数 (非调试版)
# -----------------------------------------------------------------------------
logging.getLogger("rdkit").setLevel(logging.CRITICAL)


def calculate_strain_energy(mol: Chem.Mol,
                            force_field: str = 'mmff94',
                            per_atom: bool = False):
    """
    为一个 RDKit Mol 对象计算应变能 (E_initial - E_optimized)。
    """
    try:
        # [!!] 我们在构造时已经清理过了，但再次检查以防万一
        if Chem.SanitizeMol(mol, catchErrors=True) != Chem.SanitizeFlags.SANITIZE_NONE:
            return None
        if mol.GetNumConformers() == 0:
            return None

        mol_orig = Chem.Mol(mol)
        mol_orig_h = Chem.AddHs(mol_orig, addCoords=True)
        if mol_orig_h is None: return None

        mp_orig = AllChem.MMFFGetMoleculeProperties(mol_orig_h, mmffVariant=force_field)
        if mp_orig is None: return None
        ff_orig = AllChem.MMFFGetMoleculeForceField(mol_orig_h, mp_orig)
        if ff_orig is None: return None
        energy_orig = ff_orig.CalcEnergy()

        mol_opt = Chem.Mol(mol_orig_h)
        opt_result = AllChem.MMFFOptimizeMolecule(mol_opt, mmffVariant=force_field, maxIters=500)
        if opt_result != 0: return None

        mp_opt = AllChem.MMFFGetMoleculeProperties(mol_opt, mmffVariant=force_field)
        if mp_opt is None: return None
        ff_opt = AllChem.MMFFGetMoleculeForceField(mol_opt, mp_opt)
        if ff_opt is None: return None
        energy_opt = ff_opt.CalcEnergy()

        strain_energy = energy_orig - energy_opt

        if per_atom:
            num_atoms = mol_orig_h.GetNumAtoms()
            if num_atoms == 0: return None
            return strain_energy / num_atoms
        else:
            return strain_energy
    except Exception as e:
        return None


# -----------------------------------------------------------------------------
# 2. [!! 恢复 !!] 内存中处理函数
# -----------------------------------------------------------------------------
def load_and_process_split(split_name: str, root_path: str):
    """
    加载 .smol 文件, 计算应变能, 并返回 PyG Data 列表。
    """

    # [!!] 跳过 'raw' 文件夹
    smol_path = os.path.join(root_path, f"{split_name}.smol")

    if not os.path.exists(smol_path):
        raise FileNotFoundError(f"未在 {smol_path} 找到 {split_name}.smol")

    # 1. 加载 Mol 对象
    print(f"Processing {smol_path}...")
    mol_list = load_smol_file(smol_path)

    data_list = []
    valid_count = 0
    failed_count = 0

    print(f"Calculating strain energy for {split_name} split...")
    for mol in mol_list:
        if mol is None: continue

        strain_energy = calculate_strain_energy(mol, per_atom=False)

        if strain_energy is None:
            failed_count += 1
            continue

        try:
            z = torch.tensor([atom.GetAtomicNum() for atom in mol.GetAtoms()], dtype=torch.long)
            conf = mol.GetConformer(0)
            pos = torch.tensor(conf.GetPositions(), dtype=torch.float)
        except Exception as e:
            failed_count += 1
            continue

        data = gdata.Data(z=z, pos=pos, y=torch.tensor([strain_energy], dtype=torch.float))
        data_list.append(data)
        valid_count += 1

    print(
        f"Split '{split_name}': Processed {valid_count} molecules, failed/skipped {failed_count}.")

    if len(data_list) == 0:
        print(f"Warning: {split_name} 中没有找到有效的分子。")

    return data_list


# -----------------------------------------------------------------------------
# 3. [修改后的] 微调 (Fine-Tuning) 主脚本
# -----------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser()
    # [!!] 保持你的路径
    parser.add_argument('--data_path', type=str, default='../semla-flow/data/geom-drugs/smol',
                        help='Path to dataset root (contains train.smol, val.smol)')
    parser.add_argument('--pretrained_path', type=str, default='schnet_qm9_E_strain.pth',
                        help='Path to the pretrained QM9 model weights')
    parser.add_argument('--save_path', type=str, default='schnet_geom_strain_finetuned.pth',
                      help='Path to save the fine-tuned model')
    parser.add_argument('--epochs', type=int, default=300)
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--lr', type=float, default=1e-5)
    parser.add_argument('--patience', type=int, default=20)
    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # --- [!! 核心修改 !!] ---
    # --- 切换回即时处理 ---
    root_path = args.data_path

    try:
        # 1. 手动加载和处理训练集
        train_dataset = load_and_process_split(split_name='train', root_path=root_path)
        # 2. 手动加载和处理验证集
        val_dataset = load_and_process_split(split_name='val', root_path=root_path)
    except Exception as e:
        print(f"\n数据集加载或处理失败: {e}")
        print("请检查 'load_smol_file' 函数和你的文件路径。")
        return

    print(f"\nIn-memory processing complete.")
    print(f"Train: {len(train_dataset)}, Val: {len(val_dataset)}")

    if len(train_dataset) == 0:
        print("\n错误: 训练集为空。请检查 'process' 步骤的日志。")
        return
    # --- [!! 核心修改结束 !!] ---

    # --- 统计计算 (不变) ---
    print("Calculating statistics for the fine-tuning target (strain energy)...")
    train_targets = []
    for data in train_dataset:
        train_targets.append(data.y.item())

    train_targets = torch.tensor(train_targets)
    mean = float(train_targets.mean())
    std = float(train_targets.std())
    print(f"Finetune Target Stats -> Mean: {mean:.4f}, Std: {std:.4f}")

    # --- DataLoader (不变) ---
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=0)

    print(f"Target: Fine-tuning on Strain Energy")

    # --- 加载模型和预训练权重 (不变) ---
    model = SchNet(
        hidden_channels=128,
        num_filters=128,
        num_interactions=6,
        num_gaussians=50,
        cutoff=10.0,
        readout='add',
    ).to(device)

    print(f"Loading pretrained weights from: {args.pretrained_path}")
    try:
        model.load_state_dict(torch.load(args.pretrained_path, map_location=device), strict=True)
    except FileNotFoundError:
        print(f"Warning: Pretrained model file not found at {args.pretrained_path}. Training from scratch.")
    except Exception as e:
        print(f"Error loading state dict: {e}")
        print("Warning: Could not load pretrained weights. Training from scratch.")

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.8, patience=5, min_lr=1e-7)
    best_val_loss = float('inf')
    patience_counter = 0

    # ----------------------------
    # 训练循环 (不变)
    # ----------------------------
    for epoch in range(1, args.epochs + 1):
        model.train()
        train_loss = 0
        for data in train_loader:
            data = data.to(device)
            optimizer.zero_grad()
            pred = model(data.z, data.pos, data.batch)
            target = data.y.to(device)
            target = (target - mean) / std
            target = target.unsqueeze(1)
            loss = F.l1_loss(pred, target)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            train_loss += loss.item() * data.num_graphs
        train_loss_avg = train_loss / len(train_dataset)

        model.eval()
        val_loss = 0
        with torch.no_grad():
            for data in val_loader:
                data = data.to(device)
                pred = model(data.z, data.pos, data.batch)
                target = data.y.to(device)
                target = (target - mean) / std
                target = target.unsqueeze(1)
                loss = F.l1_loss(pred, target)
                val_loss += loss.item() * data.num_graphs
        val_loss_avg = val_loss / len(val_dataset)

        scheduler.step(val_loss_avg)

        if val_loss_avg < best_val_loss:
            best_val_loss = val_loss_avg
            patience_counter = 0
            torch.save(model.state_dict(), args.save_path)
            print(f" Epoch {epoch:03d} | Train Loss: {train_loss_avg:.6f} | Val Loss: {val_loss_avg:.6f} | Saved!")
        else:
            patience_counter += 1
            print(f"Epoch {epoch:03d} | Train Loss: {train_loss_avg:.6f} | Val Loss: {val_loss_avg:.6f}")

        if patience_counter >= args.patience:
            print(f"Early stopping at epoch {epoch}")
            break

    print(f"Fine-tuning finished. Best val loss: {best_val_loss:.6f}")
    print(f"Model saved to: {os.path.abspath(args.save_path)}")


if __name__ == '__main__':
    main()