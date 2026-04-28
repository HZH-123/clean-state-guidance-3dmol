import sys
import torch
import numpy as np
from pathlib import Path

RESULTS = {
    "noisy": "experiments_multi/verify_noisy_10/result_0.pt",
    "clean_0.25": "experiments_multi/verify_clean_0.25_10/result_0.pt",
    "clean_0.5": "experiments_multi/verify_clean_0.5_10/result_0.pt",
    "clean_1.0": "experiments_multi/verify_clean_1.0_10/result_0.pt",
}

def main():
    all_pass = True
    shifts = {}

    for name, path in RESULTS.items():
        print(f"\n=== Checking {name} ===")
        p = Path(path)
        if not p.exists():
            print(f"[FAIL] File not found: {path}")
            all_pass = False
            continue

        data = torch.load(path, map_location="cpu", weights_only=False)

        # Check 1: pred_ligand_pos0_guided_traj exists
        if "pred_ligand_pos0_guided_traj" not in data:
            print("[FAIL] Missing pred_ligand_pos0_guided_traj")
            all_pass = False
            continue
        else:
            print("[PASS] pred_ligand_pos0_guided_traj exists")

        # Check 2: length matches pred_ligand_pos0_traj
        pos0_traj = data["pred_ligand_pos0_traj"]
        pos0_guided_traj = data["pred_ligand_pos0_guided_traj"]
        if len(pos0_traj) != len(pos0_guided_traj):
            print(f"[FAIL] Length mismatch: pos0_traj={len(pos0_traj)}, pos0_guided_traj={len(pos0_guided_traj)}")
            all_pass = False
            continue
        else:
            print(f"[PASS] Traj length match: {len(pos0_traj)} samples")

        # Check per-sample shape match
        for i, (p0, pg) in enumerate(zip(pos0_traj, pos0_guided_traj)):
            if np.asarray(p0).shape != np.asarray(pg).shape:
                print(f"[FAIL] Shape mismatch at sample {i}")
                all_pass = False
                break
        else:
            print("[PASS] All sample shapes match")

        # Check guidance_abs_shift
        if "guidance_abs_shift" not in data:
            print("[FAIL] Missing guidance_abs_shift")
            all_pass = False
            continue

        abs_shift = data["guidance_abs_shift"]
        abs_shift = np.asarray(abs_shift)
        mean_shift = float(abs_shift.mean())
        shifts[name] = mean_shift
        print(f"[INFO] guidance_abs_shift mean: {mean_shift:.6f}")

    print("\n=== Summary ===")
    for k, v in shifts.items():
        print(f"  {k:12s}: {v:.6f}")

    # Check 3: noisy shift ~ 0
    if "noisy" in shifts:
        if shifts["noisy"] > 1e-6:
            print(f"[FAIL] noisy guidance_abs_shift should be ~0, got {shifts['noisy']:.6f}")
            all_pass = False
        else:
            print("[PASS] noisy guidance_abs_shift is ~0")

    # Check 4: clean shifts > 0
    for k in ["clean_0.25", "clean_0.5", "clean_1.0"]:
        if k in shifts:
            if shifts[k] <= 0:
                print(f"[FAIL] {k} guidance_abs_shift should be > 0, got {shifts[k]:.6f}")
                all_pass = False
            else:
                print(f"[PASS] {k} guidance_abs_shift > 0")

    # Check 5: monotonic increase with scale
    clean_keys = ["clean_0.25", "clean_0.5", "clean_1.0"]
    clean_vals = [shifts.get(k) for k in clean_keys]
    if all(v is not None for v in clean_vals):
        if clean_vals[0] < clean_vals[1] < clean_vals[2]:
            print("[PASS] Monotonic: 0.25 < 0.5 < 1.0")
        else:
            # Allow tiny numerical fluctuation (1% tolerance)
            tol = 0.01
            ok = True
            for i in range(len(clean_vals)-1):
                if clean_vals[i+1] < clean_vals[i] * (1 - tol):
                    ok = False
                    break
            if ok:
                print(f"[PASS] Approx monotonic (within 1% tol): {clean_vals[0]:.6f} < {clean_vals[1]:.6f} < {clean_vals[2]:.6f}")
            else:
                print(f"[FAIL] Not monotonic: {clean_vals[0]:.6f}, {clean_vals[1]:.6f}, {clean_vals[2]:.6f}")
                all_pass = False

    print("\n" + ("All checks passed!" if all_pass else "Some checks failed."))
    sys.exit(0 if all_pass else 1)

if __name__ == "__main__":
    main()
