import argparse
import sys
from pathlib import Path

from torch_geometric.datasets import QM9

THIS_DIR = Path(__file__).resolve().parent
SEMLA_ROOT = THIS_DIR.parent
PROJECT_ROOT = SEMLA_ROOT.parent
if str(SEMLA_ROOT) not in sys.path:
    sys.path.insert(0, str(SEMLA_ROOT))

from semlaflow.data.qm9_protocol import (
    QM9_PROTOCOL_SEED,
    build_protocol_labels,
    create_mudm_protocol_manifest,
    export_smol_split,
    load_qm9_sdf_supplier,
    resolve_qm9_root,
    save_protocol_labels,
    save_protocol_manifest,
)


def main(args):
    protocol_dir = Path(args.output_dir)
    protocol_dir.mkdir(parents=True, exist_ok=True)

    qm9_root = resolve_qm9_root(args.qm9_root, project_root=Path(__file__).resolve().parents[2])
    dataset = QM9(root=str(qm9_root))

    manifest = create_mudm_protocol_manifest(len(dataset), seed=args.seed)
    save_protocol_manifest(manifest, protocol_dir)

    labels = build_protocol_labels(dataset)
    save_protocol_labels(labels, protocol_dir)

    if args.skip_smol_export:
        print(f"Saved manifest and labels to {protocol_dir}")
        return

    supplier = load_qm9_sdf_supplier(qm9_root)
    for split_name in ["da_train", "db_train", "val", "test"]:
        source_split = split_name
        if split_name == "da_train":
            source_split = "da"
        elif split_name == "db_train":
            source_split = "db"

        split_path = export_smol_split(dataset, manifest, source_split, protocol_dir, supplier=supplier)
        if split_name != source_split:
            renamed_path = protocol_dir / f"{split_name}.smol"
            if renamed_path.exists():
                renamed_path.unlink()
            split_path.rename(renamed_path)
            split_path = renamed_path

        print(f"Saved {split_name} -> {split_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--qm9_root", type=str, default=None)
    parser.add_argument("--seed", type=int, default=QM9_PROTOCOL_SEED)
    parser.add_argument("--skip_smol_export", action="store_true")
    main(parser.parse_args())
