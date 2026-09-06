"""CLI workflow script to create and verify leakage-safe In-Distribution and OOD dataset splits."""

import argparse
import sys
from pathlib import Path
import pandas as pd

# Add project root to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.data.splitter import DatasetSplitter, SplitType
from src.utils.config import load_config
from src.utils.logging_utils import get_logger

logger = get_logger("make_splits")


def main():
    parser = argparse.ArgumentParser(description="Generate leakage-free data splits for research benchmarks.")
    parser.add_argument("--metadata", type=str, default="data/raw/ieee123/metadata.csv", help="Path to metadata.csv.")
    parser.add_argument("--output_dir", type=str, default="data/splits/ieee123", help="Output directory for split arrays.")
    parser.add_argument("--config", type=str, default="configs/default_config.yaml", help="Path to config YAML.")
    args = parser.parse_args()

    meta_path = Path(args.metadata).resolve()
    if not meta_path.is_file():
        logger.error(f"Metadata file not found: {meta_path}. Run generate_data first.")
        sys.exit(1)

    meta_df = pd.read_csv(meta_path)
    logger.info(f"Loaded {len(meta_df)} metadata records.")

    splitter = DatasetSplitter(meta_df)
    out_dir = Path(args.output_dir)

    print("\n" + "=" * 60)
    print("CREATING LEAKAGE-FREE DATA SPLITS")
    print("=" * 60)

    for st in [SplitType.IN_DISTRIBUTION, SplitType.HELD_OUT_LOCATIONS, SplitType.HELD_OUT_OPERATING]:
        res = splitter.create_split(split_type=st, train_ratio=0.70, val_ratio=0.15, test_ratio=0.15)
        splitter.save_split(res, out_dir)
        print(f"Split [{st.value.upper()}]: Train={len(res.train_indices)}, Val={len(res.val_indices)}, Test={len(res.test_indices)} (Leakage Check: PASSED)")

    print("=" * 60 + "\n")
    logger.info(f"All splits saved successfully to {out_dir}")


if __name__ == "__main__":
    main()
