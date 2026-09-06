"""CLI workflow script to generate power-system simulation scenarios and save dataset."""

import argparse
import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.data.dataset_audit import DatasetAuditor
from src.data.dataset_generator import DatasetGenerator
from src.utils.config import load_config
from src.utils.logging_utils import get_logger

logger = get_logger("generate_data")


def main():
    parser = argparse.ArgumentParser(description="Generate simulation dataset across load/fault parameter sweeps.")
    parser.add_argument("--config", type=str, default="configs/default_config.yaml", help="Path to YAML config.")
    parser.add_argument("--max_scenarios", type=int, default=None, help="Optional scenario limit for smoke tests.")
    parser.add_argument("--output_dir", type=str, default="data/raw/ieee123", help="Output directory path.")
    args = parser.parse_args()

    cfg = load_config(args.config)
    master_path = cfg.get("feeder", {}).get("master_path", "feeders/ieee123/IEEE123Master.dss")

    logger.info(f"Initializing DatasetGenerator with feeder: {master_path}")
    generator = DatasetGenerator(master_path=master_path, config=cfg)

    sim_cfg = cfg.get("simulation", {})
    scenarios = generator.generate_scenario_list(
        num_normal=sim_cfg.get("num_normal", 50),
        load_multipliers=sim_cfg.get("load_multipliers"),
        fault_resistances=sim_cfg.get("fault_resistances"),
        der_levels_kw=sim_cfg.get("der_penetrations_kw"),
        seed=sim_cfg.get("seed", 42),
    )

    logger.info(f"Generated {len(scenarios)} total planned scenario parameters.")

    dataset = generator.run_generation(
        scenarios=scenarios,
        sensor_coverage=sim_cfg.get("sensor_coverage", "dense"),
        max_scenarios=args.max_scenarios,
    )

    out_dir = Path(args.output_dir)
    generator.save_dataset(dataset, out_dir)

    # Run audit on generated dataset
    auditor = DatasetAuditor(out_dir / "metadata.csv")
    report = auditor.run_audit(total_feeder_buses=generator.feeder_graph.num_nodes)
    auditor.save_report(report, out_dir / "audit_report.json")
    print("\n" + "=" * 60)
    print(f"DATASET GENERATION REPORT: {report.summary_message}")
    print("=" * 60 + "\n")


if __name__ == "__main__":
    main()
