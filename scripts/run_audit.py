"""CLI workflow script to run dataset audits and check split leakage."""

import argparse
import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.data.dataset_audit import DatasetAuditor
from src.utils.logging_utils import get_logger

logger = get_logger("cli_audit")


def main():
    parser = argparse.ArgumentParser(description="Audit dataset integrity, distributions, and leakage.")
    parser.add_argument("--metadata", type=str, default="data/raw/ieee123/metadata.csv", help="Path to metadata CSV.")
    parser.add_argument("--output_file", type=str, default="results/metrics/audit_report.json", help="Report output file.")
    args = parser.parse_args()

    meta_path = Path(args.metadata).resolve()
    if not meta_path.is_file():
        logger.error(f"Metadata file not found: {meta_path}")
        sys.exit(1)

    auditor = DatasetAuditor(meta_path)
    report = auditor.run_audit(total_feeder_buses=123)

    out_file = Path(args.output_file).resolve()
    auditor.save_report(report, out_file)

    print("\n" + "=" * 60)
    print("DATASET SCIENTIFIC AUDIT REPORT")
    print("=" * 60)
    print(f"Total Scenarios:         {report.total_scenarios}")
    print(f"Normal Intact Scenarios: {report.num_normal}")
    print(f"Fault Scenarios:         {report.num_faulted}")
    print(f"Fault Types:             {report.fault_type_distribution}")
    print(f"Fault Resistance Range:  {report.fault_resistance_quantiles['min']:.2f} to {report.fault_resistance_quantiles['max']:.2f} Ohm")
    print(f"Unique Fault Buses:      {report.unique_fault_buses}/{report.total_buses}")
    print(f"Missing Values Check:    {report.missing_values_detected}")
    print(f"Leakage Verification:    {'PASSED' if report.leakage_passed else 'FAILED'}")
    print("=" * 60 + "\n")


if __name__ == "__main__":
    main()
