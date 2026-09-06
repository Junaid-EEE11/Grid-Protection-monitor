"""CLI script to validate OpenDSS feeder compilation, convergence, and graph extraction."""

import argparse
import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.simulation.feeder_loader import FeederLoader
from src.topology.graph_builder import DistributionGraphBuilder
from src.utils.logging_utils import get_logger

logger = get_logger("validate_feeder")


def main():
    parser = argparse.ArgumentParser(description="Validate OpenDSS distribution feeder model.")
    parser.add_argument(
        "--master",
        type=str,
        default="feeders/ieee123/IEEE123Master.dss",
        help="Path to feeder master DSS file.",
    )
    args = parser.parse_args()

    master_path = Path(args.master).resolve()
    logger.info(f"Validating feeder from {master_path}...")

    loader = FeederLoader(master_path)
    summary = loader.get_summary()

    print("\n" + "=" * 60)
    print("FEEDER VALIDATION SUMMARY")
    print("=" * 60)
    print(f"Feeder Name:        {summary.feeder_name}")
    print(f"Master File:        {master_path}")
    print(f"Total Buses:        {summary.num_buses}")
    print(f"Total Lines:        {summary.num_lines}")
    print(f"Total Transformers: {summary.num_transformers}")
    print(f"Total Capacitors:   {summary.num_capacitors}")
    print(f"Total Loads:        {summary.num_loads}")
    print(f"Total Base Load:    {summary.total_load_kw:.2f} kW, {summary.total_load_kvar:.2f} kVAR")
    print(f"Power Flow Solved:  {summary.converged}")
    print("=" * 60)

    if not summary.converged:
        logger.error("Feeder power flow FAILED to converge!")
        sys.exit(1)

    graph_builder = DistributionGraphBuilder(master_path)
    feeder_graph = graph_builder.build_graph()
    print(f"Graph Construction: SUCCESS ({feeder_graph.num_nodes} nodes, {feeder_graph.num_edges} edges)")
    print("=" * 60 + "\n")
    logger.info("Validation completed successfully.")


if __name__ == "__main__":
    main()
