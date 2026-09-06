"""Dataset generation, sensor masking, leakage-safe splitting, and audit tools."""

from src.data.measurement_mask import SensorMaskGenerator, SensorCoverage
from src.data.dataset_generator import DatasetGenerator, SimulationDataset
from src.data.splitter import DatasetSplitter, SplitType, SplitResult, verify_no_leakage
from src.data.dataset_loader import DistributionDataset, create_dataloaders
from src.data.dataset_audit import DatasetAuditor, AuditReport

__all__ = [
    "SensorMaskGenerator",
    "SensorCoverage",
    "DatasetGenerator",
    "SimulationDataset",
    "DatasetSplitter",
    "SplitType",
    "SplitResult",
    "verify_no_leakage",
    "DistributionDataset",
    "create_dataloaders",
    "DatasetAuditor",
    "AuditReport",
]
