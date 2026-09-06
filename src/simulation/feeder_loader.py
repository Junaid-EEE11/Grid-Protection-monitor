"""Feeder loader and validation interface for OpenDSS distribution networks."""

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union
import opendssdirect as dss

from src.utils.logging_utils import get_logger

logger = get_logger("feeder_loader")


@dataclass
class FeederSummary:
    """Structured summary of distribution feeder components and power-flow state."""
    name: str
    feeder_name: str
    num_buses: int
    num_lines: int
    num_loads: int
    num_transformers: int
    num_capacitors: int
    total_active_power_kw: float
    total_reactive_power_kvar: float
    total_load_kw: float
    total_load_kvar: float
    converged: bool
    bus_names: List[str]


class FeederLoader:
    """Manages compilation, inspection, and parameter extraction for OpenDSS feeders."""

    def __init__(self, master_dss_path: Union[str, Path], auto_compile: bool = True):
        self.master_dss_path = Path(master_dss_path).resolve()
        if not self.master_dss_path.is_file():
            raise FileNotFoundError(f"DSS master file not found: {self.master_dss_path}")
        self.feeder_dir = self.master_dss_path.parent
        self.is_compiled = False
        if auto_compile:
            self.compile()

    def compile(self) -> bool:
        """Compile the OpenDSS circuit from the master DSS file.

        Returns:
            True if compilation and baseline solution succeeded.

        Raises:
            RuntimeError: If OpenDSS compilation fails.
        """
        orig_cwd = os.getcwd()
        try:
            os.chdir(self.feeder_dir)
            dss.Text.Command("Clear")
            dss.Text.Command(f'compile "{self.master_dss_path.name}"')
            dss.Text.Command("solve")
            converged = dss.Solution.Converged()
            if not converged:
                logger.warning(f"Initial power flow did not converge for {self.master_dss_path.name}")
            self.is_compiled = True
            return converged
        except Exception as exc:
            logger.error(f"Failed to compile OpenDSS circuit {self.master_dss_path}: {exc}")
            raise RuntimeError(f"OpenDSS compilation error: {exc}") from exc
        finally:
            os.chdir(orig_cwd)

    def get_summary(self) -> FeederSummary:
        """Retrieve comprehensive summary of the compiled feeder.

        Returns:
            FeederSummary dataclass containing component counts and power flow totals.
        """
        if not self.is_compiled:
            self.compile()

        orig_cwd = os.getcwd()
        try:
            os.chdir(self.feeder_dir)
            total_power = dss.Circuit.TotalPower()
            p_kw = float(total_power[0]) if total_power else 0.0
            q_kvar = float(total_power[1]) if total_power else 0.0
            buses = [str(b).lower() for b in dss.Circuit.AllBusNames()]
            feeder_name = self.master_dss_path.parent.name

            return FeederSummary(
                name=dss.Circuit.Name(),
                feeder_name=feeder_name,
                num_buses=len(buses),
                num_lines=len(dss.Lines.AllNames()),
                num_loads=len(dss.Loads.AllNames()),
                num_transformers=len(dss.Transformers.AllNames()),
                num_capacitors=len(dss.Capacitors.AllNames()),
                total_active_power_kw=abs(p_kw),
                total_reactive_power_kvar=abs(q_kvar),
                total_load_kw=abs(p_kw),
                total_load_kvar=abs(q_kvar),
                converged=dss.Solution.Converged(),
                bus_names=buses,
            )
        finally:
            os.chdir(orig_cwd)

    def get_bus_coordinates(self) -> Dict[str, Tuple[float, float]]:
        """Extract spatial (X, Y) coordinates for all buses where available.

        Returns:
            Dictionary mapping lowercase bus name to (x, y) float coordinates.
        """
        if not self.is_compiled:
            self.compile()

        orig_cwd = os.getcwd()
        try:
            os.chdir(self.feeder_dir)
            coords: Dict[str, Tuple[float, float]] = {}
            for bus in dss.Circuit.AllBusNames():
                bus_str = str(bus).lower()
                dss.Circuit.SetActiveBus(bus_str)
                x = dss.Bus.X()
                y = dss.Bus.Y()
                coords[bus_str] = (float(x), float(y))
            return coords
        finally:
            os.chdir(orig_cwd)

    def get_branches(self) -> List[Dict[str, Any]]:
        """Extract topology branch information (lines, switches, transformers).

        Returns:
            List of branch dictionaries with bus connectivity, length, and impedances.
        """
        if not self.is_compiled:
            self.compile()

        orig_cwd = os.getcwd()
        try:
            os.chdir(self.feeder_dir)
            branches = []
            for line_name in dss.Lines.AllNames():
                dss.Lines.Name(line_name)
                b1 = dss.Lines.Bus1().split(".")[0].lower()
                b2 = dss.Lines.Bus2().split(".")[0].lower()
                length = dss.Lines.Length()
                phases = dss.Lines.Phases()
                r1 = dss.Lines.R1()
                x1 = dss.Lines.X1()
                is_switch = line_name.lower().startswith("sw")

                branches.append({
                    "name": line_name,
                    "type": "switch" if is_switch else "line",
                    "bus1": b1,
                    "bus2": b2,
                    "length_kft": length,
                    "phases": phases,
                    "r1": r1,
                    "x1": x1,
                })
            return branches
        finally:
            os.chdir(orig_cwd)
