"""Power-system simulation engine using OpenDSSDirect for reproducible fault scenarios."""

import math
import os
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union
import numpy as np
import opendssdirect as dss

from src.simulation.fault_models import FaultScenario, FaultType
from src.simulation.feeder_loader import FeederLoader
from src.utils.logging_utils import get_logger

logger = get_logger("simulation_engine")


class PowerSystemSimulator:
    """Executes deterministic power-flow and fault simulations on distribution feeders."""

    def __init__(self, master_dss_path: Union[str, Path]):
        self.loader = FeederLoader(master_dss_path)
        self.master_dss_path = Path(master_dss_path).resolve()
        self.feeder_dir = self.master_dss_path.parent
        self.bus_names: List[str] = []
        self._initialize()

    def _initialize(self) -> None:
        """Compile feeder and store canonical bus ordering."""
        self.loader.compile()
        self.bus_names = sorted([str(b).lower() for b in dss.Circuit.AllBusNames()])

    def reset_feeder(self) -> None:
        """Cleanly reset OpenDSS state to intact base feeder configuration."""
        self.loader.compile()

    def execute_scenario(self, scenario: FaultScenario) -> Dict[str, Any]:
        """Run complete power-flow and optional fault sequence for a scenario.

        Args:
            scenario: FaultScenario dataclass specifying operating parameters and fault settings.

        Returns:
            Dictionary containing scenario metadata, pre-fault and post-fault measurements,
            and convergence status.
        """
        orig_cwd = os.getcwd()
        try:
            os.chdir(self.feeder_dir)
            dss.Text.Command("Clear")
            dss.Text.Command(f'compile "{self.master_dss_path.name}"')

            # 1. Apply loading multiplier
            if scenario.loading_multiplier != 1.0:
                dss.Text.Command(f"Set LoadMult = {scenario.loading_multiplier:.4f}")

            # 2. Apply topology variant if specified
            if scenario.topology_variant == "switch_reconfig":
                # Alternative feeder configuration: close tie-switch sw6 (54-94), open sectionalizer sw1 (13-152)
                dss.Text.Command("Line.Sw6.Enabled=yes")
                dss.Text.Command("Line.Sw1.Enabled=no")

            # 3. Apply DER if configured
            if scenario.der_penetration_kw > 0 and scenario.der_bus:
                der_bus = scenario.der_bus.lower()
                dss.Text.Command(
                    f"New Generator.DER1 Bus1={der_bus}.1.2.3 Phases=3 "
                    f"kW={scenario.der_penetration_kw:.2f} pf=1.0 Model=1 kV=4.16"
                )

            # 4. Solve intact pre-fault power flow
            dss.Text.Command("solve")
            pre_converged = dss.Solution.Converged()
            if not pre_converged:
                logger.warning(f"Pre-fault power flow failed to converge for scenario {scenario.scenario_id}")
                scenario.converged = False
                return {"scenario": scenario.to_dict(), "success": False, "reason": "pre_fault_non_convergence"}

            pre_measurements = self._extract_measurements()

            # 5. Inject fault if scenario is faulted
            if scenario.is_fault:
                self._inject_fault(scenario)
                dss.Text.Command("solve")
                fault_converged = dss.Solution.Converged()
                if not fault_converged:
                    logger.warning(f"Fault power flow failed to converge for scenario {scenario.scenario_id}")
                    scenario.converged = False
                    return {"scenario": scenario.to_dict(), "success": False, "reason": "fault_non_convergence"}
                scenario.converged = True
                post_measurements = self._extract_measurements()
            else:
                scenario.converged = True
                post_measurements = pre_measurements

            # 6. Apply measurement noise if configured
            if scenario.measurement_noise_std > 0:
                post_measurements = self._apply_noise(
                    post_measurements, scenario.measurement_noise_std, scenario.random_seed
                )

            return {
                "scenario": scenario.to_dict(),
                "success": True,
                "pre_fault": pre_measurements,
                "post_fault": post_measurements,
            }

        except Exception as exc:
            logger.error(f"Error during simulation of scenario {scenario.scenario_id}: {exc}")
            scenario.converged = False
            return {"scenario": scenario.to_dict(), "success": False, "reason": str(exc)}
        finally:
            os.chdir(orig_cwd)

    def _inject_fault(self, scenario: FaultScenario) -> None:
        """Inject fault into the OpenDSS circuit according to scenario specification."""
        if not scenario.fault_bus:
            raise ValueError("Fault scenario must specify a fault_bus.")

        bus = scenario.fault_bus.lower()
        rf = max(1e-4, float(scenario.fault_resistance_ohm))
        ft = scenario.fault_type

        if ft == FaultType.SLG:
            phase_char = scenario.faulted_phases[0].upper() if scenario.faulted_phases else "A"
            phase_num = {"A": "1", "B": "2", "C": "3"}.get(phase_char, "1")
            dss.Text.Command(f"New Fault.FLT phases=1 bus1={bus}.{phase_num} r={rf:.5f}")

        elif ft == FaultType.LL:
            p1_char = scenario.faulted_phases[0].upper() if len(scenario.faulted_phases) > 0 else "A"
            p2_char = scenario.faulted_phases[1].upper() if len(scenario.faulted_phases) > 1 else "B"
            p1_num = {"A": "1", "B": "2", "C": "3"}.get(p1_char, "1")
            p2_num = {"A": "1", "B": "2", "C": "3"}.get(p2_char, "2")
            dss.Text.Command(f"New Fault.FLT phases=1 bus1={bus}.{p1_num} bus2={bus}.{p2_num} r={rf:.5f}")

        elif ft == FaultType.LLG:
            p1_char = scenario.faulted_phases[0].upper() if len(scenario.faulted_phases) > 0 else "A"
            p2_char = scenario.faulted_phases[1].upper() if len(scenario.faulted_phases) > 1 else "B"
            p1_num = {"A": "1", "B": "2", "C": "3"}.get(p1_char, "1")
            p2_num = {"A": "1", "B": "2", "C": "3"}.get(p2_char, "2")
            dss.Text.Command(f"New Fault.FLT phases=2 bus1={bus}.{p1_num}.{p2_num} r={rf:.5f}")

        elif ft == FaultType.THREE_PHASE:
            dss.Text.Command(f"New Fault.FLT phases=3 bus1={bus}.1.2.3 r={rf:.5f}")
        else:
            raise ValueError(f"Unsupported fault type: {ft}")

    def _extract_measurements(self) -> Dict[str, Any]:
        """Extract phasor voltages, currents, and powers from the current power-flow state."""
        bus_data: Dict[str, Dict[str, Any]] = {}
        all_buses = [str(b).lower() for b in dss.Circuit.AllBusNames()]

        for bus in all_buses:
            dss.Circuit.SetActiveBus(bus)
            nodes = dss.Bus.Nodes()
            v_mag_angle = dss.Bus.VMagAngle()  # Array: [mag1, ang1, mag2, ang2, ...]
            pu_voltages = dss.Bus.PuVoltage()  # Complex or pair
            base_kv = dss.Bus.kVBase() * 1000.0  # Line-to-neutral base in Volts

            # Standard 3-phase arrays (A=0, B=1, C=2), default 0 if phase not present
            v_mag = [0.0, 0.0, 0.0]
            v_ang = [0.0, 0.0, 0.0]
            v_pu = [0.0, 0.0, 0.0]
            phase_mask = [0, 0, 0]

            for i, node in enumerate(nodes):
                if 1 <= node <= 3 and (2 * i + 1) < len(v_mag_angle):
                    p_idx = node - 1
                    mag = float(v_mag_angle[2 * i])
                    ang = float(v_mag_angle[2 * i + 1])
                    v_mag[p_idx] = mag
                    v_ang[p_idx] = ang
                    phase_mask[p_idx] = 1
                    if base_kv > 0:
                        v_pu[p_idx] = mag / base_kv

            bus_data[bus] = {
                "v_mag": v_mag,
                "v_ang": v_ang,
                "v_pu": v_pu,
                "phase_mask": phase_mask,
                "base_kv": float(base_kv),
            }

        # Branch current measurements
        branch_data: Dict[str, Dict[str, Any]] = {}
        for line_name in dss.Lines.AllNames():
            dss.Lines.Name(line_name)
            currents = dss.CktElement.CurrentsMagAng()
            phases = dss.Lines.Phases()

            i_mag = [0.0, 0.0, 0.0]
            i_ang = [0.0, 0.0, 0.0]

            for p in range(min(phases, 3)):
                if (2 * p + 1) < len(currents):
                    i_mag[p] = float(currents[2 * p])
                    i_ang[p] = float(currents[2 * p + 1])

            branch_data[line_name.lower()] = {
                "i_mag": i_mag,
                "i_ang": i_ang,
                "phases": phases,
            }

        return {
            "buses": bus_data,
            "branches": branch_data,
        }

    def _apply_noise(
        self, measurements: Dict[str, Any], noise_std: float, seed: int
    ) -> Dict[str, Any]:
        """Apply zero-mean Gaussian measurement noise to voltages and currents."""
        rng = np.random.RandomState(seed)
        corrupted = {
            "buses": {},
            "branches": {},
        }

        for bus, bdata in measurements["buses"].items():
            v_pu = list(bdata["v_pu"])
            v_ang = list(bdata["v_ang"])
            for p in range(3):
                if bdata["phase_mask"][p] == 1:
                    # Multiplicative Gaussian noise on magnitude, additive on angle (degrees)
                    v_pu[p] = float(max(0.0, v_pu[p] * (1.0 + rng.normal(0, noise_std))))
                    v_ang[p] = float(v_ang[p] + rng.normal(0, noise_std * 5.0))

            corrupted["buses"][bus] = {
                "v_mag": [v * bdata["base_kv"] for v in v_pu],
                "v_ang": v_ang,
                "v_pu": v_pu,
                "phase_mask": bdata["phase_mask"],
                "base_kv": bdata["base_kv"],
            }

        for line, ldata in measurements["branches"].items():
            i_mag = list(ldata["i_mag"])
            i_ang = list(ldata["i_ang"])
            for p in range(3):
                if i_mag[p] > 0:
                    i_mag[p] = float(max(0.0, i_mag[p] * (1.0 + rng.normal(0, noise_std))))
                    i_ang[p] = float(i_ang[p] + rng.normal(0, noise_std * 5.0))

            corrupted["branches"][line] = {
                "i_mag": i_mag,
                "i_ang": i_ang,
                "phases": ldata["phases"],
            }

        return corrupted
