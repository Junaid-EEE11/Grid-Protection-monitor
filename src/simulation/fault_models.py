"""Fault model definitions, phase enums, and structured scenario representations."""

from dataclasses import asdict, dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional


class FaultType(str, Enum):
    """Enumeration of fault categories in unbalanced distribution grids."""
    NORMAL = "normal"
    SLG = "single_line_to_ground"
    LL = "line_to_line"
    LLG = "double_line_to_ground"
    THREE_PHASE = "three_phase"


class Phase(str, Enum):
    """Phase conductors and neutral."""
    A = "A"
    B = "B"
    C = "C"
    AB = "AB"
    BC = "BC"
    CA = "CA"
    ABC = "ABC"


@dataclass
class FaultScenario:
    """Explicit parameters defining a power-system simulation scenario."""
    scenario_id: str
    feeder_name: str
    is_fault: bool
    fault_type: FaultType
    faulted_phases: List[str]
    fault_bus: Optional[str]
    fault_resistance_ohm: float = 0.0
    loading_multiplier: float = 1.0
    der_penetration_kw: float = 0.0
    der_bus: Optional[str] = None
    topology_variant: str = "base"
    sensor_config: str = "dense"
    measurement_noise_std: float = 0.0
    random_seed: int = 42
    converged: bool = False
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Convert scenario definition to a serializable dictionary."""
        d = asdict(self)
        d["fault_type"] = self.fault_type.value
        return d

    @classmethod
    def create_normal(
        cls,
        scenario_id: str,
        feeder_name: str,
        loading_multiplier: float = 1.0,
        der_penetration_kw: float = 0.0,
        der_bus: Optional[str] = None,
        topology_variant: str = "base",
        seed: int = 42,
    ) -> "FaultScenario":
        """Factory for generating intact normal operating scenarios."""
        return cls(
            scenario_id=scenario_id,
            feeder_name=feeder_name,
            is_fault=False,
            fault_type=FaultType.NORMAL,
            faulted_phases=[],
            fault_bus=None,
            fault_resistance_ohm=0.0,
            loading_multiplier=loading_multiplier,
            der_penetration_kw=der_penetration_kw,
            der_bus=der_bus,
            topology_variant=topology_variant,
            random_seed=seed,
        )

    @classmethod
    def create_fault(
        cls,
        scenario_id: str,
        feeder_name: str,
        fault_type: FaultType,
        fault_bus: str,
        faulted_phases: List[str],
        fault_resistance_ohm: float,
        loading_multiplier: float = 1.0,
        der_penetration_kw: float = 0.0,
        der_bus: Optional[str] = None,
        topology_variant: str = "base",
        seed: int = 42,
    ) -> "FaultScenario":
        """Factory for generating explicit fault scenarios."""
        return cls(
            scenario_id=scenario_id,
            feeder_name=feeder_name,
            is_fault=True,
            fault_type=fault_type,
            faulted_phases=faulted_phases,
            fault_bus=fault_bus.lower(),
            fault_resistance_ohm=float(fault_resistance_ohm),
            loading_multiplier=loading_multiplier,
            der_penetration_kw=der_penetration_kw,
            der_bus=der_bus,
            topology_variant=topology_variant,
            random_seed=seed,
        )
