"""Integration tests for power-system simulation engine and OpenDSS fault scenarios."""

import pytest
from src.simulation.engine import PowerSystemSimulator
from src.simulation.fault_models import FaultScenario, FaultType


@pytest.mark.simulation
def test_simulation_normal_scenario():
    """Verify normal intact power-flow simulation."""
    sim = PowerSystemSimulator("feeders/ieee123/IEEE123Master.dss")
    scenario = FaultScenario.create_normal(
        scenario_id="test_norm_01",
        feeder_name="ieee123",
        loading_multiplier=1.0,
    )
    res = sim.execute_scenario(scenario)
    assert res["success"] is True
    assert "post_fault" in res
    assert len(res["post_fault"]["buses"]) == 123


@pytest.mark.simulation
def test_simulation_slg_fault():
    """Verify single-line-to-ground fault injection and measurement extraction."""
    sim = PowerSystemSimulator("feeders/ieee123/IEEE123Master.dss")
    scenario = FaultScenario.create_fault(
        scenario_id="test_slg_01",
        feeder_name="ieee123",
        fault_type=FaultType.SLG,
        fault_bus="65",
        faulted_phases=["A"],
        fault_resistance_ohm=1.0,
    )
    res = sim.execute_scenario(scenario)
    assert res["success"] is True
    assert res["scenario"]["is_fault"] is True

    # Bus 65 voltage on phase A should drop significantly
    bus65 = res["post_fault"]["buses"]["65"]
    assert bus65["v_pu"][0] < 0.85
