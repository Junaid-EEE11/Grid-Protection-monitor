# IEEE 123-Bus Distribution Test Feeder

## 1. Overview
The IEEE 123-bus test feeder is a canonical unbalanced radial distribution network benchmark developed by the IEEE Power & Energy Society (PES) Distribution System Analysis Subcommittee. It operates at a nominal line-to-line voltage of 4.16 kV and features:
- Overhead lines and underground cables with varying phase configurations (3-phase, 2-phase, and single-phase branches).
- Highly unbalanced spot loads (single-phase and 3-phase).
- Four step-voltage regulator banks.
- Four shunt capacitor banks (three single-phase, one 3-phase).
- Multiple tie and sectionalizing switches for reconfiguration studies.

## 2. File Organization
- `IEEE123Master.dss`: Primary entry-point circuit compilation script.
- `LineCodes.dss`: Phase impedance matrices and sequence properties for conductors.
- `IEEE123Lines.dss`: Line section connectivity and switches.
- `IEEE123Loads.dss`: Unbalanced spot load definitions.
- `Capacitors.dss`: Shunt reactive compensation.
- `Regulators.dss`: Substation and in-line step-voltage regulators.
- `BusCoords.dss`: 2D spatial coordinate mapping for feeder topology visualization.

## 3. Assumptions & Adaptations
1. **Source Representation**: The primary substation is modeled as a 115 kV Thevenin equivalent source (`MVAsc3=20000`, `MVASC1=21000`) connected to a 5 MVA Delta-Wye substation transformer stepping down to 4.16 kV.
2. **Voltage Bases**: Standard base voltages are set to `[115.0, 4.16, 2.4, 0.48]` kV.
3. **Simulation Mode**: Steady-state snapshot power-flow and fault impedance solutions are used for phasor measurement extraction.
4. **License / Attribution**: Based on public IEEE PES Distribution Benchmark Models for academic and non-commercial research use.
