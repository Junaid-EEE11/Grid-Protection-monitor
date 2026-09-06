# Secondary Validation Feeder: IEEE 13-Bus

## 1. Overview
The IEEE 13-bus test feeder provides a compact yet highly challenging unbalanced test case used for secondary-network validation and cross-feeder generalization studies. Key attributes:
- Short, heavily loaded radial network operating at 4.16 kV.
- Highly unbalanced phase loading and configurations (single-phase, two-phase, and three-phase lines).
- In-line transformer (4.16 kV to 0.48 kV) and substation step regulator.
- Shunt capacitors at buses 675 (3-phase) and 611 (1-phase).

## 2. Research Purpose
This secondary feeder enables:
1. **Cross-Network Transfer Analysis**: Evaluating whether representations trained on IEEE 123-bus can transfer to or be fine-tuned on an unseen network with distinct graph topology, bus numbering, and impedance profiles.
2. **Methodological Verification**: Verifying that the proposed physics-guided GNN pipeline independently replicates qualitative robustness advantages on a distinct network architecture.
