"""Symmetrical sequence component calculation using the Fortescue transformation."""

import cmath
import math
from typing import List, Tuple, Union
import numpy as np

# Complex Fortescue operator a = exp(j * 2*pi/3)
A_OPERATOR = cmath.rect(1.0, 2.0 * math.pi / 3.0)
A2_OPERATOR = cmath.rect(1.0, 4.0 * math.pi / 3.0)


def calculate_symmetrical_phasors(
    va_complex: complex, vb_complex: complex, vc_complex: complex
) -> Tuple[complex, complex, complex]:
    """Calculate positive, negative, and zero sequence complex phasors via Fortescue transform.

    Args:
        va_complex: Phase A complex phasor.
        vb_complex: Phase B complex phasor.
        vc_complex: Phase C complex phasor.

    Returns:
        Tuple of (v0, v1, v2) complex sequence phasors:
        - v0: Zero-sequence phasor (1/3 * (Va + Vb + Vc))
        - v1: Positive-sequence phasor (1/3 * (Va + a*Vb + a^2*Vc))
        - v2: Negative-sequence phasor (1/3 * (Va + a^2*Vb + a*Vc))
    """
    v0 = (va_complex + vb_complex + vc_complex) / 3.0
    v1 = (va_complex + A_OPERATOR * vb_complex + A2_OPERATOR * vc_complex) / 3.0
    v2 = (va_complex + A2_OPERATOR * vb_complex + A_OPERATOR * vc_complex) / 3.0
    return v0, v1, v2


def compute_sequence_components(
    magnitudes: Union[List[float], np.ndarray],
    angles_deg: Union[List[float], np.ndarray],
    phase_mask: Union[List[int], np.ndarray] = (1, 1, 1),
) -> Tuple[float, float, float]:
    """Compute magnitudes of (zero, positive, negative) sequence components from magnitudes and angles.

    Handles single-phase, two-phase, and three-phase availability safely.

    Args:
        magnitudes: 3-element list/array of phase magnitudes [A, B, C].
        angles_deg: 3-element list/array of phase angles in degrees [A, B, C].
        phase_mask: 3-element binary mask indicating present phases [A, B, C].

    Returns:
        Tuple of (mag_v0, mag_v1, mag_v2) representing sequence component magnitudes.
    """
    mags = np.asarray(magnitudes, dtype=float)
    angs = np.asarray(angles_deg, dtype=float)
    mask = np.asarray(phase_mask, dtype=int)

    # Convert present phases to complex numbers (polar to rectangular)
    phasors = []
    for i in range(3):
        if mask[i] == 1 and mags[i] > 0:
            rad = math.radians(angs[i])
            phasors.append(cmath.rect(float(mags[i]), rad))
        else:
            phasors.append(0.0 + 0.0j)

    va, vb, vc = phasors[0], phasors[1], phasors[2]
    num_phases = int(mask.sum())

    if num_phases == 3:
        v0, v1, v2 = calculate_symmetrical_phasors(va, vb, vc)
        return abs(v0), abs(v1), abs(v2)
    elif num_phases == 2:
        # For 2-phase branches, sequence decomposition is partially asymmetric
        v0, v1, v2 = calculate_symmetrical_phasors(va, vb, vc)
        return abs(v0), abs(v1), abs(v2)
    elif num_phases == 1:
        # Single-phase branch: positive sequence magnitude is equivalent to single phase / 3
        single_mag = max(abs(va), abs(vb), abs(vc))
        return 0.0, float(single_mag), 0.0
    else:
        return 0.0, 0.0, 0.0
