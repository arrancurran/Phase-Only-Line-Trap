import numpy as np

from line_phase import line_phase
from grating_phase import grating_phase


def build_hologram(
    Nx,
    Ny,
    scaling_factor,
    L,
    A0,
    angle,
    randomise,
    line_offset_x=0,
    line_offset_y=0,
    spot_offset_x=0,
    spot_offset_y=0,
    holo_fresnel=None,
):
    """Build a phase-only hologram for a line trap plus background spot.

    Parameters
    ----------
    Nx, Ny : int
        SLM dimensions in pixels.
    scaling_factor : float
        Spatial-frequency scaling factor p / (f * wavelength).
    L : float
        Line length in the sample plane (µm).
    A0 : float
        Row fill fraction cap in [0, 1].
    angle : float
        Line angle in degrees (0 = horizontal, 90 = vertical).
    randomise : bool
        Whether to randomise the selection mask S.
    line_offset_x, line_offset_y : float, optional
        Grating offsets for the line (µm).
    spot_offset_x, spot_offset_y : float, optional
        Grating offsets for the background spot (µm).

    Returns
    -------
    holo_total : ndarray (Ny, Nx)
        Phase-only hologram in radians.
    S : ndarray (Ny, Nx)
        Selection mask for the line region.
    """

    # Calculate the line phase pattern and selection mask S
    holo_line, S = line_phase(Nx, Ny, L, scaling_factor, angle, A0, randomise)
    

    grating_line = grating_phase(Nx, Ny, line_offset_x, line_offset_y, scaling_factor)

    # Optionally add a Fresnel (defocus) phase only inside the line region
    if holo_fresnel is not None:
        phase_line = holo_line + grating_line + holo_fresnel
    else:
        phase_line = holo_line + grating_line

    # Combine line, grating (and optional Fresnel) inside the selected region
    holo_line = np.where(
        S == 1,
        np.mod(phase_line, 2 * np.pi).astype(np.float32),
        0,
    ).astype(np.float32)

    # Grating for unassigned pixels, where S = 0
    grating_spot = grating_phase(Nx, Ny, spot_offset_x, spot_offset_y, scaling_factor)

    # Combine phases: where S=1 use holo_line, elsewhere use grating_spot
    holo_total = np.where(S == 0, grating_spot, holo_line).astype(np.float32)

    return holo_total, S