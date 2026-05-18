import numpy as np

from line_phase import line_phase
from grating_phase import grating_phase
from fresnel_phase import fresnel_phase
from astig_phase import phase_zernike


def build_hologram(
    Nx,
    Ny,
    f, 
    wavelength,
    p,
    scaling_factor,
    dz,
    L,
    A0,
    angle,
    randomise,
    line_offset_x=0,
    line_offset_y=0,
    spot_offset_x=0,
    spot_offset_y=0,
    spot_2_offset_x=0,
    spot_2_offset_y=0,
    spot_3_offset_x=0,
    spot_3_offset_y=0,
    spot_4_offset_x=0,
    spot_4_offset_y=0,
    astig_vertical=0.0,
    astig_oblique=0.0,
    spot_1_weight=1.0,
    spot_2_weight=1.0,
    spot_3_weight=1.0,
    spot_4_weight=1.0,
    rng=None,
):
    
    """Build a phase-only hologram for a line trap plus four background spots.

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
        Grating offsets for Spot 1 (µm).
    spot_2_offset_x, spot_2_offset_y : float, optional
        Grating offsets for Spot 2 (µm).
    spot_3_offset_x, spot_3_offset_y : float, optional
        Grating offsets for Spot 3 (µm).
    spot_4_offset_x, spot_4_offset_y : float, optional
        Grating offsets for Spot 4 (µm).
    spot_1_weight, spot_2_weight, spot_3_weight, spot_4_weight : float, optional
        Relative amplitudes A_n for each spot in the complex-field sum.

    Returns
    -------
    holo_total : ndarray (Ny, Nx)
        Phase-only hologram in radians.
    S : ndarray (Ny, Nx)
        Selection mask for the line region.
    """

    # Calculate the line phase pattern and selection mask S
    holo_line, S = line_phase(Nx, Ny, L, scaling_factor, angle, A0, randomise, rng=rng)
    
    grating_line = grating_phase(Nx, Ny, line_offset_x, line_offset_y, scaling_factor)
    
    grating_spot_1 = grating_phase(Nx, Ny, spot_offset_x, spot_offset_y, scaling_factor)
    grating_spot_2 = grating_phase(Nx, Ny, spot_2_offset_x, spot_2_offset_y, scaling_factor)
    grating_spot_3 = grating_phase(Nx, Ny, spot_3_offset_x, spot_3_offset_y, scaling_factor)
    grating_spot_4 = grating_phase(Nx, Ny, spot_4_offset_x, spot_4_offset_y, scaling_factor)

    if dz != 0.0:
        fresnel_line = fresnel_phase(Nx, Ny, dz_um=dz, f=f, wavelength=wavelength, pixel_pitch=p)
        holo_line = np.where(S == 1, np.mod(holo_line + fresnel_line + grating_line, 2 * np.pi).astype(np.float32), 0).astype(np.float32)
    else:
        holo_line = np.where(S == 1, np.mod(holo_line + grating_line, 2 * np.pi).astype(np.float32), 0).astype(np.float32)
    
    if astig_vertical != 0.0 or astig_oblique != 0.0:
        phase_astig = phase_zernike(Nx, Ny, c_astig_vertical=astig_vertical, c_astig_oblique=astig_oblique, pupil_radius_pix=None)
        
        holo_line = np.where(S == 1, np.mod(holo_line + phase_astig, 2 * np.pi).astype(np.float32), holo_line).astype(np.float32)
        
        grating_spot_1 = np.mod(grating_spot_1 + phase_astig, 2 * np.pi).astype(np.float32)
        grating_spot_2 = np.mod(grating_spot_2 + phase_astig, 2 * np.pi).astype(np.float32)
        grating_spot_3 = np.mod(grating_spot_3 + phase_astig, 2 * np.pi).astype(np.float32)
        grating_spot_4 = np.mod(grating_spot_4 + phase_astig, 2 * np.pi).astype(np.float32)
        
    # Combine spot holograms using:
    # Phi_comb = arg(sum_n A_n * exp(i * Phi_n)).
    spot_field_sum = (
        spot_1_weight * np.exp(1j * grating_spot_1)
        + spot_2_weight * np.exp(1j * grating_spot_2)
        + spot_3_weight * np.exp(1j * grating_spot_3)
        + spot_4_weight * np.exp(1j * grating_spot_4)
    )

    holo_comb = np.angle(spot_field_sum).astype(np.float32)
    holo_comb = np.mod(holo_comb, 2 * np.pi).astype(np.float32)

    # Combine phases: where S=1 use holo_line, elsewhere use grating_spot
    holo_total = np.where(S == 0, holo_comb, holo_line).astype(np.float32)

    return holo_total, S