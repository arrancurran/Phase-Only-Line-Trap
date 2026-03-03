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
    astig_vertical=0.0,
    astig_oblique=0.0,
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
    
    # Grating for unassigned pixels, where S = 0
    grating_spot = grating_phase(Nx, Ny, spot_offset_x, spot_offset_y, scaling_factor)

    if dz != 0.0:
        fresnel_line = fresnel_phase(Nx, Ny, dz_um=dz, f=f, wavelength=wavelength, pixel_pitch=p)
        holo_line = np.where(S == 1, np.mod(holo_line + fresnel_line + grating_line, 2 * np.pi).astype(np.float32), 0).astype(np.float32)
    else:
        holo_line = np.where(S == 1, np.mod(holo_line + grating_line, 2 * np.pi).astype(np.float32), 0).astype(np.float32)
    
    if astig_vertical != 0.0 or astig_oblique != 0.0:
        phase_astig = phase_zernike(Nx, Ny, c_astig_vertical=astig_vertical, c_astig_oblique=astig_oblique, pupil_radius_pix=None)
        
        holo_line = np.where(S == 1, np.mod(holo_line + phase_astig, 2 * np.pi).astype(np.float32), holo_line).astype(np.float32)
        
        grating_spot = np.mod(grating_spot + phase_astig, 2 * np.pi).astype(np.float32)
        


    # Combine phases: where S=1 use holo_line, elsewhere use grating_spot
    holo_total = np.where(S == 0, grating_spot, holo_line).astype(np.float32)

    return holo_total, S