import numpy as np


def phase_zernike(
    Nx: int,
    Ny: int,
    c_astig_0: float = 0.0,
    c_astig_45: float = 0.0,
    pupil_radius_pix: float | None = None,
    wrap: bool = True,
):
    """Phase mask from Zernike astigmatism coefficients.

    This constructs a phase-only hologram made from the two lowest-order
    astigmatism Zernike modes (n = 2, m = ±2):

        - "0° / 90°" (vertical/horizontal) astigmatism  ~  r^2 cos(2θ)
        - "45° / 135°" (oblique) astigmatism            ~  r^2 sin(2θ)

    The resulting phase is

        φ(r, θ) = 2π [ c_astig_0 * Z_astig_0(r, θ) + c_astig_45 * Z_astig_45(r, θ) ]

    where Z_astig_0 ∝ r^2 cos(2θ) and Z_astig_45 ∝ r^2 sin(2θ). The overall
    normalisation is chosen such that coefficients are dimensionless and of
    order unity for typical strengths.

    Parameters
    ----------
    Nx, Ny : int
        SLM dimensions in pixels.
    c_astig_0 : float, optional
        Coefficient for the "0° / 90°" astigmatism Zernike mode
        (r^2 cos(2θ)).
    c_astig_45 : float, optional
        Coefficient for the "45° / 135°" astigmatism Zernike mode
        (r^2 sin(2θ)).
    pupil_radius_pix : float, optional
        Radius of the (circular) pupil in pixels used for normalising
        r ∈ [0, 1]. If None, min(Nx, Ny)/2 is used.
    wrap : bool, optional
        If True (default), wrap the phase into [0, 2π). If False, return
        the unwrapped phase.

    Returns
    -------
    phase : (Ny, Nx) float32
        Phase mask in radians implementing the requested Zernike
        astigmatism combination.
    """

    if pupil_radius_pix is None:
        pupil_radius_pix = min(Nx, Ny) / 2.0

    # Pixel coordinates centred at zero (in pixels)
    X_pix, Y_pix = np.meshgrid(
        np.arange(Nx, dtype=np.float32) - Nx / 2.0,
        np.arange(Ny, dtype=np.float32) - Ny / 2.0,
    )

    # Polar coordinates relative to the pupil
    r_pix = np.sqrt(X_pix**2 + Y_pix**2)
    r_norm = r_pix / float(pupil_radius_pix)
    theta = np.arctan2(Y_pix, X_pix)

    # Basic Zernike astigmatism modes (up to overall normalisation)
    Z_astig_0 = r_norm**2 * np.cos(2.0 * theta)  # "0/90" astigmatism
    Z_astig_45 = r_norm**2 * np.sin(2.0 * theta)  # "45/135" astigmatism

    # Linear combination with user-specified coefficients
    Z = c_astig_0 * Z_astig_0 + c_astig_45 * Z_astig_45

    phase = 2.0 * np.pi * Z

    if wrap:
        phase = np.mod(phase, 2.0 * np.pi)

    return phase.astype(np.float32)
