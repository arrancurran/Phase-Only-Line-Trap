import numpy as np


def astig_phase(Nx, Ny, dzx_um, dzy_um, f, wavelength, pixel_pitch=15e-6):
    """Quadratic phase mask for *cylindrical* astigmatic defocus in x and y.

    This implements different focal power along x and y (like a pair of
    orthogonal cylindrical lenses). It is *not* the pure Zernike
    astigmatism mode but can be useful for physical modelling of
    anisotropic defocus.
    """

    # Convert to metres
    dzx = dzx_um * 1e-6
    dzy = dzy_um * 1e-6

    # Pixel indices centred at zero
    X_pix, Y_pix = np.meshgrid(
        np.arange(Nx, dtype=np.float32) - Nx / 2.0,
        np.arange(Ny, dtype=np.float32) - Ny / 2.0,
    )

    # Physical coordinates on SLM [m]
    X = X_pix * pixel_pitch
    Y = Y_pix * pixel_pitch

    # Astigmatic quadratic phase:
    #   phi(x, y) = pi/(lambda * f^2) * (dzx * x^2 + dzy * y^2)
    coeff_x = np.pi * dzx / (wavelength * f**2)
    coeff_y = np.pi * dzy / (wavelength * f**2)
    phase = coeff_x * X**2 + coeff_y * Y**2

    return phase.astype(np.float32)


def zernike_astig_phase(Nx, Ny, strength=1.0, angle_deg=0.0, pupil_radius_pix=None):
    """Zernike astigmatism phase (J = ±2, n = 2).

    This generates a pure Zernike-like astigmatism term of the form

        Z_astig(r, θ) ∝ r^2 cos[2 (θ - α)],

    where `α` is the astigmatism axis orientation.

    Parameters
    ----------
    Nx, Ny : int
        SLM dimensions in pixels.
    strength : float, optional
        Dimensionless amplitude of the Zernike term. The phase is
        2π * strength * Z_astig(r, θ).
    angle_deg : float, optional
        Astigmatism axis orientation in degrees. angle_deg = 0 gives
        a pattern aligned with x/y; rotating by 45° swaps between the
        "+2" and "-2" flavours.
    pupil_radius_pix : float, optional
        Radius of the (circular) pupil in pixels used for normalising
        r ∈ [0, 1]. If None, min(Nx, Ny)/2 is used.

    Returns
    -------
    phase : (Ny, Nx) float32
        Phase mask in radians implementing the Zernike astigmatism.
    """

    if pupil_radius_pix is None:
        pupil_radius_pix = min(Nx, Ny) / 2.0

    # Pixel coordinates centred at zero (in pixels)
    X_pix, Y_pix = np.meshgrid(
        np.arange(Nx, dtype=np.float32) - Nx / 2.0,
        np.arange(Ny, dtype=np.float32) - Ny / 2.0,
    )

    # Polar coordinates in the pupil
    r_pix = np.sqrt(X_pix**2 + Y_pix**2)
    r_norm = r_pix / float(pupil_radius_pix)  # 0..~1 inside pupil
    theta = np.arctan2(Y_pix, X_pix)

    # Rotate astigmatism axis
    alpha = np.deg2rad(angle_deg)

    # Zernike astigmatism term (up to normalisation): r^2 cos[2(θ - α)]
    Z = r_norm**2 * np.cos(2.0 * (theta - alpha))

    phase = 2.0 * np.pi * strength * Z
    phase_wrapped = np.mod(phase, 2.0 * np.pi)
    return phase_wrapped.astype(np.float32)
