import numpy as np

def fresnel_phase(Nx, Ny, dz_um, f, wavelength, pixel_pitch):
    """Quadratic (Fresnel) phase for a defocus dz in a 4f system.

    Parameters
    ----------
    Nx, Ny : int
        SLM dimensions in pixels.
    dz_um : float
        Defocus distance from the nominal focal plane [µm].
    f : float
        Focal length of the objective [m].
    wavelength : float
        Wavelength of the light [m].
    pixel_pitch : float, optional
        SLM pixel pitch [m]. Defaults to 15e-6.

    Returns
    -------
    fresnel_phase : (Ny, Nx) float32
        Phase mask in radians implementing a defocus dz.
    """

    # Convert dz to metres
    dz = dz_um * 1e-6

    # Pixel indices centred at zero
    X_pix, Y_pix = np.meshgrid(
        np.arange(Nx, dtype=np.float32) - Nx / 2.0,
        np.arange(Ny, dtype=np.float32) - Ny / 2.0,
    )

    # Physical coordinates on SLM [m]
    X = X_pix * pixel_pitch
    Y = Y_pix * pixel_pitch

    # Small-defocus quadratic phase at the pupil plane:
    #   phi(x, y) = pi * dz / (lambda * f^2) * (x^2 + y^2)
    coeff = np.pi * dz / (wavelength * f**2)
    print(f"Fresnel phase coefficient: {coeff:.3e} rad/m^2")
    phase = coeff * (X**2 + Y**2)

    fresnel_phase = np.mod(phase, 2 * np.pi)
    return fresnel_phase.astype(np.float32)