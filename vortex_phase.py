import numpy as np


def vortex_phase(Nx, Ny, topological_charge, cx=None, cy=None):
    """Generate a phase map for an optical vortex (LG-like) beam.

    Parameters
    ----------
    Nx, Ny : int
        Dimensions of the SLM / phase mask in pixels.
    topological_charge : int or float
        The vortex topological charge (azimuthal index `l`). Phase winds
        by `2π * topological_charge` around the centre.
    cx, cy : float, optional
        Vortex centre in pixel coordinates. If omitted, the centre of
        the array (Nx/2, Ny/2) is used.

    Returns
    -------
    phase : (Ny, Nx) float32
        Phase mask in radians implementing the azimuthal vortex.
    """

    # Default vortex centre: array centre
    if cx is None:
        cx = Nx / 2.0
    if cy is None:
        cy = Ny / 2.0

    # Pixel coordinates relative to the vortex centre
    X, Y = np.meshgrid(
        np.arange(Nx, dtype=np.float32) - cx,
        np.arange(Ny, dtype=np.float32) - cy,
    )

    # Azimuthal angle around the vortex centre
    phi = np.arctan2(Y, X)  # range (-pi, pi]

    # Optical vortex phase: exp(i * l * phi)
    phase = topological_charge * phi

    # Wrap to [0, 2π)
    phase_wrapped = np.mod(phase, 2 * np.pi)
    
    return phase_wrapped.astype(np.float32)

def vortex_phase_anisotropic(Nx, Ny, l, ax=1.0, ay=1.0, cx=None, cy=None):
    if cx is None:
        cx = Nx / 2.0
    if cy is None:
        cy = Ny / 2.0

    X, Y = np.meshgrid(
        np.arange(Nx, dtype=np.float32) - cx,
        np.arange(Ny, dtype=np.float32) - cy,
    )

    # Stretch coordinates differently along x and y
    Xs = X / ax
    Ys = Y / ay

    phi = np.arctan2(Ys, Xs)
    phase = np.mod(l * phi, 2 * np.pi).astype(np.float32)
    
    return phase