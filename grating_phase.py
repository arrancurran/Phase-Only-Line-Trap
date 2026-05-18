import numpy as np

def grating_phase(Nx, Ny, dx_um, dy_um, scaling_factor):
    dx, dy = dx_um * 1e-6, dy_um * 1e-6  # µm -> m
    fx, fy = dx * scaling_factor, dy * scaling_factor  # cycles per pixel
    X, Y = np.meshgrid(np.arange(Nx, dtype=np.float32) - Nx/2.0, np.arange(Ny, dtype=np.float32) - Ny/2.0)

    # Return a phase map in radians (not the complex field).
    phase = np.mod(2 * np.pi * (fx * X + fy * Y), 2 * np.pi)

    return phase.astype(np.float32)
