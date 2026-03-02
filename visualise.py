import numpy as np
import numpy.fft as fft

def visualise_focal_plane(hologram, Nx, Ny, p, obj_mask):
    # ---------------- predict focal-plane intensity ----------------
    # SLM-plane pixel coordinates from centre in meters
    x = (np.arange(Nx) - Nx/2) * p
    y = (np.arange(Ny) - Ny/2) * p
    X, Y = np.meshgrid(x, y)
    
    w0x = 5e-3  # along x
    w0y = 5e-3  # along y (tighter → more oval)
    A_gauss = np.exp(-(X**2 / w0x**2 + Y**2 / w0y**2))

    
    # Quantise array in [0, 2π] to 255 discrete values.
    phi = np.clip(hologram, 0.0, 2*np.pi)

    # Quantise
    indices = np.round(phi / (2*np.pi) * 254).astype(np.uint8)

    # Map back to phase values
    phi_q = indices.astype(np.float32) * (2*np.pi / 254)

    # w0 = 5e-3  # 1/e^2 radius of the Gaussian at the SLM
    # A_gauss = np.exp(-(X**2 + Y**2) / w0**2)
    # A_gauss = A_gauss / A_gauss.max() * 2 * np.pi  # normalize to 2pi for better visualization of phase
    # Gaussian illumination overfills the SLM; S only chooses the phase pattern.
    U_pupil = A_gauss * obj_mask * np.exp(1j * phi_q)
    
    U = fft.fftshift( fft.fft2( fft.ifftshift( U_pupil ) ) )
    I = np.abs(U) ** 2
    I = I / I.max()
    
    return I

def slm_to_img_scaling(dx_um, dy_um, Nx, Ny, scaling_factor):
    """Convert SLM grating offsets (µm) to pixel coordinates in the focal plane image."""
    sx, sy = Nx * scaling_factor, Ny * scaling_factor  # pixels per metre in the focal plane
    dx, dy = dx_um * 1e-6, dy_um * 1e-6
    # Pixels from centre in the focal plane
    dx_pix, dy_pix = dx * sx, dy * sy
    ix, iy = Nx/2 + dx_pix, Ny/2 + dy_pix
    ix, iy = int(np.rint(ix)), int(np.rint(iy))
    
    return ix, iy

def plot_line_intensity(I, Nx, Ny, angle, L, ix, iy, scaling_factor):
    # Clip to valid pixel range
    ix = max(0, min(Nx - 1, ix))
    iy = max(0, min(Ny - 1, iy))
    # Sum over a band of pixels around ix_int to get a single 1D profile
    sum = 3
    
    if angle == 0:
        x0, x1 = max(0, ix - sum), min(Nx, ix + sum + 1)
        profile = I[:, x0:x1].sum(axis=1)
        coords_um = (np.arange(Ny) - Ny/2.0) * 1e6 / (Ny * scaling_factor)
    else:
        y0, y1 = max(0, iy - sum), min(Ny, iy + sum + 1)
        profile = I[y0:y1, :].sum(axis=0)
        coords_um = (np.arange(Nx) - Nx/2.0) * 1e6 / (Nx * scaling_factor)
        
    profile -= profile.min() 
    profile /= profile.max()

    return coords_um, profile
