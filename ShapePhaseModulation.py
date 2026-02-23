import numpy as np
import numpy.fft as fft
import matplotlib.pyplot as plt

# from show_hologram_on_slm import show_hologram_on_slm
from fresnel_phase import fresnel_phase
from grating_phase import grating_phase
from line_phase import line_phase

def visualise_focal_plane(hologram, Nx, Ny, p, obj_mask):
    # ---------------- predict focal-plane intensity ----------------
    # SLM-plane pixel coordinates from centre in meters
    x = (np.arange(Nx) - Nx/2) * p
    y = (np.arange(Ny) - Ny/2) * p
    X, Y = np.meshgrid(x, y)
    w0 = 5e-3  # 1/e^2 radius of the Gaussian at the SLM
    A_gauss = np.exp(-(X**2 + Y**2) / w0**2)
    A_gauss = A_gauss / A_gauss.max() * 2 * np.pi  # normalize to 2pi for better visualization of phase
    # Gaussian illumination overfills the SLM; S only chooses the phase pattern.
    U_pupil = A_gauss * obj_mask * np.exp(1j * hologram)
    
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

def plot_line_intensity(I, Nx, Ny, angle, L, ix, iy):
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

# ---------------- SLM parameters ----------------
offset_x, offset_y = 0, 0
Nx, Ny = 512, 512       # SLM pixels
p = 15e-6               # pixel pitch [m]
wavelength = 1064e-9    # wavelength [m]
f = 3.0e-3              # objective focal length [m]
M = 0.8                 # magnification from SLM to objective back aperture
obj_radius = 4.2e-3     # radius of the objective aperture in m
obj_at_slm = obj_radius * M # radius of the objective aperture at the SLM in m
obj_at_slm_px = obj_at_slm / p  # radius of the objective aperture at the SLM in pixels

Xp, Yp = np.meshgrid(np.arange(Nx) - Nx/2.0, np.arange(Ny) - Ny/2.0)
r_pix = np.sqrt(Xp**2 + Yp**2)

obj_mask = (r_pix <= obj_at_slm_px / 2).astype(np.float32)

scaling_factor = p / (f * wavelength)

# ---------------- Line parameters ----------------
randomise = True        # use random S with same pixels-per-row
L = 40                  # line length in sample plane [µm]
A0 = 1                  # row fill fraction cap [0..1]
angle = 0              # line angle in degrees (0 = horizontal, 90 = vertical)
line_offset_x = -20     # offset in x direction [µm]
line_offset_y = 0     # offset in y direction [µm]
dz = 5

# ---------------- Spot parameters ----------------
spot_offset_x = 20      # grating offset for unassigned pixels [µm]
spot_offset_y = 0      # grating offset for unassigned pixels [µm]       

# ---------------- Plots parameters ----------------
# display_on_slm = False
visualise = True

holo_fresnel = fresnel_phase(Nx, Ny, dz, f, wavelength, p)

# Calculate the line phase pattern and selection mask S
holo_line, S = line_phase(Nx, Ny, L, scaling_factor, angle, A0, randomise)

grating_line = grating_phase(Nx, Ny, line_offset_x, line_offset_y, scaling_factor)
# holo_line = np.where(S == 1, holo_line, grating_line).astype(np.float32)
holo_line = np.where(S == 1, np.mod(holo_line + grating_line + holo_fresnel, 2*np.pi).astype(np.float32), 0).astype(np.float32)

# Grating for unassigned pixels, where S = 0
grating_spot = grating_phase(Nx, Ny, spot_offset_x, spot_offset_y, scaling_factor)

# Combine phases: where S=1 use holo_line, elsewhere use grating_spot
holo_total = np.where(S == 0, grating_spot, holo_line).astype(np.float32)

# ---------------- display hologram on SLM monitor ----------------
# if display_on_slm:
#     show_hologram_on_slm(holo_total, offset_x=offset_x, offset_y=offset_y, slm_width=Nx, slm_height=Ny)

if visualise:
    fig = plt.figure(figsize=(12, 12), constrained_layout=True)
    gs = fig.add_gridspec(nrows=2, ncols=6)
    
    # Top 3 plots
    ax1 = fig.add_subplot(gs[0, 0:2])
    ax2 = fig.add_subplot(gs[0, 2:4])
    ax3 = fig.add_subplot(gs[0, 4:6])

    # Bottom plot
    ax4 = fig.add_subplot(gs[1, 0:3])
    ax5 = fig.add_subplot(gs[1, 3:6])

    im1 = ax1.imshow(holo_line, cmap="gray")
    ax1.set_title("Sinc phase")
    ax1.axis("off")

    im2 = ax2.imshow(grating_spot, cmap="gray")
    ax2.set_title("Grating phase for unassigned pixels")
    ax2.axis("off")

    im3 = ax3.imshow(holo_total, cmap="gray")
    ax3.set_title("Combined phase")
    ax3.axis("off")

    I = visualise_focal_plane(holo_total, Nx, Ny, p, obj_mask)
    
    im4 = ax4.imshow(I, cmap="gray", vmax=0.001)
    ax4.set_title("Predicted Intensity at Focus")
    ax4.axis("off")

    ix, iy = slm_to_img_scaling(line_offset_x,line_offset_y,Nx,Ny,scaling_factor)
    
    x_coords_um, profile = plot_line_intensity(I, Nx, Ny, angle, L, ix, iy)
    
    ax5.semilogy(x_coords_um, profile, 'o-', markersize=2)
    ax5.set_xlim(x_coords_um.min(), x_coords_um.max())
    ax5.set_ylim(1e-3, 1)
    ax5.set_xlabel("Offset (µm)")
    ax5.set_ylabel("Intensity (a.u.)")
    ax5.set_title("Line profile")
    ax5.grid(True, alpha=0.3)

    plt.show()

if visualise or display_on_slm:
    plt.show()