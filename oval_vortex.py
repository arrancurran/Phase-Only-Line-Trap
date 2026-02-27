import numpy as np
import numpy.fft as fft
import matplotlib.pyplot as plt

# from show_hologram_on_slm import show_hologram_on_slm
from fresnel_phase import fresnel_phase
from grating_phase import grating_phase
from vortex_phase import vortex_phase, vortex_phase_anisotropic


from visualise import visualise_focal_plane, slm_to_img_scaling, plot_line_intensity

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


# ---------------- Spot parameters ----------------      
vortex_charge = 60      # topological charge of the vortex
aspect_ratio = 1.2     # aspect ratio of the vortex (ax/ay) to create an elliptical vortex

# ---------------- Plots parameters ----------------

holo_vortex = vortex_phase_anisotropic(Nx, Ny, vortex_charge, ax=aspect_ratio, ay=1.0)

fig = plt.figure(figsize=(12, 12), constrained_layout=True)
gs = fig.add_gridspec(nrows=1, ncols=2)

# Top 3 plots
ax1 = fig.add_subplot(gs[0, 0])
ax2 = fig.add_subplot(gs[0, 1])

im1 = ax1.imshow(holo_vortex, cmap="gray")
ax1.set_title("Vortex phase")
ax1.axis("off")

I = visualise_focal_plane(holo_vortex, Nx, Ny, p, obj_mask)

im2 = ax2.imshow(I, cmap="gray")
ax2.set_title("Predicted Intensity at Focus")
ax2.axis("off")

plt.show()
