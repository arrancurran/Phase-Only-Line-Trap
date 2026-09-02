import numpy as np
import numpy.fft as fft
import matplotlib.pyplot as plt

# from show_hologram_on_slm import show_hologram_on_slm
from fresnel_phase import fresnel_phase
from grating_phase import grating_phase
from vortex_phase import vortex_phase, vortex_phase_anisotropic
from astig_phase import phase_zernike


from visualise import visualise_focal_plane, slm_to_img_scaling, plot_line_intensity

plt.rcParams["font.family"] = "Times New Roman"   # or "Arial", "DejaVu Serif", etc.
plt.rcParams["mathtext.fontset"] = "stix"         # makes Greek/math text match better

# ---------------- SLM parameters ----------------
offset_x, offset_y = 0, 0
Nx, Ny = 1024, 1024       # SLM pixels
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
vortex_charge = 20      # topological charge of the vortex
aspect_ratio = 1.0     # aspect ratio of the vortex (ax/ay) to create an elliptical vortex

# ---------------- Plots parameters ----------------

phase_vortex = vortex_phase_anisotropic(Nx, Ny, vortex_charge, ax=aspect_ratio, ay=1.0)

phase_astig = phase_zernike(Nx, Ny, c_astig_vertical=0.0, c_astig_oblique=-0.0, pupil_radius_pix=None)

holo = np.mod(phase_vortex + phase_astig, 2*np.pi)

I = visualise_focal_plane(holo, Nx, Ny, p, obj_mask)

size = 256

cy, cx = Ny // 2, Nx // 2
half = size // 2

I_crop = I[cy-half:cy+half, cx-half:cx+half]

# Save phase plot separately.
fig_phase, ax_phase = plt.subplots(figsize=(6, 6))
im_phase = ax_phase.imshow(holo, cmap="gray", vmin=0.0, vmax=2.0 * np.pi)
ax_phase.axis("off")
cbar = fig_phase.colorbar(im_phase, ax=ax_phase, fraction=0.046, pad=0.04)
cbar.set_label(rf"$\ell \theta$", rotation=270, labelpad=0, fontsize=24)
cbar.set_ticks([0.0, 2.0 * np.pi])
cbar.set_ticklabels(["0", r"$2\pi$"])
cbar.ax.tick_params(labelsize=20)  # Set the font size of the colorbar ticks
fig_phase.savefig("oval_vortex_phase.png", dpi=600, bbox_inches="tight", pad_inches=0)

# Save focal intensity crop separately.
# Save focal intensity crop separately.
fig_intensity, ax_intensity = plt.subplots(figsize=(6, 6))
im_intensity = ax_intensity.imshow(I_crop, cmap="turbo")
ax_intensity.axis("off")

cbar_I = fig_intensity.colorbar(im_intensity, ax=ax_intensity, fraction=0.046, pad=0.04)
cbar_I.set_label(r"$\left|\mathcal{F}\,\left(Ae^{i\ell\theta}\right)\right|^2$", rotation=270, labelpad=18, fontsize=24)
cbar_I.set_ticks([0.0, 1.0])
cbar_I.set_ticklabels(["0", "1"])
cbar_I.ax.tick_params(labelsize=20)  # Set the font size of the colorbar ticks

fig_intensity.savefig("oval_vortex_intensity.png", dpi=600, bbox_inches="tight", pad_inches=0)
plt.show()
