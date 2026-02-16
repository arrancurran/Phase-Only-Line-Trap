import numpy as np
import numpy.fft as fft
import matplotlib
# matplotlib.use("TkAgg")
import matplotlib.pyplot as plt

blaze = np.array([0.000000, 0.057705, 0.114756, 0.170501, 0.224285, 0.275457, 0.323361, 0.367346, 0.406758, 0.440943, 0.469248, 0.491020, 0.505781, 0.514422, 0.518472, 0.519466, 0.518936, 0.518416, 0.519441, 0.523543, 0.532256, 0.547115, 0.569651, 0.600860, 0.640003, 0.685993, 0.737743, 0.794168, 0.854179, 0.916692, 0.980618, 1.000000])

blaze_256 = np.interp(np.arange(256), np.linspace(0, 255, len(blaze)), blaze)


def show_hologram_on_slm(hologram, offset_x=2560, offset_y=0, slm_width=512, slm_height=512):
    """Display the hologram on a second monitor used as the SLM.
    hologram : 2D array 8-bit phase hologram to display.
    offset_x, offset_y : int
        Top-left corner (in desktop pixels) where the SLM monitor begins.
    """
    # Create a figure with no axes
    fig = plt.figure(figsize=(5.12, 5.12), dpi=100)
    ax = fig.add_axes([0, 0, 1, 1])
    # Use nearest-neighbor interpolation so each hologram pixel maps to SLM pixels.
    ax.imshow(hologram, cmap='gray', interpolation='nearest')
    ax.axis('off')
    manager = plt.get_current_fig_manager()
    # Try to handle both TkAgg and Qt backends for window positioning
    try:
        # TkAgg backend (common on Windows)
        window = manager.window
        # Remove title bar and window decorations
        try:
            window.overrideredirect(True)
        except Exception:
            pass
        # Hide the toolbar if present
        try:
            if hasattr(fig.canvas, "toolbar") and fig.canvas.toolbar is not None:
                fig.canvas.toolbar.pack_forget()
        except Exception:
            pass
        # Set explicit window size and position: WIDTHxHEIGHT+X+Y
        window.wm_geometry(f"{slm_width}x{slm_height}+{offset_x}+{offset_y}")
        window.lift()
    except Exception:
        pass  # Not TkAgg or failed to set geometry

    return fig

def zoom_around_peak(I, half=90, thresh_frac=0.5):
    """Return a zoom window centered on the brightest pixel in the image.
    """
    I = np.asarray(I)
    max_val = I.max()
    # Use pixels above a fraction of the maximum to define the line region
    mask = I >= (thresh_frac * max_val)
    if mask.any():
        ys, xs = np.nonzero(mask)
        iy = int(np.round(ys.mean()))
        ix = int(np.round(xs.mean()))
    else:
        # Fallback: just use the absolute maximum
        iy, ix = np.unravel_index(np.argmax(I), I.shape)

    y0 = max(0, iy - half); y1 = min(I.shape[0], iy + half)
    x0 = max(0, ix - half); x1 = min(I.shape[1], ix + half)
    return I[y0:y1, x0:x1], (iy, ix), (y0, y1, x0, x1)

def grating_phase(Nx, Ny, carrier_fx):
    """Calculate the grating phase pattern for a given carrier frequency and x coordinates."""
    x = (np.arange(Nx, dtype=np.float32) - Nx/2.0)[None, :]
    grating = 2.0 * np.pi * (carrier_fx * x)
    grating = np.remainder(grating, 2*np.pi)
    grating = np.broadcast_to(grating, (Ny, Nx)).astype(np.float32)
    return grating

# Generate hologram using shape-phase method for a uniform line tweezer
# (based on Roichman & Grier, Opt. Lett. 2006)

# ---------------- user parameters ----------------
Nx, Ny = 512, 512       # SLM pixels (square)
p = 15e-6               # pixel pitch [m]
wavelength = 1064e-9    # wavelength [m]
f = 3.0e-3              # objective focal length [m]
L = 20e-6               # line length in sample plane [m]
A0 = 1                  # row fill fraction cap [0..1]
carrier_fx = 0.2        # optional off-axis cycles per pixel along x
randomize_S = False     # use random S with same pixels-per-row

# Display parameters for SLM monitor
SHOW_ON_SLM = False
SLM_OFFSET_X = 0
SLM_OFFSET_Y = 0

# SLM-plane pixel coordinates from centre in meters
x = (np.arange(Nx) - Nx/2) * p
y = (np.arange(Ny) - Ny/2) * p

k = np.pi / L
# psi(rho) ∝ sinc(π * ρ_y * L / (f λ)) Equation 4 in Roichman & Grier, Opt. Lett. 2006
psi_rho = np.sinc((y * L) / (f * wavelength))
# Equation 6
A_rho = (A0 * np.abs(psi_rho)).astype(np.float32)

# φ(ρ) = { π  if sinc(k ρ_y) >= 0
#        { 0  otherwise
# 
# Equation 7
phi_rho = np.where(psi_rho >= 0.0, np.pi, 0.0).astype(np.float32)

# S_rho = np.where( np.abs(x) < A_rho, 1.0, 0.0).astype(np.float32)

# ---------------- selection mask S(ρ) ----------------
# Use normalized x in [-1,1]. Row j keeps columns where |x_norm| < A_rho[j].
x_norm = (np.arange(Nx, dtype=np.float32) - (Nx - 1) / 2.0) / ((Nx - 1) / 2.0)
# Equation 9
row_widths = np.clip(A_rho, 0.0, 1.0)[:, None]

S = (np.abs(x_norm)[None, :] < row_widths).astype(np.uint8)

if randomize_S:
    # Replace S with a random distribution that keeps the same
    # number of "on" pixels in each row (Grier et al.).
    rng = np.random.default_rng()
    S_rand = np.zeros_like(S)

    for j in range(S.shape[0]):
        n_on = int(S[j].sum())
        if n_on == 0:
            continue
        idx_on = rng.choice(S.shape[1], size=n_on, replace=False)
        S_rand[j, idx_on] = 1

    S = S_rand

# Phase for the line trap, applied only where S = 1 (0 : 2pi)
holo_line = (phi_rho[:, None] * S).astype(np.float32) * 2

# Grating for unassigned pixels, where S = 0

grating_spot = grating_phase(Nx, Ny, carrier_fx)

grating_line = grating_phase(Nx, Ny, 0.01)

# holo_line = np.where(S == 1, holo_line, grating_line).astype(np.float32)
# holo_line = np.where(S == 1, grating_line, holo_line).astype(np.float32)

# Combine phases: where S=1 use line-trap phase, elsewhere use grating
holo_total = np.where(S == 0, grating_spot, holo_line).astype(np.float32)

# ---------------- wrap and quantize to 8-bit phase ----------------
phi_wrapped = np.remainder(holo_total, 2*np.pi, out=np.empty_like(holo_total))
phase_8 = np.uint8(np.rint(holo_total * (255.0 / (2*np.pi))))

# Save hologram as 8-bit, 512x512 image
# plt.imsave('shape_phase_hologram_512.bmp', phase_8, cmap='gray')

# ---------------- display hologram on SLM monitor ----------------
if SHOW_ON_SLM:
    # This opens a borderless/fullscreen window on the SLM
    show_hologram_on_slm(
        phase_8,
        offset_x=SLM_OFFSET_X,
        offset_y=SLM_OFFSET_Y,
        slm_width=Nx,
        slm_height=Ny,
    )

# ---------------- predict focal-plane intensity (Fraunhofer) ----------------
X, Y = np.meshgrid(x, y)
w0 = 6e-3  # 1/e^2 radius at the SLM
A_gauss = np.exp(-(X**2 + Y**2) / w0**2)
A_gauss = A_gauss / A_gauss.max()  # normalize to 2pi for better visualization of phase
# Gaussian illumination overfills the SLM; S only chooses the phase pattern.
U_pupil = A_gauss * np.exp(1j * holo_total)
U_f = fft.fftshift(fft.fft2(fft.ifftshift(U_pupil)))
I_f = np.abs(U_f) ** 2
I_f /= I_f.max()

Iz, peak, box = zoom_around_peak(I_f, half=100)

# 1D intensity profile along the length of the line
line_profile = Iz[ :, Iz.shape[0] // 2]
line_profile /= line_profile.max()

# ---------------- quick looks (separate analysis figure) ----------------

fig, axs = plt.subplots(3, 2, figsize=(12, 16), constrained_layout=True)

# Line-trap phase where S = 1
im_holo_line = axs[0, 0].imshow(holo_line, cmap='gray')
fig.colorbar(im_holo_line, ax=axs[0, 0], fraction=0.046, pad=0.04, label='Phase (radians)')
axs[0, 0].set_title('Line-trap phase (holo_line)')
axs[0, 0].axis('off')
axs[0, 0].set_aspect('equal')

# Grating phase for diverted Gaussian (phi_spot)
im_phi_spot = axs[0, 1].imshow(grating_spot, cmap='gray')
fig.colorbar(im_phi_spot, ax=axs[0, 1], fraction=0.046, pad=0.04, label='Phase (radians)')
axs[0, 1].set_title('Grating phase (grating_spot)')
axs[0, 1].axis('off')

# Total wrapped phase shown as 8-bit hologram
hologram = axs[1, 0].imshow(holo_total, cmap='gray')
fig.colorbar(hologram, ax=axs[1, 0], fraction=0.046, pad=0.04, label='Phase (radians)')
axs[1, 0].set_title('Combined phase (holo_total)')
axs[1, 0].axis('off')

# Phase of the Gaussian beam after the SLM
phase_slm = np.angle(U_pupil)
im_phi_slm = axs[1, 1].imshow(phase_slm, cmap='gray')
fig.colorbar(im_phi_slm, ax=axs[1, 1], fraction=0.046, pad=0.04, label='Phase (radians)')
axs[1, 1].set_title('Phase after SLM')
axs[1, 1].axis('off')

# Focal-plane intensity
im_I = axs[2, 0].imshow(np.log(I_f), cmap='gray')
fig.colorbar(im_I, ax=axs[2, 0], fraction=0.046, pad=0.04, label='log(Intensity)')
axs[2, 0].set_title('Predicted Intensity at Focus')
# axs[2, 0].axis('off')

# Line intensity profile
axs[2, 1].plot(line_profile)
axs[2, 1].set_title('Line Intensity Profile')
axs[2, 1].set_xlabel('Position along line (pixels)')
axs[2, 1].set_ylabel('Normalised intensity')

plt.show()
