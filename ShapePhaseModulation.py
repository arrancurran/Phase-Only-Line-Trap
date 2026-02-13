import numpy as np
import numpy.fft as fft
import matplotlib.pyplot as plt

def zoom_around_peak(I, half=90, thresh_frac=0.5):
    """Return a zoom window centered on the *center of the bright line*.

    Instead of centering on a single max pixel (which might lie near
    one end of the line), we estimate the center of the bright region
    using a simple centroid of pixels above a fraction of the peak.
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

# Generate hologram using shape-phase method for a uniform line tweezer
# (based on Roichman & Grier, Opt. Lett. 2006)

# ---------------- user parameters ----------------
N = 512                 # SLM pixels (square)
p = 15e-6               # pixel pitch [m]
wavelength = 1064e-9    # wavelength [m]
f = 3.33e-3             # objective focal length [m]
L = 10e-6               # line length in sample plane [m]
A0 = 1.0                # row fill fraction cap [0..1]
carrier_fx = 0.02        # optional off-axis cycles per pixel along x
randomize_S = False      # use random S with same pixels-per-row (Grier et al.)

# ---------------- coordinate axes ----------------
# SLM-plane coordinates (meters)
x = (np.arange(N) - N/2) * p
y = (np.arange(N) - N/2) * p

# ---------------- line-target mapping (Fraunhofer) ----------------
# chi(rho) ∝ sinc(π * rho_y * L / (f λ))
s_row = np.sinc((y * L) / (f * wavelength))
A_row = (A0 * np.abs(s_row)).astype(np.float32)
phi_row = np.where(s_row >= 0.0, np.pi, 0.0).astype(np.float32)

# ---------------- selection mask S(rho) ----------------
# Use normalized x in [-1,1]. Row j keeps columns where |x_norm| < A_row[j].
x_norm = (np.arange(N, dtype=np.float32) - (N - 1) / 2.0) / ((N - 1) / 2.0)
row_widths = np.clip(A_row, 0.0, 1.0)[:, None]

# Deterministic mask: central contiguous block in each row
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

# ---------------- base phase mask ----------------
phi_S = (phi_row[:, None] * S).astype(np.float32)

# ---------------- off-axis carrier ----------------
cx = (np.arange(N, dtype=np.float32) - N/2.0)[None, :]
carrier = 2.0 * np.pi * (carrier_fx * cx)
phi_S = phi_S + carrier

# ---------------- wrap and quantize to 8-bit phase ----------------
phi_wrapped = np.remainder(phi_S, 2*np.pi, out=np.empty_like(phi_S))
phase_8 = np.uint8(np.rint(phi_wrapped * (255.0 / (2*np.pi))))

# Save hologram as 8-bit, 512x512 image
plt.imsave('shape_phase_hologram_512.bmp', phase_8, cmap='gray')

# ---------------- predict focal-plane intensity (Fraunhofer) ----------------
X, Y = np.meshgrid(x, y)
w0 = 6e-3  # 1/e^2 radius at the SLM
A_gauss = np.exp(-(X**2 + Y**2) / w0**2)
U_pupil = A_gauss * S.astype(np.float32) * np.exp(1j * phi_wrapped)

# U_pupil = S.astype(np.float32) * np.exp(1j * phi_wrapped)
U_f = fft.fftshift(fft.fft2(fft.ifftshift(U_pupil)))
I_f = np.abs(U_f) ** 2
I_f /= I_f.max()

Iz, peak, box = zoom_around_peak(I_f, half=100)

# 1D intensity profile along the length of the line
line_profile = Iz[ :, Iz.shape[0] // 2]
line_profile /= line_profile.max()

# ---------------- quick looks ----------------

fig, axs = plt.subplots(1, 4, figsize=(12, 4), constrained_layout=True)
axs[0].imshow(phase_8, cmap='gray')
axs[0].set_title('Shape-Phase Hologram (8-bit)')
axs[0].axis('off')

axs[1].imshow(A_gauss, cmap='gray')
axs[1].set_title('Gaussian at SLM')
axs[1].axis('off')

axs[2].imshow(Iz, cmap='gray')
axs[2].set_title('Predicted Intensity at Focus')
axs[2].axis('off')

axs[3].plot(line_profile)
axs[3].set_title('Line Intensity Profile')
axs[3].set_xlabel('Position along line (pixels)')
axs[3].set_ylabel('Normalised intensity')

plt.show()
