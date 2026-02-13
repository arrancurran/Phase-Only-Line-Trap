import numpy as np
import numpy.fft as fft
import matplotlib.pyplot as plt


def show_hologram_on_slm(phase_img,
                         offset_x=2560,
                         offset_y=0,
                         window_width=512,
                         window_height=512,
                         fullscreen=False):
    """Display a 2D 8-bit image on the SLM monitor in a 512x512 window.

    This is a copy of the helper used in ShapePhaseModulation.py, so you
    can run this script independently for optical diagnostics.
    """
    fig = plt.figure(figsize=(5.12, 5.12), dpi=100)
    ax = fig.add_axes([0, 0, 1, 1])
    ax.imshow(phase_img, cmap='gray', vmin=0, vmax=255, interpolation='nearest')
    ax.axis('off')

    manager = plt.get_current_fig_manager()

    try:
        # TkAgg backend (common on Windows)
        window = manager.window
        window.wm_geometry(f"{window_width}x{window_height}+{offset_x}+{offset_y}")
        if fullscreen:
            try:
                window.attributes("-fullscreen", True)
            except Exception:
                pass
        window.lift()
    except Exception:
        try:
            # Qt backend
            window = manager.window
            window.setGeometry(offset_x, offset_y, window_width, window_height)
            if fullscreen:
                window.showFullScreen()
            else:
                window.showNormal()
        except Exception:
            pass

    return fig


# ---------------- user parameters ----------------
N = 512                 # SLM pixels (square)

# Stripe in the pupil (SLM) domain
# Choose orientation and width of the bright stripe.
stripe_orientation = "vertical"   # "vertical" or "horizontal"
stripe_width_pixels = 400           # width of the bright stripe in pixels

# Simple amplitude grating to displace the line off-axis
# carrier_fx is in cycles per pixel; adjust to move the line further away
carrier_fx = 0.08

# Display parameters for SLM monitor
SHOW_ON_SLM = True
SLM_OFFSET_X = 2560      # adjust to your primary monitor width plus any in-monitor offset
SLM_OFFSET_Y = 0
SLM_WINDOW_WIDTH = 512
SLM_WINDOW_HEIGHT = 512

# ---------------- build simple stripe mask in pupil ----------------
A_pupil = np.zeros((N, N), dtype=np.float32)

if stripe_orientation == "vertical":
    # Bright vertical stripe through the pupil
    center = N // 2
    half_w = stripe_width_pixels // 2
    A_pupil[:, center - half_w:center + half_w] = 1.0
else:
    # Bright horizontal stripe through the pupil
    center = N // 2
    half_w = stripe_width_pixels // 2
    A_pupil[center - half_w:center + half_w, :] = 1.0

# ---------------- apply amplitude grating to displace the line ----------------
# We multiply the stripe mask by a 1D cosine grating so that the resulting
# focal pattern has diffracted orders displaced from the zeroth order.
cx = (np.arange(N, dtype=np.float32) - N / 2.0)[None, :]
grating = 0.5 * (1.0 + np.cos(2.0 * np.pi * carrier_fx * cx))

A_pupil_mod = A_pupil * grating.astype(np.float32)

# Convert amplitude mask (with grating) to 8-bit image for the SLM
phase_8 = np.uint8(np.rint(A_pupil_mod * 255.0))

# ---------------- display on SLM ----------------
if SHOW_ON_SLM:
    show_hologram_on_slm(
        phase_8,
        offset_x=SLM_OFFSET_X,
        offset_y=SLM_OFFSET_Y,
        window_width=SLM_WINDOW_WIDTH,
        window_height=SLM_WINDOW_HEIGHT,
        fullscreen=False,
    )

# ---------------- predict focal-plane intensity (Fraunhofer) ----------------
# We treat A_pupil_mod as an amplitude mask in the SLM/pupil plane.
U_pupil = A_pupil_mod
U_f = fft.fftshift(fft.fft2(fft.ifftshift(U_pupil)))
I_f = np.abs(U_f) ** 2
I_f /= I_f.max()

# Choose a central zoom region
half = 128
H, W = I_f.shape
cy, cx = H // 2, W // 2
Iz = I_f[cy - half:cy + half, cx - half:cx + half]

# 1D intensity profile across the line
if stripe_orientation == "vertical":
    # Expect a line elongated horizontally in the focal plane,
    # so plot a cross-section along x through the center.
    line_profile = Iz[Iz.shape[0] // 2, :]
else:
    # Horizontal stripe in pupil -> line elongated vertically in focal plane
    line_profile = Iz[:, Iz.shape[1] // 2]

line_profile /= line_profile.max()

# ---------------- quick looks ----------------
fig, axs = plt.subplots(1, 3, figsize=(10, 4), constrained_layout=True)
axs[0].imshow(A_pupil_mod, cmap='gray')
axs[0].set_title('Stripe + Grating in Pupil')
axs[0].axis('off')

axs[1].imshow(Iz, cmap='gray')
axs[1].set_title('Predicted Intensity at Focus')
axs[1].axis('off')

axs[2].plot(line_profile)
axs[2].set_title('Line Intensity Profile')
axs[2].set_xlabel('Position along expected line')
axs[2].set_ylabel('Normalised intensity')

plt.show()
