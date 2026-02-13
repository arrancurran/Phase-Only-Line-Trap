import numpy as np
import matplotlib.pyplot as plt


def circ_pupil(N, radius_frac=0.48):
    x = np.arange(N) - (N // 2)
    y = np.arange(N) - (N // 2)
    X, Y = np.meshgrid(x, y, indexing="xy")
    r0 = radius_frac * (N / 2.0)
    return (X**2 + Y**2) <= r0**2


def make_line_eq8_eq9_with_carrier(
    N=768,
    L_pixels=170,          # controls sinc vs y in DOE plane (proxy for L=15 µm)
    A0=0.22,               # controls half-width of the assigned wedge (fraction of half-aperture)
    grating_cycles_x=35.0  # cycles across aperture -> shifts diffraction order in focal plane
):
    """
    Eq. (8):  φ_S(ρ) = φ(ρ_y) S(ρ)
    Eq. (9):  S(ρ)=1 if |ρ_x| < A(ρ_y) else 0
    with Eq. (6)-(7) for A(ρ_y) and φ(ρ_y).  [oai_citation:2‡Roichman_Grier_1.pdf](sediment://file_00000000f220724681274fd7616a4703)

    IMPORTANT FIX vs earlier: total phase = carrier + φ_S (mod 2π),
    so the grating is present everywhere like Fig. 1(b).
    """
    # Coordinates, centered
    x = np.arange(N) - (N // 2)
    y = np.arange(N) - (N // 2)
    X, Y = np.meshgrid(x, y, indexing="xy")

    # Eq. (4) analytic DOE field for a uniform line: sinc(k ρ_y); discrete surrogate sinc(y/L_pixels)
    s = np.sinc(y / float(L_pixels))  # real, changes sign

    # Eq. (6): A(ρ_y) = A0 |sinc|
    A_y = A0 * np.abs(s)  # dimensionless (0..A0)

    # Eq. (7): φ(ρ_y) = 0 or π depending on sign
    phi_y = np.where(s >= 0, 0.0, np.pi)

    # Eq. (9): S(ρ)=1 if |ρ_x| < A(ρ_y) * (N/2)
    half_width_px = A_y * (N / 2.0)
    S = (np.abs(X) < half_width_px[None, :]).T  # (N,N) boolean

    # Carrier: a blazed grating in x (vertical stripes like the paper)
    # phase_carrier = 2π * cycles * (x/N)
    phase_carrier = 2.0 * np.pi * grating_cycles_x * ((X + (N / 2.0)) / float(N))

    # Eq. (8) term (shape–phase): φ_S = φ(y) * S
    phi_S = (phi_y[None, :] * S).astype(float)

    # Total phase displayed on SLM
    phase_total = np.mod(phase_carrier + phi_S, 2.0 * np.pi)

    return phase_total, S, A_y, phi_y


def focal_intensity_fft(phase_slm, pupil=None):
    U = np.exp(1j * phase_slm)
    if pupil is not None:
        U = U * pupil
    Uf = np.fft.fftshift(np.fft.fft2(U))
    I = np.abs(Uf) ** 2
    I /= I.max() if I.max() > 0 else 1.0
    return I


def zoom_around_peak(I, half=90):
    iy, ix = np.unravel_index(np.argmax(I), I.shape)
    y0 = max(0, iy - half); y1 = min(I.shape[0], iy + half)
    x0 = max(0, ix - half); x1 = min(I.shape[1], ix + half)
    return I[y0:y1, x0:x1], (iy, ix), (y0, y1, x0, x1)


if __name__ == "__main__":
    N = 768

    phase, S, A_y, phi_y = make_line_eq8_eq9_with_carrier(
        N=N,
        L_pixels=9,
        A0=0.7,            # try 0.12–0.30
        grating_cycles_x=100 # try 20–60
    )

    I = focal_intensity_fft(phase, pupil=circ_pupil(N, 0.1))

    Iz, peak, box = zoom_around_peak(I, half=120)

    plt.figure(figsize=(12, 5))

    plt.subplot(1, 2, 1)
    plt.imshow(phase, origin="lower", cmap="gray", vmin=0, vmax=2*np.pi)
    plt.title("Phase-only hologram: carrier + Eq. (8) term (mod 2π)")
    plt.colorbar(fraction=0.046)

    plt.subplot(1, 2, 2)
    plt.imshow(I, origin="lower")