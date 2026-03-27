import numpy as np
def line_phase(Nx, Ny, L_um, scaling_factor, angle, A0, randomize_S):
    length = L_um * 1e-6  # µm -> m
    theta = np.deg2rad(angle)
    
    # SLM pixel coordinates (centered, in pixels)
    X_pix, Y_pix = np.meshgrid(
        np.arange(Nx, dtype=np.float32) - Nx/2.0,
        np.arange(Ny, dtype=np.float32) - Ny/2.0,
    )
    
    # Rotate coordinates: u = along line, v = across line
    u =  X_pix*np.cos(theta) + Y_pix*np.sin(theta)
    v = -X_pix*np.sin(theta) + Y_pix*np.cos(theta)
    # Across-line coordinate in "metres" scaled for the sinc
    v_scaled = v * scaling_factor
    
    # Equation 4 in Roichman & Grier, Opt. Lett. 2006
    psi_rho = np.sinc(v_scaled * length)
    
    # Equation 6: A(ρ) = A0 * |ψ(ρ)|, where A0 is a user-defined cap on the row fill fraction.
    A_rho = (A0 * np.abs(psi_rho)).astype(np.float32)
    
    # Equation 7 in Roichman & Grier, Opt. Lett. 2006
    # 
    # φ(ρ) = { π  if sinc(k ρ_y) >= 0
    #        { 0  otherwise
    phi_rho = np.where(psi_rho >= 0.0, np.pi, 0.0).astype(np.float32)

    # ---------------- selection mask S(ρ) ----------------
    # Normalized along-line coordinate in [-1, 1]
    u_norm = u / np.max(np.abs(u))
    
    # Selection mask: keep points close to u = 0, with width set by A_rho
    S = (np.abs(u_norm) < A_rho).astype(np.uint8)

    # Replace S with a random distribution that keeps the same number of "on" pixels in each row.
    if randomize_S: S = randomize_mask(S, angle)
        
    # Phase for the line trap, applied only where S = 1 ( % 2pi)
    line_phase = np.mod(phi_rho * S, 2*np.pi).astype(np.float32)

    return line_phase, S

def randomize_mask(S, angle):
    """Randomize a binary mask S along the line direction.

    For any angle, we preserve the number of "on" pixels in each
    slice orthogonal to the line (i.e. constant v in the rotated
    coordinates used in `line_phase`), but shuffle their positions
    along the line (u).
    """
    S = np.asarray(S)
    ny, nx = S.shape

    # RNG for shuffling. Using a generator without a fixed seed so
    # each call can produce a different randomisation pattern.
    rng = np.random.default_rng(0)
    S_rand = np.zeros_like(S, dtype=np.uint8)

    # Fast paths for purely horizontal/vertical lines: preserve the
    # original row/column behaviour for these special cases.
    if angle % 180 == 0:  # horizontal (0 or 180 deg): per row
        for j in range(ny):
            n_on = int(S[j, :].sum())
            if n_on:
                cols = rng.choice(nx, size=n_on, replace=False)
                S_rand[j, cols] = 1
        return S_rand

    if angle % 180 == 90:  # vertical (90 or 270 deg): per column
        for i in range(nx):
            n_on = int(S[:, i].sum())
            if n_on:
                rows = rng.choice(ny, size=n_on, replace=False)
                S_rand[rows, i] = 1
        return S_rand

    # General case: arbitrary angle. Work in the same rotated
    # coordinate system (u, v) used in `line_phase`.
    theta = np.deg2rad(angle)

    X_pix, Y_pix = np.meshgrid(
        np.arange(nx, dtype=np.float32) - nx / 2.0,
        np.arange(ny, dtype=np.float32) - ny / 2.0,
    )

    # Rotate coordinates: u = along line, v = across line
    u = X_pix * np.cos(theta) + Y_pix * np.sin(theta)
    v = -X_pix * np.sin(theta) + Y_pix * np.cos(theta)

    # Group pixels by (approximate) v so that each group is a "row"
    # orthogonal to the line. We use integer bins of v for grouping.
    v_bins = np.rint(v).astype(np.int32)

    S_flat = S.ravel()
    S_rand_flat = S_rand.ravel()
    v_bins_flat = v_bins.ravel()

    for vb in np.unique(v_bins_flat):
        mask_flat = v_bins_flat == vb
        idx = np.nonzero(mask_flat)[0]
        if idx.size == 0:
            continue

        n_on = int(S_flat[idx].sum())
        if n_on == 0:
            continue

        chosen = rng.choice(idx, size=n_on, replace=False)
        S_rand_flat[chosen] = 1

    return S_rand
