#!/usr/bin/env python3
"""
Verification script for:
  "Dark energy from horizon thermodynamics: a zero-parameter prediction for DESI"
  C. G. Smith Jr. (2026)

Reproduces all numerical results from public DESI DR2 BAO data.

Requirements:
  pip install numpy
  
Data: Download from https://github.com/CobayaSampler/bao_data
  - desi_bao_dr2/desi_gaussian_bao_ALL_GCcomb_mean.txt
  - desi_bao_dr2/desi_gaussian_bao_ALL_GCcomb_cov.txt
  - desi_2024_gaussian_bao_ALL_GCcomb_mean.txt  (DR1, for cross-check)
  - desi_2024_gaussian_bao_ALL_GCcomb_cov.txt

Usage:
  python verify_paper9.py --data-dir /path/to/bao_data-master/
"""

import math, numpy as np, argparse, os, sys

c_km = 299792.458  # speed of light [km/s]

# ============================================================
# COSMOLOGICAL FUNCTIONS
# ============================================================

def H_LCDM(z, H0, Om):
    """Hubble parameter in flat LCDM [km/s/Mpc]"""
    return H0 * math.sqrt(Om * (1+z)**3 + (1-Om))

def rd_fitting(obh2, och2):
    """Sound horizon fitting formula (Aubourg+ 2015) [Mpc]"""
    omh2 = obh2 + och2
    return 147.05 * (obh2/0.02236)**(-0.255) * (omh2/0.1415)**(-0.131)

# ============================================================
# THE PREDICTION: Eq. (5) of the paper
# w(z) = -1 + (1/2pi) * [1 - H0/H(z)]
# ============================================================

g0 = 1.0 / (2.0 * math.pi)  # THE coupling constant. Not fitted.

def rho_DE_ratio(z, H0, Om):
    """rho_DE(z) / rho_Lambda from integrating w(z)"""
    if z < 1e-8:
        return 1.0
    N = 200
    dz = z / N
    integral = 0.0
    for i in range(N):
        zi = (i + 0.5) * dz
        Hi = H_LCDM(zi, H0, Om)
        wi = -1.0 + g0 * (1.0 - H0/Hi)
        integral += 3.0 * (1.0 + wi) / (1.0 + zi) * dz
    return math.exp(integral)

def H_kernel(z, H0, Om):
    """Hubble parameter with kernel-modified dark energy"""
    OL = 1.0 - Om
    rr = rho_DE_ratio(z, H0, Om)
    H2 = H0**2 * (Om * (1+z)**3 + OL * rr)
    return math.sqrt(max(H2, 1e-10))

def DH_over_rd(z, H0, Om, rd, use_kernel=False):
    Hfunc = H_kernel if use_kernel else H_LCDM
    return c_km / (Hfunc(z, H0, Om) * rd)

def DM_over_rd(z, H0, Om, rd, use_kernel=False):
    Hfunc = H_kernel if use_kernel else H_LCDM
    N = 200
    dz = z / N
    s = sum(dz / Hfunc((i+0.5)*dz, H0, Om) for i in range(N))
    return c_km * s / rd

def DV_over_rd(z, H0, Om, rd, use_kernel=False):
    dm = DM_over_rd(z, H0, Om, rd, use_kernel)
    dh = DH_over_rd(z, H0, Om, rd, use_kernel)
    return (z * dm**2 * dh) ** (1.0/3.0)

def predict(z, qty, H0, Om, rd, use_kernel=False):
    if qty == 'DH_over_rs': return DH_over_rd(z, H0, Om, rd, use_kernel)
    elif qty == 'DM_over_rs': return DM_over_rd(z, H0, Om, rd, use_kernel)
    else: return DV_over_rd(z, H0, Om, rd, use_kernel)

# ============================================================
# COMPRESSED CMB LIKELIHOOD
# ============================================================

H0rd_planck = 67.36 * 147.09  # = 9906 km/s
sigma_H0rd = 23.0
sigma_och2 = 0.0012

def chi2_total(H0, och2, use_kernel, data_vec, Cinv, dr2_list):
    obh2 = 0.02237
    h = H0 / 100.0
    Om = (obh2 + och2) / h**2
    rd = rd_fitting(obh2, och2)
    
    # CMB contribution
    chi2_cmb = ((H0*rd - H0rd_planck) / sigma_H0rd)**2
    chi2_cmb += ((och2 - 0.1200) / sigma_och2)**2
    
    # BAO contribution
    pv = np.array([predict(d[0], d[2], H0, Om, rd, use_kernel) for d in dr2_list])
    dd = data_vec - pv
    chi2_bao = float(dd @ Cinv @ dd)
    
    return chi2_cmb + chi2_bao, chi2_cmb, chi2_bao

# ============================================================
# DATA LOADING
# ============================================================

def load_data(filepath_mean, filepath_cov):
    data = []
    with open(filepath_mean) as f:
        for line in f:
            if line.startswith('#') or line.strip() == '':
                continue
            parts = line.split()
            if len(parts) == 3:
                data.append((float(parts[0]), float(parts[1]), parts[2]))
    
    cov = []
    with open(filepath_cov) as f:
        for line in f:
            cov.append([float(x) for x in line.split()])
    
    return data, np.array([d[1] for d in data]), np.array(cov)

# ============================================================
# GRID SCAN OPTIMIZER
# ============================================================

def optimize(use_kernel, data_vec, Cinv, dr2_list):
    """Two-pass grid scan: coarse then fine"""
    best = 1e10
    bp = None
    best_cc, best_cb = 0, 0
    # Coarse
    for H0 in np.arange(66.5, 69.5, 0.2):
        for och2 in np.arange(0.117, 0.124, 0.001):
            c2, cc, cb = chi2_total(H0, och2, use_kernel, data_vec, Cinv, dr2_list)
            if c2 < best:
                best = c2
                bp = (H0, och2)
                best_cc, best_cb = cc, cb
    # Fine
    H0c, oc = bp
    for H0 in np.arange(H0c-0.3, H0c+0.3, 0.04):
        for och2 in np.arange(oc-0.001, oc+0.001, 0.0002):
            c2, cc, cb = chi2_total(H0, och2, use_kernel, data_vec, Cinv, dr2_list)
            if c2 < best:
                best = c2
                bp = (H0, och2)
                best_cc, best_cb = cc, cb
    return bp[0], bp[1], best, best_cc, best_cb

# ============================================================
# MAIN
# ============================================================

def main():
    parser = argparse.ArgumentParser(description='Verify Paper 9 results')
    parser.add_argument('--data-dir', required=True,
                        help='Path to bao_data-master/ directory')
    args = parser.parse_args()
    
    d = args.data_dir
    
    # Load DR2
    dr2_mean = os.path.join(d, 'desi_bao_dr2', 'desi_gaussian_bao_ALL_GCcomb_mean.txt')
    dr2_cov = os.path.join(d, 'desi_bao_dr2', 'desi_gaussian_bao_ALL_GCcomb_cov.txt')
    
    if not os.path.exists(dr2_mean):
        print(f"ERROR: Cannot find {dr2_mean}")
        sys.exit(1)
    
    dr2_list, dv2, C2 = load_data(dr2_mean, dr2_cov)
    Ci2 = np.linalg.inv(C2)
    er2 = np.sqrt(np.diag(C2))
    
    print("=" * 65)
    print("VERIFICATION: Paper 9 numerical results")
    print("=" * 65)
    print(f"\nDESI DR2 data: {len(dr2_list)} data points loaded")
    print(f"Covariance matrix: {C2.shape}")
    print(f"g_0 = 1/(2pi) = {g0:.5f} (NOT fitted)")
    
    # --- LCDM ---
    print("\nOptimizing LCDM...")
    H0_l, och2_l, chi2_l, cc_l, cb_l = optimize(False, dv2, Ci2, dr2_list)
    h = H0_l/100; Om_l = (0.02237+och2_l)/h**2
    print(f"  H0 = {H0_l:.2f}, omega_cdm = {och2_l:.4f}, Omega_m = {Om_l:.4f}")
    print(f"  chi2 = {chi2_l:.2f}  (CMB: {cc_l:.2f}, BAO: {cb_l:.2f})")
    
    # --- Kernel ---
    print("\nOptimizing kernel prediction [w = -1 + (1/2pi)(1 - H0/H)]...")
    H0_k, och2_k, chi2_k, cc_k, cb_k = optimize(True, dv2, Ci2, dr2_list)
    h = H0_k/100; Om_k = (0.02237+och2_k)/h**2
    print(f"  H0 = {H0_k:.2f}, omega_cdm = {och2_k:.4f}, Omega_m = {Om_k:.4f}")
    print(f"  chi2 = {chi2_k:.2f}  (CMB: {cc_k:.2f}, BAO: {cb_k:.2f})")
    
    dchi = chi2_l - chi2_k
    sig = math.sqrt(abs(dchi)) if dchi > 0 else 0
    print(f"\n  Delta chi2 = {dchi:+.2f}")
    print(f"  Significance = {sig:.1f} sigma (0 extra parameters)")
    
    # --- Pulls ---
    print(f"\n{'Tracer':<8} {'z':>6} {'Qty':<14} {'LCDM pull':>10} {'Kernel pull':>12} {'Improved':>9}")
    print("-" * 65)
    rd_l = rd_fitting(0.02237, och2_l)
    rd_k = rd_fitting(0.02237, och2_k)
    n_improved = 0
    for i, d_pt in enumerate(dr2_list):
        z, val, qty = d_pt
        pred_l = predict(z, qty, H0_l, Om_l, rd_l, False)
        pred_k = predict(z, qty, H0_k, Om_k, rd_k, True)
        pl = (val - pred_l) / er2[i]
        pk = (val - pred_k) / er2[i]
        imp = "yes" if abs(pk) < abs(pl) else ""
        if imp: n_improved += 1
        tracer = ['BGS','LRG1','LRG1','LRG2','LRG2','LRG3','LRG3',
                   'ELG','ELG','QSO','QSO','Lya','Lya'][i]
        print(f"{tracer:<8} {z:>6.3f} {qty:<14} {pl:>+10.2f} {pk:>+12.2f} {imp:>9}")
    print(f"\nImproved {n_improved}/13 data points")
    
    # --- DR1 cross-check ---
    dr1_mean = os.path.join(d, 'desi_2024_gaussian_bao_ALL_GCcomb_mean.txt')
    dr1_cov = os.path.join(d, 'desi_2024_gaussian_bao_ALL_GCcomb_cov.txt')
    
    if os.path.exists(dr1_mean):
        print("\n" + "=" * 65)
        print("DR1 CROSS-CHECK")
        print("=" * 65)
        dr1_list, dv1, C1 = load_data(dr1_mean, dr1_cov)
        Ci1 = np.linalg.inv(C1)
        
        _, _, chi2_l1, _, _ = optimize(False, dv1, Ci1, dr1_list)
        _, _, chi2_k1, _, _ = optimize(True, dv1, Ci1, dr1_list)
        dchi1 = chi2_l1 - chi2_k1
        sig1 = math.sqrt(abs(dchi1)) if dchi1 > 0 else 0
        
        print(f"  DR1: Delta chi2 = {dchi1:+.2f} ({sig1:.1f} sigma)")
        print(f"  DR2: Delta chi2 = {dchi:+.2f} ({sig:.1f} sigma)")
        grew = "YES" if dchi > dchi1 else "NO"
        print(f"  Signal grew from DR1 to DR2: {grew}")
    
    # --- w(z) table ---
    print("\n" + "=" * 65)
    print("w(z) PREDICTION vs CPL")
    print("=" * 65)
    print(f"{'z':>5} {'w_kernel':>10} {'w_CPL':>10}")
    for z in [0, 0.3, 0.5, 0.7, 1.0, 1.5, 2.0, 3.0]:
        Hz = H_LCDM(z, H0_k, Om_k) if z > 0 else H0_k
        w_k = -1 + g0*(1 - H0_k/Hz) if z > 0 else -1.0
        w_cpl = -0.42 + (-1.75)*z/(1+z)
        print(f"{z:>5.1f} {w_k:>+10.4f} {w_cpl:>+10.4f}")
    
    print(f"\nAsymptotic: w_kernel -> {-1+g0:+.4f},  w_CPL -> {-0.42-1.75:+.4f}")
    print("\n" + "=" * 65)
    print("VERIFICATION COMPLETE")
    print("=" * 65)

if __name__ == '__main__':
    main()
