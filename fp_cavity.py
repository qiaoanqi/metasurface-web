# fp_cavity.py - Fabry-Perot cavity spectrum computation
import numpy as np
from engine import MaterialLibrary




_AG_NK_TABLE = {
    380: (0.17, 1.62), 385: (0.17, 1.63), 390: (0.17, 1.64), 395: (0.17, 1.66),
    400: (0.17, 1.67), 405: (0.17, 1.68), 410: (0.17, 1.70), 415: (0.17, 1.72),
    420: (0.17, 1.73), 425: (0.17, 1.75), 430: (0.17, 1.77), 435: (0.17, 1.79),
    440: (0.17, 1.81), 445: (0.17, 1.83), 450: (0.17, 1.85), 455: (0.17, 1.87),
    460: (0.17, 1.90), 465: (0.17, 1.92), 470: (0.17, 1.95), 475: (0.17, 1.98),
    480: (0.17, 2.01), 485: (0.16, 2.04), 490: (0.16, 2.07), 495: (0.16, 2.10),
    500: (0.15, 2.13), 505: (0.15, 2.16), 510: (0.15, 2.20), 515: (0.15, 2.23),
    520: (0.14, 2.27), 525: (0.14, 2.30), 530: (0.14, 2.34), 535: (0.14, 2.38),
    540: (0.13, 2.42), 545: (0.13, 2.46), 550: (0.13, 2.50), 555: (0.13, 2.54),
    560: (0.12, 2.58), 565: (0.12, 2.63), 570: (0.12, 2.67), 575: (0.12, 2.72),
    580: (0.12, 2.77), 585: (0.12, 2.82), 590: (0.12, 2.87), 595: (0.12, 2.92),
    600: (0.12, 2.97), 605: (0.12, 3.02), 610: (0.12, 3.08), 615: (0.12, 3.13),
    620: (0.12, 3.19), 625: (0.12, 3.25), 630: (0.12, 3.31), 635: (0.12, 3.37),
    640: (0.12, 3.44), 645: (0.13, 3.50), 650: (0.13, 3.57), 655: (0.13, 3.64),
    660: (0.14, 3.71), 665: (0.14, 3.78), 670: (0.14, 3.86), 675: (0.14, 3.93),
    680: (0.14, 4.01), 685: (0.14, 4.09), 690: (0.15, 4.17), 695: (0.15, 4.26),
    700: (0.15, 4.34), 705: (0.15, 4.43), 710: (0.15, 4.52), 715: (0.15, 4.61),
    720: (0.15, 4.71), 725: (0.15, 4.80), 730: (0.15, 4.90), 735: (0.15, 5.00),
    740: (0.15, 5.10), 745: (0.15, 5.21), 750: (0.15, 5.32), 755: (0.15, 5.43),
    760: (0.15, 5.55), 765: (0.15, 5.67), 770: (0.15, 5.79), 775: (0.15, 5.91),
    780: (0.15, 6.04),
}

def _ag_nk(wl_nm):
    wls = list(_AG_NK_TABLE.keys())
    if wl_nm <= wls[0]: return _AG_NK_TABLE[wls[0]]
    if wl_nm >= wls[-1]: return _AG_NK_TABLE[wls[-1]]
    for i in range(len(wls)-1):
        if wls[i] <= wl_nm <= wls[i+1]:
            frac = (wl_nm - wls[i]) / (wls[i+1] - wls[i])
            n1, k1 = _AG_NK_TABLE[wls[i]]
            n2, k2 = _AG_NK_TABLE[wls[i+1]]
            return (n1 + frac*(n2-n1), k1 + frac*(k2-k1))
    return _AG_NK_TABLE[wls[-1]]



def _ag_nk_vec(wl_arr):
    """Vectorized Ag nk lookup for numpy arrays using np.interp."""
    _wls = np.array(sorted(_AG_NK_TABLE.keys()), dtype=float)
    _ns = np.array([_AG_NK_TABLE[int(w)][0] for w in _wls])
    _ks = np.array([_AG_NK_TABLE[int(w)][1] for w in _wls])
    return np.interp(wl_arr, _wls, _ns), np.interp(wl_arr, _wls, _ks)

def _n_sio2_sellmeier(wl_nm):
    wl_um = wl_nm / 1000.0
    return np.sqrt(1 + 0.6961663*wl_um**2/(wl_um**2 - 0.0684043**2)
                   + 0.4079426*wl_um**2/(wl_um**2 - 0.1162414**2)
                   + 0.8974794*wl_um**2/(wl_um**2 - 9.896161**2))

def fp_cavity_spectrum(T_nm, angle_deg=0.0, pol_TE=True):
    """Metal-mirror FP: Ag(30nm)/TiO2(T)/Ag(bulk). 减色型 (向量化版本)"""
    wls = np.arange(380, 785, 5).astype(float)
    d_top = 30.0; theta = angle_deg * np.pi / 180.0; n_inc = 1.0
    # Vectorized refractive indices
    ag_n, ag_k = _ag_nk_vec(wls)
    n_top = ag_n + 1j * ag_k
    n_bot = ag_n + 1j * ag_k
    wl_um = np.maximum(wls / 1000.0, 0.15)
    A_tio, B_tio = 2.3000, 0.03500
    n_tio2 = (A_tio + B_tio / wl_um**2).astype(complex)
    cos_inc = np.cos(theta); sin_inc = np.sin(theta)
    def _cos_v(n): return np.emath.sqrt(1.0 - (sin_inc * n_inc / n)**2)
    cos_top = _cos_v(n_top); cos_tio2 = _cos_v(n_tio2); cos_bot = _cos_v(n_bot)
    def _layer_v(n, d, c):
        delta = 2.0 * np.pi * n * d * c / wls
        p = n * c if pol_TE else c / n
        cd = np.cos(delta); sd = np.sin(delta)
        N = len(wls)
        M = np.zeros((N, 2, 2), dtype=complex)
        M[:, 0, 0] = cd; M[:, 0, 1] = 1j * sd / p
        M[:, 1, 0] = 1j * p * sd; M[:, 1, 1] = cd
        return M
    M = _layer_v(n_top, d_top, cos_top) @ _layer_v(n_tio2, T_nm, cos_tio2)
    p_inc = n_inc * cos_inc if pol_TE else cos_inc / n_inc
    p_bot = n_bot * cos_bot if pol_TE else cos_bot / n_bot
    a = M[:, 0, 0] + M[:, 0, 1] * p_bot
    b = M[:, 1, 0] + M[:, 1, 1] * p_bot
    r = (a * p_inc - b) / (a * p_inc + b)
    refl = np.abs(r)**2
    return wls, np.nan_to_num(np.clip(refl, 0, 1), nan=0.0, posinf=1.0, neginf=0.0)

def fp_dielectric_spectrum(T_nm, target_wl=450.0, n_pairs_top=3, n_pairs_bot=5, angle_deg=0.0, pol_TE=True):
    """DBR FP cavity: (TiO2/SiO2)^n/TiO2(T)/(SiO2/TiO2)^n. 高饱和度. (向量化版本)"""
    wls = np.arange(380, 785, 5).astype(float)
    N = len(wls)
    theta = angle_deg * np.pi / 180.0; n_inc = 1.0
    n_tio2_ref = MaterialLibrary.n_at_wavelength("TiO2 (anatase)", target_wl)
    n_sio2_ref = _n_sio2_sellmeier(target_wl)
    dH = target_wl / (4.0 * n_tio2_ref); dL = target_wl / (4.0 * n_sio2_ref)
    # Vectorized refractive indices across all wavelengths
    wl_um = np.maximum(wls / 1000.0, 0.15)
    nH = (2.3000 + 0.03500 / wl_um**2).astype(complex)
    nL = _n_sio2_sellmeier(wls).astype(complex)
    cos_inc = np.cos(theta); sin_inc = np.sin(theta)
    def _cos_v(n): return np.emath.sqrt(1.0 - (sin_inc * n_inc / n)**2)
    cH = _cos_v(nH); cL = _cos_v(nL)
    def _layer_v(n, d, c):
        delta = 2.0 * np.pi * n * d * c / wls
        p = n * c
        cd = np.cos(delta); sd = np.sin(delta)
        M = np.zeros((N, 2, 2), dtype=complex)
        M[:, 0, 0] = cd; M[:, 0, 1] = 1j * sd / p
        M[:, 1, 0] = 1j * p * sd; M[:, 1, 1] = cd
        return M
    M = np.broadcast_to(np.eye(2, dtype=complex), (N, 2, 2)).copy()
    for _ in range(n_pairs_top):
        M = M @ _layer_v(nH, dH, cH) @ _layer_v(nL, dL, cL)
    M = M @ _layer_v(nH, T_nm, cH)
    for _ in range(n_pairs_bot):
        M = M @ _layer_v(nL, dL, cL) @ _layer_v(nH, dH, cH)
    p_inc = n_inc * cos_inc
    p_sub = nL * cL
    a = M[:, 0, 0] + M[:, 0, 1] * p_sub
    b = M[:, 1, 0] + M[:, 1, 1] * p_sub
    r = (a * p_inc - b) / (a * p_inc + b)
    refl = np.abs(r)**2
    return wls, np.nan_to_num(np.clip(refl, 0, 1), nan=0.0, posinf=1.0, neginf=0.0)


