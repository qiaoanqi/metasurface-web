# rcwa_batch.py v4 -- multi-material RCWA batch generator (Pool parallel)
# Usage: python rcwa_batch.py --samples 5000 --output data/rcwa_asi_5k.pkl --material a-Si --substrate SiO2

# MUST be before numpy import (BLAS thread control for multiprocessing)
import os
os.environ['OPENBLAS_NUM_THREADS'] = '1'
os.environ['OMP_NUM_THREADS'] = '1'
os.environ['MKL_NUM_THREADS'] = '1'

import numpy as np
import pickle, argparse, time, os, sys
from multiprocessing import Pool, cpu_count, TimeoutError

from grcwa import set_backend
set_backend('numpy')
from grcwa.rcwa import (obj, MakeKPMatrix, SolveLayerEigensystem_uniform,
                         SolveLayerEigensystem, SolveExterior, GetZPoyntingFlux)
from grcwa.fft_funs import Epsilon_fft
from grcwa import backend as bd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from color_utils import spectrum_to_xyz, xyz_to_srgb

# === Cauchy refractive index model (aligned with engine.py MaterialLibrary.CAUCHY) ===
CAUCHY = {
    "TiO2":  (2.3000, 0.03500, 0.0),
    "SiO2":  (1.4580, 0.00354, 0.0),
    "Si3N4": (1.9900, 0.01200, 0.0),
    "a-Si":  (3.8000, 0.08000, 0.0),
    "Al2O3": (1.7546, 0.00500, 0.0),
    # --- New materials for Δn criterion validation (2026-07-30) ---
    "GaN":   (2.2700, 0.03800, 0.0),   # wurtzite ordinary ray, Barker&Ilegems 1973; n(550)≈2.40, Δn_A=0.81
    "HfO2":  (1.9700, 0.01000, 0.0),   # Khoshman et al., Surf. Coat. Technol. 202, 2500 (2008); n(550)≈2.00, Δn_A=0.51
    "Ta2O5": (2.0600, 0.01500, 0.0),   # Palik, Handbook of Optical Constants (1998); n(550)≈2.11, Δn_A=0.60
}
CAUCHY_TiO2 = CAUCHY["TiO2"]
CAUCHY_SiO2 = CAUCHY["SiO2"]
# a-Si complex refractive index n + ik — Pierce & Spicer 1972, Phys. Rev. B 5, 3017
# (60-nm evaporated film; values as reproduced in Palik, Handbook of Optical Constants of Solids).
# Table LOADED PROGRAMMATICALLY from data/externals/aSi_PierceSpicer1972.csv (refractiveindex.info,
# retrieved 2026-08-06; the only a-Si entry on that site) so no transcription error is possible
# (audit 2026-08-06: a hand-typed copy lost precision, k=0.0812 -> 0.081).
# NOTE: the previous citation "Green & Keevers 1995" is crystalline-Si data and the old table
# matched NO public dataset (see aSi_optical_constants_audit.md). n and k are now same-source.
import os as _os
import csv as _csv
_A_SI_PS_CSV = _os.path.join(_os.path.dirname(_os.path.abspath(__file__)),
                             'data', 'externals', 'aSi_PierceSpicer1972.csv')
if not _os.path.exists(_A_SI_PS_CSV):
    raise FileNotFoundError('a-Si 光学常数源文件缺失 (Pierce & Spicer 1972): %s' % _A_SI_PS_CSV)
_A_SI_PS_TABLE = {}
with open(_A_SI_PS_CSV, encoding='utf-8') as _f:
    for _row in _csv.DictReader(_f):
        _A_SI_PS_TABLE[float(_row['wavelength_nm'])] = (float(_row['n']), float(_row['k']))

def _aSi_ps_nk(wl_nm):
    """Linear-interpolated (n, k) from Pierce & Spicer 1972.

    np.interp on the tabulated points (310-2066 nm bracket the RCWA band 380-780 nm,
    so no extrapolation is ever needed; outside the table np.interp clamps to endpoints).
    Linear interpolation avoids the step discontinuities of nearest-neighbor lookup
    (audit 2026-08-06: smooth k, e.g. k(400)=2.21 between the 387.5/413.3 anchors).
    """
    wl_arr = np.array(sorted(_A_SI_PS_TABLE.keys()))
    n_arr = np.array([_A_SI_PS_TABLE[w][0] for w in wl_arr])
    k_arr = np.array([_A_SI_PS_TABLE[w][1] for w in wl_arr])
    w = np.asarray(wl_nm, dtype=float)
    n_i = np.interp(w, wl_arr, n_arr)
    k_i = np.interp(w, wl_arr, k_arr)
    if w.ndim == 0:
        return float(n_i), float(k_i)
    return n_i, k_i

def _get_aSi_n(wl_nm):
    return _aSi_ps_nk(wl_nm)[0]

def _get_aSi_k(wl_nm):
    return _aSi_ps_nk(wl_nm)[1]

LOSSY_MATERIALS = {"a-Si", "a-Si (amorphous)"}


def n_cauchy(wl_nm, material_name):
    """Real refractive index (Cauchy model)."""
    A, B, C = CAUCHY.get(material_name, CAUCHY["SiO2"])
    wl_um = np.maximum(np.asarray(wl_nm, dtype=float) / 1000.0, 0.15)
    return A + B / (wl_um ** 2) + C / (wl_um ** 4)

def n_complex(wl_nm, material_name):
    """Complex refractive index n+ik for lossy materials.

    a-Si: n and k BOTH from the Pierce & Spicer 1972 table (same source),
    replacing the previous Cauchy-n + tabulated-k hybrid (audited 2026-08-06).
    k=0 对照模式: 设置环境变量 A_SI_K0=1 时虚部置零 (fig5 无吸收对照线数据源,
    2026-08-07 级联前置).
    """
    if material_name in LOSSY_MATERIALS:
        k0_mode = _os.environ.get('A_SI_K0', '') == '1'
        wl_arr = np.asarray(wl_nm, dtype=float)
        if wl_arr.ndim == 0:
            nn, kk = _aSi_ps_nk(float(wl_arr))
            return nn + 0j if k0_mode else nn + 1j * kk
        if k0_mode:
            return np.array([_aSi_ps_nk(float(w))[0] + 0j for w in wl_arr])
        return np.array([_aSi_ps_nk(float(w))[0] + 1j * _aSi_ps_nk(float(w))[1] for w in wl_arr])
    return n_cauchy(wl_nm, material_name)

def n_TiO2(wl_nm):
    return n_cauchy(wl_nm, "TiO2")

def n_SiO2(wl_nm):
    return n_cauchy(wl_nm, "SiO2")

# === Single-wavelength RCWA ===
def _rcwa_single_wl(args):
    D_um, H_um, P_um, W_um, wl_um, n_pillar_val, n_sub_val, nG_req, Nxy, angle_deg, pol = args
    freq = 1.0 / wl_um
    kx = np.sin(np.deg2rad(angle_deg)) / wl_um
    blk = obj(nG_req, [P_um, 0], [0, P_um], freq, kx, 0, verbose=0)
    blk.Add_LayerUniform(0, 1.0)
    blk.Add_LayerGrid(H_um, Nxy, Nxy)
    blk.Add_LayerUniform(0, n_sub_val**2)
    blk.Init_Setup(Gmethod=1)
    x = np.linspace(-P_um/2, P_um/2, Nxy)
    y = np.linspace(-P_um/2, P_um/2, Nxy)
    X, Y = np.meshgrid(x, y)
    if W_um is None or abs(W_um - D_um) < 1e-12:
        mask = (X**2 + Y**2) <= (D_um/2)**2          # circular (L=W degenerate path, bit-identical to legacy)
    else:
        mask = ((X**2)/(D_um/2)**2 + (Y**2)/(W_um/2)**2) <= 1.0   # elliptical (semi-axes D_um/2, W_um/2)
    eps2_pillar = n_pillar_val**2
    eps2_sub = n_sub_val**2
    use_complex = isinstance(eps2_pillar, complex) or (hasattr(eps2_pillar, 'dtype') and np.iscomplexobj(eps2_pillar))
    eps_grid = np.where(mask, eps2_pillar, eps2_sub)
    eps_grid = eps_grid.astype(np.complex128 if use_complex else np.float64)
    dN = 1.0 / (Nxy * Nxy)
    epsinv_nG, eps2 = Epsilon_fft(dN, eps_grid, blk.G)
    blk.Patterned_epinv_list[0] = epsinv_nG
    blk.Patterned_ep2_list[0] = eps2
    kp0 = MakeKPMatrix(blk.omega, 0, 1.0/blk.Uniform_ep_list[0], blk.kx, blk.ky)
    for i in range(blk.Layer_N):
        if blk.id_list[i][0] == 0:
            ep = blk.Uniform_ep_list[blk.id_list[i][2]]
            kp = MakeKPMatrix(blk.omega, 0, 1.0/ep, blk.kx, blk.ky)
            blk.kp_list[i] = kp
            q, phi = SolveLayerEigensystem_uniform(blk.omega, blk.kx, blk.ky, ep)
            blk.q_list[i] = q
            blk.phi_list[i] = phi
        else:
            q, phi = SolveLayerEigensystem(blk.omega, blk.kx, blk.ky, kp0, eps2)
            blk.q_list[i] = q
            blk.phi_list[i] = phi
            kp = MakeKPMatrix(blk.omega, 1, epsinv_nG, blk.kx, blk.ky)
            blk.kp_list[i] = kp
    try:
        # pol: 'p' = pure p (legacy default, (1,0,0,0)); 's' = pure s (0,0,1,0).
        # grcwa MakeExcitationPlanewave(p_amp,p_phase,s_amp,s_phase); phi=0 => p excites
        # the second half-block, s the first. TE/TM physical mapping established by the
        # three invariants (gate 1b), not assumed.
        if pol == 's':
            blk.MakeExcitationPlanewave(0, 0, 1, 0)
        else:
            blk.MakeExcitationPlanewave(1, 0, 0, 0)
        a0, bN = blk.a0, blk.bN
        aN, b0 = SolveExterior(a0, bN, blk.q_list, blk.phi_list, blk.kp_list, blk.thickness_list)
    except Exception:
        raise RuntimeError('Singular matrix')
    zero_vec = bd.zeros(2*blk.nG, dtype=complex)
    inc_for, _ = GetZPoyntingFlux(a0, zero_vec, blk.omega, blk.kp_list[0], blk.phi_list[0], blk.q_list[0])
    _, ref_back = GetZPoyntingFlux(zero_vec, b0, blk.omega, blk.kp_list[0], blk.phi_list[0], blk.q_list[0])
    tr_for, _ = GetZPoyntingFlux(aN, zero_vec, blk.omega, blk.kp_list[2], blk.phi_list[2], blk.q_list[2])
    R = float((-ref_back / inc_for).real)
    T_val = float((tr_for / inc_for).real)
    return max(0.0, min(1.0, float(R.real))), max(0.0, min(1.0, float(T_val.real)))

def rcwa_spectrum(D_nm, H_nm, P_nm, wl_nm_list, nG_req=101, Nxy=256, n_jobs=1,
                   material="TiO2", substrate="SiO2", angle_deg=0.0, W_nm=None, pol='p'):
    """Compute full RCWA spectrum for one parameter set.

    W_nm=None: circular pillar of diameter D_nm (legacy path, unchanged).
    W_nm given: elliptical pillar, semi-axis D_nm/2 along x, W_nm/2 along y;
    W_nm == D_nm degrades to the circular path bit-identically.
    pol: 'p' (default, legacy (1,0,0,0)) or 's' (pure s). TE/TM labels are assigned
    after the gate-1b invariants; see second_paper_elliptical_plan.md.
    """
    D_um, H_um, P_um = D_nm/1000, H_nm/1000, P_nm/1000
    W_um = None if W_nm is None else W_nm/1000
    n_pillar = n_complex(wl_nm_list, material)
    n_sub = n_cauchy(wl_nm_list, substrate)
    args_list = [(D_um, H_um, P_um, W_um, wl_nm_list[i]/1000, n_pillar[i], n_sub[i], nG_req, Nxy, angle_deg, pol)
                 for i in range(len(wl_nm_list))]
    results = [_rcwa_single_wl(a) for a in args_list]
    R = np.array([r[0] for r in results])
    T_val = np.array([r[1] for r in results])
    return R, T_val

# === Parameter sampling ===
def generate_params(n_samples, seed):
    rng = np.random.RandomState(seed)
    D_all = rng.uniform(80, 350, n_samples)
    H_all = rng.uniform(100, 600, n_samples)
    P_all = rng.uniform(200, 600, n_samples)
    valid = []
    for D, H, P in zip(D_all, H_all, P_all):
        if D >= P:
            continue
        fr = np.pi*(D/2)**2/(P**2)
        if fr < 0.03 or fr > 0.70:
            continue
        valid.append((int(D), int(H), int(P)))
    while len(valid) < n_samples:
        D = rng.uniform(80, 350)
        H = rng.uniform(100, 600)
        P = rng.uniform(200, 600)
        if D >= P: continue
        fr = np.pi*(D/2)**2/(P**2)
        if fr < 0.03 or fr > 0.70: continue
        valid.append((int(D), int(H), int(P)))
    return valid[:n_samples]

def convergence_check(D, H, P, wls, material, substrate):
    for nG in [65, 101, 151]:
        t0 = time.time()
        R, T_val = rcwa_spectrum(D, H, P, wls, nG_req=nG, material=material, substrate=substrate, angle_deg=0.0)
        dt = time.time() - t0
        print(f"  nG={nG:3d}: R+T={np.mean(R+T_val):.4f}, time={dt:.1f}s")


def generate_params_elliptical(n_samples, seed, r_max=3.0):
    """Uniform random + rejection sampling over (L, W, H, P) — same protocol family as
    generate_params (paper 1: uniform, NOT Latin Hypercube; see second_paper_elliptical_plan.md).

    Constraints (gate 3, hard assertions):
      L, W ~ U[80, 350] nm; H ~ U[100, 600] nm; P ~ U[200, 600] nm
      max(L, W) < P
      fill fraction f = pi*(L/2)*(W/2)/P^2 in [0.03, 0.70]
      aspect ratio r = max(L,W)/min(L,W) in [1, r_max]  (L<W swapped so r>=1)
    """
    rng = np.random.RandomState(seed)
    out = []
    guard = 0
    while len(out) < n_samples and guard < 50 * n_samples:
        guard += 1
        L = rng.uniform(80, 350)
        W = rng.uniform(80, 350)
        H = rng.uniform(100, 600)
        P = rng.uniform(200, 600)
        if max(L, W) >= P:
            continue
        f = np.pi * (L / 2) * (W / 2) / (P ** 2)
        if not (0.03 <= f <= 0.70):
            continue
        r = max(L, W) / min(L, W)
        if not (1.0 <= r <= r_max):
            continue
        out.append((float(L), float(W), float(H), float(P)))
    if len(out) < n_samples:
        raise RuntimeError('elliptical sampler exhausted after %d draws (got %d/%d)'
                           % (guard, len(out), n_samples))
    return out


# === Pool worker: compute one sample (module-level for pickling) ===
def _process_sample(task):
    """Compute RCWA spectrum for one (D,H,P) and return result dict or None."""
    D, H, P, wls, nG, material, substrate, angle_deg = task
    for retry_nG in [nG, nG + 50, nG + 100]:
        try:
            R, T_val = rcwa_spectrum(D, H, P, wls, retry_nG,
                                     material=material, substrate=substrate, angle_deg=angle_deg)
            if R is not None:
                xyz = spectrum_to_xyz(wls, R)
                rgb = xyz_to_srgb(xyz)
                r_plus_t = float(np.mean(R + T_val))
                return {
                    'D': D, 'H': H, 'P': P,
                    'material': material, 'substrate': substrate, 'angle_deg': angle_deg,
                    'wl_nm': wls.copy(), 'R': R, 'T': T_val,
                    'xyz': xyz, 'rgb': rgb, 'R_plus_T_mean': r_plus_t,
                    'success': True, 'retry_nG': retry_nG,
                }
        except Exception:
            continue
    return {'D': D, 'H': H, 'P': P, 'success': False}

def main():
    parser = argparse.ArgumentParser(description='RCWA batch spectrum generator v4')
    parser.add_argument('--samples', type=int, default=100, help='number of samples')
    parser.add_argument('--output', type=str, default='data/rcwa_out.pkl', help='output file')
    parser.add_argument('--material', type=str, default='TiO2',
                        choices=['TiO2', 'a-Si', 'Si3N4', 'Al2O3', 'GaN', 'HfO2', 'Ta2O5'],
                        help='pillar material')
    parser.add_argument('--substrate', type=str, default='SiO2',
                        choices=['SiO2', 'Si3N4', 'Al2O3'], help='substrate material')
    parser.add_argument('--nG', type=int, default=65, help='Fourier truncation order')
    parser.add_argument('--n-jobs', type=int, default=1, help='parallel cores')
    parser.add_argument('--seed', type=int, default=42, help='random seed')
    parser.add_argument('--check-convergence', action='store_true', help='run convergence check first')
    parser.add_argument('--angle', type=float, default=0.0, help='incident angle in degrees')
    parser.add_argument('--resume', action='store_true', help='resume from existing output file')
    parser.add_argument('--timeout', type=int, default=120, help='per-sample timeout in seconds')
    args = parser.parse_args()

    wls = np.linspace(380, 780, 81)
    params = generate_params(args.samples, args.seed)

    # Resume: load existing results and skip already-computed (D,H,P)
    existing_params = set()
    if args.resume and os.path.exists(args.output):
        with open(args.output, 'rb') as f:
            old_data = __import__('pickle').load(f)
        results = old_data  # start from existing results
        n_ok = len([r for r in old_data if r.get('success')])
        n_fail = len(old_data) - n_ok
        for r in old_data:
            if r.get('success'):
                existing_params.add((r['D'], r['H'], r['P']))
        print(f'Resume: {len(existing_params)} already computed, {n_ok} OK + {n_fail} FAIL')
    params = [p for p in params if (p[0], p[1], p[2]) not in existing_params]
    if not params:
        print('All samples already computed!')
        import sys; sys.exit(0)

    if args.check_convergence and len(params) > 0:
        D, H, P = params[0]
        convergence_check(D, H, P, wls, args.material, args.substrate)

    print(f'RCWA v3: {args.samples} samples x 81 wavelengths | {args.material}/{args.substrate}')
    print(f'nG={args.nG}, jobs={args.n_jobs}, D=80-350nm')
    est = args.samples * 81 * 0.05 / max(args.n_jobs, 1)
    print(f'Est. time: {est:.0f}s ({est/60:.1f}min)')
    print('='*60)

    results = []
    t_total = time.time()
    n_total = len(params)
    n_done = 0
    n_ok = 0
    n_fail = 0

    # Build task list for Pool
    tasks = [(D, H, P, wls, args.nG, args.material, args.substrate, args.angle)
             for (D, H, P) in params]

    with Pool(processes=args.n_jobs) as pool:
        async_results = [pool.apply_async(_process_sample, (task,)) for task in tasks]
        for i, async_r in enumerate(async_results):
            try:
                result = async_r.get(timeout=args.timeout)
            except TimeoutError:
                task = tasks[i]
                result = {'D': task[0], 'H': task[1], 'P': task[2], 'success': False}
                n_timeout = getattr(args, '_n_timeout', 0) + 1
                args._n_timeout = n_timeout
                print(f'[TIMEOUT] D={task[0]} H={task[1]} P={task[2]} after {args.timeout}s')

            n_done += 1
            if result['success']:
                n_ok += 1
                results.append(result)
                print(f'[{n_done}/{n_total}] D={result["D"]:3d} H={result["H"]:3d} P={result["P"]:3d} '
                      f'OK R_peak={np.max(result["R"]):.3f} R+T={result["R_plus_T_mean"]:.3f} '
                      f'(nG={result.get("retry_nG",args.nG)})')
            else:
                n_fail += 1
                print(f'[{n_done}/{n_total}] D={result["D"]:3d} H={result["H"]:3d} P={result["P"]:3d} FAIL')

            if n_done % 50 == 0:
                dt = time.time() - t_total
                rate = n_done / dt if dt > 0 else 0
                eta = (n_total - n_done) / rate if rate > 0 else 0
                print(f'  -- Progress: {n_done}/{n_total} ({100*n_done//n_total}%), '
                      f'{rate:.1f} samples/s, ETA {eta:.0f}s, OK={n_ok} FAIL={n_fail} --')
                os.makedirs(os.path.dirname(args.output) or '.', exist_ok=True)
                with open(args.output, 'wb') as f:
                    pickle.dump(results, f)

    os.makedirs(os.path.dirname(args.output) or '.', exist_ok=True)
    with open(args.output, 'wb') as f:
        pickle.dump(results, f)
    print(f'Done! {len(results)}/{args.samples} success, {time.time()-t_total:.0f}s total')
    print(f'Output: {args.output} | {args.material}/{args.substrate}')

if __name__ == '__main__':
    main()
