#!/usr/bin/env python3
"""RCWA batch generator using the 'rcwa' package - fast FFT-based RCWA solver."""

# Suppress rcwa progress bar (huge I/O overhead)
import progressbar as _pb
class _DummyBar:
    def start(self): return self
    def update(self, n): pass
    def finish(self): pass
_pb.ProgressBar = lambda *a, **kw: _DummyBar()

from rcwa import Source, Layer, LayerStack, Crystal, Solver
from rcwa.shorthand import complexArray
import numpy as np
import pickle, argparse, time, sys, os
from multiprocessing import Pool

CAUCHY = {
    "TiO2":  (2.30, 0.035), "a-Si":  (3.80, 0.100),
    "Si3N4": (1.99, 0.010), "Al2O3": (1.75, 0.008),
    "SiO2":  (1.45, 0.005),
}

def n_at_wl(name, wl_um):
    A, B = CAUCHY[name]
    return A + B / (wl_um**2)

_WL = np.linspace(380e-9, 780e-9, 81)
_SUB_NAME = None
_MAT_NAME = None

def _init_worker(args):
    global _SUB_NAME, _MAT_NAME
    _SUB_NAME, _MAT_NAME = args

def _process_sample(args_tuple):
    seed, mat, sub = args_tuple
    rng = np.random.RandomState(seed)
    D = rng.uniform(80, 320)
    H = rng.uniform(150, 550)
    P = rng.uniform(max(200, D * 1.2), 500)
    Nxy = 65
    x = np.linspace(-P/2, P/2, Nxy)
    y = np.linspace(-P/2, P/2, Nxy)
    X, Y = np.meshgrid(x, y)
    mask = (np.sqrt(X**2 + Y**2) <= D/2).astype(float)
    R_arr, T_arr = [], []
    try:
        for wl_i in _WL:
            wl_um = wl_i * 1e6
            n_mat = n_at_wl(mat, wl_um)
            n_sub = n_at_wl(sub, wl_um)
            eps_grid = 1.0 + (n_mat**2 - 1.0) * mask
            t1 = complexArray([P, 0, 0])
            t2 = complexArray([0, P, 0])
            deviceCrystal = Crystal(t1, t2, er=eps_grid, ur=np.ones_like(eps_grid))
            incidentLayer = Layer(er=1.0, ur=1.0)
            transmissionLayer = Layer(er=n_sub**2, ur=1.0)
            patternLayer = Layer(crystal=deviceCrystal, thickness=H)
            pTEM = 1/np.sqrt(2)*complexArray([1, 1j])
            source = Source(wavelength=wl_i, theta=0, phi=0, pTEM=pTEM, layer=incidentLayer)
            stack = LayerStack(patternLayer, incident_layer=incidentLayer, transmission_layer=transmissionLayer)
            solver = Solver(stack, source, (3, 3))
            res = solver.solve()
            R_arr.append(res['RTot'])
            T_arr.append(res['TTot'])
        R = np.array(R_arr)
        T = np.array(T_arr)
        rt_mean = float(np.mean(R + T))
        from color_utils_ import spectrum_to_xyz, xyz_to_srgb
        xyz = spectrum_to_xyz(np.arange(380, 785, 5).astype(float), R)
        rgb = xyz_to_srgb(xyz).tolist()
        return {
            'D': round(D, 1), 'H': round(H, 1), 'P': round(P, 1),
            'wl_nm': np.linspace(380, 780, 81).tolist(),
            'R': R.tolist(), 'T': T.tolist(),
            'xyz': xyz.tolist(), 'rgb': rgb,
            'R_plus_T_mean': rt_mean,
            'material': mat, 'substrate': sub,
        }
    except Exception as e:
        return None

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--samples', type=int, default=5000)
    parser.add_argument('--output', type=str, default='data/rcwa_fast.pkl')
    parser.add_argument('--material', type=str, default='TiO2')
    parser.add_argument('--substrate', type=str, default='SiO2')
    parser.add_argument('--n-jobs', type=int, default=4)
    parser.add_argument('--seed', type=int, default=42)
    args = parser.parse_args()
    
    print(f"RCWA Fast: {args.material}/{args.substrate}, {args.samples} samples, {args.n_jobs} jobs")
    rng = np.random.RandomState(args.seed)
    task_args = [(rng.randint(0, 2**31), args.material, args.substrate) for _ in range(args.samples)]
    
    t0 = time.time()
    results, success = [], 0
    
    if args.n_jobs > 1:
        with Pool(args.n_jobs) as pool:
            for i, res in enumerate(pool.imap_unordered(_process_sample, task_args, chunksize=1)):
                if res is not None:
                    results.append(res); success += 1
                if (i + 1) % 50 == 0:
                    elapsed = time.time() - t0
                    rate = (i + 1) / elapsed if elapsed > 0 else 0
                    eta = (args.samples - i - 1) / rate if rate > 0 else 0
                    print(f"  {i+1}/{args.samples} ({100*(i+1)/args.samples:.0f}%), ok={success}, {rate:.1f}/s, ETA {eta/60:.0f}min", flush=True)
    else:
        for i, (seed, mat, sub) in enumerate(task_args):
            res = _process_sample((seed, mat, sub))
            if res is not None: results.append(res); success += 1
            if (i + 1) % 10 == 0:
                print(f"  {i+1}/{args.samples}, ok={success}", flush=True)
    
    t1 = time.time()
    print(f"Done: {success}/{args.samples} ({100*success/max(1,args.samples):.1f}%) in {(t1-t0)/60:.1f}min")
    os.makedirs(os.path.dirname(args.output) if os.path.dirname(args.output) else '.', exist_ok=True)
    with open(args.output, 'wb') as f:
        pickle.dump(results, f)
    print(f"Saved {len(results)} samples to {args.output}")

if __name__ == '__main__':
    main()
