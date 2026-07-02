#!/usr/bin/env python3
"""
MEEP FDTD: TiO2 nanorod on SiO2, TE normal incidence.
Computes reflection R and transmission T for direct Fano model comparison.
"""
import meep as mp
import numpy as np
import pickle, os, sys
from datetime import datetime

RESOLUTION = 50         # px/um (~20nm grid)
PML_THICKNESS = 0.5     # 500nm PML
SUBSTRATE_H = 0.5       # 500nm SiO2
NANOROD_H = 0.600       # 600nm TiO2
WL_MIN, WL_MAX = 0.38, 0.78  # 380-780nm
NFREQ = 81

# TiO2 anatase ~2.5, SiO2 ~1.46
N_TIO2 = 2.5
N_SIO2 = 1.46

def run_sim(period_um, diameter_um, label=""):
    """Run MEEP for one (P,D) config. Returns (wl_nm, R, T)."""
    radius = diameter_um / 2.0
    sx = period_um
    sy = period_um
    sz = PML_THICKNESS + NANOROD_H + SUBSTRATE_H + PML_THICKNESS
    
    cell = mp.Vector3(sx, sy, sz)
    
    geometry = []
    
    # SiO2 substrate
    sub_z = -sz/2 + SUBSTRATE_H/2
    geometry.append(mp.Block(
        size=mp.Vector3(mp.inf, mp.inf, SUBSTRATE_H),
        center=mp.Vector3(0, 0, sub_z),
        material=mp.Medium(epsilon=N_SIO2**2)
    ))
    
    # TiO2 nanorod
    rod_z = sub_z + SUBSTRATE_H/2 + NANOROD_H/2
    geometry.append(mp.Cylinder(
        radius=radius,
        height=NANOROD_H,
        center=mp.Vector3(0, 0, rod_z),
        axis=mp.Vector3(0, 0, 1),
        material=mp.Medium(epsilon=N_TIO2**2)
    ))
    
    # Source
    fmin, fmax = 1.0/WL_MAX, 1.0/WL_MIN
    fcen = 0.5*(fmin + fmax)
    fwidth = fmax - fmin
    src_z = rod_z + NANOROD_H/2 + 0.1
    
    sources = [mp.Source(
        mp.GaussianSource(fcen, fwidth=fwidth),
        component=mp.Ey,  # TE (s-pol)
        center=mp.Vector3(0, 0, src_z),
        size=mp.Vector3(sx, sy, 0)
    )]
    
    # Flux monitors
    refl_z = src_z + 0.05
    trans_z = sub_z - SUBSTRATE_H/2 - 0.1
    
    refl_reg = mp.FluxRegion(center=mp.Vector3(0,0,refl_z), size=mp.Vector3(sx,sy,0))
    trans_reg = mp.FluxRegion(center=mp.Vector3(0,0,trans_z), size=mp.Vector3(sx,sy,0))
    
    refl_flux = mp.add_flux(fcen, fwidth, NFREQ, refl_reg)
    trans_flux = mp.add_flux(fcen, fwidth, NFREQ, trans_reg)
    
    sim = mp.Simulation(
        cell_size=cell,
        boundary_layers=[mp.PML(PML_THICKNESS)],
        geometry=geometry,
        sources=sources,
        resolution=RESOLUTION,
        k_point=mp.Vector3(0, 0, 0)
    )
    
    sim.run(until_after_sources=mp.stop_when_fields_decayed(
        50, mp.Ey, mp.Vector3(0, 0, src_z), 1e-5))
    
    freqs = np.asarray(mp.get_flux_freqs(refl_flux))
    wl_nm = 1.0 / freqs * 1000.0
    R_raw = np.asarray(mp.get_fluxes(refl_flux))
    T_raw = np.asarray(mp.get_fluxes(trans_flux))
    
    sim.destroy()
    
    # Sort by wavelength
    idx = np.argsort(wl_nm)
    return wl_nm[idx], R_raw[idx], T_raw[idx]


def run_reference(period_um):
    """Reference: substrate only, no rod."""
    sx = period_um
    sy = period_um
    sz = PML_THICKNESS + NANOROD_H + SUBSTRATE_H + PML_THICKNESS
    cell = mp.Vector3(sx, sy, sz)
    
    sub_z = -sz/2 + SUBSTRATE_H/2
    geometry = [mp.Block(
        size=mp.Vector3(mp.inf, mp.inf, SUBSTRATE_H),
        center=mp.Vector3(0, 0, sub_z),
        material=mp.Medium(epsilon=N_SIO2**2)
    )]
    
    fmin, fmax = 1.0/WL_MAX, 1.0/WL_MIN
    fcen = 0.5*(fmin + fmax)
    fwidth = fmax - fmin
    src_z = sub_z + SUBSTRATE_H/2 + NANOROD_H + 0.1
    
    sources = [mp.Source(
        mp.GaussianSource(fcen, fwidth=fwidth),
        component=mp.Ey,
        center=mp.Vector3(0, 0, src_z),
        size=mp.Vector3(sx, sy, 0)
    )]
    
    refl_z = src_z + 0.05
    trans_z = sub_z - SUBSTRATE_H/2 - 0.1
    
    refl_reg = mp.FluxRegion(center=mp.Vector3(0,0,refl_z), size=mp.Vector3(sx,sy,0))
    trans_reg = mp.FluxRegion(center=mp.Vector3(0,0,trans_z), size=mp.Vector3(sx,sy,0))
    
    refl_flux = mp.add_flux(fcen, fwidth, NFREQ, refl_reg)
    trans_flux = mp.add_flux(fcen, fwidth, NFREQ, trans_reg)
    
    sim = mp.Simulation(
        cell_size=cell,
        boundary_layers=[mp.PML(PML_THICKNESS)],
        geometry=geometry,
        sources=sources,
        resolution=RESOLUTION,
        k_point=mp.Vector3(0, 0, 0)
    )
    
    sim.run(until_after_sources=mp.stop_when_fields_decayed(
        50, mp.Ey, mp.Vector3(0, 0, src_z), 1e-5))
    
    freqs = np.asarray(mp.get_flux_freqs(refl_flux))
    wl_nm = 1.0 / freqs * 1000.0
    R_ref = np.asarray(mp.get_fluxes(refl_flux))
    T_ref = np.asarray(mp.get_fluxes(trans_flux))
    
    sim.destroy()
    
    idx = np.argsort(wl_nm)
    return wl_nm[idx], R_ref[idx], T_ref[idx]


# === Main ===
if __name__ == "__main__":
    out_dir = os.path.dirname(os.path.abspath(__file__))
    out_path = os.path.join(out_dir, "meep_results.pickle")
    
    print(f"MEEP FDTD TiO2 Metasurface Validation")
    print(f"Resolution: {RESOLUTION} px/um")
    print(f"Output: {out_path}")
    print()
    
    diameters_nm = [80, 100, 130, 160, 200, 260, 300]
    periods_nm = [200, 300]
    
    results = {}
    
    for P in periods_nm:
        P_um = P / 1000.0
        print(f"{'='*50}")
        print(f"Period P={P}nm  |  Reference simulation...")
        
        wl_ref, R_ref, T_ref = run_reference(P_um)
        print(f"  Reference done.")
        
        results[P] = {}
        
        for D in diameters_nm:
            D_um = D / 1000.0
            label = f"P={P} D={D}"
            print(f"  {label}...", end=" ", flush=True)
            
            try:
                wl, R_raw, T_raw = run_sim(P_um, D_um, label)
                # Normalize
                R = np.clip(R_raw / np.maximum(R_ref, 1e-10), 0, 1)
                T = np.clip(T_raw / np.maximum(T_ref, 1e-10), 0, 1)
                
                results[P][D] = {"wl_nm": wl, "R": R, "T": T}
                print(f"R_peak={R.max():.3f}")
            except Exception as e:
                print(f"FAILED: {e}")
    
    # Save
    with open(out_path, "wb") as f:
        pickle.dump(results, f)
    
    print(f"\n{'='*50}")
    print(f"Saved: {out_path}")
    
    # Quick summary
    for P in periods_nm:
        if P in results:
            n = len(results[P])
            R_peaks = [results[P][D]["R"].max() for D in results[P]]
            print(f"  P={P}nm: {n} configs, R_peak [{min(R_peaks):.3f}, {max(R_peaks):.3f}]")
    
    print("DONE")
