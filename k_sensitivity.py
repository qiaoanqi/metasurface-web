# K sensitivity: run TiO2 roundtrip N=100 with K=5,10,20,50
# Rerun closed_loop for each K, extracting hybrid success from pkl
# We already have K=20 data. Run K=5,10,50.

import subprocess, os, pickle, numpy as np, sys
os.chdir(r"C:\Users\A\Desktop\AI超表面结构色智能设计系统")

results = {}
for K in [5, 10, 20, 50]:
    outfile = f"data/closed_loop_TiO2_SiO2_roundtrip_N100_K{K}.pkl"
    if K == 20:
        outfile = "data/closed_loop_TiO2_SiO2_roundtrip_N100.pkl"
    
    if os.path.exists(outfile):
        with open(outfile, "rb") as f:
            recs = pickle.load(f)
        ok = [r for r in recs if r.get("status") == "ok" and r.get("ach_de2000") is not None]
        hyb = np.array([r["ach_de2000"] for r in ok])
        succ = sum(hyb < 2.3)
        results[K] = (len(ok), succ, np.mean(hyb), np.median(hyb))
        print(f"K={K}: exists, N={len(ok)}, success={succ}/{len(ok)} ({100*succ/len(ok):.0f}%), mean={np.mean(hyb):.2f}, med={np.median(hyb):.2f}")
    else:
        print(f"K={K}: need to run (output: {outfile})")

# Print summary for already-existing
if all(K in results for K in [5,10,20,50]):
    print("\n=== K Sensitivity Summary ===")
    print(f"{'K':<8} {'N':<6} {'Success':<12} {'Rate':<8} {'Mean DE':<10} {'Med DE'}")
    for K in [5,10,20,50]:
        n, s, m, md = results[K]
        print(f"{K:<8} {n:<6} {s}/{n:<10} {100*s/n:.0f}%{'':<4} {m:<10.2f} {md:.2f}")
