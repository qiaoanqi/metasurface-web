import pickle, os, numpy as np
os.chdir(r"C:\Users\A\Desktop\AI超表面结构色智能设计系统")
from color_utils import rgb_to_lab, delta_e2000

# Compute per-candidate prediction error std from closed-loop data
# In the hybrid N=100 data, each target has naive_pred and naive_ach
# The per-candidate error is naive_ach - naive_pred
# σ = std of these errors across all targets
with open("data/closed_loop_TiO2_SiO2_roundtrip_N100.pkl", "rb") as f:
    rt = pickle.load(f)

errors = []
for r in rt:
    if r.get("naive_ach_de2000") is not None and r.get("naive_pred_de2000") is not None:
        errors.append(r["naive_ach_de2000"] - r["naive_pred_de2000"])
errors = np.array(errors)
sigma_naive = np.std(errors)
print(f"σ from naive prediction errors (N={len(errors)}): {sigma_naive:.2f}")
print(f"  mean error: {np.mean(errors):+.2f}")
print(f"  sqrt(2ln1392) = {np.sqrt(2*np.log(1392)):.2f}")
print(f"  σ × sqrt(2lnN) = {sigma_naive * np.sqrt(2*np.log(1392)):.2f}")
print(f"  Measured curse gap median: {np.median(errors):+.2f}")

# Also compute holdout per-sample error std
# Check if holdout data exists with predictions
holdout_files = [f for f in os.listdir("data") if "forward_mlp_rcwa_TiO2" in f and "test" in f]
print(f"\nTiO2 holdout test files: {holdout_files}")

# Try loading holdout test
for fname in holdout_files[:1]:
    try:
        with open(os.path.join("data", fname), "rb") as f:
            test = pickle.load(f)
        print(f"  {fname}: type={type(test).__name__}")
        if isinstance(test, list) and len(test) > 0:
            r0 = test[0]
            print(f"  keys: {list(r0.keys())[:15]}")
        elif isinstance(test, dict):
            print(f"  keys: {list(test.keys())[:15]}")
    except Exception as e:
        print(f"  {fname}: {e}")
