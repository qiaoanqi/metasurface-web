# train_new_materials.py — Train 3-seed ResMLP ensembles for GaN, HfO2, Ta2O5
# Launched after RCWA batch completes. ~9 models total (3 materials × 3 seeds).
# Usage: python train_new_materials.py

import os, subprocess, sys

BASE = r"C:\Users\A\Desktop\AI超表面结构色智能设计系统"
os.chdir(BASE)

# (label, data_path, seed, output_name)
TRAINS = [
    # GaN (wurtzite) / SiO2 — Δn(550)=0.94, high-Δn control
    ("GaN s1", "data/rcwa_GaN_SiO2.pkl", 42,  "forward_mlp_rcwa_GaN_s1"),
    ("GaN s2", "data/rcwa_GaN_SiO2.pkl", 123, "forward_mlp_rcwa_GaN_s2"),
    ("GaN s3", "data/rcwa_GaN_SiO2.pkl", 456, "forward_mlp_rcwa_GaN_s3"),
    # HfO2 (hafnia) / SiO2 — Δn(550)=0.55, near-cutoff validation
    ("HfO2 s1", "data/rcwa_HfO2_SiO2.pkl", 42,  "forward_mlp_rcwa_HfO2_s1"),
    ("HfO2 s2", "data/rcwa_HfO2_SiO2.pkl", 123, "forward_mlp_rcwa_HfO2_s2"),
    ("HfO2 s3", "data/rcwa_HfO2_SiO2.pkl", 456, "forward_mlp_rcwa_HfO2_s3"),
    # Ta2O5 (tantala) / SiO2 — Δn(550)=0.65, above-cutoff
    ("Ta2O5 s1", "data/rcwa_Ta2O5_SiO2.pkl", 42,  "forward_mlp_rcwa_Ta2O5_s1"),
    ("Ta2O5 s2", "data/rcwa_Ta2O5_SiO2.pkl", 123, "forward_mlp_rcwa_Ta2O5_s2"),
    ("Ta2O5 s3", "data/rcwa_Ta2O5_SiO2.pkl", 456, "forward_mlp_rcwa_Ta2O5_s3"),
]

for label, data, seed, name in TRAINS:
    print(f"\n{'='*60}")
    print(f"=== Training {label} (seed={seed}) ===")
    print(f"{'='*60}")
    cmd = [
        sys.executable, "-u", "train_rcwa.py",
        "--data", data,
        "--epochs", "1000",
        "--lr", "5e-4",
        "--seed", str(seed),
        "--name", name,
    ]
    result = subprocess.run(cmd, cwd=BASE)
    if result.returncode != 0:
        print(f"!!! FAILED: {label} (exit code {result.returncode})")

print(f"\n{'='*60}")
print("=== All training complete! ===")
print("Expected models in models/:")
for _, _, _, name in TRAINS:
    print(f"  {name}.pt / {name}.onnx")
print(f"{'='*60}")
