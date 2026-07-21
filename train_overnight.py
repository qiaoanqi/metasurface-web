import os, subprocess, sys

BASE = r"C:\Users\A\Desktop\AI超表面结构色智能设计系统"
os.chdir(BASE)

TRAINS = [
    ("a-Si/Si3N4 s3", "data/rcwa_new/rcwa_a-Si_Si3N4.pkl", 456, "forward_mlp_rcwa_aSi_Si3N4_s3"),
    ("a-Si/Al2O3 s2", "data/rcwa_new/rcwa_a-Si_Al2O3.pkl", 123, "forward_mlp_rcwa_aSi_Al2O3_s2"),
    ("a-Si/Al2O3 s3", "data/rcwa_new/rcwa_a-Si_Al2O3.pkl", 456, "forward_mlp_rcwa_aSi_Al2O3_s3"),
]

for label, data, seed, name in TRAINS:
    print(f"=== Starting {label} (seed={seed}) ===")
    cmd = [
        sys.executable, "-u", "train_rcwa.py",
        "--data", data,
        "--epochs", "1000",
        "--lr", "5e-4",
        "--seed", str(seed),
        "--name", name,
    ]
    subprocess.run(cmd, cwd=BASE)

# Update registry to glob patterns
print("=== Updating registry ===")
with open("ml_module.py", encoding="utf-8-sig") as f:
    content = f.read()
content = content.replace(
    '("a-Si (amorphous)", "Si3N4 (nitride)"): ["forward_mlp_rcwa_aSi_Si3N4_s1.onnx"]',
    '("a-Si (amorphous)", "Si3N4 (nitride)"): ["forward_mlp_rcwa_aSi_Si3N4_s?.onnx"]'
)
content = content.replace(
    '("a-Si (amorphous)", "Al2O3 (sapphire)"): ["forward_mlp_rcwa_aSi_Al2O3_s1.onnx"]',
    '("a-Si (amorphous)", "Al2O3 (sapphire)"): ["forward_mlp_rcwa_aSi_Al2O3_s?.onnx"]'
)
with open("ml_module.py", "w", encoding="utf-8") as f:
    f.write(content)

# Clean up single model files
for old_name in ["forward_mlp_rcwa_aSi_Si3N4_s1.pt", "forward_mlp_rcwa_aSi_Si3N4_s1.onnx",
                 "forward_mlp_rcwa_aSi_Al2O3_s1.pt", "forward_mlp_rcwa_aSi_Al2O3_s1.onnx"]:
    pass  # Keep _s1 as part of ensemble

print("=== All done! ===")
print("Models: a-Si/Si3N4 (3-seed), a-Si/Al2O3 (3-seed)")
