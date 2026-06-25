"""color_utils.py - Unified CIE 1931 color science constants and functions.

All modules (app.py, ml_module.py, torch_model.py, rl_design.py) import from here.
Data: CIE 015:2018, CIE 1931 2-deg Standard Observer (380-780nm, 5nm step, 81 points).
"""
import math
import numpy as np

# ============================================================
# CIE 1931 2-deg Standard Observer CMFs (380-780nm, 5nm, 81 pts)
# ============================================================
CIE_WAVELENGTHS = np.arange(380, 785, 5, dtype=np.float32)

CIE_X = np.array([
    0.001368,0.002236,0.004243,0.007650,0.014310,0.023190,0.043510,0.077630,0.134380,0.214770,
    0.283900,0.328500,0.348280,0.348060,0.336200,0.318700,0.290800,0.251100,0.195360,0.142100,
    0.095640,0.058010,0.032010,0.014700,0.004900,0.002400,0.009300,0.029100,0.063270,0.109600,
    0.165500,0.225750,0.290400,0.359700,0.433450,0.512050,0.594500,0.678400,0.762100,0.842500,
    0.916300,0.978600,1.026300,1.056700,1.062200,1.045600,1.002600,0.938400,0.854450,0.751400,
    0.642400,0.541900,0.447900,0.360800,0.283500,0.218700,0.164900,0.121200,0.087400,0.063600,
    0.046770,0.032900,0.022700,0.015840,0.011359,0.008111,0.005790,0.004109,0.002899,0.002049,
    0.001440,0.001000,0.000690,0.000476,0.000332,0.000235,0.000166,0.000117,0.000083,0.000059,
    0.000042], dtype=np.float64)

CIE_Y = np.array([
    0.000039,0.000064,0.000120,0.000217,0.000396,0.000640,0.001210,0.002180,0.004000,0.007300,
    0.011600,0.016840,0.023000,0.029800,0.038000,0.048000,0.060000,0.073900,0.090980,0.112600,
    0.139020,0.169300,0.208020,0.258600,0.323000,0.407300,0.503000,0.608200,0.710000,0.793200,
    0.862000,0.914850,0.954000,0.980300,0.994950,1.000000,0.995000,0.978600,0.952000,0.915400,
    0.870000,0.816300,0.757000,0.694900,0.631000,0.566800,0.503000,0.441200,0.381000,0.321000,
    0.265000,0.217000,0.175000,0.138200,0.107000,0.081600,0.061000,0.044580,0.032000,0.023200,
    0.017000,0.011920,0.008210,0.005723,0.004102,0.002929,0.002091,0.001484,0.001047,0.000740,
    0.000520,0.000361,0.000249,0.000172,0.000120,0.000085,0.000060,0.000042,0.000030,0.000021,
    0.000015], dtype=np.float64)

# CIE_Z: full precision data (non-zero in short-wavelength region)
CIE_Z = np.array([
    0.006450,0.010550,0.020050,0.036210,0.067850,0.110200,0.207400,0.371300,0.645600,1.039050,
    1.385600,1.622960,1.747060,1.782600,1.772110,1.744100,1.669200,1.528100,1.287640,1.041900,
    0.812950,0.616200,0.465180,0.353300,0.272000,0.212300,0.158200,0.111700,0.078250,0.057250,
    0.042160,0.029840,0.020300,0.013400,0.008750,0.005750,0.003900,0.002750,0.002100,0.001800,
    0.001650,0.001400,0.001100,0.001000,0.000800,0.000600,0.000340,0.000240,0.000190,0.000100,
    0.000050,0.000030,0.000020,0.000010,0.000000,0.000000,0.000000,0.000000,0.000000,0.000000,
    0.000000,0.000000,0.000000,0.000000,0.000000,0.000000,0.000000,0.000000,0.000000,0.000000,
    0.000000,0.000000,0.000000,0.000000,0.000000,0.000000,0.000000,0.000000,0.000000,0.000000,
    0.000000], dtype=np.float64)

WL = np.linspace(380, 780, 81, dtype=np.float32)
# NumPy 1.x/2.x compatibility
try:
    _trapz = np.trapezoid
except AttributeError:
    _trapz = np.trapz
CIE_NORM = float(_trapz(CIE_Y, WL))

D65 = np.array([0.95047, 1.00000, 1.08883], dtype=np.float64)

SRGB_M = np.array([
    [ 3.2404542, -1.5371385, -0.4985314],
    [-0.9692660,  1.8760108,  0.0415560],
    [ 0.0556434, -0.2040259,  1.0572252],
], dtype=np.float64)

SRGB_M_INV = np.array([
    [0.4124564, 0.3575761, 0.1804375],
    [0.2126729, 0.7151522, 0.0721750],
    [0.0193339, 0.1191920, 0.9503041],
], dtype=np.float64)


# ============================================================
# Spectrum -> Color conversions (NumPy)
# ============================================================

def spectrum_to_xyz(wavelengths_nm, reflectance):
    """Reflectance spectrum -> CIE XYZ tristimulus."""
    dwl = 5.0
    X = dwl * np.sum(reflectance * CIE_X)
    Y = dwl * np.sum(reflectance * CIE_Y)
    Z = dwl * np.sum(reflectance * CIE_Z)
    norm = dwl * np.sum(CIE_Y)
    if norm > 1e-12:
        X /= norm; Y /= norm; Z /= norm
    return np.array([X, Y, Z])


def xyz_to_srgb(xyz):
    """CIE XYZ -> sRGB (gamma-corrected, clipped [0,1])."""
    linear = SRGB_M @ xyz
    linear = np.clip(linear, 0, 1)
    return np.where(linear <= 0.0031308, 12.92 * linear,
                    1.055 * linear ** (1 / 2.4) - 0.055)


def spectrum_to_srgb(wavelengths_nm, reflectance):
    """Reflectance spectrum -> sRGB."""
    reflectance = np.nan_to_num(np.asarray(reflectance, dtype=float), nan=0.0, posinf=1.0, neginf=0.0)
    xyz = spectrum_to_xyz(wavelengths_nm, reflectance)
    return np.nan_to_num(np.clip(xyz_to_srgb(xyz), 0, 1), nan=0.0)


def clamp01(x):
    return np.clip(np.asarray(x, dtype=float), 0.0, 1.0)


def srgb_to_linear(rgb):
    rgb = np.asarray(rgb, dtype=float)
    return np.where(rgb <= 0.04045, rgb / 12.92, ((rgb + 0.055) / 1.055) ** 2.4)


def rgb_to_xyz(rgb):
    rgb_lin = srgb_to_linear(rgb)
    return rgb_lin @ SRGB_M_INV.T


def xyz_to_xy(xyz):
    xyz = np.asarray(xyz, dtype=float)
    denom = np.sum(xyz, axis=-1, keepdims=True)
    denom = np.where(denom <= 1e-12, 1e-12, denom)
    return xyz[..., :2] / denom


def rgb_to_xy(rgb):
    return xyz_to_xy(rgb_to_xyz(rgb))


def xyz_to_lab(xyz):
    xyz_scaled = np.asarray(xyz, dtype=float) / D65
    eps, kappa = 216 / 24389, 24389 / 27
    f = np.where(xyz_scaled > eps, np.cbrt(xyz_scaled), (kappa * xyz_scaled + 16) / 116)
    L = 116 * f[..., 1] - 16
    a = 500 * (f[..., 0] - f[..., 1])
    b = 200 * (f[..., 1] - f[..., 2])
    return np.stack([L, a, b], axis=-1)


def rgb_to_lab(rgb):
    return xyz_to_lab(rgb_to_xyz(rgb))


def rgb_to_hex(rgb):
    rgb = np.nan_to_num(np.asarray(rgb, dtype=float), nan=0.0, posinf=1.0, neginf=0.0)
    r, g, b = np.rint(clamp01(rgb) * 255).astype(int)
    return f"#{r:02x}{g:02x}{b:02x}"


def rgb_255(rgb):
    rgb = np.nan_to_num(np.asarray(rgb, dtype=float), nan=0.0, posinf=1.0, neginf=0.0)
    r, g, b = np.rint(clamp01(rgb) * 255).astype(int)
    return int(r), int(g), int(b)


def delta_e76(lab1, lab2):
    return float(np.linalg.norm(np.asarray(lab1) - np.asarray(lab2)))


def delta_e2000(lab1, lab2):
    """CIEDE2000 perceptual color difference."""
    L1, a1, b1 = np.asarray(lab1, dtype=float)
    L2, a2, b2 = np.asarray(lab2, dtype=float)
    C1 = np.sqrt(a1**2 + b1**2)
    C2 = np.sqrt(a2**2 + b2**2)
    Cbar = (C1 + C2) / 2.0
    G = 0.5 * (1 - np.sqrt(Cbar**7 / (Cbar**7 + 25**7)))
    a1p = (1 + G) * a1; a2p = (1 + G) * a2
    C1p = np.sqrt(a1p**2 + b1**2); C2p = np.sqrt(a2p**2 + b2**2)
    h1p = np.degrees(np.arctan2(b1, a1p)) % 360
    h2p = np.degrees(np.arctan2(b2, a2p)) % 360
    dLp = L2 - L1; dCp = C2p - C1p
    if C1p * C2p == 0:
        dhp = 0.0
    else:
        dh = h2p - h1p
        if abs(dh) <= 180: dhp = dh
        elif dh > 180: dhp = dh - 360
        else: dhp = dh + 360
    dHp = 2 * np.sqrt(C1p * C2p) * np.sin(np.radians(dhp / 2.0))
    Lpbar = (L1 + L2) / 2.0; Cpbar = (C1p + C2p) / 2.0
    if C1p * C2p == 0: hpbar = h1p + h2p
    else:
        hpbar = (h1p + h2p) / 2.0
        if abs(h1p - h2p) > 180: hpbar = (h1p + h2p + 360) / 2.0
    T = (1 - 0.17*np.cos(np.radians(hpbar-30)) + 0.24*np.cos(np.radians(2*hpbar))
         + 0.32*np.cos(np.radians(3*hpbar+6)) - 0.20*np.cos(np.radians(4*hpbar-63)))
    dtheta = 30 * np.exp(-((hpbar - 275) / 25)**2)
    RC = 2 * np.sqrt(Cpbar**7 / (Cpbar**7 + 25**7))
    SL = 1 + (0.015*(Lpbar-50)**2) / np.sqrt(20 + (Lpbar-50)**2)
    SC = 1 + 0.045*Cpbar; SH = 1 + 0.015*Cpbar*T
    RT = -np.sin(np.radians(2*dtheta)) * RC
    return float(np.sqrt((dLp/SL)**2 + (dCp/SC)**2 + (dHp/SH)**2 + RT*(dCp/SC)*(dHp/SH)))


# ============================================================
# Fast scalar versions (no numpy overhead, for RL / single-point)
# ============================================================

def rgb_to_lab_scalar(rgb):
    """Fast scalar rgb->lab."""
    r, g, b = float(np.clip(rgb[0], 0, 1)), float(np.clip(rgb[1], 0, 1)), float(np.clip(rgb[2], 0, 1))
    rl = r / 12.92 if r <= 0.04045 else ((r + 0.055) / 1.055) ** 2.4
    gl = g / 12.92 if g <= 0.04045 else ((g + 0.055) / 1.055) ** 2.4
    bl = b / 12.92 if b <= 0.04045 else ((b + 0.055) / 1.055) ** 2.4
    x = 0.4124564*rl + 0.3575761*gl + 0.1804375*bl
    y = 0.2126729*rl + 0.7151522*gl + 0.0721750*bl
    z = 0.0193339*rl + 0.1191920*gl + 0.9503041*bl
    eps = 0.008856
    def f(v): return v**(1/3) if v > eps else (903.3*v+16)/116
    fx, fy, fz = f(x/0.95047), f(y/1.0), f(z/1.08883)
    return np.array([116*fy-16, 500*(fx-fy), 200*(fy-fz)])


def delta_e2000_scalar(lab1, lab2):
    """CIEDE2000 for scalar Lab (pure Python)."""
    L1,a1,b1 = float(lab1[0]),float(lab1[1]),float(lab1[2])
    L2,a2,b2 = float(lab2[0]),float(lab2[1]),float(lab2[2])
    C1=math.sqrt(a1**2+b1**2); C2=math.sqrt(a2**2+b2**2)
    Cavg=(C1+C2)/2; G=0.5*(1-math.sqrt(Cavg**7/(Cavg**7+25**7)))
    a1p=a1*(1+G); a2p=a2*(1+G)
    C1p=math.sqrt(a1p**2+b1**2); C2p=math.sqrt(a2p**2+b2**2)
    h1p=math.degrees(math.atan2(b1,a1p))%360
    h2p=math.degrees(math.atan2(b2,a2p))%360
    dLp=L2-L1; dCp=C2p-C1p
    if C1p*C2p==0: dh=0
    elif abs(h2p-h1p)<=180: dh=h2p-h1p
    elif h2p-h1p>180: dh=h2p-h1p-360
    else: dh=h2p-h1p+360
    dHp=2*math.sqrt(C1p*C2p)*math.sin(math.radians(dh)/2)
    Lavg=(L1+L2)/2; Cavgp=(C1p+C2p)/2
    if C1p*C2p==0: havg=h1p+h2p
    elif abs(h1p-h2p)<=180: havg=(h1p+h2p)/2
    elif h1p+h2p<360: havg=(h1p+h2p+360)/2
    else: havg=(h1p+h2p-360)/2
    T=(1-0.17*math.cos(math.radians(havg-30))+0.24*math.cos(math.radians(2*havg))
       +0.32*math.cos(math.radians(3*havg+6))-0.20*math.cos(math.radians(4*havg-63)))
    dtheta=30*math.exp(-((havg-275)/25)**2)
    RC=2*math.sqrt(Cavgp**7/(Cavgp**7+25**7))
    SL=1+0.015*(Lavg-50)**2/math.sqrt(20+(Lavg-50)**2)
    SC=1+0.045*Cavgp; SH=1+0.015*Cavgp*T
    RT=-math.sin(math.radians(2*dtheta))*RC
    return math.sqrt((dLp/SL)**2+(dCp/SC)**2+(dHp/SH)**2+RT*(dCp/SC)*(dHp/SH))
