"""Q-Learning RL agent for metasurface inverse design."""
import numpy as np
import os, math
from color_utils import rgb_to_lab_scalar, delta_e2000_scalar

D_RANGE = (50, 350)
H_RANGE = (80, 600)
P_RANGE = (200, 600)
BINS = 16
N_STATES = BINS**3
ACTIONS = [("D+5",5,0,0),("D-5",-5,0,0),("H+5",0,5,0),("H-5",0,-5,0),("P+5",0,0,5),("P-5",0,0,-5)]
N_ACTIONS = len(ACTIONS)
_MODEL_REPO = "qiaoanqi/metasurface-models"

def _ensure_model_file(rel_path):
    local = os.path.join(os.path.dirname(os.path.abspath(__file__)), rel_path)
    if os.path.exists(local):
        return local
    try:
        from huggingface_hub import hf_hub_download
        os.makedirs(os.path.dirname(local), exist_ok=True)
        return hf_hub_download(
            repo_id=_MODEL_REPO, filename=rel_path,
            cache_dir=os.path.join(os.path.dirname(os.path.abspath(__file__)), ".hf_cache"),
            local_dir=os.path.dirname(os.path.abspath(__file__)),
            local_dir_use_symlinks=False,
            timeout=5)
    except Exception:
        return local

_QTABLE_PATH = "models/rl_qtable.npy"
_META_PATH = "models/rl_meta.npy"

def _disc(v, lo, hi):
    return int(np.clip((v - lo) / (hi - lo) * (BINS - 1), 0, BINS - 1))

def _idx(d, h, p):
    return _disc(d, *D_RANGE) * BINS * BINS + _disc(h, *H_RANGE) * BINS + _disc(p, *P_RANGE)

def _compute_rgb(d, h, p, material="TiO2 (anatase)", substrate="SiO2 (fused silica)"):
    import ml_module
    rgb = ml_module.predict_rgb(float(d), float(h), float(p), 0.0, "TE", material, substrate)
    if rgb is None:
        from torch_model import batch_single_pillar_rgb
        rgb_t = batch_single_pillar_rgb(float(d), float(h), float(p))
        return rgb_t.squeeze(0).numpy()
    return rgb


class RLDesigner:
    def __init__(self):
        self.q = np.zeros((N_STATES, N_ACTIONS))
        self.alpha = 0.3
        self.gamma = 0.9
        self.eps = 0.3
        self.trained = False

    def train(self, episodes=5000, progress_cb=None):
        for ep in range(episodes):
            # Epsilon decay: explore aggressively early, exploit later
            eps = max(0.05, 0.30 * (1.0 - ep / max(episodes - 1, 1)))
            target = np.random.rand(3)
            tgt_lab = rgb_to_lab_scalar(target)
            d, h, p = np.random.uniform(*D_RANGE), np.random.uniform(*H_RANGE), np.random.uniform(*P_RANGE)
            s = _idx(d, h, p)
            rgb = _compute_rgb(d, h, p)
            best_de = delta_e2000_scalar(rgb_to_lab_scalar(rgb), tgt_lab)
            best = (d, h, p)
            for _ in range(25):
                a = np.random.randint(N_ACTIONS) if np.random.random() < eps else np.argmax(self.q[s])
                nd = np.clip(d + ACTIONS[a][1], *D_RANGE)
                nh = np.clip(h + ACTIONS[a][2], *H_RANGE)
                npp = np.clip(p + ACTIONS[a][3], *P_RANGE)
                if nd >= npp:  # physical: D must be < P
                    npp = min(P_RANGE[1], nd + 30.0)
                    nd = max(D_RANGE[0], npp - 30.0)
                nrgb = _compute_rgb(nd, nh, npp)
                de = delta_e2000_scalar(rgb_to_lab_scalar(nrgb), tgt_lab)
                reward = float(-de)
                ns = _idx(nd, nh, npp)
                self.q[s, a] += self.alpha * (reward + self.gamma * np.max(self.q[ns]) - self.q[s, a])
                if de < best_de:
                    best_de = de
                    best = (nd, nh, npp)
                d, h, p, s = nd, nh, npp, ns
                if reward > -2.0:
                    break
            if progress_cb and ep % 500 == 0:
                progress_cb(ep, episodes)
        self.trained = True

    def search(self, target_hex, steps=30, restarts=5):
        tr = int(target_hex[1:3], 16) / 255.0
        tg = int(target_hex[3:5], 16) / 255.0
        tb = int(target_hex[5:7], 16) / 255.0
        tgt = np.array([tr, tg, tb])
        tgt_lab = rgb_to_lab_scalar(tgt)
        best_overall = (0, 0, 0, "#000", float("inf"))
        for _ in range(restarts):
            d = np.random.uniform(*D_RANGE)
            h = np.random.uniform(*H_RANGE)
            p = np.random.uniform(*P_RANGE)
            if d >= p:
                p = min(P_RANGE[1], d + 30.0)
            s = _idx(d, h, p)
            init_rgb = _compute_rgb(d, h, p)
            init_de = delta_e2000_scalar(rgb_to_lab_scalar(init_rgb), tgt_lab)
            rc_init = [max(0, min(255, round(c * 255))) for c in init_rgb]
            best = (d, h, p, f"#{rc_init[0]:02x}{rc_init[1]:02x}{rc_init[2]:02x}", init_de)
            for _ in range(steps // restarts):
                a = np.argmax(self.q[s]) if (self.trained and np.max(self.q[s]) > 0) else np.random.randint(N_ACTIONS)
                d = np.clip(d + ACTIONS[a][1], *D_RANGE)
                h = np.clip(h + ACTIONS[a][2], *H_RANGE)
                p = np.clip(p + ACTIONS[a][3], *P_RANGE)
                if d >= p:
                    p = min(P_RANGE[1], d + 30.0)
                rgb = _compute_rgb(d, h, p)
                de = delta_e2000_scalar(rgb_to_lab_scalar(rgb), tgt_lab)
                if de < best[4]:
                    rc = [max(0, min(255, round(c * 255))) for c in rgb]
                    best = (d, h, p, f"#{rc[0]:02x}{rc[1]:02x}{rc[2]:02x}", de)
                s = _idx(d, h, p)
                if de < 2.0:
                    break
            if best[4] < best_overall[4]:
                best_overall = best
        return best_overall[0], best_overall[1], best_overall[2], best_overall[3], best_overall[4]

    def save(self, path=None):
        if path is None: path = _ensure_model_file(_QTABLE_PATH)
        os.makedirs(os.path.dirname(path), exist_ok=True)
        np.save(path, self.q)
        np.save(os.path.splitext(path)[0] + "_meta.npy", np.array([self.trained]))

    def load(self, path=None):
        if path is None:
            path = _ensure_model_file(_QTABLE_PATH)
        if os.path.exists(path):
            self.q = np.load(path)
            meta_path = os.path.splitext(path)[0] + "_meta.npy"
            self.trained = bool(np.load(meta_path)[0]) if os.path.exists(meta_path) else True
            return True
        return False


def get_trained_rl():
    rl = RLDesigner()
    if rl.load():
        return rl
    rl.train(5000)
    rl.save()
    return rl