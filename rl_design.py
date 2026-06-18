"""Q-Learning RL agent for metasurface inverse design."""
import numpy as np
import pickle, os, math
from color_utils import rgb_to_lab_scalar, delta_e2000_scalar

D_RANGE = (50, 350)
H_RANGE = (80, 600)
P_RANGE = (200, 600)
BINS = 8
N_STATES = BINS**3
ACTIONS = [("D+5",5,0,0),("D-5",-5,0,0),("H+5",0,5,0),("H-5",0,-5,0),("P+5",0,0,5),("P-5",0,0,-5)]
N_ACTIONS = len(ACTIONS)
CACHE_FILE = "models/rl_qtable.pkl"

def _disc(v, lo, hi):
    return int(np.clip((v - lo) / (hi - lo) * (BINS - 1), 0, BINS - 1))

def _idx(d, h, p):
    return _disc(d, *D_RANGE) * BINS * BINS + _disc(h, *H_RANGE) * BINS + _disc(p, *P_RANGE)

def _compute_rgb(d, h, p):
    """Compute RGB using torch_model (lazy import to avoid mandatory PyTorch dependency)."""
    from torch_model import batch_single_pillar_rgb
    rgb_t = batch_single_pillar_rgb(
        float(d), float(h), float(p)
    )
    return rgb_t.squeeze(0).numpy()


class RLDesigner:
    def __init__(self):
        self.q = np.zeros((N_STATES, N_ACTIONS))
        self.alpha = 0.3
        self.gamma = 0.9
        self.eps = 0.3
        self.trained = False

    def train(self, episodes=2000, progress_cb=None):
        for ep in range(episodes):
            target = np.random.rand(3)
            tgt_lab = rgb_to_lab_scalar(target)
            d = np.random.uniform(*D_RANGE)
            h = np.random.uniform(*H_RANGE)
            p = np.random.uniform(*P_RANGE)
            s = _idx(d, h, p)
            rgb = _compute_rgb(d, h, p)
            best_de = delta_e2000_scalar(rgb_to_lab_scalar(rgb), tgt_lab)
            best = (d, h, p)
            for _ in range(25):
                a = np.random.randint(N_ACTIONS) if np.random.random() < self.eps else np.argmax(self.q[s])
                nd = np.clip(d + ACTIONS[a][1], *D_RANGE)
                nh = np.clip(h + ACTIONS[a][2], *H_RANGE)
                npp = np.clip(p + ACTIONS[a][3], *P_RANGE)
                nrgb = _compute_rgb(nd, nh, npp)
                de = delta_e2000_scalar(rgb_to_lab_scalar(nrgb), tgt_lab)
                reward = -de
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

    def search(self, target_hex, steps=30):
        tr = int(target_hex[1:3], 16) / 255.0
        tg = int(target_hex[3:5], 16) / 255.0
        tb = int(target_hex[5:7], 16) / 255.0
        tgt = np.array([tr, tg, tb])
        tgt_lab = rgb_to_lab_scalar(tgt)
        d = (D_RANGE[0] + D_RANGE[1]) / 2
        h = (H_RANGE[0] + H_RANGE[1]) / 2
        p = (P_RANGE[0] + P_RANGE[1]) / 2
        s = _idx(d, h, p)
        best_de = float("inf")
        best = (d, h, p, "#000")
        for _ in range(steps):
            a = np.argmax(self.q[s]) if (self.trained and np.max(self.q[s]) > 0) else np.random.randint(N_ACTIONS)
            d = np.clip(d + ACTIONS[a][1], *D_RANGE)
            h = np.clip(h + ACTIONS[a][2], *H_RANGE)
            p = np.clip(p + ACTIONS[a][3], *P_RANGE)
            rgb = _compute_rgb(d, h, p)
            de = delta_e2000_scalar(rgb_to_lab_scalar(rgb), tgt_lab)
            if de < best_de:
                best_de = de
                rc = [max(0, min(255, int(c * 255))) for c in rgb]
                best = (d, h, p, f"#{rc[0]:02x}{rc[1]:02x}{rc[2]:02x}")
            s = _idx(d, h, p)
            if de < 2.0:
                break
        return best[0], best[1], best[2], best[3], best_de

    def save(self, path=CACHE_FILE):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "wb") as f:
            pickle.dump({"q": self.q, "trained": self.trained}, f)

    def load(self, path=CACHE_FILE):
        if os.path.exists(path):
            with open(path, "rb") as f:
                d = pickle.load(f)
            self.q = d["q"]
            self.trained = d.get("trained", True)
            return True
        return False


def get_trained_rl():
    rl = RLDesigner()
    if rl.load():
        return rl
    rl.train(2000)
    rl.save()
    return rl
