"""Q-Learning RL agent for metasurface inverse design."""
import numpy as np
import pickle, os, math
from torch_model import batch_single_pillar_rgb

def _to_rgb(tensor):
    return np.array([tensor[0,0].item(), tensor[0,1].item(), tensor[0,2].item()])

def _rgb_to_lab(rgb):
    def _lin(c):
        c = np.clip(c, 0, 1)
        return np.where(c <= 0.04045, c / 12.92, ((c + 0.055) / 1.055) ** 2.4)
    r, g, b = _lin(rgb[0]), _lin(rgb[1]), _lin(rgb[2])
    x = 0.4124564*r + 0.3575761*g + 0.1804375*b
    y = 0.2126729*r + 0.7151522*g + 0.0721750*b
    z = 0.0193339*r + 0.1191920*g + 0.9503041*b
    xn, yn, zn = 0.95047, 1.0, 1.08883
    fx, fy, fz = x/xn, y/yn, z/zn
    eps = 0.008856
    def f(v): return v**(1/3) if v > eps else (903.3*v+16)/116
    L = 116*f(fy)-16; a = 500*(f(fx)-f(fy)); b_val = 200*(f(fy)-f(fz))
    return np.array([L, a, b_val])

def _delta_e2000(lab1, lab2):
    L1,a1,b1=lab1; L2,a2,b2=lab2
    C1=math.sqrt(a1**2+b1**2); C2=math.sqrt(a2**2+b2**2)
    Cavg=(C1+C2)/2
    G=0.5*(1-math.sqrt(Cavg**7/(Cavg**7+25**7)))
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
    T=1-0.17*math.cos(math.radians(havg-30))+0.24*math.cos(math.radians(2*havg))+0.32*math.cos(math.radians(3*havg+6))-0.20*math.cos(math.radians(4*havg-63))
    dtheta=30*math.exp(-((havg-275)/25)**2)
    RC=2*math.sqrt(Cavgp**7/(Cavgp**7+25**7))
    SL=1+0.015*(Lavg-50)**2/math.sqrt(20+(Lavg-50)**2)
    SC=1+0.045*Cavgp; SH=1+0.015*Cavgp*T
    RT=-math.sin(math.radians(2*dtheta))*RC
    return math.sqrt((dLp/SL)**2+(dCp/SC)**2+(dHp/SH)**2+RT*(dCp/SC)*(dHp/SH))

D_RANGE=(50,350); H_RANGE=(80,600); P_RANGE=(200,600); BINS=8; N_STATES=BINS**3
ACTIONS=[("D+5",5,0,0),("D-5",-5,0,0),("H+5",0,5,0),("H-5",0,-5,0),("P+5",0,0,5),("P-5",0,0,-5)]
N_ACTIONS=len(ACTIONS); CACHE_FILE="models/rl_qtable.pkl"

def _disc(v,lo,hi): return int(np.clip((v-lo)/(hi-lo)*(BINS-1),0,BINS-1))
def _idx(d,h,p): return _disc(d,*D_RANGE)*BINS*BINS+_disc(h,*H_RANGE)*BINS+_disc(p,*P_RANGE)

class RLDesigner:
    def __init__(self):
        self.q=np.zeros((N_STATES,N_ACTIONS)); self.alpha=0.3; self.gamma=0.9; self.eps=0.3; self.trained=False

    def train(self, episodes=2000, progress_cb=None):
        for ep in range(episodes):
            target=np.random.rand(3); tgt_lab=_rgb_to_lab(target)
            d=np.random.uniform(*D_RANGE); h=np.random.uniform(*H_RANGE); p=np.random.uniform(*P_RANGE)
            s=_idx(d,h,p)
            rgb=_to_rgb(batch_single_pillar_rgb(d,h,p))
            best_de=_delta_e2000(_rgb_to_lab(rgb),tgt_lab); best=(d,h,p)
            for _ in range(25):
                a=np.random.randint(N_ACTIONS) if np.random.random()<self.eps else np.argmax(self.q[s])
                nd=np.clip(d+ACTIONS[a][1],*D_RANGE); nh=np.clip(h+ACTIONS[a][2],*H_RANGE); npp=np.clip(p+ACTIONS[a][3],*P_RANGE)
                nrgb=_to_rgb(batch_single_pillar_rgb(nd,nh,npp))
                de=_delta_e2000(_rgb_to_lab(nrgb),tgt_lab); reward=-de; ns=_idx(nd,nh,npp)
                self.q[s,a]+=self.alpha*(reward+self.gamma*np.max(self.q[ns])-self.q[s,a])
                if de<best_de: best_de=de; best=(nd,nh,npp)
                d,h,p,s=nd,nh,npp,ns
                if reward>-2.0: break
            if progress_cb and ep%500==0: progress_cb(ep,episodes)
        self.trained=True

    def search(self, target_hex, steps=30):
        tr=int(target_hex[1:3],16)/255.0; tg=int(target_hex[3:5],16)/255.0; tb=int(target_hex[5:7],16)/255.0
        tgt=np.array([tr,tg,tb]); tgt_lab=_rgb_to_lab(tgt)
        d=(D_RANGE[0]+D_RANGE[1])/2; h=(H_RANGE[0]+H_RANGE[1])/2; p=(P_RANGE[0]+P_RANGE[1])/2
        s=_idx(d,h,p); best_de=float("inf"); best=(d,h,p,"#000")
        for _ in range(steps):
            a=np.argmax(self.q[s]) if (self.trained and np.max(self.q[s])>0) else np.random.randint(N_ACTIONS)
            d=np.clip(d+ACTIONS[a][1],*D_RANGE); h=np.clip(h+ACTIONS[a][2],*H_RANGE); p=np.clip(p+ACTIONS[a][3],*P_RANGE)
            rgb=_to_rgb(batch_single_pillar_rgb(d,h,p)); de=_delta_e2000(_rgb_to_lab(rgb),tgt_lab)
            if de<best_de:
                best_de=de; rc=[max(0,min(255,int(c*255))) for c in rgb]
                best=(d,h,p,f"#{rc[0]:02x}{rc[1]:02x}{rc[2]:02x}")
            s=_idx(d,h,p)
            if de<2.0: break
        return best[0],best[1],best[2],best[3],best_de

    def save(self, path=CACHE_FILE):
        os.makedirs(os.path.dirname(path),exist_ok=True)
        with open(path,"wb") as f: pickle.dump({"q":self.q,"trained":self.trained},f)

    def load(self, path=CACHE_FILE):
        if os.path.exists(path):
            with open(path,"rb") as f: d=pickle.load(f); self.q=d["q"]; self.trained=d.get("trained",True)
            return True
        return False

def get_trained_rl():
    rl=RLDesigner()
    if rl.load(): return rl
    rl.train(2000); rl.save(); return rl
