"""Твердження 6: чутливість згладжувачів до шуму приростів оцінки руху (симуляція) + СКВ похибки приростів у реальному конвеєрі.
Запуск: python3 tools/noise_sensitivity.py"""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np, json
from dataclasses import replace
from vidstab.filters import PathSmoother
from vidstab.synth import render_clip, shake_path
from vidstab.stabilizer import StabConfig
from vidstab.evaluate import run_online
# --- 1. імпульсна відповідь (крок) згладжувачів
def step_resp(mode,n=800,**kw):
    sm=PathSmoother(mode,q_min=1e-6,q_max=1e-3,r=1.0,ma_window=9,adapt=False); sm.reset(np.zeros(4))
    for _ in range(4000): sm.step(np.zeros(4))
    out=[sm.step(np.ones(4))[0] for _ in range(n)]
    return np.array(out)
res={}
for mode,name in (('kalman','Kalman'),('ma','MA-9')):
    s=step_resp(mode); g=s-1.0; res[name]=dict(gain=float(np.sqrt(np.sum(g**2))),g0=float(g[0]),first_cross=int(np.argmax(s>=0.5)))
print('noise gain sqrt(sum g^2):',{k:round(v['gain'],2) for k,v in res.items()}, 'ratio',round(res['Kalman']['gain']/res['MA-9']['gain'],2))
# --- 2. симуляція: C = J + E, E — випадкове блукання з кроком sigma
rng=np.random.default_rng(1); T=6000; rows=[]
for sig in (0.0,0.1,0.25,0.5,1.0):
    out={}
    for mode,name in (('kalman','Kalman'),('ma','MA-9')):
        errs=[]
        for rep in range(8):
            J=shake_path('medium',T,rng)[:,0]/2.0       # px у масштабі 320
            E=np.cumsum(rng.normal(0,sig,T)); C=J+E
            sm=PathSmoother(mode,q_min=1e-6,q_max=1e-3,r=1.0,ma_window=9,adapt=False); sm.reset(np.array([C[0]]*4)); S=[]
            for z in C: S.append(sm.step(np.array([z]*4))[0])
            S=np.array(S); R=J+(S-C)                    # P=0
            errs.append(np.sqrt(np.mean(R[300:]**2)))
        out[name]=float(np.mean(errs))
    pred={n:float(np.sqrt(out[n]**2*0)) for n in out}
    rows.append((sig,out)); print('sigma %.2f'%sig,{k:round(v,2) for k,v in out.items()})
# аналітичний складник: err^2 ≈ err0^2 + sigma^2 * gain^2
for sig,out in rows[1:]:
    print('sigma %.2f predicted: Kalman %.2f MA-9 %.2f'%(sig,np.sqrt(rows[0][1]['Kalman']**2+sig**2*res['Kalman']['gain']**2),np.sqrt(rows[0][1]['MA-9']**2+sig**2*res['MA-9']['gain']**2)))
json.dump(dict(gain=res,sim=[(s,o) for s,o in rows]),open('noise_sensitivity.json','w'))
# --- 3. оцінка σ приростів у реальному конвеєрі: LK (medium) та ECC (lowtex)
for cat,seed in (('medium',300),('lowtex',400),('lowtex',401)):
    fr,ide,g=render_clip(cat,seed,T=300)
    out,st,corr,ms=run_online(fr,StabConfig()); C=np.array(st.history['C']); f=st.f
    dC=np.diff(C[:,:2],axis=0)*f; dT=np.diff((g['P']+g['J'])[:,:2],axis=0)
    e=dC-dT; rel=np.array(st.history['reliable'])
    print(cat,seed,'increment error std per channel (px, full frame):',e.std(0).round(3),'mean',e.mean(0).round(3),'frac reliable',rel.mean().round(2),'used_ecc',st.est.used_ecc)
