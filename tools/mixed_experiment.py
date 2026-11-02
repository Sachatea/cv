"""Нестаціонарне тремтіння (слабке → сильне → слабке): адаптивний запас проти фіксованого.
Запуск: python3 tools/mixed_experiment.py  (≈10 хв); результат — mixed_experiment.csv"""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np, pandas as pd
from dataclasses import replace
from vidstab.synth import render_clip
from vidstab.stabilizer import StabConfig
from vidstab.evaluate import run_online
BASE=StabConfig(); T=450
M={'Final':BASE}
for k in (0.02,0.03,0.04,0.06,0.08): M['fixed%.2f'%k]=replace(BASE,adaptive_crop=False,crop=k)
rows=[]
for seed in range(500,506):
    fr,ide,gt=render_clip('mixed',seed,T=T); n=T//3
    for nm,cfg in M.items():
        out,st,corr,ms=run_online(fr,cfg)
        R=(gt['P']+gt['J'])+corr; e=(R-gt['P'])[:,:2]; zt=np.array(st.history['zoom'])
        for ph,(a,b) in {'low1':(20,n),'high':(n,2*n),'low2':(2*n,T)}.items():
            acc=np.diff(R[a:b,:2],n=2,axis=0)
            rows.append(dict(seed=seed,method=nm,phase=ph,err=np.sqrt(np.mean(np.sum(e[a:b]**2,1))),jit=np.sqrt(np.mean(np.sum(acc**2,1))),area=np.mean(1/zt[a:b]**2)))
        rows.append(dict(seed=seed,method=nm,phase='all',err=np.sqrt(np.mean(np.sum(e[20:]**2,1))),jit=np.sqrt(np.mean(np.sum(np.diff(R[20:,:2],n=2,axis=0)**2,1))),area=np.mean(1/zt**2)))
    print(seed,flush=True)
d=pd.DataFrame(rows); d.to_csv('mixed_experiment.csv',index=False)
print(d.groupby(['phase','method'])[['err','jit','area']].mean().round(2).unstack(0).to_string())
# iso-area на рівні кліпу
res=[]
for sd,g in d[d.phase=='all'].groupby('seed'):
    f=g[g.method!='Final'].sort_values('area'); fin=g[g.method=='Final'].iloc[0]
    res.append((fin.area,fin.jit,np.interp(fin.area,f.area,f.jit),fin.err,np.interp(fin.area,f.area,f.err)))
r=np.array(res); print('iso-area (mixed): area %.3f | jit final %.2f vs fixed %.2f | err final %.2f vs fixed %.2f'%tuple(r.mean(0)))
