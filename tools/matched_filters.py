"""Калман проти MA-9 за однакових ECC, ваги й політики запасу (фіксований κ та адаптивний).
Запуск: python3 tools/matched_filters.py  (≈10 хв, 2 процеси); результат — matched_filters.csv"""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np, pandas as pd, time
from dataclasses import replace
from multiprocessing import Pool
from vidstab.synth import render_clip
from vidstab.stabilizer import StabConfig
from vidstab.evaluate import run_online
BASE=StabConfig(); T=300
M={'Kalman adaptive':BASE,'MA-9 adaptive':replace(BASE,method='ma',ma_window=9)}
for k in (0.02,0.03,0.04,0.06,0.08):
    M['Kalman fixed %.2f'%k]=replace(BASE,adaptive_crop=False,crop=k)
    M['MA-9 fixed %.2f'%k]=replace(BASE,method='ma',ma_window=9,adaptive_crop=False,crop=k)
def job(a):
    cat,seed=a; fr,ide,gt=render_clip(cat,seed,T=T); del ide; rows=[]
    for nm,cfg in M.items():
        out,st,corr,ms=run_online(fr,cfg); del out
        R=(gt['P']+gt['J'])+corr; e=(R-gt['P'])[20:,:2]; zt=np.array(st.history['zoom'])
        rows.append(dict(cat=cat,seed=seed,method=nm,err=float(np.sqrt(np.mean(np.sum(e**2,1)))),jit=float(np.sqrt(np.mean(np.sum(np.diff(R[20:,:2],n=2,axis=0)**2,1)))),area=float(np.mean(1/zt**2)),zmin=float(zt.min()),zmax=float(zt.max())))
    print(cat,seed,time.strftime('%X'),flush=True); return rows
if __name__=='__main__':
    cats=['low','medium','high','walking','impulse','occlusion','fastpan','blur_noise','scenecut']
    jobs=[(c,s) for s in (600,601,602) for c in cats]
    with Pool(2) as p: res=p.map(job,jobs,chunksize=1)
    pd.DataFrame([r for rr in res for r in rr]).to_csv('matched_filters.csv',index=False); print('DONE')
