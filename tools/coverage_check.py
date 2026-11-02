"""Точна геометрична перевірка (твердження 3): прообрази кутів вихідного кадру лежать усередині джерела на кожному кадрі.
Запуск: python3 tools/coverage_check.py [реальне_відео.mp4]"""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np, json
from vidstab.synth import render_clip
from vidstab.stabilizer import OnlineStabilizer, StabConfig
def check(frames,cfg):
    st=OnlineStabilizer(cfg); [st.process(f) for f in frames]
    W,H=st.W,st.H; zs=st.history['zoom']; slack=np.inf; viol=0
    for z,d in zip(zs,st.history['corr']):
        st.zoom=z; corr=(d[0]*st.f,d[1]*st.f,d[2]/st.rh,d[3]/st.rh)
        M=np.vstack([st._matrix(corr),[0,0,1]]); Mi=np.linalg.inv(M)
        for x,y in ((0,0),(W-1,0),(0,H-1),(W-1,H-1)):
            u=Mi@np.array([x,y,1.0]); s=min(u[0],W-1-u[0],u[1],H-1-u[1]); slack=min(slack,s)
            viol+= s< -1e-6
    return slack,viol,len(zs)
res=[]; 
for cat in ['low','medium','high','walking','impulse','lowtex','occlusion','blur_noise','fastpan','scenecut','mixed']:
    for seed in (700,701):
        fr,ide,g=render_clip(cat,seed,T=300); del ide
        s,v,n=check(fr,StabConfig()); res.append((cat,seed,s,v,n)); print(cat,seed,'min slack %.4f px  violations %d / %d frames'%(s,v,n),flush=True)
if len(sys.argv) > 1:      # необов'язково: шлях до реального відео
    import cv2
    cap = cv2.VideoCapture(sys.argv[1]); fr = []
    while len(fr) < 300:
        ok, f = cap.read()
        if not ok: break
        fr.append(cv2.resize(f, (640, int(round(f.shape[0] * 640 / f.shape[1] / 2)) * 2)))
    s, v, n = check(fr, StabConfig()); res.append((sys.argv[1], 0, s, v, n)); print(sys.argv[1], 'min slack %.4f px violations %d / %d' % (s, v, n))
json.dump(dict(min_slack=min(r[2] for r in res),viol=int(sum(r[3] for r in res)),frames=int(sum(r[4] for r in res)),clips=len(res)),open('coverage.json','w')); print(json.load(open('coverage.json')))
