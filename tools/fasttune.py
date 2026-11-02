import sys, pickle, itertools; sys.path.insert(0,'/tmp/claude-0/-home-claude/0eface35-0138-50e1-ac1f-21dad35a4044/scratchpad/cv')
import numpy as np, pandas as pd
from vidstab.filters import PathSmoother
def sim(d, mode, **kw):
    C=d['C']; sm=PathSmoother(mode, **kw); sm.reset(C[0]); S=[]
    for z,r in zip(C,d['rel']): S.append(sm.step(z, bool(r)))
    S=np.array(S); dd=S-C; m=d['margin']; dd=m*np.tanh(dd/m)
    f,rh=d['f'],d['rh']
    corr=np.stack([dd[:,0]*f,dd[:,1]*f,dd[:,2]/rh,dd[:,3]/rh],1)
    P,J=d['gt']['P'],d['gt']['J']; R=(P+J)+corr; e=R-P
    err=np.sqrt(np.mean(e[:,0]**2+e[:,1]**2)); acc=np.diff(R[:,:2],n=2,axis=0); jit=np.sqrt(np.mean(np.sum(acc**2,1)))
    # прибрати вплив початкового перехідного процесу
    e2=e[20:]; err2=np.sqrt(np.mean(e2[:,0]**2+e2[:,1]**2))
    return err2, jit
if __name__=='__main__':
    data=pickle.load(open(sys.argv[1],'rb'))
    rows=[]
    # baseline-и
    for name,mode,kw in [('none','none',{})]+[('ma%d'%w,'ma',dict(ma_window=w)) for w in (9,15,25,40)]+\
        [('kal q=%g'%q,'kalman',dict(q_min=q)) for q in (1e-5,3e-5,1e-4,3e-4,1e-3,3e-3)]:
        for d in data:
            e,j=sim(d,mode,**kw); rows.append(dict(cfg=name,cat=d['cat'],err=e,jit=j))
    df=pd.DataFrame(rows)
    t=df.groupby('cfg')[['err','jit']].mean(); print(t.round(2).to_string())
    print(df.pivot_table(index='cfg',columns='cat',values='err').round(1).to_string())
