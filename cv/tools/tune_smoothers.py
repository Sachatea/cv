import sys, pickle, itertools; sys.path.insert(0,'/tmp'); sys.path.insert(0,'/tmp/claude-0/-home-claude/0eface35-0138-50e1-ac1f-21dad35a4044/scratchpad/cv')
import numpy as np, pandas as pd
from fasttune import sim
data=pickle.load(open(sys.argv[1],'rb'))
def score(mode, **kw):
    r=[sim(d,mode,**kw) for d in data]
    return np.mean([x[0] for x in r]), np.mean([x[1] for x in r]), r
rows=[]
for q in (1e-6,3e-6,1e-5):
    e,j,_=score('kalman',q_min=q); rows.append(dict(cfg='kal q=%g'%q,err=e,jit=j))
res=[]
for qm,qx,al,(r0,r1),fl in itertools.product((1e-6,3e-6,1e-5,3e-5),(1e-3,1e-2,1e-1),(0.1,0.2),((0.3,0.7),(0.4,0.8)),(0.5,1.0)):
    e,j,_=score('intent',q_min=qm,q_max=qx,alpha=al,rho0=r0,rho1=r1,floor=fl)
    res.append(dict(q_min=qm,q_max=qx,alpha=al,rho0=r0,rho1=r1,floor=fl,err=e,jit=j))
df=pd.DataFrame(res).sort_values('err')
print(pd.DataFrame(rows).round(2).to_string())
print(df.head(12).round(3).to_string())
