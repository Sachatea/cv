import sys, pickle; sys.path.insert(0,'/tmp/claude-0/-home-claude/0eface35-0138-50e1-ac1f-21dad35a4044/scratchpad/cv')
import numpy as np
from dataclasses import replace
from vidstab import StabConfig, OnlineStabilizer
from vidstab.synth import render_clip
cats=sys.argv[1].split(','); seeds=[int(x) for x in sys.argv[2].split(',')]; out=sys.argv[3]
data=[]
for c in cats:
    for sd in seeds:
        fr,idl,gt=render_clip(c,seed=sd,T=150)
        st=OnlineStabilizer(replace(StabConfig(),method='none'))
        for f in fr: st.process(f)
        data.append(dict(cat=c,seed=sd,C=np.array(st.history['C']),rel=np.array(st.history['reliable']),cut=np.array(st.history['cut']),
                         f=st.f,rh=st.rh,margin=st.margin,gt=gt))
        print(c,sd,flush=True)
pickle.dump(data,open(out,'wb'))
