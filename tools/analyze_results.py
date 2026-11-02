"""Аналіз per_clip.csv: середні за методами/категоріями, парні різниці з bootstrap-ДІ, порівняння за однакової площі (iso-area).
Запуск: python3 tools/analyze_results.py results_v3/per_clip.csv analysis.json"""
import sys, pandas as pd, numpy as np, json
from scipy.stats import wilcoxon
d=pd.read_csv(sys.argv[1] if len(sys.argv)>1 else 'results_v3/per_clip.csv').rename(columns={'clip':'cid'})
gt=d[d.category!='real_shaky']; sh=d[d.category=='real_shaky']
rng=np.random.default_rng(0)
M=['none','MA-9','Kalman (fixed crop)','Final (ours)','-adaptive crop','-ECC fallback','-soft weight','fixed k=0.02','fixed k=0.03','fixed k=0.04','fixed k=0.06','fixed k=0.08','Gauss offline (ref)']
out={}
# overall
ov={}
for m in M:
    g=gt[gt.method==m]
    ov[m]={k:float(g[k].mean()) for k in ['path_err_px','rot_err_deg','jitter_px','psnr_gt','itf','crop_area','zoom_jitter','fps']}
    ov[m]['n']=int(len(g))
out['overall']=ov
# percat
cats=['low','medium','high','walking','impulse','lowtex','occlusion','blur_noise','fastpan','scenecut','real_low','real_medium','real_high','real_walking']
pc={}
for m in ['none','MA-9','Kalman (fixed crop)','Final (ours)','-ECC fallback','Gauss offline (ref)']:
    pc[m]={c:{k:float(gt[(gt.method==m)&(gt.category==c)][k].mean()) for k in ['path_err_px','jitter_px','crop_area','itf']} for c in cats}
out['percat']=pc
# fps min per category (Final)
out['fps_cat']={c:float(gt[(gt.method=='Final (ours)')&(gt.category==c)].fps.mean()) for c in cats}
out['fps_min_clip']=float(d[d.method=='Final (ours)'].fps.min())
# paired
def pair(a,b,met,sub=None,excl=None):
    g=gt
    if sub is not None: g=g[g.category.isin(sub)]
    if excl is not None: g=g[~g.category.isin(excl)]
    x=g[g.method==a].set_index('cid')[met]; y=g[g.method==b].set_index('cid')[met].reindex(x.index); diff=(x-y).values
    bs=[rng.choice(diff,len(diff)).mean() for _ in range(5000)]
    try: p=float(wilcoxon(x,y).pvalue)
    except Exception: p=float('nan')
    return dict(mean=float(diff.mean()),lo=float(np.percentile(bs,2.5)),hi=float(np.percentile(bs,97.5)),p=p,n=int(len(diff)))
pr={}
for b in ['none','MA-9','Kalman (fixed crop)','-adaptive crop','-ECC fallback','-soft weight','Gauss offline (ref)']:
    for met in ['path_err_px','jitter_px','itf','crop_area']:
        pr[f'{b}|{met}']=pair('Final (ours)',b,met)
for met in ['path_err_px','jitter_px','itf']:
    pr[f'KalmanFixed_vs_MA9|{met}']=pair('Kalman (fixed crop)','MA-9',met)
nolowtex=[c for c in cats if c!='lowtex']
for met in ['path_err_px','jitter_px']:
    pr[f'Final_vs_KalmanFixed_nolowtex|{met}']=pair('Final (ours)','Kalman (fixed crop)',met,excl=['lowtex'])
    pr[f'Final_vs_noECC_lowtex|{met}']=pair('Final (ours)','-ECC fallback',met,sub=['lowtex'])
    pr[f'Final_vs_none_lowtex|{met}']=pair('Final (ours)','none',met,sub=['lowtex'])
out['paired']=pr
# global fixed curve + bootstrap iso-area
fixed=['fixed k=0.02','fixed k=0.03','fixed k=0.04','fixed k=0.06','fixed k=0.08']
P=gt.pivot_table(index='cid',columns='method',values=['jitter_px','path_err_px','crop_area','itf'])
def iso(P,mets=('jitter_px','path_err_px','itf')):
    s=P.mean(); a=s[('crop_area','Final (ours)')]; fa=np.array([s[('crop_area',m)] for m in fixed]); o=np.argsort(fa)
    r={'area':float(a)}
    for met in mets:
        fv=np.array([s[(met,m)] for m in fixed]); r[met+'_ref']=float(np.interp(a,fa[o],fv[o])); r[met+'_final']=float(s[(met,'Final (ours)')])
    return r
base=iso(P); B=[]
cl=P.index.values
for _ in range(3000):
    idx=rng.choice(len(cl),len(cl)); r=iso(P.iloc[idx]); B.append([r[k+'_final']-r[k+'_ref'] for k in ('jitter_px','path_err_px','itf')])
B=np.array(B)
base['ci']={k:[float(np.percentile(B[:,i],2.5)),float(np.percentile(B[:,i],97.5))] for i,k in enumerate(('jitter_px','path_err_px','itf'))}
out['iso_global']=base
# per-clip oracle iso-area
rows=[]
for cid,gp in gt.groupby('cid'):
    f=gp[gp.method.isin(fixed)].sort_values('crop_area'); fin=gp[gp.method=='Final (ours)'].iloc[0]
    a=fin.crop_area
    inr = f.crop_area.min()<=a<=f.crop_area.max()
    rows.append(dict(cid=cid,cat=fin.category,area=a,inrange=inr,**{k:fin[k]-np.interp(a,f.crop_area,f[k]) for k in ('jitter_px','path_err_px','itf')}))
R=pd.DataFrame(rows); R.to_csv('iso_clip.csv',index=False)
Rin=R[R.inrange]
def bs(x):
    x=np.asarray(x); b=[rng.choice(x,len(x)).mean() for _ in range(5000)]; return [float(x.mean()),float(np.percentile(b,2.5)),float(np.percentile(b,97.5))]
out['iso_clip']={'n_in':int(len(Rin)),'n':int(len(R)),**{k:bs(Rin[k]) for k in ('jitter_px','path_err_px','itf')}}
out['iso_clip_percat']={c:{k:float(g[k].mean()) for k in ('jitter_px','path_err_px','itf')}|{'n':int(len(g)),'area':float(g.area.mean())} for c,g in Rin.groupby('cat')}
# kappa-range diagnostics: fraction of clips where Final area outside fixed range
out['iso_clip']['out_of_range_cats']=R[~R.inrange].groupby('cat').size().to_dict()
# shaky
shk={}
for cid,g in sh.groupby('cid'):
    shk[cid]={m:{k:(None if pd.isna(r[k]) else float(r[k])) for k in ('itf','jit_in','jit_out','crop_area')} for m,r in g.set_index('method').iterrows() if m in M}
out['shaky']=shk
# reliability
out['reliable']={m:float(gt[gt.method==m].reliable_frac.mean()) for m in ['Final (ours)','-ECC fallback']}
out['reliable_lowtex']={m:float(gt[(gt.method==m)&(gt.category=='lowtex')].reliable_frac.mean()) for m in ['Final (ours)','-ECC fallback']}
json.dump(out,open(sys.argv[2] if len(sys.argv)>2 else 'analysis.json','w'),indent=1)
print(json.dumps({k:out[k] for k in ['iso_global','iso_clip','iso_clip_percat','reliable','reliable_lowtex','fps_min_clip']},indent=1))
print(json.dumps(out['shaky'],indent=0)[:3500])
for k,v in out['paired'].items(): print(k,{a:round(b,3) for a,b in v.items()})
print({m:{k:round(v,3) for k,v in ov[m].items()} for m in ['none','MA-9','Kalman (fixed crop)','Final (ours)','Gauss offline (ref)']})
