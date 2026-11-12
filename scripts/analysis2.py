import pickle,collections,json,numpy as np,pandas as pd,re
from scipy import stats
S=pd.read_pickle('S.pkl');T=pd.read_pickle('T.pkl');J=pd.read_pickle('J.pkl')
R={}
FUNC={'ADP','PRON','PRON_WH','DET','CONJ','PART','PART_NEG'}
NOM={'NOUN','PROPN'}
mixed_ids=set(S[S.mixed & S.aligned].sid)
Ja=J[J.sid.isin(mixed_ids) & J.lp.notna() & J.rp.notna()].copy()
def jcat(r):
    lp,rp=r.lp,r.rp
    if r.gap: return 'Across punctuation/symbol'
    if lp=='CONJ' or rp=='CONJ': return 'Conjunction boundary'
    if lp in {'DET','ADJ','NUM','NOUN','PROPN'} and rp in NOM: return 'NP-internal (modifier/compound–noun)'
    if lp in {'NOUN','PROPN','PRON','NUM'} and rp=='ADP': return 'Noun–postposition'
    if lp=='ADP': return 'After postposition (PP boundary)'
    if lp in {'NOUN','PROPN','ADJ','PRON'} and rp=='VERB': return 'Nominal–verb (object/conjunct verb)'
    if lp=='VERB' and rp=='VERB': return 'Verb group internal'
    if lp=='VERB': return 'After verb (clause/VP boundary)'
    return 'Other'
Ja['cat']=Ja.apply(jcat,axis=1)
R['n_junctures']=len(Ja); R['n_switch_junctures']=int(Ja.sw.sum()); R['n_sent_junc']=Ja.sid.nunique()
# observed vs expected under uniform relocation of switches within sentence
k=Ja.groupby('sid').sw.transform('sum'); nj=Ja.groupby('sid').sw.transform('size')
Ja['w']=k/nj
obs=Ja.groupby('cat').sw.sum(); exp=Ja.groupby('cat').w.sum(); alln=Ja.groupby('cat').size()
# simulation
rng=np.random.default_rng(2026)
sids=Ja.sid.values; cats=Ja.cat.values; catlist=sorted(set(cats)); ci=np.array([catlist.index(c) for c in cats])
order=np.argsort(sids,kind='stable'); assert (np.diff(sids[order])>=0).all()
Ja=Ja.iloc[order].reset_index(drop=True); ci=ci[order]; sids=Ja.sid.values
ks=Ja.groupby('sid',sort=True).sw.sum().values; starts=np.r_[0,np.flatnonzero(np.diff(sids))+1]
sizes=np.diff(np.r_[starts,len(sids)])
NS=1000; sim=np.zeros((NS,len(catlist)))
grp=np.repeat(np.arange(len(starts)),sizes)
for s in range(NS):
    key=rng.random(len(sids))
    # rank within group
    o=np.lexsort((key,grp)); rank=np.empty(len(o),int); rank[o]=np.arange(len(o))-np.repeat(starts,sizes)
    chosen=rank<np.repeat(ks,sizes)
    sim[s]=np.bincount(ci[chosen],minlength=len(catlist))
tab=[]
for j,c in enumerate(catlist):
    o=int(obs[c]); e=sim[:,j].mean(); lo,hi=np.percentile(sim[:,j],[2.5,97.5])
    p=(np.sum(np.abs(sim[:,j]-e)>=abs(o-e))+1)/(NS+1)
    tab.append(dict(cat=c,junctures=int(alln[c]),switches=o,switch_rate=o/alln[c],share_obs=o/obs.sum(),expected=e,exp_lo=lo,exp_hi=hi,oe=o/e,p=p,analytic_exp=float(exp[c])))
JT=pd.DataFrame(tab).sort_values('oe',ascending=False); JT.to_csv('junctures.csv',index=False)
R['junctures']=JT.to_dict('records')
chi=stats.chisquare(obs[catlist].values, f_exp=exp[catlist].values*obs.sum()/exp.sum())
R['junc_chi']=dict(chi2=chi.statistic,df=len(catlist)-1,p=chi.pvalue)
# boundary vs internal grouping
B={'Across punctuation/symbol','Conjunction boundary','After postposition (PP boundary)','After verb (clause/VP boundary)'}
I={'NP-internal (modifier/compound–noun)','Noun–postposition','Verb group internal'}
for name,g in [('boundary',B),('internal',I)]:
    jj=[catlist.index(c) for c in g]
    o=sum(int(obs[c]) for c in g); se=sim[:,jj].sum(1)
    R['grp_'+name]=dict(obs=o,exp=se.mean(),lo=np.percentile(se,2.5),hi=np.percentile(se,97.5),oe=o/se.mean(),share=o/obs.sum())
# EC: directional detail for switches at Noun–postposition and Nominal–verb
sw=Ja[Ja.sw==1]
R['np_dir']=sw[sw.cat=='Noun–postposition'].groupby(['ll','rl']).size().rename(lambda x:x).to_dict()
R['nv_dir']=sw[sw.cat=='Nominal–verb (object/conjunct verb)'].groupby(['ll','rl']).size().rename(lambda x:x).to_dict()
R['np_internal_dir']=sw[sw.cat=='NP-internal (modifier/compound–noun)'].groupby(['ll','rl']).size().rename(lambda x:x).to_dict()
# EC non-equivalent junctures: en N + hi ADP ; en nominal + hi VERB ; en VERB + hi VERB(aux)
ne1=((sw.ll=='en')&(sw.rl=='hi')&(sw.cat=='Noun–postposition')).sum()
ne2=((sw.ll=='en')&(sw.rl=='hi')&(sw.cat=='Nominal–verb (object/conjunct verb)')).sum()
ne3=((sw.ll=='en')&(sw.rl=='hi')&(sw.cat=='Verb group internal')).sum()
R['ec']=dict(en_N_hi_P=int(ne1),en_Nom_hi_V=int(ne2),en_V_hi_V=int(ne3),total_switch=int(len(sw)),share=(ne1+ne2+ne3)/len(sw))
pickle.dump(Ja,open('Ja.pkl','wb'))
R={k:({'%s>%s'%kk:vv for kk,vv in v.items()} if isinstance(v,dict) and v and isinstance(next(iter(v)),tuple) else v) for k,v in R.items()}
json.dump(R,open('R2.json','w'),indent=1,default=float)
print(JT.to_string()); print({k:v for k,v in R.items() if k!='junctures'})
