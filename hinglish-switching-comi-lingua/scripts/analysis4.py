import pickle,collections,json,math,re,numpy as np,pandas as pd
from scipy import stats
import statsmodels.formula.api as smf
S=pd.read_pickle('S.pkl');T=pd.read_pickle('T.pkl')
R={}
mixed_ids=set(S[S.mixed & S.aligned].sid)
Ta=T[T.sid.isin(mixed_ids)&T.pos.notna()&(T.pos!='X')].copy()
freq=T.w.value_counts()
Ta['logf']=np.log10(Ta.w.map(freq))
L=S.set_index('sid').nlang; Ta['loglen']=np.log2(Ta.sid.map(L))
Ta=Ta[Ta.sclass.isin(['Devanagari','Roman','Mixed-script'])]
Ta['y']=(Ta.lang=='en').astype(int)
Ta['pos2']=Ta.pos.replace({'PRON_WH':'PRON','PART_NEG':'PART'})
Ta['logf_c']=Ta.logf-Ta.logf.mean(); Ta['loglen_c']=Ta.loglen-Ta.loglen.mean()
m=smf.logit("y ~ C(pos2, Treatment('NOUN')) + logf_c + C(sclass, Treatment('Devanagari')) + loglen_c",data=Ta).fit(disp=0,cov_type='cluster',cov_kwds={'groups':Ta.sid},maxiter=200)
ci=m.conf_int()
tab=pd.DataFrame({'b':m.params,'se':m.bse,'z':m.tvalues,'p':m.pvalues,'OR':np.exp(m.params),'lo':np.exp(ci[0]),'hi':np.exp(ci[1])})
tab.to_csv('logit.csv'); print(tab)
null=smf.logit("y ~ 1",data=Ta).fit(disp=0)
R['logit']=dict(N=int(m.nobs),clusters=int(Ta.sid.nunique()),llf=m.llf,llnull=null.llf,mcfadden=1-m.llf/null.llf,aic=m.aic,table=tab.reset_index().rename(columns={'index':'term'}).to_dict('records'))
# model without frequency for comparison of POS effect robustness
m0=smf.logit("y ~ C(pos2, Treatment('NOUN')) + C(sclass, Treatment('Devanagari')) + loglen_c",data=Ta).fit(disp=0,cov_type='cluster',cov_kwds={'groups':Ta.sid})
R['logit_nofreq_OR']=np.exp(m0.params).to_dict(); R['logit_nofreq_mcf']=1-m0.llf/null.llf
# interaction: does POS effect differ by script (content vs function x script)
Ta['content']=Ta.pos2.isin(['NOUN','PROPN','VERB','ADJ','ADV']).astype(int)
mi=smf.logit("y ~ content*C(sclass, Treatment('Devanagari')) + logf_c + loglen_c",data=Ta).fit(disp=0,cov_type='cluster',cov_kwds={'groups':Ta.sid})
R['logit_inter']=pd.DataFrame({'OR':np.exp(mi.params),'p':mi.pvalues,'lo':np.exp(mi.conf_int()[0]),'hi':np.exp(mi.conf_int()[1])}).reset_index().to_dict('records')
# POS english share by script
R['pos_en_by_script']=(Ta[Ta.sclass!='Mixed-script'].groupby(['pos2','sclass']).y.mean()*100).unstack().to_dict()
R['pos_n_by_script']=Ta[Ta.sclass!='Mixed-script'].groupby(['pos2','sclass']).size().unstack().to_dict()
# ---- RQ3 sentence-level
Sm=S[S.mixed & S.sclass.isin(['Devanagari','Roman','Mixed-script'])].copy()
Sm['swrate']=Sm.sw/(Sm.nlang-1)
Sm['en_share']=100*Sm['ne']/(Sm.nh+Sm['ne'])
Sm['en_deva_pct']=np.where(Sm['ne']>0,100*Sm.en_deva/Sm['ne'],np.nan)
Sm['mixtok_per1000']=1000*Sm.mixtok/Sm.nlang
desc={}
for v in ['nlang','cmi','sw','swrate','en_share','en_deva_pct','mixtok_per1000']:
    g={k:Sm[Sm.sclass==k][v].dropna() for k in ['Devanagari','Roman','Mixed-script']}
    kw=stats.kruskal(*g.values())
    mw=stats.mannwhitneyu(g['Devanagari'],g['Roman'])
    rb=1-2*mw.statistic/(len(g['Devanagari'])*len(g['Roman']))  # rank biserial (positive: Roman higher)
    desc[v]=dict({k:dict(mean=x.mean(),sd=x.std(),median=x.median(),n=len(x)) for k,x in g.items()},KW_H=kw.statistic,KW_p=kw.pvalue,MW_U=mw.statistic,MW_p=mw.pvalue,r_rb=rb)
R['script_sent']=desc
# token-level pooled indicators by script
for k in ['Devanagari','Roman','Mixed-script']:
    g=Sm[Sm.sclass==k]
    R.setdefault('script_pooled',{})[k]=dict(sent=len(g),tokens=int(g.nlang.sum()),en=int(g['ne'].sum()),sw=int(g.sw.sum()),iindex=g.sw.sum()/(g.nlang-1).sum(),
        en_pct=100*g['ne'].sum()/g.nlang.sum(),en_deva=int(g.en_deva.sum()),en_latn=int(g.en_latn.sum()),hi_latn=int(g.hi_latn.sum()),mixtok=int(g.mixtok.sum()))
# switch occurrence chi-square Dev vs Roman (junctures switch vs not)
a=R['script_pooled']['Devanagari']; b=R['script_pooled']['Roman']
tab2=np.array([[a['sw'],a['tokens']-len(Sm[Sm.sclass=='Devanagari'])-a['sw']],[b['sw'],b['tokens']-len(Sm[Sm.sclass=='Roman'])-b['sw']]])
c=stats.chi2_contingency(tab2,correction=False); ORs=(tab2[1,0]*tab2[0,1])/(tab2[1,1]*tab2[0,0])
R['script_switch_chi']=dict(table=tab2.tolist(),chi2=c[0],p=c[1],OR_roman_vs_deva=ORs)
# single-word insertion share by script
SP=pd.read_pickle('SP.pkl'); SP=SP[SP.mixed & SP.sclass.isin(['Devanagari','Roman'])]
ct=pd.crosstab(SP.sclass,SP.len==1); c=stats.chi2_contingency(ct,correction=False)
R['span_single_chi']=dict(table=ct.to_dict(),chi2=c[0],p=c[1],cramer=math.sqrt(c[0]/ct.values.sum()))
R['span_mean']=SP.groupby('sclass').len.agg(['mean','median','count']).to_dict()
json.dump(R,open('R4.json','w'),indent=1,default=float)
print(json.dumps({k:v for k,v in R.items() if k not in('logit','logit_inter')},indent=1,default=float))
print(pd.DataFrame(R['logit_inter']))
print('mcfadden',R['logit']['mcfadden'],R['logit']['N'],R['logit']['clusters'])
