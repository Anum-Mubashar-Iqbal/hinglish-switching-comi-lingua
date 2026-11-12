import pickle,collections,json,numpy as np,pandas as pd,re
from scipy import stats
S=pd.read_pickle('S.pkl');T=pd.read_pickle('T.pkl')
R={}
mixed_ids=set(S[S.mixed & S.aligned].sid)
Ta=T[T.sid.isin(mixed_ids)&T.pos.notna()].copy()
R['n_tok_mixed_aligned']=len(Ta); R['n_sent_mixed_aligned']=len(mixed_ids)
# --- Matrix language by last verb
ml={}
for sid,g in Ta.groupby('sid'):
    v=g[g.pos=='VERB']
    ml[sid]=v.iloc[-1].lang if len(v) else None
Ta['ml']=Ta.sid.map(ml)
mlc=pd.Series(ml).value_counts(dropna=False).to_dict()
R['ml_counts']={str(k):int(v) for k,v in mlc.items()}
# token majority as alternative
maj=Ta.groupby('sid').lang.agg(lambda x:(x=='hi').mean())
R['ml_agree_tokenmajority']=float(np.mean([ (ml[s]=='hi')==(maj[s]>0.5) for s in maj.index if ml[s] is not None]))
H=Ta[Ta.ml=='hi'].reset_index(drop=True)
# function-word language in Hindi-ML sentences
fw={}
for p in ['ADP','PART','PART_NEG','CONJ','DET','PRON','PRON_WH']:
    g=H[H.pos==p]; fw[p]=dict(n=len(g),en=int((g.lang=='en').sum()),pct_en=float((g.lang=='en').mean()*100))
R['func_lang_hiML']=fw
# English ADP in Hindi-ML: island or not
Hs=H.sort_values(['sid','i']).reset_index(drop=True)
Hs['prev_lang']=Hs.groupby('sid').lang.shift(1); Hs['next_lang']=Hs.groupby('sid').lang.shift(-1)
Hs['next_pos']=Hs.groupby('sid').pos.shift(-1); Hs['prev_pos']=Hs.groupby('sid').pos.shift(1)
ea=Hs[(Hs.pos=='ADP')&(Hs.lang=='en')]
R['en_adp_hiML']=dict(n=len(ea),island=int(((ea.prev_lang=='en')&(ea.next_lang=='en')).sum()),
   prep_before_hindi_nominal=int(((ea.next_lang=='hi')&(ea.next_pos.isin(['NOUN','PROPN','PRON']))).sum()),
   top=ea.w.value_counts().head(10).to_dict())
ex=Hs[(Hs.pos=='ADP')&(Hs.lang=='en')&(Hs.next_lang=='hi')&(Hs.next_pos.isin(['NOUN','PROPN','PRON']))]
R['en_prep_hi_examples_sids']=ex.sid.head(15).tolist()
# Hindi postposition after English noun (MOP-consistent)
R['en_noun_hi_adp']=int(((Hs.lang=='hi')&(Hs.pos=='ADP')&(Hs.prev_lang=='en')&(Hs.prev_pos.isin(['NOUN','PROPN']))).sum())
R['hi_adp_total_hiML']=int(((Hs.lang=='hi')&(Hs.pos=='ADP')).sum())
# light verbs
LV=re.compile(r'^(कर|किय|किए|किये|की$|हो|हु|दे|दि|ले|लि|kar|kiy|kie|ki$|ho|hu[aei]|de|diy|le|liy)')
Hs['next_w']=Hs.groupby('sid').w.shift(-1)
nxt_lv=(Hs.next_lang=='hi')&(Hs.next_pos=='VERB')&Hs.next_w.fillna('').str.match(LV)
env=Hs[(Hs.lang=='en')&(Hs.pos=='VERB')]
R['en_verbs_hiML']=dict(n=len(env),followed_by_hi_lightverb=int(nxt_lv[env.index].sum()),followed_by_hi_verb=int(((env.next_lang=='hi')&(env.next_pos=='VERB')).sum()),
   followed_by_en=int((env.next_lang=='en').sum()))
enn=Hs[(Hs.lang=='en')&(Hs.pos.isin(['NOUN','ADJ']))]
R['en_nounadj_lightverb']=int(nxt_lv[enn.index].sum())
R['lightverb_top']=Hs[nxt_lv&(Hs.lang=='en')].w.value_counts().head(15).to_dict()
# all hindi verbs final verb heads: proportion of verbs in English overall
R['verb_lang_all']=Ta[Ta.pos=='VERB'].lang.value_counts().to_dict()
pickle.dump(Hs,open('Hs.pkl','wb'))
# ---- RQ2 POS x language
def wilson(k,n,z=1.96):
    p=k/n; den=1+z*z/n; c=(p+z*z/(2*n))/den; h=z*math.sqrt(p*(1-p)/n+z*z/(4*n*n))/den; return c-h,c+h
import math
Tp=Ta[Ta.pos!='X']
ct=pd.crosstab(Tp.pos,Tp.lang)
chi=stats.chi2_contingency(ct)
R['pos_chi']=dict(chi2=chi[0],df=chi[2],p=chi[1],N=int(ct.values.sum()),cramer=math.sqrt(chi[0]/(ct.values.sum()*(min(ct.shape)-1))))
res=(ct-chi[3])/np.sqrt(chi[3])
rows=[]
for p in ct.index:
    k=int(ct.loc[p,'en']); n=int(ct.loc[p].sum()); lo,hi=wilson(k,n)
    rows.append(dict(pos=p,n=n,en=k,hi=int(ct.loc[p,'hi']),pct_en=100*k/n,lo=100*lo,hi_ci=100*hi,
        share_of_en=100*k/ct['en'].sum(),share_of_hi=100*ct.loc[p,'hi']/ct['hi'].sum(),resid_en=float(res.loc[p,'en'])))
PT=pd.DataFrame(rows).sort_values('pct_en',ascending=False); PT.to_csv('pos_table.csv',index=False); R['pos_table']=PT.to_dict('records')
# content vs function 2x2
Tp=Tp.assign(cls=np.where(Tp.pos.isin(['NOUN','PROPN','VERB','ADJ','ADV']),'content',np.where(Tp.pos=='NUM','num','function')))
c2=pd.crosstab(Tp[Tp.cls!='num'].cls,Tp[Tp.cls!='num'].lang)
ch2=stats.chi2_contingency(c2,correction=False)
OR=(c2.loc['content','en']*c2.loc['function','hi'])/(c2.loc['content','hi']*c2.loc['function','en'])
se=math.sqrt(sum(1/x for x in c2.values.ravel()))
R['cf']=dict(table=c2.to_dict(),chi2=ch2[0],p=ch2[1],OR=OR,OR_lo=math.exp(math.log(OR)-1.96*se),OR_hi=math.exp(math.log(OR)+1.96*se),
   pct_en_content=100*c2.loc['content','en']/c2.loc['content'].sum(),pct_en_function=100*c2.loc['function','en']/c2.loc['function'].sum(),
   phi=math.sqrt(ch2[0]/c2.values.sum()))
# noun vs verb (common nouns)
nv=pd.crosstab(Tp[Tp.pos.isin(['NOUN','VERB'])].pos,Tp[Tp.pos.isin(['NOUN','VERB'])].lang)
ORnv=(nv.loc['NOUN','en']*nv.loc['VERB','hi'])/(nv.loc['NOUN','hi']*nv.loc['VERB','en']); se=math.sqrt(sum(1/x for x in nv.values.ravel()))
R['noun_verb']=dict(OR=ORnv,lo=math.exp(math.log(ORnv)-1.96*se),hi=math.exp(math.log(ORnv)+1.96*se),chi2=stats.chi2_contingency(nv,correction=False)[0])
# switch-start tokens: POS distribution of token that begins a switch into English vs others
Tq=Tp[Tp['first']==0]
sw_en=Tq[(Tq.swstart==1)&(Tq.lang=='en')]
R['switch_into_en_pos']=(sw_en.pos.value_counts(normalize=True)*100).round(2).to_dict(); R['switch_into_en_n']=len(sw_en)
sw_hi=Tq[(Tq.swstart==1)&(Tq.lang=='hi')]
R['switch_into_hi_pos']=(sw_hi.pos.value_counts(normalize=True)*100).round(2).to_dict(); R['switch_into_hi_n']=len(sw_hi)
json.dump(R,open('R3.json','w'),indent=1,default=float)
print(json.dumps(R,indent=1,default=float,ensure_ascii=False))
