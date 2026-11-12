import pickle,collections,re,json,math,numpy as np,pandas as pd
from scipy import stats
M=pickle.load(open('merged.pkl','rb'))
R={}
def scr(w):
    dev=len(re.findall(r'[ऀ-ॿ]',w)); lat=len(re.findall(r'[A-Za-z]',w))
    if dev and lat: return 'mixed'
    if dev: return 'deva'
    if lat: return 'latn'
    return 'none'
FUNC={'ADP','PRON','PRON_WH','DET','CONJ','PART','PART_NEG'}
CONT={'NOUN','PROPN','VERB','ADJ','ADV'}
sent_rows=[]; tok_rows=[]; junc_rows=[]; spans=[]
for si,r in enumerate(M):
    toks=r['toks']; pos=r['pos']
    n=len(toks); u=sum(1 for _,l in toks if l!='hi' and l!='en')
    nh=sum(1 for _,l in toks if l=='hi'); ne=sum(1 for _,l in toks if l=='en')
    cmi=100*(1-max(nh,ne)/(n-u)) if n>u else 0.0
    # hindi script
    hs=[scr(w) for w,l in toks if l=='hi' and scr(w) in('deva','latn','mixed')]
    if not hs: sclass='No Hindi tokens'
    else:
        dv=sum(s=='deva' for s in hs)/len(hs)
        sclass='Devanagari' if dv>=0.9 else ('Roman' if dv<=0.1 else 'Mixed-script')
    # language sequence
    seq=[(i,l) for i,(w,l) in enumerate(toks) if l in('hi','en')]
    sw=0
    for a,b in zip(seq,seq[1:]):
        if a[1]!=b[1]: sw+=1
        gap=b[0]-a[0]>1
        junc_rows.append(dict(sid=si,left=a[0],right=b[0],ll=a[1],rl=b[1],sw=int(a[1]!=b[1]),gap=int(gap),
            lp=pos[a[0]] if pos else None, rp=pos[b[0]] if pos else None, sclass=sclass))
    # spans of english
    cur=None;ln=0
    for _,l in seq+[(None,'END')]:
        if l==cur: ln+=1
        else:
            if cur=='en': spans.append(dict(sid=si,len=ln,sclass=sclass,mixed=nh>0))
            cur=l;ln=1
    en_scripts=collections.Counter(scr(w) for w,l in toks if l=='en')
    sent_rows.append(dict(sid=si,split=r['split'],n=n,u=u,nh=nh,ne=ne,cmi=cmi,sw=sw,nlang=len(seq),sclass=sclass,
        aligned=pos is not None,mixed=(nh>0 and ne>0),en_deva=en_scripts['deva'],en_latn=en_scripts['latn'],
        hi_latn=sum(1 for w,l in toks if l=='hi' and scr(w)=='latn'),mixtok=sum(1 for w,l in toks if l in('hi','en') and scr(w)=='mixed')))
    prev=None
    for i,(w,l) in enumerate(toks):
        if l in('hi','en'):
            tok_rows.append(dict(sid=si,i=i,w=w.lower().strip('.,:;!?()"\'|।'),lang=l,pos=pos[i] if pos else None,
                script=scr(w),sclass=sclass,swstart=int(prev is not None and prev!=l),first=int(prev is None)))
            prev=l
S=pd.DataFrame(sent_rows); T=pd.DataFrame(tok_rows); J=pd.DataFrame(junc_rows); SP=pd.DataFrame(spans)
S.to_pickle('S.pkl');T.to_pickle('T.pkl');J.to_pickle('J.pkl');SP.to_pickle('SP.pkl')
# ---- corpus description
d={}
d['sentences']=len(S); d['aligned_sentences']=int(S.aligned.sum())
d['tokens_total']=int(S.n.sum()); d['tokens_hi']=int(S.nh.sum()); d['tokens_en']=int(S['ne'].sum()); d['tokens_other']=int(S.u.sum())
d['mixed_sentences']=int(S.mixed.sum()); d['mono_hi']=int(((S.nh>0)&(S['ne']==0)).sum()); d['mono_en']=int(((S['ne']>0)&(S.nh==0)).sum())
d['cmi_mean_all']=S.cmi.mean(); d['cmi_mean_mixed']=S[S.mixed].cmi.mean(); d['cmi_median_mixed']=S[S.mixed].cmi.median(); d['cmi_sd_mixed']=S[S.mixed].cmi.std()
d['switches_total']=int(S.sw.sum()); d['switches_per_mixed_sent_mean']=S[S.mixed].sw.mean(); d['switches_per_mixed_sent_median']=S[S.mixed].sw.median()
d['iindex']=S.sw.sum()/(S.nlang-1).clip(lower=0).sum()
p=np.array([S.nh.sum(),S['ne'].sum()])/(S.nh.sum()+S['ne'].sum()); d['mindex']=(1-(p**2).sum())/((2-1)*(p**2).sum())
d['tokens_mean_len']=S.n.mean()
d['en_dir']=int(((J.sw==1)&(J.ll=='hi')).sum()); d['hi_dir']=int(((J.sw==1)&(J.ll=='en')).sum())
d['sclass_counts']=S.sclass.value_counts().to_dict()
d['split_counts']=S.split.value_counts().to_dict()
R['desc']=d
# token script x lang
R['lang_script']=pd.crosstab(T.lang,T.script).to_dict()
# ---- spans (insertion vs alternation)
SPm=SP[SP.mixed]
R['span_len']=SPm.len.clip(upper=4).value_counts().sort_index().to_dict(); R['span_n']=len(SPm)
R['span_len_by_script']={k:(g.len.clip(upper=4).value_counts(normalize=True).sort_index().to_dict(),len(g)) for k,g in SPm.groupby('sclass')}
json.dump(R,open('R1.json','w'),indent=1,default=float)
print(json.dumps(R,indent=1,default=float))
