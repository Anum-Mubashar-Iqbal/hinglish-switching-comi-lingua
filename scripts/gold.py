import pickle,collections,re,numpy as np
d=pickle.load(open('data.pkl','rb'))
def fleiss(rows,cats):
    # rows: list of lists of 3 labels
    idx={c:i for i,c in enumerate(cats)}
    M=np.zeros((len(rows),len(cats)))
    for i,r in enumerate(rows):
        for l in r: M[i,idx[l]]+=1
    n=3; P=(np.sum(M*M,1)-n)/(n*(n-1)); Pbar=P.mean()
    p=M.sum(0)/M.sum(); Pe=(p*p).sum()
    return (Pbar-Pe)/(1-Pe)
out={}
for task in ['lid','pos']:
    rows=d[task+'_train']+d[task+'_test']
    split=['train']*len(d[task+'_train'])+['test']*len(d[task+'_test'])
    seen=set(); recs=[]; triples=[]; stat=collections.Counter()
    for r,sp in zip(rows,split):
        stat['rows']+=1
        if r['sent'] in seen: stat['dup']+=1; continue
        seen.add(r['sent'])
        A=[r['a1'],r['a2'],r['a3']]
        if len({len(a) for a in A})!=1 or len(A[0])==0: stat['len_mismatch_or_empty']+=1; continue
        if any([w for w,_ in A[0]]!=[w for w,_ in a] for a in A[1:]): stat['word_mismatch']+=1; continue
        toks=[];
        for i,(w,_) in enumerate(A[0]):
            labs=[a[i][1] for a in A]
            if task=='lid': labs=['ot' if l=='u' else l for l in labs]
            triples.append(labs)
            c=collections.Counter(labs).most_common()
            g=c[0][0] if c[0][1]>=2 else None
            stat['tok']+=1; stat['unan' if c[0][1]==3 else ('maj' if c[0][1]==2 else 'nomaj')]+=1
            toks.append((w,g))
        recs.append({'sent':r['sent'],'split':sp,'toks':toks})
    cats=sorted({l for t in triples for l in t})
    k=fleiss(triples,cats)
    print(task,dict(stat),'kappa',round(k,4),cats,'sents kept',len(recs))
    out[task]=recs; out[task+'_kappa']=k; out[task+'_stat']=dict(stat)
pickle.dump(out,open('gold.pkl','wb'))
