import pickle,collections,re,unicodedata
def clean(w): return re.sub(r'[\s\u200b-\u200f\ufeff]','',unicodedata.normalize('NFC',w))
g=pickle.load(open('gold.pkl','rb'))
pos_by={r['sent']:r['toks'] for r in g['pos']}
stat=collections.Counter(); merged=[]
for r in g['lid']:
    stat['lid_sents']+=1
    p=pos_by.get(r['sent'])
    if p is None:
        merged.append(dict(r,pos=None)); continue
    stat['in_pos']+=1
    L=''.join(clean(w) for w,_ in r['toks']); P=''.join(clean(w) for w,_ in p)
    if L!=P: stat['char_mismatch']+=1; merged.append(dict(r,pos=None)); continue
    # char -> pos token index
    owner=[]
    for j,(w,t) in enumerate(p): owner+= [j]*len(clean(w))
    posl=[]; c=0
    for w,_ in r['toks']:
        n=len(clean(w))
        js=sorted(set(owner[c:c+n])); c+=n
        tags=[p[j][1] for j in js]
        non=[t for t in tags if t not in ('X',None)]
        posl.append(non[0] if non else (tags[0] if tags else None))
        if len(js)>1: stat['lid_tok_multi_pos']+=1
    stat['aligned']+=1
    merged.append(dict(r,pos=posl))
print(stat)
pickle.dump(merged,open('merged.pkl','wb'))
