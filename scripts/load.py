import pandas as pd, ast, collections, pickle
U='data/'
MAP={'h':'hi','e':'en'}
def norm(s,kind):
    if not isinstance(s,str): return []
    p=ast.literal_eval(s); out=[]
    for e in p:
        if isinstance(e,dict):
            w=e.get('key',e.get('word')); t=e.get('value',e.get('entity'))
        else: w,t=e[0],e[1]
        out.append((w,MAP.get(t,t) if kind=='lid' else t))
    return out
data={}
for name,f,kind in [('lid_train','LID_train.csv','lid'),('lid_test','LID_test.csv','lid'),('pos_train','POS_train.csv','pos'),('pos_test','POS_test.csv','pos')]:
    df=pd.read_csv(U+f)
    rows=[]
    for _,r in df.iterrows():
        rows.append({'sent':r['Sentences'],'pred':norm(r['Predicted tags'],kind),'a1':norm(r['Annotated by: Annotator 1'],kind),'a2':norm(r['Annotated by: Annotator 2'],kind),'a3':norm(r['Annotated by: Annotator 3'],kind)})
    data[name]=rows
pickle.dump(data,open('data.pkl','wb'))
for name,rows in data.items():
    print('==',name,len(rows))
    vc={k:collections.Counter() for k in ['pred','a1','a2','a3']}
    lens=collections.Counter(); ident=collections.Counter()
    for r in rows:
        lens[len({len(r[k]) for k in vc if r[k]})]+=1
        for k in vc:
            for w,t in r[k]: vc[k][t]+=1
        for a,b in [('a1','a2'),('a1','a3'),('a2','a3'),('pred','a1'),('pred','a2'),('pred','a3')]:
            ident[a+b]+= [t for _,t in r[a]]==[t for _,t in r[b]]
    print('distinct lengths per row',lens); print('identical',ident)
    for k in vc: print(k,vc[k].most_common(40))
