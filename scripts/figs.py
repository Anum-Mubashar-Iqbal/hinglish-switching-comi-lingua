import os,json,pandas as pd,numpy as np,matplotlib
matplotlib.use('Agg'); import matplotlib.pyplot as plt
os.makedirs('fig',exist_ok=True)
plt.rcParams.update({'font.family':'DejaVu Sans','font.size':10,'axes.edgecolor':'#52514e','axes.labelcolor':'#0b0b0b','xtick.color':'#52514e','ytick.color':'#52514e','axes.spines.top':False,'axes.spines.right':False})
B='#2a78d6'; O='#eb6834'; G='#9a9a96'
NAMES={'NOUN':'Common noun','PROPN':'Proper noun','ADJ':'Adjective','NUM':'Numeral','PART':'Particle','DET':'Determiner','CONJ':'Conjunction','VERB':'Verb','PART_NEG':'Negative particle','PRON':'Pronoun','ADP':'Postposition/preposition','ADV':'Adverb','PRON_WH':'Wh-pronoun'}
# Fig1
P=pd.read_csv('pos_table.csv').sort_values('pct_en')
fig,ax=plt.subplots(figsize=(6.5,4.2),dpi=300)
y=np.arange(len(P)); ax.barh(y,P.pct_en,color=B,height=0.6)
ax.errorbar(P.pct_en,y,xerr=[P.pct_en-P.lo,P.hi_ci-P.pct_en],fmt='none',ecolor='#0b0b0b',elinewidth=0.8,capsize=2)
ax.set_yticks(y); ax.set_yticklabels([NAMES[p] for p in P.pos]); ax.set_xlabel('English tokens (% of all tokens in the category)')
for yi,v,h in zip(y,P.pct_en,P.hi_ci): ax.text(h+1.2,yi,f'{v:.1f}',va='center',fontsize=8,color='#52514e')
ax.set_xlim(0,70); ax.grid(axis='x',color='#e6e6e3',lw=0.6); ax.set_axisbelow(True)
plt.tight_layout(); plt.savefig('fig/fig1_pos.png'); plt.close()
# Fig2
J=pd.read_csv('junctures.csv').sort_values('oe')
fig,ax=plt.subplots(figsize=(6.5,3.9),dpi=300)
y=np.arange(len(J))
ax.axvline(1,color=G,lw=1,ls='--')
ax.errorbar(J.oe,y,xerr=[J.oe-J.switches/J.exp_hi,J.switches/J.exp_lo-J.oe],fmt='o',color=B,ms=6,elinewidth=1,capsize=2)
lab={'Across punctuation/symbol':'Across punctuation or symbol','Noun–postposition':'Noun + postposition','Nominal–verb (object/conjunct verb)':'Nominal + verb','Conjunction boundary':'Next to a conjunction','Other':'Other','NP-internal (modifier/compound–noun)':'Inside a noun phrase','After verb (clause/VP boundary)':'After a verb (clause edge)','After postposition (PP boundary)':'After a postposition (PP edge)','Verb group internal':'Inside a verb group'}
ax.set_yticks(y); ax.set_yticklabels([lab[c] for c in J.cat])
for yi,v in zip(y,J.oe): ax.text(v,yi+0.28,f'{v:.2f}',ha='center',fontsize=8,color='#52514e')
ax.set_xlabel('Observed ÷ expected switches (1 = chance)'); ax.set_xlim(0,1.9)
ax.grid(axis='x',color='#e6e6e3',lw=0.6); ax.set_axisbelow(True)
plt.tight_layout(); plt.savefig('fig/fig2_junctures.png'); plt.close()
# Fig3
R=json.load(open('R4.json'))
d=pd.DataFrame(R['pos_en_by_script']); n=pd.DataFrame(R['pos_n_by_script'])
d=d.drop(index=['NUM']).sort_values('Devanagari')
NAMES2=dict(NAMES,PART='Particle (incl. negation)',PRON='Pronoun (incl. wh-)')
fig,ax=plt.subplots(figsize=(6.5,4.0),dpi=300)
y=np.arange(len(d))
for yi,(a,b) in zip(y,zip(d.Devanagari,d.Roman)): ax.plot([a,b],[yi,yi],color='#d0d0cc',lw=1.5,zorder=1)
ax.scatter(d.Devanagari,y,color=B,s=36,zorder=2,label='Devanagari Hinglish')
ax.scatter(d.Roman,y,color=O,s=36,marker='s',zorder=2,label='Roman Hinglish')
ax.set_yticks(y); ax.set_yticklabels([NAMES2[p] for p in d.index]); ax.set_xlabel('English tokens (% of the category)')
ax.legend(frameon=False,loc='lower right'); ax.grid(axis='x',color='#e6e6e3',lw=0.6); ax.set_axisbelow(True); ax.set_xlim(0,65)
plt.tight_layout(); plt.savefig('fig/fig3_script_pos.png'); plt.close()
# Fig4
R1=json.load(open('R1.json'))
sl=R1['span_len_by_script']
cats=['1','2','3','4']; lbl=['1 word','2 words','3 words','4+ words']
fig,ax=plt.subplots(figsize=(6.5,3.2),dpi=300)
x=np.arange(4); w=0.36
dv=[100*sl['Devanagari'][0][c] for c in cats]; rm=[100*sl['Roman'][0][c] for c in cats]
ax.bar(x-w/2-0.01,dv,w,color=B,label=f"Devanagari Hinglish (n = {sl['Devanagari'][1]:,} stretches)")
ax.bar(x+w/2+0.01,rm,w,color=O,label=f"Roman Hinglish (n = {sl['Roman'][1]:,} stretches)")
for xi,a,b in zip(x,dv,rm):
    ax.text(xi-w/2,a+1,f'{a:.1f}',ha='center',fontsize=8,color='#52514e'); ax.text(xi+w/2,b+1,f'{b:.1f}',ha='center',fontsize=8,color='#52514e')
ax.set_xticks(x); ax.set_xticklabels(lbl); ax.set_ylabel('% of English stretches'); ax.set_ylim(0,80)
ax.legend(frameon=False); ax.grid(axis='y',color='#e6e6e3',lw=0.6); ax.set_axisbelow(True)
plt.tight_layout(); plt.savefig('fig/fig4_spans.png'); plt.close()
print('ok')
