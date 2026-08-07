import json, random
R='/home/vault/c107fa/c107fa12/synthetic-RLVL/lm_eval_results/qwen25_dolmino_post_sft_passk_20260806'
CONDS=['control','logic','nl_exact']
TASKS=['synthrlvl_longbench_2wikimqa_tagged','synthrlvl_longbench_hotpotqa_tagged','synthrlvl_longbench_musique_tagged']
random.seed(20260809)
def degenerate(s):
    e=(s.get('extracted') or '').strip()
    return e in ('','...','[answer]','<answer>') or e.startswith('...')
def load(cond,task):
    recs={}
    with open(f'{R}/{cond}/samples_{task}.jsonl') as f:
        for line in f: r=json.loads(line); recs[r['doc_id']]=r
    return recs
for task in TASKS:
    recs={c:load(c,task) for c in CONDS}
    print('==',task)
    for c in CONDS:
        alls=[s for r in recs[c].values() for s in r['samples']]
        good=[s for s in alls if s['tag_found'] and not degenerate(s)]
        deg=sum(1 for s in alls if s['tag_found'] and degenerate(s))
        print(f'  {c:9s} nondegen_rate={len(good)/len(alls):.3f} degen_tagged={deg}  EM|good={sum(s["em"] for s in good)/len(good):.4f}')
    for a,b in [('logic','control'),('logic','nl_exact'),('nl_exact','control')]:
        def docstat(r):
            t=[s['em'] for s in r['samples'] if s['tag_found'] and not degenerate(s)]
            return (sum(t)/len(t)) if t else None
        pairs=[(docstat(recs[a][i]),docstat(recs[b][i])) for i in recs[a] if i in recs[b]]
        pairs=[(x,y) for x,y in pairs if x is not None and y is not None]
        obs=sum(x-y for x,y in pairs)/len(pairs)
        ds=[]
        for _ in range(10000):
            idx=[random.randrange(len(pairs)) for _ in pairs]
            ds.append(sum(pairs[j][0]-pairs[j][1] for j in idx)/len(pairs))
        ds.sort(); lo,hi=ds[250],ds[9750]
        sig='*' if lo>0 or hi<0 else ' '
        print(f'    EM|good {a}-{b}: {obs:+.4f} [{lo:+.4f},{hi:+.4f}]{sig} (n_docs={len(pairs)})')
