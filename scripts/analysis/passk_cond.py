import json, random
import os
# Root is overridable so the identical audited analysis can be re-run against
# other pass@k bundles (e.g. the document-preserving rerun). Default unchanged.
R=os.environ.get('PASSK_ROOT','/home/vault/c107fa/c107fa12/synthetic-RLVL/lm_eval_results/qwen25_dolmino_post_sft_passk_20260806')
CONDS=['control','logic','nl_exact']
TASKS=['synthrlvl_longbench_2wikimqa_tagged','synthrlvl_longbench_hotpotqa_tagged','synthrlvl_longbench_musique_tagged']
random.seed(20260808)
def load(cond,task):
    recs={}
    with open(f'{R}/{cond}/samples_{task}.jsonl') as f:
        for line in f: r=json.loads(line); recs[r['doc_id']]=r
    return recs
# per-draw conditional accuracy: among samples with tag_found, EM rate (strict)
for task in TASKS:
    recs={c:load(c,task) for c in CONDS}
    print('==',task)
    for c in CONDS:
        alls=[s for r in recs[c].values() for s in r['samples']]
        tagged=[s for s in alls if s['tag_found']]
        em_c=sum(s['em'] for s in tagged)/len(tagged)
        print(f'  {c:9s} tag_rate={len(tagged)/len(alls):.3f}  EM|tag={em_c:.4f}  n_tag={len(tagged)}')
    # paired-by-doc bootstrap on conditional EM (doc-level mean over tagged samples, docs with >=1 tagged sample in both)
    for a,b in [('logic','control'),('logic','nl_exact'),('nl_exact','control')]:
        ids=[i for i in recs[a] if i in recs[b]]
        def docstat(r):
            t=[s['em'] for s in r['samples'] if s['tag_found']]
            return (sum(t)/len(t)) if t else None
        pairs=[(docstat(recs[a][i]),docstat(recs[b][i])) for i in ids]
        pairs=[(x,y) for x,y in pairs if x is not None and y is not None]
        obs=sum(x-y for x,y in pairs)/len(pairs)
        ds=[]
        for _ in range(10000):
            idx=[random.randrange(len(pairs)) for _ in pairs]
            ds.append(sum(pairs[j][0]-pairs[j][1] for j in idx)/len(pairs))
        ds.sort(); lo,hi=ds[250],ds[9750]
        sig='*' if lo>0 or hi<0 else ' '
        print(f'    EM|tag {a}-{b}: {obs:+.4f} [{lo:+.4f},{hi:+.4f}]{sig} (n_docs={len(pairs)})')
