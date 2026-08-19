import json, glob, random, sys
import os
# Root is overridable so the identical audited analysis can be re-run against
# other pass@k bundles (e.g. the document-preserving rerun). Default unchanged.
R=os.environ.get('PASSK_ROOT','/home/vault/c107fa/c107fa12/synthetic-RLVL/lm_eval_results/qwen25_dolmino_post_sft_passk_20260806')
CONDS=['control','logic','nl_exact']
TASKS=['gsm8k','hendrycks_math500','synthrlvl_longbench_2wikimqa_tagged','synthrlvl_longbench_hotpotqa_tagged','synthrlvl_longbench_musique_tagged']
random.seed(20260807)
def load(cond,task):
    recs={}
    with open(f'{R}/{cond}/samples_{task}.jsonl') as f:
        for line in f:
            r=json.loads(line)
            recs[r['doc_id']]=r
    return recs
def stat_pass16(r,field):
    return 1.0 if any(s[field] for s in r['samples']) else 0.0
def stat_pass1(r,field):
    ss=r['samples']; return sum(1.0 if s[field] else 0.0 for s in ss)/len(ss)
def boot(recsA,recsB,fn,n=10000):
    ids=sorted(set(recsA)&set(recsB))
    dA=[fn(recsA[i]) for i in ids]; dB=[fn(recsB[i]) for i in ids]
    obs=sum(a-b for a,b in zip(dA,dB))/len(ids)
    deltas=[]
    for _ in range(n):
        idx=[random.randrange(len(ids)) for _ in ids]
        deltas.append(sum(dA[j]-dB[j] for j in idx)/len(ids))
    deltas.sort()
    lo,hi=deltas[int(0.025*n)],deltas[int(0.975*n)]
    return obs,lo,hi
for task in TASKS:
    recs={c:load(c,task) for c in CONDS}
    tagged='tagged' in task
    print(f'== {task} (n={len(recs["control"])})')
    fields=[('correct','strict')]
    if tagged: fields.append(('em_fallback','fallback'))
    for field,name in fields:
        for a,b in [('logic','control'),('logic','nl_exact')]:
            o16,l16,h16=boot(recs[a],recs[b],lambda r:stat_pass16(r,field))
            o1,l1,h1=boot(recs[a],recs[b],lambda r:stat_pass1(r,field))
            sig16='*' if l16>0 or h16<0 else ' '
            sig1='*' if l1>0 or h1<0 else ' '
            print(f'  {name:8s} {a:>8s}-{b:8s}  pass@16 {o16:+.4f} [{l16:+.4f},{h16:+.4f}]{sig16}  pass@1 {o1:+.4f} [{l1:+.4f},{h1:+.4f}]{sig1}')
