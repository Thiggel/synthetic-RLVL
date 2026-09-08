"""Paired document bootstrap over the clean sampled readout (eosfix root).
pass@16 per doc = any of the 16 samples correct; pass@1 = mean over samples.
Pooled over depths 5-25 (1000 docs). 4000 resamples, same indices for all arms."""
import json,os,sys,random,itertools
P=sys.argv[1] if len(sys.argv)>1 else "/home/vault/c107fa/c107fa12/synthetic-RLVL/lm_eval_results/qwen25_longwin_passk_20260908_eosfix"
ARMS=["control","longdoc","logic_band25","nl_exact_band25","condensed_logic_band25"]
D=[5,10,15,20,25]
p16={}; p1={}
for a in ARMS:
    v16=[]; v1=[]
    ok=True
    for d in D:
        f=f"{P}/{a}/samples_synthrlvl_deduction_bp_cot_d{d}.jsonl"
        if not os.path.exists(f): ok=False; break
        for line in open(f):
            r=json.loads(line); c=[float(s["correct"]) for s in r["samples"]]
            v16.append(float(max(c))); v1.append(sum(c)/len(c))
    if ok: p16[a]=v16; p1[a]=v1
arms=[a for a in ARMS if a in p16]; n=len(p16[arms[0]])
print("arms:",arms,"n docs:",n)
random.seed(20260908); R=4000
idx=[[random.randrange(n) for _ in range(n)] for _ in range(R)]
def boot(v):  # returns list of R means
    return [sum(v[i] for i in ix)/n for ix in idx]
B16={a:boot(p16[a]) for a in arms}; B1={a:boot(p1[a]) for a in arms}
def ci(x): x=sorted(x); return x[int(0.025*R)],x[int(0.975*R)]
print("metric,arm,mean")
for a in arms: print(f"pass16,{a},{sum(p16[a])/n:.4f}"); 
for a in arms: print(f"pass1,{a},{sum(p1[a])/n:.4f}")
print("metric,contrast,diff,ci_lo,ci_hi,p_le0")
for name,B,V in [("pass16",B16,p16),("pass1",B1,p1)]:
    for a,b in itertools.combinations(arms,2):
        diff=sum(V[a])/n-sum(V[b])/n; dd=[x-y for x,y in zip(B[a],B[b])]; lo,hi=ci(dd)
        print(f"{name},{a}-{b},{diff:+.4f},{lo:+.4f},{hi:+.4f},{sum(1 for x in dd if x<=0)/R:.4f}")
