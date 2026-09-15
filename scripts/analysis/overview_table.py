import json,glob,os,re,collections,statistics as st,sys
ROOT="/home/vault/c107fa/c107fa12/synthetic-RLVL/lm_eval_results/dose_sweep_20260912"
def cond(r):
    m=re.match(r"qwen25_7b_dose_(.+)_p(\d+)_seed(\d+)",r)
    return m.group(1)+m.group(2) if m else None
def load(suite):
    out=collections.defaultdict(lambda: collections.defaultdict(list))
    d=os.path.join(ROOT,suite)
    if not os.path.isdir(d): return out
    for r in sorted(os.listdir(d)):
        p=os.path.join(d,r)
        if not os.path.isdir(p) or not os.path.exists(p+"/.complete"): continue
        fs=sorted(glob.glob(p+"/**/results_*.json",recursive=True))
        if not fs: continue
        for k,v in json.load(open(fs[-1]))["results"].items():
            want=METRIC.get(k)
            if want and want in v:
                out[cond(r)][k].append(100*v[want]); continue
            vals=[vv for kk,vv in v.items() if kk!="alias" and "stderr" not in kk and isinstance(vv,float)]
            if vals: out[cond(r)][k].append(100*vals[0])
    return out
METRIC={"gsm8k":"exact_match,strict-match","mmlu":"acc,none","arc_challenge":"acc_norm,none",
        "agieval_logiqa_en":"acc_norm,none","hellaswag":"acc_norm,none","piqa":"acc_norm,none",
        "winogrande":"acc,none"}
S={n:load(n) for n in ["deduction","folio_gpqa","cot","bbh_nochat","standard","native","code_nochat","multihop"]}
MH=collections.defaultdict(lambda: collections.defaultdict(list))
import gzip
sys.path.insert(0,"/home/hpc/c107fa/c107fa12/synthetic-RLVL/lm_eval_tasks/synthrlvl_ood")
from utils import qa_f1_score
STOPS=("<|im_end|>","</s>","\n")
def _trunc(t):
    cut=len(t)
    for x in STOPS:
        i=t.find(x)
        if i!=-1: cut=min(cut,i)
    return t[:cut].strip()
mhroot=os.path.join(ROOT,"multihop")
if os.path.isdir(mhroot):
    for r in sorted(os.listdir(mhroot)):
        d=os.path.join(mhroot,r)
        if not os.path.isdir(d) or not os.path.exists(d+"/.complete"): continue
        for f in glob.glob(d+"/**/samples_*_standard_*.jsonl*",recursive=True):
            task=re.search(r"samples_synthrlvl_longbench_(\w+)_standard",f).group(1)
            op=gzip.open if f.endswith(".gz") else open
            sc=[]
            for l in op(f,"rt"):
                row=json.loads(l)
                resp=row.get("filtered_resps") or row.get("resps")
                while isinstance(resp,list) and resp: resp=resp[0]
                pred=_trunc(str(resp))
                sc.append(max((qa_f1_score(pred,str(a)) for a in row["doc"]["answers"]),default=0.0))
            if sc: MH[cond(r)][task].append(100*sum(sc)/len(sc))

CH=["web_of_lies","tracking_shuffled_objects_three_objects","tracking_shuffled_objects_five_objects",
    "tracking_shuffled_objects_seven_objects","logical_deduction_three_objects",
    "logical_deduction_five_objects","logical_deduction_seven_objects","formal_fallacies"]
ALLSUB=[k[len("bbh_cot_fewshot_"):] for k in S["bbh_nochat"].get("control00",{}) if k.startswith("bbh_cot_fewshot_")]
# The general suite reports several metrics per task and the paper uses the
# normalised accuracy where the harness provides one, so the metric is named
# explicitly instead of taking whichever comes first.
METRIC={"gsm8k":"exact_match,strict-match","mmlu":"acc,none","arc_challenge":"acc_norm,none",
        "agieval_logiqa_en":"acc_norm,none","hellaswag":"acc_norm,none","piqa":"acc_norm,none",
        "winogrande":"acc,none"}
def mean(suite,c,k):
    v=S[suite].get(c,{}).get(k,[])
    return st.mean(v) if v else None
def group(c,keys):
    vals=[mean("bbh_nochat",c,"bbh_cot_fewshot_"+k) for k in keys]
    vals=[v for v in vals if v is not None]
    return st.mean(vals) if vals else None
CONDS=["control00","logic01","logic05","logic10","logic25","nl_exact01","nl_exact05","nl_exact10","nl_exact25"]
GROUPS=[
 ("Deduction benchmarks",[
   ("ProofWriter, depth 1", lambda c: mean("deduction",c,"synthrlvl_deduction_pw_d1")),
   ("ProofWriter, depth 3", lambda c: mean("deduction",c,"synthrlvl_deduction_pw_d3")),
   ("ProofWriter, depth 5", lambda c: mean("deduction",c,"synthrlvl_deduction_pw_d5")),
   ("ProofWriter, depth 3, reasoning prompt", lambda c: mean("cot",c,"synthrlvl_deduction_pw_cot_d3")),
   ("FOLIO", lambda c: mean("folio_gpqa",c,"synthrlvl_folio")),
 ]),
 ("Generator's own problems",[
   ("checkable derivation, chain 5", lambda c: mean("native",c,"synthrlvl_deduction_bp_native_d5")),
   ("checkable derivation, chain 15", lambda c: mean("native",c,"synthrlvl_deduction_bp_native_d15")),
   ("checkable derivation, chain 25", lambda c: mean("native",c,"synthrlvl_deduction_bp_native_d25")),
 ]),
 ("BIG-Bench Hard",[
   ("all 27 subtasks", lambda c: group(c,ALLSUB)),
   ("eight chain subtasks", lambda c: group(c,CH)),
   ("nineteen other subtasks", lambda c: group(c,[k for k in ALLSUB if k not in CH])),
   ("web of lies", lambda c: mean("bbh_nochat",c,"bbh_cot_fewshot_web_of_lies")),
   ("tracking objects, five", lambda c: mean("bbh_nochat",c,"bbh_cot_fewshot_tracking_shuffled_objects_five_objects")),
   ("tracking objects, seven", lambda c: mean("bbh_nochat",c,"bbh_cot_fewshot_tracking_shuffled_objects_seven_objects")),
   ("logical deduction, seven", lambda c: mean("bbh_nochat",c,"bbh_cot_fewshot_logical_deduction_seven_objects")),
   ("formal fallacies", lambda c: mean("bbh_nochat",c,"bbh_cot_fewshot_formal_fallacies")),
 ]),
 ("Multi-hop question answering",[
   ("HotpotQA", lambda c: st.mean(MH[c]["hotpotqa"]) if MH[c]["hotpotqa"] else None),
   ("2WikiMultihopQA", lambda c: st.mean(MH[c]["2wikimqa"]) if MH[c]["2wikimqa"] else None),
   ("MuSiQue", lambda c: st.mean(MH[c]["musique"]) if MH[c]["musique"] else None),
 ]),
 ("General benchmarks",[
   ("GSM8K", lambda c: mean("standard",c,"gsm8k")),
   ("MMLU", lambda c: mean("standard",c,"mmlu")),
   ("ARC-Challenge", lambda c: mean("standard",c,"arc_challenge")),
   ("LogiQA", lambda c: mean("standard",c,"agieval_logiqa_en")),
   ("HellaSwag", lambda c: mean("standard",c,"hellaswag")),
   ("PIQA", lambda c: mean("standard",c,"piqa")),
   ("WinoGrande", lambda c: mean("standard",c,"winogrande")),
   ("GPQA-Diamond", lambda c: mean("folio_gpqa",c,"synthrlvl_gpqa_diamond")),
 ]),
 ("Code",[
   ("HumanEval", lambda c: mean("code_nochat",c,"humaneval")),
   ("MBPP", lambda c: mean("code_nochat",c,"mbpp")),
 ]),
]
HEAD=["control","1\\%","5\\%","10\\%","25\\%","1\\%","5\\%","10\\%","25\\%"]
out=[]
out.append("\\begin{tabular}{l" + "c"*9 + "}")
out.append("\\toprule")
out.append(" & & \\multicolumn{4}{c}{formal} & \\multicolumn{4}{c}{English} \\\\")
out.append("\\cmidrule(lr){3-6}\\cmidrule(lr){7-10}")
out.append("benchmark & " + " & ".join(HEAD) + " \\\\")
for title, rows in GROUPS:
    out.append("\\midrule")
    out.append("\\multicolumn{10}{l}{\\textit{" + title + "}} \\\\")
    for name, fn in rows:
        vals=[fn(c) for c in CONDS]
        if all(v is None for v in vals): continue
        best=max(v for v in vals if v is not None)
        cells=[]
        for v in vals:
            if v is None: cells.append("--")
            elif abs(v-best)<1e-9: cells.append("$\\mathbf{%.1f}$" % v)
            else: cells.append("$%.1f$" % v)
        out.append("\\quad " + name + " & " + " & ".join(cells) + " \\\\")
out.append("\\bottomrule")
out.append("\\end{tabular}")
print("\n".join(out))
