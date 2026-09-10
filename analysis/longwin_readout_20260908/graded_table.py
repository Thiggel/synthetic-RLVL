import json,glob,os,sys
R=sys.argv[1]
arms=["control","longdoc","logic_band25","nl_exact_band25","condensed_logic_band25"]
tasks=["pw_d0","pw_d1","pw_d2","pw_d3","pw_d5","bp_cot_d5","bp_cot_d10","bp_cot_d15","bp_cot_d20","bp_cot_d25"]
res={}
for a in arms:
    d=glob.glob(f"{R}/qwen25_7b_longwin_{a}_2p5b_dolci_100k_lr5em6*")
    if not d: continue
    fs=sorted(glob.glob(d[0]+"/**/results*.json",recursive=True))
    if not fs: continue
    r=json.load(open(fs[-1]))["results"]
    res[a]={t:r["synthrlvl_deduction_"+t]["exact_match,none"] for t in tasks if "synthrlvl_deduction_"+t in r}
print("task,"+",".join(res))
NAN=float("nan")
for t in tasks: print(t+","+",".join("{:.3f}".format(res[a].get(t,NAN)) for a in res))
