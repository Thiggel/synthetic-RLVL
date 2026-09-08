import json,glob,os,collections,sys
R=sys.argv[1] if len(sys.argv)>1 else "/home/vault/c107fa/c107fa12/synthetic-RLVL/lm_eval_results/qwen25_longwin_graded_deduction_20260906"
print("arm,n,acc,gold_true,gold_false,gold_unknown,balanced3,twoclass,pred_true,pred_false,pred_unknown,pred_other")
for d in sorted(glob.glob(R+"/qwen25_7b_longwin_*")):
    name=os.path.basename(d).replace("qwen25_7b_longwin_","").replace("_2p5b_dolci_100k_lr5em6","")
    cnt=collections.Counter(); pred=collections.Counter()
    for dep in [1,2,3,5]:
        fs=sorted(glob.glob(d+"/*/samples_synthrlvl_deduction_pw_d%d_*.jsonl"%dep))
        for line in open(fs[-1]):
            r=json.loads(line); g=r["target"].lower(); p=(r["filtered_resps"][0] or "").strip().lower()
            cnt[(g,"n")]+=1; cnt[(g,"ok")]+=(p==g); pred[p]+=1
    n=sum(cnt[(g,"n")] for g in ["true","false","unknown"])
    acc={g:cnt[(g,"ok")]/cnt[(g,"n")] for g in ["true","false","unknown"]}
    tot=sum(cnt[(g,"ok")] for g in acc)/n
    two=(cnt[("true","ok")]+cnt[("false","ok")])/(cnt[("true","n")]+cnt[("false","n")])
    oth=1-(pred["true"]+pred["false"]+pred["unknown"])/n
    print("{},{},{:.4f},{:.4f},{:.4f},{:.4f},{:.4f},{:.4f},{:.4f},{:.4f},{:.4f},{:.4f}".format(name,n,tot,acc["true"],acc["false"],acc["unknown"],sum(acc.values())/3,two,pred["true"]/n,pred["false"]/n,pred["unknown"]/n,oth))
print("# gold class counts:",{g:cnt[(g,"n")] for g in ["true","false","unknown"]})
