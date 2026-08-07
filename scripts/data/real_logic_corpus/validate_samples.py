import json, glob, sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from logic_engine import LogicEngine
eng = LogicEngine()
base = Path(__file__).resolve().parent
tot_ok = tot_bad = 0
for path in sorted(glob.glob(str(base / "samples" / "*_sample100.jsonl"))):
    ok = bad = 0
    for line in open(path):
        row = json.loads(line)
        doc = row["formal_doc"]
        prem = doc.split("Premises:\n", 1)[1].split("\n\nDerivation:")[0]
        proof = doc.split("Derivation:\n", 1)[1].split("\n\nConclusion:")[0]
        concl = doc.split("Conclusion:\n", 1)[1].split("\n\nFinal answer:")[0].strip()
        rep = eng.analyze_proof(premises=prem, conclusion=concl, proof=proof)
        if rep.ok:
            ok += 1
        else:
            bad += 1
            print("INVALID", row["id"], rep.error)
    tot_ok += ok; tot_bad += bad
    print(Path(path).name, "ok:", ok, "bad:", bad)
print("TOTAL ok:", tot_ok, "bad:", tot_bad)
