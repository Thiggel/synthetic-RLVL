import sys
sys.path.insert(0,'scripts/analysis')
from grounded_checker import check_grounded_logic as chk

prompt = ("Mary is a knight. If Mary is a knight then John is a knave. "
          "A box holds 12 rows of 3 apples. Who is John and how many apples?")

good = """Reasoning: Mary is a knight, so John is a knave. 12 times 3 is 36.
<logic>
p1: "Mary is a knight" |- Knight(mary)
p2: "If Mary is a knight then John is a knave" |- Knight(mary) -> Knave(john)
s1: p1,p2 |- Knave(john) [mp]
concl: s1
</logic>
<answer>Knave(john)</answer>"""

assumed = """<logic>
p1: "Mary is a knight" |- Knave(john)
concl: p1
</logic>"""

invented = """<logic>
p1: "John confessed to lying" |- Knave(john)
p2: "Mary is a knight" |- Knight(mary)
s1: p1,p2 |- Knave(john) [mp]
concl: s1
</logic>"""

decorative = """<logic>
p1: "Mary is a knight" |- Knight(mary)
p2: "If Mary is a knight then John is a knave" |- Knight(mary) -> Knave(john)
s1: p1,p2 |- Knight(mary) [and_e]
concl: s1
</logic>"""

arith = """<logic>
p1: "A box holds 12 rows of 3 apples" |- rows=12
s1: 12 * 3 = 36 [arith]
concl: s1
</logic>"""

badarith = """<logic>
p1: "A box holds 12 rows of 3 apples" |- rows=12
s1: 12 * 3 = 38 [arith]
concl: s1
</logic>"""

for name, text, ans in [("good", good, "Knave(john)"),
                        ("assumed answer", assumed, "Knave(john)"),
                        ("invented premise", invented, "Knave(john)"),
                        ("decorative", decorative, "Knave(john)"),
                        ("arith ok", arith, "36"),
                        ("arith wrong", badarith, "38")]:
    r = chk(prompt, text, ans)
    print(f"{name:18s} valid={r.valid}  grounded={r.grounded} steps={r.steps_valid} "
          f"load={r.load_bearing} nontrivial={r.non_trivial}  {r.reasons[:1]}")
