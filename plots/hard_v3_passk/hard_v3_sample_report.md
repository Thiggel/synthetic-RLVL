# Hard-v3 Sample Inspection

Loaded 120 representative sample rows from 15 files.

## Aggregate Representative Samples

schema	rows	format_mean	syntactic_mean	valid_mean	correct_mean	invalid_correct_rows
correct_plus_0p1_format	24	1.000	1.000	1.000	1.000	0
correct_plus_valid_plus_0p1_format	24	1.000	1.000	1.000	1.000	0
correct_times_valid_plus_0p1_format	24	1.000	1.000	1.000	1.000	0
correct_plus_line_valid_plus_0p1_format	24	1.000	1.000	1.000	1.000	0
correct_times_line_valid_plus_0p1_format	24	1.000	1.000	1.000	1.000	0

## Representative Failure Rows

No failures appear in the compact sample JSONL files. Use the aggregate pass@k metrics for real failure rates.

## Clean Example

schema=correct_plus_0p1_format seed=3407 step=1

```text
<question>
1. All things that are new are relaxed.
2. Clara is new.
3. Clara is calm.
4. Wendy is happy.
5. For Nina, if Nina is new, then Nina is cheerful.
6. Victor is angry.
7. Nina is new.
8. Grace is sad.
9. For Clara, if Clara is new, then Clara is relaxed.
10. For Nina, if Nina is both new and angry, then Nina is relaxed.
11. For Grace, if Grace is new, then Grace is relaxed.
What mood does Nina have?
</question>
<formal>
<constants>
a = Clara
b = Wendy
c = Grace
d = Victor
e = Nina
f = Paula
</constants>
<predicates>
Ax: x is new
Bx: x is relaxed
Cx: x is calm
Dx: x is happy
Ex: x is sad
Fx: x is young
Gx: x is old
Hx: x is ancient
Ix: x is modern
Jx: x is aged
Kx: x is fresh
Lx: x is mature
Mx: x is angry
Nx: x is proud
Ox: x is nervous
Px: x is cheerful
</predicates>
<premises>
Ae
Ae -> Be
Ca
Db
Ec
Md
Aa
Aa -> Ba
Ae -> Pe
Ac -> Bc
Ae & Me -> Be
</premises>
<proof>
Ae ; R,1
Be ; ->E,2,12
</proof>
<conclusion>
Be
</conclusion>
</formal>
<answer>
relaxed
</answer>
```
