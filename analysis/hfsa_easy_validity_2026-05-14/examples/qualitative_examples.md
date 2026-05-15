# Qualitative Diagnostic Examples

## correct_cfvalid_gold_atom_derived

- condition: `train0p0_correct_only`, step: `5`, gold: `poppy`, shortcut: `ruby`, answer: `poppy`
- cf_valid: `1.0`, correct_answer_derived: `0.0`, proof lines: `12/12`

Prompt excerpt:

```text
<question>
1. a is birch.
2. a is north.
3. If a is south and a is birch, then b is maple.
4. If b is maple, then b is north.
5. If a is north and a is birch, then b is coral.
6. If b is coral, then b is north.
7. If b is north and b is coral, then c is pearl.
8. If c is pearl, then c is north.
9. If b is north and b is maple, then c is poppy.
10. If c is poppy, then c is north.
11. If c is north and c is poppy, then d is maple.
12. If d is maple, then d is south.
13. If c is north and c is pearl, then d is coral.
14. If d is coral, then d is south.
15. If d is south and d is coral, then e is pearl.
16. If e is pearl, then e is south.
17. If d is south and d is maple, then e is ruby.
18. If e is ruby, then e is north.
19. If e is south and e is pearl, then f is poppy.
20. If f is poppy, then f is south.
21. If e is north and e is ruby, then f is ruby.
22. If f is ruby, then f is south.
Which state applies to f?
</question>


```

Generation:

```text
<formal>
<constants>
a = a
b = b
c = c
d = d
e = e
f = f
</constants>
<predicates>
Ax: x is pearl
Bx: x is coral
Cx: x is ruby
Dx: x is poppy
Ex: x is birch
Fx: x is maple
Gx: x is north
Hx: x is south
</predicates>
<premises>
Ea
Ga
Ha & Ea -> Fb
Fb -> Gb
Ga & Ea -> Bb
Bb -> Gb
Gb & Bb -> Ac
Ac -> Gc
Gb & Fb -> Dc
Dc -> Gc
Gc & Dc -> Fd
Fd -> Hd
Gc & Ac -> Bd
Bd -> Hd
Hd & Bd -> Ae
Ae -> He
Hd & Fd -> Ce
Ce -> Ge
He & Ae -> Df
Df -> Hf
Ge & Ce -> Cf
Cf -> Hf
</premises>
<proof>
Ea ; R
Ga ; R
Bb ; ->E
Gb ; ->E
Ac ; ->E
Gc ; ->E
Bd ; ->E
Hd ; ->E
Ae ; ->E
He ; ->E
Df ; ->E
Hf ; ->E
</proof>
<conclusion>
Hf
</conclusion>
</formal>
<answer>
poppy
</answer>
```

- condition: `train0p0_correct_only`, step: `5`, gold: `poppy`, shortcut: `ruby`, answer: `poppy`
- cf_valid: `1.0`, correct_answer_derived: `0.0`, proof lines: `12/12`

Prompt excerpt:

```text
<question>
1. a is birch.
2. a is north.
3. If a is south and a is birch, then b is maple.
4. If b is maple, then b is north.
5. If a is north and a is birch, then b is coral.
6. If b is coral, then b is north.
7. If b is north and b is coral, then c is pearl.
8. If c is pearl, then c is north.
9. If b is north and b is maple, then c is poppy.
10. If c is poppy, then c is north.
11. If c is north and c is poppy, then d is maple.
12. If d is maple, then d is south.
13. If c is north and c is pearl, then d is coral.
14. If d is coral, then d is south.
15. If d is south and d is coral, then e is pearl.
16. If e is pearl, then e is south.
17. If d is south and d is maple, then e is ruby.
18. If e is ruby, then e is north.
19. If e is south and e is pearl, then f is poppy.
20. If f is poppy, then f is south.
21. If e is north and e is ruby, then f is ruby.
22. If f is ruby, then f is south.
Which state applies to f?
</question>


```

Generation:

```text
<formal>
<constants>
a = a
b = b
c = c
d = d
e = e
f = f
</constants>
<predicates>
Ax: x is pearl
Bx: x is coral
Cx: x is birch
Dx: x is poppy
Ex: x is ruby
Fx: x is maple
Gx: x is north
Hx: x is south
</predicates>
<premises>
Ca
Ga
Ha & Ca -> Fb
Fb -> Gb
Ga & Ca -> Bb
Bb -> Gb
Gb & Bb -> Ac
Ac -> Gc
Gb & Fb -> Dc
Dc -> Gc
Gc & Dc -> Fd
Fd -> Hd
Gc & Ac -> Bd
Bd -> Hd
Hd & Bd -> Ae
Ae -> He
Hd & Fd -> Ee
Ee -> Ge
He & Ae -> Df
Df -> Hf
Ge & Ee -> Ef
Ef -> Hf
</premises>
<proof>
Ca ; R
Ga ; R
Bb ; ->E
Gb ; ->E
Ac ; ->E
Gc ; ->E
Bd ; ->E
Hd ; ->E
Ae ; ->E
He ; ->E
Df ; ->E
Hf ; ->E
</proof>
<conclusion>
Hf
</conclusion>
</formal>
<answer>
poppy
</answer>
```
## correct_cfvalid_no_gold_atom

- condition: `train0p0_correct_only`, step: `10`, gold: `ivory`, shortcut: `willow`, answer: `ivory`
- cf_valid: `1.0`, correct_answer_derived: `0.0`, proof lines: `22/22`

Prompt excerpt:

```text
<question>
1. a is teal.
2. a is north.
3. If a is north and a is teal, then b is violet.
4. If b is violet, then b is north.
5. If a is south and a is teal, then b is cedar.
6. If b is cedar, then b is north.
7. If b is north and b is cedar, then c is granite.
8. If c is granite, then c is north.
9. If b is north and b is violet, then c is violet.
10. If c is violet, then c is south.
11. If c is south and c is violet, then d is granite.
12. If d is granite, then d is south.
13. If c is north and c is granite, then d is pearl.
14. If d is pearl, then d is north.
15. If d is south and d is granite, then e is pearl.
16. If e is pearl, then e is south.
17. If d is north and d is pearl, then e is poppy.
18. If e is poppy, then e is north.
19. If e is north and e is poppy, then f is poppy.
20. If f is poppy, then f is south.
21. If e is south and e is pearl, then f is amber.
22. If f is amber, then f is north.
23. If f is south and f is poppy, then g is cedar.
24. If g is cedar, then g is south.
25. If f is north and f is amber, then g is harbor.
26. If g is harbor, then g is north.
27. If g is north and g is harbor, then h is harbor.
28. If h is harbor, then h is south.
29. If g is south and g is cedar, then h is coral.
30. If h is coral, then h is north.
31. If h is north and h is coral, then i is willow.
32. If i is willow, then i is north.
33. If h is south and h is harbor, then i is coral.
34. If i is coral, then i is south.
35. If i is south and i is coral, then j is amber.
36. If j is amber, then j is south.
37. If i is north and i is willow, then j is ivory.
38. If j is 
...
```

Generation:

```text
<formal>
<constants>
a = a
b = b
c = c
d = d
e = e
f = f
g = g
h = h
i = i
j = j
k = k
</constants>
<predicates>
Ax: x is teal
Bx: x is pearl
Cx: x is coral
Dx: x is granite
Ex: x is amber
Fx: x is poppy
Gx: x is harbor
Hx: x is ivory
Ix: x is willow
Jx: x is cedar
Kx: x is violet
Lx: x is north
Mx: x is south
</predicates>
<premises>
Aa
La
La & Aa -> Kb
Kb -> Lb
Ma & Aa -> Jb
Jb -> Lb
Lb & Jb -> Dc
Dc -> Lc
Lb & Kb -> Kc
Kc -> Mc
Mc & Kc -> Dd
Dd -> Md
Lc & Dc -> Bd
Bd -> Ld
Md & Dd -> Be
Be -> Me
Ld & Bd -> Fe
Fe -> Le
Le & Fe -> Ff
Ff -> Mf
Me & Be -> Ef
Ef -> Lf
Lf & Ef -> Jg
Jg -> Mg
Lf & Ef -> Gg
Gg -> Lg
Lg & Gg -> Gh
Gh -> Mh
Mg & Jg -> Ch
Ch -> Lh
Lh & Ch -> Ii
Ii -> Li
Mh & Gh -> Ci
Ci -> Mi
Mi & Ci -> Ej
Ej -> Mj
Li & Ii -> Hj
Hj -> Lj
Mj & Ej -> Hk
Hk -> Mk
Lj & Hj -> Ik
Ik -> Mk
</premises>
<proof>
Aa ; R
La ; R
Kb ; ->E
Lb ; ->E
Kc ; ->E
Mc ; ->E
Dd ; ->E
Md ; ->E
Be ; ->E
Me ; ->E
Ef ; ->E
Lf ; ->E
Jg ; ->E
Mg ; ->E
Ch ; ->E
Lh ; ->E
Ii ; ->E
Li ; ->E
Hj ; ->E
Lj ; ->E
Ik ; ->E
Mk ; ->E
</proof>
<conclusion>
Mk
</conclusion>
</formal>
<answer>
ivory
</answer>
```

- condition: `train0p0_correct_only`, step: `10`, gold: `laurel`, shortcut: `ivory`, answer: `laurel`
- cf_valid: `1.0`, correct_answer_derived: `0.0`, proof lines: `22/22`

Prompt excerpt:

```text
<question>
1. a is cedar.
2. a is south.
3. If a is south and a is cedar, then b is cobalt.
4. If b is cobalt, then b is north.
5. If a is north and a is cedar, then b is teal.
6. If b is teal, then b is north.
7. If b is north and b is teal, then c is birch.
8. If c is birch, then c is north.
9. If b is north and b is cobalt, then c is granite.
10. If c is granite, then c is north.
11. If c is north and c is birch, then d is meadow.
12. If d is meadow, then d is north.
13. If c is north and c is granite, then d is birch.
14. If d is birch, then d is south.
15. If d is north and d is meadow, then e is juniper.
16. If e is juniper, then e is north.
17. If d is south and d is birch, then e is meadow.
18. If e is meadow, then e is south.
19. If e is south and e is meadow, then f is teal.
20. If f is teal, then f is south.
21. If e is north and e is juniper, then f is violet.
22. If f is violet, then f is north.
23. If f is south and f is teal, then g is lime.
24. If g is lime, then g is north.
25. If f is north and f is violet, then g is laurel.
26. If g is laurel, then g is north.
27. If g is north and g is laurel, then h is violet.
28. If h is violet, then h is south.
29. If g is north and g is lime, then h is juniper.
30. If h is juniper, then h is south.
31. If h is south and h is juniper, then i is cobalt.
32. If i is cobalt, then i is south.
33. If h is south and h is violet, then i is lime.
34. If i is lime, then i is south.
35. If i is south and i is lime, then j is ivory.
36. If j is ivory, then j is north.
37. If i is south and i is cobalt, then j is granite.
38. If 
...
```

Generation:

```text
<formal>
<constants>
a = a
b = b
c = c
d = d
e = e
f = f
g = g
h = h
i = i
j = j
k = k
</constants>
<predicates>
Ax: x is cedar
Bx: x is granite
Cx: x is meadow
Dx: x is teal
Ex: x is violet
Fx: x is birch
Gx: x is juniper
Hx: x is cobalt
Ix: x is lime
Jx: x is ivory
Kx: x is laurel
Lx: x is north
Mx: x is south
</predicates>
<premises>
Aa
Ma
Ma & Aa -> Hb
Hb -> Lb
La & Aa -> Db
Db -> Lb
Lb & Db -> Fc
Fc -> Lc
Lb & Hb -> Bc
Bc -> Lc
Lc & Fc -> Cd
Cd -> Ld
Lc & Bc -> Fd
Fd -> Md
Ld & Cd -> Ge
Ge -> Le
Md & Fd -> Ce
Ce -> Me
Me & Ce -> Df
Df -> Mf
Le & Ge -> Ef
Ef -> Lf
Mf & Df -> Ig
Ig -> Lg
Lf & Ef -> Kg
Kg -> Lg
Lg & Kg -> Eh
Eh -> Mh
Lg & Ig -> Gh
Gh -> Mh
Mh & Gh -> Hi
Hi -> Mi
Mh & Eh -> Ii
Ii -> Mi
Mi & Ii -> Jj
Jj -> Lj
Mi & Hi -> Bk
Bk -> Mk
Mk & Bk -> Kl
Kl -> Mk
Lj & Jj -> Jk
Jk -> Mk
</premises>
<proof>
Aa ; R
Ma ; R
Hb ; ->E
Lb ; ->E
Bc ; ->E
Lc ; ->E
Fd ; ->E
Md ; ->E
Ce ; ->E
Me ; ->E
Df ; ->E
Mf ; ->E
Ig ; ->E
Lg ; ->E
Gh ; ->E
Mh ; ->E
Hi ; ->E
Mi ; ->E
Bk ; ->E
Mk ; ->E
Kl ; ->E
Mk ; ->E
</proof>
<conclusion>
Mk
</conclusion>
</formal>
<answer>
laurel
</answer>
```
## correct_cfinvalid_gold_atom_derived

- condition: `train0p0_correct_only`, step: `10`, gold: `ivory`, shortcut: `willow`, answer: `ivory`
- cf_valid: `0.0`, correct_answer_derived: `0.0`, proof lines: `20/22`

Prompt excerpt:

```text
<question>
1. a is teal.
2. a is north.
3. If a is north and a is teal, then b is violet.
4. If b is violet, then b is north.
5. If a is south and a is teal, then b is cedar.
6. If b is cedar, then b is north.
7. If b is north and b is cedar, then c is granite.
8. If c is granite, then c is north.
9. If b is north and b is violet, then c is violet.
10. If c is violet, then c is south.
11. If c is south and c is violet, then d is granite.
12. If d is granite, then d is south.
13. If c is north and c is granite, then d is pearl.
14. If d is pearl, then d is north.
15. If d is south and d is granite, then e is pearl.
16. If e is pearl, then e is south.
17. If d is north and d is pearl, then e is poppy.
18. If e is poppy, then e is north.
19. If e is north and e is poppy, then f is poppy.
20. If f is poppy, then f is south.
21. If e is south and e is pearl, then f is amber.
22. If f is amber, then f is north.
23. If f is south and f is poppy, then g is cedar.
24. If g is cedar, then g is south.
25. If f is north and f is amber, then g is harbor.
26. If g is harbor, then g is north.
27. If g is north and g is harbor, then h is harbor.
28. If h is harbor, then h is south.
29. If g is south and g is cedar, then h is coral.
30. If h is coral, then h is north.
31. If h is north and h is coral, then i is willow.
32. If i is willow, then i is north.
33. If h is south and h is harbor, then i is coral.
34. If i is coral, then i is south.
35. If i is south and i is coral, then j is amber.
36. If j is amber, then j is south.
37. If i is north and i is willow, then j is ivory.
38. If j is 
...
```

Generation:

```text
<formal>
<constants>
a = a
b = b
c = c
d = d
e = e
f = f
g = g
h = h
i = i
j = j
k = k
</constants>
<predicates>
Ax: x is pearl
Bx: x is violet
Cx: x is harbor
Dx: x is amber
Ex: x is poppy
Fx: x is coral
Gx: x is granite
Hx: x is willow
Ix: x is teal
Jx: x is ivory
Kx: x is cedar
Lx: x is granite
Mx: x is willow
Nx: x is amber
Ox: x is cedar
Px: x is pearl
Qx: x is violet
Rx: x is coral
Sx: x is ivory
Tx: x is teal
Ux: x is ivory
Vx: x is poppy
Wx: x is pearl
</predicates>
<premises>
Ia
Na
Ha & Ia -> Bb
Bb -> Nb
Sa & Ia -> Kb
Kb -> Nb
Nb & Kb -> Gc
Gc -> Nc
Nb & Bb -> Bc
Bc -> Tc
Tc & Bc -> Ld
Ld -> Rd
Nc & Gc -> Ad
Ad -> Nd
Rd & Ld -> Ee
Ee -> Se
Nd & Ad -> Ee
Ee -> Ne
Ne & Ee -> Ef
Ef -> Sf
Se & Ee -> Df
Df -> Nf
Sf & Ef -> Og
Og -> Sg
Nf & Df -> Cg
Cg -> Ng
Ng & Cg -> Ch
Ch -> Rh
Sg & Og -> Fh
Fh -> Rh
Rh & Fh -> Hi
Hi -> Ni
Rh & Ch -> Fj
Fj -> Rj
Ni & Hi -> Jk
Jk -> Nk
Rj & Fj -> Mk
Mk -> Nk
</premises>
<proof>
Ia ; R
Na ; R
Bb ; ->E
Nb ; ->E
Bc ; ->E
Tc ; ->E
Ld ; ->E
Rd ; ->E
Ee ; ->E
Se ; ->E
Df ; ->E
Nf ; ->E
Cg ; ->E
Ng ; ->E
Ch ; ->E
Rh ; ->E
Hi ; ->E
Ni ; ->E
Jk ; ->E
Nk ; ->E
</proof>
<conclusion>
Nk
</conclusion>
</formal>
<answer>
ivory
</answer>
```

- condition: `train0p0_correct_only`, step: `10`, gold: `ivory`, shortcut: `willow`, answer: `ivory`
- cf_valid: `0.0`, correct_answer_derived: `0.0`, proof lines: `18/22`

Prompt excerpt:

```text
<question>
1. a is teal.
2. a is north.
3. If a is north and a is teal, then b is violet.
4. If b is violet, then b is north.
5. If a is south and a is teal, then b is cedar.
6. If b is cedar, then b is north.
7. If b is north and b is cedar, then c is granite.
8. If c is granite, then c is north.
9. If b is north and b is violet, then c is violet.
10. If c is violet, then c is south.
11. If c is south and c is violet, then d is granite.
12. If d is granite, then d is south.
13. If c is north and c is granite, then d is pearl.
14. If d is pearl, then d is north.
15. If d is south and d is granite, then e is pearl.
16. If e is pearl, then e is south.
17. If d is north and d is pearl, then e is poppy.
18. If e is poppy, then e is north.
19. If e is north and e is poppy, then f is poppy.
20. If f is poppy, then f is south.
21. If e is south and e is pearl, then f is amber.
22. If f is amber, then f is north.
23. If f is south and f is poppy, then g is cedar.
24. If g is cedar, then g is south.
25. If f is north and f is amber, then g is harbor.
26. If g is harbor, then g is north.
27. If g is north and g is harbor, then h is harbor.
28. If h is harbor, then h is south.
29. If g is south and g is cedar, then h is coral.
30. If h is coral, then h is north.
31. If h is north and h is coral, then i is willow.
32. If i is willow, then i is north.
33. If h is south and h is harbor, then i is coral.
34. If i is coral, then i is south.
35. If i is south and i is coral, then j is amber.
36. If j is amber, then j is south.
37. If i is north and i is willow, then j is ivory.
38. If j is 
...
```

Generation:

```text
<formal>
<constants>
a = a
b = b
c = c
d = d
e = e
f = f
g = g
h = h
i = i
j = j
k = k
</constants>
<predicates>
Ax: x is willow
Bx: x is pearl
Cx: x is cedar
Dx: x is amber
Ex: x is violet
Fx: x is coral
Gx: x is granite
Hx: x is poppy
Jx: x is teal
Kx: x is harbor
Lx: x is ivory
Mx: x is granite
Nx: x is harbor
Ox: x is teal
Px: x is poppy
Qx: x is violet
Rx: x is ivory
Sx: x is harbor
Tx: x is teal
Ux: x is granite
Vx: x is cedar
Wx: x is willow
</predicates>
<premises>
Ja
Na
Ha & Ja -> Eb
Eb -> Nb
Sa & Ja -> Cb
Cb -> Nb
Nb & Cb -> Gc
Gc -> Nc
Nb & Eb -> Ec
Ec -> Sc
Sc & Ec -> Md
Md -> Sd
Nc & Gc -> Bd
Bd -> Nd
Sd & Md -> Be
Be -> Se
Nd & Bd -> Fi
Fi -> Ni
Se & Be -> Df
Df -> Nf
Nf & Df -> Ph
Ph -> Fh
Nf & Fi -> Fh
Fh -> Ih
Ih -> Hi
Ih & Fh -> Ai
Ai -> Ik
Hi & Fi -> Fk
Fk -> Jk
Ik & Ai -> Lk
Lk -> Jk
</premises>
<proof>
Ja ; R
Na ; R
Eb ; ->E
Nb ; ->E
Ec ; ->E
Sc ; ->E
Md ; ->E
Sd ; ->E
Be ; ->E
Se ; ->E
Df ; ->E
Nf ; ->E
Ph ; ->E
Fh ; ->E
Ai ; ->E
Ik ; ->E
Lk ; ->E
Jk ; ->E
</proof>
<conclusion>
Jk
</conclusion>
</formal>
<answer>
ivory
</answer>
```
## correct_cfinvalid_no_gold_atom

- condition: `train0p0_correct_only`, step: `10`, gold: `laurel`, shortcut: `ivory`, answer: `laurel`
- cf_valid: `0.0`, correct_answer_derived: `0.0`, proof lines: `26/22`

Prompt excerpt:

```text
<question>
1. a is cedar.
2. a is south.
3. If a is south and a is cedar, then b is cobalt.
4. If b is cobalt, then b is north.
5. If a is north and a is cedar, then b is teal.
6. If b is teal, then b is north.
7. If b is north and b is teal, then c is birch.
8. If c is birch, then c is north.
9. If b is north and b is cobalt, then c is granite.
10. If c is granite, then c is north.
11. If c is north and c is birch, then d is meadow.
12. If d is meadow, then d is north.
13. If c is north and c is granite, then d is birch.
14. If d is birch, then d is south.
15. If d is north and d is meadow, then e is juniper.
16. If e is juniper, then e is north.
17. If d is south and d is birch, then e is meadow.
18. If e is meadow, then e is south.
19. If e is south and e is meadow, then f is teal.
20. If f is teal, then f is south.
21. If e is north and e is juniper, then f is violet.
22. If f is violet, then f is north.
23. If f is south and f is teal, then g is lime.
24. If g is lime, then g is north.
25. If f is north and f is violet, then g is laurel.
26. If g is laurel, then g is north.
27. If g is north and g is laurel, then h is violet.
28. If h is violet, then h is south.
29. If g is north and g is lime, then h is juniper.
30. If h is juniper, then h is south.
31. If h is south and h is juniper, then i is cobalt.
32. If i is cobalt, then i is south.
33. If h is south and h is violet, then i is lime.
34. If i is lime, then i is south.
35. If i is south and i is lime, then j is ivory.
36. If j is ivory, then j is north.
37. If i is south and i is cobalt, then j is granite.
38. If 
...
```

Generation:

```text
<formal>
<constants>
a = a
b = b
c = c
d = d
e = e
f = f
g = g
h = h
i = i
j = j
k = k
</constants>
<predicates>
Ax: x is lacquer
Bx: x is laurel
Cx: x is lavender
Dx: x is meadow
Ex: x is juniper
Fx: x is cherry
Gx: x is ivory
Hx: x is cobalt
Ix: x is teal
Jx: x is meadow
Kx: x is violet
Lx: x is granite
Mx: x is cedar
Nx: x is lime
Ox: x is birch
Px: x is cedar
Qx: x is cobalt
Rx: x is violet
Sx: x is ivory
Tx: x is teal
Ux: x is laurel
Vx: x is juniper
Wx: x is teal
</predicates>
<premises>
Ma
aR
aHa
HaQ
bQ
bB
bW
Wc
cN
cO
OcD
DcN
cd
dE
de
eF
ef
fT
fg
gS
gh
hR
hi
iN
ij
jA
jk
kU
</premises>
<proof>
Ma ; R
aR ; Ha
HaQ ; Q
bQ ; B
bW ; W
Wc ; c
cN ; N
cO ; O
OcD ; D
DcN ; N
cd ; d
dE ; E
de ; e
eF ; F
ef ; f
fT ; T
fg ; g
gS ; S
gh ; h
hR ; R
hi ; i
iN ; N
ij ; j
jA ; A
jk ; k
kU ; U
</proof>
<conclusion>
kU
</conclusion>
</formal>
<answer>
laurel
</answer>
```

- condition: `train0p0_correct_only`, step: `10`, gold: `maple`, shortcut: `willow`, answer: `maple`
- cf_valid: `0.0`, correct_answer_derived: `0.0`, proof lines: `16/22`

Prompt excerpt:

```text
<question>
1. a is elm.
2. a is north.
3. If a is north and a is elm, then b is pearl.
4. If b is pearl, then b is north.
5. If a is south and a is elm, then b is violet.
6. If b is violet, then b is north.
7. If b is north and b is pearl, then c is orchid.
8. If c is orchid, then c is north.
9. If b is north and b is violet, then c is laurel.
10. If c is laurel, then c is north.
11. If c is north and c is orchid, then d is pearl.
12. If d is pearl, then d is south.
13. If c is north and c is laurel, then d is laurel.
14. If d is laurel, then d is south.
15. If d is south and d is laurel, then e is granite.
16. If e is granite, then e is north.
17. If d is south and d is pearl, then e is birch.
18. If e is birch, then e is north.
19. If e is north and e is birch, then f is birch.
20. If f is birch, then f is south.
21. If e is north and e is granite, then f is granite.
22. If f is granite, then f is south.
23. If f is south and f is granite, then g is olive.
24. If g is olive, then g is north.
25. If f is south and f is birch, then g is orchid.
26. If g is orchid, then g is south.
27. If g is north and g is olive, then h is cedar.
28. If h is cedar, then h is north.
29. If g is south and g is orchid, then h is olive.
30. If h is olive, then h is south.
31. If h is south and h is olive, then i is maple.
32. If i is maple, then i is north.
33. If h is north and h is cedar, then i is cedar.
34. If i is cedar, then i is south.
35. If i is north and i is maple, then j is violet.
36. If j is violet, then j is south.
37. If i is south and i is cedar, then j is willow.
38. If j is 
...
```

Generation:

```text
<formal>
<constants>
a = a
b = b
c = c
d = d
e = e
f = f
g = g
h = h
i = i
j = j
k = k
</constants>
<predicates>
Ax: x is cedar
Bx: x is granite
Cx: x is violet
Dx: x is birch
Ex: x is laurel
Fx: x is olive
Gx: x is elm
Hx: x is maple
Ix: x is orchid
Jx: x is willow
Kx: x is pearl
Lx: x is north
Mx: x is south
</predicates>
<premises>
Ga
La
La & Ga -> Kb
Kb -> Lb
Ma & Ga -> Cb
Cb -> Lb
Lb & Kb -> Ic
Ic -> Lc
Lb & Cb -> Ec
Ec -> Lc
Lc & Ic -> Kd
Kd -> Md
Lc & Ec -> Ed
Ed -> Md
Md & Ed -> Be
Be -> Le
Md & Kd -> De
De -> Le
Le & De -> Hi
Hi -> Li
Le & Be -> Aj
Aj -> Mj
Le & De -> Jj
Jj -> Lj
Mj & Aj -> Ik
Ik -> Mk
Lj & Jj -> Jk
Jk -> Mk
</premises>
<proof>
Ga ; R
La ; R
Kb ; ->E
Lb ; ->E
Ic ; ->E
Lc ; ->E
Kd ; ->E
Md ; ->E
De ; ->E
Le ; ->E
Hi ; ->E
Li ; ->E
Aj ; ->E
Mj ; ->E
Ik ; ->E
Mk ; ->E
</proof>
<conclusion>
Mk
</conclusion>
</formal>
<answer>
maple
</answer>
```
## shortcut_wrong

- condition: `train0p0_correct_only`, step: `10`, gold: `ivory`, shortcut: `willow`, answer: `willow`
- cf_valid: `0.0`, correct_answer_derived: `0.0`, proof lines: `24/22`

Prompt excerpt:

```text
<question>
1. a is teal.
2. a is north.
3. If a is north and a is teal, then b is violet.
4. If b is violet, then b is north.
5. If a is south and a is teal, then b is cedar.
6. If b is cedar, then b is north.
7. If b is north and b is cedar, then c is granite.
8. If c is granite, then c is north.
9. If b is north and b is violet, then c is violet.
10. If c is violet, then c is south.
11. If c is south and c is violet, then d is granite.
12. If d is granite, then d is south.
13. If c is north and c is granite, then d is pearl.
14. If d is pearl, then d is north.
15. If d is south and d is granite, then e is pearl.
16. If e is pearl, then e is south.
17. If d is north and d is pearl, then e is poppy.
18. If e is poppy, then e is north.
19. If e is north and e is poppy, then f is poppy.
20. If f is poppy, then f is south.
21. If e is south and e is pearl, then f is amber.
22. If f is amber, then f is north.
23. If f is south and f is poppy, then g is cedar.
24. If g is cedar, then g is south.
25. If f is north and f is amber, then g is harbor.
26. If g is harbor, then g is north.
27. If g is north and g is harbor, then h is harbor.
28. If h is harbor, then h is south.
29. If g is south and g is cedar, then h is coral.
30. If h is coral, then h is north.
31. If h is north and h is coral, then i is willow.
32. If i is willow, then i is north.
33. If h is south and h is harbor, then i is coral.
34. If i is coral, then i is south.
35. If i is south and i is coral, then j is amber.
36. If j is amber, then j is south.
37. If i is north and i is willow, then j is ivory.
38. If j is 
...
```

Generation:

```text
<formal>
<constants>
a = a
b = b
c = c
d = d
e = e
f = f
g = g
h = h
i = i
j = j
k = k
</constants>
<predicates>
Ax: x is teal
Bx: x is ivory
Cx: x is coral
Dx: x is amber
Ex: x is poppy
Fx: x is granite
Gx: x is pearl
Hx: x is harbor
Ix: x is willow
Jx: x is cedar
Kx: x is violet
Lx: x is north
Mx: x is south
</predicates>
<premises>
Aa
La
La & Aa -> Kb
Kb -> Lb
Ma & Aa -> Jb
Jb -> Lb
Lb & Jb -> Fc
Fc -> Lc
Lb & Kb -> Kc
Kc -> Mc
Mc & Kc -> Fd
Fd -> Md
Lc & Fc -> Gd
Gd -> Ld
Md & Fd -> Ge
Ge -> Me
Ld & Gd -> Ee
Ee -> Le
Le & Ee -> Ef
Ef -> Mf
Me & Ge -> Df
Df -> Lf
Mf & Ef -> Jg
Jg -> Mg
Lf & Df -> Hg
Hg -> Lg
Lg & Hg -> Hh
Hh -> Mh
Mg & Jg -> Ch
Ch -> Lh
Lh & Ch -> Ii
Ii -> Li
Mh & Hh -> Ci
Ci -> Mi
Mi & Ci -> Dj
Dj -> Mj
Li & Ii -> Bj
Bj -> Lj
Mj & Dj -> Bk
Bk -> Mk
Lj & Bj -> Ik
Ik -> Mk
</premises>
<proof>
Aa ; R
La ; R
Kb ; ->E
Lb ; ->E
Kc ; ->E
Mc ; ->E
Fd ; ->E
Md ; ->E
Ge ; ->E
Me ; ->E
Df ; ->E
Lf ; ->E
Hg ; ->E
Lg ; ->E
Hh ; ->E
Mh ; ->E
Ci ; ->E
Mi ; ->E
Dj ; ->E
Mj ; ->E
Bk ; ->E
Mk ; ->E
Ik ; ->E
Mk ; ->E
</proof>
<conclusion>
Mk
</conclusion>
</formal>
<answer>
willow
</answer>
```

- condition: `train0p0_correct_only`, step: `10`, gold: `ivory`, shortcut: `willow`, answer: `willow`
- cf_valid: `0.0`, correct_answer_derived: `0.0`, proof lines: `24/22`

Prompt excerpt:

```text
<question>
1. a is teal.
2. a is north.
3. If a is north and a is teal, then b is violet.
4. If b is violet, then b is north.
5. If a is south and a is teal, then b is cedar.
6. If b is cedar, then b is north.
7. If b is north and b is cedar, then c is granite.
8. If c is granite, then c is north.
9. If b is north and b is violet, then c is violet.
10. If c is violet, then c is south.
11. If c is south and c is violet, then d is granite.
12. If d is granite, then d is south.
13. If c is north and c is granite, then d is pearl.
14. If d is pearl, then d is north.
15. If d is south and d is granite, then e is pearl.
16. If e is pearl, then e is south.
17. If d is north and d is pearl, then e is poppy.
18. If e is poppy, then e is north.
19. If e is north and e is poppy, then f is poppy.
20. If f is poppy, then f is south.
21. If e is south and e is pearl, then f is amber.
22. If f is amber, then f is north.
23. If f is south and f is poppy, then g is cedar.
24. If g is cedar, then g is south.
25. If f is north and f is amber, then g is harbor.
26. If g is harbor, then g is north.
27. If g is north and g is harbor, then h is harbor.
28. If h is harbor, then h is south.
29. If g is south and g is cedar, then h is coral.
30. If h is coral, then h is north.
31. If h is north and h is coral, then i is willow.
32. If i is willow, then i is north.
33. If h is south and h is harbor, then i is coral.
34. If i is coral, then i is south.
35. If i is south and i is coral, then j is amber.
36. If j is amber, then j is south.
37. If i is north and i is willow, then j is ivory.
38. If j is 
...
```

Generation:

```text
<formal>
<constants>
a = a
b = b
c = c
d = d
e = e
f = f
g = g
h = h
i = i
j = j
k = k
</constants>
<predicates>
Ax: x is poppy
Bx: x is amber
Cx: x is pearl
Dx: x is cedar
Ex: x is willow
Fx: x is granite
Gx: x is coral
Hx: x is teal
Ix: x is ivory
Jx: x is violet
Kx: x is harbor
Lx: x is north
Mx: x is south
</predicates>
<premises>
Ha
La
Ha & La -> Jb
Jb -> Lb
Ma & Ha -> Db
Db -> Lb
Lb & Db -> Fc
Fc -> Lc
Lb & Jb -> Jc
Jc -> Mc
Mc & Jc -> Fd
Fd -> Md
Lc & Fc -> Cd
Cd -> Ld
Md & Fd -> Ce
Ce -> Me
Ld & Cd -> Ae
Ae -> Le
Le & Ae -> Af
Af -> Mf
Me & Ce -> Bf
Bf -> Lf
Mf & Af -> Dg
Dg -> Mg
Lf & Bf -> Kg
Kg -> Lg
Lg & Kg -> Kh
Kh -> Mh
Mg & Dg -> Gh
Gh -> Lh
Lh & Gh -> Ei
Ei -> Li
Mh & Kh -> Gi
Gi -> Mi
Mi & Gi -> Ij
Ij -> Mj
Li & Ei -> Ij
Ij -> Lj
Mj & Ij -> Ik
Ik -> Mk
Lj & Ij -> Ek
Ek -> Mk
</premises>
<proof>
Ha ; R
La ; R
Jb ; ->E
Lb ; ->E
Jc ; ->E
Mc ; ->E
Fd ; ->E
Md ; ->E
Ce ; ->E
Me ; ->E
Bf ; ->E
Lf ; ->E
Kg ; ->E
Lg ; ->E
Kh ; ->E
Mh ; ->E
Gi ; ->E
Mi ; ->E
Ij ; ->E
Mj ; ->E
Ik ; ->E
Mk ; ->E
Ek ; ->E
Mk ; ->E
</proof>
<conclusion>
Mk
</conclusion>
</formal>
<answer>
willow
</answer>
```
## wrong_cfvalid_gold_atom_derived

- condition: `train0p0_correct_only`, step: `10`, gold: `ivory`, shortcut: `willow`, answer: `amber`
- cf_valid: `1.0`, correct_answer_derived: `0.0`, proof lines: `22/22`

Prompt excerpt:

```text
<question>
1. a is teal.
2. a is north.
3. If a is north and a is teal, then b is violet.
4. If b is violet, then b is north.
5. If a is south and a is teal, then b is cedar.
6. If b is cedar, then b is north.
7. If b is north and b is cedar, then c is granite.
8. If c is granite, then c is north.
9. If b is north and b is violet, then c is violet.
10. If c is violet, then c is south.
11. If c is south and c is violet, then d is granite.
12. If d is granite, then d is south.
13. If c is north and c is granite, then d is pearl.
14. If d is pearl, then d is north.
15. If d is south and d is granite, then e is pearl.
16. If e is pearl, then e is south.
17. If d is north and d is pearl, then e is poppy.
18. If e is poppy, then e is north.
19. If e is north and e is poppy, then f is poppy.
20. If f is poppy, then f is south.
21. If e is south and e is pearl, then f is amber.
22. If f is amber, then f is north.
23. If f is south and f is poppy, then g is cedar.
24. If g is cedar, then g is south.
25. If f is north and f is amber, then g is harbor.
26. If g is harbor, then g is north.
27. If g is north and g is harbor, then h is harbor.
28. If h is harbor, then h is south.
29. If g is south and g is cedar, then h is coral.
30. If h is coral, then h is north.
31. If h is north and h is coral, then i is willow.
32. If i is willow, then i is north.
33. If h is south and h is harbor, then i is coral.
34. If i is coral, then i is south.
35. If i is south and i is coral, then j is amber.
36. If j is amber, then j is south.
37. If i is north and i is willow, then j is ivory.
38. If j is 
...
```

Generation:

```text
<formal>
<constants>
a = a
b = b
c = c
d = d
e = e
f = f
g = g
h = h
i = i
j = j
k = k
</constants>
<predicates>
Ax: x is teal
Bx: x is cedar
Cx: x is granite
Dx: x is coral
Ex: x is pearl
Fx: x is amber
Gx: x is violet
Hx: x is poppy
Jx: x is willow
Kx: x is harbor
Lx: x is ivory
Mx: x is north
Nx: x is south
Ox: x is violet
Px: x is granite
Qx: x is teal
Rx: x is harbor
Sx: x is poppy
Tx: x is cedar
Ux: x is amber
Vx: x is ivory
Wx: x is pearl
Xx: x is willow
Yx: x is pearl
Zx: x is violet
</predicates>
<premises>
Aa
Ma
Ma & Aa -> Gb
Gb -> Mb
Na & Aa -> Bb
Bb -> Mb
Mb & Bb -> Cc
Cc -> Mc
Mb & Gb -> Oc
Oc -> Nc
Nc & Oc -> Pd
Pd -> Nd
Mc & Cc -> Ed
Ed -> Md
Nd & Pd -> Ee
Ee -> Ne
Md & Ed -> He
He -> Me
Me & He -> Hf
Hf -> Nf
Ne & Ee -> Uf
Uf -> Mf
Nf & Hf -> Tg
Tg -> Ng
Mf & Uf -> Rg
Rg -> Mg
Mg & Rg -> Rh
Rh -> Nh
Ng & Tg -> Dh
Dh -> Mh
Mh & Dh -> Xi
Xi -> Mi
Nh & Rh -> Di
Di -> Ni
Ni & Di -> Fj
Fj -> Nj
Mi & Xi -> Vj
Vj -> Mj
Nj & Fj -> Vk
Vk -> Nk
Mj & Vj -> Xk
Xk -> Nk
</premises>
<proof>
Aa ; R
Ma ; R
Gb ; ->E
Mb ; ->E
Oc ; ->E
Nc ; ->E
Pd ; ->E
Nd ; ->E
Ee ; ->E
Ne ; ->E
Uf ; ->E
Mf ; ->E
Rg ; ->E
Mg ; ->E
Rh ; ->E
Nh ; ->E
Di ; ->E
Ni ; ->E
Fj ; ->E
Nj ; ->E
Vk ; ->E
Nk ; ->E
</proof>
<conclusion>
Nk
</conclusion>
</formal>
<answer>
amber
</answer>
```

- condition: `train0p0_validity_gated`, step: `10`, gold: `laurel`, shortcut: `ivory`, answer: `marble`
- cf_valid: `1.0`, correct_answer_derived: `0.0`, proof lines: `22/22`

Prompt excerpt:

```text
<question>
1. a is cedar.
2. a is south.
3. If a is south and a is cedar, then b is cobalt.
4. If b is cobalt, then b is north.
5. If a is north and a is cedar, then b is teal.
6. If b is teal, then b is north.
7. If b is north and b is teal, then c is birch.
8. If c is birch, then c is north.
9. If b is north and b is cobalt, then c is granite.
10. If c is granite, then c is north.
11. If c is north and c is birch, then d is meadow.
12. If d is meadow, then d is north.
13. If c is north and c is granite, then d is birch.
14. If d is birch, then d is south.
15. If d is north and d is meadow, then e is juniper.
16. If e is juniper, then e is north.
17. If d is south and d is birch, then e is meadow.
18. If e is meadow, then e is south.
19. If e is south and e is meadow, then f is teal.
20. If f is teal, then f is south.
21. If e is north and e is juniper, then f is violet.
22. If f is violet, then f is north.
23. If f is south and f is teal, then g is lime.
24. If g is lime, then g is north.
25. If f is north and f is violet, then g is laurel.
26. If g is laurel, then g is north.
27. If g is north and g is laurel, then h is violet.
28. If h is violet, then h is south.
29. If g is north and g is lime, then h is juniper.
30. If h is juniper, then h is south.
31. If h is south and h is juniper, then i is cobalt.
32. If i is cobalt, then i is south.
33. If h is south and h is violet, then i is lime.
34. If i is lime, then i is south.
35. If i is south and i is lime, then j is ivory.
36. If j is ivory, then j is north.
37. If i is south and i is cobalt, then j is granite.
38. If 
...
```

Generation:

```text
<formal>
<constants>
a = a
b = b
c = c
d = d
e = e
f = f
g = g
h = h
i = i
j = j
k = k
</constants>
<predicates>
Ax: x is lime
Bx: x is juniper
Cx: x is marble
Dx: x is meadow
Ex: x is teal
Fx: x is birch
Gx: x is cobalt
Hx: x is laurel
Ix: x is ivory
Jx: x is cedar
Kx: x is granite
Lx: x is violet
Mx: x is north
Nx: x is south
Ox: x is cobalt
Px: x is birch
Qx: x is meadow
Rx: x is meadow
Sx: x is ivory
Tx: x is teal
Ux: x is juniper
Vx: x is laurel
Wx: x is granite
Xx: x is marble
Yx: x is cedar
Zx: x is violet
</predicates>
<premises>
Ja
Na
Na & Ja -> Gb
Gb -> Mb
Ma & Ja -> Eb
Eb -> Mb
Mb & Eb -> Fc
Fc -> Mc
Mb & Gb -> Kc
Kc -> Mc
Mc & Fc -> Dd
Dd -> Md
Mc & Kc -> Fd
Fd -> Nd
Md & Dd -> Be
Be -> Me
Nd & Fd -> Re
Re -> Ne
Ne & Re -> Tf
Tf -> Nf
Me & Be -> Lf
Lf -> Mf
Nf & Tf -> Ag
Ag -> Mf
Mf & Lf -> Vh
Vh -> Nh
Mf & Ag -> Bh
Bh -> Nh
Nh & Bh -> Oi
Oi -> Ni
Nh & Vh -> Ai
Ai -> Ni
Ni & Ai -> Ij
Ij -> Mj
Ni & Oi -> Wj
Wj -> Nj
Nj & Wj -> Vk
Vk -> Nk
Mj & Ij -> Ik
Ik -> Nk
</premises>
<proof>
Ja ; R
Na ; R
Gb ; ->E
Mb ; ->E
Kc ; ->E
Mc ; ->E
Fd ; ->E
Nd ; ->E
Re ; ->E
Ne ; ->E
Tf ; ->E
Nf ; ->E
Ag ; ->E
Mf ; ->E
Bh ; ->E
Nh ; ->E
Oi ; ->E
Ni ; ->E
Wj ; ->E
Nj ; ->E
Vk ; ->E
Nk ; ->E
</proof>
<conclusion>
Nk
</conclusion>
</formal>
<answer>
marble
</answer>
```
## wrong_cfvalid_no_gold_atom

- condition: `train0p0_correct_only`, step: `10`, gold: `laurel`, shortcut: `ivory`, answer: `lime`
- cf_valid: `1.0`, correct_answer_derived: `0.0`, proof lines: `14/22`

Prompt excerpt:

```text
<question>
1. a is cedar.
2. a is south.
3. If a is south and a is cedar, then b is cobalt.
4. If b is cobalt, then b is north.
5. If a is north and a is cedar, then b is teal.
6. If b is teal, then b is north.
7. If b is north and b is teal, then c is birch.
8. If c is birch, then c is north.
9. If b is north and b is cobalt, then c is granite.
10. If c is granite, then c is north.
11. If c is north and c is birch, then d is meadow.
12. If d is meadow, then d is north.
13. If c is north and c is granite, then d is birch.
14. If d is birch, then d is south.
15. If d is north and d is meadow, then e is juniper.
16. If e is juniper, then e is north.
17. If d is south and d is birch, then e is meadow.
18. If e is meadow, then e is south.
19. If e is south and e is meadow, then f is teal.
20. If f is teal, then f is south.
21. If e is north and e is juniper, then f is violet.
22. If f is violet, then f is north.
23. If f is south and f is teal, then g is lime.
24. If g is lime, then g is north.
25. If f is north and f is violet, then g is laurel.
26. If g is laurel, then g is north.
27. If g is north and g is laurel, then h is violet.
28. If h is violet, then h is south.
29. If g is north and g is lime, then h is juniper.
30. If h is juniper, then h is south.
31. If h is south and h is juniper, then i is cobalt.
32. If i is cobalt, then i is south.
33. If h is south and h is violet, then i is lime.
34. If i is lime, then i is south.
35. If i is south and i is lime, then j is ivory.
36. If j is ivory, then j is north.
37. If i is south and i is cobalt, then j is granite.
38. If 
...
```

Generation:

```text
<formal>
<constants>
a = a
b = b
c = c
d = d
e = e
f = f
g = g
h = h
i = i
j = j
k = k
</constants>
<predicates>
Ax: x is cobalt
Bx: x is teal
Cx: x is laurel
Dx: x is ivory
Ex: x is cedar
Fx: x is violet
Gx: x is juniper
Hx: x is meadow
Ix: x is granite
Jx: x is birch
Kx: x is lime
Lx: x is north
Mx: x is south
</predicates>
<premises>
Ea
Ma
Ma & Ea -> Ab
Ab -> Lb
La & Ea -> Bb
Bb -> Lb
Lb & Bb -> Jc
Jc -> Lc
Lb & Ab -> Ic
Ic -> Lc
Lc & Jc -> Hd
Hd -> Ld
Lc & Ic -> Jd
Jd -> Md
Ld & Hd -> Ge
Ge -> Le
Md & Jd -> He
He -> Me
Me & He -> Bf
Bf -> Mf
Le & Ge -> Ff
Ff -> Lf
Mf & Bf -> Kg
Kg -> Lk
Lf & Ff -> Ch
Ch -> Lk
</premises>
<proof>
Ea ; R
Ma ; R
Ab ; ->E
Lb ; ->E
Ic ; ->E
Lc ; ->E
Jd ; ->E
Md ; ->E
He ; ->E
Me ; ->E
Bf ; ->E
Mf ; ->E
Kg ; ->E
Lk ; ->E
</proof>
<conclusion>
Lk
</conclusion>
</formal>
<answer>
lime
</answer>
```

- condition: `train0p0_correct_only`, step: `10`, gold: `ruby`, shortcut: `harbor`, answer: `teal`
- cf_valid: `1.0`, correct_answer_derived: `0.0`, proof lines: `20/22`

Prompt excerpt:

```text
<question>
1. a is cedar.
2. a is north.
3. If a is south and a is cedar, then b is cobalt.
4. If b is cobalt, then b is north.
5. If a is north and a is cedar, then b is pearl.
6. If b is pearl, then b is north.
7. If b is north and b is pearl, then c is ivory.
8. If c is ivory, then c is north.
9. If b is north and b is cobalt, then c is birch.
10. If c is birch, then c is north.
11. If c is north and c is birch, then d is ivory.
12. If d is ivory, then d is south.
13. If c is north and c is ivory, then d is ruby.
14. If d is ruby, then d is north.
15. If d is south and d is ivory, then e is harbor.
16. If e is harbor, then e is north.
17. If d is north and d is ruby, then e is olive.
18. If e is olive, then e is north.
19. If e is north and e is olive, then f is birch.
20. If f is birch, then f is south.
21. If e is north and e is harbor, then f is teal.
22. If f is teal, then f is north.
23. If f is south and f is birch, then g is laurel.
24. If g is laurel, then g is north.
25. If f is north and f is teal, then g is cobalt.
26. If g is cobalt, then g is south.
27. If g is north and g is laurel, then h is pearl.
28. If h is pearl, then h is south.
29. If g is south and g is cobalt, then h is laurel.
30. If h is laurel, then h is south.
31. If h is south and h is pearl, then i is olive.
32. If i is olive, then i is south.
33. If h is south and h is laurel, then i is willow.
34. If i is willow, then i is north.
35. If i is north and i is willow, then j is willow.
36. If j is willow, then j is south.
37. If i is south and i is olive, then j is teal.
38. If j is teal, then 
...
```

Generation:

```text
<formal>
<constants>
a = a
b = b
c = c
d = d
e = e
f = f
g = g
h = h
i = i
j = j
k = k
</constants>
<predicates>
Ax: x is olive
Bx: x is laurel
Cx: x is willow
Dx: x is cedar
Ex: x is harbor
Fx: x is cobalt
Gx: x is pearl
Hx: x is birch
Ix: x is ruby
Jx: x is teal
Kx: x is ivory
Lx: x is north
Mx: x is south
</predicates>
<premises>
Da
La
Ma & Da -> Fb
Fb -> Lb
La & Da -> Gb
Gb -> Lb
Lb & Gb -> Kc
Kc -> Lc
Lb & Fb -> Hc
Hc -> Lc
Lc & Hc -> Kd
Kd -> Md
Lc & Kc -> Id
Id -> Ld
Md & Kd -> Ee
Ee -> Le
Ld & Id -> Ae
Ae -> Le
Le & Ae -> Hf
Hf -> Mf
Le & Ee -> Jf
Jf -> Lf
Mf & Hf -> Bg
Bg -> Lg
Lf & Jf -> Fg
Fg -> Mg
Lg & Bg -> Gh
Gh -> Mh
Mg & Fg -> Bh
Bh -> Mh
Mh & Gh -> Ai
Ai -> Mi
Mh & Bh -> Ci
Ci -> Li
Li & Ci -> Cj
Cj -> Mj
Mi & Ai -> Jk
Jk -> Mk
Mj & Jk -> Ek
Ek -> Mk
</premises>
<proof>
Da ; R
La ; R
Gb ; ->E
Lb ; ->E
Kc ; ->E
Lc ; ->E
Id ; ->E
Ld ; ->E
Ae ; ->E
Le ; ->E
Hf ; ->E
Mf ; ->E
Bg ; ->E
Lg ; ->E
Gh ; ->E
Mh ; ->E
Ai ; ->E
Mi ; ->E
Jk ; ->E
Mk ; ->E
</proof>
<conclusion>
Mk
</conclusion>
</formal>
<answer>
teal
</answer>
```
## wrong_cfinvalid

- condition: `train0p0_correct_only`, step: `10`, gold: `ivory`, shortcut: `willow`, answer: `poppy`
- cf_valid: `0.0`, correct_answer_derived: `0.0`, proof lines: `20/22`

Prompt excerpt:

```text
<question>
1. a is teal.
2. a is north.
3. If a is north and a is teal, then b is violet.
4. If b is violet, then b is north.
5. If a is south and a is teal, then b is cedar.
6. If b is cedar, then b is north.
7. If b is north and b is cedar, then c is granite.
8. If c is granite, then c is north.
9. If b is north and b is violet, then c is violet.
10. If c is violet, then c is south.
11. If c is south and c is violet, then d is granite.
12. If d is granite, then d is south.
13. If c is north and c is granite, then d is pearl.
14. If d is pearl, then d is north.
15. If d is south and d is granite, then e is pearl.
16. If e is pearl, then e is south.
17. If d is north and d is pearl, then e is poppy.
18. If e is poppy, then e is north.
19. If e is north and e is poppy, then f is poppy.
20. If f is poppy, then f is south.
21. If e is south and e is pearl, then f is amber.
22. If f is amber, then f is north.
23. If f is south and f is poppy, then g is cedar.
24. If g is cedar, then g is south.
25. If f is north and f is amber, then g is harbor.
26. If g is harbor, then g is north.
27. If g is north and g is harbor, then h is harbor.
28. If h is harbor, then h is south.
29. If g is south and g is cedar, then h is coral.
30. If h is coral, then h is north.
31. If h is north and h is coral, then i is willow.
32. If i is willow, then i is north.
33. If h is south and h is harbor, then i is coral.
34. If i is coral, then i is south.
35. If i is south and i is coral, then j is amber.
36. If j is amber, then j is south.
37. If i is north and i is willow, then j is ivory.
38. If j is 
...
```

Generation:

```text
<formal>
<constants>
a = a
b = b
c = c
d = d
e = e
f = f
g = g
h = h
i = i
j = j
k = k
</constants>
<predicates>
Ax: x is coral
Bx: x is amber
Cx: x is violet
Dx: x is pearl
Ex: x is cedar
Fx: x is teal
Gx: x is granite
Hx: x is harbor
Ix: x is willow
Jx: x is ivory
Kx: x is poppy
Lx: x is teal
Mx: x is violet
Nx: x is granite
Ox: x is harbor
Px: x is pearl
Qx: x is cedar
Rx: x is amber
Sx: x is willow
Tx: x is coral
Ux: x is teal
Vx: x is teal
Wx: x is ivory
Xx: x is granite
Yx: x is amber
Zx: x is harbor
</predicates>
<premises>
Fa
Na
Ha & Fa -> Ab
Ab -> Nb
La & Fa -> Eb
Eb -> Nb
Nb & Eb -> Gc
Gc -> Nc
Nb & Ab -> Mc
Mc -> Tc
Tc & Mc -> Nd
Nd -> Sd
Nc & Gc -> Pd
Pd -> Nd
Sd & Nd -> Pe
Pe -> Se
Nd & Pd -> Ke
Ke -> Ne
Ne & Ke -> Kf
Kf -> Sf
Se & Pe -> Rf
Rf -> Nf
Sf & Kf -> Qg
Qg -> Sg
Nf & Rf -> Hg
Hg -> Ng
Ng & Hg -> Oh
Oh -> Th
Sg & Qg -> Ah
Ah -> Nh
Nh & Ah -> Si
Si -> Ni
Th & Oh -> Ti
Ti -> Sj
Ni & Si -> Jj
Jj -> Nj
Sj & Ti -> Kk
Kk -> Sk
Nj & Jj -> Ik
Ik -> Sk
</premises>
<proof>
Fa ; R
Na ; R
Ab ; ->E
Nb ; ->E
Mc ; ->E
Tc ; ->E
Nd ; ->E
Sd ; ->E
Pe ; ->E
Se ; ->E
Rf ; ->E
Nf ; ->E
Hg ; ->E
Ng ; ->E
Oh ; ->E
Th ; ->E
Ti ; ->E
Sj ; ->E
Kk ; ->E
Sk ; ->E
</proof>
<conclusion>
Sk
</conclusion>
</formal>
<answer>
poppy
</answer>
```

- condition: `train0p0_correct_only`, step: `10`, gold: `ivory`, shortcut: `willow`, answer: `amber`
- cf_valid: `0.0`, correct_answer_derived: `0.0`, proof lines: `20/22`

Prompt excerpt:

```text
<question>
1. a is teal.
2. a is north.
3. If a is north and a is teal, then b is violet.
4. If b is violet, then b is north.
5. If a is south and a is teal, then b is cedar.
6. If b is cedar, then b is north.
7. If b is north and b is cedar, then c is granite.
8. If c is granite, then c is north.
9. If b is north and b is violet, then c is violet.
10. If c is violet, then c is south.
11. If c is south and c is violet, then d is granite.
12. If d is granite, then d is south.
13. If c is north and c is granite, then d is pearl.
14. If d is pearl, then d is north.
15. If d is south and d is granite, then e is pearl.
16. If e is pearl, then e is south.
17. If d is north and d is pearl, then e is poppy.
18. If e is poppy, then e is north.
19. If e is north and e is poppy, then f is poppy.
20. If f is poppy, then f is south.
21. If e is south and e is pearl, then f is amber.
22. If f is amber, then f is north.
23. If f is south and f is poppy, then g is cedar.
24. If g is cedar, then g is south.
25. If f is north and f is amber, then g is harbor.
26. If g is harbor, then g is north.
27. If g is north and g is harbor, then h is harbor.
28. If h is harbor, then h is south.
29. If g is south and g is cedar, then h is coral.
30. If h is coral, then h is north.
31. If h is north and h is coral, then i is willow.
32. If i is willow, then i is north.
33. If h is south and h is harbor, then i is coral.
34. If i is coral, then i is south.
35. If i is south and i is coral, then j is amber.
36. If j is amber, then j is south.
37. If i is north and i is willow, then j is ivory.
38. If j is 
...
```

Generation:

```text
<formal>
<constants>
a = a
b = b
c = c
d = d
e = e
f = f
g = g
h = h
i = i
j = j
k = k
</constants>
<predicates>
Ax: x is amber
Bx: x is cedar
Cx: x is pearl
Dx: x is willow
Ex: x is violet
Fx: x is coral
Gx: x is harbor
Hx: x is granite
Ix: x is poppy
Jx: x is teal
Kx: x is ivory
Lx: x is granite
Mx: x is ivory
Nx: x is harbor
Ox: x is teal
Px: x is poppy
Qx: x is granite
Rx: x is coral
Sx: x is teal
Tx: x is amber
Ux: x is teal
Vx: x is pearl
Wx: x is amber
</predicates>
<premises>
Ja
aRb
aHa
bHa
bBa
cHb
cHc
cEc
dHc
dCd
eCd
eIe
fIe
fTf
gBf
gNg
hNg
hFh
iFi
iSj
jAj
kKj
</premises>
<proof>
Ja ; R
aRb ; ->E
bHa ; ->E
cHb ; ->E
cHc ; ->E
cEc ; ->E
dHc ; ->E
dCd ; ->E
eCd ; ->E
eIe ; ->E
fIe ; ->E
fTf ; ->E
gBf ; ->E
gNg ; ->E
hNg ; ->E
hFh ; ->E
iFi ; ->E
iSj ; ->E
jAj ; ->E
kKj ; ->E
</proof>
<conclusion>
kA
</conclusion>
</formal>
<answer>
amber
</answer>
```
## malformed

- condition: `train0p0_correct_only`, step: `10`, gold: `ivory`, shortcut: `willow`, answer: ``
- cf_valid: `0.0`, correct_answer_derived: `0.0`, proof lines: `0/22`

Prompt excerpt:

```text
<question>
1. a is teal.
2. a is north.
3. If a is north and a is teal, then b is violet.
4. If b is violet, then b is north.
5. If a is south and a is teal, then b is cedar.
6. If b is cedar, then b is north.
7. If b is north and b is cedar, then c is granite.
8. If c is granite, then c is north.
9. If b is north and b is violet, then c is violet.
10. If c is violet, then c is south.
11. If c is south and c is violet, then d is granite.
12. If d is granite, then d is south.
13. If c is north and c is granite, then d is pearl.
14. If d is pearl, then d is north.
15. If d is south and d is granite, then e is pearl.
16. If e is pearl, then e is south.
17. If d is north and d is pearl, then e is poppy.
18. If e is poppy, then e is north.
19. If e is north and e is poppy, then f is poppy.
20. If f is poppy, then f is south.
21. If e is south and e is pearl, then f is amber.
22. If f is amber, then f is north.
23. If f is south and f is poppy, then g is cedar.
24. If g is cedar, then g is south.
25. If f is north and f is amber, then g is harbor.
26. If g is harbor, then g is north.
27. If g is north and g is harbor, then h is harbor.
28. If h is harbor, then h is south.
29. If g is south and g is cedar, then h is coral.
30. If h is coral, then h is north.
31. If h is north and h is coral, then i is willow.
32. If i is willow, then i is north.
33. If h is south and h is harbor, then i is coral.
34. If i is coral, then i is south.
35. If i is south and i is coral, then j is amber.
36. If j is amber, then j is south.
37. If i is north and i is willow, then j is ivory.
38. If j is 
...
```

Generation:

```text
<formal>
<constants>
a = a
b = b
c = c
d = d
e = e
f = f
g = g
h = h
i = i
j = j
k = k
</constants>
<predicates>
Ax: x is teal
Bx: x is violet
Cx: x is pearl
Dx: x is cedar
Ex: x is willow
Fx: x is harbor
Gx: x is coral
Hx: x is amber
Jx: x is granite
Kx: x is poppy
Lx: x is ivory
Mx: x is harbor
Nx: x is granite
Ox: x is amber
Px: x is teal
Qx: x is ivory
Rx: x is willow
Sx: x is poppy
Tx: x is ivory
Ux: x is teal
Vx: x is granite
Wx: x is cedar
</predicates>
<premises>
Aa
Na
Aa & Aa -> Bb
Bb -> Bb
Ea & Aa -> Db
Db -> Db
Bb & Db -> Jc
Jc -> Jc
Bb & Bb -> Bc
Bc -> Bc
Bc & Bc -> Nd
Nd -> Nd
Bc & Bc -> Cc
Cc -> Cc
Cc & Bc -> Jd
Jd -> Jd
Cc & Jc -> Cd
Cd -> Cd
Jc & Jc -> Cd
Cd -> Cd
Cd & Jd -> Ce
Ce -> Ce
Cd & Cd -> Hf
Hf -> Hf
Cd & Ce -> Og
Og -> Og
Ce & Og -> Hf
Hf -> Hf
Ce & Hf -> Hf
Hf -> Hf
Hf & Hf -> Wg
Wg -> Wg
Ce & Hf -> Fg
Fg -> Fg
Hf & Wg -> Fh
Fh -> Fh
Ce & Fg -> Fh
Fh -> Fh
Fh & Fh -> Gh
Gh -> Gh
Ce & Fh -> Gh
Gh -> Gh
Gh & Gh -> Gj
Gj -> Gj
Gh & Gh -> Ej
Ej -> Ej
Gh & Gh -> Ri
Ri -> Ri
Gh & Gj -> Ei
Ei -> Ei
Gh & Gh -> Si
Si -> Si
Gh & Ej -> Ti
Ti -> Ti
Gh & Ei -> Tk
Tk -> Tk
Gh & Si -> Tj
Tj -> Tj
Gh & Ei -> Tj
Tj -> Tj
Gh & Si -> Tj
Tj -> Tj
Gh & Gj -> Tj
Tj -> Tj
Gh & Si -> Tj
Tj -> Tj
Gh & Gj -> Tk
Tk -> Tk
Gh & Ei -> Tk
Tk -> Tk
Tj & Tk -> Tk
Tk -> Tk
Tj & Tk -> Tk
Tk -> Tk
Tk & Tk -> Tj
Tj -> Tj
Tk & Tk -> Tj
Tj -> Tj
Tk & Tk -> Tk
Tk -> Tk
Tj & Tk -> Tj
Tj -> Tj
Tk & Tk -> Tk
Tk -> Tk
Tj & Tk -> Tj
Tj -> Tj
Tj & Tk -> Tj
Tj -> Tj
Tj & Tk -> Tj
Tj -> Tj
Tj & Tj -> Tj
Tj -> Tj
Tj & Tj -> Tj
Tj -> Tj
Tj & Tj -> Tk
Tk -> Tk
Tj & Tj -> Tk
Tk -> Tk
Tj & Tj -> Tk
Tk -> Tk
Tj & Tj -> Tk
Tk -> Tk
Tj & Tj -> Tj
Tj -> Tj
Tj & Tk -> Tj
Tj -> Tj
Tk & Tj -> Tk
Tk -> Tk
Tj & Tk -> Tj
Tj -> Tj
Tk & Tj -> Tk
Tk -> Tk
Tj & Tk -> Tj
Tj -> Tj
Tk & Tj -> Tj
Tj -> Tj
Tk & Tj -> Tk
Tk -> Tk
Tj & Tk -> Tj
Tj -> Tj
Tk & Tj -> Tj
Tj -> Tj
Tk & Tk -> Tj
Tk -> Tj
Tj & Tk -> Tj
T
```

- condition: `train0p0_correct_only`, step: `10`, gold: `ivory`, shortcut: `willow`, answer: ``
- cf_valid: `0.0`, correct_answer_derived: `0.0`, proof lines: `0/22`

Prompt excerpt:

```text
<question>
1. a is teal.
2. a is north.
3. If a is north and a is teal, then b is violet.
4. If b is violet, then b is north.
5. If a is south and a is teal, then b is cedar.
6. If b is cedar, then b is north.
7. If b is north and b is cedar, then c is granite.
8. If c is granite, then c is north.
9. If b is north and b is violet, then c is violet.
10. If c is violet, then c is south.
11. If c is south and c is violet, then d is granite.
12. If d is granite, then d is south.
13. If c is north and c is granite, then d is pearl.
14. If d is pearl, then d is north.
15. If d is south and d is granite, then e is pearl.
16. If e is pearl, then e is south.
17. If d is north and d is pearl, then e is poppy.
18. If e is poppy, then e is north.
19. If e is north and e is poppy, then f is poppy.
20. If f is poppy, then f is south.
21. If e is south and e is pearl, then f is amber.
22. If f is amber, then f is north.
23. If f is south and f is poppy, then g is cedar.
24. If g is cedar, then g is south.
25. If f is north and f is amber, then g is harbor.
26. If g is harbor, then g is north.
27. If g is north and g is harbor, then h is harbor.
28. If h is harbor, then h is south.
29. If g is south and g is cedar, then h is coral.
30. If h is coral, then h is north.
31. If h is north and h is coral, then i is willow.
32. If i is willow, then i is north.
33. If h is south and h is harbor, then i is coral.
34. If i is coral, then i is south.
35. If i is south and i is coral, then j is amber.
36. If j is amber, then j is south.
37. If i is north and i is willow, then j is ivory.
38. If j is 
...
```

Generation:

```text
<formal>
<constants>
a = a
b = b
c = c
d = d
e = e
f = f
g = g
h = h
i = i
j = j
k = k
</constants>
<predicates>
Ax: x is amber
Bx: x is granite
Cx: x is violet
Dx: x is coral
Ex: x is harbor
Fx: x is teal
Gx: x is ivory
Hx: x is willow
Ix: x is poppy
Jx: x is pearl
Kx: x is cedar
Lx: x is teal
Mx: x is granite
Nx: x is pearl
Ox: x is poppy
Px: x is willow
Qx: x is ivory
Rx: x is amber
Sx: x is granite
Tx: x is cedar
Ux: x is teal
Vx: x is harbor
Wx: x is coral
Xx: x is violet
Yx: x is pearl
Zx: x is poppy
</predicates>
<premises>
Fa
a = a
Ha
a = a
Da & Ha -> Bi
Bi = b
Ba & Fa -> Kb
Kb = b
Kb & Kb -> Bc
Bc = c
Kb & Da -> Xc
Xc = c
Xc & Xc -> Bd
Bd = d
Bc & Bc -> Jd
Jd = d
Bd & Bd -> Ne
Ne = e
Bd & Jd -> Io
Io = o
Ne & Ne -> Ie
Ie = e
Bd & Jd -> Oe
Oe = e
Oe & Oe -> If
If = f
Ne & Ne -> Af
Af = f
Ne & If -> Tg
Tg = g
Ne & Af -> Eg
Eg = g
Eg & Eg -> Eh
Eh = h
Eg & Tg -> Dh
Dh = h
Dh & Dh -> Wi
Wi = i
Dh & Eh -> Di
Di = i
Di & Di -> Aj
Aj = j
Di & Wi -> Gj
Gj = j
Di & Dj -> Gk
Gk = k
Dj & Dj -> Hk
Hk = k
Dj & Wi -> Hj
Hj = j
Hk & Hk -> Pj
Pj = j
Pj & Pj -> Zk
Zk = k
Hj & Hj -> Qk
Qk = k
Hj & Pj -> Rk
Rk = k
Hj & Qk -> Sk
Sk = k
Hk & Rk -> Qj
Qj = j
Rk & Rk -> Tk
Tk = k
Rk & Sk -> Ej
Ej = j
Hk & Qj -> Tk
Tk = k
Hj & Tk -> Ei
Ei = i
Sk & Tk -> Ki
Ki = i
Ej & Ei -> Ki
Ki = i
Ei & Ki -> Fj
Fj = j
Ei & Ej -> Fi
Fi = i
Fj & Fj -> Fk
Fk = k
Fj & Fi -> Gk
Gk = k
Fi & Fi -> Fk
Fk = k
Ei & Fj -> Gk
Gk = k
Ki & Ki -> Ii
Ii = i
Ki & Gj -> Ji
Ji = i
Fi & Ej -> Ji
Ji = i
Gk & Gk -> Ik
Ik = k
Fi & Gk -> Hk
Hk = k
Gk & Ik -> Ij
Ij = j
Gk & Hk -> Kj
Kj = j
Gk & Hj -> Ik
Ik = i
Ki & Ji -> Il
Il = l
Gk & Kj -> Il
Il = l
Gk & Ij -> Il
Il = l
Hk & Kj -> Il
Il = l
Hj & Il -> Li
Li = i
Hk & Il -> Lj
Lj = j
Lj & Lj -> Lk
Lk = k
Lj & Li -> Lk
Lk = k
Lk & Lk -> Lj
Lj = j
Lk & Lk -> Lj
Lj = j
Lj & Lk -> Lk
Lk = k
Lk & Lk -> Lj
Lj = j
Lj & Lk -> Lk
Lk =
```
