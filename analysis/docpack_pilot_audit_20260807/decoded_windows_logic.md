# Decoded-batch audit examples (document-preserving docpack loader)

## Window 0 (1 documents, 366 pad tokens)
```
1. c0 is teal.
2. c0 is east.
3. If c0 is west and c0 is teal, then c1 is orchid.
4. If c1 is orchid, then c1 is east.
5. If c0 is north and c0 is teal, then c1 is maple.
6. If c1 is maple, then c1 is north.
7. If c0 is south and c0 is teal, then c1 is cobalt.
8. If c1 is cobalt, then c1 is south.
9. If c0 is east and c0 is teal, then c1 is birch.
10. If c1 is birch, then c1 is west.
11. If c1 is south and c1 is cobalt, then c2 is coral.
12. If c2 is coral, then c2 is east.
13. If c1 is west and c1 is birch, then c2 is maple.
14. If c2 is maple, then c2 is west.
15. If c1 is east and c1 is orchid, then c2 is olive.
16. If c2 is olive, then c2 is south.
17. If c1 is north and c1 is maple, then c2 is poppy.
18. If c2 is poppy, then c2 is north.
19. If c2 is east and c2 is coral, then c3 is birch.
20. If c3 is birch, then c3 is north.
21. If c2 is west and c2 is maple, then c3 is cobalt.
22. If c3 is cobalt, then c3 is south.
23. If c2 is north and c2 is poppy, then c3 is maple.
24. If c3 is maple, then c3 is west.
25. If c2 is south and c2 is olive, then c3 is meadow.
26. If c3 is meadow, then c3 is east.
27. If c3 is south and c3 is cobalt, then c4 is amber.
28. If c4 is amber, then c4 is north.
29. If c3 is east and c3 is meadow, then c4 is lime.
30. If c4 is lime, then c4 is south.
31. If c3 is west and c3 is maple, then c4 is cobalt.
32. If c4 is cobalt, then c4 is east.
33. If c3 is north and c3 is birch, then c4 is granite.
34. If c4 is granite, then c4 is west.
35. If c4 is west and c4 is granite, then c5 is granite.
36. If c5 is granite, then c5 is north.
37. If c4 is north and c4 is amber, then c5 is lime.
38. If c5 is lime, then c5 is west.
39. If c4 is south and c4 is lime, then c5 is juniper.
40. If c5 is juniper, then c5 is east.
41. If c4 is east and c4 is cobalt, then c5 is orchid.
42. If c5 is orchid, then c5 is south.
43. If c5 is east and c5 is juniper, then c6 is poppy.
44. If c6 is poppy, then c6 is south.
45. If c5 is north and c5 is granite, then c6 is coral.
46. If c6 is coral, then c6 is east.
47. If c5 is west and c5 is lime, then c6 is juniper.
48. If c6 is juniper, then c6 is west.
49. If c5 is south and c5 is orchid, then c6 is granite.
50. If c6 is granite, then c6 is north.
51. If c6 is east and c6 is coral, then c7 is maple.
52. If c7 is maple, then c7 is east.
53. If c6 is north and c6 is granite, then c7 is lime.
54. If c7 is lime, then c7 is north.
55. If c6 is south and c6 is poppy, then c7 is orchid.
56. If c7 is orchid, then c7 is west.
57. If c6 is west and c6 is juniper, then c7 is coral.
58. If c7 is coral, then c7 is south.
59. If c7 is south and c7 is coral, then c8 is orchid.
60. If c8 is orchid, then c8 is north.
61. If c7 is east and c7 is maple, then c8 is birch.
62. If c8 is birch, then c8 is east.
63. If c7 is west and c7 is orchid, then c8 is amber.
64. If c8 is amber, then c8 is west.
65. If c7 is north and c7 is lime, then c8 is poppy.
66. If c8 is poppy, then c8 is south.
67. If c8 is south and c8 is poppy, then c9 is lime.
68. If c9 is lime, then c9 is west.
69. If c8 is north and c8 is orchid, then c9 is birch.
70. If c9 is birch, then c9 is north.
71. If c8 is east and c8 is birch, then c9 is willow.
72. If c9 is willow, then c9 is south.
73. If c8 is west and c8 is amber, then c9 is maple.
74. If c9 is maple, then c9 is east.
75. If c9 is north and c9 is birch, then c10 is poppy.
76. If c10 is poppy, then c10 is west.
77. If c9 is west and c9 is lime, then c10 is juniper.
78. If c10 is juniper, then c10 is east.
79. If c9 is south and c9 is willow, then c10 is meadow.
80. If c10 is meadow, then c10 is south.
81. If c9 is east and c9 is maple, then c10 is cobalt.
82. If c10 is cobalt, then c10 is north.
83. If c10 is east and c10 is juniper, then c11 is coral.
84. If c11 is coral, then c11 is east.
85. If c10 is north and c10 is cobalt, then c11 is cobalt.
86. If c11 is cobalt, then c11 is south.
87. If c10 is west and c10 is poppy, then c11 is amber.
88. If c11 is amber, then c11 is north.
89. If c10 is south and c10 is meadow, then c11 is orchid.
90. If c11 is orchid, then c11 is west.
91. If c11 is south and c11 is cobalt, then c12 is poppy.
92. If c12 is poppy, then c12 is east.
93. If c11 is north and c11 is amber, then c12 is maple.
94. If c12 is maple, then c12 is west.
95. If c11 is west and c11 is orchid, then c12 is orchid.
96. If c12 is orchid, then c12 is south.
97. If c11 is east and c11 is coral, then c12 is juniper.
98. If c12 is juniper, then c12 is north.
99. If c12 is west and c12 is maple, then c13 is coral.
100. If c13 is coral, then c13 is west.
101. If c12 is south and c12 is orchid, then c13 is meadow.
102. If c13 is meadow, then c13 is east.
103. If c12 is east and c12 is poppy, then c13 is willow.
104. If c13 is willow, then c13 is north.
105. If c12 is north and c12 is juniper, then c13 is cobalt.
106. If c13 is cobalt, then c13 is south.
Which state applies to c13?

Solution:
Definitions:
c0 = c0
c1 = c1
c2 = c2
c3 = c3
c4 = c4
c5 = c5
c6 = c6
c7 = c7
c8 = c8
c9 = c9
c10 = c10
c11 = c11
c12 = c12
c13 = c13
Ax: x is meadow
Bx: x is teal
Cx: x is willow
Dx: x is birch
Ex: x is olive
Fx: x is lime
Gx: x is granite
Hx: x is juniper
Ix: x is maple
Jx: x is coral
Kx: x is orchid
Lx: x is amber
Mx: x is cobalt
Nx: x is poppy
Ox: x is north
Px: x is south
Qx: x is east
Rx: x is west

Formal premises:
B(c0)
Q(c0)
R(c0) & B(c0) -> K(c1)
K(c1) -> Q(c1)
O(c0) & B(c0) -> I(c1)
I(c1) -> O(c1)
P(c0) & B(c0) -> M(c1)
M(c1) -> P(c1)
Q(c0) & B(c0) -> D(c1)
D(c1) -> R(c1)
P(c1) & M(c1) -> J(c2)
J(c2) -> Q(c2)
R(c1) & D(c1) -> I(c2)
I(c2) -> R(c2)
Q(c1) & K(c1) -> E(c2)
E(c2) -> P(c2)
O(c1) & I(c1) -> N(c2)
N(c2) -> O(c2)
Q(c2) & J(c2) -> D(c3)
D(c3) -> O(c3)
R(c2) & I(c2) -> M(c3)
M(c3) -> P(c3)
O(c2) & N(c2) -> I(c3)
I(c3) -> R(c3)
P(c2) & E(c2) -> A(c3)
A(c3) -> Q(c3)
P(c3) & M(c3) -> L(c4)
L(c4) -> O(c4)
Q(c3) & A(c3) -> F(c4)
F(c4) -> P(c4)
R(c3) & I(c3) -> M(c4)
M(c4) -> Q(c4)
O(c3) & D(c3) -> G(c4)
G(c4) -> R(c4)
R(c4) & G(c4) -> G(c5)
G(c5) -> O(c5)
O(c4) & L(c4) -> F(c5)
F(c5) -> R(c5)
P(c4) & F(c4) -> H(c5)
H(c5) -> Q(c5)
Q(c4) & M(c4) -> K(c5)
K(c5) -> P(c5)
Q(c5) & H(c5) -> N(c6)
N(c6) -> P(c6)
O(c5) & G(c5) -> J(c6)
J(c6) -> Q(c6)
R(c5) & F(c5) -> H(c6)
H(c6) -> R(c6)
P(c5) & K(c5) -> G(c6)
G(c6) -> O(c6)
Q(c6) & J(c6) -> I(c7)
I(c7) -> Q(c7)
O(c6) & G(c6) -> F(c7)
F(c7) -> O(c7)
P(c6) & N(c6) -> K(c7)
K(c7) -> R(c7)
R(c6) & H(c6) -> J(c7)
J(c7) -> P(c7)
P(c7) & J(c7) -> K(c8)
K(c8) -> O(c8)
Q(c7) & I(c7) -> D(c8)
D(c8) -> Q(c8)
R(c7) & K(c7) -> L(c8)
L(c8) -> R(c8)
O(c7) & F(c7) -> N(c8)
N(c8) -> P(c8)
P(c8) & N(c8) -> F(c9)
F(c9) -> R(c9)
O(c8) & K(c8) -> D(c9)
D(c9) -> O(c9)
Q(c8) & D(c8) -> C(c9)
C(c9) -> P(c9)
R(c8) & L(c8) -> I(c9)
I(c9) -> Q(c9)
O(c9) & D(c9) -> N(c10)
N(c10) -> R(c10)
R(c9) & F(c9) -> H(c10)
H(c10) -> Q(c10)
P(c9) & C(c9) -> A(c10)
A(c10) -> P(c10)
Q(c9) & I(c9) -> M(c10)
M(c10) -> O(c10)
Q(c10) & H(c10) -> J(c11)
J(c11) -> Q(c11)
O(c10) & M(c10) -> M(c11)
M(c11) -> P(c11)
R(c10) & N(c10) -> L(c11)
L(c11) -> O(c11)
P(c10) & A(c10) -> K(c11)
K(c11) -> R(c11)
P(c11) & M(c11) -> N(c12)
N(c12) -> Q(c12)
O(c11) & L(c11) -> I(c12)
I(c12) -> R(c12)
R(c11) & K(c11) -> K(c12)
K(c12) -> P(c12)
Q(c11) & J(c11) -> H(c12)
H(c12) -> O(c12)
R(c12) & I(c12) -> J(c13)
J(c13) -> R(c13)
P(c12) & K(c12) -> A(c13)
A(c13) -> Q(c13)
Q(c12) & N(c12) -> C(c13)
C(c13) -> O(c13)
O(c12) & H(c12) -> M(c13)
M(c13) -> P(c13)

Derivation:
B(c0) ; R
Q(c0) ; R
D(c1) ; ->E
R(c1) ; ->E
I(c2) ; ->E
R(c2) ; ->E
M(c3) ; ->E
P(c3) ; ->E
L(c4) ; ->E
O(c4) ; ->E
F(c5) ; ->E
R(c5) ; ->E
H(c6) ; ->E
R(c6) ; ->E
J(c7) ; ->E
P(c7) ; ->E
K(c8) ; ->E
O(c8) ; ->E
D(c9) ; ->E
O(c9) ; ->E
N(c10) ; ->E
R(c10) ; ->E
L(c11) ; ->E
O(c11) ; ->E
I(c12) ; ->E
R(c12) ; ->E
J(c13) ; ->E

Final answer: coral<|endoftext|>
```

## Window 1 (2 documents, 474 pad tokens)
```
1. c0 is pearl.
2. c0 is south.
3. If c0 is south and c0 is pearl, then c1 is harbor.
4. If c1 is harbor, then c1 is east.
5. If c0 is west and c0 is pearl, then c1 is willow.
6. If c1 is willow, then c1 is north.
7. If c0 is east and c0 is pearl, then c1 is laurel.
8. If c1 is laurel, then c1 is west.
9. If c0 is north and c0 is pearl, then c1 is juniper.
10. If c1 is juniper, then c1 is south.
11. If c1 is east and c1 is harbor, then c2 is laurel.
12. If c2 is laurel, then c2 is east.
13. If c1 is west and c1 is laurel, then c2 is willow.
14. If c2 is willow, then c2 is north.
15. If c1 is south and c1 is juniper, then c2 is slate.
16. If c2 is slate, then c2 is west.
17. If c1 is north and c1 is willow, then c2 is harbor.
18. If c2 is harbor, then c2 is south.
19. If c2 is south and c2 is harbor, then c3 is granite.
20. If c3 is granite, then c3 is north.
21. If c2 is east and c2 is laurel, then c3 is willow.
22. If c3 is willow, then c3 is south.
23. If c2 is west and c2 is slate, then c3 is juniper.
24. If c3 is juniper, then c3 is west.
25. If c2 is north and c2 is willow, then c3 is ivory.
26. If c3 is ivory, then c3 is east.
27. If c3 is south and c3 is willow, then c4 is harbor.
28. If c4 is harbor, then c4 is south.
29. If c3 is east and c3 is ivory, then c4 is granite.
30. If c4 is granite, then c4 is east.
31. If c3 is north and c3 is granite, then c4 is juniper.
32. If c4 is juniper, then c4 is west.
33. If c3 is west and c3 is juniper, then c4 is willow.
34. If c4 is willow, then c4 is north.
35. If c4 is west and c4 is juniper, then c5 is laurel.
36. If c5 is laurel, then c5 is north.
37. If c4 is east and c4 is granite, then c5 is slate.
38. If c5 is slate, then c5 is west.
39. If c4 is north and c4 is willow, then c5 is meadow.
40. If c5 is meadow, then c5 is east.
41. If c4 is south and c4 is harbor, then c5 is elm.
42. If c5 is elm, then c5 is south.
43. If c5 is west and c5 is slate, then c6 is laurel.
44. If c6 is laurel, then c6 is north.
45. If c5 is south and c5 is elm, then c6 is harbor.
46. If c6 is harbor, then c6 is east.
47. If c5 is north and c5 is laurel, then c6 is granite.
48. If c6 is granite, then c6 is west.
49. If c5 is east and c5 is meadow, then c6 is slate.
50. If c6 is slate, then c6 is south.
51. If c6 is north and c6 is laurel, then c7 is elm.
52. If c7 is elm, then c7 is north.
53. If c6 is east and c6 is harbor, then c7 is willow.
54. If c7 is willow, then c7 is east.
55. If c6 is west and c6 is granite, then c7 is meadow.
56. If c7 is meadow, then c7 is west.
57. If c6 is south and c6 is slate, then c7 is slate.
58. If c7 is slate, then c7 is south.
59. If c7 is north and c7 is elm, then c8 is meadow.
60. If c8 is meadow, then c8 is east.
61. If c7 is west and c7 is meadow, then c8 is juniper.
62. If c8 is juniper, then c8 is west.
63. If c7 is south and c7 is slate, then c8 is granite.
64. If c8 is granite, then c8 is north.
65. If c7 is east and c7 is willow, then c8 is harbor.
66. If c8 is harbor, then c8 is south.
67. If c8 is south and c8 is harbor, then c9 is willow.
68. If c9 is willow, then c9 is west.
69. If c8 is east and c8 is meadow, then c9 is juniper.
70. If c9 is juniper, then c9 is north.
71. If c8 is north and c8 is granite, then c9 is slate.
72. If c9 is slate, then c9 is east.
73. If c8 is west and c8 is juniper, then c9 is laurel.
74. If c9 is laurel, then c9 is south.
Which state applies to c9?

Solution:
Definitions:
c0 = c0
c1 = c1
c2 = c2
c3 = c3
c4 = c4
c5 = c5
c6 = c6
c7 = c7
c8 = c8
c9 = c9
Ax: x is laurel
Bx: x is elm
Cx: x is slate
Dx: x is willow
Ex: x is pearl
Fx: x is ivory
Gx: x is meadow
Hx: x is granite
Ix: x is juniper
Jx: x is harbor
Kx: x is north
Lx: x is south
Mx: x is east
Nx: x is west

Formal premises:
E(c0)
L(c0)
L(c0) & E(c0) -> J(c1)
J(c1) -> M(c1)
N(c0) & E(c0) -> D(c1)
D(c1) -> K(c1)
M(c0) & E(c0) -> A(c1)
A(c1) -> N(c1)
K(c0) & E(c0) -> I(c1)
I(c1) -> L(c1)
M(c1) & J(c1) -> A(c2)
A(c2) -> M(c2)
N(c1) & A(c1) -> D(c2)
D(c2) -> K(c2)
L(c1) & I(c1) -> C(c2)
C(c2) -> N(c2)
K(c1) & D(c1) -> J(c2)
J(c2) -> L(c2)
L(c2) & J(c2) -> H(c3)
H(c3) -> K(c3)
M(c2) & A(c2) -> D(c3)
D(c3) -> L(c3)
N(c2) & C(c2) -> I(c3)
I(c3) -> N(c3)
K(c2) & D(c2) -> F(c3)
F(c3) -> M(c3)
L(c3) & D(c3) -> J(c4)
J(c4) -> L(c4)
M(c3) & F(c3) -> H(c4)
H(c4) -> M(c4)
K(c3) & H(c3) -> I(c4)
I(c4) -> N(c4)
N(c3) & I(c3) -> D(c4)
D(c4) -> K(c4)
N(c4) & I(c4) -> A(c5)
A(c5) -> K(c5)
M(c4) & H(c4) -> C(c5)
C(c5) -> N(c5)
K(c4) & D(c4) -> G(c5)
G(c5) -> M(c5)
L(c4) & J(c4) -> B(c5)
B(c5) -> L(c5)
N(c5) & C(c5) -> A(c6)
A(c6) -> K(c6)
L(c5) & B(c5) -> J(c6)
J(c6) -> M(c6)
K(c5) & A(c5) -> H(c6)
H(c6) -> N(c6)
M(c5) & G(c5) -> C(c6)
C(c6) -> L(c6)
K(c6) & A(c6) -> B(c7)
B(c7) -> K(c7)
M(c6) & J(c6) -> D(c7)
D(c7) -> M(c7)
N(c6) & H(c6) -> G(c7)
G(c7) -> N(c7)
L(c6) & C(c6) -> C(c7)
C(c7) -> L(c7)
K(c7) & B(c7) -> G(c8)
G(c8) -> M(c8)
N(c7) & G(c7) -> I(c8)
I(c8) -> N(c8)
L(c7) & C(c7) -> H(c8)
H(c8) -> K(c8)
M(c7) & D(c7) -> J(c8)
J(c8) -> L(c8)
L(c8) & J(c8) -> D(c9)
D(c9) -> N(c9)
M(c8) & G(c8) -> I(c9)
I(c9) -> K(c9)
K(c8) & H(c8) -> C(c9)
C(c9) -> M(c9)
N(c8) & I(c8) -> A(c9)
A(c9) -> L(c9)

Derivation:
E(c0) ; R
L(c0) ; R
J(c1) ; ->E
M(c1) ; ->E
A(c2) ; ->E
M(c2) ; ->E
D(c3) ; ->E
L(c3) ; ->E
J(c4) ; ->E
L(c4) ; ->E
B(c5) ; ->E
L(c5) ; ->E
J(c6) ; ->E
M(c6) ; ->E
D(c7) ; ->E
M(c7) ; ->E
J(c8) ; ->E
L(c8) ; ->E
D(c9) ; ->E

Final answer: willow<|endoftext|>1. c0 is harbor.
2. c0 is south.
3. If c0 is north and c0 is harbor, then c1 is violet.
4. If c1 is violet, then c1 is west.
5. If c0 is south and c0 is harbor, then c1 is lime.
6. If c1 is lime, then c1 is north.
7. If c0 is east and c0 is harbor, then c1 is birch.
8. If c1 is birch, then c1 is south.
9. If c0 is west and c0 is harbor, then c1 is slate.
10. If c1 is slate, then c1 is east.
11. If c1 is west and c1 is violet, then c2 is slate.
12. If c2 is slate, then c2 is north.
13. If c1 is south and c1 is birch, then c2 is violet.
14. If c2 is violet, then c2 is east.
15. If c1 is east and c1 is slate, then c2 is birch.
16. If c2 is birch, then c2 is west.
17. If c1 is north and c1 is lime, then c2 is lime.
18. If c2 is lime, then c2 is south.
19. If c2 is west and c2 is birch, then c3 is slate.
20. If c3 is slate, then c3 is west.
21. If c2 is north and c2 is slate, then c3 is violet.
22. If c3 is violet, then c3 is east.
23. If c2 is east and c2 is violet, then c3 is birch.
24. If c3 is birch, then c3 is north.
25. If c2 is south and c2 is lime, then c3 is lime.
26. If c3 is lime, then c3 is south.
27. If c3 is south and c3 is lime, then c4 is lime.
28. If c4 is lime, then c4 is north.
29. If c3 is north and c3 is birch, then c4 is slate.
30. If c4 is slate, then c4 is east.
31. If c3 is west and c3 is slate, then c4 is birch.
32. If c4 is birch, then c4 is west.
33. If c3 is east and c3 is violet, then c4 is violet.
34. If c4 is violet, then c4 is south.
Which state applies to c4?

Solution:
Definitions:
c0 = c0
c1 = c1
c2 = c2
c3 = c3
c4 = c4
Ax: x is lime
Bx: x is violet
Cx: x is harbor
Dx: x is birch
Ex: x is slate
Fx: x is north
Gx: x is south
Hx: x is east
Ix: x is west

Formal premises:
C(c0)
G(c0)
F(c0) & C(c0) -> B(c1)
B(c1) -> I(c1)
G(c0) & C(c0) -> A(c1)
A(c1) -> F(c1)
H(c0) & C(c0) -> D(c1)
D(c1) -> G(c1)
I(c0) & C(c0) -> E(c1)
E(c1) -> H(c1)
I(c1) & B(c1) -> E(c2)
E(c2) -> F(c2)
G(c1) & D(c1) -> B(c2)
B(c2) -> H(c2)
H(c1) & E(c1) -> D(c2)
D(c2) -> I(c2)
F(c1) & A(c1) -> A(c2)
A(c2) -> G(c2)
I(c2) & D(c2) -> E(c3)
E(c3) -> I(c3)
F(c2) & E(c2) -> B(c3)
B(c3) -> H(c3)
H(c2) & B(c2) -> D(c3)
D(c3) -> F(c3)
G(c2) & A(c2) -> A(c3)
A(c3) -> G(c3)
G(c3) & A(c3) -> A(c4)
A(c4) -> F(c4)
F(c3) & D(c3) -> E(c4)
E(c4) -> H(c4)
I(c3) & E(c3) -> D(c4)
D(c4) -> I(c4)
H(c3) & B(c3) -> B(c4)
B(c4) -> G(c4)

Derivation:
C(c0) ; R
G(c0) ; R
A(c1) ; ->E
F(c1) ; ->E
A(c2) ; ->E
G(c2) ; ->E
A(c3) ; ->E
G(c3) ; ->E
A(c4) ; ->E

Final answer: lime<|endoftext|>
```

## Window 2 (1 documents, 104 pad tokens)
```
1. c0 is granite.
2. c0 is east.
3. If c0 is south and c0 is granite, then c1 is harbor.
4. If c1 is harbor, then c1 is north.
5. If c0 is east and c0 is granite, then c1 is pearl.
6. If c1 is pearl, then c1 is south.
7. If c0 is north and c0 is granite, then c1 is ivory.
8. If c1 is ivory, then c1 is west.
9. If c0 is west and c0 is granite, then c1 is poppy.
10. If c1 is poppy, then c1 is east.
11. If c1 is east and c1 is poppy, then c2 is hazel.
12. If c2 is hazel, then c2 is east.
13. If c1 is north and c1 is harbor, then c2 is maple.
14. If c2 is maple, then c2 is west.
15. If c1 is south and c1 is pearl, then c2 is elm.
16. If c2 is elm, then c2 is south.
17. If c1 is west and c1 is ivory, then c2 is ruby.
18. If c2 is ruby, then c2 is north.
19. If c2 is west and c2 is maple, then c3 is harbor.
20. If c3 is harbor, then c3 is south.
21. If c2 is east and c2 is hazel, then c3 is hazel.
22. If c3 is hazel, then c3 is west.
23. If c2 is south and c2 is elm, then c3 is ivory.
24. If c3 is ivory, then c3 is north.
25. If c2 is north and c2 is ruby, then c3 is cedar.
26. If c3 is cedar, then c3 is east.
27. If c3 is south and c3 is harbor, then c4 is cedar.
28. If c4 is cedar, then c4 is east.
29. If c3 is north and c3 is ivory, then c4 is hazel.
30. If c4 is hazel, then c4 is south.
31. If c3 is east and c3 is cedar, then c4 is harbor.
32. If c4 is harbor, then c4 is west.
33. If c3 is west and c3 is hazel, then c4 is teal.
34. If c4 is teal, then c4 is north.
35. If c4 is south and c4 is hazel, then c5 is pearl.
36. If c5 is pearl, then c5 is east.
37. If c4 is north and c4 is teal, then c5 is harbor.
38. If c5 is harbor, then c5 is west.
39. If c4 is east and c4 is cedar, then c5 is poppy.
40. If c5 is poppy, then c5 is south.
41. If c4 is west and c4 is harbor, then c5 is ruby.
42. If c5 is ruby, then c5 is north.
43. If c5 is east and c5 is pearl, then c6 is juniper.
44. If c6 is juniper, then c6 is west.
45. If c5 is north and c5 is ruby, then c6 is coral.
46. If c6 is coral, then c6 is south.
47. If c5 is south and c5 is poppy, then c6 is cedar.
48. If c6 is cedar, then c6 is east.
49. If c5 is west and c5 is harbor, then c6 is amber.
50. If c6 is amber, then c6 is north.
51. If c6 is north and c6 is amber, then c7 is pearl.
52. If c7 is pearl, then c7 is north.
53. If c6 is west and c6 is juniper, then c7 is amber.
54. If c7 is amber, then c7 is east.
55. If c6 is south and c6 is coral, then c7 is juniper.
56. If c7 is juniper, then c7 is west.
57. If c6 is east and c6 is cedar, then c7 is harbor.
58. If c7 is harbor, then c7 is south.
59. If c7 is east and c7 is amber, then c8 is teal.
60. If c8 is teal, then c8 is east.
61. If c7 is south and c7 is harbor, then c8 is juniper.
62. If c8 is juniper, then c8 is south.
63. If c7 is west and c7 is juniper, then c8 is poppy.
64. If c8 is poppy, then c8 is west.
65. If c7 is north and c7 is pearl, then c8 is birch.
66. If c8 is birch, then c8 is north.
67. If c8 is east and c8 is teal, then c9 is amber.
68. If c9 is amber, then c9 is east.
69. If c8 is south and c8 is juniper, then c9 is cedar.
70. If c9 is cedar, then c9 is south.
71. If c8 is north and c8 is birch, then c9 is coral.
72. If c9 is coral, then c9 is north.
73. If c8 is west and c8 is poppy, then c9 is harbor.
74. If c9 is harbor, then c9 is west.
75. If c9 is east and c9 is amber, then c10 is amber.
76. If c10 is amber, then c10 is north.
77. If c9 is west and c9 is harbor, then c10 is pearl.
78. If c10 is pearl, then c10 is west.
79. If c9 is south and c9 is cedar, then c10 is maple.
80. If c10 is maple, then c10 is south.
81. If c9 is north and c9 is coral, then c10 is ruby.
82. If c10 is ruby, then c10 is east.
83. If c10 is south and c10 is maple, then c11 is amber.
84. If c11 is amber, then c11 is south.
85. If c10 is west and c10 is pearl, then c11 is poppy.
86. If c11 is poppy, then c11 is west.
87. If c10 is north and c10 is amber, then c11 is hazel.
88. If c11 is hazel, then c11 is north.
89. If c10 is east and c10 is ruby, then c11 is ivory.
90. If c11 is ivory, then c11 is east.
91. If c11 is east and c11 is ivory, then c12 is hazel.
92. If c12 is hazel, then c12 is north.
93. If c11 is west and c11 is poppy, then c12 is teal.
94. If c12 is teal, then c12 is east.
95. If c11 is north and c11 is hazel, then c12 is elm.
96. If c12 is elm, then c12 is south.
97. If c11 is south and c11 is amber, then c12 is pearl.
98. If c12 is pearl, then c12 is west.
99. If c12 is west and c12 is pearl, then c13 is pearl.
100. If c13 is pearl, then c13 is west.
101. If c12 is north and c12 is hazel, then c13 is cedar.
102. If c13 is cedar, then c13 is south.
103. If c12 is east and c12 is teal, then c13 is amber.
104. If c13 is amber, then c13 is east.
105. If c12 is south and c12 is elm, then c13 is harbor.
106. If c13 is harbor, then c13 is north.
107. If c13 is west and c13 is pearl, then c14 is elm.
108. If c14 is elm, then c14 is south.
109. If c13 is east and c13 is amber, then c14 is ruby.
110. If c14 is ruby, then c14 is west.
111. If c13 is north and c13 is harbor, then c14 is coral.
112. If c14 is coral, then c14 is north.
113. If c13 is south and c13 is cedar, then c14 is amber.
114. If c14 is amber, then c14 is east.
Which state applies to c14?

Solution:
Definitions:
c0 = c0
c1 = c1
c2 = c2
c3 = c3
c4 = c4
c5 = c5
c6 = c6
c7 = c7
c8 = c8
c9 = c9
c10 = c10
c11 = c11
c12 = c12
c13 = c13
c14 = c14
Ax: x is juniper
Bx: x is amber
Cx: x is poppy
Dx: x is coral
Ex: x is harbor
Fx: x is ivory
Gx: x is hazel
Hx: x is teal
Ix: x is pearl
Jx: x is granite
Kx: x is cedar
Lx: x is maple
Mx: x is birch
Nx: x is ruby
Ox: x is elm
Px: x is north
Qx: x is south
Rx: x is east
Sx: x is west

Formal premises:
J(c0)
R(c0)
Q(c0) & J(c0) -> E(c1)
E(c1) -> P(c1)
R(c0) & J(c0) -> I(c1)
I(c1) -> Q(c1)
P(c0) & J(c0) -> F(c1)
F(c1) -> S(c1)
S(c0) & J(c0) -> C(c1)
C(c1) -> R(c1)
R(c1) & C(c1) -> G(c2)
G(c2) -> R(c2)
P(c1) & E(c1) -> L(c2)
L(c2) -> S(c2)
Q(c1) & I(c1) -> O(c2)
O(c2) -> Q(c2)
S(c1) & F(c1) -> N(c2)
N(c2) -> P(c2)
S(c2) & L(c2) -> E(c3)
E(c3) -> Q(c3)
R(c2) & G(c2) -> G(c3)
G(c3) -> S(c3)
Q(c2) & O(c2) -> F(c3)
F(c3) -> P(c3)
P(c2) & N(c2) -> K(c3)
K(c3) -> R(c3)
Q(c3) & E(c3) -> K(c4)
K(c4) -> R(c4)
P(c3) & F(c3) -> G(c4)
G(c4) -> Q(c4)
R(c3) & K(c3) -> E(c4)
E(c4) -> S(c4)
S(c3) & G(c3) -> H(c4)
H(c4) -> P(c4)
Q(c4) & G(c4) -> I(c5)
I(c5) -> R(c5)
P(c4) & H(c4) -> E(c5)
E(c5) -> S(c5)
R(c4) & K(c4) -> C(c5)
C(c5) -> Q(c5)
S(c4) & E(c4) -> N(c5)
N(c5) -> P(c5)
R(c5) & I(c5) -> A(c6)
A(c6) -> S(c6)
P(c5) & N(c5) -> D(c6)
D(c6) -> Q(c6)
Q(c5) & C(c5) -> K(c6)
K(c6) -> R(c6)
S(c5) & E(c5) -> B(c6)
B(c6) -> P(c6)
P(c6) & B(c6) -> I(c7)
I(c7) -> P(c7)
S(c6) & A(c6) -> B(c7)
B(c7) -> R(c7)
Q(c6) & D(c6) -> A(c7)
A(c7) -> S(c7)
R(c6) & K(c6) -> E(c7)
E(c7) -> Q(c7)
R(c7) & B(c7) -> H(c8)
H(c8) -> R(c8)
Q(c7) & E(c7) -> A(c8)
A(c8) -> Q(c8)
S(c7) & A(c7) -> C(c8)
C(c8) -> S(c8)
P(c7) & I(c7) -> M(c8)
M(c8) -> P(c8)
R(c8) & H(c8) -> B(c9)
B(c9) -> R(c9)
Q(c8) & A(c8) -> K(c9)
K(c9) -> Q(c9)
P(c8) & M(c8) -> D(c9)
D(c9) -> P(c9)
S(c8) & C(c8) -> E(c9)
E(c9) -> S(c9)
R(c9) & B(c9) -> B(c10)
B(c10) -> P(c10)
S(c9) & E(c9) -> I(c10)
I(c10) -> S(c10)
Q(c9) & K(c9) -> L(c10)
L(c10) -> Q(c10)
P(c9) & D(c9) -> N(c10)
N(c10) -> R(c10)
Q(c10) & L(c10) -> B(c11)
B(c11) -> Q(c11)
S(c10) & I(c10) -> C(c11)
C(c11) -> S(c11)
P(c10) & B(c10) -> G(c11)
G(c11) -> P(c11)
R(c10) & N(c10) -> F(c11)
F(c11) -> R(c11)
R(c11) & F(c11) -> G(c12)
G(c12) -> P(c12)
S(c11) & C(c11) -> H(c12)
H(c12) -> R(c12)
P(c11) & G(c11) -> O(c12)
O(c12) -> Q(c12)
Q(c11) & B(c11) -> I(c12)
I(c12) -> S(c12)
S(c12) & I(c12) -> I(c13)
I(c13) -> S(c13)
P(c12) & G(c12) -> K(c13)
K(c13) -> Q(c13)
R(c12) & H(c12) -> B(c13)
B(c13) -> R(c13)
Q(c12) & O(c12) -> E(c13)
E(c13) -> P(c13)
S(c13) & I(c13) -> O(c14)
O(c14) -> Q(c14)
R(c13) & B(c13) -> N(c14)
N(c14) -> S(c14)
P(c13) & E(c13) -> D(c14)
D(c14) -> P(c14)
Q(c13) & K(c13) -> B(c14)
B(c14) -> R(c14)

Derivation:
J(c0) ; R
R(c0) ; R
I(c1) ; ->E
Q(c1) ; ->E
O(c2) ; ->E
Q(c2) ; ->E
F(c3) ; ->E
P(c3) ; ->E
G(c4) ; ->E
Q(c4) ; ->E
I(c5) ; ->E
R(c5) ; ->E
A(c6) ; ->E
S(c6) ; ->E
B(c7) ; ->E
R(c7) ; ->E
H(c8) ; ->E
R(c8) ; ->E
B(c9) ; ->E
R(c9) ; ->E
B(c10) ; ->E
P(c10) ; ->E
G(c11) ; ->E
P(c11) ; ->E
O(c12) ; ->E
Q(c12) ; ->E
E(c13) ; ->E
P(c13) ; ->E
D(c14) ; ->E

Final answer: coral<|endoftext|>
```

## Window 3 (2 documents, 361 pad tokens)
```
1. c0 is pearl.
2. c0 is north.
3. If c0 is south and c0 is pearl, then c1 is violet.
4. If c1 is violet, then c1 is east.
5. If c0 is west and c0 is pearl, then c1 is lime.
6. If c1 is lime, then c1 is north.
7. If c0 is east and c0 is pearl, then c1 is orchid.
8. If c1 is orchid, then c1 is west.
9. If c0 is north and c0 is pearl, then c1 is amber.
10. If c1 is amber, then c1 is south.
11. If c1 is east and c1 is violet, then c2 is orchid.
12. If c2 is orchid, then c2 is east.
13. If c1 is west and c1 is orchid, then c2 is harbor.
14. If c2 is harbor, then c2 is south.
15. If c1 is south and c1 is amber, then c2 is coral.
16. If c2 is coral, then c2 is north.
17. If c1 is north and c1 is lime, then c2 is amber.
18. If c2 is amber, then c2 is west.
19. If c2 is west and c2 is amber, then c3 is ruby.
20. If c3 is ruby, then c3 is south.
21. If c2 is south and c2 is harbor, then c3 is violet.
22. If c3 is violet, then c3 is east.
23. If c2 is north and c2 is coral, then c3 is orchid.
24. If c3 is orchid, then c3 is north.
25. If c2 is east and c2 is orchid, then c3 is lime.
26. If c3 is lime, then c3 is west.
27. If c3 is west and c3 is lime, then c4 is harbor.
28. If c4 is harbor, then c4 is east.
29. If c3 is north and c3 is orchid, then c4 is ivory.
30. If c4 is ivory, then c4 is north.
31. If c3 is south and c3 is ruby, then c4 is coral.
32. If c4 is coral, then c4 is south.
33. If c3 is east and c3 is violet, then c4 is meadow.
34. If c4 is meadow, then c4 is west.
35. If c4 is west and c4 is meadow, then c5 is coral.
36. If c5 is coral, then c5 is east.
37. If c4 is south and c4 is coral, then c5 is orchid.
38. If c5 is orchid, then c5 is west.
39. If c4 is north and c4 is ivory, then c5 is willow.
40. If c5 is willow, then c5 is north.
41. If c4 is east and c4 is harbor, then c5 is lime.
42. If c5 is lime, then c5 is south.
43. If c5 is west and c5 is orchid, then c6 is coral.
44. If c6 is coral, then c6 is south.
45. If c5 is south and c5 is lime, then c6 is poppy.
46. If c6 is poppy, then c6 is west.
47. If c5 is east and c5 is coral, then c6 is violet.
48. If c6 is violet, then c6 is east.
49. If c5 is north and c5 is willow, then c6 is harbor.
50. If c6 is harbor, then c6 is north.
51. If c6 is west and c6 is poppy, then c7 is meadow.
52. If c7 is meadow, then c7 is south.
53. If c6 is north and c6 is harbor, then c7 is violet.
54. If c7 is violet, then c7 is east.
55. If c6 is east and c6 is violet, then c7 is laurel.
56. If c7 is laurel, then c7 is north.
57. If c6 is south and c6 is coral, then c7 is coral.
58. If c7 is coral, then c7 is west.
59. If c7 is west and c7 is coral, then c8 is coral.
60. If c8 is coral, then c8 is north.
61. If c7 is east and c7 is violet, then c8 is lime.
62. If c8 is lime, then c8 is west.
63. If c7 is north and c7 is laurel, then c8 is violet.
64. If c8 is violet, then c8 is south.
65. If c7 is south and c7 is meadow, then c8 is poppy.
66. If c8 is poppy, then c8 is east.
67. If c8 is east and c8 is poppy, then c9 is amber.
68. If c9 is amber, then c9 is south.
69. If c8 is south and c8 is violet, then c9 is coral.
70. If c9 is coral, then c9 is north.
71. If c8 is north and c8 is coral, then c9 is harbor.
72. If c9 is harbor, then c9 is east.
73. If c8 is west and c8 is lime, then c9 is violet.
74. If c9 is violet, then c9 is west.
75. If c9 is east and c9 is harbor, then c10 is lime.
76. If c10 is lime, then c10 is east.
77. If c9 is west and c9 is violet, then c10 is ruby.
78. If c10 is ruby, then c10 is west.
79. If c9 is north and c9 is coral, then c10 is violet.
80. If c10 is violet, then c10 is north.
81. If c9 is south and c9 is amber, then c10 is harbor.
82. If c10 is harbor, then c10 is south.
83. If c10 is south and c10 is harbor, then c11 is amber.
84. If c11 is amber, then c11 is south.
85. If c10 is west and c10 is ruby, then c11 is willow.
86. If c11 is willow, then c11 is east.
87. If c10 is north and c10 is violet, then c11 is violet.
88. If c11 is violet, then c11 is west.
89. If c10 is east and c10 is lime, then c11 is coral.
90. If c11 is coral, then c11 is north.
91. If c11 is west and c11 is violet, then c12 is poppy.
92. If c12 is poppy, then c12 is south.
93. If c11 is north and c11 is coral, then c12 is ivory.
94. If c12 is ivory, then c12 is west.
95. If c11 is south and c11 is amber, then c12 is meadow.
96. If c12 is meadow, then c12 is east.
97. If c11 is east and c11 is willow, then c12 is willow.
98. If c12 is willow, then c12 is north.
Which state applies to c12?

Solution:
Definitions:
c0 = c0
c1 = c1
c2 = c2
c3 = c3
c4 = c4
c5 = c5
c6 = c6
c7 = c7
c8 = c8
c9 = c9
c10 = c10
c11 = c11
c12 = c12
Ax: x is harbor
Bx: x is ruby
Cx: x is pearl
Dx: x is laurel
Ex: x is orchid
Fx: x is coral
Gx: x is violet
Hx: x is willow
Ix: x is ivory
Jx: x is meadow
Kx: x is lime
Lx: x is amber
Mx: x is poppy
Nx: x is north
Ox: x is south
Px: x is east
Qx: x is west

Formal premises:
C(c0)
N(c0)
O(c0) & C(c0) -> G(c1)
G(c1) -> P(c1)
Q(c0) & C(c0) -> K(c1)
K(c1) -> N(c1)
P(c0) & C(c0) -> E(c1)
E(c1) -> Q(c1)
N(c0) & C(c0) -> L(c1)
L(c1) -> O(c1)
P(c1) & G(c1) -> E(c2)
E(c2) -> P(c2)
Q(c1) & E(c1) -> A(c2)
A(c2) -> O(c2)
O(c1) & L(c1) -> F(c2)
F(c2) -> N(c2)
N(c1) & K(c1) -> L(c2)
L(c2) -> Q(c2)
Q(c2) & L(c2) -> B(c3)
B(c3) -> O(c3)
O(c2) & A(c2) -> G(c3)
G(c3) -> P(c3)
N(c2) & F(c2) -> E(c3)
E(c3) -> N(c3)
P(c2) & E(c2) -> K(c3)
K(c3) -> Q(c3)
Q(c3) & K(c3) -> A(c4)
A(c4) -> P(c4)
N(c3) & E(c3) -> I(c4)
I(c4) -> N(c4)
O(c3) & B(c3) -> F(c4)
F(c4) -> O(c4)
P(c3) & G(c3) -> J(c4)
J(c4) -> Q(c4)
Q(c4) & J(c4) -> F(c5)
F(c5) -> P(c5)
O(c4) & F(c4) -> E(c5)
E(c5) -> Q(c5)
N(c4) & I(c4) -> H(c5)
H(c5) -> N(c5)
P(c4) & A(c4) -> K(c5)
K(c5) -> O(c5)
Q(c5) & E(c5) -> F(c6)
F(c6) -> O(c6)
O(c5) & K(c5) -> M(c6)
M(c6) -> Q(c6)
P(c5) & F(c5) -> G(c6)
G(c6) -> P(c6)
N(c5) & H(c5) -> A(c6)
A(c6) -> N(c6)
Q(c6) & M(c6) -> J(c7)
J(c7) -> O(c7)
N(c6) & A(c6) -> G(c7)
G(c7) -> P(c7)
P(c6) & G(c6) -> D(c7)
D(c7) -> N(c7)
O(c6) & F(c6) -> F(c7)
F(c7) -> Q(c7)
Q(c7) & F(c7) -> F(c8)
F(c8) -> N(c8)
P(c7) & G(c7) -> K(c8)
K(c8) -> Q(c8)
N(c7) & D(c7) -> G(c8)
G(c8) -> O(c8)
O(c7) & J(c7) -> M(c8)
M(c8) -> P(c8)
P(c8) & M(c8) -> L(c9)
L(c9) -> O(c9)
O(c8) & G(c8) -> F(c9)
F(c9) -> N(c9)
N(c8) & F(c8) -> A(c9)
A(c9) -> P(c9)
Q(c8) & K(c8) -> G(c9)
G(c9) -> Q(c9)
P(c9) & A(c9) -> K(c10)
K(c10) -> P(c10)
Q(c9) & G(c9) -> B(c10)
B(c10) -> Q(c10)
N(c9) & F(c9) -> G(c10)
G(c10) -> N(c10)
O(c9) & L(c9) -> A(c10)
A(c10) -> O(c10)
O(c10) & A(c10) -> L(c11)
L(c11) -> O(c11)
Q(c10) & B(c10) -> H(c11)
H(c11) -> P(c11)
N(c10) & G(c10) -> G(c11)
G(c11) -> Q(c11)
P(c10) & K(c10) -> F(c11)
F(c11) -> N(c11)
Q(c11) & G(c11) -> M(c12)
M(c12) -> O(c12)
N(c11) & F(c11) -> I(c12)
I(c12) -> Q(c12)
O(c11) & L(c11) -> J(c12)
J(c12) -> P(c12)
P(c11) & H(c11) -> H(c12)
H(c12) -> N(c12)

Derivation:
C(c0) ; R
N(c0) ; R
L(c1) ; ->E
O(c1) ; ->E
F(c2) ; ->E
N(c2) ; ->E
E(c3) ; ->E
N(c3) ; ->E
I(c4) ; ->E
N(c4) ; ->E
H(c5) ; ->E
N(c5) ; ->E
A(c6) ; ->E
N(c6) ; ->E
G(c7) ; ->E
P(c7) ; ->E
K(c8) ; ->E
Q(c8) ; ->E
G(c9) ; ->E
Q(c9) ; ->E
B(c10) ; ->E
Q(c10) ; ->E
H(c11) ; ->E
P(c11) ; ->E
H(c12) ; ->E

Final answer: willow<|endoftext|>1. c0 is pearl.
2. c0 is east.
3. If c0 is south and c0 is pearl, then c1 is amber.
4. If c1 is amber, then c1 is south.
5. If c0 is west and c0 is pearl, then c1 is juniper.
6. If c1 is juniper, then c1 is north.
7. If c0 is north and c0 is pearl, then c1 is violet.
8. If c1 is violet, then c1 is west.
9. If c0 is east and c0 is pearl, then c1 is elm.
10. If c1 is elm, then c1 is east.
Which state applies to c1?

Solution:
Definitions:
c0 = c0
c1 = c1
Ax: x is amber
Bx: x is juniper
Cx: x is violet
Dx: x is pearl
Ex: x is elm
Fx: x is north
Gx: x is south
Hx: x is east
Ix: x is west

Formal premises:
D(c0)
H(c0)
G(c0) & D(c0) -> A(c1)
A(c1) -> G(c1)
I(c0) & D(c0) -> B(c1)
B(c1) -> F(c1)
F(c0) & D(c0) -> C(c1)
C(c1) -> I(c1)
H(c0) & D(c0) -> E(c1)
E(c1) -> H(c1)

Derivation:
D(c0) ; R
H(c0) ; R
E(c1) ; ->E

Final answer: elm<|endoftext|>
```

## Window 246 (2 documents, 241 pad tokens)
```
1. c0 is lime.
2. c0 is north.
3. If c0 is south and c0 is lime, then c1 is pearl.
4. If c1 is pearl, then c1 is south.
5. If c0 is north and c0 is lime, then c1 is willow.
6. If c1 is willow, then c1 is north.
7. If c0 is east and c0 is lime, then c1 is olive.
8. If c1 is olive, then c1 is east.
9. If c0 is west and c0 is lime, then c1 is granite.
10. If c1 is granite, then c1 is west.
11. If c1 is west and c1 is granite, then c2 is granite.
12. If c2 is granite, then c2 is north.
13. If c1 is south and c1 is pearl, then c2 is willow.
14. If c2 is willow, then c2 is south.
15. If c1 is east and c1 is olive, then c2 is cobalt.
16. If c2 is cobalt, then c2 is east.
17. If c1 is north and c1 is willow, then c2 is olive.
18. If c2 is olive, then c2 is west.
19. If c2 is north and c2 is granite, then c3 is pearl.
20. If c3 is pearl, then c3 is west.
21. If c2 is east and c2 is cobalt, then c3 is juniper.
22. If c3 is juniper, then c3 is north.
23. If c2 is south and c2 is willow, then c3 is cobalt.
24. If c3 is cobalt, then c3 is east.
25. If c2 is west and c2 is olive, then c3 is cedar.
26. If c3 is cedar, then c3 is south.
27. If c3 is west and c3 is pearl, then c4 is olive.
28. If c4 is olive, then c4 is east.
29. If c3 is north and c3 is juniper, then c4 is slate.
30. If c4 is slate, then c4 is south.
31. If c3 is east and c3 is cobalt, then c4 is willow.
32. If c4 is willow, then c4 is north.
33. If c3 is south and c3 is cedar, then c4 is granite.
34. If c4 is granite, then c4 is west.
35. If c4 is north and c4 is willow, then c5 is olive.
36. If c5 is olive, then c5 is west.
37. If c4 is east and c4 is olive, then c5 is willow.
38. If c5 is willow, then c5 is north.
39. If c4 is west and c4 is granite, then c5 is slate.
40. If c5 is slate, then c5 is south.
41. If c4 is south and c4 is slate, then c5 is cedar.
42. If c5 is cedar, then c5 is east.
43. If c5 is west and c5 is olive, then c6 is slate.
44. If c6 is slate, then c6 is east.
45. If c5 is east and c5 is cedar, then c6 is cedar.
46. If c6 is cedar, then c6 is north.
47. If c5 is south and c5 is slate, then c6 is olive.
48. If c6 is olive, then c6 is south.
49. If c5 is north and c5 is willow, then c6 is granite.
50. If c6 is granite, then c6 is west.
51. If c6 is west and c6 is granite, then c7 is juniper.
52. If c7 is juniper, then c7 is west.
53. If c6 is south and c6 is olive, then c7 is olive.
54. If c7 is olive, then c7 is south.
55. If c6 is north and c6 is cedar, then c7 is granite.
56. If c7 is granite, then c7 is north.
57. If c6 is east and c6 is slate, then c7 is cobalt.
58. If c7 is cobalt, then c7 is east.
59. If c7 is east and c7 is cobalt, then c8 is willow.
60. If c8 is willow, then c8 is west.
61. If c7 is north and c7 is granite, then c8 is granite.
62. If c8 is granite, then c8 is south.
63. If c7 is west and c7 is juniper, then c8 is pearl.
64. If c8 is pearl, then c8 is east.
65. If c7 is south and c7 is olive, then c8 is cedar.
66. If c8 is cedar, then c8 is north.
Which state applies to c8?

Solution:
Definitions:
c0 = c0
c1 = c1
c2 = c2
c3 = c3
c4 = c4
c5 = c5
c6 = c6
c7 = c7
c8 = c8
Ax: x is willow
Bx: x is juniper
Cx: x is cedar
Dx: x is granite
Ex: x is lime
Fx: x is slate
Gx: x is cobalt
Hx: x is olive
Ix: x is pearl
Jx: x is north
Kx: x is south
Lx: x is east
Mx: x is west

Formal premises:
E(c0)
J(c0)
K(c0) & E(c0) -> I(c1)
I(c1) -> K(c1)
J(c0) & E(c0) -> A(c1)
A(c1) -> J(c1)
L(c0) & E(c0) -> H(c1)
H(c1) -> L(c1)
M(c0) & E(c0) -> D(c1)
D(c1) -> M(c1)
M(c1) & D(c1) -> D(c2)
D(c2) -> J(c2)
K(c1) & I(c1) -> A(c2)
A(c2) -> K(c2)
L(c1) & H(c1) -> G(c2)
G(c2) -> L(c2)
J(c1) & A(c1) -> H(c2)
H(c2) -> M(c2)
J(c2) & D(c2) -> I(c3)
I(c3) -> M(c3)
L(c2) & G(c2) -> B(c3)
B(c3) -> J(c3)
K(c2) & A(c2) -> G(c3)
G(c3) -> L(c3)
M(c2) & H(c2) -> C(c3)
C(c3) -> K(c3)
M(c3) & I(c3) -> H(c4)
H(c4) -> L(c4)
J(c3) & B(c3) -> F(c4)
F(c4) -> K(c4)
L(c3) & G(c3) -> A(c4)
A(c4) -> J(c4)
K(c3) & C(c3) -> D(c4)
D(c4) -> M(c4)
J(c4) & A(c4) -> H(c5)
H(c5) -> M(c5)
L(c4) & H(c4) -> A(c5)
A(c5) -> J(c5)
M(c4) & D(c4) -> F(c5)
F(c5) -> K(c5)
K(c4) & F(c4) -> C(c5)
C(c5) -> L(c5)
M(c5) & H(c5) -> F(c6)
F(c6) -> L(c6)
L(c5) & C(c5) -> C(c6)
C(c6) -> J(c6)
K(c5) & F(c5) -> H(c6)
H(c6) -> K(c6)
J(c5) & A(c5) -> D(c6)
D(c6) -> M(c6)
M(c6) & D(c6) -> B(c7)
B(c7) -> M(c7)
K(c6) & H(c6) -> H(c7)
H(c7) -> K(c7)
J(c6) & C(c6) -> D(c7)
D(c7) -> J(c7)
L(c6) & F(c6) -> G(c7)
G(c7) -> L(c7)
L(c7) & G(c7) -> A(c8)
A(c8) -> M(c8)
J(c7) & D(c7) -> D(c8)
D(c8) -> K(c8)
M(c7) & B(c7) -> I(c8)
I(c8) -> L(c8)
K(c7) & H(c7) -> C(c8)
C(c8) -> J(c8)

Derivation:
E(c0) ; R
J(c0) ; R
A(c1) ; ->E
J(c1) ; ->E
H(c2) ; ->E
M(c2) ; ->E
C(c3) ; ->E
K(c3) ; ->E
D(c4) ; ->E
M(c4) ; ->E
F(c5) ; ->E
K(c5) ; ->E
H(c6) ; ->E
K(c6) ; ->E
H(c7) ; ->E
K(c7) ; ->E
C(c8) ; ->E

Final answer: cedar<|endoftext|>1. c0 is coral.
2. c0 is west.
3. If c0 is east and c0 is coral, then c1 is pearl.
4. If c1 is pearl, then c1 is east.
5. If c0 is north and c0 is coral, then c1 is olive.
6. If c1 is olive, then c1 is north.
7. If c0 is south and c0 is coral, then c1 is cedar.
8. If c1 is cedar, then c1 is west.
9. If c0 is west and c0 is coral, then c1 is maple.
10. If c1 is maple, then c1 is south.
11. If c1 is south and c1 is maple, then c2 is meadow.
12. If c2 is meadow, then c2 is west.
13. If c1 is north and c1 is olive, then c2 is olive.
14. If c2 is olive, then c2 is south.
15. If c1 is west and c1 is cedar, then c2 is maple.
16. If c2 is maple, then c2 is east.
17. If c1 is east and c1 is pearl, then c2 is ruby.
18. If c2 is ruby, then c2 is north.
19. If c2 is east and c2 is maple, then c3 is cedar.
20. If c3 is cedar, then c3 is south.
21. If c2 is north and c2 is ruby, then c3 is pearl.
22. If c3 is pearl, then c3 is west.
23. If c2 is south and c2 is olive, then c3 is olive.
24. If c3 is olive, then c3 is east.
25. If c2 is west and c2 is meadow, then c3 is ruby.
26. If c3 is ruby, then c3 is north.
27. If c3 is north and c3 is ruby, then c4 is cedar.
28. If c4 is cedar, then c4 is south.
29. If c3 is west and c3 is pearl, then c4 is meadow.
30. If c4 is meadow, then c4 is west.
31. If c3 is south and c3 is cedar, then c4 is pearl.
32. If c4 is pearl, then c4 is east.
33. If c3 is east and c3 is olive, then c4 is olive.
34. If c4 is olive, then c4 is north.
35. If c4 is south and c4 is cedar, then c5 is ruby.
36. If c5 is ruby, then c5 is west.
37. If c4 is north and c4 is olive, then c5 is pearl.
38. If c5 is pearl, then c5 is north.
39. If c4 is west and c4 is meadow, then c5 is maple.
40. If c5 is maple, then c5 is east.
41. If c4 is east and c4 is pearl, then c5 is cedar.
42. If c5 is cedar, then c5 is south.
43. If c5 is south and c5 is cedar, then c6 is cedar.
44. If c6 is cedar, then c6 is south.
45. If c5 is east and c5 is maple, then c6 is ruby.
46. If c6 is ruby, then c6 is west.
47. If c5 is north and c5 is pearl, then c6 is olive.
48. If c6 is olive, then c6 is east.
49. If c5 is west and c5 is ruby, then c6 is meadow.
50. If c6 is meadow, then c6 is north.
Which state applies to c6?

Solution:
Definitions:
c0 = c0
c1 = c1
c2 = c2
c3 = c3
c4 = c4
c5 = c5
c6 = c6
Ax: x is maple
Bx: x is olive
Cx: x is meadow
Dx: x is coral
Ex: x is ruby
Fx: x is cedar
Gx: x is pearl
Hx: x is north
Ix: x is south
Jx: x is east
Kx: x is west

Formal premises:
D(c0)
K(c0)
J(c0) & D(c0) -> G(c1)
G(c1) -> J(c1)
H(c0) & D(c0) -> B(c1)
B(c1) -> H(c1)
I(c0) & D(c0) -> F(c1)
F(c1) -> K(c1)
K(c0) & D(c0) -> A(c1)
A(c1) -> I(c1)
I(c1) & A(c1) -> C(c2)
C(c2) -> K(c2)
H(c1) & B(c1) -> B(c2)
B(c2) -> I(c2)
K(c1) & F(c1) -> A(c2)
A(c2) -> J(c2)
J(c1) & G(c1) -> E(c2)
E(c2) -> H(c2)
J(c2) & A(c2) -> F(c3)
F(c3) -> I(c3)
H(c2) & E(c2) -> G(c3)
G(c3) -> K(c3)
I(c2) & B(c2) -> B(c3)
B(c3) -> J(c3)
K(c2) & C(c2) -> E(c3)
E(c3) -> H(c3)
H(c3) & E(c3) -> F(c4)
F(c4) -> I(c4)
K(c3) & G(c3) -> C(c4)
C(c4) -> K(c4)
I(c3) & F(c3) -> G(c4)
G(c4) -> J(c4)
J(c3) & B(c3) -> B(c4)
B(c4) -> H(c4)
I(c4) & F(c4) -> E(c5)
E(c5) -> K(c5)
H(c4) & B(c4) -> G(c5)
G(c5) -> H(c5)
K(c4) & C(c4) -> A(c5)
A(c5) -> J(c5)
J(c4) & G(c4) -> F(c5)
F(c5) -> I(c5)
I(c5) & F(c5) -> F(c6)
F(c6) -> I(c6)
J(c5) & A(c5) -> E(c6)
E(c6) -> K(c6)
H(c5) & G(c5) -> B(c6)
B(c6) -> J(c6)
K(c5) & E(c5) -> C(c6)
C(c6) -> H(c6)

Derivation:
D(c0) ; R
K(c0) ; R
A(c1) ; ->E
I(c1) ; ->E
C(c2) ; ->E
K(c2) ; ->E
E(c3) ; ->E
H(c3) ; ->E
F(c4) ; ->E
I(c4) ; ->E
E(c5) ; ->E
K(c5) ; ->E
C(c6) ; ->E

Final answer: meadow<|endoftext|>
```

## Window 664 (2 documents, 225 pad tokens)
```
1. c0 is coral.
2. c0 is north.
3. If c0 is south and c0 is coral, then c1 is cobalt.
4. If c1 is cobalt, then c1 is south.
5. If c0 is east and c0 is coral, then c1 is elm.
6. If c1 is elm, then c1 is north.
7. If c0 is north and c0 is coral, then c1 is amber.
8. If c1 is amber, then c1 is west.
9. If c0 is west and c0 is coral, then c1 is ivory.
10. If c1 is ivory, then c1 is east.
11. If c1 is north and c1 is elm, then c2 is elm.
12. If c2 is elm, then c2 is west.
13. If c1 is west and c1 is amber, then c2 is pearl.
14. If c2 is pearl, then c2 is east.
15. If c1 is south and c1 is cobalt, then c2 is willow.
16. If c2 is willow, then c2 is north.
17. If c1 is east and c1 is ivory, then c2 is amber.
18. If c2 is amber, then c2 is south.
19. If c2 is south and c2 is amber, then c3 is harbor.
20. If c3 is harbor, then c3 is south.
21. If c2 is east and c2 is pearl, then c3 is pearl.
22. If c3 is pearl, then c3 is west.
23. If c2 is west and c2 is elm, then c3 is willow.
24. If c3 is willow, then c3 is east.
25. If c2 is north and c2 is willow, then c3 is ivory.
26. If c3 is ivory, then c3 is north.
27. If c3 is west and c3 is pearl, then c4 is cedar.
28. If c4 is cedar, then c4 is north.
29. If c3 is north and c3 is ivory, then c4 is pearl.
30. If c4 is pearl, then c4 is east.
31. If c3 is east and c3 is willow, then c4 is ivory.
32. If c4 is ivory, then c4 is west.
33. If c3 is south and c3 is harbor, then c4 is cobalt.
34. If c4 is cobalt, then c4 is south.
35. If c4 is north and c4 is cedar, then c5 is violet.
36. If c5 is violet, then c5 is north.
37. If c4 is west and c4 is ivory, then c5 is amber.
38. If c5 is amber, then c5 is south.
39. If c4 is south and c4 is cobalt, then c5 is olive.
40. If c5 is olive, then c5 is west.
41. If c4 is east and c4 is pearl, then c5 is ivory.
42. If c5 is ivory, then c5 is east.
43. If c5 is west and c5 is olive, then c6 is pearl.
44. If c6 is pearl, then c6 is east.
45. If c5 is east and c5 is ivory, then c6 is ivory.
46. If c6 is ivory, then c6 is west.
47. If c5 is south and c5 is amber, then c6 is harbor.
48. If c6 is harbor, then c6 is south.
49. If c5 is north and c5 is violet, then c6 is amber.
50. If c6 is amber, then c6 is north.
51. If c6 is north and c6 is amber, then c7 is elm.
52. If c7 is elm, then c7 is north.
53. If c6 is south and c6 is harbor, then c7 is pearl.
54. If c7 is pearl, then c7 is east.
55. If c6 is east and c6 is pearl, then c7 is amber.
56. If c7 is amber, then c7 is south.
57. If c6 is west and c6 is ivory, then c7 is cedar.
58. If c7 is cedar, then c7 is west.
59. If c7 is south and c7 is amber, then c8 is harbor.
60. If c8 is harbor, then c8 is north.
61. If c7 is east and c7 is pearl, then c8 is elm.
62. If c8 is elm, then c8 is east.
63. If c7 is west and c7 is cedar, then c8 is violet.
64. If c8 is violet, then c8 is south.
65. If c7 is north and c7 is elm, then c8 is olive.
66. If c8 is olive, then c8 is west.
67. If c8 is south and c8 is violet, then c9 is elm.
68. If c9 is elm, then c9 is west.
69. If c8 is north and c8 is harbor, then c9 is harbor.
70. If c9 is harbor, then c9 is south.
71. If c8 is west and c8 is olive, then c9 is olive.
72. If c9 is olive, then c9 is east.
73. If c8 is east and c8 is elm, then c9 is cobalt.
74. If c9 is cobalt, then c9 is north.
75. If c9 is north and c9 is cobalt, then c10 is cobalt.
76. If c10 is cobalt, then c10 is west.
77. If c9 is south and c9 is harbor, then c10 is cedar.
78. If c10 is cedar, then c10 is north.
79. If c9 is east and c9 is olive, then c10 is olive.
80. If c10 is olive, then c10 is south.
81. If c9 is west and c9 is elm, then c10 is amber.
82. If c10 is amber, then c10 is east.
Which state applies to c10?

Solution:
Definitions:
c0 = c0
c1 = c1
c2 = c2
c3 = c3
c4 = c4
c5 = c5
c6 = c6
c7 = c7
c8 = c8
c9 = c9
c10 = c10
Ax: x is violet
Bx: x is ivory
Cx: x is cobalt
Dx: x is harbor
Ex: x is amber
Fx: x is elm
Gx: x is cedar
Hx: x is pearl
Ix: x is willow
Jx: x is olive
Kx: x is coral
Lx: x is north
Mx: x is south
Nx: x is east
Ox: x is west

Formal premises:
K(c0)
L(c0)
M(c0) & K(c0) -> C(c1)
C(c1) -> M(c1)
N(c0) & K(c0) -> F(c1)
F(c1) -> L(c1)
L(c0) & K(c0) -> E(c1)
E(c1) -> O(c1)
O(c0) & K(c0) -> B(c1)
B(c1) -> N(c1)
L(c1) & F(c1) -> F(c2)
F(c2) -> O(c2)
O(c1) & E(c1) -> H(c2)
H(c2) -> N(c2)
M(c1) & C(c1) -> I(c2)
I(c2) -> L(c2)
N(c1) & B(c1) -> E(c2)
E(c2) -> M(c2)
M(c2) & E(c2) -> D(c3)
D(c3) -> M(c3)
N(c2) & H(c2) -> H(c3)
H(c3) -> O(c3)
O(c2) & F(c2) -> I(c3)
I(c3) -> N(c3)
L(c2) & I(c2) -> B(c3)
B(c3) -> L(c3)
O(c3) & H(c3) -> G(c4)
G(c4) -> L(c4)
L(c3) & B(c3) -> H(c4)
H(c4) -> N(c4)
N(c3) & I(c3) -> B(c4)
B(c4) -> O(c4)
M(c3) & D(c3) -> C(c4)
C(c4) -> M(c4)
L(c4) & G(c4) -> A(c5)
A(c5) -> L(c5)
O(c4) & B(c4) -> E(c5)
E(c5) -> M(c5)
M(c4) & C(c4) -> J(c5)
J(c5) -> O(c5)
N(c4) & H(c4) -> B(c5)
B(c5) -> N(c5)
O(c5) & J(c5) -> H(c6)
H(c6) -> N(c6)
N(c5) & B(c5) -> B(c6)
B(c6) -> O(c6)
M(c5) & E(c5) -> D(c6)
D(c6) -> M(c6)
L(c5) & A(c5) -> E(c6)
E(c6) -> L(c6)
L(c6) & E(c6) -> F(c7)
F(c7) -> L(c7)
M(c6) & D(c6) -> H(c7)
H(c7) -> N(c7)
N(c6) & H(c6) -> E(c7)
E(c7) -> M(c7)
O(c6) & B(c6) -> G(c7)
G(c7) -> O(c7)
M(c7) & E(c7) -> D(c8)
D(c8) -> L(c8)
N(c7) & H(c7) -> F(c8)
F(c8) -> N(c8)
O(c7) & G(c7) -> A(c8)
A(c8) -> M(c8)
L(c7) & F(c7) -> J(c8)
J(c8) -> O(c8)
M(c8) & A(c8) -> F(c9)
F(c9) -> O(c9)
L(c8) & D(c8) -> D(c9)
D(c9) -> M(c9)
O(c8) & J(c8) -> J(c9)
J(c9) -> N(c9)
N(c8) & F(c8) -> C(c9)
C(c9) -> L(c9)
L(c9) & C(c9) -> C(c10)
C(c10) -> O(c10)
M(c9) & D(c9) -> G(c10)
G(c10) -> L(c10)
N(c9) & J(c9) -> J(c10)
J(c10) -> M(c10)
O(c9) & F(c9) -> E(c10)
E(c10) -> N(c10)

Derivation:
K(c0) ; R
L(c0) ; R
E(c1) ; ->E
O(c1) ; ->E
H(c2) ; ->E
N(c2) ; ->E
H(c3) ; ->E
O(c3) ; ->E
G(c4) ; ->E
L(c4) ; ->E
A(c5) ; ->E
L(c5) ; ->E
E(c6) ; ->E
L(c6) ; ->E
F(c7) ; ->E
L(c7) ; ->E
J(c8) ; ->E
O(c8) ; ->E
J(c9) ; ->E
N(c9) ; ->E
J(c10) ; ->E

Final answer: olive<|endoftext|>1. c0 is ivory.
2. c0 is north.
3. If c0 is north and c0 is ivory, then c1 is elm.
4. If c1 is elm, then c1 is south.
5. If c0 is south and c0 is ivory, then c1 is lime.
6. If c1 is lime, then c1 is east.
7. If c0 is west and c0 is ivory, then c1 is violet.
8. If c1 is violet, then c1 is north.
9. If c0 is east and c0 is ivory, then c1 is orchid.
10. If c1 is orchid, then c1 is west.
11. If c1 is north and c1 is violet, then c2 is elm.
12. If c2 is elm, then c2 is south.
13. If c1 is east and c1 is lime, then c2 is violet.
14. If c2 is violet, then c2 is north.
15. If c1 is west and c1 is orchid, then c2 is lime.
16. If c2 is lime, then c2 is west.
17. If c1 is south and c1 is elm, then c2 is orchid.
18. If c2 is orchid, then c2 is east.
19. If c2 is west and c2 is lime, then c3 is violet.
20. If c3 is violet, then c3 is east.
21. If c2 is east and c2 is orchid, then c3 is orchid.
22. If c3 is orchid, then c3 is south.
23. If c2 is north and c2 is violet, then c3 is elm.
24. If c3 is elm, then c3 is north.
25. If c2 is south and c2 is elm, then c3 is lime.
26. If c3 is lime, then c3 is west.
27. If c3 is west and c3 is lime, then c4 is violet.
28. If c4 is violet, then c4 is south.
29. If c3 is east and c3 is violet, then c4 is elm.
30. If c4 is elm, then c4 is east.
31. If c3 is south and c3 is orchid, then c4 is lime.
32. If c4 is lime, then c4 is north.
33. If c3 is north and c3 is elm, then c4 is orchid.
34. If c4 is orchid, then c4 is west.
Which state applies to c4?

Solution:
Definitions:
c0 = c0
c1 = c1
c2 = c2
c3 = c3
c4 = c4
Ax: x is lime
Bx: x is orchid
Cx: x is elm
Dx: x is violet
Ex: x is ivory
Fx: x is north
Gx: x is south
Hx: x is east
Ix: x is west

Formal premises:
E(c0)
F(c0)
F(c0) & E(c0) -> C(c1)
C(c1) -> G(c1)
G(c0) & E(c0) -> A(c1)
A(c1) -> H(c1)
I(c0) & E(c0) -> D(c1)
D(c1) -> F(c1)
H(c0) & E(c0) -> B(c1)
B(c1) -> I(c1)
F(c1) & D(c1) -> C(c2)
C(c2) -> G(c2)
H(c1) & A(c1) -> D(c2)
D(c2) -> F(c2)
I(c1) & B(c1) -> A(c2)
A(c2) -> I(c2)
G(c1) & C(c1) -> B(c2)
B(c2) -> H(c2)
I(c2) & A(c2) -> D(c3)
D(c3) -> H(c3)
H(c2) & B(c2) -> B(c3)
B(c3) -> G(c3)
F(c2) & D(c2) -> C(c3)
C(c3) -> F(c3)
G(c2) & C(c2) -> A(c3)
A(c3) -> I(c3)
I(c3) & A(c3) -> D(c4)
D(c4) -> G(c4)
H(c3) & D(c3) -> C(c4)
C(c4) -> H(c4)
G(c3) & B(c3) -> A(c4)
A(c4) -> F(c4)
F(c3) & C(c3) -> B(c4)
B(c4) -> I(c4)

Derivation:
E(c0) ; R
F(c0) ; R
C(c1) ; ->E
G(c1) ; ->E
B(c2) ; ->E
H(c2) ; ->E
B(c3) ; ->E
G(c3) ; ->E
A(c4) ; ->E

Final answer: lime<|endoftext|>
```

## Window 1182 summary: [{"tokens": 1938, "head": "1. c0 is pearl. 2. c0 is west. 3. If c0 is east and c0 is pearl, then c1 is ceda", "tail": "->E E(c6) ; ->E J(c6) ; ->E G(c7) ; ->E  Final answer: coral"}, {"tokens": 1940, "head": "1. c0 is birch. 2. c0 is east. 3. If c0 is west and c0 is birch, then c1 is slat", "tail": "; ->E C(c6) ; ->E I(c6) ; ->E B(c7) ; ->E  Final answer: elm"}]

## Window 1318 summary: [{"tokens": 4041, "head": "1. c0 is cobalt. 2. c0 is west. 3. If c0 is south and c0 is cobalt, then c1 is m", "tail": " H(c13) ; ->E P(c13) ; ->E J(c14) ; ->E  Final answer: slate"}]

## Window 2431 summary: [{"tokens": 3394, "head": "1. c0 is granite. 2. c0 is south. 3. If c0 is west and c0 is granite, then c1 is", "tail": " B(c11) ; ->E O(c11) ; ->E I(c12) ; ->E  Final answer: pearl"}, {"tokens": 368, "head": "1. c0 is amber. 2. c0 is east. 3. If c0 is south and c0 is amber, then c1 is jun", "tail": "ivation: A(c0) ; R H(c0) ; R D(c1) ; ->E  Final answer: lime"}]

## Window 2558 summary: [{"tokens": 4021, "head": "1. c0 is poppy. 2. c0 is north. 3. If c0 is west and c0 is poppy, then c1 is gra", "tail": "E F(c13) ; ->E P(c13) ; ->E A(c14) ; ->E  Final answer: ruby"}]

## Window 3503 summary: [{"tokens": 2441, "head": "1. c0 is orchid. 2. c0 is east. 3. If c0 is south and c0 is orchid, then c1 is h", "tail": "->E B(c8) ; ->E L(c8) ; ->E H(c9) ; ->E  Final answer: coral"}, {"tokens": 1124, "head": "1. c0 is ruby. 2. c0 is south. 3. If c0 is west and c0 is ruby, then c1 is pearl", "tail": "->E B(c3) ; ->E F(c3) ; ->E B(c4) ; ->E  Final answer: pearl"}, {"tokens": 372, "head": "1. c0 is teal. 2. c0 is north. 3. If c0 is east and c0 is teal, then c1 is birch", "tail": "ation: C(c0) ; R F(c0) ; R A(c1) ; ->E  Final answer: cobalt"}]

## Window 3824 summary: [{"tokens": 3717, "head": "1. c0 is coral. 2. c0 is south. 3. If c0 is west and c0 is coral, then c1 is tea", "tail": " K(c12) ; ->E R(c12) ; ->E H(c13) ; ->E  Final answer: hazel"}]

## Window 5145 summary: [{"tokens": 3091, "head": "1. c0 is ruby. 2. c0 is west. 3. If c0 is north and c0 is ruby, then c1 is elm. ", "tail": " A(c10) ; ->E N(c10) ; ->E J(c11) ; ->E  Final answer: poppy"}, {"tokens": 622, "head": "1. c0 is willow. 2. c0 is south. 3. If c0 is north and c0 is willow, then c1 is ", "tail": " ; R D(c1) ; ->E I(c1) ; ->E C(c2) ; ->E  Final answer: lime"}, {"tokens": 369, "head": "1. c0 is granite. 2. c0 is north. 3. If c0 is north and c0 is granite, then c1 i", "tail": "ation: D(c0) ; R F(c0) ; R A(c1) ; ->E  Final answer: cobalt"}]

## Window 5146 summary: [{"tokens": 3691, "head": "1. c0 is teal. 2. c0 is south. 3. If c0 is north and c0 is teal, then c1 is pear", "tail": " C(c12) ; ->E O(c12) ; ->E A(c13) ; ->E  Final answer: pearl"}]

## Window 5682 summary: [{"tokens": 4003, "head": "1. c0 is cedar. 2. c0 is west. 3. If c0 is east and c0 is cedar, then c1 is ivor", "tail": "E D(c13) ; ->E P(c13) ; ->E E(c14) ; ->E  Final answer: teal"}]

## Window 5883 summary: [{"tokens": 2213, "head": "1. c0 is violet. 2. c0 is south. 3. If c0 is north and c0 is violet, then c1 is ", "tail": "->E B(c7) ; ->E M(c7) ; ->E I(c8) ; ->E  Final answer: olive"}, {"tokens": 1646, "head": "1. c0 is coral. 2. c0 is south. 3. If c0 is south and c0 is coral, then c1 is gr", "tail": "->E A(c5) ; ->E J(c5) ; ->E C(c6) ; ->E  Final answer: pearl"}]

## Window 6137 summary: [{"tokens": 4004, "head": "1. c0 is meadow. 2. c0 is south. 3. If c0 is north and c0 is meadow, then c1 is ", "tail": " C(c13) ; ->E S(c13) ; ->E N(c14) ; ->E  Final answer: olive"}]

## Window 6833 summary: [{"tokens": 2210, "head": "1. c0 is maple. 2. c0 is west. 3. If c0 is east and c0 is maple, then c1 is cora", "tail": ">E B(c7) ; ->E K(c7) ; ->E H(c8) ; ->E  Final answer: meadow"}, {"tokens": 1671, "head": "1. c0 is cedar. 2. c0 is south. 3. If c0 is north and c0 is cedar, then c1 is ha", "tail": "->E E(c5) ; ->E K(c5) ; ->E F(c6) ; ->E  Final answer: amber"}]

## Window 7221 summary: [{"tokens": 2789, "head": "1. c0 is coral. 2. c0 is west. 3. If c0 is west and c0 is coral, then c1 is haze", "tail": " A(c9) ; ->E O(c9) ; ->E C(c10) ; ->E  Final answer: granite"}, {"tokens": 1148, "head": "1. c0 is coral. 2. c0 is west. 3. If c0 is north and c0 is coral, then c1 is orc", "tail": " ->E C(c3) ; ->E H(c3) ; ->E C(c4) ; ->E  Final answer: teal"}]

## Window 7322 summary: [{"tokens": 3699, "head": "1. c0 is willow. 2. c0 is east. 3. If c0 is south and c0 is willow, then c1 is l", "tail": " K(c12) ; ->E Q(c12) ; ->E B(c13) ; ->E  Final answer: poppy"}]

## Window 7540 summary: [{"tokens": 2239, "head": "1. c0 is elm. 2. c0 is west. 3. If c0 is south and c0 is elm, then c1 is willow.", "tail": "->E C(c7) ; ->E M(c7) ; ->E B(c8) ; ->E  Final answer: birch"}, {"tokens": 1142, "head": "1. c0 is birch. 2. c0 is east. 3. If c0 is north and c0 is birch, then c1 is lim", "tail": "; ->E E(c3) ; ->E G(c3) ; ->E E(c4) ; ->E  Final answer: elm"}, {"tokens": 374, "head": "1. c0 is cobalt. 2. c0 is south. 3. If c0 is west and c0 is cobalt, then c1 is l", "tail": "ivation: A(c0) ; R G(c0) ; R E(c1) ; ->E  Final answer: teal"}]

## Window 7644 summary: [{"tokens": 3398, "head": "1. c0 is birch. 2. c0 is east. 3. If c0 is south and c0 is birch, then c1 is orc", "tail": "M(c11) ; ->E P(c11) ; ->E M(c12) ; ->E  Final answer: orchid"}, {"tokens": 622, "head": "1. c0 is cedar. 2. c0 is south. 3. If c0 is south and c0 is cedar, then c1 is li", "tail": " ; R D(c1) ; ->E F(c1) ; ->E D(c2) ; ->E  Final answer: lime"}]

## Window 7916 summary: [{"tokens": 1947, "head": "1. c0 is teal. 2. c0 is north. 3. If c0 is east and c0 is teal, then c1 is willo", "tail": "->E B(c6) ; ->E L(c6) ; ->E C(c7) ; ->E  Final answer: maple"}, {"tokens": 1938, "head": "1. c0 is laurel. 2. c0 is west. 3. If c0 is north and c0 is laurel, then c1 is v", "tail": "->E F(c6) ; ->E K(c6) ; ->E E(c7) ; ->E  Final answer: pearl"}]

## Window 7962 summary: [{"tokens": 3094, "head": "1. c0 is willow. 2. c0 is east. 3. If c0 is west and c0 is willow, then c1 is ha", "tail": "E J(c10) ; ->E M(c10) ; ->E I(c11) ; ->E  Final answer: lime"}, {"tokens": 879, "head": "1. c0 is granite. 2. c0 is south. 3. If c0 is west and c0 is granite, then c1 is", "tail": " ->E C(c2) ; ->E I(c2) ; ->E C(c3) ; ->E  Final answer: teal"}]

## Window 8060 summary: [{"tokens": 2768, "head": "1. c0 is olive. 2. c0 is south. 3. If c0 is east and c0 is olive, then c1 is lau", "tail": " J(c9) ; ->E L(c9) ; ->E H(c10) ; ->E  Final answer: granite"}, {"tokens": 1124, "head": "1. c0 is maple. 2. c0 is south. 3. If c0 is north and c0 is maple, then c1 is ce", "tail": "->E D(c3) ; ->E F(c3) ; ->E E(c4) ; ->E  Final answer: cedar"}]

## Window 8222 summary: [{"tokens": 2770, "head": "1. c0 is harbor. 2. c0 is south. 3. If c0 is west and c0 is harbor, then c1 is o", "tail": ">E I(c9) ; ->E L(c9) ; ->E D(c10) ; ->E  Final answer: amber"}, {"tokens": 1154, "head": "1. c0 is hazel. 2. c0 is west. 3. If c0 is west and c0 is hazel, then c1 is slat", "tail": " ->E D(c3) ; ->E I(c3) ; ->E A(c4) ; ->E  Final answer: lime"}]

## Window 8471 summary: [{"tokens": 3387, "head": "1. c0 is birch. 2. c0 is south. 3. If c0 is west and c0 is birch, then c1 is gra", "tail": "J(c11) ; ->E P(c11) ; ->E I(c12) ; ->E  Final answer: willow"}, {"tokens": 371, "head": "1. c0 is lime. 2. c0 is north. 3. If c0 is south and c0 is lime, then c1 is popp", "tail": "ation: D(c0) ; R F(c0) ; R E(c1) ; ->E  Final answer: violet"}]

## Window 8915 summary: [{"tokens": 2191, "head": "1. c0 is birch. 2. c0 is east. 3. If c0 is north and c0 is birch, then c1 is ivo", "tail": "; ->E E(c7) ; ->E M(c7) ; ->E D(c8) ; ->E  Final answer: elm"}, {"tokens": 870, "head": "1. c0 is ruby. 2. c0 is west. 3. If c0 is south and c0 is ruby, then c1 is elm. ", "tail": "; ->E E(c2) ; ->E H(c2) ; ->E E(c3) ; ->E  Final answer: elm"}, {"tokens": 622, "head": "1. c0 is birch. 2. c0 is east. 3. If c0 is west and c0 is birch, then c1 is oliv", "tail": "R D(c1) ; ->E G(c1) ; ->E D(c2) ; ->E  Final answer: granite"}, {"tokens": 369, "head": "1. c0 is lime. 2. c0 is north. 3. If c0 is south and c0 is lime, then c1 is laur", "tail": "ation: C(c0) ; R F(c0) ; R A(c1) ; ->E  Final answer: cobalt"}]

## Window 9161 summary: [{"tokens": 2209, "head": "1. c0 is amber. 2. c0 is east. 3. If c0 is south and c0 is amber, then c1 is lau", "tail": "->E F(c7) ; ->E K(c7) ; ->E D(c8) ; ->E  Final answer: pearl"}, {"tokens": 1693, "head": "1. c0 is birch. 2. c0 is south. 3. If c0 is east and c0 is birch, then c1 is map", "tail": "->E E(c5) ; ->E J(c5) ; ->E G(c6) ; ->E  Final answer: hazel"}]

## Window 9449 summary: [{"tokens": 2241, "head": "1. c0 is slate. 2. c0 is east. 3. If c0 is west and c0 is slate, then c1 is haze", "tail": ">E E(c7) ; ->E M(c7) ; ->E F(c8) ; ->E  Final answer: orchid"}, {"tokens": 1671, "head": "1. c0 is ruby. 2. c0 is south. 3. If c0 is south and c0 is ruby, then c1 is will", "tail": ">E C(c5) ; ->E I(c5) ; ->E F(c6) ; ->E  Final answer: harbor"}]

## Window 10288 summary: [{"tokens": 4033, "head": "1. c0 is cedar. 2. c0 is east. 3. If c0 is east and c0 is cedar, then c1 is mead", "tail": "(c13) ; ->E P(c13) ; ->E I(c14) ; ->E  Final answer: juniper"}]

## Window 10289 summary: [{"tokens": 3390, "head": "1. c0 is elm. 2. c0 is east. 3. If c0 is north and c0 is elm, then c1 is laurel.", "tail": "M(c11) ; ->E O(c11) ; ->E A(c12) ; ->E  Final answer: meadow"}, {"tokens": 372, "head": "1. c0 is cobalt. 2. c0 is east. 3. If c0 is west and c0 is cobalt, then c1 is ce", "tail": "vation: A(c0) ; R H(c0) ; R D(c1) ; ->E  Final answer: poppy"}]
