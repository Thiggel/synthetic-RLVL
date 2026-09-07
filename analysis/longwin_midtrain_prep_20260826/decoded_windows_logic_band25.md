# Decoded-batch audit examples (document-preserving docpack loader)

## Window 0 (2 documents, 469 pad tokens)
```
<question>
1. c0 is juniper.
2. c0 is east.
3. If c0 is west and c0 is juniper, then c1 is lime.
4. If c1 is lime, then c1 is west.
5. If c0 is north and c0 is juniper, then c1 is slate.
6. If c1 is slate, then c1 is north.
7. If c0 is east and c0 is juniper, then c1 is ivory.
8. If c1 is ivory, then c1 is east.
9. If c0 is south and c0 is juniper, then c1 is olive.
10. If c1 is olive, then c1 is south.
11. If c1 is west and c1 is lime, then c2 is ivory.
12. If c2 is ivory, then c2 is south.
13. If c1 is south and c1 is olive, then c2 is orchid.
14. If c2 is orchid, then c2 is east.
15. If c1 is north and c1 is slate, then c2 is cobalt.
16. If c2 is cobalt, then c2 is north.
17. If c1 is east and c1 is ivory, then c2 is pearl.
18. If c2 is pearl, then c2 is west.
19. If c2 is north and c2 is cobalt, then c3 is cobalt.
20. If c3 is cobalt, then c3 is east.
21. If c2 is south and c2 is ivory, then c3 is willow.
22. If c3 is willow, then c3 is west.
23. If c2 is east and c2 is orchid, then c3 is violet.
24. If c3 is violet, then c3 is north.
25. If c2 is west and c2 is pearl, then c3 is coral.
26. If c3 is coral, then c3 is south.
27. If c3 is west and c3 is willow, then c4 is cedar.
28. If c4 is cedar, then c4 is east.
29. If c3 is east and c3 is cobalt, then c4 is cobalt.
30. If c4 is cobalt, then c4 is north.
31. If c3 is south and c3 is coral, then c4 is ivory.
32. If c4 is ivory, then c4 is west.
33. If c3 is north and c3 is violet, then c4 is teal.
34. If c4 is teal, then c4 is south.
35. If c4 is north and c4 is cobalt, then c5 is olive.
36. If c5 is olive, then c5 is east.
37. If c4 is west and c4 is ivory, then c5 is cedar.
38. If c5 is cedar, then c5 is north.
39. If c4 is east and c4 is cedar, then c5 is granite.
40. If c5 is granite, then c5 is south.
41. If c4 is south and c4 is teal, then c5 is poppy.
42. If c5 is poppy, then c5 is west.
43. If c5 is north and c5 is cedar, then c6 is cobalt.
44. If c6 is cobalt, then c6 is south.
45. If c5 is west and c5 is poppy, then c6 is cedar.
46. If c6 is cedar, then c6 is west.
47. If c5 is east and c5 is olive, then c6 is harbor.
48. If c6 is harbor, then c6 is east.
49. If c5 is south and c5 is granite, then c6 is slate.
50. If c6 is slate, then c6 is north.
51. If c6 is west and c6 is cedar, then c7 is elm.
52. If c7 is elm, then c7 is north.
53. If c6 is south and c6 is cobalt, then c7 is ruby.
54. If c7 is ruby, then c7 is east.
55. If c6 is north and c6 is slate, then c7 is coral.
56. If c7 is coral, then c7 is west.
57. If c6 is east and c6 is harbor, then c7 is olive.
58. If c7 is olive, then c7 is south.
59. If c7 is south and c7 is olive, then c8 is olive.
60. If c8 is olive, then c8 is east.
61. If c7 is west and c7 is coral, then c8 is cedar.
62. If c8 is cedar, then c8 is south.
63. If c7 is north and c7 is elm, then c8 is violet.
64. If c8 is violet, then c8 is north.
65. If c7 is east and c7 is ruby, then c8 is birch.
66. If c8 is birch, then c8 is west.
67. If c8 is east and c8 is olive, then c9 is ruby.
68. If c9 is ruby, then c9 is west.
69. If c8 is south and c8 is cedar, then c9 is cedar.
70. If c9 is cedar, then c9 is south.
71. If c8 is west and c8 is birch, then c9 is ivory.
72. If c9 is ivory, then c9 is east.
73. If c8 is north and c8 is violet, then c9 is hazel.
74. If c9 is hazel, then c9 is north.
75. If c9 is south and c9 is cedar, then c10 is harbor.
76. If c10 is harbor, then c10 is south.
77. If c9 is east and c9 is ivory, then c10 is cedar.
78. If c10 is cedar, then c10 is north.
79. If c9 is west and c9 is ruby, then c10 is ruby.
80. If c10 is ruby, then c10 is east.
81. If c9 is north and c9 is hazel, then c10 is teal.
82. If c10 is teal, then c10 is west.
83. If c10 is north and c10 is cedar, then c11 is poppy.
84. If c11 is poppy, then c11 is north.
85. If c10 is east and c10 is ruby, then c11 is teal.
86. If c11 is teal, then c11 is east.
87. If c10 is south and c10 is harbor, then c11 is cedar.
88. If c11 is cedar, then c11 is west.
89. If c10 is west and c10 is teal, then c11 is ruby.
90. If c11 is ruby, then c11 is south.
91. If c11 is west and c11 is cedar, then c12 is coral.
92. If c12 is coral, then c12 is east.
93. If c11 is south and c11 is ruby, then c12 is amber.
94. If c12 is amber, then c12 is south.
95. If c11 is east and c11 is teal, then c12 is willow.
96. If c12 is willow, then c12 is north.
97. If c11 is north and c11 is poppy, then c12 is orchid.
98. If c12 is orchid, then c12 is west.
99. If c12 is north and c12 is willow, then c13 is laurel.
100. If c13 is laurel, then c13 is north.
101. If c12 is west and c12 is orchid, then c13 is harbor.
102. If c13 is harbor, then c13 is east.
103. If c12 is east and c12 is coral, then c13 is olive.
104. If c13 is olive, then c13 is west.
105. If c12 is south and c12 is amber, then c13 is coral.
106. If c13 is coral, then c13 is south.
107. If c13 is west and c13 is olive, then c14 is ivory.
108. If c14 is ivory, then c14 is south.
109. If c13 is north and c13 is laurel, then c14 is coral.
110. If c14 is coral, then c14 is east.
111. If c13 is east and c13 is harbor, then c14 is slate.
112. If c14 is slate, then c14 is north.
113. If c13 is south and c13 is coral, then c14 is laurel.
114. If c14 is laurel, then c14 is west.
115. If c14 is east and c14 is coral, then c15 is cedar.
116. If c15 is cedar, then c15 is east.
117. If c14 is south and c14 is ivory, then c15 is elm.
118. If c15 is elm, then c15 is north.
119. If c14 is north and c14 is slate, then c15 is birch.
120. If c15 is birch, then c15 is south.
121. If c14 is west and c14 is laurel, then c15 is hazel.
122. If c15 is hazel, then c15 is west.
123. If c15 is north and c15 is elm, then c16 is teal.
124. If c16 is teal, then c16 is north.
125. If c15 is east and c15 is cedar, then c16 is olive.
126. If c16 is olive, then c16 is east.
127. If c15 is south and c15 is birch, then c16 is pearl.
128. If c16 is pearl, then c16 is south.
129. If c15 is west and c15 is hazel, then c16 is lime.
130. If c16 is lime, then c16 is west.
131. If c16 is west and c16 is lime, then c17 is laurel.
132. If c17 is laurel, then c17 is north.
133. If c16 is north and c16 is teal, then c17 is orchid.
134. If c17 is orchid, then c17 is south.
135. If c16 is east and c16 is olive, then c17 is hazel.
136. If c17 is hazel, then c17 is west.
137. If c16 is south and c16 is pearl, then c17 is ruby.
138. If c17 is ruby, then c17 is east.
139. If c17 is west and c17 is hazel, then c18 is harbor.
140. If c18 is harbor, then c18 is north.
141. If c17 is north and c17 is laurel, then c18 is granite.
142. If c18 is granite, then c18 is west.
143. If c17 is east and c17 is ruby, then c18 is pearl.
144. If c18 is pearl, then c18 is south.
145. If c17 is south and c17 is orchid, then c18 is ivory.
146. If c18 is ivory, then c18 is east.
147. If c18 is east and c18 is ivory, then c19 is pearl.
148. If c19 is pearl, then c19 is west.
149. If c18 is west and c18 is granite, then c19 is violet.
150. If c19 is violet, then c19 is south.
151. If c18 is north and c18 is harbor, then c19 is granite.
152. If c19 is granite, then c19 is north.
153. If c18 is south and c18 is pearl, then c19 is laurel.
154. If c19 is laurel, then c19 is east.
155. If c19 is west and c19 is pearl, then c20 is birch.
156. If c20 is birch, then c20 is north.
157. If c19 is south and c19 is violet, then c20 is teal.
158. If c20 is teal, then c20 is west.
159. If c19 is east and c19 is laurel, then c20 is poppy.
160. If c20 is poppy, then c20 is east.
161. If c19 is north and c19 is granite, then c20 is ruby.
162. If c20 is ruby, then c20 is south.
163. If c20 is south and c20 is ruby, then c21 is birch.
164. If c21 is birch, then c21 is west.
165. If c20 is east and c20 is poppy, then c21 is coral.
166. If c21 is coral, then c21 is east.
167. If c20 is north and c20 is birch, then c21 is harbor.
168. If c21 is harbor, then c21 is north.
169. If c20 is west and c20 is teal, then c21 is ruby.
170. If c21 is ruby, then c21 is south.
Which state applies to c21?
</question>

<formal>
<constants>
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
c15 = c15
c16 = c16
c17 = c17
c18 = c18
c19 = c19
c20 = c20
c21 = c21
</constants>
<predicates>
Ax: x is laurel
Bx: x is hazel
Cx: x is coral
Dx: x is amber
Ex: x is ruby
Fx: x is lime
Gx: x is elm
Hx: x is poppy
Ix: x is teal
Jx: x is olive
Kx: x is harbor
Lx: x is birch
Mx: x is willow
Nx: x is orchid
Ox: x is granite
Px: x is juniper
Qx: x is cobalt
Rx: x is violet
Sx: x is cedar
Tx: x is ivory
Ux: x is pearl
Vx: x is slate
Wx: x is north
Xx: x is south
Yx: x is east
Zx: x is west
</predicates>
<premises>
P(c0)
Y(c0)
Z(c0) & P(c0) -> F(c1)
F(c1) -> Z(c1)
W(c0) & P(c0) -> V(c1)
V(c1) -> W(c1)
Y(c0) & P(c0) -> T(c1)
T(c1) -> Y(c1)
X(c0) & P(c0) -> J(c1)
J(c1) -> X(c1)
Z(c1) & F(c1) -> T(c2)
T(c2) -> X(c2)
X(c1) & J(c1) -> N(c2)
N(c2) -> Y(c2)
W(c1) & V(c1) -> Q(c2)
Q(c2) -> W(c2)
Y(c1) & T(c1) -> U(c2)
U(c2) -> Z(c2)
W(c2) & Q(c2) -> Q(c3)
Q(c3) -> Y(c3)
X(c2) & T(c2) -> M(c3)
M(c3) -> Z(c3)
Y(c2) & N(c2) -> R(c3)
R(c3) -> W(c3)
Z(c2) & U(c2) -> C(c3)
C(c3) -> X(c3)
Z(c3) & M(c3) -> S(c4)
S(c4) -> Y(c4)
Y(c3) & Q(c3) -> Q(c4)
Q(c4) -> W(c4)
X(c3) & C(c3) -> T(c4)
T(c4) -> Z(c4)
W(c3) & R(c3) -> I(c4)
I(c4) -> X(c4)
W(c4) & Q(c4) -> J(c5)
J(c5) -> Y(c5)
Z(c4) & T(c4) -> S(c5)
S(c5) -> W(c5)
Y(c4) & S(c4) -> O(c5)
O(c5) -> X(c5)
X(c4) & I(c4) -> H(c5)
H(c5) -> Z(c5)
W(c5) & S(c5) -> Q(c6)
Q(c6) -> X(c6)
Z(c5) & H(c5) -> S(c6)
S(c6) -> Z(c6)
Y(c5) & J(c5) -> K(c6)
K(c6) -> Y(c6)
X(c5) & O(c5) -> V(c6)
V(c6) -> W(c6)
Z(c6) & S(c6) -> G(c7)
G(c7) -> W(c7)
X(c6) & Q(c6) -> E(c7)
E(c7) -> Y(c7)
W(c6) & V(c6) -> C(c7)
C(c7) -> Z(c7)
Y(c6) & K(c6) -> J(c7)
J(c7) -> X(c7)
X(c7) & J(c7) -> J(c8)
J(c8) -> Y(c8)
Z(c7) & C(c7) -> S(c8)
S(c8) -> X(c8)
W(c7) & G(c7) -> R(c8)
R(c8) -> W(c8)
Y(c7) & E(c7) -> L(c8)
L(c8) -> Z(c8)
Y(c8) & J(c8) -> E(c9)
E(c9) -> Z(c9)
X(c8) & S(c8) -> S(c9)
S(c9) -> X(c9)
Z(c8) & L(c8) -> T(c9)
T(c9) -> Y(c9)
W(c8) & R(c8) -> B(c9)
B(c9) -> W(c9)
X(c9) & S(c9) -> K(c10)
K(c10) -> X(c10)
Y(c9) & T(c9) -> S(c10)
S(c10) -> W(c10)
Z(c9) & E(c9) -> E(c10)
E(c10) -> Y(c10)
W(c9) & B(c9) -> I(c10)
I(c10) -> Z(c10)
W(c10) & S(c10) -> H(c11)
H(c11) -> W(c11)
Y(c10) & E(c10) -> I(c11)
I(c11) -> Y(c11)
X(c10) & K(c10) -> S(c11)
S(c11) -> Z(c11)
Z(c10) & I(c10) -> E(c11)
E(c11) -> X(c11)
Z(c11) & S(c11) -> C(c12)
C(c12) -> Y(c12)
X(c11) & E(c11) -> D(c12)
D(c12) -> X(c12)
Y(c11) & I(c11) -> M(c12)
M(c12) -> W(c12)
W(c11) & H(c11) -> N(c12)
N(c12) -> Z(c12)
W(c12) & M(c12) -> A(c13)
A(c13) -> W(c13)
Z(c12) & N(c12) -> K(c13)
K(c13) -> Y(c13)
Y(c12) & C(c12) -> J(c13)
J(c13) -> Z(c13)
X(c12) & D(c12) -> C(c13)
C(c13) -> X(c13)
Z(c13) & J(c13) -> T(c14)
T(c14) -> X(c14)
W(c13) & A(c13) -> C(c14)
C(c14) -> Y(c14)
Y(c13) & K(c13) -> V(c14)
V(c14) -> W(c14)
X(c13) & C(c13) -> A(c14)
A(c14) -> Z(c14)
Y(c14) & C(c14) -> S(c15)
S(c15) -> Y(c15)
X(c14) & T(c14) -> G(c15)
G(c15) -> W(c15)
W(c14) & V(c14) -> L(c15)
L(c15) -> X(c15)
Z(c14) & A(c14) -> B(c15)
B(c15) -> Z(c15)
W(c15) & G(c15) -> I(c16)
I(c16) -> W(c16)
Y(c15) & S(c15) -> J(c16)
J(c16) -> Y(c16)
X(c15) & L(c15) -> U(c16)
U(c16) -> X(c16)
Z(c15) & B(c15) -> F(c16)
F(c16) -> Z(c16)
Z(c16) & F(c16) -> A(c17)
A(c17) -> W(c17)
W(c16) & I(c16) -> N(c17)
N(c17) -> X(c17)
Y(c16) & J(c16) -> B(c17)
B(c17) -> Z(c17)
X(c16) & U(c16) -> E(c17)
E(c17) -> Y(c17)
Z(c17) & B(c17) -> K(c18)
K(c18) -> W(c18)
W(c17) & A(c17) -> O(c18)
O(c18) -> Z(c18)
Y(c17) & E(c17) -> U(c18)
U(c18) -> X(c18)
X(c17) & N(c17) -> T(c18)
T(c18) -> Y(c18)
Y(c18) & T(c18) -> U(c19)
U(c19) -> Z(c19)
Z(c18) & O(c18) -> R(c19)
R(c19) -> X(c19)
W(c18) & K(c18) -> O(c19)
O(c19) -> W(c19)
X(c18) & U(c18) -> A(c19)
A(c19) -> Y(c19)
Z(c19) & U(c19) -> L(c20)
L(c20) -> W(c20)
X(c19) & R(c19) -> I(c20)
I(c20) -> Z(c20)
Y(c19) & A(c19) -> H(c20)
H(c20) -> Y(c20)
W(c19) & O(c19) -> E(c20)
E(c20) -> X(c20)
X(c20) & E(c20) -> L(c21)
L(c21) -> Z(c21)
Y(c20) & H(c20) -> C(c21)
C(c21) -> Y(c21)
W(c20) & L(c20) -> K(c21)
K(c21) -> W(c21)
Z(c20) & I(c20) -> E(c21)
E(c21) -> X(c21)
</premises>
<proof>
P(c0) ; R
Y(c0) ; R
T(c1) ; ->E
Y(c1) ; ->E
U(c2) ; ->E
Z(c2) ; ->E
C(c3) ; ->E
X(c3) ; ->E
T(c4) ; ->E
Z(c4) ; ->E
S(c5) ; ->E
W(c5) ; ->E
Q(c6) ; ->E
X(c6) ; ->E
E(c7) ; ->E
Y(c7) ; ->E
L(c8) ; ->E
Z(c8) ; ->E
T(c9) ; ->E
Y(c9) ; ->E
S(c10) ; ->E
W(c10) ; ->E
H(c11) ; ->E
W(c11) ; ->E
N(c12) ; ->E
Z(c12) ; ->E
K(c13) ; ->E
Y(c13) ; ->E
V(c14) ; ->E
W(c14) ; ->E
L(c15) ; ->E
X(c15) ; ->E
U(c16) ; ->E
X(c16) ; ->E
E(c17) ; ->E
Y(c17) ; ->E
U(c18) ; ->E
X(c18) ; ->E
A(c19) ; ->E
Y(c19) ; ->E
H(c20) ; ->E
Y(c20) ; ->E
C(c21) ; ->E
</proof>
<conclusion>
C(c21)
</conclusion>
</formal>
<answer>
coral
</answer><|endoftext|><question>
1. c0 is olive.
2. c0 is south.
3. If c0 is south and c0 is olive, then c1 is laurel.
4. If c1 is laurel, then c1 is west.
5. If c0 is west and c0 is olive, then c1 is granite.
6. If c1 is granite, then c1 is south.
7. If c0 is north and c0 is olive, then c1 is harbor.
8. If c1 is harbor, then c1 is east.
9. If c0 is east and c0 is olive, then c1 is elm.
10. If c1 is elm, then c1 is north.
11. If c1 is west and c1 is laurel, then c2 is harbor.
12. If c2 is harbor, then c2 is north.
13. If c1 is north and c1 is elm, then c2 is laurel.
14. If c2 is laurel, then c2 is south.
15. If c1 is east and c1 is harbor, then c2 is birch.
16. If c2 is birch, then c2 is east.
17. If c1 is south and c1 is granite, then c2 is granite.
18. If c2 is granite, then c2 is west.
19. If c2 is west and c2 is granite, then c3 is harbor.
20. If c3 is harbor, then c3 is north.
21. If c2 is east and c2 is birch, then c3 is granite.
22. If c3 is granite, then c3 is east.
23. If c2 is north and c2 is harbor, then c3 is elm.
24. If c3 is elm, then c3 is south.
25. If c2 is south and c2 is laurel, then c3 is laurel.
26. If c3 is laurel, then c3 is west.
27. If c3 is east and c3 is granite, then c4 is elm.
28. If c4 is elm, then c4 is west.
29. If c3 is south and c3 is elm, then c4 is laurel.
30. If c4 is laurel, then c4 is east.
31. If c3 is west and c3 is laurel, then c4 is granite.
32. If c4 is granite, then c4 is south.
33. If c3 is north and c3 is harbor, then c4 is harbor.
34. If c4 is harbor, then c4 is north.
35. If c4 is south and c4 is granite, then c5 is birch.
36. If c5 is birch, then c5 is east.
37. If c4 is north and c4 is harbor, then c5 is harbor.
38. If c5 is harbor, then c5 is west.
39. If c4 is east and c4 is laurel, then c5 is granite.
40. If c5 is granite, then c5 is south.
41. If c4 is west and c4 is elm, then c5 is laurel.
42. If c5 is laurel, then c5 is north.
Which state applies to c5?
</question>

<formal>
<constants>
c0 = c0
c1 = c1
c2 = c2
c3 = c3
c4 = c4
c5 = c5
</constants>
<predicates>
Ax: x is laurel
Bx: x is birch
Cx: x is granite
Dx: x is harbor
Ex: x is elm
Fx: x is olive
Gx: x is north
Hx: x is south
Ix: x is east
Jx: x is west
</predicates>
<premises>
F(c0)
H(c0)
H(c0) & F(c0) -> A(c1)
A(c1) -> J(c1)
J(c0) & F(c0) -> C(c1)
C(c1) -> H(c1)
G(c0) & F(c0) -> D(c1)
D(c1) -> I(c1)
I(c0) & F(c0) -> E(c1)
E(c1) -> G(c1)
J(c1) & A(c1) -> D(c2)
D(c2) -> G(c2)
G(c1) & E(c1) -> A(c2)
A(c2) -> H(c2)
I(c1) & D(c1) -> B(c2)
B(c2) -> I(c2)
H(c1) & C(c1) -> C(c2)
C(c2) -> J(c2)
J(c2) & C(c2) -> D(c3)
D(c3) -> G(c3)
I(c2) & B(c2) -> C(c3)
C(c3) -> I(c3)
G(c2) & D(c2) -> E(c3)
E(c3) -> H(c3)
H(c2) & A(c2) -> A(c3)
A(c3) -> J(c3)
I(c3) & C(c3) -> E(c4)
E(c4) -> J(c4)
H(c3) & E(c3) -> A(c4)
A(c4) -> I(c4)
J(c3) & A(c3) -> C(c4)
C(c4) -> H(c4)
G(c3) & D(c3) -> D(c4)
D(c4) -> G(c4)
H(c4) & C(c4) -> B(c5)
B(c5) -> I(c5)
G(c4) & D(c4) -> D(c5)
D(c5) -> J(c5)
I(c4) & A(c4) -> C(c5)
C(c5) -> H(c5)
J(c4) & E(c4) -> A(c5)
A(c5) -> G(c5)
</premises>
<proof>
F(c0) ; R
H(c0) ; R
A(c1) ; ->E
J(c1) ; ->E
D(c2) ; ->E
G(c2) ; ->E
E(c3) ; ->E
H(c3) ; ->E
A(c4) ; ->E
I(c4) ; ->E
C(c5) ; ->E
</proof>
<conclusion>
C(c5)
</conclusion>
</formal>
<answer>
granite
</answer><|endoftext|>
```

## Window 1 (2 documents, 253 pad tokens)
```
<question>
1. c0 is elm.
2. c0 is west.
3. If c0 is east and c0 is elm, then c1 is pearl.
4. If c1 is pearl, then c1 is north.
5. If c0 is north and c0 is elm, then c1 is ivory.
6. If c1 is ivory, then c1 is south.
7. If c0 is west and c0 is elm, then c1 is olive.
8. If c1 is olive, then c1 is west.
9. If c0 is south and c0 is elm, then c1 is maple.
10. If c1 is maple, then c1 is east.
11. If c1 is south and c1 is ivory, then c2 is meadow.
12. If c2 is meadow, then c2 is north.
13. If c1 is north and c1 is pearl, then c2 is pearl.
14. If c2 is pearl, then c2 is south.
15. If c1 is west and c1 is olive, then c2 is slate.
16. If c2 is slate, then c2 is east.
17. If c1 is east and c1 is maple, then c2 is amber.
18. If c2 is amber, then c2 is west.
19. If c2 is south and c2 is pearl, then c3 is coral.
20. If c3 is coral, then c3 is north.
21. If c2 is east and c2 is slate, then c3 is birch.
22. If c3 is birch, then c3 is south.
23. If c2 is north and c2 is meadow, then c3 is ivory.
24. If c3 is ivory, then c3 is west.
25. If c2 is west and c2 is amber, then c3 is poppy.
26. If c3 is poppy, then c3 is east.
27. If c3 is south and c3 is birch, then c4 is granite.
28. If c4 is granite, then c4 is south.
29. If c3 is north and c3 is coral, then c4 is coral.
30. If c4 is coral, then c4 is west.
31. If c3 is east and c3 is poppy, then c4 is olive.
32. If c4 is olive, then c4 is north.
33. If c3 is west and c3 is ivory, then c4 is juniper.
34. If c4 is juniper, then c4 is east.
35. If c4 is west and c4 is coral, then c5 is orchid.
36. If c5 is orchid, then c5 is west.
37. If c4 is north and c4 is olive, then c5 is violet.
38. If c5 is violet, then c5 is north.
39. If c4 is south and c4 is granite, then c5 is amber.
40. If c5 is amber, then c5 is south.
41. If c4 is east and c4 is juniper, then c5 is cobalt.
42. If c5 is cobalt, then c5 is east.
43. If c5 is east and c5 is cobalt, then c6 is meadow.
44. If c6 is meadow, then c6 is west.
45. If c5 is south and c5 is amber, then c6 is amber.
46. If c6 is amber, then c6 is south.
47. If c5 is west and c5 is orchid, then c6 is cedar.
48. If c6 is cedar, then c6 is east.
49. If c5 is north and c5 is violet, then c6 is birch.
50. If c6 is birch, then c6 is north.
51. If c6 is south and c6 is amber, then c7 is hazel.
52. If c7 is hazel, then c7 is north.
53. If c6 is west and c6 is meadow, then c7 is violet.
54. If c7 is violet, then c7 is east.
55. If c6 is north and c6 is birch, then c7 is coral.
56. If c7 is coral, then c7 is west.
57. If c6 is east and c6 is cedar, then c7 is cobalt.
58. If c7 is cobalt, then c7 is south.
59. If c7 is east and c7 is violet, then c8 is maple.
60. If c8 is maple, then c8 is east.
61. If c7 is west and c7 is coral, then c8 is amber.
62. If c8 is amber, then c8 is north.
63. If c7 is north and c7 is hazel, then c8 is willow.
64. If c8 is willow, then c8 is west.
65. If c7 is south and c7 is cobalt, then c8 is meadow.
66. If c8 is meadow, then c8 is south.
67. If c8 is north and c8 is amber, then c9 is olive.
68. If c9 is olive, then c9 is west.
69. If c8 is south and c8 is meadow, then c9 is orchid.
70. If c9 is orchid, then c9 is east.
71. If c8 is east and c8 is maple, then c9 is pearl.
72. If c9 is pearl, then c9 is south.
73. If c8 is west and c8 is willow, then c9 is ivory.
74. If c9 is ivory, then c9 is north.
75. If c9 is south and c9 is pearl, then c10 is ivory.
76. If c10 is ivory, then c10 is east.
77. If c9 is west and c9 is olive, then c10 is cedar.
78. If c10 is cedar, then c10 is south.
79. If c9 is east and c9 is orchid, then c10 is amber.
80. If c10 is amber, then c10 is west.
81. If c9 is north and c9 is ivory, then c10 is hazel.
82. If c10 is hazel, then c10 is north.
83. If c10 is north and c10 is hazel, then c11 is violet.
84. If c11 is violet, then c11 is north.
85. If c10 is east and c10 is ivory, then c11 is birch.
86. If c11 is birch, then c11 is east.
87. If c10 is south and c10 is cedar, then c11 is olive.
88. If c11 is olive, then c11 is west.
89. If c10 is west and c10 is amber, then c11 is maple.
90. If c11 is maple, then c11 is south.
91. If c11 is east and c11 is birch, then c12 is coral.
92. If c12 is coral, then c12 is west.
93. If c11 is south and c11 is maple, then c12 is hazel.
94. If c12 is hazel, then c12 is east.
95. If c11 is north and c11 is violet, then c12 is birch.
96. If c12 is birch, then c12 is north.
97. If c11 is west and c11 is olive, then c12 is laurel.
98. If c12 is laurel, then c12 is south.
99. If c12 is north and c12 is birch, then c13 is birch.
100. If c13 is birch, then c13 is south.
101. If c12 is west and c12 is coral, then c13 is poppy.
102. If c13 is poppy, then c13 is east.
103. If c12 is east and c12 is hazel, then c13 is laurel.
104. If c13 is laurel, then c13 is west.
105. If c12 is south and c12 is laurel, then c13 is coral.
106. If c13 is coral, then c13 is north.
107. If c13 is south and c13 is birch, then c14 is slate.
108. If c14 is slate, then c14 is south.
109. If c13 is east and c13 is poppy, then c14 is cobalt.
110. If c14 is cobalt, then c14 is west.
111. If c13 is north and c13 is coral, then c14 is juniper.
112. If c14 is juniper, then c14 is north.
113. If c13 is west and c13 is laurel, then c14 is maple.
114. If c14 is maple, then c14 is east.
115. If c14 is west and c14 is cobalt, then c15 is meadow.
116. If c15 is meadow, then c15 is east.
117. If c14 is north and c14 is juniper, then c15 is cobalt.
118. If c15 is cobalt, then c15 is west.
119. If c14 is south and c14 is slate, then c15 is laurel.
120. If c15 is laurel, then c15 is north.
121. If c14 is east and c14 is maple, then c15 is violet.
122. If c15 is violet, then c15 is south.
123. If c15 is south and c15 is violet, then c16 is slate.
124. If c16 is slate, then c16 is west.
125. If c15 is west and c15 is cobalt, then c16 is amber.
126. If c16 is amber, then c16 is east.
127. If c15 is east and c15 is meadow, then c16 is olive.
128. If c16 is olive, then c16 is south.
129. If c15 is north and c15 is laurel, then c16 is violet.
130. If c16 is violet, then c16 is north.
131. If c16 is east and c16 is amber, then c17 is violet.
132. If c17 is violet, then c17 is east.
133. If c16 is north and c16 is violet, then c17 is laurel.
134. If c17 is laurel, then c17 is north.
135. If c16 is south and c16 is olive, then c17 is orchid.
136. If c17 is orchid, then c17 is west.
137. If c16 is west and c16 is slate, then c17 is pearl.
138. If c17 is pearl, then c17 is south.
139. If c17 is south and c17 is pearl, then c18 is pearl.
140. If c18 is pearl, then c18 is west.
141. If c17 is north and c17 is laurel, then c18 is granite.
142. If c18 is granite, then c18 is north.
143. If c17 is east and c17 is violet, then c18 is olive.
144. If c18 is olive, then c18 is south.
145. If c17 is west and c17 is orchid, then c18 is ivory.
146. If c18 is ivory, then c18 is east.
147. If c18 is east and c18 is ivory, then c19 is slate.
148. If c19 is slate, then c19 is south.
149. If c18 is north and c18 is granite, then c19 is olive.
150. If c19 is olive, then c19 is west.
151. If c18 is west and c18 is pearl, then c19 is orchid.
152. If c19 is orchid, then c19 is north.
153. If c18 is south and c18 is olive, then c19 is laurel.
154. If c19 is laurel, then c19 is east.
Which state applies to c19?
</question>

<formal>
<constants>
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
c15 = c15
c16 = c16
c17 = c17
c18 = c18
c19 = c19
</constants>
<predicates>
Ax: x is olive
Bx: x is willow
Cx: x is birch
Dx: x is granite
Ex: x is maple
Fx: x is laurel
Gx: x is pearl
Hx: x is slate
Ix: x is violet
Jx: x is juniper
Kx: x is elm
Lx: x is cedar
Mx: x is poppy
Nx: x is orchid
Ox: x is hazel
Px: x is meadow
Qx: x is amber
Rx: x is cobalt
Sx: x is coral
Tx: x is ivory
Ux: x is north
Vx: x is south
Wx: x is east
Xx: x is west
</predicates>
<premises>
K(c0)
X(c0)
W(c0) & K(c0) -> G(c1)
G(c1) -> U(c1)
U(c0) & K(c0) -> T(c1)
T(c1) -> V(c1)
X(c0) & K(c0) -> A(c1)
A(c1) -> X(c1)
V(c0) & K(c0) -> E(c1)
E(c1) -> W(c1)
V(c1) & T(c1) -> P(c2)
P(c2) -> U(c2)
U(c1) & G(c1) -> G(c2)
G(c2) -> V(c2)
X(c1) & A(c1) -> H(c2)
H(c2) -> W(c2)
W(c1) & E(c1) -> Q(c2)
Q(c2) -> X(c2)
V(c2) & G(c2) -> S(c3)
S(c3) -> U(c3)
W(c2) & H(c2) -> C(c3)
C(c3) -> V(c3)
U(c2) & P(c2) -> T(c3)
T(c3) -> X(c3)
X(c2) & Q(c2) -> M(c3)
M(c3) -> W(c3)
V(c3) & C(c3) -> D(c4)
D(c4) -> V(c4)
U(c3) & S(c3) -> S(c4)
S(c4) -> X(c4)
W(c3) & M(c3) -> A(c4)
A(c4) -> U(c4)
X(c3) & T(c3) -> J(c4)
J(c4) -> W(c4)
X(c4) & S(c4) -> N(c5)
N(c5) -> X(c5)
U(c4) & A(c4) -> I(c5)
I(c5) -> U(c5)
V(c4) & D(c4) -> Q(c5)
Q(c5) -> V(c5)
W(c4) & J(c4) -> R(c5)
R(c5) -> W(c5)
W(c5) & R(c5) -> P(c6)
P(c6) -> X(c6)
V(c5) & Q(c5) -> Q(c6)
Q(c6) -> V(c6)
X(c5) & N(c5) -> L(c6)
L(c6) -> W(c6)
U(c5) & I(c5) -> C(c6)
C(c6) -> U(c6)
V(c6) & Q(c6) -> O(c7)
O(c7) -> U(c7)
X(c6) & P(c6) -> I(c7)
I(c7) -> W(c7)
U(c6) & C(c6) -> S(c7)
S(c7) -> X(c7)
W(c6) & L(c6) -> R(c7)
R(c7) -> V(c7)
W(c7) & I(c7) -> E(c8)
E(c8) -> W(c8)
X(c7) & S(c7) -> Q(c8)
Q(c8) -> U(c8)
U(c7) & O(c7) -> B(c8)
B(c8) -> X(c8)
V(c7) & R(c7) -> P(c8)
P(c8) -> V(c8)
U(c8) & Q(c8) -> A(c9)
A(c9) -> X(c9)
V(c8) & P(c8) -> N(c9)
N(c9) -> W(c9)
W(c8) & E(c8) -> G(c9)
G(c9) -> V(c9)
X(c8) & B(c8) -> T(c9)
T(c9) -> U(c9)
V(c9) & G(c9) -> T(c10)
T(c10) -> W(c10)
X(c9) & A(c9) -> L(c10)
L(c10) -> V(c10)
W(c9) & N(c9) -> Q(c10)
Q(c10) -> X(c10)
U(c9) & T(c9) -> O(c10)
O(c10) -> U(c10)
U(c10) & O(c10) -> I(c11)
I(c11) -> U(c11)
W(c10) & T(c10) -> C(c11)
C(c11) -> W(c11)
V(c10) & L(c10) -> A(c11)
A(c11) -> X(c11)
X(c10) & Q(c10) -> E(c11)
E(c11) -> V(c11)
W(c11) & C(c11) -> S(c12)
S(c12) -> X(c12)
V(c11) & E(c11) -> O(c12)
O(c12) -> W(c12)
U(c11) & I(c11) -> C(c12)
C(c12) -> U(c12)
X(c11) & A(c11) -> F(c12)
F(c12) -> V(c12)
U(c12) & C(c12) -> C(c13)
C(c13) -> V(c13)
X(c12) & S(c12) -> M(c13)
M(c13) -> W(c13)
W(c12) & O(c12) -> F(c13)
F(c13) -> X(c13)
V(c12) & F(c12) -> S(c13)
S(c13) -> U(c13)
V(c13) & C(c13) -> H(c14)
H(c14) -> V(c14)
W(c13) & M(c13) -> R(c14)
R(c14) -> X(c14)
U(c13) & S(c13) -> J(c14)
J(c14) -> U(c14)
X(c13) & F(c13) -> E(c14)
E(c14) -> W(c14)
X(c14) & R(c14) -> P(c15)
P(c15) -> W(c15)
U(c14) & J(c14) -> R(c15)
R(c15) -> X(c15)
V(c14) & H(c14) -> F(c15)
F(c15) -> U(c15)
W(c14) & E(c14) -> I(c15)
I(c15) -> V(c15)
V(c15) & I(c15) -> H(c16)
H(c16) -> X(c16)
X(c15) & R(c15) -> Q(c16)
Q(c16) -> W(c16)
W(c15) & P(c15) -> A(c16)
A(c16) -> V(c16)
U(c15) & F(c15) -> I(c16)
I(c16) -> U(c16)
W(c16) & Q(c16) -> I(c17)
I(c17) -> W(c17)
U(c16) & I(c16) -> F(c17)
F(c17) -> U(c17)
V(c16) & A(c16) -> N(c17)
N(c17) -> X(c17)
X(c16) & H(c16) -> G(c17)
G(c17) -> V(c17)
V(c17) & G(c17) -> G(c18)
G(c18) -> X(c18)
U(c17) & F(c17) -> D(c18)
D(c18) -> U(c18)
W(c17) & I(c17) -> A(c18)
A(c18) -> V(c18)
X(c17) & N(c17) -> T(c18)
T(c18) -> W(c18)
W(c18) & T(c18) -> H(c19)
H(c19) -> V(c19)
U(c18) & D(c18) -> A(c19)
A(c19) -> X(c19)
X(c18) & G(c18) -> N(c19)
N(c19) -> U(c19)
V(c18) & A(c18) -> F(c19)
F(c19) -> W(c19)
</premises>
<proof>
K(c0) ; R
X(c0) ; R
A(c1) ; ->E
X(c1) ; ->E
H(c2) ; ->E
W(c2) ; ->E
C(c3) ; ->E
V(c3) ; ->E
D(c4) ; ->E
V(c4) ; ->E
Q(c5) ; ->E
V(c5) ; ->E
Q(c6) ; ->E
V(c6) ; ->E
O(c7) ; ->E
U(c7) ; ->E
B(c8) ; ->E
X(c8) ; ->E
T(c9) ; ->E
U(c9) ; ->E
O(c10) ; ->E
U(c10) ; ->E
I(c11) ; ->E
U(c11) ; ->E
C(c12) ; ->E
U(c12) ; ->E
C(c13) ; ->E
V(c13) ; ->E
H(c14) ; ->E
V(c14) ; ->E
F(c15) ; ->E
U(c15) ; ->E
I(c16) ; ->E
U(c16) ; ->E
F(c17) ; ->E
U(c17) ; ->E
D(c18) ; ->E
U(c18) ; ->E
A(c19) ; ->E
</proof>
<conclusion>
A(c19)
</conclusion>
</formal>
<answer>
olive
</answer><|endoftext|><question>
1. c0 is meadow.
2. c0 is west.
3. If c0 is south and c0 is meadow, then c1 is birch.
4. If c1 is birch, then c1 is east.
5. If c0 is north and c0 is meadow, then c1 is orchid.
6. If c1 is orchid, then c1 is west.
7. If c0 is west and c0 is meadow, then c1 is hazel.
8. If c1 is hazel, then c1 is south.
9. If c0 is east and c0 is meadow, then c1 is harbor.
10. If c1 is harbor, then c1 is north.
11. If c1 is south and c1 is hazel, then c2 is violet.
12. If c2 is violet, then c2 is south.
13. If c1 is east and c1 is birch, then c2 is coral.
14. If c2 is coral, then c2 is north.
15. If c1 is north and c1 is harbor, then c2 is hazel.
16. If c2 is hazel, then c2 is west.
17. If c1 is west and c1 is orchid, then c2 is orchid.
18. If c2 is orchid, then c2 is east.
19. If c2 is east and c2 is orchid, then c3 is laurel.
20. If c3 is laurel, then c3 is west.
21. If c2 is west and c2 is hazel, then c3 is birch.
22. If c3 is birch, then c3 is north.
23. If c2 is south and c2 is violet, then c3 is harbor.
24. If c3 is harbor, then c3 is south.
25. If c2 is north and c2 is coral, then c3 is violet.
26. If c3 is violet, then c3 is east.
27. If c3 is south and c3 is harbor, then c4 is violet.
28. If c4 is violet, then c4 is north.
29. If c3 is north and c3 is birch, then c4 is orchid.
30. If c4 is orchid, then c4 is east.
31. If c3 is east and c3 is violet, then c4 is harbor.
32. If c4 is harbor, then c4 is south.
33. If c3 is west and c3 is laurel, then c4 is hazel.
34. If c4 is hazel, then c4 is west.
35. If c4 is east and c4 is orchid, then c5 is hazel.
36. If c5 is hazel, then c5 is east.
37. If c4 is north and c4 is violet, then c5 is birch.
38. If c5 is birch, then c5 is south.
39. If c4 is west and c4 is hazel, then c5 is teal.
40. If c5 is teal, then c5 is west.
41. If c4 is south and c4 is harbor, then c5 is harbor.
42. If c5 is harbor, then c5 is north.
43. If c5 is west and c5 is teal, then c6 is birch.
44. If c6 is birch, then c6 is north.
45. If c5 is north and c5 is harbor, then c6 is coral.
46. If c6 is coral, then c6 is south.
47. If c5 is south and c5 is birch, then c6 is laurel.
48. If c6 is laurel, then c6 is east.
49. If c5 is east and c5 is hazel, then c6 is hazel.
50. If c6 is hazel, then c6 is west.
51. If c6 is south and c6 is coral, then c7 is teal.
52. If c7 is teal, then c7 is west.
53. If c6 is north and c6 is birch, then c7 is coral.
54. If c7 is coral, then c7 is south.
55. If c6 is east and c6 is laurel, then c7 is hazel.
56. If c7 is hazel, then c7 is north.
57. If c6 is west and c6 is hazel, then c7 is laurel.
58. If c7 is laurel, then c7 is east.
59. If c7 is south and c7 is coral, then c8 is birch.
60. If c8 is birch, then c8 is east.
61. If c7 is west and c7 is teal, then c8 is laurel.
62. If c8 is laurel, then c8 is west.
63. If c7 is east and c7 is laurel, then c8 is orchid.
64. If c8 is orchid, then c8 is north.
65. If c7 is north and c7 is hazel, then c8 is coral.
66. If c8 is coral, then c8 is south.
Which state applies to c8?
</question>

<formal>
<constants>
c0 = c0
c1 = c1
c2 = c2
c3 = c3
c4 = c4
c5 = c5
c6 = c6
c7 = c7
c8 = c8
</constants>
<predicates>
Ax: x is laurel
Bx: x is meadow
Cx: x is orchid
Dx: x is violet
Ex: x is harbor
Fx: x is hazel
Gx: x is birch
Hx: x is teal
Ix: x is coral
Jx: x is north
Kx: x is south
Lx: x is east
Mx: x is west
</predicates>
<premises>
B(c0)
M(c0)
K(c0) & B(c0) -> G(c1)
G(c1) -> L(c1)
J(c0) & B(c0) -> C(c1)
C(c1) -> M(c1)
M(c0) & B(c0) -> F(c1)
F(c1) -> K(c1)
L(c0) & B(c0) -> E(c1)
E(c1) -> J(c1)
K(c1) & F(c1) -> D(c2)
D(c2) -> K(c2)
L(c1) & G(c1) -> I(c2)
I(c2) -> J(c2)
J(c1) & E(c1) -> F(c2)
F(c2) -> M(c2)
M(c1) & C(c1) -> C(c2)
C(c2) -> L(c2)
L(c2) & C(c2) -> A(c3)
A(c3) -> M(c3)
M(c2) & F(c2) -> G(c3)
G(c3) -> J(c3)
K(c2) & D(c2) -> E(c3)
E(c3) -> K(c3)
J(c2) & I(c2) -> D(c3)
D(c3) -> L(c3)
K(c3) & E(c3) -> D(c4)
D(c4) -> J(c4)
J(c3) & G(c3) -> C(c4)
C(c4) -> L(c4)
L(c3) & D(c3) -> E(c4)
E(c4) -> K(c4)
M(c3) & A(c3) -> F(c4)
F(c4) -> M(c4)
L(c4) & C(c4) -> F(c5)
F(c5) -> L(c5)
J(c4) & D(c4) -> G(c5)
G(c5) -> K(c5)
M(c4) & F(c4) -> H(c5)
H(c5) -> M(c5)
K(c4) & E(c4) -> E(c5)
E(c5) -> J(c5)
M(c5) & H(c5) -> G(c6)
G(c6) -> J(c6)
J(c5) & E(c5) -> I(c6)
I(c6) -> K(c6)
K(c5) & G(c5) -> A(c6)
A(c6) -> L(c6)
L(c5) & F(c5) -> F(c6)
F(c6) -> M(c6)
K(c6) & I(c6) -> H(c7)
H(c7) -> M(c7)
J(c6) & G(c6) -> I(c7)
I(c7) -> K(c7)
L(c6) & A(c6) -> F(c7)
F(c7) -> J(c7)
M(c6) & F(c6) -> A(c7)
A(c7) -> L(c7)
K(c7) & I(c7) -> G(c8)
G(c8) -> L(c8)
M(c7) & H(c7) -> A(c8)
A(c8) -> M(c8)
L(c7) & A(c7) -> C(c8)
C(c8) -> J(c8)
J(c7) & F(c7) -> I(c8)
I(c8) -> K(c8)
</premises>
<proof>
B(c0) ; R
M(c0) ; R
F(c1) ; ->E
K(c1) ; ->E
D(c2) ; ->E
K(c2) ; ->E
E(c3) ; ->E
K(c3) ; ->E
D(c4) ; ->E
J(c4) ; ->E
G(c5) ; ->E
K(c5) ; ->E
A(c6) ; ->E
L(c6) ; ->E
F(c7) ; ->E
J(c7) ; ->E
I(c8) ; ->E
</proof>
<conclusion>
I(c8)
</conclusion>
</formal>
<answer>
coral
</answer><|endoftext|>
```

## Window 2 (2 documents, 489 pad tokens)
```
<question>
1. c0 is maple.
2. c0 is east.
3. If c0 is east and c0 is maple, then c1 is pearl.
4. If c1 is pearl, then c1 is south.
5. If c0 is north and c0 is maple, then c1 is olive.
6. If c1 is olive, then c1 is west.
7. If c0 is west and c0 is maple, then c1 is cobalt.
8. If c1 is cobalt, then c1 is east.
9. If c0 is south and c0 is maple, then c1 is elm.
10. If c1 is elm, then c1 is north.
11. If c1 is south and c1 is pearl, then c2 is harbor.
12. If c2 is harbor, then c2 is south.
13. If c1 is north and c1 is elm, then c2 is willow.
14. If c2 is willow, then c2 is west.
15. If c1 is west and c1 is olive, then c2 is hazel.
16. If c2 is hazel, then c2 is north.
17. If c1 is east and c1 is cobalt, then c2 is violet.
18. If c2 is violet, then c2 is east.
19. If c2 is east and c2 is violet, then c3 is pearl.
20. If c3 is pearl, then c3 is east.
21. If c2 is south and c2 is harbor, then c3 is slate.
22. If c3 is slate, then c3 is south.
23. If c2 is north and c2 is hazel, then c3 is meadow.
24. If c3 is meadow, then c3 is north.
25. If c2 is west and c2 is willow, then c3 is violet.
26. If c3 is violet, then c3 is west.
27. If c3 is east and c3 is pearl, then c4 is slate.
28. If c4 is slate, then c4 is north.
29. If c3 is south and c3 is slate, then c4 is orchid.
30. If c4 is orchid, then c4 is south.
31. If c3 is north and c3 is meadow, then c4 is willow.
32. If c4 is willow, then c4 is west.
33. If c3 is west and c3 is violet, then c4 is lime.
34. If c4 is lime, then c4 is east.
35. If c4 is east and c4 is lime, then c5 is lime.
36. If c5 is lime, then c5 is east.
37. If c4 is north and c4 is slate, then c5 is pearl.
38. If c5 is pearl, then c5 is south.
39. If c4 is south and c4 is orchid, then c5 is cedar.
40. If c5 is cedar, then c5 is north.
41. If c4 is west and c4 is willow, then c5 is cobalt.
42. If c5 is cobalt, then c5 is west.
43. If c5 is north and c5 is cedar, then c6 is willow.
44. If c6 is willow, then c6 is east.
45. If c5 is east and c5 is lime, then c6 is harbor.
46. If c6 is harbor, then c6 is north.
47. If c5 is west and c5 is cobalt, then c6 is laurel.
48. If c6 is laurel, then c6 is south.
49. If c5 is south and c5 is pearl, then c6 is granite.
50. If c6 is granite, then c6 is west.
51. If c6 is east and c6 is willow, then c7 is teal.
52. If c7 is teal, then c7 is north.
53. If c6 is north and c6 is harbor, then c7 is orchid.
54. If c7 is orchid, then c7 is south.
55. If c6 is south and c6 is laurel, then c7 is harbor.
56. If c7 is harbor, then c7 is east.
57. If c6 is west and c6 is granite, then c7 is meadow.
58. If c7 is meadow, then c7 is west.
59. If c7 is east and c7 is harbor, then c8 is harbor.
60. If c8 is harbor, then c8 is north.
61. If c7 is north and c7 is teal, then c8 is cobalt.
62. If c8 is cobalt, then c8 is east.
63. If c7 is south and c7 is orchid, then c8 is poppy.
64. If c8 is poppy, then c8 is south.
65. If c7 is west and c7 is meadow, then c8 is olive.
66. If c8 is olive, then c8 is west.
67. If c8 is east and c8 is cobalt, then c9 is hazel.
68. If c9 is hazel, then c9 is east.
69. If c8 is north and c8 is harbor, then c9 is willow.
70. If c9 is willow, then c9 is west.
71. If c8 is south and c8 is poppy, then c9 is coral.
72. If c9 is coral, then c9 is south.
73. If c8 is west and c8 is olive, then c9 is violet.
74. If c9 is violet, then c9 is north.
75. If c9 is north and c9 is violet, then c10 is cobalt.
76. If c10 is cobalt, then c10 is west.
77. If c9 is west and c9 is willow, then c10 is amber.
78. If c10 is amber, then c10 is south.
79. If c9 is south and c9 is coral, then c10 is slate.
80. If c10 is slate, then c10 is east.
81. If c9 is east and c9 is hazel, then c10 is meadow.
82. If c10 is meadow, then c10 is north.
83. If c10 is south and c10 is amber, then c11 is slate.
84. If c11 is slate, then c11 is south.
85. If c10 is west and c10 is cobalt, then c11 is poppy.
86. If c11 is poppy, then c11 is east.
87. If c10 is east and c10 is slate, then c11 is hazel.
88. If c11 is hazel, then c11 is west.
89. If c10 is north and c10 is meadow, then c11 is olive.
90. If c11 is olive, then c11 is north.
91. If c11 is south and c11 is slate, then c12 is slate.
92. If c12 is slate, then c12 is west.
93. If c11 is east and c11 is poppy, then c12 is laurel.
94. If c12 is laurel, then c12 is south.
95. If c11 is west and c11 is hazel, then c12 is amber.
96. If c12 is amber, then c12 is north.
97. If c11 is north and c11 is olive, then c12 is poppy.
98. If c12 is poppy, then c12 is east.
99. If c12 is east and c12 is poppy, then c13 is orchid.
100. If c13 is orchid, then c13 is south.
101. If c12 is south and c12 is laurel, then c13 is granite.
102. If c13 is granite, then c13 is north.
103. If c12 is north and c12 is amber, then c13 is willow.
104. If c13 is willow, then c13 is west.
105. If c12 is west and c12 is slate, then c13 is juniper.
106. If c13 is juniper, then c13 is east.
107. If c13 is south and c13 is orchid, then c14 is willow.
108. If c14 is willow, then c14 is north.
109. If c13 is east and c13 is juniper, then c14 is juniper.
110. If c14 is juniper, then c14 is east.
111. If c13 is north and c13 is granite, then c14 is pearl.
112. If c14 is pearl, then c14 is south.
113. If c13 is west and c13 is willow, then c14 is meadow.
114. If c14 is meadow, then c14 is west.
115. If c14 is south and c14 is pearl, then c15 is poppy.
116. If c15 is poppy, then c15 is north.
117. If c14 is north and c14 is willow, then c15 is elm.
118. If c15 is elm, then c15 is south.
119. If c14 is west and c14 is meadow, then c15 is cobalt.
120. If c15 is cobalt, then c15 is west.
121. If c14 is east and c14 is juniper, then c15 is coral.
122. If c15 is coral, then c15 is east.
123. If c15 is east and c15 is coral, then c16 is pearl.
124. If c16 is pearl, then c16 is west.
125. If c15 is north and c15 is poppy, then c16 is teal.
126. If c16 is teal, then c16 is north.
127. If c15 is west and c15 is cobalt, then c16 is lime.
128. If c16 is lime, then c16 is south.
129. If c15 is south and c15 is elm, then c16 is laurel.
130. If c16 is laurel, then c16 is east.
131. If c16 is north and c16 is teal, then c17 is coral.
132. If c17 is coral, then c17 is south.
133. If c16 is south and c16 is lime, then c17 is violet.
134. If c17 is violet, then c17 is north.
135. If c16 is west and c16 is pearl, then c17 is teal.
136. If c17 is teal, then c17 is east.
137. If c16 is east and c16 is laurel, then c17 is granite.
138. If c17 is granite, then c17 is west.
139. If c17 is east and c17 is teal, then c18 is teal.
140. If c18 is teal, then c18 is east.
141. If c17 is north and c17 is violet, then c18 is harbor.
142. If c18 is harbor, then c18 is north.
143. If c17 is south and c17 is coral, then c18 is meadow.
144. If c18 is meadow, then c18 is west.
145. If c17 is west and c17 is granite, then c18 is juniper.
146. If c18 is juniper, then c18 is south.
147. If c18 is north and c18 is harbor, then c19 is harbor.
148. If c19 is harbor, then c19 is east.
149. If c18 is south and c18 is juniper, then c19 is willow.
150. If c19 is willow, then c19 is south.
151. If c18 is west and c18 is meadow, then c19 is pearl.
152. If c19 is pearl, then c19 is north.
153. If c18 is east and c18 is teal, then c19 is slate.
154. If c19 is slate, then c19 is west.
155. If c19 is north and c19 is pearl, then c20 is teal.
156. If c20 is teal, then c20 is west.
157. If c19 is south and c19 is willow, then c20 is cedar.
158. If c20 is cedar, then c20 is south.
159. If c19 is east and c19 is harbor, then c20 is slate.
160. If c20 is slate, then c20 is north.
161. If c19 is west and c19 is slate, then c20 is olive.
162. If c20 is olive, then c20 is east.
Which state applies to c20?
</question>

<formal>
<constants>
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
c15 = c15
c16 = c16
c17 = c17
c18 = c18
c19 = c19
c20 = c20
</constants>
<predicates>
Ax: x is hazel
Bx: x is amber
Cx: x is maple
Dx: x is coral
Ex: x is lime
Fx: x is willow
Gx: x is olive
Hx: x is harbor
Ix: x is teal
Jx: x is granite
Kx: x is laurel
Lx: x is meadow
Mx: x is elm
Nx: x is juniper
Ox: x is slate
Px: x is poppy
Qx: x is violet
Rx: x is orchid
Sx: x is cobalt
Tx: x is cedar
Ux: x is pearl
Vx: x is north
Wx: x is south
Xx: x is east
Yx: x is west
</predicates>
<premises>
C(c0)
X(c0)
X(c0) & C(c0) -> U(c1)
U(c1) -> W(c1)
V(c0) & C(c0) -> G(c1)
G(c1) -> Y(c1)
Y(c0) & C(c0) -> S(c1)
S(c1) -> X(c1)
W(c0) & C(c0) -> M(c1)
M(c1) -> V(c1)
W(c1) & U(c1) -> H(c2)
H(c2) -> W(c2)
V(c1) & M(c1) -> F(c2)
F(c2) -> Y(c2)
Y(c1) & G(c1) -> A(c2)
A(c2) -> V(c2)
X(c1) & S(c1) -> Q(c2)
Q(c2) -> X(c2)
X(c2) & Q(c2) -> U(c3)
U(c3) -> X(c3)
W(c2) & H(c2) -> O(c3)
O(c3) -> W(c3)
V(c2) & A(c2) -> L(c3)
L(c3) -> V(c3)
Y(c2) & F(c2) -> Q(c3)
Q(c3) -> Y(c3)
X(c3) & U(c3) -> O(c4)
O(c4) -> V(c4)
W(c3) & O(c3) -> R(c4)
R(c4) -> W(c4)
V(c3) & L(c3) -> F(c4)
F(c4) -> Y(c4)
Y(c3) & Q(c3) -> E(c4)
E(c4) -> X(c4)
X(c4) & E(c4) -> E(c5)
E(c5) -> X(c5)
V(c4) & O(c4) -> U(c5)
U(c5) -> W(c5)
W(c4) & R(c4) -> T(c5)
T(c5) -> V(c5)
Y(c4) & F(c4) -> S(c5)
S(c5) -> Y(c5)
V(c5) & T(c5) -> F(c6)
F(c6) -> X(c6)
X(c5) & E(c5) -> H(c6)
H(c6) -> V(c6)
Y(c5) & S(c5) -> K(c6)
K(c6) -> W(c6)
W(c5) & U(c5) -> J(c6)
J(c6) -> Y(c6)
X(c6) & F(c6) -> I(c7)
I(c7) -> V(c7)
V(c6) & H(c6) -> R(c7)
R(c7) -> W(c7)
W(c6) & K(c6) -> H(c7)
H(c7) -> X(c7)
Y(c6) & J(c6) -> L(c7)
L(c7) -> Y(c7)
X(c7) & H(c7) -> H(c8)
H(c8) -> V(c8)
V(c7) & I(c7) -> S(c8)
S(c8) -> X(c8)
W(c7) & R(c7) -> P(c8)
P(c8) -> W(c8)
Y(c7) & L(c7) -> G(c8)
G(c8) -> Y(c8)
X(c8) & S(c8) -> A(c9)
A(c9) -> X(c9)
V(c8) & H(c8) -> F(c9)
F(c9) -> Y(c9)
W(c8) & P(c8) -> D(c9)
D(c9) -> W(c9)
Y(c8) & G(c8) -> Q(c9)
Q(c9) -> V(c9)
V(c9) & Q(c9) -> S(c10)
S(c10) -> Y(c10)
Y(c9) & F(c9) -> B(c10)
B(c10) -> W(c10)
W(c9) & D(c9) -> O(c10)
O(c10) -> X(c10)
X(c9) & A(c9) -> L(c10)
L(c10) -> V(c10)
W(c10) & B(c10) -> O(c11)
O(c11) -> W(c11)
Y(c10) & S(c10) -> P(c11)
P(c11) -> X(c11)
X(c10) & O(c10) -> A(c11)
A(c11) -> Y(c11)
V(c10) & L(c10) -> G(c11)
G(c11) -> V(c11)
W(c11) & O(c11) -> O(c12)
O(c12) -> Y(c12)
X(c11) & P(c11) -> K(c12)
K(c12) -> W(c12)
Y(c11) & A(c11) -> B(c12)
B(c12) -> V(c12)
V(c11) & G(c11) -> P(c12)
P(c12) -> X(c12)
X(c12) & P(c12) -> R(c13)
R(c13) -> W(c13)
W(c12) & K(c12) -> J(c13)
J(c13) -> V(c13)
V(c12) & B(c12) -> F(c13)
F(c13) -> Y(c13)
Y(c12) & O(c12) -> N(c13)
N(c13) -> X(c13)
W(c13) & R(c13) -> F(c14)
F(c14) -> V(c14)
X(c13) & N(c13) -> N(c14)
N(c14) -> X(c14)
V(c13) & J(c13) -> U(c14)
U(c14) -> W(c14)
Y(c13) & F(c13) -> L(c14)
L(c14) -> Y(c14)
W(c14) & U(c14) -> P(c15)
P(c15) -> V(c15)
V(c14) & F(c14) -> M(c15)
M(c15) -> W(c15)
Y(c14) & L(c14) -> S(c15)
S(c15) -> Y(c15)
X(c14) & N(c14) -> D(c15)
D(c15) -> X(c15)
X(c15) & D(c15) -> U(c16)
U(c16) -> Y(c16)
V(c15) & P(c15) -> I(c16)
I(c16) -> V(c16)
Y(c15) & S(c15) -> E(c16)
E(c16) -> W(c16)
W(c15) & M(c15) -> K(c16)
K(c16) -> X(c16)
V(c16) & I(c16) -> D(c17)
D(c17) -> W(c17)
W(c16) & E(c16) -> Q(c17)
Q(c17) -> V(c17)
Y(c16) & U(c16) -> I(c17)
I(c17) -> X(c17)
X(c16) & K(c16) -> J(c17)
J(c17) -> Y(c17)
X(c17) & I(c17) -> I(c18)
I(c18) -> X(c18)
V(c17) & Q(c17) -> H(c18)
H(c18) -> V(c18)
W(c17) & D(c17) -> L(c18)
L(c18) -> Y(c18)
Y(c17) & J(c17) -> N(c18)
N(c18) -> W(c18)
V(c18) & H(c18) -> H(c19)
H(c19) -> X(c19)
W(c18) & N(c18) -> F(c19)
F(c19) -> W(c19)
Y(c18) & L(c18) -> U(c19)
U(c19) -> V(c19)
X(c18) & I(c18) -> O(c19)
O(c19) -> Y(c19)
V(c19) & U(c19) -> I(c20)
I(c20) -> Y(c20)
W(c19) & F(c19) -> T(c20)
T(c20) -> W(c20)
X(c19) & H(c19) -> O(c20)
O(c20) -> V(c20)
Y(c19) & O(c19) -> G(c20)
G(c20) -> X(c20)
</premises>
<proof>
C(c0) ; R
X(c0) ; R
U(c1) ; ->E
W(c1) ; ->E
H(c2) ; ->E
W(c2) ; ->E
O(c3) ; ->E
W(c3) ; ->E
R(c4) ; ->E
W(c4) ; ->E
T(c5) ; ->E
V(c5) ; ->E
F(c6) ; ->E
X(c6) ; ->E
I(c7) ; ->E
V(c7) ; ->E
S(c8) ; ->E
X(c8) ; ->E
A(c9) ; ->E
X(c9) ; ->E
L(c10) ; ->E
V(c10) ; ->E
G(c11) ; ->E
V(c11) ; ->E
P(c12) ; ->E
X(c12) ; ->E
R(c13) ; ->E
W(c13) ; ->E
F(c14) ; ->E
V(c14) ; ->E
M(c15) ; ->E
W(c15) ; ->E
K(c16) ; ->E
X(c16) ; ->E
J(c17) ; ->E
Y(c17) ; ->E
N(c18) ; ->E
W(c18) ; ->E
F(c19) ; ->E
W(c19) ; ->E
T(c20) ; ->E
</proof>
<conclusion>
T(c20)
</conclusion>
</formal>
<answer>
cedar
</answer><|endoftext|><question>
1. c0 is poppy.
2. c0 is east.
3. If c0 is south and c0 is poppy, then c1 is ruby.
4. If c1 is ruby, then c1 is east.
5. If c0 is east and c0 is poppy, then c1 is willow.
6. If c1 is willow, then c1 is south.
7. If c0 is north and c0 is poppy, then c1 is ivory.
8. If c1 is ivory, then c1 is north.
9. If c0 is west and c0 is poppy, then c1 is maple.
10. If c1 is maple, then c1 is west.
11. If c1 is south and c1 is willow, then c2 is olive.
12. If c2 is olive, then c2 is south.
13. If c1 is east and c1 is ruby, then c2 is maple.
14. If c2 is maple, then c2 is east.
15. If c1 is west and c1 is maple, then c2 is ivory.
16. If c2 is ivory, then c2 is north.
17. If c1 is north and c1 is ivory, then c2 is willow.
18. If c2 is willow, then c2 is west.
19. If c2 is south and c2 is olive, then c3 is willow.
20. If c3 is willow, then c3 is south.
21. If c2 is north and c2 is ivory, then c3 is ruby.
22. If c3 is ruby, then c3 is north.
23. If c2 is east and c2 is maple, then c3 is ivory.
24. If c3 is ivory, then c3 is west.
25. If c2 is west and c2 is willow, then c3 is maple.
26. If c3 is maple, then c3 is east.
27. If c3 is south and c3 is willow, then c4 is orchid.
28. If c4 is orchid, then c4 is east.
29. If c3 is north and c3 is ruby, then c4 is olive.
30. If c4 is olive, then c4 is west.
31. If c3 is west and c3 is ivory, then c4 is maple.
32. If c4 is maple, then c4 is south.
33. If c3 is east and c3 is maple, then c4 is ruby.
34. If c4 is ruby, then c4 is north.
35. If c4 is north and c4 is ruby, then c5 is orchid.
36. If c5 is orchid, then c5 is north.
37. If c4 is south and c4 is maple, then c5 is maple.
38. If c5 is maple, then c5 is east.
39. If c4 is west and c4 is olive, then c5 is ruby.
40. If c5 is ruby, then c5 is south.
41. If c4 is east and c4 is orchid, then c5 is ivory.
42. If c5 is ivory, then c5 is west.
43. If c5 is east and c5 is maple, then c6 is ruby.
44. If c6 is ruby, then c6 is west.
45. If c5 is west and c5 is ivory, then c6 is maple.
46. If c6 is maple, then c6 is south.
47. If c5 is south and c5 is ruby, then c6 is orchid.
48. If c6 is orchid, then c6 is north.
49. If c5 is north and c5 is orchid, then c6 is willow.
50. If c6 is willow, then c6 is east.
Which state applies to c6?
</question>

<formal>
<constants>
c0 = c0
c1 = c1
c2 = c2
c3 = c3
c4 = c4
c5 = c5
c6 = c6
</constants>
<predicates>
Ax: x is orchid
Bx: x is maple
Cx: x is poppy
Dx: x is willow
Ex: x is olive
Fx: x is ruby
Gx: x is ivory
Hx: x is north
Ix: x is south
Jx: x is east
Kx: x is west
</predicates>
<premises>
C(c0)
J(c0)
I(c0) & C(c0) -> F(c1)
F(c1) -> J(c1)
J(c0) & C(c0) -> D(c1)
D(c1) -> I(c1)
H(c0) & C(c0) -> G(c1)
G(c1) -> H(c1)
K(c0) & C(c0) -> B(c1)
B(c1) -> K(c1)
I(c1) & D(c1) -> E(c2)
E(c2) -> I(c2)
J(c1) & F(c1) -> B(c2)
B(c2) -> J(c2)
K(c1) & B(c1) -> G(c2)
G(c2) -> H(c2)
H(c1) & G(c1) -> D(c2)
D(c2) -> K(c2)
I(c2) & E(c2) -> D(c3)
D(c3) -> I(c3)
H(c2) & G(c2) -> F(c3)
F(c3) -> H(c3)
J(c2) & B(c2) -> G(c3)
G(c3) -> K(c3)
K(c2) & D(c2) -> B(c3)
B(c3) -> J(c3)
I(c3) & D(c3) -> A(c4)
A(c4) -> J(c4)
H(c3) & F(c3) -> E(c4)
E(c4) -> K(c4)
K(c3) & G(c3) -> B(c4)
B(c4) -> I(c4)
J(c3) & B(c3) -> F(c4)
F(c4) -> H(c4)
H(c4) & F(c4) -> A(c5)
A(c5) -> H(c5)
I(c4) & B(c4) -> B(c5)
B(c5) -> J(c5)
K(c4) & E(c4) -> F(c5)
F(c5) -> I(c5)
J(c4) & A(c4) -> G(c5)
G(c5) -> K(c5)
J(c5) & B(c5) -> F(c6)
F(c6) -> K(c6)
K(c5) & G(c5) -> B(c6)
B(c6) -> I(c6)
I(c5) & F(c5) -> A(c6)
A(c6) -> H(c6)
H(c5) & A(c5) -> D(c6)
D(c6) -> J(c6)
</premises>
<proof>
C(c0) ; R
J(c0) ; R
D(c1) ; ->E
I(c1) ; ->E
E(c2) ; ->E
I(c2) ; ->E
D(c3) ; ->E
I(c3) ; ->E
A(c4) ; ->E
J(c4) ; ->E
G(c5) ; ->E
K(c5) ; ->E
B(c6) ; ->E
</proof>
<conclusion>
B(c6)
</conclusion>
</formal>
<answer>
maple
</answer><|endoftext|>
```

## Window 3 (2 documents, 494 pad tokens)
```
<question>
1. c0 is coral.
2. c0 is south.
3. If c0 is south and c0 is coral, then c1 is meadow.
4. If c1 is meadow, then c1 is west.
5. If c0 is east and c0 is coral, then c1 is hazel.
6. If c1 is hazel, then c1 is east.
7. If c0 is west and c0 is coral, then c1 is laurel.
8. If c1 is laurel, then c1 is south.
9. If c0 is north and c0 is coral, then c1 is ivory.
10. If c1 is ivory, then c1 is north.
11. If c1 is east and c1 is hazel, then c2 is ruby.
12. If c2 is ruby, then c2 is south.
13. If c1 is south and c1 is laurel, then c2 is poppy.
14. If c2 is poppy, then c2 is west.
15. If c1 is west and c1 is meadow, then c2 is laurel.
16. If c2 is laurel, then c2 is north.
17. If c1 is north and c1 is ivory, then c2 is willow.
18. If c2 is willow, then c2 is east.
19. If c2 is north and c2 is laurel, then c3 is willow.
20. If c3 is willow, then c3 is east.
21. If c2 is south and c2 is ruby, then c3 is meadow.
22. If c3 is meadow, then c3 is west.
23. If c2 is west and c2 is poppy, then c3 is amber.
24. If c3 is amber, then c3 is south.
25. If c2 is east and c2 is willow, then c3 is slate.
26. If c3 is slate, then c3 is north.
27. If c3 is north and c3 is slate, then c4 is granite.
28. If c4 is granite, then c4 is west.
29. If c3 is west and c3 is meadow, then c4 is cedar.
30. If c4 is cedar, then c4 is east.
31. If c3 is south and c3 is amber, then c4 is meadow.
32. If c4 is meadow, then c4 is north.
33. If c3 is east and c3 is willow, then c4 is maple.
34. If c4 is maple, then c4 is south.
35. If c4 is south and c4 is maple, then c5 is juniper.
36. If c5 is juniper, then c5 is east.
37. If c4 is west and c4 is granite, then c5 is ivory.
38. If c5 is ivory, then c5 is north.
39. If c4 is north and c4 is meadow, then c5 is lime.
40. If c5 is lime, then c5 is west.
41. If c4 is east and c4 is cedar, then c5 is meadow.
42. If c5 is meadow, then c5 is south.
43. If c5 is west and c5 is lime, then c6 is birch.
44. If c6 is birch, then c6 is north.
45. If c5 is north and c5 is ivory, then c6 is granite.
46. If c6 is granite, then c6 is east.
47. If c5 is east and c5 is juniper, then c6 is laurel.
48. If c6 is laurel, then c6 is west.
49. If c5 is south and c5 is meadow, then c6 is ivory.
50. If c6 is ivory, then c6 is south.
51. If c6 is south and c6 is ivory, then c7 is slate.
52. If c7 is slate, then c7 is south.
53. If c6 is east and c6 is granite, then c7 is hazel.
54. If c7 is hazel, then c7 is north.
55. If c6 is west and c6 is laurel, then c7 is lime.
56. If c7 is lime, then c7 is west.
57. If c6 is north and c6 is birch, then c7 is cedar.
58. If c7 is cedar, then c7 is east.
59. If c7 is north and c7 is hazel, then c8 is cobalt.
60. If c8 is cobalt, then c8 is north.
61. If c7 is south and c7 is slate, then c8 is birch.
62. If c8 is birch, then c8 is west.
63. If c7 is east and c7 is cedar, then c8 is juniper.
64. If c8 is juniper, then c8 is east.
65. If c7 is west and c7 is lime, then c8 is harbor.
66. If c8 is harbor, then c8 is south.
67. If c8 is west and c8 is birch, then c9 is laurel.
68. If c9 is laurel, then c9 is south.
69. If c8 is south and c8 is harbor, then c9 is ivory.
70. If c9 is ivory, then c9 is east.
71. If c8 is east and c8 is juniper, then c9 is amber.
72. If c9 is amber, then c9 is west.
73. If c8 is north and c8 is cobalt, then c9 is elm.
74. If c9 is elm, then c9 is north.
75. If c9 is south and c9 is laurel, then c10 is meadow.
76. If c10 is meadow, then c10 is west.
77. If c9 is north and c9 is elm, then c10 is willow.
78. If c10 is willow, then c10 is south.
79. If c9 is west and c9 is amber, then c10 is amber.
80. If c10 is amber, then c10 is east.
81. If c9 is east and c9 is ivory, then c10 is elm.
82. If c10 is elm, then c10 is north.
83. If c10 is east and c10 is amber, then c11 is maple.
84. If c11 is maple, then c11 is east.
85. If c10 is south and c10 is willow, then c11 is poppy.
86. If c11 is poppy, then c11 is north.
87. If c10 is west and c10 is meadow, then c11 is cedar.
88. If c11 is cedar, then c11 is west.
89. If c10 is north and c10 is elm, then c11 is meadow.
90. If c11 is meadow, then c11 is south.
91. If c11 is north and c11 is poppy, then c12 is elm.
92. If c12 is elm, then c12 is south.
93. If c11 is west and c11 is cedar, then c12 is amber.
94. If c12 is amber, then c12 is east.
95. If c11 is east and c11 is maple, then c12 is granite.
96. If c12 is granite, then c12 is west.
97. If c11 is south and c11 is meadow, then c12 is ruby.
98. If c12 is ruby, then c12 is north.
99. If c12 is north and c12 is ruby, then c13 is ivory.
100. If c13 is ivory, then c13 is north.
101. If c12 is south and c12 is elm, then c13 is cedar.
102. If c13 is cedar, then c13 is west.
103. If c12 is east and c12 is amber, then c13 is harbor.
104. If c13 is harbor, then c13 is south.
105. If c12 is west and c12 is granite, then c13 is cobalt.
106. If c13 is cobalt, then c13 is east.
107. If c13 is east and c13 is cobalt, then c14 is lime.
108. If c14 is lime, then c14 is south.
109. If c13 is north and c13 is ivory, then c14 is slate.
110. If c14 is slate, then c14 is west.
111. If c13 is south and c13 is harbor, then c14 is harbor.
112. If c14 is harbor, then c14 is north.
113. If c13 is west and c13 is cedar, then c14 is elm.
114. If c14 is elm, then c14 is east.
115. If c14 is north and c14 is harbor, then c15 is lime.
116. If c15 is lime, then c15 is east.
117. If c14 is south and c14 is lime, then c15 is meadow.
118. If c15 is meadow, then c15 is south.
119. If c14 is west and c14 is slate, then c15 is ivory.
120. If c15 is ivory, then c15 is north.
121. If c14 is east and c14 is elm, then c15 is elm.
122. If c15 is elm, then c15 is west.
123. If c15 is west and c15 is elm, then c16 is poppy.
124. If c16 is poppy, then c16 is west.
125. If c15 is north and c15 is ivory, then c16 is birch.
126. If c16 is birch, then c16 is south.
127. If c15 is south and c15 is meadow, then c16 is laurel.
128. If c16 is laurel, then c16 is north.
129. If c15 is east and c15 is lime, then c16 is harbor.
130. If c16 is harbor, then c16 is east.
131. If c16 is east and c16 is harbor, then c17 is granite.
132. If c17 is granite, then c17 is east.
133. If c16 is west and c16 is poppy, then c17 is ruby.
134. If c17 is ruby, then c17 is west.
135. If c16 is south and c16 is birch, then c17 is hazel.
136. If c17 is hazel, then c17 is north.
137. If c16 is north and c16 is laurel, then c17 is juniper.
138. If c17 is juniper, then c17 is south.
139. If c17 is north and c17 is hazel, then c18 is elm.
140. If c18 is elm, then c18 is east.
141. If c17 is east and c17 is granite, then c18 is harbor.
142. If c18 is harbor, then c18 is north.
143. If c17 is west and c17 is ruby, then c18 is maple.
144. If c18 is maple, then c18 is west.
145. If c17 is south and c17 is juniper, then c18 is willow.
146. If c18 is willow, then c18 is south.
147. If c18 is east and c18 is elm, then c19 is teal.
148. If c19 is teal, then c19 is west.
149. If c18 is west and c18 is maple, then c19 is slate.
150. If c19 is slate, then c19 is north.
151. If c18 is north and c18 is harbor, then c19 is hazel.
152. If c19 is hazel, then c19 is south.
153. If c18 is south and c18 is willow, then c19 is laurel.
154. If c19 is laurel, then c19 is east.
155. If c19 is east and c19 is laurel, then c20 is lime.
156. If c20 is lime, then c20 is south.
157. If c19 is south and c19 is hazel, then c20 is olive.
158. If c20 is olive, then c20 is west.
159. If c19 is west and c19 is teal, then c20 is amber.
160. If c20 is amber, then c20 is north.
161. If c19 is north and c19 is slate, then c20 is laurel.
162. If c20 is laurel, then c20 is east.
Which state applies to c20?
</question>

<formal>
<constants>
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
c15 = c15
c16 = c16
c17 = c17
c18 = c18
c19 = c19
c20 = c20
</constants>
<predicates>
Ax: x is amber
Bx: x is birch
Cx: x is lime
Dx: x is cobalt
Ex: x is teal
Fx: x is willow
Gx: x is poppy
Hx: x is laurel
Ix: x is harbor
Jx: x is hazel
Kx: x is ruby
Lx: x is olive
Mx: x is granite
Nx: x is elm
Ox: x is maple
Px: x is slate
Qx: x is cedar
Rx: x is ivory
Sx: x is coral
Tx: x is juniper
Ux: x is meadow
Vx: x is north
Wx: x is south
Xx: x is east
Yx: x is west
</predicates>
<premises>
S(c0)
W(c0)
W(c0) & S(c0) -> U(c1)
U(c1) -> Y(c1)
X(c0) & S(c0) -> J(c1)
J(c1) -> X(c1)
Y(c0) & S(c0) -> H(c1)
H(c1) -> W(c1)
V(c0) & S(c0) -> R(c1)
R(c1) -> V(c1)
X(c1) & J(c1) -> K(c2)
K(c2) -> W(c2)
W(c1) & H(c1) -> G(c2)
G(c2) -> Y(c2)
Y(c1) & U(c1) -> H(c2)
H(c2) -> V(c2)
V(c1) & R(c1) -> F(c2)
F(c2) -> X(c2)
V(c2) & H(c2) -> F(c3)
F(c3) -> X(c3)
W(c2) & K(c2) -> U(c3)
U(c3) -> Y(c3)
Y(c2) & G(c2) -> A(c3)
A(c3) -> W(c3)
X(c2) & F(c2) -> P(c3)
P(c3) -> V(c3)
V(c3) & P(c3) -> M(c4)
M(c4) -> Y(c4)
Y(c3) & U(c3) -> Q(c4)
Q(c4) -> X(c4)
W(c3) & A(c3) -> U(c4)
U(c4) -> V(c4)
X(c3) & F(c3) -> O(c4)
O(c4) -> W(c4)
W(c4) & O(c4) -> T(c5)
T(c5) -> X(c5)
Y(c4) & M(c4) -> R(c5)
R(c5) -> V(c5)
V(c4) & U(c4) -> C(c5)
C(c5) -> Y(c5)
X(c4) & Q(c4) -> U(c5)
U(c5) -> W(c5)
Y(c5) & C(c5) -> B(c6)
B(c6) -> V(c6)
V(c5) & R(c5) -> M(c6)
M(c6) -> X(c6)
X(c5) & T(c5) -> H(c6)
H(c6) -> Y(c6)
W(c5) & U(c5) -> R(c6)
R(c6) -> W(c6)
W(c6) & R(c6) -> P(c7)
P(c7) -> W(c7)
X(c6) & M(c6) -> J(c7)
J(c7) -> V(c7)
Y(c6) & H(c6) -> C(c7)
C(c7) -> Y(c7)
V(c6) & B(c6) -> Q(c7)
Q(c7) -> X(c7)
V(c7) & J(c7) -> D(c8)
D(c8) -> V(c8)
W(c7) & P(c7) -> B(c8)
B(c8) -> Y(c8)
X(c7) & Q(c7) -> T(c8)
T(c8) -> X(c8)
Y(c7) & C(c7) -> I(c8)
I(c8) -> W(c8)
Y(c8) & B(c8) -> H(c9)
H(c9) -> W(c9)
W(c8) & I(c8) -> R(c9)
R(c9) -> X(c9)
X(c8) & T(c8) -> A(c9)
A(c9) -> Y(c9)
V(c8) & D(c8) -> N(c9)
N(c9) -> V(c9)
W(c9) & H(c9) -> U(c10)
U(c10) -> Y(c10)
V(c9) & N(c9) -> F(c10)
F(c10) -> W(c10)
Y(c9) & A(c9) -> A(c10)
A(c10) -> X(c10)
X(c9) & R(c9) -> N(c10)
N(c10) -> V(c10)
X(c10) & A(c10) -> O(c11)
O(c11) -> X(c11)
W(c10) & F(c10) -> G(c11)
G(c11) -> V(c11)
Y(c10) & U(c10) -> Q(c11)
Q(c11) -> Y(c11)
V(c10) & N(c10) -> U(c11)
U(c11) -> W(c11)
V(c11) & G(c11) -> N(c12)
N(c12) -> W(c12)
Y(c11) & Q(c11) -> A(c12)
A(c12) -> X(c12)
X(c11) & O(c11) -> M(c12)
M(c12) -> Y(c12)
W(c11) & U(c11) -> K(c12)
K(c12) -> V(c12)
V(c12) & K(c12) -> R(c13)
R(c13) -> V(c13)
W(c12) & N(c12) -> Q(c13)
Q(c13) -> Y(c13)
X(c12) & A(c12) -> I(c13)
I(c13) -> W(c13)
Y(c12) & M(c12) -> D(c13)
D(c13) -> X(c13)
X(c13) & D(c13) -> C(c14)
C(c14) -> W(c14)
V(c13) & R(c13) -> P(c14)
P(c14) -> Y(c14)
W(c13) & I(c13) -> I(c14)
I(c14) -> V(c14)
Y(c13) & Q(c13) -> N(c14)
N(c14) -> X(c14)
V(c14) & I(c14) -> C(c15)
C(c15) -> X(c15)
W(c14) & C(c14) -> U(c15)
U(c15) -> W(c15)
Y(c14) & P(c14) -> R(c15)
R(c15) -> V(c15)
X(c14) & N(c14) -> N(c15)
N(c15) -> Y(c15)
Y(c15) & N(c15) -> G(c16)
G(c16) -> Y(c16)
V(c15) & R(c15) -> B(c16)
B(c16) -> W(c16)
W(c15) & U(c15) -> H(c16)
H(c16) -> V(c16)
X(c15) & C(c15) -> I(c16)
I(c16) -> X(c16)
X(c16) & I(c16) -> M(c17)
M(c17) -> X(c17)
Y(c16) & G(c16) -> K(c17)
K(c17) -> Y(c17)
W(c16) & B(c16) -> J(c17)
J(c17) -> V(c17)
V(c16) & H(c16) -> T(c17)
T(c17) -> W(c17)
V(c17) & J(c17) -> N(c18)
N(c18) -> X(c18)
X(c17) & M(c17) -> I(c18)
I(c18) -> V(c18)
Y(c17) & K(c17) -> O(c18)
O(c18) -> Y(c18)
W(c17) & T(c17) -> F(c18)
F(c18) -> W(c18)
X(c18) & N(c18) -> E(c19)
E(c19) -> Y(c19)
Y(c18) & O(c18) -> P(c19)
P(c19) -> V(c19)
V(c18) & I(c18) -> J(c19)
J(c19) -> W(c19)
W(c18) & F(c18) -> H(c19)
H(c19) -> X(c19)
X(c19) & H(c19) -> C(c20)
C(c20) -> W(c20)
W(c19) & J(c19) -> L(c20)
L(c20) -> Y(c20)
Y(c19) & E(c19) -> A(c20)
A(c20) -> V(c20)
V(c19) & P(c19) -> H(c20)
H(c20) -> X(c20)
</premises>
<proof>
S(c0) ; R
W(c0) ; R
U(c1) ; ->E
Y(c1) ; ->E
H(c2) ; ->E
V(c2) ; ->E
F(c3) ; ->E
X(c3) ; ->E
O(c4) ; ->E
W(c4) ; ->E
T(c5) ; ->E
X(c5) ; ->E
H(c6) ; ->E
Y(c6) ; ->E
C(c7) ; ->E
Y(c7) ; ->E
I(c8) ; ->E
W(c8) ; ->E
R(c9) ; ->E
X(c9) ; ->E
N(c10) ; ->E
V(c10) ; ->E
U(c11) ; ->E
W(c11) ; ->E
K(c12) ; ->E
V(c12) ; ->E
R(c13) ; ->E
V(c13) ; ->E
P(c14) ; ->E
Y(c14) ; ->E
R(c15) ; ->E
V(c15) ; ->E
B(c16) ; ->E
W(c16) ; ->E
J(c17) ; ->E
V(c17) ; ->E
N(c18) ; ->E
X(c18) ; ->E
E(c19) ; ->E
Y(c19) ; ->E
A(c20) ; ->E
</proof>
<conclusion>
A(c20)
</conclusion>
</formal>
<answer>
amber
</answer><|endoftext|><question>
1. c0 is maple.
2. c0 is west.
3. If c0 is east and c0 is maple, then c1 is slate.
4. If c1 is slate, then c1 is west.
5. If c0 is west and c0 is maple, then c1 is juniper.
6. If c1 is juniper, then c1 is south.
7. If c0 is south and c0 is maple, then c1 is poppy.
8. If c1 is poppy, then c1 is north.
9. If c0 is north and c0 is maple, then c1 is pearl.
10. If c1 is pearl, then c1 is east.
11. If c1 is north and c1 is poppy, then c2 is juniper.
12. If c2 is juniper, then c2 is west.
13. If c1 is east and c1 is pearl, then c2 is pearl.
14. If c2 is pearl, then c2 is south.
15. If c1 is south and c1 is juniper, then c2 is ruby.
16. If c2 is ruby, then c2 is east.
17. If c1 is west and c1 is slate, then c2 is slate.
18. If c2 is slate, then c2 is north.
19. If c2 is south and c2 is pearl, then c3 is slate.
20. If c3 is slate, then c3 is south.
21. If c2 is north and c2 is slate, then c3 is ruby.
22. If c3 is ruby, then c3 is west.
23. If c2 is west and c2 is juniper, then c3 is pearl.
24. If c3 is pearl, then c3 is east.
25. If c2 is east and c2 is ruby, then c3 is juniper.
26. If c3 is juniper, then c3 is north.
27. If c3 is west and c3 is ruby, then c4 is pearl.
28. If c4 is pearl, then c4 is south.
29. If c3 is south and c3 is slate, then c4 is slate.
30. If c4 is slate, then c4 is east.
31. If c3 is east and c3 is pearl, then c4 is poppy.
32. If c4 is poppy, then c4 is north.
33. If c3 is north and c3 is juniper, then c4 is violet.
34. If c4 is violet, then c4 is west.
35. If c4 is north and c4 is poppy, then c5 is pearl.
36. If c5 is pearl, then c5 is west.
37. If c4 is east and c4 is slate, then c5 is juniper.
38. If c5 is juniper, then c5 is south.
39. If c4 is south and c4 is pearl, then c5 is slate.
40. If c5 is slate, then c5 is east.
41. If c4 is west and c4 is violet, then c5 is violet.
42. If c5 is violet, then c5 is north.
43. If c5 is east and c5 is slate, then c6 is ruby.
44. If c6 is ruby, then c6 is north.
45. If c5 is south and c5 is juniper, then c6 is juniper.
46. If c6 is juniper, then c6 is east.
47. If c5 is north and c5 is violet, then c6 is pearl.
48. If c6 is pearl, then c6 is south.
49. If c5 is west and c5 is pearl, then c6 is poppy.
50. If c6 is poppy, then c6 is west.
Which state applies to c6?
</question>

<formal>
<constants>
c0 = c0
c1 = c1
c2 = c2
c3 = c3
c4 = c4
c5 = c5
c6 = c6
</constants>
<predicates>
Ax: x is juniper
Bx: x is violet
Cx: x is poppy
Dx: x is maple
Ex: x is pearl
Fx: x is slate
Gx: x is ruby
Hx: x is north
Ix: x is south
Jx: x is east
Kx: x is west
</predicates>
<premises>
D(c0)
K(c0)
J(c0) & D(c0) -> F(c1)
F(c1) -> K(c1)
K(c0) & D(c0) -> A(c1)
A(c1) -> I(c1)
I(c0) & D(c0) -> C(c1)
C(c1) -> H(c1)
H(c0) & D(c0) -> E(c1)
E(c1) -> J(c1)
H(c1) & C(c1) -> A(c2)
A(c2) -> K(c2)
J(c1) & E(c1) -> E(c2)
E(c2) -> I(c2)
I(c1) & A(c1) -> G(c2)
G(c2) -> J(c2)
K(c1) & F(c1) -> F(c2)
F(c2) -> H(c2)
I(c2) & E(c2) -> F(c3)
F(c3) -> I(c3)
H(c2) & F(c2) -> G(c3)
G(c3) -> K(c3)
K(c2) & A(c2) -> E(c3)
E(c3) -> J(c3)
J(c2) & G(c2) -> A(c3)
A(c3) -> H(c3)
K(c3) & G(c3) -> E(c4)
E(c4) -> I(c4)
I(c3) & F(c3) -> F(c4)
F(c4) -> J(c4)
J(c3) & E(c3) -> C(c4)
C(c4) -> H(c4)
H(c3) & A(c3) -> B(c4)
B(c4) -> K(c4)
H(c4) & C(c4) -> E(c5)
E(c5) -> K(c5)
J(c4) & F(c4) -> A(c5)
A(c5) -> I(c5)
I(c4) & E(c4) -> F(c5)
F(c5) -> J(c5)
K(c4) & B(c4) -> B(c5)
B(c5) -> H(c5)
J(c5) & F(c5) -> G(c6)
G(c6) -> H(c6)
I(c5) & A(c5) -> A(c6)
A(c6) -> J(c6)
H(c5) & B(c5) -> E(c6)
E(c6) -> I(c6)
K(c5) & E(c5) -> C(c6)
C(c6) -> K(c6)
</premises>
<proof>
D(c0) ; R
K(c0) ; R
A(c1) ; ->E
I(c1) ; ->E
G(c2) ; ->E
J(c2) ; ->E
A(c3) ; ->E
H(c3) ; ->E
B(c4) ; ->E
K(c4) ; ->E
B(c5) ; ->E
H(c5) ; ->E
E(c6) ; ->E
</proof>
<conclusion>
E(c6)
</conclusion>
</formal>
<answer>
pearl
</answer><|endoftext|>
```

## Window 2558 (2 documents, 336 pad tokens)
```
<question>
1. c0 is birch.
2. c0 is north.
3. If c0 is east and c0 is birch, then c1 is harbor.
4. If c1 is harbor, then c1 is north.
5. If c0 is south and c0 is birch, then c1 is slate.
6. If c1 is slate, then c1 is west.
7. If c0 is west and c0 is birch, then c1 is teal.
8. If c1 is teal, then c1 is east.
9. If c0 is north and c0 is birch, then c1 is coral.
10. If c1 is coral, then c1 is south.
11. If c1 is south and c1 is coral, then c2 is hazel.
12. If c2 is hazel, then c2 is west.
13. If c1 is north and c1 is harbor, then c2 is granite.
14. If c2 is granite, then c2 is east.
15. If c1 is west and c1 is slate, then c2 is cedar.
16. If c2 is cedar, then c2 is south.
17. If c1 is east and c1 is teal, then c2 is maple.
18. If c2 is maple, then c2 is north.
19. If c2 is east and c2 is granite, then c3 is hazel.
20. If c3 is hazel, then c3 is north.
21. If c2 is south and c2 is cedar, then c3 is ivory.
22. If c3 is ivory, then c3 is west.
23. If c2 is west and c2 is hazel, then c3 is lime.
24. If c3 is lime, then c3 is south.
25. If c2 is north and c2 is maple, then c3 is meadow.
26. If c3 is meadow, then c3 is east.
27. If c3 is south and c3 is lime, then c4 is maple.
28. If c4 is maple, then c4 is west.
29. If c3 is north and c3 is hazel, then c4 is cobalt.
30. If c4 is cobalt, then c4 is north.
31. If c3 is east and c3 is meadow, then c4 is juniper.
32. If c4 is juniper, then c4 is south.
33. If c3 is west and c3 is ivory, then c4 is poppy.
34. If c4 is poppy, then c4 is east.
35. If c4 is north and c4 is cobalt, then c5 is hazel.
36. If c5 is hazel, then c5 is west.
37. If c4 is west and c4 is maple, then c5 is coral.
38. If c5 is coral, then c5 is north.
39. If c4 is south and c4 is juniper, then c5 is maple.
40. If c5 is maple, then c5 is east.
41. If c4 is east and c4 is poppy, then c5 is lime.
42. If c5 is lime, then c5 is south.
43. If c5 is south and c5 is lime, then c6 is teal.
44. If c6 is teal, then c6 is south.
45. If c5 is west and c5 is hazel, then c6 is meadow.
46. If c6 is meadow, then c6 is north.
47. If c5 is north and c5 is coral, then c6 is willow.
48. If c6 is willow, then c6 is west.
49. If c5 is east and c5 is maple, then c6 is harbor.
50. If c6 is harbor, then c6 is east.
51. If c6 is east and c6 is harbor, then c7 is willow.
52. If c7 is willow, then c7 is east.
53. If c6 is south and c6 is teal, then c7 is violet.
54. If c7 is violet, then c7 is north.
55. If c6 is north and c6 is meadow, then c7 is cobalt.
56. If c7 is cobalt, then c7 is west.
57. If c6 is west and c6 is willow, then c7 is teal.
58. If c7 is teal, then c7 is south.
59. If c7 is south and c7 is teal, then c8 is meadow.
60. If c8 is meadow, then c8 is west.
61. If c7 is east and c7 is willow, then c8 is willow.
62. If c8 is willow, then c8 is south.
63. If c7 is west and c7 is cobalt, then c8 is orchid.
64. If c8 is orchid, then c8 is north.
65. If c7 is north and c7 is violet, then c8 is slate.
66. If c8 is slate, then c8 is east.
67. If c8 is north and c8 is orchid, then c9 is hazel.
68. If c9 is hazel, then c9 is north.
69. If c8 is west and c8 is meadow, then c9 is willow.
70. If c9 is willow, then c9 is south.
71. If c8 is south and c8 is willow, then c9 is teal.
72. If c9 is teal, then c9 is east.
73. If c8 is east and c8 is slate, then c9 is orchid.
74. If c9 is orchid, then c9 is west.
75. If c9 is south and c9 is willow, then c10 is juniper.
76. If c10 is juniper, then c10 is south.
77. If c9 is west and c9 is orchid, then c10 is harbor.
78. If c10 is harbor, then c10 is west.
79. If c9 is east and c9 is teal, then c10 is violet.
80. If c10 is violet, then c10 is north.
81. If c9 is north and c9 is hazel, then c10 is lime.
82. If c10 is lime, then c10 is east.
83. If c10 is south and c10 is juniper, then c11 is orchid.
84. If c11 is orchid, then c11 is east.
85. If c10 is north and c10 is violet, then c11 is coral.
86. If c11 is coral, then c11 is north.
87. If c10 is west and c10 is harbor, then c11 is willow.
88. If c11 is willow, then c11 is west.
89. If c10 is east and c10 is lime, then c11 is violet.
90. If c11 is violet, then c11 is south.
91. If c11 is north and c11 is coral, then c12 is maple.
92. If c12 is maple, then c12 is east.
93. If c11 is south and c11 is violet, then c12 is cedar.
94. If c12 is cedar, then c12 is south.
95. If c11 is west and c11 is willow, then c12 is violet.
96. If c12 is violet, then c12 is north.
97. If c11 is east and c11 is orchid, then c12 is juniper.
98. If c12 is juniper, then c12 is west.
99. If c12 is east and c12 is maple, then c13 is cobalt.
100. If c13 is cobalt, then c13 is north.
101. If c12 is west and c12 is juniper, then c13 is maple.
102. If c13 is maple, then c13 is west.
103. If c12 is north and c12 is violet, then c13 is teal.
104. If c13 is teal, then c13 is east.
105. If c12 is south and c12 is cedar, then c13 is orchid.
106. If c13 is orchid, then c13 is south.
107. If c13 is west and c13 is maple, then c14 is coral.
108. If c14 is coral, then c14 is east.
109. If c13 is east and c13 is teal, then c14 is lime.
110. If c14 is lime, then c14 is south.
111. If c13 is south and c13 is orchid, then c14 is meadow.
112. If c14 is meadow, then c14 is west.
113. If c13 is north and c13 is cobalt, then c14 is cedar.
114. If c14 is cedar, then c14 is north.
115. If c14 is north and c14 is cedar, then c15 is maple.
116. If c15 is maple, then c15 is west.
117. If c14 is east and c14 is coral, then c15 is cobalt.
118. If c15 is cobalt, then c15 is south.
119. If c14 is west and c14 is meadow, then c15 is poppy.
120. If c15 is poppy, then c15 is east.
121. If c14 is south and c14 is lime, then c15 is willow.
122. If c15 is willow, then c15 is north.
123. If c15 is east and c15 is poppy, then c16 is juniper.
124. If c16 is juniper, then c16 is south.
125. If c15 is west and c15 is maple, then c16 is lime.
126. If c16 is lime, then c16 is west.
127. If c15 is north and c15 is willow, then c16 is slate.
128. If c16 is slate, then c16 is east.
129. If c15 is south and c15 is cobalt, then c16 is harbor.
130. If c16 is harbor, then c16 is north.
131. If c16 is south and c16 is juniper, then c17 is lime.
132. If c17 is lime, then c17 is north.
133. If c16 is west and c16 is lime, then c17 is slate.
134. If c17 is slate, then c17 is east.
135. If c16 is east and c16 is slate, then c17 is ivory.
136. If c17 is ivory, then c17 is south.
137. If c16 is north and c16 is harbor, then c17 is ruby.
138. If c17 is ruby, then c17 is west.
139. If c17 is north and c17 is lime, then c18 is harbor.
140. If c18 is harbor, then c18 is west.
141. If c17 is west and c17 is ruby, then c18 is coral.
142. If c18 is coral, then c18 is east.
143. If c17 is east and c17 is slate, then c18 is lime.
144. If c18 is lime, then c18 is south.
145. If c17 is south and c17 is ivory, then c18 is teal.
146. If c18 is teal, then c18 is north.
Which state applies to c18?
</question>

<formal>
<constants>
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
c15 = c15
c16 = c16
c17 = c17
c18 = c18
</constants>
<predicates>
Ax: x is granite
Bx: x is ivory
Cx: x is juniper
Dx: x is coral
Ex: x is cobalt
Fx: x is harbor
Gx: x is slate
Hx: x is hazel
Ix: x is birch
Jx: x is poppy
Kx: x is orchid
Lx: x is cedar
Mx: x is willow
Nx: x is meadow
Ox: x is maple
Px: x is ruby
Qx: x is lime
Rx: x is teal
Sx: x is violet
Tx: x is north
Ux: x is south
Vx: x is east
Wx: x is west
</predicates>
<premises>
I(c0)
T(c0)
V(c0) & I(c0) -> F(c1)
F(c1) -> T(c1)
U(c0) & I(c0) -> G(c1)
G(c1) -> W(c1)
W(c0) & I(c0) -> R(c1)
R(c1) -> V(c1)
T(c0) & I(c0) -> D(c1)
D(c1) -> U(c1)
U(c1) & D(c1) -> H(c2)
H(c2) -> W(c2)
T(c1) & F(c1) -> A(c2)
A(c2) -> V(c2)
W(c1) & G(c1) -> L(c2)
L(c2) -> U(c2)
V(c1) & R(c1) -> O(c2)
O(c2) -> T(c2)
V(c2) & A(c2) -> H(c3)
H(c3) -> T(c3)
U(c2) & L(c2) -> B(c3)
B(c3) -> W(c3)
W(c2) & H(c2) -> Q(c3)
Q(c3) -> U(c3)
T(c2) & O(c2) -> N(c3)
N(c3) -> V(c3)
U(c3) & Q(c3) -> O(c4)
O(c4) -> W(c4)
T(c3) & H(c3) -> E(c4)
E(c4) -> T(c4)
V(c3) & N(c3) -> C(c4)
C(c4) -> U(c4)
W(c3) & B(c3) -> J(c4)
J(c4) -> V(c4)
T(c4) & E(c4) -> H(c5)
H(c5) -> W(c5)
W(c4) & O(c4) -> D(c5)
D(c5) -> T(c5)
U(c4) & C(c4) -> O(c5)
O(c5) -> V(c5)
V(c4) & J(c4) -> Q(c5)
Q(c5) -> U(c5)
U(c5) & Q(c5) -> R(c6)
R(c6) -> U(c6)
W(c5) & H(c5) -> N(c6)
N(c6) -> T(c6)
T(c5) & D(c5) -> M(c6)
M(c6) -> W(c6)
V(c5) & O(c5) -> F(c6)
F(c6) -> V(c6)
V(c6) & F(c6) -> M(c7)
M(c7) -> V(c7)
U(c6) & R(c6) -> S(c7)
S(c7) -> T(c7)
T(c6) & N(c6) -> E(c7)
E(c7) -> W(c7)
W(c6) & M(c6) -> R(c7)
R(c7) -> U(c7)
U(c7) & R(c7) -> N(c8)
N(c8) -> W(c8)
V(c7) & M(c7) -> M(c8)
M(c8) -> U(c8)
W(c7) & E(c7) -> K(c8)
K(c8) -> T(c8)
T(c7) & S(c7) -> G(c8)
G(c8) -> V(c8)
T(c8) & K(c8) -> H(c9)
H(c9) -> T(c9)
W(c8) & N(c8) -> M(c9)
M(c9) -> U(c9)
U(c8) & M(c8) -> R(c9)
R(c9) -> V(c9)
V(c8) & G(c8) -> K(c9)
K(c9) -> W(c9)
U(c9) & M(c9) -> C(c10)
C(c10) -> U(c10)
W(c9) & K(c9) -> F(c10)
F(c10) -> W(c10)
V(c9) & R(c9) -> S(c10)
S(c10) -> T(c10)
T(c9) & H(c9) -> Q(c10)
Q(c10) -> V(c10)
U(c10) & C(c10) -> K(c11)
K(c11) -> V(c11)
T(c10) & S(c10) -> D(c11)
D(c11) -> T(c11)
W(c10) & F(c10) -> M(c11)
M(c11) -> W(c11)
V(c10) & Q(c10) -> S(c11)
S(c11) -> U(c11)
T(c11) & D(c11) -> O(c12)
O(c12) -> V(c12)
U(c11) & S(c11) -> L(c12)
L(c12) -> U(c12)
W(c11) & M(c11) -> S(c12)
S(c12) -> T(c12)
V(c11) & K(c11) -> C(c12)
C(c12) -> W(c12)
V(c12) & O(c12) -> E(c13)
E(c13) -> T(c13)
W(c12) & C(c12) -> O(c13)
O(c13) -> W(c13)
T(c12) & S(c12) -> R(c13)
R(c13) -> V(c13)
U(c12) & L(c12) -> K(c13)
K(c13) -> U(c13)
W(c13) & O(c13) -> D(c14)
D(c14) -> V(c14)
V(c13) & R(c13) -> Q(c14)
Q(c14) -> U(c14)
U(c13) & K(c13) -> N(c14)
N(c14) -> W(c14)
T(c13) & E(c13) -> L(c14)
L(c14) -> T(c14)
T(c14) & L(c14) -> O(c15)
O(c15) -> W(c15)
V(c14) & D(c14) -> E(c15)
E(c15) -> U(c15)
W(c14) & N(c14) -> J(c15)
J(c15) -> V(c15)
U(c14) & Q(c14) -> M(c15)
M(c15) -> T(c15)
V(c15) & J(c15) -> C(c16)
C(c16) -> U(c16)
W(c15) & O(c15) -> Q(c16)
Q(c16) -> W(c16)
T(c15) & M(c15) -> G(c16)
G(c16) -> V(c16)
U(c15) & E(c15) -> F(c16)
F(c16) -> T(c16)
U(c16) & C(c16) -> Q(c17)
Q(c17) -> T(c17)
W(c16) & Q(c16) -> G(c17)
G(c17) -> V(c17)
V(c16) & G(c16) -> B(c17)
B(c17) -> U(c17)
T(c16) & F(c16) -> P(c17)
P(c17) -> W(c17)
T(c17) & Q(c17) -> F(c18)
F(c18) -> W(c18)
W(c17) & P(c17) -> D(c18)
D(c18) -> V(c18)
V(c17) & G(c17) -> Q(c18)
Q(c18) -> U(c18)
U(c17) & B(c17) -> R(c18)
R(c18) -> T(c18)
</premises>
<proof>
I(c0) ; R
T(c0) ; R
D(c1) ; ->E
U(c1) ; ->E
H(c2) ; ->E
W(c2) ; ->E
Q(c3) ; ->E
U(c3) ; ->E
O(c4) ; ->E
W(c4) ; ->E
D(c5) ; ->E
T(c5) ; ->E
M(c6) ; ->E
W(c6) ; ->E
R(c7) ; ->E
U(c7) ; ->E
N(c8) ; ->E
W(c8) ; ->E
M(c9) ; ->E
U(c9) ; ->E
C(c10) ; ->E
U(c10) ; ->E
K(c11) ; ->E
V(c11) ; ->E
C(c12) ; ->E
W(c12) ; ->E
O(c13) ; ->E
W(c13) ; ->E
D(c14) ; ->E
V(c14) ; ->E
E(c15) ; ->E
U(c15) ; ->E
F(c16) ; ->E
T(c16) ; ->E
P(c17) ; ->E
W(c17) ; ->E
D(c18) ; ->E
</proof>
<conclusion>
D(c18)
</conclusion>
</formal>
<answer>
coral
</answer><|endoftext|><question>
1. c0 is ruby.
2. c0 is east.
3. If c0 is west and c0 is ruby, then c1 is pearl.
4. If c1 is pearl, then c1 is west.
5. If c0 is east and c0 is ruby, then c1 is violet.
6. If c1 is violet, then c1 is north.
7. If c0 is north and c0 is ruby, then c1 is lime.
8. If c1 is lime, then c1 is south.
9. If c0 is south and c0 is ruby, then c1 is amber.
10. If c1 is amber, then c1 is east.
11. If c1 is south and c1 is lime, then c2 is elm.
12. If c2 is elm, then c2 is west.
13. If c1 is north and c1 is violet, then c2 is meadow.
14. If c2 is meadow, then c2 is east.
15. If c1 is east and c1 is amber, then c2 is pearl.
16. If c2 is pearl, then c2 is south.
17. If c1 is west and c1 is pearl, then c2 is lime.
18. If c2 is lime, then c2 is north.
19. If c2 is north and c2 is lime, then c3 is amber.
20. If c3 is amber, then c3 is south.
21. If c2 is east and c2 is meadow, then c3 is pearl.
22. If c3 is pearl, then c3 is north.
23. If c2 is west and c2 is elm, then c3 is lime.
24. If c3 is lime, then c3 is east.
25. If c2 is south and c2 is pearl, then c3 is elm.
26. If c3 is elm, then c3 is west.
27. If c3 is west and c3 is elm, then c4 is orchid.
28. If c4 is orchid, then c4 is south.
29. If c3 is east and c3 is lime, then c4 is hazel.
30. If c4 is hazel, then c4 is north.
31. If c3 is south and c3 is amber, then c4 is elm.
32. If c4 is elm, then c4 is east.
33. If c3 is north and c3 is pearl, then c4 is violet.
34. If c4 is violet, then c4 is west.
35. If c4 is west and c4 is violet, then c5 is pearl.
36. If c5 is pearl, then c5 is west.
37. If c4 is east and c4 is elm, then c5 is elm.
38. If c5 is elm, then c5 is north.
39. If c4 is north and c4 is hazel, then c5 is violet.
40. If c5 is violet, then c5 is south.
41. If c4 is south and c4 is orchid, then c5 is amber.
42. If c5 is amber, then c5 is east.
43. If c5 is east and c5 is amber, then c6 is violet.
44. If c6 is violet, then c6 is west.
45. If c5 is north and c5 is elm, then c6 is meadow.
46. If c6 is meadow, then c6 is east.
47. If c5 is west and c5 is pearl, then c6 is elm.
48. If c6 is elm, then c6 is south.
49. If c5 is south and c5 is violet, then c6 is orchid.
50. If c6 is orchid, then c6 is north.
51. If c6 is north and c6 is orchid, then c7 is hazel.
52. If c7 is hazel, then c7 is north.
53. If c6 is south and c6 is elm, then c7 is amber.
54. If c7 is amber, then c7 is west.
55. If c6 is west and c6 is violet, then c7 is orchid.
56. If c7 is orchid, then c7 is south.
57. If c6 is east and c6 is meadow, then c7 is violet.
58. If c7 is violet, then c7 is east.
59. If c7 is north and c7 is hazel, then c8 is meadow.
60. If c8 is meadow, then c8 is south.
61. If c7 is south and c7 is orchid, then c8 is elm.
62. If c8 is elm, then c8 is west.
63. If c7 is east and c7 is violet, then c8 is orchid.
64. If c8 is orchid, then c8 is north.
65. If c7 is west and c7 is amber, then c8 is lime.
66. If c8 is lime, then c8 is east.
67. If c8 is north and c8 is orchid, then c9 is elm.
68. If c9 is elm, then c9 is south.
69. If c8 is east and c8 is lime, then c9 is meadow.
70. If c9 is meadow, then c9 is east.
71. If c8 is west and c8 is elm, then c9 is amber.
72. If c9 is amber, then c9 is north.
73. If c8 is south and c8 is meadow, then c9 is orchid.
74. If c9 is orchid, then c9 is west.
Which state applies to c9?
</question>

<formal>
<constants>
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
</constants>
<predicates>
Ax: x is lime
Bx: x is cedar
Cx: x is elm
Dx: x is orchid
Ex: x is pearl
Fx: x is ruby
Gx: x is amber
Hx: x is hazel
Ix: x is meadow
Jx: x is violet
Kx: x is north
Lx: x is south
Mx: x is east
Nx: x is west
</predicates>
<premises>
F(c0)
M(c0)
N(c0) & F(c0) -> E(c1)
E(c1) -> N(c1)
M(c0) & F(c0) -> J(c1)
J(c1) -> K(c1)
K(c0) & F(c0) -> A(c1)
A(c1) -> L(c1)
L(c0) & F(c0) -> G(c1)
G(c1) -> M(c1)
L(c1) & A(c1) -> C(c2)
C(c2) -> N(c2)
K(c1) & J(c1) -> I(c2)
I(c2) -> M(c2)
M(c1) & G(c1) -> E(c2)
E(c2) -> L(c2)
N(c1) & E(c1) -> A(c2)
A(c2) -> K(c2)
K(c2) & A(c2) -> G(c3)
G(c3) -> L(c3)
M(c2) & I(c2) -> E(c3)
E(c3) -> K(c3)
N(c2) & C(c2) -> A(c3)
A(c3) -> M(c3)
L(c2) & E(c2) -> C(c3)
C(c3) -> N(c3)
N(c3) & C(c3) -> D(c4)
D(c4) -> L(c4)
M(c3) & A(c3) -> H(c4)
H(c4) -> K(c4)
L(c3) & G(c3) -> C(c4)
C(c4) -> M(c4)
K(c3) & E(c3) -> J(c4)
J(c4) -> N(c4)
N(c4) & J(c4) -> E(c5)
E(c5) -> N(c5)
M(c4) & C(c4) -> C(c5)
C(c5) -> K(c5)
K(c4) & H(c4) -> J(c5)
J(c5) -> L(c5)
L(c4) & D(c4) -> G(c5)
G(c5) -> M(c5)
M(c5) & G(c5) -> J(c6)
J(c6) -> N(c6)
K(c5) & C(c5) -> I(c6)
I(c6) -> M(c6)
N(c5) & E(c5) -> C(c6)
C(c6) -> L(c6)
L(c5) & J(c5) -> D(c6)
D(c6) -> K(c6)
K(c6) & D(c6) -> H(c7)
H(c7) -> K(c7)
L(c6) & C(c6) -> G(c7)
G(c7) -> N(c7)
N(c6) & J(c6) -> D(c7)
D(c7) -> L(c7)
M(c6) & I(c6) -> J(c7)
J(c7) -> M(c7)
K(c7) & H(c7) -> I(c8)
I(c8) -> L(c8)
L(c7) & D(c7) -> C(c8)
C(c8) -> N(c8)
M(c7) & J(c7) -> D(c8)
D(c8) -> K(c8)
N(c7) & G(c7) -> A(c8)
A(c8) -> M(c8)
K(c8) & D(c8) -> C(c9)
C(c9) -> L(c9)
M(c8) & A(c8) -> I(c9)
I(c9) -> M(c9)
N(c8) & C(c8) -> G(c9)
G(c9) -> K(c9)
L(c8) & I(c8) -> D(c9)
D(c9) -> N(c9)
</premises>
<proof>
F(c0) ; R
M(c0) ; R
J(c1) ; ->E
K(c1) ; ->E
I(c2) ; ->E
M(c2) ; ->E
E(c3) ; ->E
K(c3) ; ->E
J(c4) ; ->E
N(c4) ; ->E
E(c5) ; ->E
N(c5) ; ->E
C(c6) ; ->E
L(c6) ; ->E
G(c7) ; ->E
N(c7) ; ->E
A(c8) ; ->E
M(c8) ; ->E
I(c9) ; ->E
</proof>
<conclusion>
I(c9)
</conclusion>
</formal>
<answer>
meadow
</answer><|endoftext|>
```

## Window 3824 (3 documents, 59 pad tokens)
```
<question>
1. c0 is coral.
2. c0 is north.
3. If c0 is east and c0 is coral, then c1 is amber.
4. If c1 is amber, then c1 is west.
5. If c0 is west and c0 is coral, then c1 is meadow.
6. If c1 is meadow, then c1 is north.
7. If c0 is south and c0 is coral, then c1 is granite.
8. If c1 is granite, then c1 is east.
9. If c0 is north and c0 is coral, then c1 is lime.
10. If c1 is lime, then c1 is south.
11. If c1 is east and c1 is granite, then c2 is granite.
12. If c2 is granite, then c2 is west.
13. If c1 is west and c1 is amber, then c2 is juniper.
14. If c2 is juniper, then c2 is south.
15. If c1 is south and c1 is lime, then c2 is orchid.
16. If c2 is orchid, then c2 is north.
17. If c1 is north and c1 is meadow, then c2 is slate.
18. If c2 is slate, then c2 is east.
19. If c2 is south and c2 is juniper, then c3 is harbor.
20. If c3 is harbor, then c3 is south.
21. If c2 is west and c2 is granite, then c3 is hazel.
22. If c3 is hazel, then c3 is east.
23. If c2 is north and c2 is orchid, then c3 is orchid.
24. If c3 is orchid, then c3 is west.
25. If c2 is east and c2 is slate, then c3 is juniper.
26. If c3 is juniper, then c3 is north.
27. If c3 is south and c3 is harbor, then c4 is maple.
28. If c4 is maple, then c4 is west.
29. If c3 is east and c3 is hazel, then c4 is cobalt.
30. If c4 is cobalt, then c4 is south.
31. If c3 is north and c3 is juniper, then c4 is meadow.
32. If c4 is meadow, then c4 is east.
33. If c3 is west and c3 is orchid, then c4 is pearl.
34. If c4 is pearl, then c4 is north.
35. If c4 is east and c4 is meadow, then c5 is juniper.
36. If c5 is juniper, then c5 is east.
37. If c4 is south and c4 is cobalt, then c5 is meadow.
38. If c5 is meadow, then c5 is west.
39. If c4 is west and c4 is maple, then c5 is slate.
40. If c5 is slate, then c5 is north.
41. If c4 is north and c4 is pearl, then c5 is amber.
42. If c5 is amber, then c5 is south.
43. If c5 is south and c5 is amber, then c6 is willow.
44. If c6 is willow, then c6 is south.
45. If c5 is west and c5 is meadow, then c6 is lime.
46. If c6 is lime, then c6 is east.
47. If c5 is east and c5 is juniper, then c6 is granite.
48. If c6 is granite, then c6 is north.
49. If c5 is north and c5 is slate, then c6 is cedar.
50. If c6 is cedar, then c6 is west.
51. If c6 is east and c6 is lime, then c7 is maple.
52. If c7 is maple, then c7 is south.
53. If c6 is north and c6 is granite, then c7 is lime.
54. If c7 is lime, then c7 is north.
55. If c6 is south and c6 is willow, then c7 is slate.
56. If c7 is slate, then c7 is west.
57. If c6 is west and c6 is cedar, then c7 is birch.
58. If c7 is birch, then c7 is east.
59. If c7 is east and c7 is birch, then c8 is elm.
60. If c8 is elm, then c8 is west.
61. If c7 is north and c7 is lime, then c8 is ivory.
62. If c8 is ivory, then c8 is east.
63. If c7 is south and c7 is maple, then c8 is juniper.
64. If c8 is juniper, then c8 is north.
65. If c7 is west and c7 is slate, then c8 is birch.
66. If c8 is birch, then c8 is south.
67. If c8 is south and c8 is birch, then c9 is laurel.
68. If c9 is laurel, then c9 is south.
69. If c8 is east and c8 is ivory, then c9 is elm.
70. If c9 is elm, then c9 is north.
71. If c8 is north and c8 is juniper, then c9 is violet.
72. If c9 is violet, then c9 is west.
73. If c8 is west and c8 is elm, then c9 is olive.
74. If c9 is olive, then c9 is east.
75. If c9 is south and c9 is laurel, then c10 is granite.
76. If c10 is granite, then c10 is south.
77. If c9 is east and c9 is olive, then c10 is amber.
78. If c10 is amber, then c10 is north.
79. If c9 is west and c9 is violet, then c10 is orchid.
80. If c10 is orchid, then c10 is west.
81. If c9 is north and c9 is elm, then c10 is meadow.
82. If c10 is meadow, then c10 is east.
83. If c10 is north and c10 is amber, then c11 is cedar.
84. If c11 is cedar, then c11 is north.
85. If c10 is west and c10 is orchid, then c11 is ruby.
86. If c11 is ruby, then c11 is south.
87. If c10 is east and c10 is meadow, then c11 is granite.
88. If c11 is granite, then c11 is east.
89. If c10 is south and c10 is granite, then c11 is amber.
90. If c11 is amber, then c11 is west.
91. If c11 is west and c11 is amber, then c12 is birch.
92. If c12 is birch, then c12 is east.
93. If c11 is north and c11 is cedar, then c12 is hazel.
94. If c12 is hazel, then c12 is south.
95. If c11 is south and c11 is ruby, then c12 is elm.
96. If c12 is elm, then c12 is west.
97. If c11 is east and c11 is granite, then c12 is willow.
98. If c12 is willow, then c12 is north.
99. If c12 is north and c12 is willow, then c13 is juniper.
100. If c13 is juniper, then c13 is south.
101. If c12 is east and c12 is birch, then c13 is lime.
102. If c13 is lime, then c13 is north.
103. If c12 is west and c12 is elm, then c13 is slate.
104. If c13 is slate, then c13 is west.
105. If c12 is south and c12 is hazel, then c13 is poppy.
106. If c13 is poppy, then c13 is east.
107. If c13 is east and c13 is poppy, then c14 is willow.
108. If c14 is willow, then c14 is north.
109. If c13 is west and c13 is slate, then c14 is olive.
110. If c14 is olive, then c14 is south.
111. If c13 is north and c13 is lime, then c14 is ivory.
112. If c14 is ivory, then c14 is west.
113. If c13 is south and c13 is juniper, then c14 is birch.
114. If c14 is birch, then c14 is east.
115. If c14 is west and c14 is ivory, then c15 is birch.
116. If c15 is birch, then c15 is south.
117. If c14 is east and c14 is birch, then c15 is teal.
118. If c15 is teal, then c15 is north.
119. If c14 is south and c14 is olive, then c15 is willow.
120. If c15 is willow, then c15 is east.
121. If c14 is north and c14 is willow, then c15 is violet.
122. If c15 is violet, then c15 is west.
123. If c15 is north and c15 is teal, then c16 is harbor.
124. If c16 is harbor, then c16 is east.
125. If c15 is west and c15 is violet, then c16 is maple.
126. If c16 is maple, then c16 is south.
127. If c15 is east and c15 is willow, then c16 is granite.
128. If c16 is granite, then c16 is north.
129. If c15 is south and c15 is birch, then c16 is meadow.
130. If c16 is meadow, then c16 is west.
131. If c16 is south and c16 is maple, then c17 is orchid.
132. If c17 is orchid, then c17 is south.
133. If c16 is east and c16 is harbor, then c17 is lime.
134. If c17 is lime, then c17 is north.
135. If c16 is west and c16 is meadow, then c17 is harbor.
136. If c17 is harbor, then c17 is east.
137. If c16 is north and c16 is granite, then c17 is willow.
138. If c17 is willow, then c17 is west.
139. If c17 is east and c17 is harbor, then c18 is cobalt.
140. If c18 is cobalt, then c18 is south.
141. If c17 is west and c17 is willow, then c18 is hazel.
142. If c18 is hazel, then c18 is west.
143. If c17 is south and c17 is orchid, then c18 is orchid.
144. If c18 is orchid, then c18 is east.
145. If c17 is north and c17 is lime, then c18 is amber.
146. If c18 is amber, then c18 is north.
147. If c18 is north and c18 is amber, then c19 is cobalt.
148. If c19 is cobalt, then c19 is east.
149. If c18 is west and c18 is hazel, then c19 is lime.
150. If c19 is lime, then c19 is north.
151. If c18 is south and c18 is cobalt, then c19 is olive.
152. If c19 is olive, then c19 is south.
153. If c18 is east and c18 is orchid, then c19 is willow.
154. If c19 is willow, then c19 is west.
155. If c19 is north and c19 is lime, then c20 is meadow.
156. If c20 is meadow, then c20 is east.
157. If c19 is west and c19 is willow, then c20 is lime.
158. If c20 is lime, then c20 is north.
159. If c19 is east and c19 is cobalt, then c20 is hazel.
160. If c20 is hazel, then c20 is west.
161. If c19 is south and c19 is olive, then c20 is juniper.
162. If c20 is juniper, then c20 is south.
163. If c20 is south and c20 is juniper, then c21 is birch.
164. If c21 is birch, then c21 is west.
165. If c20 is west and c20 is hazel, then c21 is granite.
166. If c21 is granite, then c21 is south.
167. If c20 is east and c20 is meadow, then c21 is pearl.
168. If c21 is pearl, then c21 is north.
169. If c20 is north and c20 is lime, then c21 is ivory.
170. If c21 is ivory, then c21 is east.
171. If c21 is west and c21 is birch, then c22 is juniper.
172. If c22 is juniper, then c22 is south.
173. If c21 is east and c21 is ivory, then c22 is olive.
174. If c22 is olive, then c22 is east.
175. If c21 is south and c21 is granite, then c22 is lime.
176. If c22 is lime, then c22 is west.
177. If c21 is north and c21 is pearl, then c22 is willow.
178. If c22 is willow, then c22 is north.
179. If c22 is north and c22 is willow, then c23 is cobalt.
180. If c23 is cobalt, then c23 is west.
181. If c22 is south and c22 is juniper, then c23 is violet.
182. If c23 is violet, then c23 is north.
183. If c22 is west and c22 is lime, then c23 is meadow.
184. If c23 is meadow, then c23 is east.
185. If c22 is east and c22 is olive, then c23 is elm.
186. If c23 is elm, then c23 is south.
Which state applies to c23?
</question>

<formal>
<constants>
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
c15 = c15
c16 = c16
c17 = c17
c18 = c18
c19 = c19
c20 = c20
c21 = c21
c22 = c22
c23 = c23
</constants>
<predicates>
Ax: x is granite
Bx: x is orchid
Cx: x is ivory
Dx: x is juniper
Ex: x is coral
Fx: x is violet
Gx: x is willow
Hx: x is maple
Ix: x is olive
Jx: x is hazel
Kx: x is cobalt
Lx: x is teal
Mx: x is amber
Nx: x is birch
Ox: x is elm
Px: x is slate
Qx: x is meadow
Rx: x is harbor
Sx: x is ruby
Tx: x is cedar
Ux: x is lime
Vx: x is laurel
Wx: x is pearl
Xx: x is poppy
Yx: x is north
Zx: x is south
P0(x): x is east
P1(x): x is west
</predicates>
<premises>
E(c0)
Y(c0)
P0(c0) & E(c0) -> M(c1)
M(c1) -> P1(c1)
P1(c0) & E(c0) -> Q(c1)
Q(c1) -> Y(c1)
Z(c0) & E(c0) -> A(c1)
A(c1) -> P0(c1)
Y(c0) & E(c0) -> U(c1)
U(c1) -> Z(c1)
P0(c1) & A(c1) -> A(c2)
A(c2) -> P1(c2)
P1(c1) & M(c1) -> D(c2)
D(c2) -> Z(c2)
Z(c1) & U(c1) -> B(c2)
B(c2) -> Y(c2)
Y(c1) & Q(c1) -> P(c2)
P(c2) -> P0(c2)
Z(c2) & D(c2) -> R(c3)
R(c3) -> Z(c3)
P1(c2) & A(c2) -> J(c3)
J(c3) -> P0(c3)
Y(c2) & B(c2) -> B(c3)
B(c3) -> P1(c3)
P0(c2) & P(c2) -> D(c3)
D(c3) -> Y(c3)
Z(c3) & R(c3) -> H(c4)
H(c4) -> P1(c4)
P0(c3) & J(c3) -> K(c4)
K(c4) -> Z(c4)
Y(c3) & D(c3) -> Q(c4)
Q(c4) -> P0(c4)
P1(c3) & B(c3) -> W(c4)
W(c4) -> Y(c4)
P0(c4) & Q(c4) -> D(c5)
D(c5) -> P0(c5)
Z(c4) & K(c4) -> Q(c5)
Q(c5) -> P1(c5)
P1(c4) & H(c4) -> P(c5)
P(c5) -> Y(c5)
Y(c4) & W(c4) -> M(c5)
M(c5) -> Z(c5)
Z(c5) & M(c5) -> G(c6)
G(c6) -> Z(c6)
P1(c5) & Q(c5) -> U(c6)
U(c6) -> P0(c6)
P0(c5) & D(c5) -> A(c6)
A(c6) -> Y(c6)
Y(c5) & P(c5) -> T(c6)
T(c6) -> P1(c6)
P0(c6) & U(c6) -> H(c7)
H(c7) -> Z(c7)
Y(c6) & A(c6) -> U(c7)
U(c7) -> Y(c7)
Z(c6) & G(c6) -> P(c7)
P(c7) -> P1(c7)
P1(c6) & T(c6) -> N(c7)
N(c7) -> P0(c7)
P0(c7) & N(c7) -> O(c8)
O(c8) -> P1(c8)
Y(c7) & U(c7) -> C(c8)
C(c8) -> P0(c8)
Z(c7) & H(c7) -> D(c8)
D(c8) -> Y(c8)
P1(c7) & P(c7) -> N(c8)
N(c8) -> Z(c8)
Z(c8) & N(c8) -> V(c9)
V(c9) -> Z(c9)
P0(c8) & C(c8) -> O(c9)
O(c9) -> Y(c9)
Y(c8) & D(c8) -> F(c9)
F(c9) -> P1(c9)
P1(c8) & O(c8) -> I(c9)
I(c9) -> P0(c9)
Z(c9) & V(c9) -> A(c10)
A(c10) -> Z(c10)
P0(c9) & I(c9) -> M(c10)
M(c10) -> Y(c10)
P1(c9) & F(c9) -> B(c10)
B(c10) -> P1(c10)
Y(c9) & O(c9) -> Q(c10)
Q(c10) -> P0(c10)
Y(c10) & M(c10) -> T(c11)
T(c11) -> Y(c11)
P1(c10) & B(c10) -> S(c11)
S(c11) -> Z(c11)
P0(c10) & Q(c10) -> A(c11)
A(c11) -> P0(c11)
Z(c10) & A(c10) -> M(c11)
M(c11) -> P1(c11)
P1(c11) & M(c11) -> N(c12)
N(c12) -> P0(c12)
Y(c11) & T(c11) -> J(c12)
J(c12) -> Z(c12)
Z(c11) & S(c11) -> O(c12)
O(c12) -> P1(c12)
P0(c11) & A(c11) -> G(c12)
G(c12) -> Y(c12)
Y(c12) & G(c12) -> D(c13)
D(c13) -> Z(c13)
P0(c12) & N(c12) -> U(c13)
U(c13) -> Y(c13)
P1(c12) & O(c12) -> P(c13)
P(c13) -> P1(c13)
Z(c12) & J(c12) -> X(c13)
X(c13) -> P0(c13)
P0(c13) & X(c13) -> G(c14)
G(c14) -> Y(c14)
P1(c13) & P(c13) -> I(c14)
I(c14) -> Z(c14)
Y(c13) & U(c13) -> C(c14)
C(c14) -> P1(c14)
Z(c13) & D(c13) -> N(c14)
N(c14) -> P0(c14)
P1(c14) & C(c14) -> N(c15)
N(c15) -> Z(c15)
P0(c14) & N(c14) -> L(c15)
L(c15) -> Y(c15)
Z(c14) & I(c14) -> G(c15)
G(c15) -> P0(c15)
Y(c14) & G(c14) -> F(c15)
F(c15) -> P1(c15)
Y(c15) & L(c15) -> R(c16)
R(c16) -> P0(c16)
P1(c15) & F(c15) -> H(c16)
H(c16) -> Z(c16)
P0(c15) & G(c15) -> A(c16)
A(c16) -> Y(c16)
Z(c15) & N(c15) -> Q(c16)
Q(c16) -> P1(c16)
Z(c16) & H(c16) -> B(c17)
B(c17) -> Z(c17)
P0(c16) & R(c16) -> U(c17)
U(c17) -> Y(c17)
P1(c16) & Q(c16) -> R(c17)
R(c17) -> P0(c17)
Y(c16) & A(c16) -> G(c17)
G(c17) -> P1(c17)
P0(c17) & R(c17) -> K(c18)
K(c18) -> Z(c18)
P1(c17) & G(c17) -> J(c18)
J(c18) -> P1(c18)
Z(c17) & B(c17) -> B(c18)
B(c18) -> P0(c18)
Y(c17) & U(c17) -> M(c18)
M(c18) -> Y(c18)
Y(c18) & M(c18) -> K(c19)
K(c19) -> P0(c19)
P1(c18) & J(c18) -> U(c19)
U(c19) -> Y(c19)
Z(c18) & K(c18) -> I(c19)
I(c19) -> Z(c19)
P0(c18) & B(c18) -> G(c19)
G(c19) -> P1(c19)
Y(c19) & U(c19) -> Q(c20)
Q(c20) -> P0(c20)
P1(c19) & G(c19) -> U(c20)
U(c20) -> Y(c20)
P0(c19) & K(c19) -> J(c20)
J(c20) -> P1(c20)
Z(c19) & I(c19) -> D(c20)
D(c20) -> Z(c20)
Z(c20) & D(c20) -> N(c21)
N(c21) -> P1(c21)
P1(c20) & J(c20) -> A(c21)
A(c21) -> Z(c21)
P0(c20) & Q(c20) -> W(c21)
W(c21) -> Y(c21)
Y(c20) & U(c20) -> C(c21)
C(c21) -> P0(c21)
P1(c21) & N(c21) -> D(c22)
D(c22) -> Z(c22)
P0(c21) & C(c21) -> I(c22)
I(c22) -> P0(c22)
Z(c21) & A(c21) -> U(c22)
U(c22) -> P1(c22)
Y(c21) & W(c21) -> G(c22)
G(c22) -> Y(c22)
Y(c22) & G(c22) -> K(c23)
K(c23) -> P1(c23)
Z(c22) & D(c22) -> F(c23)
F(c23) -> Y(c23)
P1(c22) & U(c22) -> Q(c23)
Q(c23) -> P0(c23)
P0(c22) & I(c22) -> O(c23)
O(c23) -> Z(c23)
</premises>
<proof>
E(c0) ; R
Y(c0) ; R
U(c1) ; ->E
Z(c1) ; ->E
B(c2) ; ->E
Y(c2) ; ->E
B(c3) ; ->E
P1(c3) ; ->E
W(c4) ; ->E
Y(c4) ; ->E
M(c5) ; ->E
Z(c5) ; ->E
G(c6) ; ->E
Z(c6) ; ->E
P(c7) ; ->E
P1(c7) ; ->E
N(c8) ; ->E
Z(c8) ; ->E
V(c9) ; ->E
Z(c9) ; ->E
A(c10) ; ->E
Z(c10) ; ->E
M(c11) ; ->E
P1(c11) ; ->E
N(c12) ; ->E
P0(c12) ; ->E
U(c13) ; ->E
Y(c13) ; ->E
C(c14) ; ->E
P1(c14) ; ->E
N(c15) ; ->E
Z(c15) ; ->E
Q(c16) ; ->E
P1(c16) ; ->E
R(c17) ; ->E
P0(c17) ; ->E
K(c18) ; ->E
Z(c18) ; ->E
I(c19) ; ->E
Z(c19) ; ->E
D(c20) ; ->E
Z(c20) ; ->E
N(c21) ; ->E
P1(c21) ; ->E
D(c22) ; ->E
Z(c22) ; ->E
F(c23) ; ->E
</proof>
<conclusion>
F(c23)
</conclusion>
</formal>
<answer>
violet
</answer><|endoftext|><question>
1. c0 is poppy.
2. c0 is north.
3. If c0 is west and c0 is poppy, then c1 is ruby.
4. If c1 is ruby, then c1 is north.
5. If c0 is north and c0 is poppy, then c1 is laurel.
6. If c1 is laurel, then c1 is west.
7. If c0 is south and c0 is poppy, then c1 is olive.
8. If c1 is olive, then c1 is south.
9. If c0 is east and c0 is poppy, then c1 is teal.
10. If c1 is teal, then c1 is east.
Which state applies to c1?
</question>

<formal>
<constants>
c0 = c0
c1 = c1
</constants>
<predicates>
Ax: x is olive
Bx: x is teal
Cx: x is ruby
Dx: x is laurel
Ex: x is poppy
Fx: x is north
Gx: x is south
Hx: x is east
Ix: x is west
</predicates>
<premises>
E(c0)
F(c0)
I(c0) & E(c0) -> C(c1)
C(c1) -> F(c1)
F(c0) & E(c0) -> D(c1)
D(c1) -> I(c1)
G(c0) & E(c0) -> A(c1)
A(c1) -> G(c1)
H(c0) & E(c0) -> B(c1)
B(c1) -> H(c1)
</premises>
<proof>
E(c0) ; R
F(c0) ; R
D(c1) ; ->E
</proof>
<conclusion>
D(c1)
</conclusion>
</formal>
<answer>
laurel
</answer><|endoftext|><question>
1. c0 is amber.
2. c0 is east.
3. If c0 is east and c0 is amber, then c1 is olive.
4. If c1 is olive, then c1 is south.
5. If c0 is south and c0 is amber, then c1 is violet.
6. If c1 is violet, then c1 is east.
7. If c0 is west and c0 is amber, then c1 is ruby.
8. If c1 is ruby, then c1 is west.
9. If c0 is north and c0 is amber, then c1 is coral.
10. If c1 is coral, then c1 is north.
11. If c1 is north and c1 is coral, then c2 is ruby.
12. If c2 is ruby, then c2 is south.
13. If c1 is south and c1 is olive, then c2 is olive.
14. If c2 is olive, then c2 is west.
15. If c1 is east and c1 is violet, then c2 is coral.
16. If c2 is coral, then c2 is east.
17. If c1 is west and c1 is ruby, then c2 is violet.
18. If c2 is violet, then c2 is north.
Which state applies to c2?
</question>

<formal>
<constants>
c0 = c0
c1 = c1
c2 = c2
</constants>
<predicates>
Ax: x is coral
Bx: x is violet
Cx: x is amber
Dx: x is olive
Ex: x is ruby
Fx: x is north
Gx: x is south
Hx: x is east
Ix: x is west
</predicates>
<premises>
C(c0)
H(c0)
H(c0) & C(c0) -> D(c1)
D(c1) -> G(c1)
G(c0) & C(c0) -> B(c1)
B(c1) -> H(c1)
I(c0) & C(c0) -> E(c1)
E(c1) -> I(c1)
F(c0) & C(c0) -> A(c1)
A(c1) -> F(c1)
F(c1) & A(c1) -> E(c2)
E(c2) -> G(c2)
G(c1) & D(c1) -> D(c2)
D(c2) -> I(c2)
H(c1) & B(c1) -> A(c2)
A(c2) -> H(c2)
I(c1) & E(c1) -> B(c2)
B(c2) -> F(c2)
</premises>
<proof>
C(c0) ; R
H(c0) ; R
D(c1) ; ->E
G(c1) ; ->E
D(c2) ; ->E
</proof>
<conclusion>
D(c2)
</conclusion>
</formal>
<answer>
olive
</answer><|endoftext|>
```

## Window 7644 summary: [{"tokens": 5964, "head": "<question> 1. c0 is amber. 2. c0 is north. 3. If c0 is east and c0 is amber, the", "tail": "n> T(c20) </conclusion> </formal> <answer> juniper </answer>"}, {"tokens": 2002, "head": "<question> 1. c0 is ivory. 2. c0 is north. 3. If c0 is east and c0 is ivory, the", "tail": "ion> A(c7) </conclusion> </formal> <answer> willow </answer>"}]

## Window 7962 summary: [{"tokens": 7039, "head": "<question> 1. c0 is granite. 2. c0 is west. 3. If c0 is east and c0 is granite, ", "tail": "on> S(c23) </conclusion> </formal> <answer> willow </answer>"}, {"tokens": 683, "head": "<question> 1. c0 is poppy. 2. c0 is north. 3. If c0 is west and c0 is poppy, the", "tail": "ion> E(c2) </conclusion> </formal> <answer> laurel </answer>"}]

## Window 8060 summary: [{"tokens": 5334, "head": "<question> 1. c0 is slate. 2. c0 is west. 3. If c0 is south and c0 is slate, the", "tail": "on> S(c18) </conclusion> </formal> <answer> meadow </answer>"}, {"tokens": 2808, "head": "<question> 1. c0 is laurel. 2. c0 is north. 3. If c0 is north and c0 is laurel, ", "tail": "ion> I(c10) </conclusion> </formal> <answer> coral </answer>"}]

## Window 8222 summary: [{"tokens": 6304, "head": "<question> 1. c0 is teal. 2. c0 is east. 3. If c0 is north and c0 is teal, then ", "tail": "ion> A(c21) </conclusion> </formal> <answer> poppy </answer>"}, {"tokens": 1442, "head": "<question> 1. c0 is lime. 2. c0 is north. 3. If c0 is south and c0 is lime, then", "tail": "on> F(c5) </conclusion> </formal> <answer> granite </answer>"}]

## Window 9449 summary: [{"tokens": 5017, "head": "<question> 1. c0 is poppy. 2. c0 is east. 3. If c0 is west and c0 is poppy, then", "tail": "ion> J(c17) </conclusion> </formal> <answer> coral </answer>"}, {"tokens": 2244, "head": "<question> 1. c0 is meadow. 2. c0 is west. 3. If c0 is east and c0 is meadow, th", "tail": "usion> G(c8) </conclusion> </formal> <answer> lime </answer>"}, {"tokens": 413, "head": "<question> 1. c0 is lime. 2. c0 is north. 3. If c0 is east and c0 is lime, then ", "tail": "sion> A(c1) </conclusion> </formal> <answer> olive </answer>"}]

## Window 10983 summary: [{"tokens": 4353, "head": "<question> 1. c0 is granite. 2. c0 is south. 3. If c0 is south and c0 is granite", "tail": "ion> N(c15) </conclusion> </formal> <answer> coral </answer>"}, {"tokens": 3093, "head": "<question> 1. c0 is cedar. 2. c0 is south. 3. If c0 is south and c0 is cedar, th", "tail": "on> J(c11) </conclusion> </formal> <answer> laurel </answer>"}]

## Window 12198 summary: [{"tokens": 4717, "head": "<question> 1. c0 is birch. 2. c0 is east. 3. If c0 is south and c0 is birch, the", "tail": "ion> I(c16) </conclusion> </formal> <answer> hazel </answer>"}, {"tokens": 3440, "head": "<question> 1. c0 is cedar. 2. c0 is east. 3. If c0 is west and c0 is cedar, then", "tail": "on> J(c12) </conclusion> </formal> <answer> meadow </answer>"}]

## Window 13686 summary: [{"tokens": 5307, "head": "<question> 1. c0 is poppy. 2. c0 is west. 3. If c0 is east and c0 is poppy, then", "tail": "on> J(c18) </conclusion> </formal> <answer> cobalt </answer>"}, {"tokens": 2520, "head": "<question> 1. c0 is olive. 2. c0 is south. 3. If c0 is south and c0 is olive, th", "tail": "sion> J(c9) </conclusion> </formal> <answer> ivory </answer>"}]

## Window 14192 summary: [{"tokens": 3780, "head": "<question> 1. c0 is amber. 2. c0 is west. 3. If c0 is west and c0 is amber, then", "tail": "usion> I(c13) </conclusion> </formal> <answer> elm </answer>"}, {"tokens": 4028, "head": "<question> 1. c0 is pearl. 2. c0 is north. 3. If c0 is west and c0 is pearl, the", "tail": "sion> H(c14) </conclusion> </formal> <answer> teal </answer>"}]

## Window 14629 summary: [{"tokens": 5329, "head": "<question> 1. c0 is olive. 2. c0 is south. 3. If c0 is south and c0 is olive, th", "tail": "ion> N(c18) </conclusion> </formal> <answer> amber </answer>"}, {"tokens": 2265, "head": "<question> 1. c0 is violet. 2. c0 is south. 3. If c0 is south and c0 is violet, ", "tail": "ion> E(c8) </conclusion> </formal> <answer> cobalt </answer>"}]

## Window 17048 summary: [{"tokens": 7042, "head": "<question> 1. c0 is hazel. 2. c0 is west. 3. If c0 is south and c0 is hazel, the", "tail": "usion> K(c23) </conclusion> </formal> <answer> elm </answer>"}, {"tokens": 933, "head": "<question> 1. c0 is hazel. 2. c0 is east. 3. If c0 is north and c0 is hazel, the", "tail": "usion> A(c3) </conclusion> </formal> <answer> teal </answer>"}]

## Window 17746 summary: [{"tokens": 5994, "head": "<question> 1. c0 is maple. 2. c0 is north. 3. If c0 is east and c0 is maple, the", "tail": "on> Q(c20) </conclusion> </formal> <answer> cobalt </answer>"}, {"tokens": 1711, "head": "<question> 1. c0 is poppy. 2. c0 is east. 3. If c0 is north and c0 is poppy, the", "tail": "usion> E(c6) </conclusion> </formal> <answer> lime </answer>"}]

## Window 17747 summary: [{"tokens": 5941, "head": "<question> 1. c0 is hazel. 2. c0 is west. 3. If c0 is north and c0 is hazel, the", "tail": "sion> Q(c20) </conclusion> </formal> <answer> teal </answer>"}, {"tokens": 1731, "head": "<question> 1. c0 is ivory. 2. c0 is south. 3. If c0 is north and c0 is ivory, th", "tail": "ion> B(c6) </conclusion> </formal> <answer> willow </answer>"}]

## Window 21580 summary: [{"tokens": 5979, "head": "<question> 1. c0 is olive. 2. c0 is south. 3. If c0 is west and c0 is olive, the", "tail": "ion> Q(c20) </conclusion> </formal> <answer> slate </answer>"}, {"tokens": 1983, "head": "<question> 1. c0 is granite. 2. c0 is north. 3. If c0 is south and c0 is granite", "tail": "ion> D(c7) </conclusion> </formal> <answer> orchid </answer>"}]

## Window 22267 summary: [{"tokens": 5630, "head": "<question> 1. c0 is hazel. 2. c0 is south. 3. If c0 is north and c0 is hazel, th", "tail": "sion> I(c19) </conclusion> </formal> <answer> lime </answer>"}, {"tokens": 1981, "head": "<question> 1. c0 is meadow. 2. c0 is south. 3. If c0 is south and c0 is meadow, ", "tail": "usion> F(c7) </conclusion> </formal> <answer> ruby </answer>"}]

## Window 23706 summary: [{"tokens": 5340, "head": "<question> 1. c0 is meadow. 2. c0 is east. 3. If c0 is east and c0 is meadow, th", "tail": "usion> B(c18) </conclusion> </formal> <answer> elm </answer>"}, {"tokens": 2803, "head": "<question> 1. c0 is teal. 2. c0 is south. 3. If c0 is north and c0 is teal, then", "tail": "n> D(c10) </conclusion> </formal> <answer> juniper </answer>"}]

## Window 23924 summary: [{"tokens": 7672, "head": "<question> 1. c0 is cedar. 2. c0 is west. 3. If c0 is west and c0 is cedar, then", "tail": "n> R(c25) </conclusion> </formal> <answer> juniper </answer>"}]

## Window 27439 summary: [{"tokens": 7666, "head": "<question> 1. c0 is cedar. 2. c0 is south. 3. If c0 is south and c0 is cedar, th", "tail": "on> G(c25) </conclusion> </formal> <answer> willow </answer>"}]

## Window 27973 summary: [{"tokens": 6270, "head": "<question> 1. c0 is meadow. 2. c0 is south. 3. If c0 is west and c0 is meadow, t", "tail": "on> G(c21) </conclusion> </formal> <answer> orchid </answer>"}, {"tokens": 1445, "head": "<question> 1. c0 is coral. 2. c0 is north. 3. If c0 is north and c0 is coral, th", "tail": "ion> F(c5) </conclusion> </formal> <answer> violet </answer>"}]

## Window 27984 summary: [{"tokens": 3460, "head": "<question> 1. c0 is orchid. 2. c0 is north. 3. If c0 is south and c0 is orchid, ", "tail": "ion> D(c12) </conclusion> </formal> <answer> birch </answer>"}, {"tokens": 3746, "head": "<question> 1. c0 is birch. 2. c0 is north. 3. If c0 is east and c0 is birch, the", "tail": "on> A(c13) </conclusion> </formal> <answer> harbor </answer>"}, {"tokens": 935, "head": "<question> 1. c0 is cedar. 2. c0 is south. 3. If c0 is south and c0 is cedar, th", "tail": "usion> B(c3) </conclusion> </formal> <answer> lime </answer>"}]

## Window 32399 summary: [{"tokens": 4998, "head": "<question> 1. c0 is laurel. 2. c0 is east. 3. If c0 is north and c0 is laurel, t", "tail": "n> B(c17) </conclusion> </formal> <answer> granite </answer>"}, {"tokens": 2519, "head": "<question> 1. c0 is elm. 2. c0 is south. 3. If c0 is north and c0 is elm, then c", "tail": "sion> F(c9) </conclusion> </formal> <answer> birch </answer>"}]

## Window 32427 summary: [{"tokens": 7344, "head": "<question> 1. c0 is laurel. 2. c0 is south. 3. If c0 is east and c0 is laurel, t", "tail": "on> S(c24) </conclusion> </formal> <answer> cobalt </answer>"}, {"tokens": 419, "head": "<question> 1. c0 is elm. 2. c0 is north. 3. If c0 is north and c0 is elm, then c", "tail": "sion> B(c1) </conclusion> </formal> <answer> cedar </answer>"}]

## Window 33950 summary: [{"tokens": 5979, "head": "<question> 1. c0 is meadow. 2. c0 is north. 3. If c0 is north and c0 is meadow, ", "tail": "sion> A(c20) </conclusion> </formal> <answer> ruby </answer>"}, {"tokens": 1728, "head": "<question> 1. c0 is elm. 2. c0 is east. 3. If c0 is east and c0 is elm, then c1 ", "tail": "sion> C(c6) </conclusion> </formal> <answer> pearl </answer>"}]

## Window 34086 summary: [{"tokens": 4711, "head": "<question> 1. c0 is juniper. 2. c0 is south. 3. If c0 is south and c0 is juniper", "tail": "ion> C(c16) </conclusion> </formal> <answer> maple </answer>"}, {"tokens": 2530, "head": "<question> 1. c0 is orchid. 2. c0 is west. 3. If c0 is east and c0 is orchid, th", "tail": "sion> G(c9) </conclusion> </formal> <answer> coral </answer>"}, {"tokens": 918, "head": "<question> 1. c0 is ruby. 2. c0 is south. 3. If c0 is north and c0 is ruby, then", "tail": "sion> B(c3) </conclusion> </formal> <answer> slate </answer>"}]

## Window 35490 summary: [{"tokens": 5313, "head": "<question> 1. c0 is birch. 2. c0 is south. 3. If c0 is north and c0 is birch, th", "tail": "sion> P(c18) </conclusion> </formal> <answer> lime </answer>"}, {"tokens": 2515, "head": "<question> 1. c0 is meadow. 2. c0 is north. 3. If c0 is north and c0 is meadow, ", "tail": "sion> E(c9) </conclusion> </formal> <answer> coral </answer>"}]

## Window 35491 summary: [{"tokens": 6655, "head": "<question> 1. c0 is elm. 2. c0 is west. 3. If c0 is south and c0 is elm, then c1", "tail": "on> J(c22) </conclusion> </formal> <answer> orchid </answer>"}, {"tokens": 1190, "head": "<question> 1. c0 is meadow. 2. c0 is south. 3. If c0 is east and c0 is meadow, t", "tail": "sion> A(c4) </conclusion> </formal> <answer> coral </answer>"}]
