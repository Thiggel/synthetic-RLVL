# Decoded-batch audit examples (document-preserving docpack loader)

## Window 0 (2 documents, 226 pad tokens)
```
<question>
1. c0 is pearl.
2. c0 is north.
3. If c0 is north and c0 is pearl, then c1 is slate.
4. If c1 is slate, then c1 is south.
5. If c0 is west and c0 is pearl, then c1 is willow.
6. If c1 is willow, then c1 is east.
7. If c0 is east and c0 is pearl, then c1 is orchid.
8. If c1 is orchid, then c1 is west.
9. If c0 is south and c0 is pearl, then c1 is violet.
10. If c1 is violet, then c1 is north.
11. If c1 is west and c1 is orchid, then c2 is hazel.
12. If c2 is hazel, then c2 is south.
13. If c1 is north and c1 is violet, then c2 is meadow.
14. If c2 is meadow, then c2 is north.
15. If c1 is east and c1 is willow, then c2 is teal.
16. If c2 is teal, then c2 is west.
17. If c1 is south and c1 is slate, then c2 is cobalt.
18. If c2 is cobalt, then c2 is east.
19. If c2 is south and c2 is hazel, then c3 is lime.
20. If c3 is lime, then c3 is east.
21. If c2 is east and c2 is cobalt, then c3 is birch.
22. If c3 is birch, then c3 is north.
23. If c2 is north and c2 is meadow, then c3 is harbor.
24. If c3 is harbor, then c3 is south.
25. If c2 is west and c2 is teal, then c3 is orchid.
26. If c3 is orchid, then c3 is west.
27. If c3 is east and c3 is lime, then c4 is olive.
28. If c4 is olive, then c4 is south.
29. If c3 is north and c3 is birch, then c4 is coral.
30. If c4 is coral, then c4 is west.
31. If c3 is south and c3 is harbor, then c4 is cedar.
32. If c4 is cedar, then c4 is north.
33. If c3 is west and c3 is orchid, then c4 is violet.
34. If c4 is violet, then c4 is east.
35. If c4 is west and c4 is coral, then c5 is granite.
36. If c5 is granite, then c5 is west.
37. If c4 is east and c4 is violet, then c5 is lime.
38. If c5 is lime, then c5 is east.
39. If c4 is north and c4 is cedar, then c5 is amber.
40. If c5 is amber, then c5 is south.
41. If c4 is south and c4 is olive, then c5 is ruby.
42. If c5 is ruby, then c5 is north.
43. If c5 is south and c5 is amber, then c6 is slate.
44. If c6 is slate, then c6 is west.
45. If c5 is north and c5 is ruby, then c6 is hazel.
46. If c6 is hazel, then c6 is south.
47. If c5 is west and c5 is granite, then c6 is cedar.
48. If c6 is cedar, then c6 is east.
49. If c5 is east and c5 is lime, then c6 is meadow.
50. If c6 is meadow, then c6 is north.
51. If c6 is north and c6 is meadow, then c7 is cedar.
52. If c7 is cedar, then c7 is west.
53. If c6 is east and c6 is cedar, then c7 is ivory.
54. If c7 is ivory, then c7 is east.
55. If c6 is west and c6 is slate, then c7 is teal.
56. If c7 is teal, then c7 is north.
57. If c6 is south and c6 is hazel, then c7 is hazel.
58. If c7 is hazel, then c7 is south.
59. If c7 is west and c7 is cedar, then c8 is slate.
60. If c8 is slate, then c8 is south.
61. If c7 is south and c7 is hazel, then c8 is ivory.
62. If c8 is ivory, then c8 is north.
63. If c7 is north and c7 is teal, then c8 is poppy.
64. If c8 is poppy, then c8 is east.
65. If c7 is east and c7 is ivory, then c8 is maple.
66. If c8 is maple, then c8 is west.
67. If c8 is south and c8 is slate, then c9 is ruby.
68. If c9 is ruby, then c9 is north.
69. If c8 is north and c8 is ivory, then c9 is orchid.
70. If c9 is orchid, then c9 is west.
71. If c8 is west and c8 is maple, then c9 is granite.
72. If c9 is granite, then c9 is east.
73. If c8 is east and c8 is poppy, then c9 is harbor.
74. If c9 is harbor, then c9 is south.
75. If c9 is east and c9 is granite, then c10 is granite.
76. If c10 is granite, then c10 is east.
77. If c9 is south and c9 is harbor, then c10 is olive.
78. If c10 is olive, then c10 is north.
79. If c9 is north and c9 is ruby, then c10 is coral.
80. If c10 is coral, then c10 is south.
81. If c9 is west and c9 is orchid, then c10 is poppy.
82. If c10 is poppy, then c10 is west.
83. If c10 is east and c10 is granite, then c11 is hazel.
84. If c11 is hazel, then c11 is west.
85. If c10 is south and c10 is coral, then c11 is juniper.
86. If c11 is juniper, then c11 is south.
87. If c10 is west and c10 is poppy, then c11 is maple.
88. If c11 is maple, then c11 is north.
89. If c10 is north and c10 is olive, then c11 is laurel.
90. If c11 is laurel, then c11 is east.
91. If c11 is south and c11 is juniper, then c12 is ruby.
92. If c12 is ruby, then c12 is south.
93. If c11 is west and c11 is hazel, then c12 is juniper.
94. If c12 is juniper, then c12 is west.
95. If c11 is east and c11 is laurel, then c12 is willow.
96. If c12 is willow, then c12 is east.
97. If c11 is north and c11 is maple, then c12 is slate.
98. If c12 is slate, then c12 is north.
99. If c12 is north and c12 is slate, then c13 is ruby.
100. If c13 is ruby, then c13 is east.
101. If c12 is east and c12 is willow, then c13 is granite.
102. If c13 is granite, then c13 is north.
103. If c12 is west and c12 is juniper, then c13 is cedar.
104. If c13 is cedar, then c13 is west.
105. If c12 is south and c12 is ruby, then c13 is slate.
106. If c13 is slate, then c13 is south.
107. If c13 is west and c13 is cedar, then c14 is granite.
108. If c14 is granite, then c14 is east.
109. If c13 is east and c13 is ruby, then c14 is elm.
110. If c14 is elm, then c14 is west.
111. If c13 is south and c13 is slate, then c14 is meadow.
112. If c14 is meadow, then c14 is north.
113. If c13 is north and c13 is granite, then c14 is ruby.
114. If c14 is ruby, then c14 is south.
115. If c14 is west and c14 is elm, then c15 is ruby.
116. If c15 is ruby, then c15 is west.
117. If c14 is south and c14 is ruby, then c15 is orchid.
118. If c15 is orchid, then c15 is north.
119. If c14 is north and c14 is meadow, then c15 is poppy.
120. If c15 is poppy, then c15 is east.
121. If c14 is east and c14 is granite, then c15 is granite.
122. If c15 is granite, then c15 is south.
123. If c15 is west and c15 is ruby, then c16 is juniper.
124. If c16 is juniper, then c16 is east.
125. If c15 is north and c15 is orchid, then c16 is amber.
126. If c16 is amber, then c16 is north.
127. If c15 is south and c15 is granite, then c16 is ivory.
128. If c16 is ivory, then c16 is south.
129. If c15 is east and c15 is poppy, then c16 is cobalt.
130. If c16 is cobalt, then c16 is west.
131. If c16 is east and c16 is juniper, then c17 is coral.
132. If c17 is coral, then c17 is south.
133. If c16 is north and c16 is amber, then c17 is violet.
134. If c17 is violet, then c17 is west.
135. If c16 is west and c16 is cobalt, then c17 is poppy.
136. If c17 is poppy, then c17 is north.
137. If c16 is south and c16 is ivory, then c17 is amber.
138. If c17 is amber, then c17 is east.
139. If c17 is west and c17 is violet, then c18 is ruby.
140. If c18 is ruby, then c18 is west.
141. If c17 is south and c17 is coral, then c18 is juniper.
142. If c18 is juniper, then c18 is north.
143. If c17 is north and c17 is poppy, then c18 is poppy.
144. If c18 is poppy, then c18 is south.
145. If c17 is east and c17 is amber, then c18 is coral.
146. If c18 is coral, then c18 is east.
147. If c18 is north and c18 is juniper, then c19 is teal.
148. If c19 is teal, then c19 is south.
149. If c18 is south and c18 is poppy, then c19 is orchid.
150. If c19 is orchid, then c19 is west.
151. If c18 is west and c18 is ruby, then c19 is slate.
152. If c19 is slate, then c19 is north.
153. If c18 is east and c18 is coral, then c19 is ruby.
154. If c19 is ruby, then c19 is east.
155. If c19 is north and c19 is slate, then c20 is lime.
156. If c20 is lime, then c20 is north.
157. If c19 is west and c19 is orchid, then c20 is amber.
158. If c20 is amber, then c20 is east.
159. If c19 is east and c19 is ruby, then c20 is willow.
160. If c20 is willow, then c20 is south.
161. If c19 is south and c19 is teal, then c20 is coral.
162. If c20 is coral, then c20 is west.
163. If c20 is north and c20 is lime, then c21 is maple.
164. If c21 is maple, then c21 is east.
165. If c20 is south and c20 is willow, then c21 is poppy.
166. If c21 is poppy, then c21 is north.
167. If c20 is west and c20 is coral, then c21 is violet.
168. If c21 is violet, then c21 is south.
169. If c20 is east and c20 is amber, then c21 is juniper.
170. If c21 is juniper, then c21 is west.
171. If c21 is west and c21 is juniper, then c22 is hazel.
172. If c22 is hazel, then c22 is south.
173. If c21 is north and c21 is poppy, then c22 is cobalt.
174. If c22 is cobalt, then c22 is west.
175. If c21 is east and c21 is maple, then c22 is granite.
176. If c22 is granite, then c22 is east.
177. If c21 is south and c21 is violet, then c22 is amber.
178. If c22 is amber, then c22 is north.
179. If c22 is east and c22 is granite, then c23 is granite.
180. If c23 is granite, then c23 is west.
181. If c22 is south and c22 is hazel, then c23 is maple.
182. If c23 is maple, then c23 is north.
183. If c22 is west and c22 is cobalt, then c23 is hazel.
184. If c23 is hazel, then c23 is south.
185. If c22 is north and c22 is amber, then c23 is lime.
186. If c23 is lime, then c23 is east.
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
Ax: x is juniper
Bx: x is lime
Cx: x is maple
Dx: x is olive
Ex: x is pearl
Fx: x is harbor
Gx: x is violet
Hx: x is birch
Ix: x is poppy
Jx: x is ruby
Kx: x is orchid
Lx: x is ivory
Mx: x is teal
Nx: x is laurel
Ox: x is cobalt
Px: x is willow
Qx: x is slate
Rx: x is amber
Sx: x is granite
Tx: x is elm
Ux: x is cedar
Vx: x is meadow
Wx: x is hazel
Xx: x is coral
Yx: x is north
Zx: x is south
P0(x): x is east
P1(x): x is west
</predicates>
<premises>
E(c0)
Y(c0)
Y(c0) & E(c0) -> Q(c1)
Q(c1) -> Z(c1)
P1(c0) & E(c0) -> P(c1)
P(c1) -> P0(c1)
P0(c0) & E(c0) -> K(c1)
K(c1) -> P1(c1)
Z(c0) & E(c0) -> G(c1)
G(c1) -> Y(c1)
P1(c1) & K(c1) -> W(c2)
W(c2) -> Z(c2)
Y(c1) & G(c1) -> V(c2)
V(c2) -> Y(c2)
P0(c1) & P(c1) -> M(c2)
M(c2) -> P1(c2)
Z(c1) & Q(c1) -> O(c2)
O(c2) -> P0(c2)
Z(c2) & W(c2) -> B(c3)
B(c3) -> P0(c3)
P0(c2) & O(c2) -> H(c3)
H(c3) -> Y(c3)
Y(c2) & V(c2) -> F(c3)
F(c3) -> Z(c3)
P1(c2) & M(c2) -> K(c3)
K(c3) -> P1(c3)
P0(c3) & B(c3) -> D(c4)
D(c4) -> Z(c4)
Y(c3) & H(c3) -> X(c4)
X(c4) -> P1(c4)
Z(c3) & F(c3) -> U(c4)
U(c4) -> Y(c4)
P1(c3) & K(c3) -> G(c4)
G(c4) -> P0(c4)
P1(c4) & X(c4) -> S(c5)
S(c5) -> P1(c5)
P0(c4) & G(c4) -> B(c5)
B(c5) -> P0(c5)
Y(c4) & U(c4) -> R(c5)
R(c5) -> Z(c5)
Z(c4) & D(c4) -> J(c5)
J(c5) -> Y(c5)
Z(c5) & R(c5) -> Q(c6)
Q(c6) -> P1(c6)
Y(c5) & J(c5) -> W(c6)
W(c6) -> Z(c6)
P1(c5) & S(c5) -> U(c6)
U(c6) -> P0(c6)
P0(c5) & B(c5) -> V(c6)
V(c6) -> Y(c6)
Y(c6) & V(c6) -> U(c7)
U(c7) -> P1(c7)
P0(c6) & U(c6) -> L(c7)
L(c7) -> P0(c7)
P1(c6) & Q(c6) -> M(c7)
M(c7) -> Y(c7)
Z(c6) & W(c6) -> W(c7)
W(c7) -> Z(c7)
P1(c7) & U(c7) -> Q(c8)
Q(c8) -> Z(c8)
Z(c7) & W(c7) -> L(c8)
L(c8) -> Y(c8)
Y(c7) & M(c7) -> I(c8)
I(c8) -> P0(c8)
P0(c7) & L(c7) -> C(c8)
C(c8) -> P1(c8)
Z(c8) & Q(c8) -> J(c9)
J(c9) -> Y(c9)
Y(c8) & L(c8) -> K(c9)
K(c9) -> P1(c9)
P1(c8) & C(c8) -> S(c9)
S(c9) -> P0(c9)
P0(c8) & I(c8) -> F(c9)
F(c9) -> Z(c9)
P0(c9) & S(c9) -> S(c10)
S(c10) -> P0(c10)
Z(c9) & F(c9) -> D(c10)
D(c10) -> Y(c10)
Y(c9) & J(c9) -> X(c10)
X(c10) -> Z(c10)
P1(c9) & K(c9) -> I(c10)
I(c10) -> P1(c10)
P0(c10) & S(c10) -> W(c11)
W(c11) -> P1(c11)
Z(c10) & X(c10) -> A(c11)
A(c11) -> Z(c11)
P1(c10) & I(c10) -> C(c11)
C(c11) -> Y(c11)
Y(c10) & D(c10) -> N(c11)
N(c11) -> P0(c11)
Z(c11) & A(c11) -> J(c12)
J(c12) -> Z(c12)
P1(c11) & W(c11) -> A(c12)
A(c12) -> P1(c12)
P0(c11) & N(c11) -> P(c12)
P(c12) -> P0(c12)
Y(c11) & C(c11) -> Q(c12)
Q(c12) -> Y(c12)
Y(c12) & Q(c12) -> J(c13)
J(c13) -> P0(c13)
P0(c12) & P(c12) -> S(c13)
S(c13) -> Y(c13)
P1(c12) & A(c12) -> U(c13)
U(c13) -> P1(c13)
Z(c12) & J(c12) -> Q(c13)
Q(c13) -> Z(c13)
P1(c13) & U(c13) -> S(c14)
S(c14) -> P0(c14)
P0(c13) & J(c13) -> T(c14)
T(c14) -> P1(c14)
Z(c13) & Q(c13) -> V(c14)
V(c14) -> Y(c14)
Y(c13) & S(c13) -> J(c14)
J(c14) -> Z(c14)
P1(c14) & T(c14) -> J(c15)
J(c15) -> P1(c15)
Z(c14) & J(c14) -> K(c15)
K(c15) -> Y(c15)
Y(c14) & V(c14) -> I(c15)
I(c15) -> P0(c15)
P0(c14) & S(c14) -> S(c15)
S(c15) -> Z(c15)
P1(c15) & J(c15) -> A(c16)
A(c16) -> P0(c16)
Y(c15) & K(c15) -> R(c16)
R(c16) -> Y(c16)
Z(c15) & S(c15) -> L(c16)
L(c16) -> Z(c16)
P0(c15) & I(c15) -> O(c16)
O(c16) -> P1(c16)
P0(c16) & A(c16) -> X(c17)
X(c17) -> Z(c17)
Y(c16) & R(c16) -> G(c17)
G(c17) -> P1(c17)
P1(c16) & O(c16) -> I(c17)
I(c17) -> Y(c17)
Z(c16) & L(c16) -> R(c17)
R(c17) -> P0(c17)
P1(c17) & G(c17) -> J(c18)
J(c18) -> P1(c18)
Z(c17) & X(c17) -> A(c18)
A(c18) -> Y(c18)
Y(c17) & I(c17) -> I(c18)
I(c18) -> Z(c18)
P0(c17) & R(c17) -> X(c18)
X(c18) -> P0(c18)
Y(c18) & A(c18) -> M(c19)
M(c19) -> Z(c19)
Z(c18) & I(c18) -> K(c19)
K(c19) -> P1(c19)
P1(c18) & J(c18) -> Q(c19)
Q(c19) -> Y(c19)
P0(c18) & X(c18) -> J(c19)
J(c19) -> P0(c19)
Y(c19) & Q(c19) -> B(c20)
B(c20) -> Y(c20)
P1(c19) & K(c19) -> R(c20)
R(c20) -> P0(c20)
P0(c19) & J(c19) -> P(c20)
P(c20) -> Z(c20)
Z(c19) & M(c19) -> X(c20)
X(c20) -> P1(c20)
Y(c20) & B(c20) -> C(c21)
C(c21) -> P0(c21)
Z(c20) & P(c20) -> I(c21)
I(c21) -> Y(c21)
P1(c20) & X(c20) -> G(c21)
G(c21) -> Z(c21)
P0(c20) & R(c20) -> A(c21)
A(c21) -> P1(c21)
P1(c21) & A(c21) -> W(c22)
W(c22) -> Z(c22)
Y(c21) & I(c21) -> O(c22)
O(c22) -> P1(c22)
P0(c21) & C(c21) -> S(c22)
S(c22) -> P0(c22)
Z(c21) & G(c21) -> R(c22)
R(c22) -> Y(c22)
P0(c22) & S(c22) -> S(c23)
S(c23) -> P1(c23)
Z(c22) & W(c22) -> C(c23)
C(c23) -> Y(c23)
P1(c22) & O(c22) -> W(c23)
W(c23) -> Z(c23)
Y(c22) & R(c22) -> B(c23)
B(c23) -> P0(c23)
</premises>
<proof>
E(c0) ; R
Y(c0) ; R
Q(c1) ; ->E
Z(c1) ; ->E
O(c2) ; ->E
P0(c2) ; ->E
H(c3) ; ->E
Y(c3) ; ->E
X(c4) ; ->E
P1(c4) ; ->E
S(c5) ; ->E
P1(c5) ; ->E
U(c6) ; ->E
P0(c6) ; ->E
L(c7) ; ->E
P0(c7) ; ->E
C(c8) ; ->E
P1(c8) ; ->E
S(c9) ; ->E
P0(c9) ; ->E
S(c10) ; ->E
P0(c10) ; ->E
W(c11) ; ->E
P1(c11) ; ->E
A(c12) ; ->E
P1(c12) ; ->E
U(c13) ; ->E
P1(c13) ; ->E
S(c14) ; ->E
P0(c14) ; ->E
S(c15) ; ->E
Z(c15) ; ->E
L(c16) ; ->E
Z(c16) ; ->E
R(c17) ; ->E
P0(c17) ; ->E
X(c18) ; ->E
P0(c18) ; ->E
J(c19) ; ->E
P0(c19) ; ->E
P(c20) ; ->E
Z(c20) ; ->E
I(c21) ; ->E
Y(c21) ; ->E
O(c22) ; ->E
P1(c22) ; ->E
W(c23) ; ->E
</proof>
<conclusion>
W(c23)
</conclusion>
</formal>
<answer>
hazel
</answer><|endoftext|><question>
1. c0 is harbor.
2. c0 is north.
3. If c0 is north and c0 is harbor, then c1 is amber.
4. If c1 is amber, then c1 is north.
5. If c0 is south and c0 is harbor, then c1 is orchid.
6. If c1 is orchid, then c1 is east.
7. If c0 is west and c0 is harbor, then c1 is elm.
8. If c1 is elm, then c1 is south.
9. If c0 is east and c0 is harbor, then c1 is poppy.
10. If c1 is poppy, then c1 is west.
11. If c1 is west and c1 is poppy, then c2 is orchid.
12. If c2 is orchid, then c2 is south.
13. If c1 is north and c1 is amber, then c2 is amber.
14. If c2 is amber, then c2 is west.
15. If c1 is south and c1 is elm, then c2 is elm.
16. If c2 is elm, then c2 is east.
17. If c1 is east and c1 is orchid, then c2 is poppy.
18. If c2 is poppy, then c2 is north.
19. If c2 is east and c2 is elm, then c3 is poppy.
20. If c3 is poppy, then c3 is north.
21. If c2 is south and c2 is orchid, then c3 is amber.
22. If c3 is amber, then c3 is south.
23. If c2 is west and c2 is amber, then c3 is elm.
24. If c3 is elm, then c3 is west.
25. If c2 is north and c2 is poppy, then c3 is orchid.
26. If c3 is orchid, then c3 is east.
Which state applies to c3?
</question>

<formal>
<constants>
c0 = c0
c1 = c1
c2 = c2
c3 = c3
</constants>
<predicates>
Ax: x is poppy
Bx: x is harbor
Cx: x is amber
Dx: x is orchid
Ex: x is elm
Fx: x is north
Gx: x is south
Hx: x is east
Ix: x is west
</predicates>
<premises>
B(c0)
F(c0)
F(c0) & B(c0) -> C(c1)
C(c1) -> F(c1)
G(c0) & B(c0) -> D(c1)
D(c1) -> H(c1)
I(c0) & B(c0) -> E(c1)
E(c1) -> G(c1)
H(c0) & B(c0) -> A(c1)
A(c1) -> I(c1)
I(c1) & A(c1) -> D(c2)
D(c2) -> G(c2)
F(c1) & C(c1) -> C(c2)
C(c2) -> I(c2)
G(c1) & E(c1) -> E(c2)
E(c2) -> H(c2)
H(c1) & D(c1) -> A(c2)
A(c2) -> F(c2)
H(c2) & E(c2) -> A(c3)
A(c3) -> F(c3)
G(c2) & D(c2) -> C(c3)
C(c3) -> G(c3)
I(c2) & C(c2) -> E(c3)
E(c3) -> I(c3)
F(c2) & A(c2) -> D(c3)
D(c3) -> H(c3)
</premises>
<proof>
B(c0) ; R
F(c0) ; R
C(c1) ; ->E
F(c1) ; ->E
C(c2) ; ->E
I(c2) ; ->E
E(c3) ; ->E
</proof>
<conclusion>
E(c3)
</conclusion>
</formal>
<answer>
elm
</answer><|endoftext|>
```

## Window 1 (2 documents, 374 pad tokens)
```
<question>
1. c0 is hazel.
2. c0 is east.
3. If c0 is east and c0 is hazel, then c1 is slate.
4. If c1 is slate, then c1 is west.
5. If c0 is north and c0 is hazel, then c1 is juniper.
6. If c1 is juniper, then c1 is east.
7. If c0 is west and c0 is hazel, then c1 is cedar.
8. If c1 is cedar, then c1 is south.
9. If c0 is south and c0 is hazel, then c1 is cobalt.
10. If c1 is cobalt, then c1 is north.
11. If c1 is east and c1 is juniper, then c2 is cobalt.
12. If c2 is cobalt, then c2 is east.
13. If c1 is north and c1 is cobalt, then c2 is violet.
14. If c2 is violet, then c2 is north.
15. If c1 is west and c1 is slate, then c2 is elm.
16. If c2 is elm, then c2 is west.
17. If c1 is south and c1 is cedar, then c2 is meadow.
18. If c2 is meadow, then c2 is south.
19. If c2 is north and c2 is violet, then c3 is slate.
20. If c3 is slate, then c3 is north.
21. If c2 is east and c2 is cobalt, then c3 is cedar.
22. If c3 is cedar, then c3 is south.
23. If c2 is south and c2 is meadow, then c3 is elm.
24. If c3 is elm, then c3 is east.
25. If c2 is west and c2 is elm, then c3 is olive.
26. If c3 is olive, then c3 is west.
27. If c3 is west and c3 is olive, then c4 is cedar.
28. If c4 is cedar, then c4 is west.
29. If c3 is east and c3 is elm, then c4 is amber.
30. If c4 is amber, then c4 is north.
31. If c3 is north and c3 is slate, then c4 is meadow.
32. If c4 is meadow, then c4 is east.
33. If c3 is south and c3 is cedar, then c4 is juniper.
34. If c4 is juniper, then c4 is south.
35. If c4 is east and c4 is meadow, then c5 is elm.
36. If c5 is elm, then c5 is north.
37. If c4 is west and c4 is cedar, then c5 is maple.
38. If c5 is maple, then c5 is east.
39. If c4 is north and c4 is amber, then c5 is juniper.
40. If c5 is juniper, then c5 is west.
41. If c4 is south and c4 is juniper, then c5 is amber.
42. If c5 is amber, then c5 is south.
43. If c5 is north and c5 is elm, then c6 is juniper.
44. If c6 is juniper, then c6 is east.
45. If c5 is east and c5 is maple, then c6 is birch.
46. If c6 is birch, then c6 is west.
47. If c5 is south and c5 is amber, then c6 is olive.
48. If c6 is olive, then c6 is south.
49. If c5 is west and c5 is juniper, then c6 is elm.
50. If c6 is elm, then c6 is north.
51. If c6 is south and c6 is olive, then c7 is olive.
52. If c7 is olive, then c7 is south.
53. If c6 is west and c6 is birch, then c7 is meadow.
54. If c7 is meadow, then c7 is west.
55. If c6 is east and c6 is juniper, then c7 is teal.
56. If c7 is teal, then c7 is north.
57. If c6 is north and c6 is elm, then c7 is juniper.
58. If c7 is juniper, then c7 is east.
59. If c7 is east and c7 is juniper, then c8 is violet.
60. If c8 is violet, then c8 is south.
61. If c7 is west and c7 is meadow, then c8 is olive.
62. If c8 is olive, then c8 is north.
63. If c7 is north and c7 is teal, then c8 is ivory.
64. If c8 is ivory, then c8 is west.
65. If c7 is south and c7 is olive, then c8 is slate.
66. If c8 is slate, then c8 is east.
67. If c8 is east and c8 is slate, then c9 is ivory.
68. If c9 is ivory, then c9 is north.
69. If c8 is west and c8 is ivory, then c9 is maple.
70. If c9 is maple, then c9 is west.
71. If c8 is north and c8 is olive, then c9 is violet.
72. If c9 is violet, then c9 is south.
73. If c8 is south and c8 is violet, then c9 is cobalt.
74. If c9 is cobalt, then c9 is east.
75. If c9 is east and c9 is cobalt, then c10 is juniper.
76. If c10 is juniper, then c10 is north.
77. If c9 is south and c9 is violet, then c10 is ivory.
78. If c10 is ivory, then c10 is west.
79. If c9 is north and c9 is ivory, then c10 is cobalt.
80. If c10 is cobalt, then c10 is south.
81. If c9 is west and c9 is maple, then c10 is amber.
82. If c10 is amber, then c10 is east.
83. If c10 is west and c10 is ivory, then c11 is cobalt.
84. If c11 is cobalt, then c11 is north.
85. If c10 is north and c10 is juniper, then c11 is cedar.
86. If c11 is cedar, then c11 is west.
87. If c10 is east and c10 is amber, then c11 is juniper.
88. If c11 is juniper, then c11 is south.
89. If c10 is south and c10 is cobalt, then c11 is teal.
90. If c11 is teal, then c11 is east.
91. If c11 is west and c11 is cedar, then c12 is elm.
92. If c12 is elm, then c12 is west.
93. If c11 is south and c11 is juniper, then c12 is ivory.
94. If c12 is ivory, then c12 is north.
95. If c11 is north and c11 is cobalt, then c12 is meadow.
96. If c12 is meadow, then c12 is south.
97. If c11 is east and c11 is teal, then c12 is teal.
98. If c12 is teal, then c12 is east.
99. If c12 is east and c12 is teal, then c13 is maple.
100. If c13 is maple, then c13 is south.
101. If c12 is north and c12 is ivory, then c13 is ivory.
102. If c13 is ivory, then c13 is east.
103. If c12 is west and c12 is elm, then c13 is slate.
104. If c13 is slate, then c13 is north.
105. If c12 is south and c12 is meadow, then c13 is juniper.
106. If c13 is juniper, then c13 is west.
Which state applies to c13?
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
</constants>
<predicates>
Ax: x is elm
Bx: x is violet
Cx: x is amber
Dx: x is hazel
Ex: x is cobalt
Fx: x is maple
Gx: x is meadow
Hx: x is birch
Ix: x is cedar
Jx: x is juniper
Kx: x is teal
Lx: x is slate
Mx: x is ivory
Nx: x is olive
Ox: x is north
Px: x is south
Qx: x is east
Rx: x is west
</predicates>
<premises>
D(c0)
Q(c0)
Q(c0) & D(c0) -> L(c1)
L(c1) -> R(c1)
O(c0) & D(c0) -> J(c1)
J(c1) -> Q(c1)
R(c0) & D(c0) -> I(c1)
I(c1) -> P(c1)
P(c0) & D(c0) -> E(c1)
E(c1) -> O(c1)
Q(c1) & J(c1) -> E(c2)
E(c2) -> Q(c2)
O(c1) & E(c1) -> B(c2)
B(c2) -> O(c2)
R(c1) & L(c1) -> A(c2)
A(c2) -> R(c2)
P(c1) & I(c1) -> G(c2)
G(c2) -> P(c2)
O(c2) & B(c2) -> L(c3)
L(c3) -> O(c3)
Q(c2) & E(c2) -> I(c3)
I(c3) -> P(c3)
P(c2) & G(c2) -> A(c3)
A(c3) -> Q(c3)
R(c2) & A(c2) -> N(c3)
N(c3) -> R(c3)
R(c3) & N(c3) -> I(c4)
I(c4) -> R(c4)
Q(c3) & A(c3) -> C(c4)
C(c4) -> O(c4)
O(c3) & L(c3) -> G(c4)
G(c4) -> Q(c4)
P(c3) & I(c3) -> J(c4)
J(c4) -> P(c4)
Q(c4) & G(c4) -> A(c5)
A(c5) -> O(c5)
R(c4) & I(c4) -> F(c5)
F(c5) -> Q(c5)
O(c4) & C(c4) -> J(c5)
J(c5) -> R(c5)
P(c4) & J(c4) -> C(c5)
C(c5) -> P(c5)
O(c5) & A(c5) -> J(c6)
J(c6) -> Q(c6)
Q(c5) & F(c5) -> H(c6)
H(c6) -> R(c6)
P(c5) & C(c5) -> N(c6)
N(c6) -> P(c6)
R(c5) & J(c5) -> A(c6)
A(c6) -> O(c6)
P(c6) & N(c6) -> N(c7)
N(c7) -> P(c7)
R(c6) & H(c6) -> G(c7)
G(c7) -> R(c7)
Q(c6) & J(c6) -> K(c7)
K(c7) -> O(c7)
O(c6) & A(c6) -> J(c7)
J(c7) -> Q(c7)
Q(c7) & J(c7) -> B(c8)
B(c8) -> P(c8)
R(c7) & G(c7) -> N(c8)
N(c8) -> O(c8)
O(c7) & K(c7) -> M(c8)
M(c8) -> R(c8)
P(c7) & N(c7) -> L(c8)
L(c8) -> Q(c8)
Q(c8) & L(c8) -> M(c9)
M(c9) -> O(c9)
R(c8) & M(c8) -> F(c9)
F(c9) -> R(c9)
O(c8) & N(c8) -> B(c9)
B(c9) -> P(c9)
P(c8) & B(c8) -> E(c9)
E(c9) -> Q(c9)
Q(c9) & E(c9) -> J(c10)
J(c10) -> O(c10)
P(c9) & B(c9) -> M(c10)
M(c10) -> R(c10)
O(c9) & M(c9) -> E(c10)
E(c10) -> P(c10)
R(c9) & F(c9) -> C(c10)
C(c10) -> Q(c10)
R(c10) & M(c10) -> E(c11)
E(c11) -> O(c11)
O(c10) & J(c10) -> I(c11)
I(c11) -> R(c11)
Q(c10) & C(c10) -> J(c11)
J(c11) -> P(c11)
P(c10) & E(c10) -> K(c11)
K(c11) -> Q(c11)
R(c11) & I(c11) -> A(c12)
A(c12) -> R(c12)
P(c11) & J(c11) -> M(c12)
M(c12) -> O(c12)
O(c11) & E(c11) -> G(c12)
G(c12) -> P(c12)
Q(c11) & K(c11) -> K(c12)
K(c12) -> Q(c12)
Q(c12) & K(c12) -> F(c13)
F(c13) -> P(c13)
O(c12) & M(c12) -> M(c13)
M(c13) -> Q(c13)
R(c12) & A(c12) -> L(c13)
L(c13) -> O(c13)
P(c12) & G(c12) -> J(c13)
J(c13) -> R(c13)
</premises>
<proof>
D(c0) ; R
Q(c0) ; R
L(c1) ; ->E
R(c1) ; ->E
A(c2) ; ->E
R(c2) ; ->E
N(c3) ; ->E
R(c3) ; ->E
I(c4) ; ->E
R(c4) ; ->E
F(c5) ; ->E
Q(c5) ; ->E
H(c6) ; ->E
R(c6) ; ->E
G(c7) ; ->E
R(c7) ; ->E
N(c8) ; ->E
O(c8) ; ->E
B(c9) ; ->E
P(c9) ; ->E
M(c10) ; ->E
R(c10) ; ->E
E(c11) ; ->E
O(c11) ; ->E
G(c12) ; ->E
P(c12) ; ->E
J(c13) ; ->E
</proof>
<conclusion>
J(c13)
</conclusion>
</formal>
<answer>
juniper
</answer><|endoftext|><question>
1. c0 is maple.
2. c0 is north.
3. If c0 is east and c0 is maple, then c1 is birch.
4. If c1 is birch, then c1 is east.
5. If c0 is south and c0 is maple, then c1 is juniper.
6. If c1 is juniper, then c1 is north.
7. If c0 is north and c0 is maple, then c1 is amber.
8. If c1 is amber, then c1 is west.
9. If c0 is west and c0 is maple, then c1 is coral.
10. If c1 is coral, then c1 is south.
11. If c1 is east and c1 is birch, then c2 is poppy.
12. If c2 is poppy, then c2 is south.
13. If c1 is north and c1 is juniper, then c2 is elm.
14. If c2 is elm, then c2 is east.
15. If c1 is west and c1 is amber, then c2 is amber.
16. If c2 is amber, then c2 is west.
17. If c1 is south and c1 is coral, then c2 is lime.
18. If c2 is lime, then c2 is north.
19. If c2 is west and c2 is amber, then c3 is birch.
20. If c3 is birch, then c3 is south.
21. If c2 is south and c2 is poppy, then c3 is pearl.
22. If c3 is pearl, then c3 is north.
23. If c2 is north and c2 is lime, then c3 is laurel.
24. If c3 is laurel, then c3 is east.
25. If c2 is east and c2 is elm, then c3 is juniper.
26. If c3 is juniper, then c3 is west.
27. If c3 is south and c3 is birch, then c4 is lime.
28. If c4 is lime, then c4 is north.
29. If c3 is east and c3 is laurel, then c4 is coral.
30. If c4 is coral, then c4 is south.
31. If c3 is west and c3 is juniper, then c4 is hazel.
32. If c4 is hazel, then c4 is east.
33. If c3 is north and c3 is pearl, then c4 is juniper.
34. If c4 is juniper, then c4 is west.
35. If c4 is east and c4 is hazel, then c5 is juniper.
36. If c5 is juniper, then c5 is south.
37. If c4 is south and c4 is coral, then c5 is pearl.
38. If c5 is pearl, then c5 is west.
39. If c4 is north and c4 is lime, then c5 is elm.
40. If c5 is elm, then c5 is north.
41. If c4 is west and c4 is juniper, then c5 is olive.
42. If c5 is olive, then c5 is east.
43. If c5 is west and c5 is pearl, then c6 is ruby.
44. If c6 is ruby, then c6 is east.
45. If c5 is east and c5 is olive, then c6 is elm.
46. If c6 is elm, then c6 is west.
47. If c5 is south and c5 is juniper, then c6 is laurel.
48. If c6 is laurel, then c6 is north.
49. If c5 is north and c5 is elm, then c6 is meadow.
50. If c6 is meadow, then c6 is south.
51. If c6 is west and c6 is elm, then c7 is ruby.
52. If c7 is ruby, then c7 is west.
53. If c6 is east and c6 is ruby, then c7 is amber.
54. If c7 is amber, then c7 is south.
55. If c6 is north and c6 is laurel, then c7 is pearl.
56. If c7 is pearl, then c7 is east.
57. If c6 is south and c6 is meadow, then c7 is coral.
58. If c7 is coral, then c7 is north.
59. If c7 is south and c7 is amber, then c8 is olive.
60. If c8 is olive, then c8 is south.
61. If c7 is west and c7 is ruby, then c8 is poppy.
62. If c8 is poppy, then c8 is north.
63. If c7 is east and c7 is pearl, then c8 is amber.
64. If c8 is amber, then c8 is east.
65. If c7 is north and c7 is coral, then c8 is meadow.
66. If c8 is meadow, then c8 is west.
67. If c8 is east and c8 is amber, then c9 is olive.
68. If c9 is olive, then c9 is east.
69. If c8 is north and c8 is poppy, then c9 is lime.
70. If c9 is lime, then c9 is north.
71. If c8 is west and c8 is meadow, then c9 is birch.
72. If c9 is birch, then c9 is west.
73. If c8 is south and c8 is olive, then c9 is laurel.
74. If c9 is laurel, then c9 is south.
75. If c9 is south and c9 is laurel, then c10 is meadow.
76. If c10 is meadow, then c10 is east.
77. If c9 is west and c9 is birch, then c10 is hazel.
78. If c10 is hazel, then c10 is north.
79. If c9 is north and c9 is lime, then c10 is laurel.
80. If c10 is laurel, then c10 is south.
81. If c9 is east and c9 is olive, then c10 is olive.
82. If c10 is olive, then c10 is west.
83. If c10 is east and c10 is meadow, then c11 is teal.
84. If c11 is teal, then c11 is east.
85. If c10 is north and c10 is hazel, then c11 is ruby.
86. If c11 is ruby, then c11 is west.
87. If c10 is west and c10 is olive, then c11 is meadow.
88. If c11 is meadow, then c11 is north.
89. If c10 is south and c10 is laurel, then c11 is hazel.
90. If c11 is hazel, then c11 is south.
91. If c11 is west and c11 is ruby, then c12 is ruby.
92. If c12 is ruby, then c12 is west.
93. If c11 is south and c11 is hazel, then c12 is pearl.
94. If c12 is pearl, then c12 is south.
95. If c11 is east and c11 is teal, then c12 is lime.
96. If c12 is lime, then c12 is east.
97. If c11 is north and c11 is meadow, then c12 is elm.
98. If c12 is elm, then c12 is north.
99. If c12 is west and c12 is ruby, then c13 is birch.
100. If c13 is birch, then c13 is north.
101. If c12 is east and c12 is lime, then c13 is coral.
102. If c13 is coral, then c13 is south.
103. If c12 is north and c12 is elm, then c13 is lime.
104. If c13 is lime, then c13 is west.
105. If c12 is south and c12 is pearl, then c13 is elm.
106. If c13 is elm, then c13 is east.
107. If c13 is north and c13 is birch, then c14 is teal.
108. If c14 is teal, then c14 is south.
109. If c13 is west and c13 is lime, then c14 is juniper.
110. If c14 is juniper, then c14 is east.
111. If c13 is south and c13 is coral, then c14 is lime.
112. If c14 is lime, then c14 is north.
113. If c13 is east and c13 is elm, then c14 is elm.
114. If c14 is elm, then c14 is west.
Which state applies to c14?
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
</constants>
<predicates>
Ax: x is coral
Bx: x is amber
Cx: x is ruby
Dx: x is elm
Ex: x is hazel
Fx: x is laurel
Gx: x is birch
Hx: x is lime
Ix: x is pearl
Jx: x is juniper
Kx: x is olive
Lx: x is teal
Mx: x is maple
Nx: x is meadow
Ox: x is poppy
Px: x is north
Qx: x is south
Rx: x is east
Sx: x is west
</predicates>
<premises>
M(c0)
P(c0)
R(c0) & M(c0) -> G(c1)
G(c1) -> R(c1)
Q(c0) & M(c0) -> J(c1)
J(c1) -> P(c1)
P(c0) & M(c0) -> B(c1)
B(c1) -> S(c1)
S(c0) & M(c0) -> A(c1)
A(c1) -> Q(c1)
R(c1) & G(c1) -> O(c2)
O(c2) -> Q(c2)
P(c1) & J(c1) -> D(c2)
D(c2) -> R(c2)
S(c1) & B(c1) -> B(c2)
B(c2) -> S(c2)
Q(c1) & A(c1) -> H(c2)
H(c2) -> P(c2)
S(c2) & B(c2) -> G(c3)
G(c3) -> Q(c3)
Q(c2) & O(c2) -> I(c3)
I(c3) -> P(c3)
P(c2) & H(c2) -> F(c3)
F(c3) -> R(c3)
R(c2) & D(c2) -> J(c3)
J(c3) -> S(c3)
Q(c3) & G(c3) -> H(c4)
H(c4) -> P(c4)
R(c3) & F(c3) -> A(c4)
A(c4) -> Q(c4)
S(c3) & J(c3) -> E(c4)
E(c4) -> R(c4)
P(c3) & I(c3) -> J(c4)
J(c4) -> S(c4)
R(c4) & E(c4) -> J(c5)
J(c5) -> Q(c5)
Q(c4) & A(c4) -> I(c5)
I(c5) -> S(c5)
P(c4) & H(c4) -> D(c5)
D(c5) -> P(c5)
S(c4) & J(c4) -> K(c5)
K(c5) -> R(c5)
S(c5) & I(c5) -> C(c6)
C(c6) -> R(c6)
R(c5) & K(c5) -> D(c6)
D(c6) -> S(c6)
Q(c5) & J(c5) -> F(c6)
F(c6) -> P(c6)
P(c5) & D(c5) -> N(c6)
N(c6) -> Q(c6)
S(c6) & D(c6) -> C(c7)
C(c7) -> S(c7)
R(c6) & C(c6) -> B(c7)
B(c7) -> Q(c7)
P(c6) & F(c6) -> I(c7)
I(c7) -> R(c7)
Q(c6) & N(c6) -> A(c7)
A(c7) -> P(c7)
Q(c7) & B(c7) -> K(c8)
K(c8) -> Q(c8)
S(c7) & C(c7) -> O(c8)
O(c8) -> P(c8)
R(c7) & I(c7) -> B(c8)
B(c8) -> R(c8)
P(c7) & A(c7) -> N(c8)
N(c8) -> S(c8)
R(c8) & B(c8) -> K(c9)
K(c9) -> R(c9)
P(c8) & O(c8) -> H(c9)
H(c9) -> P(c9)
S(c8) & N(c8) -> G(c9)
G(c9) -> S(c9)
Q(c8) & K(c8) -> F(c9)
F(c9) -> Q(c9)
Q(c9) & F(c9) -> N(c10)
N(c10) -> R(c10)
S(c9) & G(c9) -> E(c10)
E(c10) -> P(c10)
P(c9) & H(c9) -> F(c10)
F(c10) -> Q(c10)
R(c9) & K(c9) -> K(c10)
K(c10) -> S(c10)
R(c10) & N(c10) -> L(c11)
L(c11) -> R(c11)
P(c10) & E(c10) -> C(c11)
C(c11) -> S(c11)
S(c10) & K(c10) -> N(c11)
N(c11) -> P(c11)
Q(c10) & F(c10) -> E(c11)
E(c11) -> Q(c11)
S(c11) & C(c11) -> C(c12)
C(c12) -> S(c12)
Q(c11) & E(c11) -> I(c12)
I(c12) -> Q(c12)
R(c11) & L(c11) -> H(c12)
H(c12) -> R(c12)
P(c11) & N(c11) -> D(c12)
D(c12) -> P(c12)
S(c12) & C(c12) -> G(c13)
G(c13) -> P(c13)
R(c12) & H(c12) -> A(c13)
A(c13) -> Q(c13)
P(c12) & D(c12) -> H(c13)
H(c13) -> S(c13)
Q(c12) & I(c12) -> D(c13)
D(c13) -> R(c13)
P(c13) & G(c13) -> L(c14)
L(c14) -> Q(c14)
S(c13) & H(c13) -> J(c14)
J(c14) -> R(c14)
Q(c13) & A(c13) -> H(c14)
H(c14) -> P(c14)
R(c13) & D(c13) -> D(c14)
D(c14) -> S(c14)
</premises>
<proof>
M(c0) ; R
P(c0) ; R
B(c1) ; ->E
S(c1) ; ->E
B(c2) ; ->E
S(c2) ; ->E
G(c3) ; ->E
Q(c3) ; ->E
H(c4) ; ->E
P(c4) ; ->E
D(c5) ; ->E
P(c5) ; ->E
N(c6) ; ->E
Q(c6) ; ->E
A(c7) ; ->E
P(c7) ; ->E
N(c8) ; ->E
S(c8) ; ->E
G(c9) ; ->E
S(c9) ; ->E
E(c10) ; ->E
P(c10) ; ->E
C(c11) ; ->E
S(c11) ; ->E
C(c12) ; ->E
S(c12) ; ->E
G(c13) ; ->E
P(c13) ; ->E
L(c14) ; ->E
</proof>
<conclusion>
L(c14)
</conclusion>
</formal>
<answer>
teal
</answer><|endoftext|>
```

## Window 2 (1 documents, 534 pad tokens)
```
<question>
1. c0 is violet.
2. c0 is west.
3. If c0 is south and c0 is violet, then c1 is laurel.
4. If c1 is laurel, then c1 is south.
5. If c0 is west and c0 is violet, then c1 is ivory.
6. If c1 is ivory, then c1 is east.
7. If c0 is east and c0 is violet, then c1 is orchid.
8. If c1 is orchid, then c1 is west.
9. If c0 is north and c0 is violet, then c1 is elm.
10. If c1 is elm, then c1 is north.
11. If c1 is south and c1 is laurel, then c2 is maple.
12. If c2 is maple, then c2 is south.
13. If c1 is west and c1 is orchid, then c2 is harbor.
14. If c2 is harbor, then c2 is north.
15. If c1 is east and c1 is ivory, then c2 is poppy.
16. If c2 is poppy, then c2 is east.
17. If c1 is north and c1 is elm, then c2 is elm.
18. If c2 is elm, then c2 is west.
19. If c2 is south and c2 is maple, then c3 is maple.
20. If c3 is maple, then c3 is south.
21. If c2 is west and c2 is elm, then c3 is pearl.
22. If c3 is pearl, then c3 is east.
23. If c2 is east and c2 is poppy, then c3 is slate.
24. If c3 is slate, then c3 is west.
25. If c2 is north and c2 is harbor, then c3 is coral.
26. If c3 is coral, then c3 is north.
27. If c3 is north and c3 is coral, then c4 is pearl.
28. If c4 is pearl, then c4 is south.
29. If c3 is west and c3 is slate, then c4 is poppy.
30. If c4 is poppy, then c4 is west.
31. If c3 is south and c3 is maple, then c4 is teal.
32. If c4 is teal, then c4 is north.
33. If c3 is east and c3 is pearl, then c4 is elm.
34. If c4 is elm, then c4 is east.
35. If c4 is east and c4 is elm, then c5 is teal.
36. If c5 is teal, then c5 is east.
37. If c4 is south and c4 is pearl, then c5 is maple.
38. If c5 is maple, then c5 is north.
39. If c4 is north and c4 is teal, then c5 is ruby.
40. If c5 is ruby, then c5 is west.
41. If c4 is west and c4 is poppy, then c5 is coral.
42. If c5 is coral, then c5 is south.
43. If c5 is north and c5 is maple, then c6 is cedar.
44. If c6 is cedar, then c6 is west.
45. If c5 is east and c5 is teal, then c6 is ivory.
46. If c6 is ivory, then c6 is south.
47. If c5 is west and c5 is ruby, then c6 is orchid.
48. If c6 is orchid, then c6 is north.
49. If c5 is south and c5 is coral, then c6 is ruby.
50. If c6 is ruby, then c6 is east.
51. If c6 is south and c6 is ivory, then c7 is lime.
52. If c7 is lime, then c7 is south.
53. If c6 is north and c6 is orchid, then c7 is ruby.
54. If c7 is ruby, then c7 is west.
55. If c6 is west and c6 is cedar, then c7 is granite.
56. If c7 is granite, then c7 is north.
57. If c6 is east and c6 is ruby, then c7 is cedar.
58. If c7 is cedar, then c7 is east.
59. If c7 is north and c7 is granite, then c8 is ruby.
60. If c8 is ruby, then c8 is east.
61. If c7 is east and c7 is cedar, then c8 is elm.
62. If c8 is elm, then c8 is north.
63. If c7 is west and c7 is ruby, then c8 is ivory.
64. If c8 is ivory, then c8 is west.
65. If c7 is south and c7 is lime, then c8 is orchid.
66. If c8 is orchid, then c8 is south.
67. If c8 is west and c8 is ivory, then c9 is maple.
68. If c9 is maple, then c9 is east.
69. If c8 is north and c8 is elm, then c9 is slate.
70. If c9 is slate, then c9 is south.
71. If c8 is east and c8 is ruby, then c9 is ivory.
72. If c9 is ivory, then c9 is north.
73. If c8 is south and c8 is orchid, then c9 is cobalt.
74. If c9 is cobalt, then c9 is west.
75. If c9 is east and c9 is maple, then c10 is coral.
76. If c10 is coral, then c10 is north.
77. If c9 is south and c9 is slate, then c10 is hazel.
78. If c10 is hazel, then c10 is south.
79. If c9 is north and c9 is ivory, then c10 is maple.
80. If c10 is maple, then c10 is west.
81. If c9 is west and c9 is cobalt, then c10 is laurel.
82. If c10 is laurel, then c10 is east.
83. If c10 is east and c10 is laurel, then c11 is maple.
84. If c11 is maple, then c11 is south.
85. If c10 is west and c10 is maple, then c11 is meadow.
86. If c11 is meadow, then c11 is east.
87. If c10 is north and c10 is coral, then c11 is slate.
88. If c11 is slate, then c11 is north.
89. If c10 is south and c10 is hazel, then c11 is hazel.
90. If c11 is hazel, then c11 is west.
91. If c11 is north and c11 is slate, then c12 is harbor.
92. If c12 is harbor, then c12 is east.
93. If c11 is west and c11 is hazel, then c12 is lime.
94. If c12 is lime, then c12 is south.
95. If c11 is south and c11 is maple, then c12 is cobalt.
96. If c12 is cobalt, then c12 is north.
97. If c11 is east and c11 is meadow, then c12 is ruby.
98. If c12 is ruby, then c12 is west.
99. If c12 is north and c12 is cobalt, then c13 is meadow.
100. If c13 is meadow, then c13 is south.
101. If c12 is south and c12 is lime, then c13 is amber.
102. If c13 is amber, then c13 is east.
103. If c12 is west and c12 is ruby, then c13 is willow.
104. If c13 is willow, then c13 is west.
105. If c12 is east and c12 is harbor, then c13 is granite.
106. If c13 is granite, then c13 is north.
107. If c13 is east and c13 is amber, then c14 is olive.
108. If c14 is olive, then c14 is east.
109. If c13 is north and c13 is granite, then c14 is teal.
110. If c14 is teal, then c14 is west.
111. If c13 is west and c13 is willow, then c14 is juniper.
112. If c14 is juniper, then c14 is south.
113. If c13 is south and c13 is meadow, then c14 is cedar.
114. If c14 is cedar, then c14 is north.
115. If c14 is west and c14 is teal, then c15 is meadow.
116. If c15 is meadow, then c15 is south.
117. If c14 is north and c14 is cedar, then c15 is teal.
118. If c15 is teal, then c15 is east.
119. If c14 is south and c14 is juniper, then c15 is hazel.
120. If c15 is hazel, then c15 is north.
121. If c14 is east and c14 is olive, then c15 is juniper.
122. If c15 is juniper, then c15 is west.
123. If c15 is east and c15 is teal, then c16 is ruby.
124. If c16 is ruby, then c16 is south.
125. If c15 is north and c15 is hazel, then c16 is laurel.
126. If c16 is laurel, then c16 is west.
127. If c15 is south and c15 is meadow, then c16 is cobalt.
128. If c16 is cobalt, then c16 is north.
129. If c15 is west and c15 is juniper, then c16 is amber.
130. If c16 is amber, then c16 is east.
131. If c16 is south and c16 is ruby, then c17 is orchid.
132. If c17 is orchid, then c17 is north.
133. If c16 is east and c16 is amber, then c17 is teal.
134. If c17 is teal, then c17 is east.
135. If c16 is west and c16 is laurel, then c17 is ruby.
136. If c17 is ruby, then c17 is south.
137. If c16 is north and c16 is cobalt, then c17 is granite.
138. If c17 is granite, then c17 is west.
139. If c17 is east and c17 is teal, then c18 is willow.
140. If c18 is willow, then c18 is east.
141. If c17 is south and c17 is ruby, then c18 is teal.
142. If c18 is teal, then c18 is west.
143. If c17 is north and c17 is orchid, then c18 is orchid.
144. If c18 is orchid, then c18 is north.
145. If c17 is west and c17 is granite, then c18 is ivory.
146. If c18 is ivory, then c18 is south.
147. If c18 is west and c18 is teal, then c19 is coral.
148. If c19 is coral, then c19 is west.
149. If c18 is south and c18 is ivory, then c19 is cobalt.
150. If c19 is cobalt, then c19 is north.
151. If c18 is north and c18 is orchid, then c19 is maple.
152. If c19 is maple, then c19 is east.
153. If c18 is east and c18 is willow, then c19 is birch.
154. If c19 is birch, then c19 is south.
155. If c19 is east and c19 is maple, then c20 is meadow.
156. If c20 is meadow, then c20 is south.
157. If c19 is north and c19 is cobalt, then c20 is juniper.
158. If c20 is juniper, then c20 is west.
159. If c19 is west and c19 is coral, then c20 is teal.
160. If c20 is teal, then c20 is east.
161. If c19 is south and c19 is birch, then c20 is poppy.
162. If c20 is poppy, then c20 is north.
163. If c20 is south and c20 is meadow, then c21 is cedar.
164. If c21 is cedar, then c21 is west.
165. If c20 is north and c20 is poppy, then c21 is granite.
166. If c21 is granite, then c21 is south.
167. If c20 is east and c20 is teal, then c21 is harbor.
168. If c21 is harbor, then c21 is east.
169. If c20 is west and c20 is juniper, then c21 is willow.
170. If c21 is willow, then c21 is north.
171. If c21 is north and c21 is willow, then c22 is amber.
172. If c22 is amber, then c22 is west.
173. If c21 is west and c21 is cedar, then c22 is cedar.
174. If c22 is cedar, then c22 is north.
175. If c21 is south and c21 is granite, then c22 is juniper.
176. If c22 is juniper, then c22 is east.
177. If c21 is east and c21 is harbor, then c22 is laurel.
178. If c22 is laurel, then c22 is south.
179. If c22 is south and c22 is laurel, then c23 is ruby.
180. If c23 is ruby, then c23 is south.
181. If c22 is west and c22 is amber, then c23 is cobalt.
182. If c23 is cobalt, then c23 is east.
183. If c22 is north and c22 is cedar, then c23 is willow.
184. If c23 is willow, then c23 is west.
185. If c22 is east and c22 is juniper, then c23 is olive.
186. If c23 is olive, then c23 is north.
187. If c23 is south and c23 is ruby, then c24 is granite.
188. If c24 is granite, then c24 is south.
189. If c23 is west and c23 is willow, then c24 is cobalt.
190. If c24 is cobalt, then c24 is north.
191. If c23 is north and c23 is olive, then c24 is pearl.
192. If c24 is pearl, then c24 is west.
193. If c23 is east and c23 is cobalt, then c24 is slate.
194. If c24 is slate, then c24 is east.
195. If c24 is west and c24 is pearl, then c25 is willow.
196. If c25 is willow, then c25 is west.
197. If c24 is south and c24 is granite, then c25 is olive.
198. If c25 is olive, then c25 is south.
199. If c24 is north and c24 is cobalt, then c25 is amber.
200. If c25 is amber, then c25 is east.
201. If c24 is east and c24 is slate, then c25 is maple.
202. If c25 is maple, then c25 is north.
Which state applies to c25?
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
c24 = c24
c25 = c25
</constants>
<predicates>
Ax: x is coral
Bx: x is harbor
Cx: x is ruby
Dx: x is elm
Ex: x is lime
Fx: x is slate
Gx: x is juniper
Hx: x is orchid
Ix: x is cobalt
Jx: x is willow
Kx: x is pearl
Lx: x is poppy
Mx: x is laurel
Nx: x is olive
Ox: x is violet
Px: x is meadow
Qx: x is teal
Rx: x is birch
Sx: x is maple
Tx: x is cedar
Ux: x is hazel
Vx: x is amber
Wx: x is granite
Xx: x is ivory
Yx: x is north
Zx: x is south
P0(x): x is east
P1(x): x is west
</predicates>
<premises>
O(c0)
P1(c0)
Z(c0) & O(c0) -> M(c1)
M(c1) -> Z(c1)
P1(c0) & O(c0) -> X(c1)
X(c1) -> P0(c1)
P0(c0) & O(c0) -> H(c1)
H(c1) -> P1(c1)
Y(c0) & O(c0) -> D(c1)
D(c1) -> Y(c1)
Z(c1) & M(c1) -> S(c2)
S(c2) -> Z(c2)
P1(c1) & H(c1) -> B(c2)
B(c2) -> Y(c2)
P0(c1) & X(c1) -> L(c2)
L(c2) -> P0(c2)
Y(c1) & D(c1) -> D(c2)
D(c2) -> P1(c2)
Z(c2) & S(c2) -> S(c3)
S(c3) -> Z(c3)
P1(c2) & D(c2) -> K(c3)
K(c3) -> P0(c3)
P0(c2) & L(c2) -> F(c3)
F(c3) -> P1(c3)
Y(c2) & B(c2) -> A(c3)
A(c3) -> Y(c3)
Y(c3) & A(c3) -> K(c4)
K(c4) -> Z(c4)
P1(c3) & F(c3) -> L(c4)
L(c4) -> P1(c4)
Z(c3) & S(c3) -> Q(c4)
Q(c4) -> Y(c4)
P0(c3) & K(c3) -> D(c4)
D(c4) -> P0(c4)
P0(c4) & D(c4) -> Q(c5)
Q(c5) -> P0(c5)
Z(c4) & K(c4) -> S(c5)
S(c5) -> Y(c5)
Y(c4) & Q(c4) -> C(c5)
C(c5) -> P1(c5)
P1(c4) & L(c4) -> A(c5)
A(c5) -> Z(c5)
Y(c5) & S(c5) -> T(c6)
T(c6) -> P1(c6)
P0(c5) & Q(c5) -> X(c6)
X(c6) -> Z(c6)
P1(c5) & C(c5) -> H(c6)
H(c6) -> Y(c6)
Z(c5) & A(c5) -> C(c6)
C(c6) -> P0(c6)
Z(c6) & X(c6) -> E(c7)
E(c7) -> Z(c7)
Y(c6) & H(c6) -> C(c7)
C(c7) -> P1(c7)
P1(c6) & T(c6) -> W(c7)
W(c7) -> Y(c7)
P0(c6) & C(c6) -> T(c7)
T(c7) -> P0(c7)
Y(c7) & W(c7) -> C(c8)
C(c8) -> P0(c8)
P0(c7) & T(c7) -> D(c8)
D(c8) -> Y(c8)
P1(c7) & C(c7) -> X(c8)
X(c8) -> P1(c8)
Z(c7) & E(c7) -> H(c8)
H(c8) -> Z(c8)
P1(c8) & X(c8) -> S(c9)
S(c9) -> P0(c9)
Y(c8) & D(c8) -> F(c9)
F(c9) -> Z(c9)
P0(c8) & C(c8) -> X(c9)
X(c9) -> Y(c9)
Z(c8) & H(c8) -> I(c9)
I(c9) -> P1(c9)
P0(c9) & S(c9) -> A(c10)
A(c10) -> Y(c10)
Z(c9) & F(c9) -> U(c10)
U(c10) -> Z(c10)
Y(c9) & X(c9) -> S(c10)
S(c10) -> P1(c10)
P1(c9) & I(c9) -> M(c10)
M(c10) -> P0(c10)
P0(c10) & M(c10) -> S(c11)
S(c11) -> Z(c11)
P1(c10) & S(c10) -> P(c11)
P(c11) -> P0(c11)
Y(c10) & A(c10) -> F(c11)
F(c11) -> Y(c11)
Z(c10) & U(c10) -> U(c11)
U(c11) -> P1(c11)
Y(c11) & F(c11) -> B(c12)
B(c12) -> P0(c12)
P1(c11) & U(c11) -> E(c12)
E(c12) -> Z(c12)
Z(c11) & S(c11) -> I(c12)
I(c12) -> Y(c12)
P0(c11) & P(c11) -> C(c12)
C(c12) -> P1(c12)
Y(c12) & I(c12) -> P(c13)
P(c13) -> Z(c13)
Z(c12) & E(c12) -> V(c13)
V(c13) -> P0(c13)
P1(c12) & C(c12) -> J(c13)
J(c13) -> P1(c13)
P0(c12) & B(c12) -> W(c13)
W(c13) -> Y(c13)
P0(c13) & V(c13) -> N(c14)
N(c14) -> P0(c14)
Y(c13) & W(c13) -> Q(c14)
Q(c14) -> P1(c14)
P1(c13) & J(c13) -> G(c14)
G(c14) -> Z(c14)
Z(c13) & P(c13) -> T(c14)
T(c14) -> Y(c14)
P1(c14) & Q(c14) -> P(c15)
P(c15) -> Z(c15)
Y(c14) & T(c14) -> Q(c15)
Q(c15) -> P0(c15)
Z(c14) & G(c14) -> U(c15)
U(c15) -> Y(c15)
P0(c14) & N(c14) -> G(c15)
G(c15) -> P1(c15)
P0(c15) & Q(c15) -> C(c16)
C(c16) -> Z(c16)
Y(c15) & U(c15) -> M(c16)
M(c16) -> P1(c16)
Z(c15) & P(c15) -> I(c16)
I(c16) -> Y(c16)
P1(c15) & G(c15) -> V(c16)
V(c16) -> P0(c16)
Z(c16) & C(c16) -> H(c17)
H(c17) -> Y(c17)
P0(c16) & V(c16) -> Q(c17)
Q(c17) -> P0(c17)
P1(c16) & M(c16) -> C(c17)
C(c17) -> Z(c17)
Y(c16) & I(c16) -> W(c17)
W(c17) -> P1(c17)
P0(c17) & Q(c17) -> J(c18)
J(c18) -> P0(c18)
Z(c17) & C(c17) -> Q(c18)
Q(c18) -> P1(c18)
Y(c17) & H(c17) -> H(c18)
H(c18) -> Y(c18)
P1(c17) & W(c17) -> X(c18)
X(c18) -> Z(c18)
P1(c18) & Q(c18) -> A(c19)
A(c19) -> P1(c19)
Z(c18) & X(c18) -> I(c19)
I(c19) -> Y(c19)
Y(c18) & H(c18) -> S(c19)
S(c19) -> P0(c19)
P0(c18) & J(c18) -> R(c19)
R(c19) -> Z(c19)
P0(c19) & S(c19) -> P(c20)
P(c20) -> Z(c20)
Y(c19) & I(c19) -> G(c20)
G(c20) -> P1(c20)
P1(c19) & A(c19) -> Q(c20)
Q(c20) -> P0(c20)
Z(c19) & R(c19) -> L(c20)
L(c20) -> Y(c20)
Z(c20) & P(c20) -> T(c21)
T(c21) -> P1(c21)
Y(c20) & L(c20) -> W(c21)
W(c21) -> Z(c21)
P0(c20) & Q(c20) -> B(c21)
B(c21) -> P0(c21)
P1(c20) & G(c20) -> J(c21)
J(c21) -> Y(c21)
Y(c21) & J(c21) -> V(c22)
V(c22) -> P1(c22)
P1(c21) & T(c21) -> T(c22)
T(c22) -> Y(c22)
Z(c21) & W(c21) -> G(c22)
G(c22) -> P0(c22)
P0(c21) & B(c21) -> M(c22)
M(c22) -> Z(c22)
Z(c22) & M(c22) -> C(c23)
C(c23) -> Z(c23)
P1(c22) & V(c22) -> I(c23)
I(c23) -> P0(c23)
Y(c22) & T(c22) -> J(c23)
J(c23) -> P1(c23)
P0(c22) & G(c22) -> N(c23)
N(c23) -> Y(c23)
Z(c23) & C(c23) -> W(c24)
W(c24) -> Z(c24)
P1(c23) & J(c23) -> I(c24)
I(c24) -> Y(c24)
Y(c23) & N(c23) -> K(c24)
K(c24) -> P1(c24)
P0(c23) & I(c23) -> F(c24)
F(c24) -> P0(c24)
P1(c24) & K(c24) -> J(c25)
J(c25) -> P1(c25)
Z(c24) & W(c24) -> N(c25)
N(c25) -> Z(c25)
Y(c24) & I(c24) -> V(c25)
V(c25) -> P0(c25)
P0(c24) & F(c24) -> S(c25)
S(c25) -> Y(c25)
</premises>
<proof>
O(c0) ; R
P1(c0) ; R
X(c1) ; ->E
P0(c1) ; ->E
L(c2) ; ->E
P0(c2) ; ->E
F(c3) ; ->E
P1(c3) ; ->E
L(c4) ; ->E
P1(c4) ; ->E
A(c5) ; ->E
Z(c5) ; ->E
C(c6) ; ->E
P0(c6) ; ->E
T(c7) ; ->E
P0(c7) ; ->E
D(c8) ; ->E
Y(c8) ; ->E
F(c9) ; ->E
Z(c9) ; ->E
U(c10) ; ->E
Z(c10) ; ->E
U(c11) ; ->E
P1(c11) ; ->E
E(c12) ; ->E
Z(c12) ; ->E
V(c13) ; ->E
P0(c13) ; ->E
N(c14) ; ->E
P0(c14) ; ->E
G(c15) ; ->E
P1(c15) ; ->E
V(c16) ; ->E
P0(c16) ; ->E
Q(c17) ; ->E
P0(c17) ; ->E
J(c18) ; ->E
P0(c18) ; ->E
R(c19) ; ->E
Z(c19) ; ->E
L(c20) ; ->E
Y(c20) ; ->E
W(c21) ; ->E
Z(c21) ; ->E
G(c22) ; ->E
P0(c22) ; ->E
N(c23) ; ->E
Y(c23) ; ->E
K(c24) ; ->E
P1(c24) ; ->E
J(c25) ; ->E
</proof>
<conclusion>
J(c25)
</conclusion>
</formal>
<answer>
willow
</answer><|endoftext|>
```

## Window 3 (2 documents, 314 pad tokens)
```
<question>
1. c0 is lime.
2. c0 is east.
3. If c0 is south and c0 is lime, then c1 is violet.
4. If c1 is violet, then c1 is east.
5. If c0 is north and c0 is lime, then c1 is elm.
6. If c1 is elm, then c1 is north.
7. If c0 is west and c0 is lime, then c1 is slate.
8. If c1 is slate, then c1 is west.
9. If c0 is east and c0 is lime, then c1 is meadow.
10. If c1 is meadow, then c1 is south.
11. If c1 is east and c1 is violet, then c2 is harbor.
12. If c2 is harbor, then c2 is south.
13. If c1 is south and c1 is meadow, then c2 is meadow.
14. If c2 is meadow, then c2 is west.
15. If c1 is north and c1 is elm, then c2 is amber.
16. If c2 is amber, then c2 is north.
17. If c1 is west and c1 is slate, then c2 is poppy.
18. If c2 is poppy, then c2 is east.
19. If c2 is south and c2 is harbor, then c3 is coral.
20. If c3 is coral, then c3 is south.
21. If c2 is north and c2 is amber, then c3 is ivory.
22. If c3 is ivory, then c3 is west.
23. If c2 is east and c2 is poppy, then c3 is orchid.
24. If c3 is orchid, then c3 is east.
25. If c2 is west and c2 is meadow, then c3 is amber.
26. If c3 is amber, then c3 is north.
27. If c3 is east and c3 is orchid, then c4 is juniper.
28. If c4 is juniper, then c4 is west.
29. If c3 is north and c3 is amber, then c4 is cedar.
30. If c4 is cedar, then c4 is south.
31. If c3 is south and c3 is coral, then c4 is slate.
32. If c4 is slate, then c4 is east.
33. If c3 is west and c3 is ivory, then c4 is amber.
34. If c4 is amber, then c4 is north.
35. If c4 is south and c4 is cedar, then c5 is meadow.
36. If c5 is meadow, then c5 is south.
37. If c4 is north and c4 is amber, then c5 is teal.
38. If c5 is teal, then c5 is east.
39. If c4 is west and c4 is juniper, then c5 is cobalt.
40. If c5 is cobalt, then c5 is west.
41. If c4 is east and c4 is slate, then c5 is slate.
42. If c5 is slate, then c5 is north.
43. If c5 is south and c5 is meadow, then c6 is teal.
44. If c6 is teal, then c6 is west.
45. If c5 is east and c5 is teal, then c6 is slate.
46. If c6 is slate, then c6 is east.
47. If c5 is west and c5 is cobalt, then c6 is orchid.
48. If c6 is orchid, then c6 is north.
49. If c5 is north and c5 is slate, then c6 is harbor.
50. If c6 is harbor, then c6 is south.
51. If c6 is east and c6 is slate, then c7 is birch.
52. If c7 is birch, then c7 is west.
53. If c6 is south and c6 is harbor, then c7 is pearl.
54. If c7 is pearl, then c7 is east.
55. If c6 is west and c6 is teal, then c7 is ivory.
56. If c7 is ivory, then c7 is north.
57. If c6 is north and c6 is orchid, then c7 is coral.
58. If c7 is coral, then c7 is south.
59. If c7 is south and c7 is coral, then c8 is violet.
60. If c8 is violet, then c8 is east.
61. If c7 is west and c7 is birch, then c8 is cedar.
62. If c8 is cedar, then c8 is south.
63. If c7 is north and c7 is ivory, then c8 is elm.
64. If c8 is elm, then c8 is north.
65. If c7 is east and c7 is pearl, then c8 is teal.
66. If c8 is teal, then c8 is west.
67. If c8 is south and c8 is cedar, then c9 is amber.
68. If c9 is amber, then c9 is north.
69. If c8 is north and c8 is elm, then c9 is slate.
70. If c9 is slate, then c9 is east.
71. If c8 is east and c8 is violet, then c9 is violet.
72. If c9 is violet, then c9 is west.
73. If c8 is west and c8 is teal, then c9 is orchid.
74. If c9 is orchid, then c9 is south.
75. If c9 is west and c9 is violet, then c10 is elm.
76. If c10 is elm, then c10 is west.
77. If c9 is east and c9 is slate, then c10 is meadow.
78. If c10 is meadow, then c10 is south.
79. If c9 is north and c9 is amber, then c10 is violet.
80. If c10 is violet, then c10 is north.
81. If c9 is south and c9 is orchid, then c10 is cobalt.
82. If c10 is cobalt, then c10 is east.
83. If c10 is west and c10 is elm, then c11 is coral.
84. If c11 is coral, then c11 is south.
85. If c10 is east and c10 is cobalt, then c11 is elm.
86. If c11 is elm, then c11 is west.
87. If c10 is north and c10 is violet, then c11 is orchid.
88. If c11 is orchid, then c11 is north.
89. If c10 is south and c10 is meadow, then c11 is ivory.
90. If c11 is ivory, then c11 is east.
91. If c11 is south and c11 is coral, then c12 is ruby.
92. If c12 is ruby, then c12 is west.
93. If c11 is west and c11 is elm, then c12 is teal.
94. If c12 is teal, then c12 is south.
95. If c11 is east and c11 is ivory, then c12 is amber.
96. If c12 is amber, then c12 is north.
97. If c11 is north and c11 is orchid, then c12 is maple.
98. If c12 is maple, then c12 is east.
99. If c12 is south and c12 is teal, then c13 is ruby.
100. If c13 is ruby, then c13 is south.
101. If c12 is north and c12 is amber, then c13 is birch.
102. If c13 is birch, then c13 is west.
103. If c12 is west and c12 is ruby, then c13 is pearl.
104. If c13 is pearl, then c13 is east.
105. If c12 is east and c12 is maple, then c13 is violet.
106. If c13 is violet, then c13 is north.
107. If c13 is east and c13 is pearl, then c14 is birch.
108. If c14 is birch, then c14 is west.
109. If c13 is west and c13 is birch, then c14 is elm.
110. If c14 is elm, then c14 is south.
111. If c13 is north and c13 is violet, then c14 is slate.
112. If c14 is slate, then c14 is north.
113. If c13 is south and c13 is ruby, then c14 is teal.
114. If c14 is teal, then c14 is east.
115. If c14 is north and c14 is slate, then c15 is maple.
116. If c15 is maple, then c15 is north.
117. If c14 is east and c14 is teal, then c15 is cedar.
118. If c15 is cedar, then c15 is south.
119. If c14 is south and c14 is elm, then c15 is poppy.
120. If c15 is poppy, then c15 is east.
121. If c14 is west and c14 is birch, then c15 is orchid.
122. If c15 is orchid, then c15 is west.
123. If c15 is south and c15 is cedar, then c16 is elm.
124. If c16 is elm, then c16 is west.
125. If c15 is north and c15 is maple, then c16 is amber.
126. If c16 is amber, then c16 is south.
127. If c15 is east and c15 is poppy, then c16 is birch.
128. If c16 is birch, then c16 is north.
129. If c15 is west and c15 is orchid, then c16 is juniper.
130. If c16 is juniper, then c16 is east.
131. If c16 is east and c16 is juniper, then c17 is teal.
132. If c17 is teal, then c17 is north.
133. If c16 is west and c16 is elm, then c17 is poppy.
134. If c17 is poppy, then c17 is east.
135. If c16 is south and c16 is amber, then c17 is elm.
136. If c17 is elm, then c17 is west.
137. If c16 is north and c16 is birch, then c17 is juniper.
138. If c17 is juniper, then c17 is south.
139. If c17 is east and c17 is poppy, then c18 is cedar.
140. If c18 is cedar, then c18 is south.
141. If c17 is west and c17 is elm, then c18 is cobalt.
142. If c18 is cobalt, then c18 is east.
143. If c17 is south and c17 is juniper, then c18 is violet.
144. If c18 is violet, then c18 is north.
145. If c17 is north and c17 is teal, then c18 is orchid.
146. If c18 is orchid, then c18 is west.
147. If c18 is south and c18 is cedar, then c19 is birch.
148. If c19 is birch, then c19 is north.
149. If c18 is west and c18 is orchid, then c19 is orchid.
150. If c19 is orchid, then c19 is east.
151. If c18 is east and c18 is cobalt, then c19 is violet.
152. If c19 is violet, then c19 is south.
153. If c18 is north and c18 is violet, then c19 is ivory.
154. If c19 is ivory, then c19 is west.
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
Ax: x is elm
Bx: x is amber
Cx: x is lime
Dx: x is coral
Ex: x is poppy
Fx: x is harbor
Gx: x is meadow
Hx: x is ruby
Ix: x is ivory
Jx: x is violet
Kx: x is juniper
Lx: x is pearl
Mx: x is teal
Nx: x is slate
Ox: x is cobalt
Px: x is olive
Qx: x is maple
Rx: x is cedar
Sx: x is orchid
Tx: x is birch
Ux: x is north
Vx: x is south
Wx: x is east
Xx: x is west
</predicates>
<premises>
C(c0)
W(c0)
V(c0) & C(c0) -> J(c1)
J(c1) -> W(c1)
U(c0) & C(c0) -> A(c1)
A(c1) -> U(c1)
X(c0) & C(c0) -> N(c1)
N(c1) -> X(c1)
W(c0) & C(c0) -> G(c1)
G(c1) -> V(c1)
W(c1) & J(c1) -> F(c2)
F(c2) -> V(c2)
V(c1) & G(c1) -> G(c2)
G(c2) -> X(c2)
U(c1) & A(c1) -> B(c2)
B(c2) -> U(c2)
X(c1) & N(c1) -> E(c2)
E(c2) -> W(c2)
V(c2) & F(c2) -> D(c3)
D(c3) -> V(c3)
U(c2) & B(c2) -> I(c3)
I(c3) -> X(c3)
W(c2) & E(c2) -> S(c3)
S(c3) -> W(c3)
X(c2) & G(c2) -> B(c3)
B(c3) -> U(c3)
W(c3) & S(c3) -> K(c4)
K(c4) -> X(c4)
U(c3) & B(c3) -> R(c4)
R(c4) -> V(c4)
V(c3) & D(c3) -> N(c4)
N(c4) -> W(c4)
X(c3) & I(c3) -> B(c4)
B(c4) -> U(c4)
V(c4) & R(c4) -> G(c5)
G(c5) -> V(c5)
U(c4) & B(c4) -> M(c5)
M(c5) -> W(c5)
X(c4) & K(c4) -> O(c5)
O(c5) -> X(c5)
W(c4) & N(c4) -> N(c5)
N(c5) -> U(c5)
V(c5) & G(c5) -> M(c6)
M(c6) -> X(c6)
W(c5) & M(c5) -> N(c6)
N(c6) -> W(c6)
X(c5) & O(c5) -> S(c6)
S(c6) -> U(c6)
U(c5) & N(c5) -> F(c6)
F(c6) -> V(c6)
W(c6) & N(c6) -> T(c7)
T(c7) -> X(c7)
V(c6) & F(c6) -> L(c7)
L(c7) -> W(c7)
X(c6) & M(c6) -> I(c7)
I(c7) -> U(c7)
U(c6) & S(c6) -> D(c7)
D(c7) -> V(c7)
V(c7) & D(c7) -> J(c8)
J(c8) -> W(c8)
X(c7) & T(c7) -> R(c8)
R(c8) -> V(c8)
U(c7) & I(c7) -> A(c8)
A(c8) -> U(c8)
W(c7) & L(c7) -> M(c8)
M(c8) -> X(c8)
V(c8) & R(c8) -> B(c9)
B(c9) -> U(c9)
U(c8) & A(c8) -> N(c9)
N(c9) -> W(c9)
W(c8) & J(c8) -> J(c9)
J(c9) -> X(c9)
X(c8) & M(c8) -> S(c9)
S(c9) -> V(c9)
X(c9) & J(c9) -> A(c10)
A(c10) -> X(c10)
W(c9) & N(c9) -> G(c10)
G(c10) -> V(c10)
U(c9) & B(c9) -> J(c10)
J(c10) -> U(c10)
V(c9) & S(c9) -> O(c10)
O(c10) -> W(c10)
X(c10) & A(c10) -> D(c11)
D(c11) -> V(c11)
W(c10) & O(c10) -> A(c11)
A(c11) -> X(c11)
U(c10) & J(c10) -> S(c11)
S(c11) -> U(c11)
V(c10) & G(c10) -> I(c11)
I(c11) -> W(c11)
V(c11) & D(c11) -> H(c12)
H(c12) -> X(c12)
X(c11) & A(c11) -> M(c12)
M(c12) -> V(c12)
W(c11) & I(c11) -> B(c12)
B(c12) -> U(c12)
U(c11) & S(c11) -> Q(c12)
Q(c12) -> W(c12)
V(c12) & M(c12) -> H(c13)
H(c13) -> V(c13)
U(c12) & B(c12) -> T(c13)
T(c13) -> X(c13)
X(c12) & H(c12) -> L(c13)
L(c13) -> W(c13)
W(c12) & Q(c12) -> J(c13)
J(c13) -> U(c13)
W(c13) & L(c13) -> T(c14)
T(c14) -> X(c14)
X(c13) & T(c13) -> A(c14)
A(c14) -> V(c14)
U(c13) & J(c13) -> N(c14)
N(c14) -> U(c14)
V(c13) & H(c13) -> M(c14)
M(c14) -> W(c14)
U(c14) & N(c14) -> Q(c15)
Q(c15) -> U(c15)
W(c14) & M(c14) -> R(c15)
R(c15) -> V(c15)
V(c14) & A(c14) -> E(c15)
E(c15) -> W(c15)
X(c14) & T(c14) -> S(c15)
S(c15) -> X(c15)
V(c15) & R(c15) -> A(c16)
A(c16) -> X(c16)
U(c15) & Q(c15) -> B(c16)
B(c16) -> V(c16)
W(c15) & E(c15) -> T(c16)
T(c16) -> U(c16)
X(c15) & S(c15) -> K(c16)
K(c16) -> W(c16)
W(c16) & K(c16) -> M(c17)
M(c17) -> U(c17)
X(c16) & A(c16) -> E(c17)
E(c17) -> W(c17)
V(c16) & B(c16) -> A(c17)
A(c17) -> X(c17)
U(c16) & T(c16) -> K(c17)
K(c17) -> V(c17)
W(c17) & E(c17) -> R(c18)
R(c18) -> V(c18)
X(c17) & A(c17) -> O(c18)
O(c18) -> W(c18)
V(c17) & K(c17) -> J(c18)
J(c18) -> U(c18)
U(c17) & M(c17) -> S(c18)
S(c18) -> X(c18)
V(c18) & R(c18) -> T(c19)
T(c19) -> U(c19)
X(c18) & S(c18) -> S(c19)
S(c19) -> W(c19)
W(c18) & O(c18) -> J(c19)
J(c19) -> V(c19)
U(c18) & J(c18) -> I(c19)
I(c19) -> X(c19)
</premises>
<proof>
C(c0) ; R
W(c0) ; R
G(c1) ; ->E
V(c1) ; ->E
G(c2) ; ->E
X(c2) ; ->E
B(c3) ; ->E
U(c3) ; ->E
R(c4) ; ->E
V(c4) ; ->E
G(c5) ; ->E
V(c5) ; ->E
M(c6) ; ->E
X(c6) ; ->E
I(c7) ; ->E
U(c7) ; ->E
A(c8) ; ->E
U(c8) ; ->E
N(c9) ; ->E
W(c9) ; ->E
G(c10) ; ->E
V(c10) ; ->E
I(c11) ; ->E
W(c11) ; ->E
B(c12) ; ->E
U(c12) ; ->E
T(c13) ; ->E
X(c13) ; ->E
A(c14) ; ->E
V(c14) ; ->E
E(c15) ; ->E
W(c15) ; ->E
T(c16) ; ->E
U(c16) ; ->E
K(c17) ; ->E
V(c17) ; ->E
J(c18) ; ->E
U(c18) ; ->E
I(c19) ; ->E
</proof>
<conclusion>
I(c19)
</conclusion>
</formal>
<answer>
ivory
</answer><|endoftext|><question>
1. c0 is cobalt.
2. c0 is west.
3. If c0 is north and c0 is cobalt, then c1 is maple.
4. If c1 is maple, then c1 is north.
5. If c0 is south and c0 is cobalt, then c1 is cedar.
6. If c1 is cedar, then c1 is west.
7. If c0 is west and c0 is cobalt, then c1 is violet.
8. If c1 is violet, then c1 is south.
9. If c0 is east and c0 is cobalt, then c1 is ruby.
10. If c1 is ruby, then c1 is east.
11. If c1 is west and c1 is cedar, then c2 is cedar.
12. If c2 is cedar, then c2 is east.
13. If c1 is north and c1 is maple, then c2 is teal.
14. If c2 is teal, then c2 is north.
15. If c1 is east and c1 is ruby, then c2 is laurel.
16. If c2 is laurel, then c2 is south.
17. If c1 is south and c1 is violet, then c2 is harbor.
18. If c2 is harbor, then c2 is west.
19. If c2 is north and c2 is teal, then c3 is harbor.
20. If c3 is harbor, then c3 is south.
21. If c2 is south and c2 is laurel, then c3 is violet.
22. If c3 is violet, then c3 is north.
23. If c2 is west and c2 is harbor, then c3 is teal.
24. If c3 is teal, then c3 is west.
25. If c2 is east and c2 is cedar, then c3 is laurel.
26. If c3 is laurel, then c3 is east.
27. If c3 is south and c3 is harbor, then c4 is laurel.
28. If c4 is laurel, then c4 is south.
29. If c3 is west and c3 is teal, then c4 is cedar.
30. If c4 is cedar, then c4 is east.
31. If c3 is north and c3 is violet, then c4 is harbor.
32. If c4 is harbor, then c4 is north.
33. If c3 is east and c3 is laurel, then c4 is violet.
34. If c4 is violet, then c4 is west.
35. If c4 is east and c4 is cedar, then c5 is harbor.
36. If c5 is harbor, then c5 is east.
37. If c4 is south and c4 is laurel, then c5 is ivory.
38. If c5 is ivory, then c5 is south.
39. If c4 is north and c4 is harbor, then c5 is maple.
40. If c5 is maple, then c5 is west.
41. If c4 is west and c4 is violet, then c5 is cedar.
42. If c5 is cedar, then c5 is north.
43. If c5 is east and c5 is harbor, then c6 is maple.
44. If c6 is maple, then c6 is south.
45. If c5 is west and c5 is maple, then c6 is violet.
46. If c6 is violet, then c6 is north.
47. If c5 is south and c5 is ivory, then c6 is harbor.
48. If c6 is harbor, then c6 is west.
49. If c5 is north and c5 is cedar, then c6 is cedar.
50. If c6 is cedar, then c6 is east.
51. If c6 is east and c6 is cedar, then c7 is laurel.
52. If c7 is laurel, then c7 is north.
53. If c6 is west and c6 is harbor, then c7 is harbor.
54. If c7 is harbor, then c7 is west.
55. If c6 is south and c6 is maple, then c7 is teal.
56. If c7 is teal, then c7 is south.
57. If c6 is north and c6 is violet, then c7 is ruby.
58. If c7 is ruby, then c7 is east.
59. If c7 is east and c7 is ruby, then c8 is laurel.
60. If c8 is laurel, then c8 is south.
61. If c7 is west and c7 is harbor, then c8 is teal.
62. If c8 is teal, then c8 is east.
63. If c7 is south and c7 is teal, then c8 is maple.
64. If c8 is maple, then c8 is north.
65. If c7 is north and c7 is laurel, then c8 is violet.
66. If c8 is violet, then c8 is west.
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
Ax: x is cobalt
Bx: x is harbor
Cx: x is laurel
Dx: x is teal
Ex: x is ivory
Fx: x is cedar
Gx: x is maple
Hx: x is ruby
Ix: x is violet
Jx: x is north
Kx: x is south
Lx: x is east
Mx: x is west
</predicates>
<premises>
A(c0)
M(c0)
J(c0) & A(c0) -> G(c1)
G(c1) -> J(c1)
K(c0) & A(c0) -> F(c1)
F(c1) -> M(c1)
M(c0) & A(c0) -> I(c1)
I(c1) -> K(c1)
L(c0) & A(c0) -> H(c1)
H(c1) -> L(c1)
M(c1) & F(c1) -> F(c2)
F(c2) -> L(c2)
J(c1) & G(c1) -> D(c2)
D(c2) -> J(c2)
L(c1) & H(c1) -> C(c2)
C(c2) -> K(c2)
K(c1) & I(c1) -> B(c2)
B(c2) -> M(c2)
J(c2) & D(c2) -> B(c3)
B(c3) -> K(c3)
K(c2) & C(c2) -> I(c3)
I(c3) -> J(c3)
M(c2) & B(c2) -> D(c3)
D(c3) -> M(c3)
L(c2) & F(c2) -> C(c3)
C(c3) -> L(c3)
K(c3) & B(c3) -> C(c4)
C(c4) -> K(c4)
M(c3) & D(c3) -> F(c4)
F(c4) -> L(c4)
J(c3) & I(c3) -> B(c4)
B(c4) -> J(c4)
L(c3) & C(c3) -> I(c4)
I(c4) -> M(c4)
L(c4) & F(c4) -> B(c5)
B(c5) -> L(c5)
K(c4) & C(c4) -> E(c5)
E(c5) -> K(c5)
J(c4) & B(c4) -> G(c5)
G(c5) -> M(c5)
M(c4) & I(c4) -> F(c5)
F(c5) -> J(c5)
L(c5) & B(c5) -> G(c6)
G(c6) -> K(c6)
M(c5) & G(c5) -> I(c6)
I(c6) -> J(c6)
K(c5) & E(c5) -> B(c6)
B(c6) -> M(c6)
J(c5) & F(c5) -> F(c6)
F(c6) -> L(c6)
L(c6) & F(c6) -> C(c7)
C(c7) -> J(c7)
M(c6) & B(c6) -> B(c7)
B(c7) -> M(c7)
K(c6) & G(c6) -> D(c7)
D(c7) -> K(c7)
J(c6) & I(c6) -> H(c7)
H(c7) -> L(c7)
L(c7) & H(c7) -> C(c8)
C(c8) -> K(c8)
M(c7) & B(c7) -> D(c8)
D(c8) -> L(c8)
K(c7) & D(c7) -> G(c8)
G(c8) -> J(c8)
J(c7) & C(c7) -> I(c8)
I(c8) -> M(c8)
</premises>
<proof>
A(c0) ; R
M(c0) ; R
I(c1) ; ->E
K(c1) ; ->E
B(c2) ; ->E
M(c2) ; ->E
D(c3) ; ->E
M(c3) ; ->E
F(c4) ; ->E
L(c4) ; ->E
B(c5) ; ->E
L(c5) ; ->E
G(c6) ; ->E
K(c6) ; ->E
D(c7) ; ->E
K(c7) ; ->E
G(c8) ; ->E
</proof>
<conclusion>
G(c8)
</conclusion>
</formal>
<answer>
maple
</answer><|endoftext|>
```

## Window 8060 (1 documents, 539 pad tokens)
```
<question>
1. c0 is cobalt.
2. c0 is east.
3. If c0 is east and c0 is cobalt, then c1 is granite.
4. If c1 is granite, then c1 is west.
5. If c0 is north and c0 is cobalt, then c1 is ivory.
6. If c1 is ivory, then c1 is north.
7. If c0 is south and c0 is cobalt, then c1 is coral.
8. If c1 is coral, then c1 is south.
9. If c0 is west and c0 is cobalt, then c1 is poppy.
10. If c1 is poppy, then c1 is east.
11. If c1 is west and c1 is granite, then c2 is birch.
12. If c2 is birch, then c2 is north.
13. If c1 is east and c1 is poppy, then c2 is ruby.
14. If c2 is ruby, then c2 is west.
15. If c1 is south and c1 is coral, then c2 is juniper.
16. If c2 is juniper, then c2 is south.
17. If c1 is north and c1 is ivory, then c2 is coral.
18. If c2 is coral, then c2 is east.
19. If c2 is south and c2 is juniper, then c3 is lime.
20. If c3 is lime, then c3 is south.
21. If c2 is west and c2 is ruby, then c3 is willow.
22. If c3 is willow, then c3 is east.
23. If c2 is east and c2 is coral, then c3 is violet.
24. If c3 is violet, then c3 is north.
25. If c2 is north and c2 is birch, then c3 is poppy.
26. If c3 is poppy, then c3 is west.
27. If c3 is north and c3 is violet, then c4 is granite.
28. If c4 is granite, then c4 is north.
29. If c3 is east and c3 is willow, then c4 is amber.
30. If c4 is amber, then c4 is east.
31. If c3 is south and c3 is lime, then c4 is poppy.
32. If c4 is poppy, then c4 is west.
33. If c3 is west and c3 is poppy, then c4 is laurel.
34. If c4 is laurel, then c4 is south.
35. If c4 is south and c4 is laurel, then c5 is coral.
36. If c5 is coral, then c5 is east.
37. If c4 is west and c4 is poppy, then c5 is granite.
38. If c5 is granite, then c5 is south.
39. If c4 is east and c4 is amber, then c5 is elm.
40. If c5 is elm, then c5 is north.
41. If c4 is north and c4 is granite, then c5 is orchid.
42. If c5 is orchid, then c5 is west.
43. If c5 is west and c5 is orchid, then c6 is laurel.
44. If c6 is laurel, then c6 is south.
45. If c5 is north and c5 is elm, then c6 is amber.
46. If c6 is amber, then c6 is north.
47. If c5 is south and c5 is granite, then c6 is slate.
48. If c6 is slate, then c6 is east.
49. If c5 is east and c5 is coral, then c6 is orchid.
50. If c6 is orchid, then c6 is west.
51. If c6 is north and c6 is amber, then c7 is violet.
52. If c7 is violet, then c7 is east.
53. If c6 is west and c6 is orchid, then c7 is ivory.
54. If c7 is ivory, then c7 is north.
55. If c6 is south and c6 is laurel, then c7 is elm.
56. If c7 is elm, then c7 is west.
57. If c6 is east and c6 is slate, then c7 is harbor.
58. If c7 is harbor, then c7 is south.
59. If c7 is south and c7 is harbor, then c8 is slate.
60. If c8 is slate, then c8 is south.
61. If c7 is north and c7 is ivory, then c8 is teal.
62. If c8 is teal, then c8 is north.
63. If c7 is west and c7 is elm, then c8 is violet.
64. If c8 is violet, then c8 is east.
65. If c7 is east and c7 is violet, then c8 is maple.
66. If c8 is maple, then c8 is west.
67. If c8 is east and c8 is violet, then c9 is laurel.
68. If c9 is laurel, then c9 is north.
69. If c8 is west and c8 is maple, then c9 is elm.
70. If c9 is elm, then c9 is east.
71. If c8 is south and c8 is slate, then c9 is pearl.
72. If c9 is pearl, then c9 is south.
73. If c8 is north and c8 is teal, then c9 is granite.
74. If c9 is granite, then c9 is west.
75. If c9 is south and c9 is pearl, then c10 is maple.
76. If c10 is maple, then c10 is west.
77. If c9 is east and c9 is elm, then c10 is slate.
78. If c10 is slate, then c10 is south.
79. If c9 is north and c9 is laurel, then c10 is coral.
80. If c10 is coral, then c10 is east.
81. If c9 is west and c9 is granite, then c10 is juniper.
82. If c10 is juniper, then c10 is north.
83. If c10 is east and c10 is coral, then c11 is birch.
84. If c11 is birch, then c11 is north.
85. If c10 is north and c10 is juniper, then c11 is orchid.
86. If c11 is orchid, then c11 is south.
87. If c10 is south and c10 is slate, then c11 is poppy.
88. If c11 is poppy, then c11 is east.
89. If c10 is west and c10 is maple, then c11 is coral.
90. If c11 is coral, then c11 is west.
91. If c11 is east and c11 is poppy, then c12 is slate.
92. If c12 is slate, then c12 is south.
93. If c11 is west and c11 is coral, then c12 is olive.
94. If c12 is olive, then c12 is west.
95. If c11 is south and c11 is orchid, then c12 is maple.
96. If c12 is maple, then c12 is north.
97. If c11 is north and c11 is birch, then c12 is cedar.
98. If c12 is cedar, then c12 is east.
99. If c12 is north and c12 is maple, then c13 is maple.
100. If c13 is maple, then c13 is north.
101. If c12 is east and c12 is cedar, then c13 is granite.
102. If c13 is granite, then c13 is south.
103. If c12 is south and c12 is slate, then c13 is laurel.
104. If c13 is laurel, then c13 is west.
105. If c12 is west and c12 is olive, then c13 is birch.
106. If c13 is birch, then c13 is east.
107. If c13 is west and c13 is laurel, then c14 is laurel.
108. If c14 is laurel, then c14 is south.
109. If c13 is south and c13 is granite, then c14 is juniper.
110. If c14 is juniper, then c14 is north.
111. If c13 is north and c13 is maple, then c14 is harbor.
112. If c14 is harbor, then c14 is east.
113. If c13 is east and c13 is birch, then c14 is orchid.
114. If c14 is orchid, then c14 is west.
115. If c14 is south and c14 is laurel, then c15 is granite.
116. If c15 is granite, then c15 is south.
117. If c14 is east and c14 is harbor, then c15 is teal.
118. If c15 is teal, then c15 is east.
119. If c14 is north and c14 is juniper, then c15 is lime.
120. If c15 is lime, then c15 is west.
121. If c14 is west and c14 is orchid, then c15 is meadow.
122. If c15 is meadow, then c15 is north.
123. If c15 is north and c15 is meadow, then c16 is slate.
124. If c16 is slate, then c16 is north.
125. If c15 is south and c15 is granite, then c16 is lime.
126. If c16 is lime, then c16 is west.
127. If c15 is west and c15 is lime, then c16 is olive.
128. If c16 is olive, then c16 is south.
129. If c15 is east and c15 is teal, then c16 is ivory.
130. If c16 is ivory, then c16 is east.
131. If c16 is north and c16 is slate, then c17 is ivory.
132. If c17 is ivory, then c17 is west.
133. If c16 is south and c16 is olive, then c17 is teal.
134. If c17 is teal, then c17 is north.
135. If c16 is east and c16 is ivory, then c17 is juniper.
136. If c17 is juniper, then c17 is south.
137. If c16 is west and c16 is lime, then c17 is maple.
138. If c17 is maple, then c17 is east.
139. If c17 is west and c17 is ivory, then c18 is lime.
140. If c18 is lime, then c18 is north.
141. If c17 is south and c17 is juniper, then c18 is amber.
142. If c18 is amber, then c18 is east.
143. If c17 is east and c17 is maple, then c18 is hazel.
144. If c18 is hazel, then c18 is south.
145. If c17 is north and c17 is teal, then c18 is elm.
146. If c18 is elm, then c18 is west.
147. If c18 is south and c18 is hazel, then c19 is orchid.
148. If c19 is orchid, then c19 is west.
149. If c18 is west and c18 is elm, then c19 is cedar.
150. If c19 is cedar, then c19 is south.
151. If c18 is east and c18 is amber, then c19 is ivory.
152. If c19 is ivory, then c19 is north.
153. If c18 is north and c18 is lime, then c19 is violet.
154. If c19 is violet, then c19 is east.
155. If c19 is west and c19 is orchid, then c20 is violet.
156. If c20 is violet, then c20 is north.
157. If c19 is south and c19 is cedar, then c20 is teal.
158. If c20 is teal, then c20 is west.
159. If c19 is east and c19 is violet, then c20 is pearl.
160. If c20 is pearl, then c20 is east.
161. If c19 is north and c19 is ivory, then c20 is cedar.
162. If c20 is cedar, then c20 is south.
163. If c20 is east and c20 is pearl, then c21 is harbor.
164. If c21 is harbor, then c21 is west.
165. If c20 is west and c20 is teal, then c21 is meadow.
166. If c21 is meadow, then c21 is south.
167. If c20 is north and c20 is violet, then c21 is poppy.
168. If c21 is poppy, then c21 is north.
169. If c20 is south and c20 is cedar, then c21 is juniper.
170. If c21 is juniper, then c21 is east.
171. If c21 is west and c21 is harbor, then c22 is amber.
172. If c22 is amber, then c22 is east.
173. If c21 is south and c21 is meadow, then c22 is meadow.
174. If c22 is meadow, then c22 is west.
175. If c21 is east and c21 is juniper, then c22 is poppy.
176. If c22 is poppy, then c22 is north.
177. If c21 is north and c21 is poppy, then c22 is ivory.
178. If c22 is ivory, then c22 is south.
179. If c22 is east and c22 is amber, then c23 is meadow.
180. If c23 is meadow, then c23 is north.
181. If c22 is west and c22 is meadow, then c23 is orchid.
182. If c23 is orchid, then c23 is west.
183. If c22 is north and c22 is poppy, then c23 is violet.
184. If c23 is violet, then c23 is south.
185. If c22 is south and c22 is ivory, then c23 is slate.
186. If c23 is slate, then c23 is east.
187. If c23 is south and c23 is violet, then c24 is hazel.
188. If c24 is hazel, then c24 is west.
189. If c23 is east and c23 is slate, then c24 is teal.
190. If c24 is teal, then c24 is east.
191. If c23 is north and c23 is meadow, then c24 is cedar.
192. If c24 is cedar, then c24 is south.
193. If c23 is west and c23 is orchid, then c24 is granite.
194. If c24 is granite, then c24 is north.
195. If c24 is west and c24 is hazel, then c25 is poppy.
196. If c25 is poppy, then c25 is south.
197. If c24 is east and c24 is teal, then c25 is elm.
198. If c25 is elm, then c25 is east.
199. If c24 is north and c24 is granite, then c25 is pearl.
200. If c25 is pearl, then c25 is north.
201. If c24 is south and c24 is cedar, then c25 is maple.
202. If c25 is maple, then c25 is west.
Which state applies to c25?
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
c24 = c24
c25 = c25
</constants>
<predicates>
Ax: x is laurel
Bx: x is maple
Cx: x is coral
Dx: x is violet
Ex: x is amber
Fx: x is slate
Gx: x is cedar
Hx: x is orchid
Ix: x is cobalt
Jx: x is juniper
Kx: x is meadow
Lx: x is pearl
Mx: x is elm
Nx: x is granite
Ox: x is ruby
Px: x is willow
Qx: x is poppy
Rx: x is birch
Sx: x is ivory
Tx: x is lime
Ux: x is harbor
Vx: x is teal
Wx: x is olive
Xx: x is hazel
Yx: x is north
Zx: x is south
P0(x): x is east
P1(x): x is west
</predicates>
<premises>
I(c0)
P0(c0)
P0(c0) & I(c0) -> N(c1)
N(c1) -> P1(c1)
Y(c0) & I(c0) -> S(c1)
S(c1) -> Y(c1)
Z(c0) & I(c0) -> C(c1)
C(c1) -> Z(c1)
P1(c0) & I(c0) -> Q(c1)
Q(c1) -> P0(c1)
P1(c1) & N(c1) -> R(c2)
R(c2) -> Y(c2)
P0(c1) & Q(c1) -> O(c2)
O(c2) -> P1(c2)
Z(c1) & C(c1) -> J(c2)
J(c2) -> Z(c2)
Y(c1) & S(c1) -> C(c2)
C(c2) -> P0(c2)
Z(c2) & J(c2) -> T(c3)
T(c3) -> Z(c3)
P1(c2) & O(c2) -> P(c3)
P(c3) -> P0(c3)
P0(c2) & C(c2) -> D(c3)
D(c3) -> Y(c3)
Y(c2) & R(c2) -> Q(c3)
Q(c3) -> P1(c3)
Y(c3) & D(c3) -> N(c4)
N(c4) -> Y(c4)
P0(c3) & P(c3) -> E(c4)
E(c4) -> P0(c4)
Z(c3) & T(c3) -> Q(c4)
Q(c4) -> P1(c4)
P1(c3) & Q(c3) -> A(c4)
A(c4) -> Z(c4)
Z(c4) & A(c4) -> C(c5)
C(c5) -> P0(c5)
P1(c4) & Q(c4) -> N(c5)
N(c5) -> Z(c5)
P0(c4) & E(c4) -> M(c5)
M(c5) -> Y(c5)
Y(c4) & N(c4) -> H(c5)
H(c5) -> P1(c5)
P1(c5) & H(c5) -> A(c6)
A(c6) -> Z(c6)
Y(c5) & M(c5) -> E(c6)
E(c6) -> Y(c6)
Z(c5) & N(c5) -> F(c6)
F(c6) -> P0(c6)
P0(c5) & C(c5) -> H(c6)
H(c6) -> P1(c6)
Y(c6) & E(c6) -> D(c7)
D(c7) -> P0(c7)
P1(c6) & H(c6) -> S(c7)
S(c7) -> Y(c7)
Z(c6) & A(c6) -> M(c7)
M(c7) -> P1(c7)
P0(c6) & F(c6) -> U(c7)
U(c7) -> Z(c7)
Z(c7) & U(c7) -> F(c8)
F(c8) -> Z(c8)
Y(c7) & S(c7) -> V(c8)
V(c8) -> Y(c8)
P1(c7) & M(c7) -> D(c8)
D(c8) -> P0(c8)
P0(c7) & D(c7) -> B(c8)
B(c8) -> P1(c8)
P0(c8) & D(c8) -> A(c9)
A(c9) -> Y(c9)
P1(c8) & B(c8) -> M(c9)
M(c9) -> P0(c9)
Z(c8) & F(c8) -> L(c9)
L(c9) -> Z(c9)
Y(c8) & V(c8) -> N(c9)
N(c9) -> P1(c9)
Z(c9) & L(c9) -> B(c10)
B(c10) -> P1(c10)
P0(c9) & M(c9) -> F(c10)
F(c10) -> Z(c10)
Y(c9) & A(c9) -> C(c10)
C(c10) -> P0(c10)
P1(c9) & N(c9) -> J(c10)
J(c10) -> Y(c10)
P0(c10) & C(c10) -> R(c11)
R(c11) -> Y(c11)
Y(c10) & J(c10) -> H(c11)
H(c11) -> Z(c11)
Z(c10) & F(c10) -> Q(c11)
Q(c11) -> P0(c11)
P1(c10) & B(c10) -> C(c11)
C(c11) -> P1(c11)
P0(c11) & Q(c11) -> F(c12)
F(c12) -> Z(c12)
P1(c11) & C(c11) -> W(c12)
W(c12) -> P1(c12)
Z(c11) & H(c11) -> B(c12)
B(c12) -> Y(c12)
Y(c11) & R(c11) -> G(c12)
G(c12) -> P0(c12)
Y(c12) & B(c12) -> B(c13)
B(c13) -> Y(c13)
P0(c12) & G(c12) -> N(c13)
N(c13) -> Z(c13)
Z(c12) & F(c12) -> A(c13)
A(c13) -> P1(c13)
P1(c12) & W(c12) -> R(c13)
R(c13) -> P0(c13)
P1(c13) & A(c13) -> A(c14)
A(c14) -> Z(c14)
Z(c13) & N(c13) -> J(c14)
J(c14) -> Y(c14)
Y(c13) & B(c13) -> U(c14)
U(c14) -> P0(c14)
P0(c13) & R(c13) -> H(c14)
H(c14) -> P1(c14)
Z(c14) & A(c14) -> N(c15)
N(c15) -> Z(c15)
P0(c14) & U(c14) -> V(c15)
V(c15) -> P0(c15)
Y(c14) & J(c14) -> T(c15)
T(c15) -> P1(c15)
P1(c14) & H(c14) -> K(c15)
K(c15) -> Y(c15)
Y(c15) & K(c15) -> F(c16)
F(c16) -> Y(c16)
Z(c15) & N(c15) -> T(c16)
T(c16) -> P1(c16)
P1(c15) & T(c15) -> W(c16)
W(c16) -> Z(c16)
P0(c15) & V(c15) -> S(c16)
S(c16) -> P0(c16)
Y(c16) & F(c16) -> S(c17)
S(c17) -> P1(c17)
Z(c16) & W(c16) -> V(c17)
V(c17) -> Y(c17)
P0(c16) & S(c16) -> J(c17)
J(c17) -> Z(c17)
P1(c16) & T(c16) -> B(c17)
B(c17) -> P0(c17)
P1(c17) & S(c17) -> T(c18)
T(c18) -> Y(c18)
Z(c17) & J(c17) -> E(c18)
E(c18) -> P0(c18)
P0(c17) & B(c17) -> X(c18)
X(c18) -> Z(c18)
Y(c17) & V(c17) -> M(c18)
M(c18) -> P1(c18)
Z(c18) & X(c18) -> H(c19)
H(c19) -> P1(c19)
P1(c18) & M(c18) -> G(c19)
G(c19) -> Z(c19)
P0(c18) & E(c18) -> S(c19)
S(c19) -> Y(c19)
Y(c18) & T(c18) -> D(c19)
D(c19) -> P0(c19)
P1(c19) & H(c19) -> D(c20)
D(c20) -> Y(c20)
Z(c19) & G(c19) -> V(c20)
V(c20) -> P1(c20)
P0(c19) & D(c19) -> L(c20)
L(c20) -> P0(c20)
Y(c19) & S(c19) -> G(c20)
G(c20) -> Z(c20)
P0(c20) & L(c20) -> U(c21)
U(c21) -> P1(c21)
P1(c20) & V(c20) -> K(c21)
K(c21) -> Z(c21)
Y(c20) & D(c20) -> Q(c21)
Q(c21) -> Y(c21)
Z(c20) & G(c20) -> J(c21)
J(c21) -> P0(c21)
P1(c21) & U(c21) -> E(c22)
E(c22) -> P0(c22)
Z(c21) & K(c21) -> K(c22)
K(c22) -> P1(c22)
P0(c21) & J(c21) -> Q(c22)
Q(c22) -> Y(c22)
Y(c21) & Q(c21) -> S(c22)
S(c22) -> Z(c22)
P0(c22) & E(c22) -> K(c23)
K(c23) -> Y(c23)
P1(c22) & K(c22) -> H(c23)
H(c23) -> P1(c23)
Y(c22) & Q(c22) -> D(c23)
D(c23) -> Z(c23)
Z(c22) & S(c22) -> F(c23)
F(c23) -> P0(c23)
Z(c23) & D(c23) -> X(c24)
X(c24) -> P1(c24)
P0(c23) & F(c23) -> V(c24)
V(c24) -> P0(c24)
Y(c23) & K(c23) -> G(c24)
G(c24) -> Z(c24)
P1(c23) & H(c23) -> N(c24)
N(c24) -> Y(c24)
P1(c24) & X(c24) -> Q(c25)
Q(c25) -> Z(c25)
P0(c24) & V(c24) -> M(c25)
M(c25) -> P0(c25)
Y(c24) & N(c24) -> L(c25)
L(c25) -> Y(c25)
Z(c24) & G(c24) -> B(c25)
B(c25) -> P1(c25)
</premises>
<proof>
I(c0) ; R
P0(c0) ; R
N(c1) ; ->E
P1(c1) ; ->E
R(c2) ; ->E
Y(c2) ; ->E
Q(c3) ; ->E
P1(c3) ; ->E
A(c4) ; ->E
Z(c4) ; ->E
C(c5) ; ->E
P0(c5) ; ->E
H(c6) ; ->E
P1(c6) ; ->E
S(c7) ; ->E
Y(c7) ; ->E
V(c8) ; ->E
Y(c8) ; ->E
N(c9) ; ->E
P1(c9) ; ->E
J(c10) ; ->E
Y(c10) ; ->E
H(c11) ; ->E
Z(c11) ; ->E
B(c12) ; ->E
Y(c12) ; ->E
B(c13) ; ->E
Y(c13) ; ->E
U(c14) ; ->E
P0(c14) ; ->E
V(c15) ; ->E
P0(c15) ; ->E
S(c16) ; ->E
P0(c16) ; ->E
J(c17) ; ->E
Z(c17) ; ->E
E(c18) ; ->E
P0(c18) ; ->E
S(c19) ; ->E
Y(c19) ; ->E
G(c20) ; ->E
Z(c20) ; ->E
J(c21) ; ->E
P0(c21) ; ->E
Q(c22) ; ->E
Y(c22) ; ->E
D(c23) ; ->E
Z(c23) ; ->E
X(c24) ; ->E
P1(c24) ; ->E
Q(c25) ; ->E
</proof>
<conclusion>
Q(c25)
</conclusion>
</formal>
<answer>
poppy
</answer><|endoftext|>
```

## Window 9449 (2 documents, 354 pad tokens)
```
<question>
1. c0 is granite.
2. c0 is west.
3. If c0 is south and c0 is granite, then c1 is maple.
4. If c1 is maple, then c1 is south.
5. If c0 is east and c0 is granite, then c1 is orchid.
6. If c1 is orchid, then c1 is east.
7. If c0 is west and c0 is granite, then c1 is elm.
8. If c1 is elm, then c1 is north.
9. If c0 is north and c0 is granite, then c1 is harbor.
10. If c1 is harbor, then c1 is west.
11. If c1 is north and c1 is elm, then c2 is orchid.
12. If c2 is orchid, then c2 is west.
13. If c1 is south and c1 is maple, then c2 is lime.
14. If c2 is lime, then c2 is south.
15. If c1 is west and c1 is harbor, then c2 is maple.
16. If c2 is maple, then c2 is north.
17. If c1 is east and c1 is orchid, then c2 is teal.
18. If c2 is teal, then c2 is east.
19. If c2 is north and c2 is maple, then c3 is amber.
20. If c3 is amber, then c3 is south.
21. If c2 is west and c2 is orchid, then c3 is meadow.
22. If c3 is meadow, then c3 is north.
23. If c2 is south and c2 is lime, then c3 is coral.
24. If c3 is coral, then c3 is west.
25. If c2 is east and c2 is teal, then c3 is willow.
26. If c3 is willow, then c3 is east.
27. If c3 is west and c3 is coral, then c4 is amber.
28. If c4 is amber, then c4 is west.
29. If c3 is east and c3 is willow, then c4 is lime.
30. If c4 is lime, then c4 is south.
31. If c3 is north and c3 is meadow, then c4 is willow.
32. If c4 is willow, then c4 is east.
33. If c3 is south and c3 is amber, then c4 is teal.
34. If c4 is teal, then c4 is north.
35. If c4 is south and c4 is lime, then c5 is hazel.
36. If c5 is hazel, then c5 is north.
37. If c4 is east and c4 is willow, then c5 is amber.
38. If c5 is amber, then c5 is west.
39. If c4 is north and c4 is teal, then c5 is lime.
40. If c5 is lime, then c5 is east.
41. If c4 is west and c4 is amber, then c5 is harbor.
42. If c5 is harbor, then c5 is south.
43. If c5 is north and c5 is hazel, then c6 is coral.
44. If c6 is coral, then c6 is east.
45. If c5 is west and c5 is amber, then c6 is cobalt.
46. If c6 is cobalt, then c6 is south.
47. If c5 is east and c5 is lime, then c6 is maple.
48. If c6 is maple, then c6 is north.
49. If c5 is south and c5 is harbor, then c6 is juniper.
50. If c6 is juniper, then c6 is west.
51. If c6 is south and c6 is cobalt, then c7 is meadow.
52. If c7 is meadow, then c7 is south.
53. If c6 is west and c6 is juniper, then c7 is amber.
54. If c7 is amber, then c7 is west.
55. If c6 is east and c6 is coral, then c7 is elm.
56. If c7 is elm, then c7 is north.
57. If c6 is north and c6 is maple, then c7 is hazel.
58. If c7 is hazel, then c7 is east.
59. If c7 is south and c7 is meadow, then c8 is amber.
60. If c8 is amber, then c8 is west.
61. If c7 is north and c7 is elm, then c8 is willow.
62. If c8 is willow, then c8 is north.
63. If c7 is east and c7 is hazel, then c8 is cobalt.
64. If c8 is cobalt, then c8 is east.
65. If c7 is west and c7 is amber, then c8 is maple.
66. If c8 is maple, then c8 is south.
67. If c8 is north and c8 is willow, then c9 is coral.
68. If c9 is coral, then c9 is west.
69. If c8 is east and c8 is cobalt, then c9 is meadow.
70. If c9 is meadow, then c9 is east.
71. If c8 is west and c8 is amber, then c9 is lime.
72. If c9 is lime, then c9 is north.
73. If c8 is south and c8 is maple, then c9 is elm.
74. If c9 is elm, then c9 is south.
75. If c9 is south and c9 is elm, then c10 is orchid.
76. If c10 is orchid, then c10 is south.
77. If c9 is north and c9 is lime, then c10 is cobalt.
78. If c10 is cobalt, then c10 is north.
79. If c9 is east and c9 is meadow, then c10 is harbor.
80. If c10 is harbor, then c10 is east.
81. If c9 is west and c9 is coral, then c10 is meadow.
82. If c10 is meadow, then c10 is west.
83. If c10 is east and c10 is harbor, then c11 is lime.
84. If c11 is lime, then c11 is south.
85. If c10 is west and c10 is meadow, then c11 is teal.
86. If c11 is teal, then c11 is west.
87. If c10 is south and c10 is orchid, then c11 is orchid.
88. If c11 is orchid, then c11 is north.
89. If c10 is north and c10 is cobalt, then c11 is willow.
90. If c11 is willow, then c11 is east.
91. If c11 is east and c11 is willow, then c12 is hazel.
92. If c12 is hazel, then c12 is north.
93. If c11 is south and c11 is lime, then c12 is coral.
94. If c12 is coral, then c12 is south.
95. If c11 is west and c11 is teal, then c12 is cobalt.
96. If c12 is cobalt, then c12 is east.
97. If c11 is north and c11 is orchid, then c12 is maple.
98. If c12 is maple, then c12 is west.
99. If c12 is south and c12 is coral, then c13 is coral.
100. If c13 is coral, then c13 is south.
101. If c12 is west and c12 is maple, then c13 is meadow.
102. If c13 is meadow, then c13 is west.
103. If c12 is north and c12 is hazel, then c13 is lime.
104. If c13 is lime, then c13 is north.
105. If c12 is east and c12 is cobalt, then c13 is harbor.
106. If c13 is harbor, then c13 is east.
Which state applies to c13?
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
</constants>
<predicates>
Ax: x is elm
Bx: x is willow
Cx: x is teal
Dx: x is orchid
Ex: x is maple
Fx: x is juniper
Gx: x is granite
Hx: x is lime
Ix: x is meadow
Jx: x is coral
Kx: x is harbor
Lx: x is hazel
Mx: x is cobalt
Nx: x is amber
Ox: x is north
Px: x is south
Qx: x is east
Rx: x is west
</predicates>
<premises>
G(c0)
R(c0)
P(c0) & G(c0) -> E(c1)
E(c1) -> P(c1)
Q(c0) & G(c0) -> D(c1)
D(c1) -> Q(c1)
R(c0) & G(c0) -> A(c1)
A(c1) -> O(c1)
O(c0) & G(c0) -> K(c1)
K(c1) -> R(c1)
O(c1) & A(c1) -> D(c2)
D(c2) -> R(c2)
P(c1) & E(c1) -> H(c2)
H(c2) -> P(c2)
R(c1) & K(c1) -> E(c2)
E(c2) -> O(c2)
Q(c1) & D(c1) -> C(c2)
C(c2) -> Q(c2)
O(c2) & E(c2) -> N(c3)
N(c3) -> P(c3)
R(c2) & D(c2) -> I(c3)
I(c3) -> O(c3)
P(c2) & H(c2) -> J(c3)
J(c3) -> R(c3)
Q(c2) & C(c2) -> B(c3)
B(c3) -> Q(c3)
R(c3) & J(c3) -> N(c4)
N(c4) -> R(c4)
Q(c3) & B(c3) -> H(c4)
H(c4) -> P(c4)
O(c3) & I(c3) -> B(c4)
B(c4) -> Q(c4)
P(c3) & N(c3) -> C(c4)
C(c4) -> O(c4)
P(c4) & H(c4) -> L(c5)
L(c5) -> O(c5)
Q(c4) & B(c4) -> N(c5)
N(c5) -> R(c5)
O(c4) & C(c4) -> H(c5)
H(c5) -> Q(c5)
R(c4) & N(c4) -> K(c5)
K(c5) -> P(c5)
O(c5) & L(c5) -> J(c6)
J(c6) -> Q(c6)
R(c5) & N(c5) -> M(c6)
M(c6) -> P(c6)
Q(c5) & H(c5) -> E(c6)
E(c6) -> O(c6)
P(c5) & K(c5) -> F(c6)
F(c6) -> R(c6)
P(c6) & M(c6) -> I(c7)
I(c7) -> P(c7)
R(c6) & F(c6) -> N(c7)
N(c7) -> R(c7)
Q(c6) & J(c6) -> A(c7)
A(c7) -> O(c7)
O(c6) & E(c6) -> L(c7)
L(c7) -> Q(c7)
P(c7) & I(c7) -> N(c8)
N(c8) -> R(c8)
O(c7) & A(c7) -> B(c8)
B(c8) -> O(c8)
Q(c7) & L(c7) -> M(c8)
M(c8) -> Q(c8)
R(c7) & N(c7) -> E(c8)
E(c8) -> P(c8)
O(c8) & B(c8) -> J(c9)
J(c9) -> R(c9)
Q(c8) & M(c8) -> I(c9)
I(c9) -> Q(c9)
R(c8) & N(c8) -> H(c9)
H(c9) -> O(c9)
P(c8) & E(c8) -> A(c9)
A(c9) -> P(c9)
P(c9) & A(c9) -> D(c10)
D(c10) -> P(c10)
O(c9) & H(c9) -> M(c10)
M(c10) -> O(c10)
Q(c9) & I(c9) -> K(c10)
K(c10) -> Q(c10)
R(c9) & J(c9) -> I(c10)
I(c10) -> R(c10)
Q(c10) & K(c10) -> H(c11)
H(c11) -> P(c11)
R(c10) & I(c10) -> C(c11)
C(c11) -> R(c11)
P(c10) & D(c10) -> D(c11)
D(c11) -> O(c11)
O(c10) & M(c10) -> B(c11)
B(c11) -> Q(c11)
Q(c11) & B(c11) -> L(c12)
L(c12) -> O(c12)
P(c11) & H(c11) -> J(c12)
J(c12) -> P(c12)
R(c11) & C(c11) -> M(c12)
M(c12) -> Q(c12)
O(c11) & D(c11) -> E(c12)
E(c12) -> R(c12)
P(c12) & J(c12) -> J(c13)
J(c13) -> P(c13)
R(c12) & E(c12) -> I(c13)
I(c13) -> R(c13)
O(c12) & L(c12) -> H(c13)
H(c13) -> O(c13)
Q(c12) & M(c12) -> K(c13)
K(c13) -> Q(c13)
</premises>
<proof>
G(c0) ; R
R(c0) ; R
A(c1) ; ->E
O(c1) ; ->E
D(c2) ; ->E
R(c2) ; ->E
I(c3) ; ->E
O(c3) ; ->E
B(c4) ; ->E
Q(c4) ; ->E
N(c5) ; ->E
R(c5) ; ->E
M(c6) ; ->E
P(c6) ; ->E
I(c7) ; ->E
P(c7) ; ->E
N(c8) ; ->E
R(c8) ; ->E
H(c9) ; ->E
O(c9) ; ->E
M(c10) ; ->E
O(c10) ; ->E
B(c11) ; ->E
Q(c11) ; ->E
L(c12) ; ->E
O(c12) ; ->E
H(c13) ; ->E
</proof>
<conclusion>
H(c13)
</conclusion>
</formal>
<answer>
lime
</answer><|endoftext|><question>
1. c0 is juniper.
2. c0 is south.
3. If c0 is east and c0 is juniper, then c1 is hazel.
4. If c1 is hazel, then c1 is west.
5. If c0 is south and c0 is juniper, then c1 is harbor.
6. If c1 is harbor, then c1 is east.
7. If c0 is west and c0 is juniper, then c1 is amber.
8. If c1 is amber, then c1 is north.
9. If c0 is north and c0 is juniper, then c1 is pearl.
10. If c1 is pearl, then c1 is south.
11. If c1 is north and c1 is amber, then c2 is cobalt.
12. If c2 is cobalt, then c2 is north.
13. If c1 is south and c1 is pearl, then c2 is cedar.
14. If c2 is cedar, then c2 is south.
15. If c1 is east and c1 is harbor, then c2 is meadow.
16. If c2 is meadow, then c2 is west.
17. If c1 is west and c1 is hazel, then c2 is maple.
18. If c2 is maple, then c2 is east.
19. If c2 is west and c2 is meadow, then c3 is granite.
20. If c3 is granite, then c3 is south.
21. If c2 is east and c2 is maple, then c3 is cobalt.
22. If c3 is cobalt, then c3 is north.
23. If c2 is north and c2 is cobalt, then c3 is cedar.
24. If c3 is cedar, then c3 is east.
25. If c2 is south and c2 is cedar, then c3 is coral.
26. If c3 is coral, then c3 is west.
27. If c3 is west and c3 is coral, then c4 is hazel.
28. If c4 is hazel, then c4 is west.
29. If c3 is east and c3 is cedar, then c4 is maple.
30. If c4 is maple, then c4 is east.
31. If c3 is south and c3 is granite, then c4 is amber.
32. If c4 is amber, then c4 is north.
33. If c3 is north and c3 is cobalt, then c4 is granite.
34. If c4 is granite, then c4 is south.
35. If c4 is south and c4 is granite, then c5 is cedar.
36. If c5 is cedar, then c5 is south.
37. If c4 is east and c4 is maple, then c5 is elm.
38. If c5 is elm, then c5 is east.
39. If c4 is west and c4 is hazel, then c5 is amber.
40. If c5 is amber, then c5 is north.
41. If c4 is north and c4 is amber, then c5 is pearl.
42. If c5 is pearl, then c5 is west.
43. If c5 is north and c5 is amber, then c6 is maple.
44. If c6 is maple, then c6 is west.
45. If c5 is south and c5 is cedar, then c6 is laurel.
46. If c6 is laurel, then c6 is south.
47. If c5 is east and c5 is elm, then c6 is granite.
48. If c6 is granite, then c6 is east.
49. If c5 is west and c5 is pearl, then c6 is cobalt.
50. If c6 is cobalt, then c6 is north.
51. If c6 is north and c6 is cobalt, then c7 is meadow.
52. If c7 is meadow, then c7 is east.
53. If c6 is west and c6 is maple, then c7 is elm.
54. If c7 is elm, then c7 is west.
55. If c6 is east and c6 is granite, then c7 is cedar.
56. If c7 is cedar, then c7 is south.
57. If c6 is south and c6 is laurel, then c7 is coral.
58. If c7 is coral, then c7 is north.
59. If c7 is east and c7 is meadow, then c8 is amber.
60. If c8 is amber, then c8 is south.
61. If c7 is west and c7 is elm, then c8 is hazel.
62. If c8 is hazel, then c8 is east.
63. If c7 is north and c7 is coral, then c8 is orchid.
64. If c8 is orchid, then c8 is west.
65. If c7 is south and c7 is cedar, then c8 is cobalt.
66. If c8 is cobalt, then c8 is north.
67. If c8 is north and c8 is cobalt, then c9 is hazel.
68. If c9 is hazel, then c9 is north.
69. If c8 is west and c8 is orchid, then c9 is elm.
70. If c9 is elm, then c9 is south.
71. If c8 is south and c8 is amber, then c9 is laurel.
72. If c9 is laurel, then c9 is east.
73. If c8 is east and c8 is hazel, then c9 is poppy.
74. If c9 is poppy, then c9 is west.
75. If c9 is east and c9 is laurel, then c10 is laurel.
76. If c10 is laurel, then c10 is west.
77. If c9 is north and c9 is hazel, then c10 is elm.
78. If c10 is elm, then c10 is south.
79. If c9 is south and c9 is elm, then c10 is orchid.
80. If c10 is orchid, then c10 is east.
81. If c9 is west and c9 is poppy, then c10 is cedar.
82. If c10 is cedar, then c10 is north.
83. If c10 is north and c10 is cedar, then c11 is poppy.
84. If c11 is poppy, then c11 is north.
85. If c10 is east and c10 is orchid, then c11 is coral.
86. If c11 is coral, then c11 is west.
87. If c10 is south and c10 is elm, then c11 is harbor.
88. If c11 is harbor, then c11 is south.
89. If c10 is west and c10 is laurel, then c11 is laurel.
90. If c11 is laurel, then c11 is east.
91. If c11 is north and c11 is poppy, then c12 is cobalt.
92. If c12 is cobalt, then c12 is west.
93. If c11 is south and c11 is harbor, then c12 is coral.
94. If c12 is coral, then c12 is south.
95. If c11 is east and c11 is laurel, then c12 is poppy.
96. If c12 is poppy, then c12 is east.
97. If c11 is west and c11 is coral, then c12 is laurel.
98. If c12 is laurel, then c12 is north.
99. If c12 is west and c12 is cobalt, then c13 is pearl.
100. If c13 is pearl, then c13 is north.
101. If c12 is east and c12 is poppy, then c13 is coral.
102. If c13 is coral, then c13 is west.
103. If c12 is south and c12 is coral, then c13 is cobalt.
104. If c13 is cobalt, then c13 is south.
105. If c12 is north and c12 is laurel, then c13 is orchid.
106. If c13 is orchid, then c13 is east.
107. If c13 is east and c13 is orchid, then c14 is granite.
108. If c14 is granite, then c14 is east.
109. If c13 is west and c13 is coral, then c14 is orchid.
110. If c14 is orchid, then c14 is west.
111. If c13 is north and c13 is pearl, then c14 is meadow.
112. If c14 is meadow, then c14 is south.
113. If c13 is south and c13 is cobalt, then c14 is poppy.
114. If c14 is poppy, then c14 is north.
Which state applies to c14?
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
</constants>
<predicates>
Ax: x is harbor
Bx: x is granite
Cx: x is hazel
Dx: x is amber
Ex: x is orchid
Fx: x is cobalt
Gx: x is laurel
Hx: x is elm
Ix: x is poppy
Jx: x is maple
Kx: x is meadow
Lx: x is juniper
Mx: x is cedar
Nx: x is coral
Ox: x is pearl
Px: x is north
Qx: x is south
Rx: x is east
Sx: x is west
</predicates>
<premises>
L(c0)
Q(c0)
R(c0) & L(c0) -> C(c1)
C(c1) -> S(c1)
Q(c0) & L(c0) -> A(c1)
A(c1) -> R(c1)
S(c0) & L(c0) -> D(c1)
D(c1) -> P(c1)
P(c0) & L(c0) -> O(c1)
O(c1) -> Q(c1)
P(c1) & D(c1) -> F(c2)
F(c2) -> P(c2)
Q(c1) & O(c1) -> M(c2)
M(c2) -> Q(c2)
R(c1) & A(c1) -> K(c2)
K(c2) -> S(c2)
S(c1) & C(c1) -> J(c2)
J(c2) -> R(c2)
S(c2) & K(c2) -> B(c3)
B(c3) -> Q(c3)
R(c2) & J(c2) -> F(c3)
F(c3) -> P(c3)
P(c2) & F(c2) -> M(c3)
M(c3) -> R(c3)
Q(c2) & M(c2) -> N(c3)
N(c3) -> S(c3)
S(c3) & N(c3) -> C(c4)
C(c4) -> S(c4)
R(c3) & M(c3) -> J(c4)
J(c4) -> R(c4)
Q(c3) & B(c3) -> D(c4)
D(c4) -> P(c4)
P(c3) & F(c3) -> B(c4)
B(c4) -> Q(c4)
Q(c4) & B(c4) -> M(c5)
M(c5) -> Q(c5)
R(c4) & J(c4) -> H(c5)
H(c5) -> R(c5)
S(c4) & C(c4) -> D(c5)
D(c5) -> P(c5)
P(c4) & D(c4) -> O(c5)
O(c5) -> S(c5)
P(c5) & D(c5) -> J(c6)
J(c6) -> S(c6)
Q(c5) & M(c5) -> G(c6)
G(c6) -> Q(c6)
R(c5) & H(c5) -> B(c6)
B(c6) -> R(c6)
S(c5) & O(c5) -> F(c6)
F(c6) -> P(c6)
P(c6) & F(c6) -> K(c7)
K(c7) -> R(c7)
S(c6) & J(c6) -> H(c7)
H(c7) -> S(c7)
R(c6) & B(c6) -> M(c7)
M(c7) -> Q(c7)
Q(c6) & G(c6) -> N(c7)
N(c7) -> P(c7)
R(c7) & K(c7) -> D(c8)
D(c8) -> Q(c8)
S(c7) & H(c7) -> C(c8)
C(c8) -> R(c8)
P(c7) & N(c7) -> E(c8)
E(c8) -> S(c8)
Q(c7) & M(c7) -> F(c8)
F(c8) -> P(c8)
P(c8) & F(c8) -> C(c9)
C(c9) -> P(c9)
S(c8) & E(c8) -> H(c9)
H(c9) -> Q(c9)
Q(c8) & D(c8) -> G(c9)
G(c9) -> R(c9)
R(c8) & C(c8) -> I(c9)
I(c9) -> S(c9)
R(c9) & G(c9) -> G(c10)
G(c10) -> S(c10)
P(c9) & C(c9) -> H(c10)
H(c10) -> Q(c10)
Q(c9) & H(c9) -> E(c10)
E(c10) -> R(c10)
S(c9) & I(c9) -> M(c10)
M(c10) -> P(c10)
P(c10) & M(c10) -> I(c11)
I(c11) -> P(c11)
R(c10) & E(c10) -> N(c11)
N(c11) -> S(c11)
Q(c10) & H(c10) -> A(c11)
A(c11) -> Q(c11)
S(c10) & G(c10) -> G(c11)
G(c11) -> R(c11)
P(c11) & I(c11) -> F(c12)
F(c12) -> S(c12)
Q(c11) & A(c11) -> N(c12)
N(c12) -> Q(c12)
R(c11) & G(c11) -> I(c12)
I(c12) -> R(c12)
S(c11) & N(c11) -> G(c12)
G(c12) -> P(c12)
S(c12) & F(c12) -> O(c13)
O(c13) -> P(c13)
R(c12) & I(c12) -> N(c13)
N(c13) -> S(c13)
Q(c12) & N(c12) -> F(c13)
F(c13) -> Q(c13)
P(c12) & G(c12) -> E(c13)
E(c13) -> R(c13)
R(c13) & E(c13) -> B(c14)
B(c14) -> R(c14)
S(c13) & N(c13) -> E(c14)
E(c14) -> S(c14)
P(c13) & O(c13) -> K(c14)
K(c14) -> Q(c14)
Q(c13) & F(c13) -> I(c14)
I(c14) -> P(c14)
</premises>
<proof>
L(c0) ; R
Q(c0) ; R
A(c1) ; ->E
R(c1) ; ->E
K(c2) ; ->E
S(c2) ; ->E
B(c3) ; ->E
Q(c3) ; ->E
D(c4) ; ->E
P(c4) ; ->E
O(c5) ; ->E
S(c5) ; ->E
F(c6) ; ->E
P(c6) ; ->E
K(c7) ; ->E
R(c7) ; ->E
D(c8) ; ->E
Q(c8) ; ->E
G(c9) ; ->E
R(c9) ; ->E
G(c10) ; ->E
S(c10) ; ->E
G(c11) ; ->E
R(c11) ; ->E
I(c12) ; ->E
R(c12) ; ->E
N(c13) ; ->E
S(c13) ; ->E
E(c14) ; ->E
</proof>
<conclusion>
E(c14)
</conclusion>
</formal>
<answer>
orchid
</answer><|endoftext|>
```

## Window 32399 summary: [{"tokens": 7682, "head": "<question> 1. c0 is pearl. 2. c0 is west. 3. If c0 is west and c0 is pearl, then", "tail": "ion> Q(c25) </conclusion> </formal> <answer> cedar </answer>"}]

## Window 40684 summary: [{"tokens": 4693, "head": "<question> 1. c0 is pearl. 2. c0 is south. 3. If c0 is east and c0 is pearl, the", "tail": "ion> M(c16) </conclusion> </formal> <answer> poppy </answer>"}, {"tokens": 3123, "head": "<question> 1. c0 is elm. 2. c0 is south. 3. If c0 is north and c0 is elm, then c", "tail": "ion> C(c11) </conclusion> </formal> <answer> poppy </answer>"}]

## Window 41239 summary: [{"tokens": 5644, "head": "<question> 1. c0 is granite. 2. c0 is north. 3. If c0 is east and c0 is granite,", "tail": "sion> Q(c19) </conclusion> </formal> <answer> teal </answer>"}, {"tokens": 2251, "head": "<question> 1. c0 is violet. 2. c0 is east. 3. If c0 is west and c0 is violet, th", "tail": "usion> F(c8) </conclusion> </formal> <answer> teal </answer>"}]

## Window 55985 summary: [{"tokens": 5989, "head": "<question> 1. c0 is ruby. 2. c0 is west. 3. If c0 is south and c0 is ruby, then ", "tail": "on> A(c20) </conclusion> </formal> <answer> meadow </answer>"}, {"tokens": 1721, "head": "<question> 1. c0 is juniper. 2. c0 is east. 3. If c0 is east and c0 is juniper, ", "tail": "sion> C(c6) </conclusion> </formal> <answer> poppy </answer>"}]

## Window 73180 summary: [{"tokens": 4347, "head": "<question> 1. c0 is amber. 2. c0 is east. 3. If c0 is west and c0 is amber, then", "tail": "ion> J(c15) </conclusion> </formal> <answer> pearl </answer>"}, {"tokens": 3139, "head": "<question> 1. c0 is violet. 2. c0 is north. 3. If c0 is east and c0 is violet, t", "tail": "sion> B(c11) </conclusion> </formal> <answer> teal </answer>"}]

## Window 73498 summary: [{"tokens": 4997, "head": "<question> 1. c0 is olive. 2. c0 is north. 3. If c0 is north and c0 is olive, th", "tail": "sion> L(c17) </conclusion> </formal> <answer> lime </answer>"}, {"tokens": 2508, "head": "<question> 1. c0 is maple. 2. c0 is north. 3. If c0 is west and c0 is maple, the", "tail": "on> C(c9) </conclusion> </formal> <answer> granite </answer>"}]

## Window 79728 summary: [{"tokens": 5316, "head": "<question> 1. c0 is elm. 2. c0 is north. 3. If c0 is west and c0 is elm, then c1", "tail": "ion> L(c18) </conclusion> </formal> <answer> hazel </answer>"}, {"tokens": 2523, "head": "<question> 1. c0 is elm. 2. c0 is south. 3. If c0 is west and c0 is elm, then c1", "tail": "on> E(c9) </conclusion> </formal> <answer> juniper </answer>"}]

## Window 82584 summary: [{"tokens": 5345, "head": "<question> 1. c0 is olive. 2. c0 is east. 3. If c0 is south and c0 is olive, the", "tail": "on> P(c18) </conclusion> </formal> <answer> willow </answer>"}, {"tokens": 2532, "head": "<question> 1. c0 is olive. 2. c0 is east. 3. If c0 is north and c0 is olive, the", "tail": "sion> F(c9) </conclusion> </formal> <answer> poppy </answer>"}]

## Window 121909 summary: [{"tokens": 7042, "head": "<question> 1. c0 is birch. 2. c0 is west. 3. If c0 is south and c0 is birch, the", "tail": "on> U(c23) </conclusion> </formal> <answer> harbor </answer>"}, {"tokens": 945, "head": "<question> 1. c0 is slate. 2. c0 is east. 3. If c0 is east and c0 is slate, then", "tail": "sion> D(c3) </conclusion> </formal> <answer> ivory </answer>"}]

## Window 125048 summary: [{"tokens": 3748, "head": "<question> 1. c0 is birch. 2. c0 is south. 3. If c0 is south and c0 is birch, th", "tail": "n> K(c13) </conclusion> </formal> <answer> juniper </answer>"}, {"tokens": 4056, "head": "<question> 1. c0 is amber. 2. c0 is east. 3. If c0 is north and c0 is amber, the", "tail": "sion> I(c14) </conclusion> </formal> <answer> lime </answer>"}]

## Window 125370 summary: [{"tokens": 5346, "head": "<question> 1. c0 is coral. 2. c0 is south. 3. If c0 is north and c0 is coral, th", "tail": "on> S(c18) </conclusion> </formal> <answer> laurel </answer>"}, {"tokens": 2818, "head": "<question> 1. c0 is cedar. 2. c0 is north. 3. If c0 is north and c0 is cedar, th", "tail": "ion> E(c10) </conclusion> </formal> <answer> poppy </answer>"}]

## Window 128123 summary: [{"tokens": 4095, "head": "<question> 1. c0 is granite. 2. c0 is south. 3. If c0 is south and c0 is granite", "tail": "ion> D(c14) </conclusion> </formal> <answer> maple </answer>"}, {"tokens": 3127, "head": "<question> 1. c0 is poppy. 2. c0 is north. 3. If c0 is south and c0 is poppy, th", "tail": "sion> B(c11) </conclusion> </formal> <answer> teal </answer>"}, {"tokens": 670, "head": "<question> 1. c0 is maple. 2. c0 is north. 3. If c0 is east and c0 is maple, the", "tail": "on> D(c2) </conclusion> </formal> <answer> granite </answer>"}]

## Window 128124 summary: [{"tokens": 5935, "head": "<question> 1. c0 is cobalt. 2. c0 is west. 3. If c0 is south and c0 is cobalt, t", "tail": "ion> U(c20) </conclusion> </formal> <answer> ivory </answer>"}, {"tokens": 2250, "head": "<question> 1. c0 is violet. 2. c0 is south. 3. If c0 is east and c0 is violet, t", "tail": "sion> G(c8) </conclusion> </formal> <answer> coral </answer>"}]

## Window 139294 summary: [{"tokens": 7357, "head": "<question> 1. c0 is maple. 2. c0 is north. 3. If c0 is east and c0 is maple, the", "tail": "ion> X(c24) </conclusion> </formal> <answer> coral </answer>"}]

## Window 154778 summary: [{"tokens": 7669, "head": "<question> 1. c0 is pearl. 2. c0 is south. 3. If c0 is south and c0 is pearl, th", "tail": "ion> H(c25) </conclusion> </formal> <answer> poppy </answer>"}]

## Window 165022 summary: [{"tokens": 6636, "head": "<question> 1. c0 is maple. 2. c0 is north. 3. If c0 is west and c0 is maple, the", "tail": "on> S(c22) </conclusion> </formal> <answer> willow </answer>"}, {"tokens": 1202, "head": "<question> 1. c0 is laurel. 2. c0 is east. 3. If c0 is west and c0 is laurel, th", "tail": "ion> D(c4) </conclusion> </formal> <answer> cobalt </answer>"}]

## Window 165158 summary: [{"tokens": 7028, "head": "<question> 1. c0 is harbor. 2. c0 is south. 3. If c0 is west and c0 is harbor, t", "tail": "on> U(c23) </conclusion> </formal> <answer> violet </answer>"}, {"tokens": 664, "head": "<question> 1. c0 is cedar. 2. c0 is north. 3. If c0 is east and c0 is cedar, the", "tail": "sion> E(c2) </conclusion> </formal> <answer> olive </answer>"}]

## Window 189139 summary: [{"tokens": 3787, "head": "<question> 1. c0 is harbor. 2. c0 is west. 3. If c0 is north and c0 is harbor, t", "tail": "on> H(c13) </conclusion> </formal> <answer> violet </answer>"}, {"tokens": 4066, "head": "<question> 1. c0 is juniper. 2. c0 is east. 3. If c0 is east and c0 is juniper, ", "tail": "ion> I(c14) </conclusion> </formal> <answer> slate </answer>"}]

## Window 191692 summary: [{"tokens": 4706, "head": "<question> 1. c0 is meadow. 2. c0 is west. 3. If c0 is north and c0 is meadow, t", "tail": "on> F(c16) </conclusion> </formal> <answer> orchid </answer>"}, {"tokens": 3097, "head": "<question> 1. c0 is laurel. 2. c0 is west. 3. If c0 is south and c0 is laurel, t", "tail": "sion> H(c11) </conclusion> </formal> <answer> lime </answer>"}]

## Window 210294 summary: [{"tokens": 5969, "head": "<question> 1. c0 is birch. 2. c0 is east. 3. If c0 is north and c0 is birch, the", "tail": "ion> Q(c20) </conclusion> </formal> <answer> olive </answer>"}, {"tokens": 1980, "head": "<question> 1. c0 is violet. 2. c0 is north. 3. If c0 is west and c0 is violet, t", "tail": "ion> A(c7) </conclusion> </formal> <answer> cobalt </answer>"}]

## Window 220532 summary: [{"tokens": 7349, "head": "<question> 1. c0 is olive. 2. c0 is north. 3. If c0 is south and c0 is olive, th", "tail": "ion> L(c24) </conclusion> </formal> <answer> poppy </answer>"}, {"tokens": 419, "head": "<question> 1. c0 is cobalt. 2. c0 is east. 3. If c0 is east and c0 is cobalt, th", "tail": "sion> C(c1) </conclusion> </formal> <answer> pearl </answer>"}]

## Window 224047 summary: [{"tokens": 5326, "head": "<question> 1. c0 is laurel. 2. c0 is east. 3. If c0 is south and c0 is laurel, t", "tail": "ion> M(c18) </conclusion> </formal> <answer> ivory </answer>"}, {"tokens": 2820, "head": "<question> 1. c0 is ruby. 2. c0 is north. 3. If c0 is west and c0 is ruby, then ", "tail": "sion> H(c10) </conclusion> </formal> <answer> lime </answer>"}]

## Window 229035 summary: [{"tokens": 4716, "head": "<question> 1. c0 is elm. 2. c0 is south. 3. If c0 is east and c0 is elm, then c1", "tail": "on> G(c16) </conclusion> </formal> <answer> violet </answer>"}, {"tokens": 3165, "head": "<question> 1. c0 is teal. 2. c0 is north. 3. If c0 is south and c0 is teal, then", "tail": "sion> J(c11) </conclusion> </formal> <answer> ruby </answer>"}]

## Window 251897 summary: [{"tokens": 4720, "head": "<question> 1. c0 is cobalt. 2. c0 is south. 3. If c0 is west and c0 is cobalt, t", "tail": "on> K(c16) </conclusion> </formal> <answer> violet </answer>"}, {"tokens": 2502, "head": "<question> 1. c0 is olive. 2. c0 is east. 3. If c0 is east and c0 is olive, then", "tail": "ion> H(c9) </conclusion> </formal> <answer> harbor </answer>"}, {"tokens": 682, "head": "<question> 1. c0 is laurel. 2. c0 is south. 3. If c0 is south and c0 is laurel, ", "tail": "sion> B(c2) </conclusion> </formal> <answer> olive </answer>"}]

## Window 256245 summary: [{"tokens": 4383, "head": "<question> 1. c0 is harbor. 2. c0 is west. 3. If c0 is north and c0 is harbor, t", "tail": "on> B(c15) </conclusion> </formal> <answer> willow </answer>"}, {"tokens": 2798, "head": "<question> 1. c0 is violet. 2. c0 is south. 3. If c0 is south and c0 is violet, ", "tail": "ion> G(c10) </conclusion> </formal> <answer> ivory </answer>"}, {"tokens": 945, "head": "<question> 1. c0 is coral. 2. c0 is west. 3. If c0 is east and c0 is coral, then", "tail": "ion> B(c3) </conclusion> </formal> <answer> orchid </answer>"}]

## Window 256246 summary: [{"tokens": 6651, "head": "<question> 1. c0 is orchid. 2. c0 is west. 3. If c0 is north and c0 is orchid, t", "tail": "sion> P(c22) </conclusion> </formal> <answer> teal </answer>"}, {"tokens": 1201, "head": "<question> 1. c0 is hazel. 2. c0 is west. 3. If c0 is west and c0 is hazel, then", "tail": "usion> D(c4) </conclusion> </formal> <answer> lime </answer>"}]
