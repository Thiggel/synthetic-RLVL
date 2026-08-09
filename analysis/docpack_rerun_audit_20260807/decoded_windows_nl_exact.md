# Decoded-batch audit examples (document-preserving docpack loader)

## Window 0 (3 documents, 36 pad tokens)
```
1. c0 is olive.
2. c0 is south.
3. If c0 is east and c0 is olive, then c1 is elm.
4. If c1 is elm, then c1 is south.
5. If c0 is north and c0 is olive, then c1 is violet.
6. If c1 is violet, then c1 is east.
7. If c0 is south and c0 is olive, then c1 is ivory.
8. If c1 is ivory, then c1 is north.
9. If c0 is west and c0 is olive, then c1 is harbor.
10. If c1 is harbor, then c1 is west.
11. If c1 is north and c1 is ivory, then c2 is violet.
12. If c2 is violet, then c2 is west.
13. If c1 is west and c1 is harbor, then c2 is orchid.
14. If c2 is orchid, then c2 is east.
15. If c1 is east and c1 is violet, then c2 is harbor.
16. If c2 is harbor, then c2 is north.
17. If c1 is south and c1 is elm, then c2 is ivory.
18. If c2 is ivory, then c2 is south.
19. If c2 is east and c2 is orchid, then c3 is harbor.
20. If c3 is harbor, then c3 is east.
21. If c2 is south and c2 is ivory, then c3 is orchid.
22. If c3 is orchid, then c3 is north.
23. If c2 is north and c2 is harbor, then c3 is ivory.
24. If c3 is ivory, then c3 is south.
25. If c2 is west and c2 is violet, then c3 is violet.
26. If c3 is violet, then c3 is west.
27. If c3 is east and c3 is harbor, then c4 is orchid.
28. If c4 is orchid, then c4 is east.
29. If c3 is south and c3 is ivory, then c4 is violet.
30. If c4 is violet, then c4 is south.
31. If c3 is north and c3 is orchid, then c4 is harbor.
32. If c4 is harbor, then c4 is west.
33. If c3 is west and c3 is violet, then c4 is elm.
34. If c4 is elm, then c4 is north.
35. If c4 is south and c4 is violet, then c5 is ivory.
36. If c5 is ivory, then c5 is south.
37. If c4 is east and c4 is orchid, then c5 is elm.
38. If c5 is elm, then c5 is north.
39. If c4 is north and c4 is elm, then c5 is harbor.
40. If c5 is harbor, then c5 is west.
41. If c4 is west and c4 is harbor, then c5 is maple.
42. If c5 is maple, then c5 is east.
43. If c5 is south and c5 is ivory, then c6 is maple.
44. If c6 is maple, then c6 is east.
45. If c5 is east and c5 is maple, then c6 is violet.
46. If c6 is violet, then c6 is north.
47. If c5 is west and c5 is harbor, then c6 is orchid.
48. If c6 is orchid, then c6 is south.
49. If c5 is north and c5 is elm, then c6 is harbor.
50. If c6 is harbor, then c6 is west.
Which state applies to c6?

Solution:
Derivation:
c0 is olive.
c0 is south.
c1 is ivory.
c1 is north.
c2 is violet.
c2 is west.
c3 is violet.
c3 is west.
c4 is elm.
c4 is north.
c5 is harbor.
c5 is west.
c6 is orchid.

Final answer: orchid<|endoftext|>1. c0 is juniper.
2. c0 is west.
3. If c0 is north and c0 is juniper, then c1 is amber.
4. If c1 is amber, then c1 is west.
5. If c0 is east and c0 is juniper, then c1 is granite.
6. If c1 is granite, then c1 is north.
7. If c0 is west and c0 is juniper, then c1 is teal.
8. If c1 is teal, then c1 is east.
9. If c0 is south and c0 is juniper, then c1 is meadow.
10. If c1 is meadow, then c1 is south.
11. If c1 is north and c1 is granite, then c2 is amber.
12. If c2 is amber, then c2 is east.
13. If c1 is south and c1 is meadow, then c2 is slate.
14. If c2 is slate, then c2 is south.
15. If c1 is west and c1 is amber, then c2 is ivory.
16. If c2 is ivory, then c2 is north.
17. If c1 is east and c1 is teal, then c2 is meadow.
18. If c2 is meadow, then c2 is west.
19. If c2 is east and c2 is amber, then c3 is cedar.
20. If c3 is cedar, then c3 is north.
21. If c2 is south and c2 is slate, then c3 is laurel.
22. If c3 is laurel, then c3 is east.
23. If c2 is north and c2 is ivory, then c3 is poppy.
24. If c3 is poppy, then c3 is west.
25. If c2 is west and c2 is meadow, then c3 is willow.
26. If c3 is willow, then c3 is south.
27. If c3 is west and c3 is poppy, then c4 is olive.
28. If c4 is olive, then c4 is east.
29. If c3 is north and c3 is cedar, then c4 is granite.
30. If c4 is granite, then c4 is north.
31. If c3 is east and c3 is laurel, then c4 is laurel.
32. If c4 is laurel, then c4 is west.
33. If c3 is south and c3 is willow, then c4 is ivory.
34. If c4 is ivory, then c4 is south.
35. If c4 is west and c4 is laurel, then c5 is amber.
36. If c5 is amber, then c5 is south.
37. If c4 is north and c4 is granite, then c5 is harbor.
38. If c5 is harbor, then c5 is west.
39. If c4 is east and c4 is olive, then c5 is laurel.
40. If c5 is laurel, then c5 is east.
41. If c4 is south and c4 is ivory, then c5 is olive.
42. If c5 is olive, then c5 is north.
43. If c5 is south and c5 is amber, then c6 is ivory.
44. If c6 is ivory, then c6 is west.
45. If c5 is west and c5 is harbor, then c6 is harbor.
46. If c6 is harbor, then c6 is north.
47. If c5 is north and c5 is olive, then c6 is teal.
48. If c6 is teal, then c6 is south.
49. If c5 is east and c5 is laurel, then c6 is meadow.
50. If c6 is meadow, then c6 is east.
51. If c6 is east and c6 is meadow, then c7 is willow.
52. If c7 is willow, then c7 is west.
53. If c6 is west and c6 is ivory, then c7 is laurel.
54. If c7 is laurel, then c7 is south.
55. If c6 is north and c6 is harbor, then c7 is granite.
56. If c7 is granite, then c7 is east.
57. If c6 is south and c6 is teal, then c7 is hazel.
58. If c7 is hazel, then c7 is north.
59. If c7 is west and c7 is willow, then c8 is cedar.
60. If c8 is cedar, then c8 is south.
61. If c7 is east and c7 is granite, then c8 is willow.
62. If c8 is willow, then c8 is north.
63. If c7 is south and c7 is laurel, then c8 is poppy.
64. If c8 is poppy, then c8 is east.
65. If c7 is north and c7 is hazel, then c8 is teal.
66. If c8 is teal, then c8 is west.
67. If c8 is west and c8 is teal, then c9 is amber.
68. If c9 is amber, then c9 is north.
69. If c8 is north and c8 is willow, then c9 is slate.
70. If c9 is slate, then c9 is east.
71. If c8 is east and c8 is poppy, then c9 is willow.
72. If c9 is willow, then c9 is south.
73. If c8 is south and c8 is cedar, then c9 is teal.
74. If c9 is teal, then c9 is west.
75. If c9 is south and c9 is willow, then c10 is hazel.
76. If c10 is hazel, then c10 is south.
77. If c9 is west and c9 is teal, then c10 is granite.
78. If c10 is granite, then c10 is north.
79. If c9 is east and c9 is slate, then c10 is teal.
80. If c10 is teal, then c10 is east.
81. If c9 is north and c9 is amber, then c10 is ivory.
82. If c10 is ivory, then c10 is west.
83. If c10 is south and c10 is hazel, then c11 is meadow.
84. If c11 is meadow, then c11 is north.
85. If c10 is north and c10 is granite, then c11 is ivory.
86. If c11 is ivory, then c11 is south.
87. If c10 is east and c10 is teal, then c11 is granite.
88. If c11 is granite, then c11 is west.
89. If c10 is west and c10 is ivory, then c11 is harbor.
90. If c11 is harbor, then c11 is east.
91. If c11 is east and c11 is harbor, then c12 is harbor.
92. If c12 is harbor, then c12 is east.
93. If c11 is west and c11 is granite, then c12 is granite.
94. If c12 is granite, then c12 is north.
95. If c11 is north and c11 is meadow, then c12 is olive.
96. If c12 is olive, then c12 is south.
97. If c11 is south and c11 is ivory, then c12 is meadow.
98. If c12 is meadow, then c12 is west.
99. If c12 is west and c12 is meadow, then c13 is laurel.
100. If c13 is laurel, then c13 is east.
101. If c12 is south and c12 is olive, then c13 is slate.
102. If c13 is slate, then c13 is south.
103. If c12 is east and c12 is harbor, then c13 is olive.
104. If c13 is olive, then c13 is north.
105. If c12 is north and c12 is granite, then c13 is ivory.
106. If c13 is ivory, then c13 is west.
107. If c13 is west and c13 is ivory, then c14 is laurel.
108. If c14 is laurel, then c14 is west.
109. If c13 is south and c13 is slate, then c14 is harbor.
110. If c14 is harbor, then c14 is south.
111. If c13 is east and c13 is laurel, then c14 is orchid.
112. If c14 is orchid, then c14 is north.
113. If c13 is north and c13 is olive, then c14 is slate.
114. If c14 is slate, then c14 is east.
Which state applies to c14?

Solution:
Derivation:
c0 is juniper.
c0 is west.
c1 is teal.
c1 is east.
c2 is meadow.
c2 is west.
c3 is willow.
c3 is south.
c4 is ivory.
c4 is south.
c5 is olive.
c5 is north.
c6 is teal.
c6 is south.
c7 is hazel.
c7 is north.
c8 is teal.
c8 is west.
c9 is amber.
c9 is north.
c10 is ivory.
c10 is west.
c11 is harbor.
c11 is east.
c12 is harbor.
c12 is east.
c13 is olive.
c13 is north.
c14 is slate.

Final answer: slate<|endoftext|>1. c0 is amber.
2. c0 is north.
3. If c0 is west and c0 is amber, then c1 is birch.
4. If c1 is birch, then c1 is west.
5. If c0 is east and c0 is amber, then c1 is maple.
6. If c1 is maple, then c1 is south.
7. If c0 is south and c0 is amber, then c1 is hazel.
8. If c1 is hazel, then c1 is east.
9. If c0 is north and c0 is amber, then c1 is ruby.
10. If c1 is ruby, then c1 is north.
11. If c1 is north and c1 is ruby, then c2 is ruby.
12. If c2 is ruby, then c2 is north.
13. If c1 is west and c1 is birch, then c2 is birch.
14. If c2 is birch, then c2 is east.
15. If c1 is south and c1 is maple, then c2 is maple.
16. If c2 is maple, then c2 is west.
17. If c1 is east and c1 is hazel, then c2 is hazel.
18. If c2 is hazel, then c2 is south.
19. If c2 is east and c2 is birch, then c3 is ruby.
20. If c3 is ruby, then c3 is east.
21. If c2 is south and c2 is hazel, then c3 is birch.
22. If c3 is birch, then c3 is north.
23. If c2 is west and c2 is maple, then c3 is hazel.
24. If c3 is hazel, then c3 is south.
25. If c2 is north and c2 is ruby, then c3 is maple.
26. If c3 is maple, then c3 is west.
27. If c3 is north and c3 is birch, then c4 is hazel.
28. If c4 is hazel, then c4 is east.
29. If c3 is east and c3 is ruby, then c4 is ruby.
30. If c4 is ruby, then c4 is west.
31. If c3 is south and c3 is hazel, then c4 is maple.
32. If c4 is maple, then c4 is north.
33. If c3 is west and c3 is maple, then c4 is birch.
34. If c4 is birch, then c4 is south.
35. If c4 is east and c4 is hazel, then c5 is maple.
36. If c5 is maple, then c5 is south.
37. If c4 is west and c4 is ruby, then c5 is birch.
38. If c5 is birch, then c5 is north.
39. If c4 is south and c4 is birch, then c5 is ruby.
40. If c5 is ruby, then c5 is east.
41. If c4 is north and c4 is maple, then c5 is teal.
42. If c5 is teal, then c5 is west.
Which state applies to c5?

Solution:
Derivation:
c0 is amber.
c0 is north.
c1 is ruby.
c1 is north.
c2 is ruby.
c2 is north.
c3 is maple.
c3 is west.
c4 is birch.
c4 is south.
c5 is ruby.

Final answer: ruby<|endoftext|>
```

## Window 1 (3 documents, 44 pad tokens)
```
1. c0 is juniper.
2. c0 is south.
3. If c0 is south and c0 is juniper, then c1 is ivory.
4. If c1 is ivory, then c1 is south.
5. If c0 is east and c0 is juniper, then c1 is birch.
6. If c1 is birch, then c1 is east.
7. If c0 is west and c0 is juniper, then c1 is slate.
8. If c1 is slate, then c1 is north.
9. If c0 is north and c0 is juniper, then c1 is coral.
10. If c1 is coral, then c1 is west.
11. If c1 is north and c1 is slate, then c2 is coral.
12. If c2 is coral, then c2 is west.
13. If c1 is south and c1 is ivory, then c2 is ivory.
14. If c2 is ivory, then c2 is north.
15. If c1 is east and c1 is birch, then c2 is slate.
16. If c2 is slate, then c2 is south.
17. If c1 is west and c1 is coral, then c2 is birch.
18. If c2 is birch, then c2 is east.
19. If c2 is west and c2 is coral, then c3 is teal.
20. If c3 is teal, then c3 is west.
21. If c2 is south and c2 is slate, then c3 is coral.
22. If c3 is coral, then c3 is north.
23. If c2 is east and c2 is birch, then c3 is birch.
24. If c3 is birch, then c3 is east.
25. If c2 is north and c2 is ivory, then c3 is slate.
26. If c3 is slate, then c3 is south.
27. If c3 is west and c3 is teal, then c4 is coral.
28. If c4 is coral, then c4 is south.
29. If c3 is north and c3 is coral, then c4 is birch.
30. If c4 is birch, then c4 is west.
31. If c3 is south and c3 is slate, then c4 is ivory.
32. If c4 is ivory, then c4 is north.
33. If c3 is east and c3 is birch, then c4 is teal.
34. If c4 is teal, then c4 is east.
35. If c4 is east and c4 is teal, then c5 is birch.
36. If c5 is birch, then c5 is north.
37. If c4 is north and c4 is ivory, then c5 is ivory.
38. If c5 is ivory, then c5 is east.
39. If c4 is west and c4 is birch, then c5 is coral.
40. If c5 is coral, then c5 is south.
41. If c4 is south and c4 is coral, then c5 is teal.
42. If c5 is teal, then c5 is west.
Which state applies to c5?

Solution:
Derivation:
c0 is juniper.
c0 is south.
c1 is ivory.
c1 is south.
c2 is ivory.
c2 is north.
c3 is slate.
c3 is south.
c4 is ivory.
c4 is north.
c5 is ivory.

Final answer: ivory<|endoftext|>1. c0 is lime.
2. c0 is north.
3. If c0 is south and c0 is lime, then c1 is harbor.
4. If c1 is harbor, then c1 is west.
5. If c0 is west and c0 is lime, then c1 is coral.
6. If c1 is coral, then c1 is south.
7. If c0 is north and c0 is lime, then c1 is juniper.
8. If c1 is juniper, then c1 is north.
9. If c0 is east and c0 is lime, then c1 is ivory.
10. If c1 is ivory, then c1 is east.
11. If c1 is west and c1 is harbor, then c2 is willow.
12. If c2 is willow, then c2 is north.
13. If c1 is east and c1 is ivory, then c2 is maple.
14. If c2 is maple, then c2 is east.
15. If c1 is south and c1 is coral, then c2 is poppy.
16. If c2 is poppy, then c2 is west.
17. If c1 is north and c1 is juniper, then c2 is ivory.
18. If c2 is ivory, then c2 is south.
19. If c2 is north and c2 is willow, then c3 is juniper.
20. If c3 is juniper, then c3 is north.
21. If c2 is south and c2 is ivory, then c3 is violet.
22. If c3 is violet, then c3 is west.
23. If c2 is east and c2 is maple, then c3 is ivory.
24. If c3 is ivory, then c3 is south.
25. If c2 is west and c2 is poppy, then c3 is hazel.
26. If c3 is hazel, then c3 is east.
27. If c3 is west and c3 is violet, then c4 is maple.
28. If c4 is maple, then c4 is north.
29. If c3 is east and c3 is hazel, then c4 is harbor.
30. If c4 is harbor, then c4 is west.
31. If c3 is south and c3 is ivory, then c4 is pearl.
32. If c4 is pearl, then c4 is east.
33. If c3 is north and c3 is juniper, then c4 is coral.
34. If c4 is coral, then c4 is south.
35. If c4 is north and c4 is maple, then c5 is juniper.
36. If c5 is juniper, then c5 is west.
37. If c4 is south and c4 is coral, then c5 is hazel.
38. If c5 is hazel, then c5 is south.
39. If c4 is east and c4 is pearl, then c5 is cobalt.
40. If c5 is cobalt, then c5 is east.
41. If c4 is west and c4 is harbor, then c5 is willow.
42. If c5 is willow, then c5 is north.
43. If c5 is east and c5 is cobalt, then c6 is ivory.
44. If c6 is ivory, then c6 is south.
45. If c5 is north and c5 is willow, then c6 is cobalt.
46. If c6 is cobalt, then c6 is north.
47. If c5 is west and c5 is juniper, then c6 is pearl.
48. If c6 is pearl, then c6 is west.
49. If c5 is south and c5 is hazel, then c6 is hazel.
50. If c6 is hazel, then c6 is east.
51. If c6 is north and c6 is cobalt, then c7 is birch.
52. If c7 is birch, then c7 is north.
53. If c6 is east and c6 is hazel, then c7 is coral.
54. If c7 is coral, then c7 is south.
55. If c6 is west and c6 is pearl, then c7 is willow.
56. If c7 is willow, then c7 is west.
57. If c6 is south and c6 is ivory, then c7 is juniper.
58. If c7 is juniper, then c7 is east.
59. If c7 is south and c7 is coral, then c8 is harbor.
60. If c8 is harbor, then c8 is south.
61. If c7 is north and c7 is birch, then c8 is hazel.
62. If c8 is hazel, then c8 is north.
63. If c7 is west and c7 is willow, then c8 is juniper.
64. If c8 is juniper, then c8 is east.
65. If c7 is east and c7 is juniper, then c8 is willow.
66. If c8 is willow, then c8 is west.
67. If c8 is east and c8 is juniper, then c9 is harbor.
68. If c9 is harbor, then c9 is north.
69. If c8 is west and c8 is willow, then c9 is cobalt.
70. If c9 is cobalt, then c9 is south.
71. If c8 is north and c8 is hazel, then c9 is coral.
72. If c9 is coral, then c9 is east.
73. If c8 is south and c8 is harbor, then c9 is poppy.
74. If c9 is poppy, then c9 is west.
75. If c9 is west and c9 is poppy, then c10 is ivory.
76. If c10 is ivory, then c10 is south.
77. If c9 is east and c9 is coral, then c10 is poppy.
78. If c10 is poppy, then c10 is north.
79. If c9 is north and c9 is harbor, then c10 is hazel.
80. If c10 is hazel, then c10 is west.
81. If c9 is south and c9 is cobalt, then c10 is violet.
82. If c10 is violet, then c10 is east.
83. If c10 is east and c10 is violet, then c11 is maple.
84. If c11 is maple, then c11 is south.
85. If c10 is north and c10 is poppy, then c11 is juniper.
86. If c11 is juniper, then c11 is west.
87. If c10 is south and c10 is ivory, then c11 is coral.
88. If c11 is coral, then c11 is east.
89. If c10 is west and c10 is hazel, then c11 is poppy.
90. If c11 is poppy, then c11 is north.
91. If c11 is west and c11 is juniper, then c12 is cobalt.
92. If c12 is cobalt, then c12 is north.
93. If c11 is east and c11 is coral, then c12 is willow.
94. If c12 is willow, then c12 is east.
95. If c11 is south and c11 is maple, then c12 is pearl.
96. If c12 is pearl, then c12 is south.
97. If c11 is north and c11 is poppy, then c12 is violet.
98. If c12 is violet, then c12 is west.
Which state applies to c12?

Solution:
Derivation:
c0 is lime.
c0 is north.
c1 is juniper.
c1 is north.
c2 is ivory.
c2 is south.
c3 is violet.
c3 is west.
c4 is maple.
c4 is north.
c5 is juniper.
c5 is west.
c6 is pearl.
c6 is west.
c7 is willow.
c7 is west.
c8 is juniper.
c8 is east.
c9 is harbor.
c9 is north.
c10 is hazel.
c10 is west.
c11 is poppy.
c11 is north.
c12 is violet.

Final answer: violet<|endoftext|>1. c0 is violet.
2. c0 is west.
3. If c0 is west and c0 is violet, then c1 is olive.
4. If c1 is olive, then c1 is west.
5. If c0 is south and c0 is violet, then c1 is laurel.
6. If c1 is laurel, then c1 is south.
7. If c0 is east and c0 is violet, then c1 is ruby.
8. If c1 is ruby, then c1 is north.
9. If c0 is north and c0 is violet, then c1 is willow.
10. If c1 is willow, then c1 is east.
11. If c1 is north and c1 is ruby, then c2 is hazel.
12. If c2 is hazel, then c2 is south.
13. If c1 is west and c1 is olive, then c2 is willow.
14. If c2 is willow, then c2 is north.
15. If c1 is south and c1 is laurel, then c2 is ivory.
16. If c2 is ivory, then c2 is west.
17. If c1 is east and c1 is willow, then c2 is birch.
18. If c2 is birch, then c2 is east.
19. If c2 is south and c2 is hazel, then c3 is ruby.
20. If c3 is ruby, then c3 is south.
21. If c2 is west and c2 is ivory, then c3 is birch.
22. If c3 is birch, then c3 is north.
23. If c2 is east and c2 is birch, then c3 is willow.
24. If c3 is willow, then c3 is west.
25. If c2 is north and c2 is willow, then c3 is ivory.
26. If c3 is ivory, then c3 is east.
27. If c3 is south and c3 is ruby, then c4 is olive.
28. If c4 is olive, then c4 is east.
29. If c3 is north and c3 is birch, then c4 is juniper.
30. If c4 is juniper, then c4 is west.
31. If c3 is west and c3 is willow, then c4 is ivory.
32. If c4 is ivory, then c4 is north.
33. If c3 is east and c3 is ivory, then c4 is birch.
34. If c4 is birch, then c4 is south.
35. If c4 is west and c4 is juniper, then c5 is ivory.
36. If c5 is ivory, then c5 is north.
37. If c4 is south and c4 is birch, then c5 is olive.
38. If c5 is olive, then c5 is south.
39. If c4 is north and c4 is ivory, then c5 is hazel.
40. If c5 is hazel, then c5 is west.
41. If c4 is east and c4 is olive, then c5 is ruby.
42. If c5 is ruby, then c5 is east.
43. If c5 is west and c5 is hazel, then c6 is ruby.
44. If c6 is ruby, then c6 is east.
45. If c5 is north and c5 is ivory, then c6 is laurel.
46. If c6 is laurel, then c6 is south.
47. If c5 is south and c5 is olive, then c6 is willow.
48. If c6 is willow, then c6 is north.
49. If c5 is east and c5 is ruby, then c6 is olive.
50. If c6 is olive, then c6 is west.
51. If c6 is east and c6 is ruby, then c7 is olive.
52. If c7 is olive, then c7 is north.
53. If c6 is west and c6 is olive, then c7 is laurel.
54. If c7 is laurel, then c7 is east.
55. If c6 is north and c6 is willow, then c7 is willow.
56. If c7 is willow, then c7 is west.
57. If c6 is south and c6 is laurel, then c7 is birch.
58. If c7 is birch, then c7 is south.
59. If c7 is west and c7 is willow, then c8 is juniper.
60. If c8 is juniper, then c8 is east.
61. If c7 is north and c7 is olive, then c8 is hazel.
62. If c8 is hazel, then c8 is west.
63. If c7 is east and c7 is laurel, then c8 is willow.
64. If c8 is willow, then c8 is south.
65. If c7 is south and c7 is birch, then c8 is laurel.
66. If c8 is laurel, then c8 is north.
Which state applies to c8?

Solution:
Derivation:
c0 is violet.
c0 is west.
c1 is olive.
c1 is west.
c2 is willow.
c2 is north.
c3 is ivory.
c3 is east.
c4 is birch.
c4 is south.
c5 is olive.
c5 is south.
c6 is willow.
c6 is north.
c7 is willow.
c7 is west.
c8 is juniper.

Final answer: juniper<|endoftext|>
```

## Window 2 (5 documents, 27 pad tokens)
```
1. c0 is violet.
2. c0 is west.
3. If c0 is east and c0 is violet, then c1 is maple.
4. If c1 is maple, then c1 is south.
5. If c0 is west and c0 is violet, then c1 is juniper.
6. If c1 is juniper, then c1 is east.
7. If c0 is south and c0 is violet, then c1 is meadow.
8. If c1 is meadow, then c1 is west.
9. If c0 is north and c0 is violet, then c1 is lime.
10. If c1 is lime, then c1 is north.
11. If c1 is south and c1 is maple, then c2 is meadow.
12. If c2 is meadow, then c2 is north.
13. If c1 is north and c1 is lime, then c2 is maple.
14. If c2 is maple, then c2 is east.
15. If c1 is west and c1 is meadow, then c2 is willow.
16. If c2 is willow, then c2 is west.
17. If c1 is east and c1 is juniper, then c2 is birch.
18. If c2 is birch, then c2 is south.
19. If c2 is west and c2 is willow, then c3 is elm.
20. If c3 is elm, then c3 is south.
21. If c2 is north and c2 is meadow, then c3 is granite.
22. If c3 is granite, then c3 is east.
23. If c2 is east and c2 is maple, then c3 is maple.
24. If c3 is maple, then c3 is north.
25. If c2 is south and c2 is birch, then c3 is pearl.
26. If c3 is pearl, then c3 is west.
27. If c3 is north and c3 is maple, then c4 is maple.
28. If c4 is maple, then c4 is east.
29. If c3 is east and c3 is granite, then c4 is lime.
30. If c4 is lime, then c4 is south.
31. If c3 is south and c3 is elm, then c4 is granite.
32. If c4 is granite, then c4 is north.
33. If c3 is west and c3 is pearl, then c4 is ivory.
34. If c4 is ivory, then c4 is west.
35. If c4 is south and c4 is lime, then c5 is lime.
36. If c5 is lime, then c5 is north.
37. If c4 is north and c4 is granite, then c5 is granite.
38. If c5 is granite, then c5 is east.
39. If c4 is west and c4 is ivory, then c5 is meadow.
40. If c5 is meadow, then c5 is south.
41. If c4 is east and c4 is maple, then c5 is slate.
42. If c5 is slate, then c5 is west.
43. If c5 is north and c5 is lime, then c6 is elm.
44. If c6 is elm, then c6 is north.
45. If c5 is south and c5 is meadow, then c6 is olive.
46. If c6 is olive, then c6 is west.
47. If c5 is east and c5 is granite, then c6 is pearl.
48. If c6 is pearl, then c6 is south.
49. If c5 is west and c5 is slate, then c6 is juniper.
50. If c6 is juniper, then c6 is east.
51. If c6 is west and c6 is olive, then c7 is slate.
52. If c7 is slate, then c7 is west.
53. If c6 is south and c6 is pearl, then c7 is ivory.
54. If c7 is ivory, then c7 is south.
55. If c6 is north and c6 is elm, then c7 is willow.
56. If c7 is willow, then c7 is east.
57. If c6 is east and c6 is juniper, then c7 is birch.
58. If c7 is birch, then c7 is north.
59. If c7 is south and c7 is ivory, then c8 is meadow.
60. If c8 is meadow, then c8 is east.
61. If c7 is north and c7 is birch, then c8 is lime.
62. If c8 is lime, then c8 is north.
63. If c7 is west and c7 is slate, then c8 is birch.
64. If c8 is birch, then c8 is south.
65. If c7 is east and c7 is willow, then c8 is ivory.
66. If c8 is ivory, then c8 is west.
67. If c8 is north and c8 is lime, then c9 is granite.
68. If c9 is granite, then c9 is east.
69. If c8 is west and c8 is ivory, then c9 is slate.
70. If c9 is slate, then c9 is north.
71. If c8 is east and c8 is meadow, then c9 is juniper.
72. If c9 is juniper, then c9 is south.
73. If c8 is south and c8 is birch, then c9 is willow.
74. If c9 is willow, then c9 is west.
75. If c9 is north and c9 is slate, then c10 is olive.
76. If c10 is olive, then c10 is east.
77. If c9 is east and c9 is granite, then c10 is willow.
78. If c10 is willow, then c10 is west.
79. If c9 is west and c9 is willow, then c10 is juniper.
80. If c10 is juniper, then c10 is south.
81. If c9 is south and c9 is juniper, then c10 is granite.
82. If c10 is granite, then c10 is north.
83. If c10 is east and c10 is olive, then c11 is granite.
84. If c11 is granite, then c11 is north.
85. If c10 is south and c10 is juniper, then c11 is birch.
86. If c11 is birch, then c11 is east.
87. If c10 is north and c10 is granite, then c11 is pearl.
88. If c11 is pearl, then c11 is south.
89. If c10 is west and c10 is willow, then c11 is slate.
90. If c11 is slate, then c11 is west.
91. If c11 is south and c11 is pearl, then c12 is granite.
92. If c12 is granite, then c12 is north.
93. If c11 is north and c11 is granite, then c12 is slate.
94. If c12 is slate, then c12 is south.
95. If c11 is west and c11 is slate, then c12 is olive.
96. If c12 is olive, then c12 is west.
97. If c11 is east and c11 is birch, then c12 is meadow.
98. If c12 is meadow, then c12 is east.
Which state applies to c12?

Solution:
Derivation:
c0 is violet.
c0 is west.
c1 is juniper.
c1 is east.
c2 is birch.
c2 is south.
c3 is pearl.
c3 is west.
c4 is ivory.
c4 is west.
c5 is meadow.
c5 is south.
c6 is olive.
c6 is west.
c7 is slate.
c7 is west.
c8 is birch.
c8 is south.
c9 is willow.
c9 is west.
c10 is juniper.
c10 is south.
c11 is birch.
c11 is east.
c12 is meadow.

Final answer: meadow<|endoftext|>1. c0 is elm.
2. c0 is west.
3. If c0 is east and c0 is elm, then c1 is granite.
4. If c1 is granite, then c1 is east.
5. If c0 is south and c0 is elm, then c1 is cedar.
6. If c1 is cedar, then c1 is west.
7. If c0 is west and c0 is elm, then c1 is ivory.
8. If c1 is ivory, then c1 is north.
9. If c0 is north and c0 is elm, then c1 is cobalt.
10. If c1 is cobalt, then c1 is south.
11. If c1 is south and c1 is cobalt, then c2 is cedar.
12. If c2 is cedar, then c2 is north.
13. If c1 is east and c1 is granite, then c2 is granite.
14. If c2 is granite, then c2 is east.
15. If c1 is west and c1 is cedar, then c2 is slate.
16. If c2 is slate, then c2 is south.
17. If c1 is north and c1 is ivory, then c2 is juniper.
18. If c2 is juniper, then c2 is west.
19. If c2 is west and c2 is juniper, then c3 is maple.
20. If c3 is maple, then c3 is south.
21. If c2 is east and c2 is granite, then c3 is cobalt.
22. If c3 is cobalt, then c3 is north.
23. If c2 is south and c2 is slate, then c3 is granite.
24. If c3 is granite, then c3 is east.
25. If c2 is north and c2 is cedar, then c3 is juniper.
26. If c3 is juniper, then c3 is west.
27. If c3 is north and c3 is cobalt, then c4 is juniper.
28. If c4 is juniper, then c4 is north.
29. If c3 is east and c3 is granite, then c4 is granite.
30. If c4 is granite, then c4 is west.
31. If c3 is west and c3 is juniper, then c4 is ivory.
32. If c4 is ivory, then c4 is south.
33. If c3 is south and c3 is maple, then c4 is maple.
34. If c4 is maple, then c4 is east.
35. If c4 is north and c4 is juniper, then c5 is cedar.
36. If c5 is cedar, then c5 is east.
37. If c4 is south and c4 is ivory, then c5 is ivory.
38. If c5 is ivory, then c5 is north.
39. If c4 is east and c4 is maple, then c5 is slate.
40. If c5 is slate, then c5 is west.
41. If c4 is west and c4 is granite, then c5 is maple.
42. If c5 is maple, then c5 is south.
43. If c5 is north and c5 is ivory, then c6 is juniper.
44. If c6 is juniper, then c6 is west.
45. If c5 is south and c5 is maple, then c6 is ivory.
46. If c6 is ivory, then c6 is north.
47. If c5 is west and c5 is slate, then c6 is granite.
48. If c6 is granite, then c6 is south.
49. If c5 is east and c5 is cedar, then c6 is cedar.
50. If c6 is cedar, then c6 is east.
51. If c6 is west and c6 is juniper, then c7 is poppy.
52. If c7 is poppy, then c7 is south.
53. If c6 is south and c6 is granite, then c7 is juniper.
54. If c7 is juniper, then c7 is north.
55. If c6 is north and c6 is ivory, then c7 is cobalt.
56. If c7 is cobalt, then c7 is west.
57. If c6 is east and c6 is cedar, then c7 is granite.
58. If c7 is granite, then c7 is east.
59. If c7 is north and c7 is juniper, then c8 is maple.
60. If c8 is maple, then c8 is west.
61. If c7 is east and c7 is granite, then c8 is ivory.
62. If c8 is ivory, then c8 is north.
63. If c7 is west and c7 is cobalt, then c8 is juniper.
64. If c8 is juniper, then c8 is east.
65. If c7 is south and c7 is poppy, then c8 is cobalt.
66. If c8 is cobalt, then c8 is south.
Which state applies to c8?

Solution:
Derivation:
c0 is elm.
c0 is west.
c1 is ivory.
c1 is north.
c2 is juniper.
c2 is west.
c3 is maple.
c3 is south.
c4 is maple.
c4 is east.
c5 is slate.
c5 is west.
c6 is granite.
c6 is south.
c7 is juniper.
c7 is north.
c8 is maple.

Final answer: maple<|endoftext|>1. c0 is olive.
2. c0 is north.
3. If c0 is east and c0 is olive, then c1 is willow.
4. If c1 is willow, then c1 is south.
5. If c0 is north and c0 is olive, then c1 is maple.
6. If c1 is maple, then c1 is west.
7. If c0 is west and c0 is olive, then c1 is slate.
8. If c1 is slate, then c1 is east.
9. If c0 is south and c0 is olive, then c1 is ivory.
10. If c1 is ivory, then c1 is north.
11. If c1 is east and c1 is slate, then c2 is willow.
12. If c2 is willow, then c2 is east.
13. If c1 is north and c1 is ivory, then c2 is maple.
14. If c2 is maple, then c2 is west.
15. If c1 is west and c1 is maple, then c2 is ivory.
16. If c2 is ivory, then c2 is north.
17. If c1 is south and c1 is willow, then c2 is slate.
18. If c2 is slate, then c2 is south.
Which state applies to c2?

Solution:
Derivation:
c0 is olive.
c0 is north.
c1 is maple.
c1 is west.
c2 is ivory.

Final answer: ivory<|endoftext|>1. c0 is olive.
2. c0 is west.
3. If c0 is north and c0 is olive, then c1 is willow.
4. If c1 is willow, then c1 is west.
5. If c0 is west and c0 is olive, then c1 is harbor.
6. If c1 is harbor, then c1 is east.
7. If c0 is east and c0 is olive, then c1 is cobalt.
8. If c1 is cobalt, then c1 is north.
9. If c0 is south and c0 is olive, then c1 is violet.
10. If c1 is violet, then c1 is south.
11. If c1 is west and c1 is willow, then c2 is violet.
12. If c2 is violet, then c2 is west.
13. If c1 is south and c1 is violet, then c2 is willow.
14. If c2 is willow, then c2 is north.
15. If c1 is north and c1 is cobalt, then c2 is harbor.
16. If c2 is harbor, then c2 is east.
17. If c1 is east and c1 is harbor, then c2 is cobalt.
18. If c2 is cobalt, then c2 is south.
Which state applies to c2?

Solution:
Derivation:
c0 is olive.
c0 is west.
c1 is harbor.
c1 is east.
c2 is cobalt.

Final answer: cobalt<|endoftext|>1. c0 is meadow.
2. c0 is north.
3. If c0 is west and c0 is meadow, then c1 is lime.
4. If c1 is lime, then c1 is west.
5. If c0 is east and c0 is meadow, then c1 is olive.
6. If c1 is olive, then c1 is south.
7. If c0 is north and c0 is meadow, then c1 is amber.
8. If c1 is amber, then c1 is east.
9. If c0 is south and c0 is meadow, then c1 is birch.
10. If c1 is birch, then c1 is north.
Which state applies to c1?

Solution:
Derivation:
c0 is meadow.
c0 is north.
c1 is amber.

Final answer: amber<|endoftext|>
```

## Window 3 (4 documents, 53 pad tokens)
```
1. c0 is pearl.
2. c0 is west.
3. If c0 is east and c0 is pearl, then c1 is ruby.
4. If c1 is ruby, then c1 is north.
5. If c0 is north and c0 is pearl, then c1 is lime.
6. If c1 is lime, then c1 is west.
7. If c0 is west and c0 is pearl, then c1 is amber.
8. If c1 is amber, then c1 is south.
9. If c0 is south and c0 is pearl, then c1 is violet.
10. If c1 is violet, then c1 is east.
11. If c1 is south and c1 is amber, then c2 is ruby.
12. If c2 is ruby, then c2 is east.
13. If c1 is west and c1 is lime, then c2 is violet.
14. If c2 is violet, then c2 is west.
15. If c1 is north and c1 is ruby, then c2 is lime.
16. If c2 is lime, then c2 is north.
17. If c1 is east and c1 is violet, then c2 is olive.
18. If c2 is olive, then c2 is south.
19. If c2 is east and c2 is ruby, then c3 is willow.
20. If c3 is willow, then c3 is east.
21. If c2 is west and c2 is violet, then c3 is teal.
22. If c3 is teal, then c3 is south.
23. If c2 is south and c2 is olive, then c3 is violet.
24. If c3 is violet, then c3 is north.
25. If c2 is north and c2 is lime, then c3 is elm.
26. If c3 is elm, then c3 is west.
27. If c3 is west and c3 is elm, then c4 is amber.
28. If c4 is amber, then c4 is west.
29. If c3 is north and c3 is violet, then c4 is elm.
30. If c4 is elm, then c4 is east.
31. If c3 is south and c3 is teal, then c4 is olive.
32. If c4 is olive, then c4 is south.
33. If c3 is east and c3 is willow, then c4 is willow.
34. If c4 is willow, then c4 is north.
35. If c4 is south and c4 is olive, then c5 is lime.
36. If c5 is lime, then c5 is east.
37. If c4 is east and c4 is elm, then c5 is violet.
38. If c5 is violet, then c5 is south.
39. If c4 is north and c4 is willow, then c5 is amber.
40. If c5 is amber, then c5 is west.
41. If c4 is west and c4 is amber, then c5 is willow.
42. If c5 is willow, then c5 is north.
43. If c5 is south and c5 is violet, then c6 is olive.
44. If c6 is olive, then c6 is south.
45. If c5 is east and c5 is lime, then c6 is teal.
46. If c6 is teal, then c6 is west.
47. If c5 is north and c5 is willow, then c6 is elm.
48. If c6 is elm, then c6 is north.
49. If c5 is west and c5 is amber, then c6 is amber.
50. If c6 is amber, then c6 is east.
51. If c6 is west and c6 is teal, then c7 is amber.
52. If c7 is amber, then c7 is north.
53. If c6 is north and c6 is elm, then c7 is olive.
54. If c7 is olive, then c7 is west.
55. If c6 is east and c6 is amber, then c7 is violet.
56. If c7 is violet, then c7 is south.
57. If c6 is south and c6 is olive, then c7 is willow.
58. If c7 is willow, then c7 is east.
59. If c7 is west and c7 is olive, then c8 is lime.
60. If c8 is lime, then c8 is south.
61. If c7 is east and c7 is willow, then c8 is ruby.
62. If c8 is ruby, then c8 is north.
63. If c7 is south and c7 is violet, then c8 is amber.
64. If c8 is amber, then c8 is east.
65. If c7 is north and c7 is amber, then c8 is elm.
66. If c8 is elm, then c8 is west.
Which state applies to c8?

Solution:
Derivation:
c0 is pearl.
c0 is west.
c1 is amber.
c1 is south.
c2 is ruby.
c2 is east.
c3 is willow.
c3 is east.
c4 is willow.
c4 is north.
c5 is amber.
c5 is west.
c6 is amber.
c6 is east.
c7 is violet.
c7 is south.
c8 is amber.

Final answer: amber<|endoftext|>1. c0 is lime.
2. c0 is east.
3. If c0 is west and c0 is lime, then c1 is willow.
4. If c1 is willow, then c1 is west.
5. If c0 is south and c0 is lime, then c1 is birch.
6. If c1 is birch, then c1 is south.
7. If c0 is north and c0 is lime, then c1 is elm.
8. If c1 is elm, then c1 is north.
9. If c0 is east and c0 is lime, then c1 is juniper.
10. If c1 is juniper, then c1 is east.
11. If c1 is north and c1 is elm, then c2 is birch.
12. If c2 is birch, then c2 is north.
13. If c1 is south and c1 is birch, then c2 is juniper.
14. If c2 is juniper, then c2 is east.
15. If c1 is east and c1 is juniper, then c2 is willow.
16. If c2 is willow, then c2 is west.
17. If c1 is west and c1 is willow, then c2 is elm.
18. If c2 is elm, then c2 is south.
19. If c2 is south and c2 is elm, then c3 is juniper.
20. If c3 is juniper, then c3 is north.
21. If c2 is west and c2 is willow, then c3 is birch.
22. If c3 is birch, then c3 is south.
23. If c2 is north and c2 is birch, then c3 is willow.
24. If c3 is willow, then c3 is west.
25. If c2 is east and c2 is juniper, then c3 is elm.
26. If c3 is elm, then c3 is east.
27. If c3 is north and c3 is juniper, then c4 is elm.
28. If c4 is elm, then c4 is north.
29. If c3 is west and c3 is willow, then c4 is birch.
30. If c4 is birch, then c4 is east.
31. If c3 is south and c3 is birch, then c4 is willow.
32. If c4 is willow, then c4 is west.
33. If c3 is east and c3 is elm, then c4 is juniper.
34. If c4 is juniper, then c4 is south.
Which state applies to c4?

Solution:
Derivation:
c0 is lime.
c0 is east.
c1 is juniper.
c1 is east.
c2 is willow.
c2 is west.
c3 is birch.
c3 is south.
c4 is willow.

Final answer: willow<|endoftext|>1. c0 is pearl.
2. c0 is south.
3. If c0 is west and c0 is pearl, then c1 is juniper.
4. If c1 is juniper, then c1 is north.
5. If c0 is north and c0 is pearl, then c1 is willow.
6. If c1 is willow, then c1 is south.
7. If c0 is south and c0 is pearl, then c1 is granite.
8. If c1 is granite, then c1 is east.
9. If c0 is east and c0 is pearl, then c1 is olive.
10. If c1 is olive, then c1 is west.
11. If c1 is south and c1 is willow, then c2 is olive.
12. If c2 is olive, then c2 is west.
13. If c1 is west and c1 is olive, then c2 is elm.
14. If c2 is elm, then c2 is north.
15. If c1 is north and c1 is juniper, then c2 is lime.
16. If c2 is lime, then c2 is east.
17. If c1 is east and c1 is granite, then c2 is hazel.
18. If c2 is hazel, then c2 is south.
19. If c2 is east and c2 is lime, then c3 is juniper.
20. If c3 is juniper, then c3 is east.
21. If c2 is west and c2 is olive, then c3 is meadow.
22. If c3 is meadow, then c3 is north.
23. If c2 is south and c2 is hazel, then c3 is ivory.
24. If c3 is ivory, then c3 is south.
25. If c2 is north and c2 is elm, then c3 is elm.
26. If c3 is elm, then c3 is west.
27. If c3 is west and c3 is elm, then c4 is juniper.
28. If c4 is juniper, then c4 is south.
29. If c3 is east and c3 is juniper, then c4 is violet.
30. If c4 is violet, then c4 is east.
31. If c3 is south and c3 is ivory, then c4 is lime.
32. If c4 is lime, then c4 is north.
33. If c3 is north and c3 is meadow, then c4 is meadow.
34. If c4 is meadow, then c4 is west.
35. If c4 is west and c4 is meadow, then c5 is amber.
36. If c5 is amber, then c5 is south.
37. If c4 is east and c4 is violet, then c5 is violet.
38. If c5 is violet, then c5 is north.
39. If c4 is south and c4 is juniper, then c5 is slate.
40. If c5 is slate, then c5 is west.
41. If c4 is north and c4 is lime, then c5 is olive.
42. If c5 is olive, then c5 is east.
43. If c5 is north and c5 is violet, then c6 is meadow.
44. If c6 is meadow, then c6 is north.
45. If c5 is west and c5 is slate, then c6 is willow.
46. If c6 is willow, then c6 is east.
47. If c5 is south and c5 is amber, then c6 is juniper.
48. If c6 is juniper, then c6 is south.
49. If c5 is east and c5 is olive, then c6 is granite.
50. If c6 is granite, then c6 is west.
51. If c6 is east and c6 is willow, then c7 is meadow.
52. If c7 is meadow, then c7 is west.
53. If c6 is west and c6 is granite, then c7 is hazel.
54. If c7 is hazel, then c7 is south.
55. If c6 is south and c6 is juniper, then c7 is lime.
56. If c7 is lime, then c7 is north.
57. If c6 is north and c6 is meadow, then c7 is elm.
58. If c7 is elm, then c7 is east.
59. If c7 is south and c7 is hazel, then c8 is hazel.
60. If c8 is hazel, then c8 is east.
61. If c7 is north and c7 is lime, then c8 is elm.
62. If c8 is elm, then c8 is west.
63. If c7 is west and c7 is meadow, then c8 is juniper.
64. If c8 is juniper, then c8 is north.
65. If c7 is east and c7 is elm, then c8 is olive.
66. If c8 is olive, then c8 is south.
67. If c8 is east and c8 is hazel, then c9 is slate.
68. If c9 is slate, then c9 is west.
69. If c8 is north and c8 is juniper, then c9 is ivory.
70. If c9 is ivory, then c9 is north.
71. If c8 is south and c8 is olive, then c9 is willow.
72. If c9 is willow, then c9 is east.
73. If c8 is west and c8 is elm, then c9 is granite.
74. If c9 is granite, then c9 is south.
75. If c9 is north and c9 is ivory, then c10 is slate.
76. If c10 is slate, then c10 is north.
77. If c9 is east and c9 is willow, then c10 is granite.
78. If c10 is granite, then c10 is west.
79. If c9 is south and c9 is granite, then c10 is willow.
80. If c10 is willow, then c10 is east.
81. If c9 is west and c9 is slate, then c10 is ivory.
82. If c10 is ivory, then c10 is south.
83. If c10 is east and c10 is willow, then c11 is granite.
84. If c11 is granite, then c11 is east.
85. If c10 is north and c10 is slate, then c11 is hazel.
86. If c11 is hazel, then c11 is north.
87. If c10 is south and c10 is ivory, then c11 is willow.
88. If c11 is willow, then c11 is west.
89. If c10 is west and c10 is granite, then c11 is violet.
90. If c11 is violet, then c11 is south.
91. If c11 is west and c11 is willow, then c12 is ivory.
92. If c12 is ivory, then c12 is south.
93. If c11 is east and c11 is granite, then c12 is amber.
94. If c12 is amber, then c12 is east.
95. If c11 is north and c11 is hazel, then c12 is elm.
96. If c12 is elm, then c12 is north.
97. If c11 is south and c11 is violet, then c12 is lime.
98. If c12 is lime, then c12 is west.
Which state applies to c12?

Solution:
Derivation:
c0 is pearl.
c0 is south.
c1 is granite.
c1 is east.
c2 is hazel.
c2 is south.
c3 is ivory.
c3 is south.
c4 is lime.
c4 is north.
c5 is olive.
c5 is east.
c6 is granite.
c6 is west.
c7 is hazel.
c7 is south.
c8 is hazel.
c8 is east.
c9 is slate.
c9 is west.
c10 is ivory.
c10 is south.
c11 is willow.
c11 is west.
c12 is ivory.

Final answer: ivory<|endoftext|>1. c0 is willow.
2. c0 is east.
3. If c0 is south and c0 is willow, then c1 is cobalt.
4. If c1 is cobalt, then c1 is west.
5. If c0 is west and c0 is willow, then c1 is birch.
6. If c1 is birch, then c1 is north.
7. If c0 is east and c0 is willow, then c1 is amber.
8. If c1 is amber, then c1 is south.
9. If c0 is north and c0 is willow, then c1 is maple.
10. If c1 is maple, then c1 is east.
Which state applies to c1?

Solution:
Derivation:
c0 is willow.
c0 is east.
c1 is amber.

Final answer: amber<|endoftext|>
```

## Window 246 (3 documents, 31 pad tokens)
```
1. c0 is granite.
2. c0 is south.
3. If c0 is north and c0 is granite, then c1 is birch.
4. If c1 is birch, then c1 is west.
5. If c0 is west and c0 is granite, then c1 is maple.
6. If c1 is maple, then c1 is north.
7. If c0 is east and c0 is granite, then c1 is juniper.
8. If c1 is juniper, then c1 is south.
9. If c0 is south and c0 is granite, then c1 is meadow.
10. If c1 is meadow, then c1 is east.
11. If c1 is east and c1 is meadow, then c2 is willow.
12. If c2 is willow, then c2 is east.
13. If c1 is west and c1 is birch, then c2 is lime.
14. If c2 is lime, then c2 is south.
15. If c1 is south and c1 is juniper, then c2 is juniper.
16. If c2 is juniper, then c2 is west.
17. If c1 is north and c1 is maple, then c2 is pearl.
18. If c2 is pearl, then c2 is north.
19. If c2 is south and c2 is lime, then c3 is maple.
20. If c3 is maple, then c3 is south.
21. If c2 is west and c2 is juniper, then c3 is harbor.
22. If c3 is harbor, then c3 is east.
23. If c2 is north and c2 is pearl, then c3 is olive.
24. If c3 is olive, then c3 is north.
25. If c2 is east and c2 is willow, then c3 is orchid.
26. If c3 is orchid, then c3 is west.
27. If c3 is south and c3 is maple, then c4 is violet.
28. If c4 is violet, then c4 is west.
29. If c3 is west and c3 is orchid, then c4 is olive.
30. If c4 is olive, then c4 is south.
31. If c3 is east and c3 is harbor, then c4 is pearl.
32. If c4 is pearl, then c4 is north.
33. If c3 is north and c3 is olive, then c4 is cedar.
34. If c4 is cedar, then c4 is east.
35. If c4 is south and c4 is olive, then c5 is cedar.
36. If c5 is cedar, then c5 is south.
37. If c4 is north and c4 is pearl, then c5 is orchid.
38. If c5 is orchid, then c5 is west.
39. If c4 is east and c4 is cedar, then c5 is juniper.
40. If c5 is juniper, then c5 is east.
41. If c4 is west and c4 is violet, then c5 is olive.
42. If c5 is olive, then c5 is north.
43. If c5 is north and c5 is olive, then c6 is harbor.
44. If c6 is harbor, then c6 is west.
45. If c5 is south and c5 is cedar, then c6 is lime.
46. If c6 is lime, then c6 is south.
47. If c5 is east and c5 is juniper, then c6 is poppy.
48. If c6 is poppy, then c6 is north.
49. If c5 is west and c5 is orchid, then c6 is orchid.
50. If c6 is orchid, then c6 is east.
51. If c6 is west and c6 is harbor, then c7 is juniper.
52. If c7 is juniper, then c7 is south.
53. If c6 is south and c6 is lime, then c7 is lime.
54. If c7 is lime, then c7 is west.
55. If c6 is east and c6 is orchid, then c7 is meadow.
56. If c7 is meadow, then c7 is east.
57. If c6 is north and c6 is poppy, then c7 is laurel.
58. If c7 is laurel, then c7 is north.
59. If c7 is south and c7 is juniper, then c8 is poppy.
60. If c8 is poppy, then c8 is east.
61. If c7 is east and c7 is meadow, then c8 is laurel.
62. If c8 is laurel, then c8 is north.
63. If c7 is north and c7 is laurel, then c8 is harbor.
64. If c8 is harbor, then c8 is south.
65. If c7 is west and c7 is lime, then c8 is willow.
66. If c8 is willow, then c8 is west.
67. If c8 is south and c8 is harbor, then c9 is birch.
68. If c9 is birch, then c9 is north.
69. If c8 is north and c8 is laurel, then c9 is laurel.
70. If c9 is laurel, then c9 is west.
71. If c8 is west and c8 is willow, then c9 is pearl.
72. If c9 is pearl, then c9 is east.
73. If c8 is east and c8 is poppy, then c9 is olive.
74. If c9 is olive, then c9 is south.
75. If c9 is west and c9 is laurel, then c10 is pearl.
76. If c10 is pearl, then c10 is north.
77. If c9 is north and c9 is birch, then c10 is laurel.
78. If c10 is laurel, then c10 is south.
79. If c9 is east and c9 is pearl, then c10 is meadow.
80. If c10 is meadow, then c10 is east.
81. If c9 is south and c9 is olive, then c10 is orchid.
82. If c10 is orchid, then c10 is west.
83. If c10 is north and c10 is pearl, then c11 is lime.
84. If c11 is lime, then c11 is west.
85. If c10 is west and c10 is orchid, then c11 is laurel.
86. If c11 is laurel, then c11 is east.
87. If c10 is south and c10 is laurel, then c11 is juniper.
88. If c11 is juniper, then c11 is north.
89. If c10 is east and c10 is meadow, then c11 is maple.
90. If c11 is maple, then c11 is south.
91. If c11 is east and c11 is laurel, then c12 is willow.
92. If c12 is willow, then c12 is north.
93. If c11 is west and c11 is lime, then c12 is lime.
94. If c12 is lime, then c12 is south.
95. If c11 is north and c11 is juniper, then c12 is harbor.
96. If c12 is harbor, then c12 is west.
97. If c11 is south and c11 is maple, then c12 is meadow.
98. If c12 is meadow, then c12 is east.
99. If c12 is south and c12 is lime, then c13 is birch.
100. If c13 is birch, then c13 is south.
101. If c12 is west and c12 is harbor, then c13 is pearl.
102. If c13 is pearl, then c13 is west.
103. If c12 is east and c12 is meadow, then c13 is juniper.
104. If c13 is juniper, then c13 is north.
105. If c12 is north and c12 is willow, then c13 is harbor.
106. If c13 is harbor, then c13 is east.
107. If c13 is north and c13 is juniper, then c14 is poppy.
108. If c14 is poppy, then c14 is east.
109. If c13 is south and c13 is birch, then c14 is willow.
110. If c14 is willow, then c14 is west.
111. If c13 is west and c13 is pearl, then c14 is maple.
112. If c14 is maple, then c14 is south.
113. If c13 is east and c13 is harbor, then c14 is lime.
114. If c14 is lime, then c14 is north.
Which state applies to c14?

Solution:
Derivation:
c0 is granite.
c0 is south.
c1 is meadow.
c1 is east.
c2 is willow.
c2 is east.
c3 is orchid.
c3 is west.
c4 is olive.
c4 is south.
c5 is cedar.
c5 is south.
c6 is lime.
c6 is south.
c7 is lime.
c7 is west.
c8 is willow.
c8 is west.
c9 is pearl.
c9 is east.
c10 is meadow.
c10 is east.
c11 is maple.
c11 is south.
c12 is meadow.
c12 is east.
c13 is juniper.
c13 is north.
c14 is poppy.

Final answer: poppy<|endoftext|>1. c0 is granite.
2. c0 is east.
3. If c0 is west and c0 is granite, then c1 is teal.
4. If c1 is teal, then c1 is east.
5. If c0 is east and c0 is granite, then c1 is cedar.
6. If c1 is cedar, then c1 is west.
7. If c0 is north and c0 is granite, then c1 is maple.
8. If c1 is maple, then c1 is south.
9. If c0 is south and c0 is granite, then c1 is poppy.
10. If c1 is poppy, then c1 is north.
11. If c1 is south and c1 is maple, then c2 is maple.
12. If c2 is maple, then c2 is west.
13. If c1 is north and c1 is poppy, then c2 is laurel.
14. If c2 is laurel, then c2 is south.
15. If c1 is west and c1 is cedar, then c2 is cedar.
16. If c2 is cedar, then c2 is north.
17. If c1 is east and c1 is teal, then c2 is ivory.
18. If c2 is ivory, then c2 is east.
19. If c2 is north and c2 is cedar, then c3 is ivory.
20. If c3 is ivory, then c3 is south.
21. If c2 is west and c2 is maple, then c3 is laurel.
22. If c3 is laurel, then c3 is east.
23. If c2 is south and c2 is laurel, then c3 is teal.
24. If c3 is teal, then c3 is west.
25. If c2 is east and c2 is ivory, then c3 is maple.
26. If c3 is maple, then c3 is north.
27. If c3 is north and c3 is maple, then c4 is poppy.
28. If c4 is poppy, then c4 is south.
29. If c3 is east and c3 is laurel, then c4 is maple.
30. If c4 is maple, then c4 is east.
31. If c3 is west and c3 is teal, then c4 is teal.
32. If c4 is teal, then c4 is north.
33. If c3 is south and c3 is ivory, then c4 is ivory.
34. If c4 is ivory, then c4 is west.
35. If c4 is east and c4 is maple, then c5 is poppy.
36. If c5 is poppy, then c5 is south.
37. If c4 is west and c4 is ivory, then c5 is ivory.
38. If c5 is ivory, then c5 is east.
39. If c4 is north and c4 is teal, then c5 is laurel.
40. If c5 is laurel, then c5 is west.
41. If c4 is south and c4 is poppy, then c5 is cedar.
42. If c5 is cedar, then c5 is north.
43. If c5 is west and c5 is laurel, then c6 is ivory.
44. If c6 is ivory, then c6 is west.
45. If c5 is north and c5 is cedar, then c6 is cedar.
46. If c6 is cedar, then c6 is east.
47. If c5 is south and c5 is poppy, then c6 is teal.
48. If c6 is teal, then c6 is south.
49. If c5 is east and c5 is ivory, then c6 is maple.
50. If c6 is maple, then c6 is north.
Which state applies to c6?

Solution:
Derivation:
c0 is granite.
c0 is east.
c1 is cedar.
c1 is west.
c2 is cedar.
c2 is north.
c3 is ivory.
c3 is south.
c4 is ivory.
c4 is west.
c5 is ivory.
c5 is east.
c6 is maple.

Final answer: maple<|endoftext|>1. c0 is teal.
2. c0 is west.
3. If c0 is south and c0 is teal, then c1 is ivory.
4. If c1 is ivory, then c1 is west.
5. If c0 is east and c0 is teal, then c1 is elm.
6. If c1 is elm, then c1 is east.
7. If c0 is west and c0 is teal, then c1 is cedar.
8. If c1 is cedar, then c1 is south.
9. If c0 is north and c0 is teal, then c1 is pearl.
10. If c1 is pearl, then c1 is north.
11. If c1 is east and c1 is elm, then c2 is cedar.
12. If c2 is cedar, then c2 is west.
13. If c1 is west and c1 is ivory, then c2 is ivory.
14. If c2 is ivory, then c2 is north.
15. If c1 is north and c1 is pearl, then c2 is pearl.
16. If c2 is pearl, then c2 is south.
17. If c1 is south and c1 is cedar, then c2 is elm.
18. If c2 is elm, then c2 is east.
19. If c2 is north and c2 is ivory, then c3 is ivory.
20. If c3 is ivory, then c3 is west.
21. If c2 is south and c2 is pearl, then c3 is pearl.
22. If c3 is pearl, then c3 is north.
23. If c2 is east and c2 is elm, then c3 is elm.
24. If c3 is elm, then c3 is south.
25. If c2 is west and c2 is cedar, then c3 is ruby.
26. If c3 is ruby, then c3 is east.
27. If c3 is south and c3 is elm, then c4 is cedar.
28. If c4 is cedar, then c4 is east.
29. If c3 is east and c3 is ruby, then c4 is elm.
30. If c4 is elm, then c4 is north.
31. If c3 is west and c3 is ivory, then c4 is pearl.
32. If c4 is pearl, then c4 is south.
33. If c3 is north and c3 is pearl, then c4 is ruby.
34. If c4 is ruby, then c4 is west.
35. If c4 is west and c4 is ruby, then c5 is elm.
36. If c5 is elm, then c5 is south.
37. If c4 is east and c4 is cedar, then c5 is pearl.
38. If c5 is pearl, then c5 is east.
39. If c4 is north and c4 is elm, then c5 is cedar.
40. If c5 is cedar, then c5 is west.
41. If c4 is south and c4 is pearl, then c5 is ruby.
42. If c5 is ruby, then c5 is north.
Which state applies to c5?

Solution:
Derivation:
c0 is teal.
c0 is west.
c1 is cedar.
c1 is south.
c2 is elm.
c2 is east.
c3 is elm.
c3 is south.
c4 is cedar.
c4 is east.
c5 is pearl.

Final answer: pearl<|endoftext|>
```

## Window 664 (3 documents, 13 pad tokens)
```
1. c0 is maple.
2. c0 is west.
3. If c0 is north and c0 is maple, then c1 is teal.
4. If c1 is teal, then c1 is east.
5. If c0 is east and c0 is maple, then c1 is birch.
6. If c1 is birch, then c1 is north.
7. If c0 is south and c0 is maple, then c1 is violet.
8. If c1 is violet, then c1 is west.
9. If c0 is west and c0 is maple, then c1 is granite.
10. If c1 is granite, then c1 is south.
11. If c1 is north and c1 is birch, then c2 is coral.
12. If c2 is coral, then c2 is north.
13. If c1 is east and c1 is teal, then c2 is granite.
14. If c2 is granite, then c2 is east.
15. If c1 is west and c1 is violet, then c2 is birch.
16. If c2 is birch, then c2 is west.
17. If c1 is south and c1 is granite, then c2 is juniper.
18. If c2 is juniper, then c2 is south.
19. If c2 is south and c2 is juniper, then c3 is cobalt.
20. If c3 is cobalt, then c3 is south.
21. If c2 is north and c2 is coral, then c3 is juniper.
22. If c3 is juniper, then c3 is east.
23. If c2 is east and c2 is granite, then c3 is birch.
24. If c3 is birch, then c3 is north.
25. If c2 is west and c2 is birch, then c3 is harbor.
26. If c3 is harbor, then c3 is west.
27. If c3 is north and c3 is birch, then c4 is harbor.
28. If c4 is harbor, then c4 is north.
29. If c3 is east and c3 is juniper, then c4 is poppy.
30. If c4 is poppy, then c4 is west.
31. If c3 is south and c3 is cobalt, then c4 is coral.
32. If c4 is coral, then c4 is south.
33. If c3 is west and c3 is harbor, then c4 is cedar.
34. If c4 is cedar, then c4 is east.
35. If c4 is west and c4 is poppy, then c5 is hazel.
36. If c5 is hazel, then c5 is west.
37. If c4 is south and c4 is coral, then c5 is poppy.
38. If c5 is poppy, then c5 is north.
39. If c4 is north and c4 is harbor, then c5 is cobalt.
40. If c5 is cobalt, then c5 is south.
41. If c4 is east and c4 is cedar, then c5 is violet.
42. If c5 is violet, then c5 is east.
43. If c5 is north and c5 is poppy, then c6 is violet.
44. If c6 is violet, then c6 is east.
45. If c5 is south and c5 is cobalt, then c6 is juniper.
46. If c6 is juniper, then c6 is north.
47. If c5 is west and c5 is hazel, then c6 is hazel.
48. If c6 is hazel, then c6 is south.
49. If c5 is east and c5 is violet, then c6 is poppy.
50. If c6 is poppy, then c6 is west.
51. If c6 is east and c6 is violet, then c7 is granite.
52. If c7 is granite, then c7 is south.
53. If c6 is north and c6 is juniper, then c7 is violet.
54. If c7 is violet, then c7 is west.
55. If c6 is west and c6 is poppy, then c7 is teal.
56. If c7 is teal, then c7 is east.
57. If c6 is south and c6 is hazel, then c7 is cobalt.
58. If c7 is cobalt, then c7 is north.
59. If c7 is east and c7 is teal, then c8 is teal.
60. If c8 is teal, then c8 is south.
61. If c7 is west and c7 is violet, then c8 is coral.
62. If c8 is coral, then c8 is west.
63. If c7 is north and c7 is cobalt, then c8 is cobalt.
64. If c8 is cobalt, then c8 is north.
65. If c7 is south and c7 is granite, then c8 is violet.
66. If c8 is violet, then c8 is east.
67. If c8 is west and c8 is coral, then c9 is violet.
68. If c9 is violet, then c9 is north.
69. If c8 is north and c8 is cobalt, then c9 is granite.
70. If c9 is granite, then c9 is south.
71. If c8 is south and c8 is teal, then c9 is birch.
72. If c9 is birch, then c9 is east.
73. If c8 is east and c8 is violet, then c9 is harbor.
74. If c9 is harbor, then c9 is west.
75. If c9 is south and c9 is granite, then c10 is cedar.
76. If c10 is cedar, then c10 is west.
77. If c9 is north and c9 is violet, then c10 is birch.
78. If c10 is birch, then c10 is east.
79. If c9 is east and c9 is birch, then c10 is poppy.
80. If c10 is poppy, then c10 is north.
81. If c9 is west and c9 is harbor, then c10 is juniper.
82. If c10 is juniper, then c10 is south.
83. If c10 is south and c10 is juniper, then c11 is poppy.
84. If c11 is poppy, then c11 is south.
85. If c10 is west and c10 is cedar, then c11 is harbor.
86. If c11 is harbor, then c11 is east.
87. If c10 is east and c10 is birch, then c11 is teal.
88. If c11 is teal, then c11 is west.
89. If c10 is north and c10 is poppy, then c11 is juniper.
90. If c11 is juniper, then c11 is north.
Which state applies to c11?

Solution:
Derivation:
c0 is maple.
c0 is west.
c1 is granite.
c1 is south.
c2 is juniper.
c2 is south.
c3 is cobalt.
c3 is south.
c4 is coral.
c4 is south.
c5 is poppy.
c5 is north.
c6 is violet.
c6 is east.
c7 is granite.
c7 is south.
c8 is violet.
c8 is east.
c9 is harbor.
c9 is west.
c10 is juniper.
c10 is south.
c11 is poppy.

Final answer: poppy<|endoftext|>1. c0 is laurel.
2. c0 is north.
3. If c0 is north and c0 is laurel, then c1 is harbor.
4. If c1 is harbor, then c1 is south.
5. If c0 is west and c0 is laurel, then c1 is poppy.
6. If c1 is poppy, then c1 is west.
7. If c0 is south and c0 is laurel, then c1 is orchid.
8. If c1 is orchid, then c1 is north.
9. If c0 is east and c0 is laurel, then c1 is amber.
10. If c1 is amber, then c1 is east.
11. If c1 is north and c1 is orchid, then c2 is teal.
12. If c2 is teal, then c2 is east.
13. If c1 is east and c1 is amber, then c2 is ruby.
14. If c2 is ruby, then c2 is west.
15. If c1 is west and c1 is poppy, then c2 is cedar.
16. If c2 is cedar, then c2 is north.
17. If c1 is south and c1 is harbor, then c2 is poppy.
18. If c2 is poppy, then c2 is south.
19. If c2 is south and c2 is poppy, then c3 is cedar.
20. If c3 is cedar, then c3 is north.
21. If c2 is west and c2 is ruby, then c3 is maple.
22. If c3 is maple, then c3 is south.
23. If c2 is north and c2 is cedar, then c3 is teal.
24. If c3 is teal, then c3 is east.
25. If c2 is east and c2 is teal, then c3 is willow.
26. If c3 is willow, then c3 is west.
27. If c3 is east and c3 is teal, then c4 is coral.
28. If c4 is coral, then c4 is south.
29. If c3 is west and c3 is willow, then c4 is lime.
30. If c4 is lime, then c4 is east.
31. If c3 is south and c3 is maple, then c4 is amber.
32. If c4 is amber, then c4 is west.
33. If c3 is north and c3 is cedar, then c4 is ruby.
34. If c4 is ruby, then c4 is north.
35. If c4 is north and c4 is ruby, then c5 is pearl.
36. If c5 is pearl, then c5 is north.
37. If c4 is west and c4 is amber, then c5 is harbor.
38. If c5 is harbor, then c5 is west.
39. If c4 is east and c4 is lime, then c5 is cedar.
40. If c5 is cedar, then c5 is east.
41. If c4 is south and c4 is coral, then c5 is teal.
42. If c5 is teal, then c5 is south.
43. If c5 is north and c5 is pearl, then c6 is amber.
44. If c6 is amber, then c6 is south.
45. If c5 is west and c5 is harbor, then c6 is teal.
46. If c6 is teal, then c6 is east.
47. If c5 is east and c5 is cedar, then c6 is orchid.
48. If c6 is orchid, then c6 is west.
49. If c5 is south and c5 is teal, then c6 is lime.
50. If c6 is lime, then c6 is north.
51. If c6 is north and c6 is lime, then c7 is coral.
52. If c7 is coral, then c7 is west.
53. If c6 is south and c6 is amber, then c7 is teal.
54. If c7 is teal, then c7 is north.
55. If c6 is east and c6 is teal, then c7 is amber.
56. If c7 is amber, then c7 is south.
57. If c6 is west and c6 is orchid, then c7 is harbor.
58. If c7 is harbor, then c7 is east.
59. If c7 is south and c7 is amber, then c8 is lime.
60. If c8 is lime, then c8 is south.
61. If c7 is east and c7 is harbor, then c8 is amber.
62. If c8 is amber, then c8 is east.
63. If c7 is west and c7 is coral, then c8 is willow.
64. If c8 is willow, then c8 is west.
65. If c7 is north and c7 is teal, then c8 is violet.
66. If c8 is violet, then c8 is north.
67. If c8 is east and c8 is amber, then c9 is willow.
68. If c9 is willow, then c9 is west.
69. If c8 is north and c8 is violet, then c9 is lime.
70. If c9 is lime, then c9 is south.
71. If c8 is west and c8 is willow, then c9 is cedar.
72. If c9 is cedar, then c9 is east.
73. If c8 is south and c8 is lime, then c9 is harbor.
74. If c9 is harbor, then c9 is north.
75. If c9 is north and c9 is harbor, then c10 is poppy.
76. If c10 is poppy, then c10 is west.
77. If c9 is south and c9 is lime, then c10 is maple.
78. If c10 is maple, then c10 is south.
79. If c9 is west and c9 is willow, then c10 is cedar.
80. If c10 is cedar, then c10 is east.
81. If c9 is east and c9 is cedar, then c10 is willow.
82. If c10 is willow, then c10 is north.
83. If c10 is west and c10 is poppy, then c11 is maple.
84. If c11 is maple, then c11 is west.
85. If c10 is east and c10 is cedar, then c11 is coral.
86. If c11 is coral, then c11 is north.
87. If c10 is north and c10 is willow, then c11 is poppy.
88. If c11 is poppy, then c11 is east.
89. If c10 is south and c10 is maple, then c11 is willow.
90. If c11 is willow, then c11 is south.
91. If c11 is south and c11 is willow, then c12 is lime.
92. If c12 is lime, then c12 is north.
93. If c11 is north and c11 is coral, then c12 is orchid.
94. If c12 is orchid, then c12 is east.
95. If c11 is west and c11 is maple, then c12 is harbor.
96. If c12 is harbor, then c12 is west.
97. If c11 is east and c11 is poppy, then c12 is willow.
98. If c12 is willow, then c12 is south.
99. If c12 is west and c12 is harbor, then c13 is lime.
100. If c13 is lime, then c13 is west.
101. If c12 is east and c12 is orchid, then c13 is violet.
102. If c13 is violet, then c13 is south.
103. If c12 is south and c12 is willow, then c13 is ruby.
104. If c13 is ruby, then c13 is east.
105. If c12 is north and c12 is lime, then c13 is coral.
106. If c13 is coral, then c13 is north.
Which state applies to c13?

Solution:
Derivation:
c0 is laurel.
c0 is north.
c1 is harbor.
c1 is south.
c2 is poppy.
c2 is south.
c3 is cedar.
c3 is north.
c4 is ruby.
c4 is north.
c5 is pearl.
c5 is north.
c6 is amber.
c6 is south.
c7 is teal.
c7 is north.
c8 is violet.
c8 is north.
c9 is lime.
c9 is south.
c10 is maple.
c10 is south.
c11 is willow.
c11 is south.
c12 is lime.
c12 is north.
c13 is coral.

Final answer: coral<|endoftext|>1. c0 is maple.
2. c0 is east.
3. If c0 is east and c0 is maple, then c1 is cobalt.
4. If c1 is cobalt, then c1 is south.
5. If c0 is north and c0 is maple, then c1 is laurel.
6. If c1 is laurel, then c1 is west.
7. If c0 is west and c0 is maple, then c1 is willow.
8. If c1 is willow, then c1 is east.
9. If c0 is south and c0 is maple, then c1 is orchid.
10. If c1 is orchid, then c1 is north.
Which state applies to c1?

Solution:
Derivation:
c0 is maple.
c0 is east.
c1 is cobalt.

Final answer: cobalt<|endoftext|>
```

## Window 1182 summary: [{"tokens": 2158, "head": "1. c0 is birch. 2. c0 is east. 3. If c0 is west and c0 is birch, then c1 is gran", "tail": "orchid. c12 is north. c13 is granite.  Final answer: granite"}, {"tokens": 1748, "head": "1. c0 is hazel. 2. c0 is west. 3. If c0 is east and c0 is hazel, then c1 is pear", "tail": "0 is willow. c10 is east. c11 is pearl.  Final answer: pearl"}, {"tokens": 181, "head": "1. c0 is coral. 2. c0 is west. 3. If c0 is east and c0 is coral, then c1 is coba", "tail": ": c0 is coral. c0 is west. c1 is olive.  Final answer: olive"}]

## Window 1318 summary: [{"tokens": 1765, "head": "1. c0 is elm. 2. c0 is north. 3. If c0 is south and c0 is elm, then c1 is maple.", "tail": "10 is laurel. c10 is south. c11 is teal.  Final answer: teal"}, {"tokens": 1442, "head": "1. c0 is amber. 2. c0 is south. 3. If c0 is north and c0 is amber, then c1 is ju", "tail": "is laurel. c8 is west. c9 is granite.  Final answer: granite"}, {"tokens": 496, "head": "1. c0 is maple. 2. c0 is east. 3. If c0 is south and c0 is maple, then c1 is cob", "tail": "h. c2 is willow. c2 is west. c3 is ruby.  Final answer: ruby"}, {"tokens": 334, "head": "1. c0 is granite. 2. c0 is west. 3. If c0 is east and c0 is granite, then c1 is ", "tail": ". c1 is lime. c1 is south. c2 is coral.  Final answer: coral"}]

## Window 2431 summary: [{"tokens": 1929, "head": "1. c0 is coral. 2. c0 is south. 3. If c0 is west and c0 is coral, then c1 is lim", "tail": ". c11 is ivory. c11 is north. c12 is elm.  Final answer: elm"}, {"tokens": 1948, "head": "1. c0 is poppy. 2. c0 is west. 3. If c0 is west and c0 is poppy, then c1 is mead", "tail": " is lime. c11 is north. c12 is cobalt.  Final answer: cobalt"}, {"tokens": 181, "head": "1. c0 is amber. 2. c0 is north. 3. If c0 is west and c0 is amber, then c1 is orc", "tail": " c0 is amber. c0 is north. c1 is cedar.  Final answer: cedar"}]

## Window 2558 summary: [{"tokens": 2361, "head": "1. c0 is cedar. 2. c0 is east. 3. If c0 is south and c0 is cedar, then c1 is mea", "tail": " is willow. c13 is north. c14 is hazel.  Final answer: hazel"}, {"tokens": 957, "head": "1. c0 is cedar. 2. c0 is east. 3. If c0 is west and c0 is cedar, then c1 is viol", "tail": "5 is violet. c5 is east. c6 is harbor.  Final answer: harbor"}, {"tokens": 485, "head": "1. c0 is birch. 2. c0 is east. 3. If c0 is north and c0 is birch, then c1 is pea", "tail": ". c2 is slate. c2 is east. c3 is coral.  Final answer: coral"}, {"tokens": 183, "head": "1. c0 is amber. 2. c0 is south. 3. If c0 is west and c0 is amber, then c1 is lim", "tail": "ion: c0 is amber. c0 is south. c1 is elm.  Final answer: elm"}]

## Window 3503 summary: [{"tokens": 1296, "head": "1. c0 is violet. 2. c0 is north. 3. If c0 is south and c0 is violet, then c1 is ", "tail": ". c7 is poppy. c7 is east. c8 is maple.  Final answer: maple"}, {"tokens": 479, "head": "1. c0 is olive. 2. c0 is north. 3. If c0 is north and c0 is olive, then c1 is ha", "tail": "2 is ivory. c2 is north. c3 is harbor.  Final answer: harbor"}, {"tokens": 1452, "head": "1. c0 is slate. 2. c0 is west. 3. If c0 is east and c0 is slate, then c1 is will", "tail": "8 is ivory. c8 is north. c9 is orchid.  Final answer: orchid"}, {"tokens": 640, "head": "1. c0 is slate. 2. c0 is north. 3. If c0 is north and c0 is slate, then c1 is co", "tail": "ast. c3 is teal. c3 is west. c4 is teal.  Final answer: teal"}, {"tokens": 179, "head": "1. c0 is pearl. 2. c0 is east. 3. If c0 is east and c0 is pearl, then c1 is ambe", "tail": ": c0 is pearl. c0 is east. c1 is amber.  Final answer: amber"}]

## Window 3824 summary: [{"tokens": 1431, "head": "1. c0 is orchid. 2. c0 is north. 3. If c0 is south and c0 is orchid, then c1 is ", "tail": " is laurel. c8 is north. c9 is meadow.  Final answer: meadow"}, {"tokens": 818, "head": "1. c0 is ruby. 2. c0 is north. 3. If c0 is east and c0 is ruby, then c1 is hazel", "tail": " c4 is amber. c4 is south. c5 is hazel.  Final answer: hazel"}, {"tokens": 797, "head": "1. c0 is poppy. 2. c0 is west. 3. If c0 is north and c0 is poppy, then c1 is tea", "tail": "4 is juniper. c4 is north. c5 is ivory.  Final answer: ivory"}, {"tokens": 967, "head": "1. c0 is elm. 2. c0 is west. 3. If c0 is south and c0 is elm, then c1 is hazel. ", "tail": " is orchid. c5 is south. c6 is orchid.  Final answer: orchid"}]

## Window 4971 summary: [{"tokens": 2117, "head": "1. c0 is hazel. 2. c0 is north. 3. If c0 is east and c0 is hazel, then c1 is map", "tail": "12 is harbor. c12 is north. c13 is lime.  Final answer: lime"}, {"tokens": 1603, "head": "1. c0 is juniper. 2. c0 is east. 3. If c0 is east and c0 is juniper, then c1 is ", "tail": "c9 is amber. c9 is south. c10 is amber.  Final answer: amber"}, {"tokens": 334, "head": "1. c0 is pearl. 2. c0 is east. 3. If c0 is north and c0 is pearl, then c1 is lim", "tail": "t. c1 is ivory. c1 is north. c2 is lime.  Final answer: lime"}]

## Window 4972 summary: [{"tokens": 1968, "head": "1. c0 is teal. 2. c0 is north. 3. If c0 is south and c0 is teal, then c1 is birc", "tail": "is willow. c11 is west. c12 is violet.  Final answer: violet"}, {"tokens": 1950, "head": "1. c0 is meadow. 2. c0 is east. 3. If c0 is west and c0 is meadow, then c1 is co", "tail": " is cobalt. c11 is south. c12 is maple.  Final answer: maple"}, {"tokens": 179, "head": "1. c0 is amber. 2. c0 is north. 3. If c0 is west and c0 is amber, then c1 is lim", "tail": " c0 is amber. c0 is north. c1 is cedar.  Final answer: cedar"}]

## Window 5682 summary: [{"tokens": 938, "head": "1. c0 is coral. 2. c0 is west. 3. If c0 is south and c0 is coral, then c1 is tea", "tail": ". c5 is ivory. c5 is east. c6 is maple.  Final answer: maple"}, {"tokens": 935, "head": "1. c0 is cobalt. 2. c0 is west. 3. If c0 is east and c0 is cobalt, then c1 is vi", "tail": "st. c5 is lime. c5 is south. c6 is ruby.  Final answer: ruby"}, {"tokens": 818, "head": "1. c0 is amber. 2. c0 is east. 3. If c0 is west and c0 is amber, then c1 is orch", "tail": "4 is willow. c4 is west. c5 is willow.  Final answer: willow"}, {"tokens": 1279, "head": "1. c0 is slate. 2. c0 is east. 3. If c0 is north and c0 is slate, then c1 is elm", "tail": "c7 is elm. c7 is east. c8 is juniper.  Final answer: juniper"}]

## Window 5883 summary: [{"tokens": 1916, "head": "1. c0 is willow. 2. c0 is east. 3. If c0 is south and c0 is willow, then c1 is m", "tail": "is cedar. c11 is north. c12 is harbor.  Final answer: harbor"}, {"tokens": 1294, "head": "1. c0 is cobalt. 2. c0 is west. 3. If c0 is north and c0 is cobalt, then c1 is j", "tail": "c7 is juniper. c7 is east. c8 is pearl.  Final answer: pearl"}, {"tokens": 488, "head": "1. c0 is teal. 2. c0 is north. 3. If c0 is east and c0 is teal, then c1 is ruby.", "tail": " is violet. c2 is north. c3 is violet.  Final answer: violet"}, {"tokens": 334, "head": "1. c0 is cedar. 2. c0 is south. 3. If c0 is west and c0 is cedar, then c1 is pea", "tail": ". c1 is slate. c1 is west. c2 is pearl.  Final answer: pearl"}]

## Window 6137 summary: [{"tokens": 1620, "head": "1. c0 is pearl. 2. c0 is south. 3. If c0 is north and c0 is pearl, then c1 is co", "tail": "is cobalt. c9 is north. c10 is willow.  Final answer: willow"}, {"tokens": 2336, "head": "1. c0 is harbor. 2. c0 is west. 3. If c0 is south and c0 is harbor, then c1 is c", "tail": "c13 is lime. c13 is east. c14 is birch.  Final answer: birch"}]

## Window 6833 summary: [{"tokens": 1956, "head": "1. c0 is cobalt. 2. c0 is west. 3. If c0 is west and c0 is cobalt, then c1 is ma", "tail": " is laurel. c11 is north. c12 is maple.  Final answer: maple"}, {"tokens": 1750, "head": "1. c0 is elm. 2. c0 is south. 3. If c0 is east and c0 is elm, then c1 is ruby. 4", "tail": "0 is violet. c10 is east. c11 is ivory.  Final answer: ivory"}, {"tokens": 341, "head": "1. c0 is teal. 2. c0 is north. 3. If c0 is east and c0 is teal, then c1 is cobal", "tail": "c1 is ivory. c1 is east. c2 is cobalt.  Final answer: cobalt"}]

## Window 7221 summary: [{"tokens": 1953, "head": "1. c0 is amber. 2. c0 is west. 3. If c0 is south and c0 is amber, then c1 is oli", "tail": "juniper. c11 is west. c12 is juniper.  Final answer: juniper"}, {"tokens": 1611, "head": "1. c0 is slate. 2. c0 is east. 3. If c0 is east and c0 is slate, then c1 is lime", "tail": "th. c9 is lime. c9 is east. c10 is teal.  Final answer: teal"}, {"tokens": 334, "head": "1. c0 is maple. 2. c0 is north. 3. If c0 is west and c0 is maple, then c1 is ivo", "tail": "h. c1 is elm. c1 is north. c2 is slate.  Final answer: slate"}]

## Window 7322 summary: [{"tokens": 2321, "head": "1. c0 is teal. 2. c0 is east. 3. If c0 is south and c0 is teal, then c1 is maple", "tail": "13 is poppy. c13 is east. c14 is birch.  Final answer: birch"}, {"tokens": 1762, "head": "1. c0 is orchid. 2. c0 is west. 3. If c0 is west and c0 is orchid, then c1 is bi", "tail": ". c10 is elm. c10 is south. c11 is ruby.  Final answer: ruby"}]

## Window 7540 summary: [{"tokens": 1774, "head": "1. c0 is meadow. 2. c0 is north. 3. If c0 is east and c0 is meadow, then c1 is g", "tail": ". c10 is slate. c10 is south. c11 is elm.  Final answer: elm"}, {"tokens": 1955, "head": "1. c0 is birch. 2. c0 is south. 3. If c0 is west and c0 is birch, then c1 is jun", "tail": "is meadow. c11 is east. c12 is orchid.  Final answer: orchid"}, {"tokens": 334, "head": "1. c0 is olive. 2. c0 is west. 3. If c0 is north and c0 is olive, then c1 is rub", "tail": " c1 is granite. c1 is north. c2 is ruby.  Final answer: ruby"}]

## Window 7644 summary: [{"tokens": 1748, "head": "1. c0 is meadow. 2. c0 is south. 3. If c0 is south and c0 is meadow, then c1 is ", "tail": " cedar. c10 is south. c11 is granite.  Final answer: granite"}, {"tokens": 1403, "head": "1. c0 is laurel. 2. c0 is north. 3. If c0 is south and c0 is laurel, then c1 is ", "tail": ". c8 is ivory. c8 is east. c9 is ivory.  Final answer: ivory"}, {"tokens": 802, "head": "1. c0 is pearl. 2. c0 is east. 3. If c0 is south and c0 is pearl, then c1 is cor", "tail": "west. c4 is birch. c4 is east. c5 is elm.  Final answer: elm"}]

## Window 7916 summary: [{"tokens": 2153, "head": "1. c0 is hazel. 2. c0 is east. 3. If c0 is south and c0 is hazel, then c1 is amb", "tail": "2 is harbor. c12 is east. c13 is cedar.  Final answer: cedar"}, {"tokens": 1576, "head": "1. c0 is olive. 2. c0 is east. 3. If c0 is north and c0 is olive, then c1 is sla", "tail": "9 is teal. c9 is north. c10 is violet.  Final answer: violet"}, {"tokens": 342, "head": "1. c0 is olive. 2. c0 is north. 3. If c0 is east and c0 is olive, then c1 is vio", "tail": "1 is cobalt. c1 is west. c2 is cobalt.  Final answer: cobalt"}]

## Window 7962 summary: [{"tokens": 1752, "head": "1. c0 is meadow. 2. c0 is west. 3. If c0 is west and c0 is meadow, then c1 is el", "tail": "10 is cedar. c10 is east. c11 is hazel.  Final answer: hazel"}, {"tokens": 1605, "head": "1. c0 is poppy. 2. c0 is south. 3. If c0 is west and c0 is poppy, then c1 is wil", "tail": " c9 is lime. c9 is south. c10 is maple.  Final answer: maple"}, {"tokens": 340, "head": "1. c0 is ruby. 2. c0 is east. 3. If c0 is north and c0 is ruby, then c1 is hazel", "tail": " c1 is poppy. c1 is south. c2 is ivory.  Final answer: ivory"}, {"tokens": 335, "head": "1. c0 is willow. 2. c0 is south. 3. If c0 is north and c0 is willow, then c1 is ", "tail": "th. c1 is slate. c1 is west. c2 is lime.  Final answer: lime"}]

## Window 8060 summary: [{"tokens": 1579, "head": "1. c0 is cedar. 2. c0 is west. 3. If c0 is west and c0 is cedar, then c1 is mead", "tail": "c9 is amber. c9 is south. c10 is slate.  Final answer: slate"}, {"tokens": 1112, "head": "1. c0 is ivory. 2. c0 is north. 3. If c0 is south and c0 is ivory, then c1 is ru", "tail": " c6 is laurel. c6 is east. c7 is hazel.  Final answer: hazel"}, {"tokens": 1272, "head": "1. c0 is cedar. 2. c0 is north. 3. If c0 is north and c0 is cedar, then c1 is or", "tail": " is willow. c7 is south. c8 is orchid.  Final answer: orchid"}]

## Window 8222 summary: [{"tokens": 1947, "head": "1. c0 is meadow. 2. c0 is west. 3. If c0 is west and c0 is meadow, then c1 is wi", "tail": " is willow. c11 is north. c12 is maple.  Final answer: maple"}, {"tokens": 661, "head": "1. c0 is laurel. 2. c0 is west. 3. If c0 is north and c0 is laurel, then c1 is h", "tail": " c3 is coral. c3 is south. c4 is hazel.  Final answer: hazel"}, {"tokens": 789, "head": "1. c0 is maple. 2. c0 is north. 3. If c0 is south and c0 is maple, then c1 is pe", "tail": "c4 is coral. c4 is east. c5 is laurel.  Final answer: laurel"}, {"tokens": 635, "head": "1. c0 is poppy. 2. c0 is north. 3. If c0 is north and c0 is poppy, then c1 is ol", "tail": ". c3 is maple. c3 is east. c4 is olive.  Final answer: olive"}]

## Window 8471 summary: [{"tokens": 1756, "head": "1. c0 is coral. 2. c0 is east. 3. If c0 is west and c0 is coral, then c1 is birc", "tail": "harbor. c10 is south. c11 is granite.  Final answer: granite"}, {"tokens": 635, "head": "1. c0 is birch. 2. c0 is south. 3. If c0 is north and c0 is birch, then c1 is ru", "tail": "t. c3 is ruby. c3 is west. c4 is amber.  Final answer: amber"}, {"tokens": 1426, "head": "1. c0 is cedar. 2. c0 is east. 3. If c0 is north and c0 is cedar, then c1 is cob", "tail": "s orchid. c8 is north. c9 is juniper.  Final answer: juniper"}, {"tokens": 185, "head": "1. c0 is amber. 2. c0 is south. 3. If c0 is south and c0 is amber, then c1 is or", "tail": "0 is amber. c0 is south. c1 is orchid.  Final answer: orchid"}]

## Window 8915 summary: [{"tokens": 1744, "head": "1. c0 is elm. 2. c0 is east. 3. If c0 is south and c0 is elm, then c1 is cedar. ", "tail": " c10 is amber. c10 is west. c11 is ruby.  Final answer: ruby"}, {"tokens": 487, "head": "1. c0 is olive. 2. c0 is west. 3. If c0 is north and c0 is olive, then c1 is map", "tail": "c2 is granite. c2 is east. c3 is maple.  Final answer: maple"}, {"tokens": 796, "head": "1. c0 is willow. 2. c0 is east. 3. If c0 is north and c0 is willow, then c1 is j", "tail": ". c4 is maple. c4 is west. c5 is maple.  Final answer: maple"}, {"tokens": 495, "head": "1. c0 is hazel. 2. c0 is east. 3. If c0 is east and c0 is hazel, then c1 is ambe", "tail": "t. c2 is lime. c2 is east. c3 is birch.  Final answer: birch"}, {"tokens": 337, "head": "1. c0 is cedar. 2. c0 is west. 3. If c0 is south and c0 is cedar, then c1 is gra", "tail": ". c1 is hazel. c1 is west. c2 is hazel.  Final answer: hazel"}, {"tokens": 179, "head": "1. c0 is elm. 2. c0 is south. 3. If c0 is east and c0 is elm, then c1 is ivory. ", "tail": "n: c0 is elm. c0 is south. c1 is slate.  Final answer: slate"}]

## Window 9161 summary: [{"tokens": 2328, "head": "1. c0 is hazel. 2. c0 is west. 3. If c0 is east and c0 is hazel, then c1 is coba", "tail": "3 is olive. c13 is north. c14 is slate.  Final answer: slate"}, {"tokens": 1751, "head": "1. c0 is willow. 2. c0 is north. 3. If c0 is east and c0 is willow, then c1 is h", "tail": ". c10 is teal. c10 is west. c11 is lime.  Final answer: lime"}]

## Window 9449 summary: [{"tokens": 941, "head": "1. c0 is violet. 2. c0 is west. 3. If c0 is south and c0 is violet, then c1 is m", "tail": ". c5 is maple. c5 is east. c6 is slate.  Final answer: slate"}, {"tokens": 952, "head": "1. c0 is coral. 2. c0 is south. 3. If c0 is east and c0 is coral, then c1 is pea", "tail": " c5 is willow. c5 is west. c6 is pearl.  Final answer: pearl"}, {"tokens": 1603, "head": "1. c0 is olive. 2. c0 is north. 3. If c0 is east and c0 is olive, then c1 is jun", "tail": "9 is poppy. c9 is west. c10 is meadow.  Final answer: meadow"}, {"tokens": 495, "head": "1. c0 is cobalt. 2. c0 is south. 3. If c0 is north and c0 is cobalt, then c1 is ", "tail": " c2 is willow. c2 is west. c3 is slate.  Final answer: slate"}]

## Window 9940 summary: [{"tokens": 1907, "head": "1. c0 is ivory. 2. c0 is south. 3. If c0 is east and c0 is ivory, then c1 is amb", "tail": " is ruby. c11 is north. c12 is violet.  Final answer: violet"}, {"tokens": 1630, "head": "1. c0 is birch. 2. c0 is north. 3. If c0 is west and c0 is birch, then c1 is jun", "tail": "s maple. c9 is north. c10 is juniper.  Final answer: juniper"}, {"tokens": 494, "head": "1. c0 is laurel. 2. c0 is east. 3. If c0 is west and c0 is laurel, then c1 is sl", "tail": " c2 is hazel. c2 is south. c3 is slate.  Final answer: slate"}]

## Window 9941 summary: [{"tokens": 2136, "head": "1. c0 is coral. 2. c0 is east. 3. If c0 is west and c0 is coral, then c1 is lime", "tail": " is cobalt. c12 is north. c13 is birch.  Final answer: birch"}, {"tokens": 1619, "head": "1. c0 is poppy. 2. c0 is west. 3. If c0 is west and c0 is poppy, then c1 is teal", "tail": " is birch. c9 is south. c10 is cobalt.  Final answer: cobalt"}, {"tokens": 340, "head": "1. c0 is juniper. 2. c0 is east. 3. If c0 is south and c0 is juniper, then c1 is", "tail": "st. c1 is olive. c1 is west. c2 is teal.  Final answer: teal"}]
