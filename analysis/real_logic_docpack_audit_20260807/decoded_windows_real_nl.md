# Decoded-batch audit examples (document-preserving docpack loader)

## Window 0 (8 documents, 164 pad tokens)
```
The bald eagle chases the tiger. The bald eagle is round. The bald eagle likes the bear. The bald eagle likes the tiger. The bear likes the tiger. The bear needs the bald eagle. The cow chases the tiger. The cow is green. The cow is red. The cow does not need the bald eagle. The tiger likes the cow. If the bald eagle likes the bear then the bald eagle is young. If something is cold then it needs the tiger. If something chases the bear and the bear needs the bald eagle then it needs the cow. If something needs the bald eagle then it is cold. If something is round and it needs the bald eagle then the bald eagle does not need the cow. If something is cold and it needs the tiger then the tiger needs the bald eagle. If something needs the cow then the cow needs the bear.

True or false: The cow is not red.

Solution:
Context:
Premises:
1. The bald eagle chases the tiger.
2. The bald eagle is round.
3. The bald eagle likes the bear.
4. The bald eagle likes the tiger.
5. The bear likes the tiger.
6. The bear needs the bald eagle.
7. The cow chases the tiger.
8. The cow is green.
9. The cow is red.
10. The cow does not need the bald eagle.
11. The tiger likes the cow.
12. If the bald eagle likes the bear then the bald eagle is young.
13. If something is cold then it needs the tiger.
14. If something chases the bear and the bear needs the bald eagle then it needs the cow.
15. If something needs the bald eagle then it is cold.
16. If something is round and it needs the bald eagle then the bald eagle does not need the cow.
17. If something is cold and it needs the tiger then the tiger needs the bald eagle.
18. If something needs the cow then the cow needs the bear.

Derivation:
19. We are given: The cow is red.

Conclusion:
The cow is red.

Final answer: False<|endoftext|>The bald eagle chases the tiger. The bald eagle is round. The bald eagle likes the bear. The bald eagle likes the tiger. The bear likes the tiger. The bear needs the bald eagle. The cow chases the tiger. The cow is green. The cow is red. The cow does not need the bald eagle. The tiger likes the cow. If the bald eagle likes the bear then the bald eagle is young. If something is cold then it needs the tiger. If something chases the bear and the bear needs the bald eagle then it needs the cow. If something needs the bald eagle then it is cold. If something is round and it needs the bald eagle then the bald eagle does not need the cow. If something is cold and it needs the tiger then the tiger needs the bald eagle. If something needs the cow then the cow needs the bear.

True or false: The bald eagle is young.

Solution:
Context:
Premises:
1. The bald eagle chases the tiger.
2. The bald eagle is round.
3. The bald eagle likes the bear.
4. The bald eagle likes the tiger.
5. The bear likes the tiger.
6. The bear needs the bald eagle.
7. The cow chases the tiger.
8. The cow is green.
9. The cow is red.
10. The cow does not need the bald eagle.
11. The tiger likes the cow.
12. If the bald eagle likes the bear then the bald eagle is young.
13. If something is cold then it needs the tiger.
14. If something chases the bear and the bear needs the bald eagle then it needs the cow.
15. If something needs the bald eagle then it is cold.
16. If something is round and it needs the bald eagle then the bald eagle does not need the cow.
17. If something is cold and it needs the tiger then the tiger needs the bald eagle.
18. If something needs the cow then the cow needs the bear.

Derivation:
19. We are given: The bald eagle likes the bear.
20. Therefore, The bald eagle is young.

Conclusion:
The bald eagle is young.

Final answer: True<|endoftext|>The bald eagle chases the tiger. The bald eagle is round. The bald eagle likes the bear. The bald eagle likes the tiger. The bear likes the tiger. The bear needs the bald eagle. The cow chases the tiger. The cow is green. The cow is red. The cow does not need the bald eagle. The tiger likes the cow. If the bald eagle likes the bear then the bald eagle is young. If something is cold then it needs the tiger. If something chases the bear and the bear needs the bald eagle then it needs the cow. If something needs the bald eagle then it is cold. If something is round and it needs the bald eagle then the bald eagle does not need the cow. If something is cold and it needs the tiger then the tiger needs the bald eagle. If something needs the cow then the cow needs the bear.

True or false: The bear is not cold.

Solution:
Context:
Premises:
1. The bald eagle chases the tiger.
2. The bald eagle is round.
3. The bald eagle likes the bear.
4. The bald eagle likes the tiger.
5. The bear likes the tiger.
6. The bear needs the bald eagle.
7. The cow chases the tiger.
8. The cow is green.
9. The cow is red.
10. The cow does not need the bald eagle.
11. The tiger likes the cow.
12. If the bald eagle likes the bear then the bald eagle is young.
13. If something is cold then it needs the tiger.
14. If something chases the bear and the bear needs the bald eagle then it needs the cow.
15. If something needs the bald eagle then it is cold.
16. If something is round and it needs the bald eagle then the bald eagle does not need the cow.
17. If something is cold and it needs the tiger then the tiger needs the bald eagle.
18. If something needs the cow then the cow needs the bear.

Derivation:
19. We are given: The bear needs the bald eagle.
20. Instantiating rule 15 for the bear: if The bear needs the bald eagle, then The bear is cold.
21. Therefore, The bear is cold.

Conclusion:
The bear is cold.

Final answer: False<|endoftext|>The bald eagle chases the tiger. The bald eagle is round. The bald eagle likes the bear. The bald eagle likes the tiger. The bear likes the tiger. The bear needs the bald eagle. The cow chases the tiger. The cow is green. The cow is red. The cow does not need the bald eagle. The tiger likes the cow. If the bald eagle likes the bear then the bald eagle is young. If something is cold then it needs the tiger. If something chases the bear and the bear needs the bald eagle then it needs the cow. If something needs the bald eagle then it is cold. If something is round and it needs the bald eagle then the bald eagle does not need the cow. If something is cold and it needs the tiger then the tiger needs the bald eagle. If something needs the cow then the cow needs the bear.

True or false: The bear needs the tiger.

Solution:
Context:
Premises:
1. The bald eagle chases the tiger.
2. The bald eagle is round.
3. The bald eagle likes the bear.
4. The bald eagle likes the tiger.
5. The bear likes the tiger.
6. The bear needs the bald eagle.
7. The cow chases the tiger.
8. The cow is green.
9. The cow is red.
10. The cow does not need the bald eagle.
11. The tiger likes the cow.
12. If the bald eagle likes the bear then the bald eagle is young.
13. If something is cold then it needs the tiger.
14. If something chases the bear and the bear needs the bald eagle then it needs the cow.
15. If something needs the bald eagle then it is cold.
16. If something is round and it needs the bald eagle then the bald eagle does not need the cow.
17. If something is cold and it needs the tiger then the tiger needs the bald eagle.
18. If something needs the cow then the cow needs the bear.

Derivation:
19. We are given: The bear needs the bald eagle.
20. Instantiating rule 15 for the bear: if The bear needs the bald eagle, then The bear is cold.
21. Therefore, The bear is cold.
22. Instantiating rule 13 for the bear: if The bear is cold, then The bear needs the tiger.
23. Therefore, The bear needs the tiger.

Conclusion:
The bear needs the tiger.

Final answer: True<|endoftext|>The bald eagle chases the tiger. The bald eagle is round. The bald eagle likes the bear. The bald eagle likes the tiger. The bear likes the tiger. The bear needs the bald eagle. The cow chases the tiger. The cow is green. The cow is red. The cow does not need the bald eagle. The tiger likes the cow. If the bald eagle likes the bear then the bald eagle is young. If something is cold then it needs the tiger. If something chases the bear and the bear needs the bald eagle then it needs the cow. If something needs the bald eagle then it is cold. If something is round and it needs the bald eagle then the bald eagle does not need the cow. If something is cold and it needs the tiger then the tiger needs the bald eagle. If something needs the cow then the cow needs the bear.

True or false: The bear does not need the tiger.

Solution:
Context:
Premises:
1. The bald eagle chases the tiger.
2. The bald eagle is round.
3. The bald eagle likes the bear.
4. The bald eagle likes the tiger.
5. The bear likes the tiger.
6. The bear needs the bald eagle.
7. The cow chases the tiger.
8. The cow is green.
9. The cow is red.
10. The cow does not need the bald eagle.
11. The tiger likes the cow.
12. If the bald eagle likes the bear then the bald eagle is young.
13. If something is cold then it needs the tiger.
14. If something chases the bear and the bear needs the bald eagle then it needs the cow.
15. If something needs the bald eagle then it is cold.
16. If something is round and it needs the bald eagle then the bald eagle does not need the cow.
17. If something is cold and it needs the tiger then the tiger needs the bald eagle.
18. If something needs the cow then the cow needs the bear.

Derivation:
19. We are given: The bear needs the bald eagle.
20. Instantiating rule 15 for the bear: if The bear needs the bald eagle, then The bear is cold.
21. Therefore, The bear is cold.
22. Instantiating rule 13 for the bear: if The bear is cold, then The bear needs the tiger.
23. Therefore, The bear needs the tiger.

Conclusion:
The bear needs the tiger.

Final answer: False<|endoftext|>The bald eagle chases the tiger. The bald eagle is round. The bald eagle likes the bear. The bald eagle likes the tiger. The bear likes the tiger. The bear needs the bald eagle. The cow chases the tiger. The cow is green. The cow is red. The cow does not need the bald eagle. The tiger likes the cow. If the bald eagle likes the bear then the bald eagle is young. If something is cold then it needs the tiger. If something chases the bear and the bear needs the bald eagle then it needs the cow. If something needs the bald eagle then it is cold. If something is round and it needs the bald eagle then the bald eagle does not need the cow. If something is cold and it needs the tiger then the tiger needs the bald eagle. If something needs the cow then the cow needs the bear.

True or false: The tiger needs the bald eagle.

Solution:
Context:
Premises:
1. The bald eagle chases the tiger.
2. The bald eagle is round.
3. The bald eagle likes the bear.
4. The bald eagle likes the tiger.
5. The bear likes the tiger.
6. The bear needs the bald eagle.
7. The cow chases the tiger.
8. The cow is green.
9. The cow is red.
10. The cow does not need the bald eagle.
11. The tiger likes the cow.
12. If the bald eagle likes the bear then the bald eagle is young.
13. If something is cold then it needs the tiger.
14. If something chases the bear and the bear needs the bald eagle then it needs the cow.
15. If something needs the bald eagle then it is cold.
16. If something is round and it needs the bald eagle then the bald eagle does not need the cow.
17. If something is cold and it needs the tiger then the tiger needs the bald eagle.
18. If something needs the cow then the cow needs the bear.

Derivation:
19. We are given: The bear needs the bald eagle.
20. Instantiating rule 15 for the bear: if The bear needs the bald eagle, then The bear is cold.
21. Therefore, The bear is cold.
22. Instantiating rule 13 for the bear: if The bear is cold, then The bear needs the tiger.
23. Therefore, The bear needs the tiger.
24. Combining: The bear is cold and The bear needs the tiger.
25. Instantiating rule 17 for the bear: if The bear is cold and The bear needs the tiger, then The tiger needs the bald eagle.
26. Therefore, The tiger needs the bald eagle.

Conclusion:
The tiger needs the bald eagle.

Final answer: True<|endoftext|>The bald eagle chases the tiger. The bald eagle is round. The bald eagle likes the bear. The bald eagle likes the tiger. The bear likes the tiger. The bear needs the bald eagle. The cow chases the tiger. The cow is green. The cow is red. The cow does not need the bald eagle. The tiger likes the cow. If the bald eagle likes the bear then the bald eagle is young. If something is cold then it needs the tiger. If something chases the bear and the bear needs the bald eagle then it needs the cow. If something needs the bald eagle then it is cold. If something is round and it needs the bald eagle then the bald eagle does not need the cow. If something is cold and it needs the tiger then the tiger needs the bald eagle. If something needs the cow then the cow needs the bear.

True or false: The tiger does not need the bald eagle.

Solution:
Context:
Premises:
1. The bald eagle chases the tiger.
2. The bald eagle is round.
3. The bald eagle likes the bear.
4. The bald eagle likes the tiger.
5. The bear likes the tiger.
6. The bear needs the bald eagle.
7. The cow chases the tiger.
8. The cow is green.
9. The cow is red.
10. The cow does not need the bald eagle.
11. The tiger likes the cow.
12. If the bald eagle likes the bear then the bald eagle is young.
13. If something is cold then it needs the tiger.
14. If something chases the bear and the bear needs the bald eagle then it needs the cow.
15. If something needs the bald eagle then it is cold.
16. If something is round and it needs the bald eagle then the bald eagle does not need the cow.
17. If something is cold and it needs the tiger then the tiger needs the bald eagle.
18. If something needs the cow then the cow needs the bear.

Derivation:
19. We are given: The bear needs the bald eagle.
20. Instantiating rule 15 for the bear: if The bear needs the bald eagle, then The bear is cold.
21. Therefore, The bear is cold.
22. Instantiating rule 13 for the bear: if The bear is cold, then The bear needs the tiger.
23. Therefore, The bear needs the tiger.
24. Combining: The bear is cold and The bear needs the tiger.
25. Instantiating rule 17 for the bear: if The bear is cold and The bear needs the tiger, then The tiger needs the bald eagle.
26. Therefore, The tiger needs the bald eagle.

Conclusion:
The tiger needs the bald eagle.

Final answer: False<|endoftext|>Anne is cold. Anne is green. Anne is kind. Anne is quiet. Anne is rough. Anne is smart. Anne is young. Charlie is cold. Charlie is rough. Charlie is smart. Erin is cold. Erin is quiet. Gary is cold. Gary is smart. All smart, quiet things are green. If Gary is smart then Gary is rough. If something is quiet and green then it is cold. All quiet, green things are young. Young, smart things are green. If Anne is rough and Anne is kind then Anne is young. Young, quiet things are kind. If something is rough then it is quiet.

True or false: Gary is cold.

Solution:
Context:
Premises:
1. Anne is cold.
2. Anne is green.
3. Anne is kind.
4. Anne is quiet.
5. Anne is rough.
6. Anne is smart.
7. Anne is young.
8. Charlie is cold.
9. Charlie is rough.
10. Charlie is smart.
11. Erin is cold.
12. Erin is quiet.
13. Gary is cold.
14. Gary is smart.
15. All smart, quiet things are green.
16. If Gary is smart then Gary is rough.
17. If something is quiet and green then it is cold.
18. All quiet, green things are young.
19. Young, smart things are green.
20. If Anne is rough and Anne is kind then Anne is young.
21. Young, quiet things are kind.
22. If something is rough then it is quiet.

Derivation:
23. We are given: Gary is cold.

Conclusion:
Gary is cold.

Final answer: True<|endoftext|>
```

## Window 1 (9 documents, 11 pad tokens)
```
The wolf is rough. The wolf is dull. The wolf is slow. The wolf likes the dog. The snake visits the rabbit. The snake is fierce. The snake is big. The dog is round. The dog is quiet. The dog is smart. The rabbit is cute. The rabbit is beautiful. The rabbit is lovely. Round animals are cute. If something is dull then it chases the dog. If something chases the dog then it is sleepy. If something is rough and dull then it is slow. If something is cute and beautiful then it is furry. If something is fierce and big then it is heavy. All slow animals are lazy. All cute animals are beautiful. All heavy animals are strong. All furry animals are small.

True or false: The snake is strong.

Solution:
Context:
Premises:
1. The wolf is rough.
2. The wolf is dull.
3. The wolf is slow.
4. The wolf likes the dog.
5. The snake visits the rabbit.
6. The snake is fierce.
7. The snake is big.
8. The dog is round.
9. The dog is quiet.
10. The dog is smart.
11. The rabbit is cute.
12. The rabbit is beautiful.
13. The rabbit is lovely.
14. Round animals are cute.
15. If something is dull then it chases the dog.
16. If something chases the dog then it is sleepy.
17. If something is rough and dull then it is slow.
18. If something is cute and beautiful then it is furry.
19. If something is fierce and big then it is heavy.
20. All slow animals are lazy.
21. All cute animals are beautiful.
22. All heavy animals are strong.
23. All furry animals are small.

Derivation:
24. We are given: The snake is fierce.
25. We are given: The snake is big.
26. Combining: The snake is fierce and The snake is big.
27. Instantiating rule 19 for the snake: if The snake is fierce and The snake is big, then The snake is heavy.
28. Therefore, The snake is heavy.
29. Instantiating rule 22 for the snake: if The snake is heavy, then The snake is strong.
30. Therefore, The snake is strong.

Conclusion:
The snake is strong.

Final answer: True<|endoftext|>The wolf is rough. The wolf is dull. The wolf is slow. The wolf likes the dog. The snake visits the rabbit. The snake is fierce. The snake is big. The dog is round. The dog is quiet. The dog is smart. The rabbit is cute. The rabbit is beautiful. The rabbit is lovely. Round animals are cute. If something is dull then it chases the dog. If something chases the dog then it is sleepy. If something is rough and dull then it is slow. If something is cute and beautiful then it is furry. If something is fierce and big then it is heavy. All slow animals are lazy. All cute animals are beautiful. All heavy animals are strong. All furry animals are small.

True or false: The snake is not strong.

Solution:
Context:
Premises:
1. The wolf is rough.
2. The wolf is dull.
3. The wolf is slow.
4. The wolf likes the dog.
5. The snake visits the rabbit.
6. The snake is fierce.
7. The snake is big.
8. The dog is round.
9. The dog is quiet.
10. The dog is smart.
11. The rabbit is cute.
12. The rabbit is beautiful.
13. The rabbit is lovely.
14. Round animals are cute.
15. If something is dull then it chases the dog.
16. If something chases the dog then it is sleepy.
17. If something is rough and dull then it is slow.
18. If something is cute and beautiful then it is furry.
19. If something is fierce and big then it is heavy.
20. All slow animals are lazy.
21. All cute animals are beautiful.
22. All heavy animals are strong.
23. All furry animals are small.

Derivation:
24. We are given: The snake is fierce.
25. We are given: The snake is big.
26. Combining: The snake is fierce and The snake is big.
27. Instantiating rule 19 for the snake: if The snake is fierce and The snake is big, then The snake is heavy.
28. Therefore, The snake is heavy.
29. Instantiating rule 22 for the snake: if The snake is heavy, then The snake is strong.
30. Therefore, The snake is strong.

Conclusion:
The snake is strong.

Final answer: False<|endoftext|>The wolf is rough. The wolf is dull. The wolf is slow. The wolf likes the dog. The snake visits the rabbit. The snake is fierce. The snake is big. The dog is round. The dog is quiet. The dog is smart. The rabbit is cute. The rabbit is beautiful. The rabbit is lovely. Round animals are cute. If something is dull then it chases the dog. If something chases the dog then it is sleepy. If something is rough and dull then it is slow. If something is cute and beautiful then it is furry. If something is fierce and big then it is heavy. All slow animals are lazy. All cute animals are beautiful. All heavy animals are strong. All furry animals are small.

True or false: The rabbit is small.

Solution:
Context:
Premises:
1. The wolf is rough.
2. The wolf is dull.
3. The wolf is slow.
4. The wolf likes the dog.
5. The snake visits the rabbit.
6. The snake is fierce.
7. The snake is big.
8. The dog is round.
9. The dog is quiet.
10. The dog is smart.
11. The rabbit is cute.
12. The rabbit is beautiful.
13. The rabbit is lovely.
14. Round animals are cute.
15. If something is dull then it chases the dog.
16. If something chases the dog then it is sleepy.
17. If something is rough and dull then it is slow.
18. If something is cute and beautiful then it is furry.
19. If something is fierce and big then it is heavy.
20. All slow animals are lazy.
21. All cute animals are beautiful.
22. All heavy animals are strong.
23. All furry animals are small.

Derivation:
24. We are given: The rabbit is cute.
25. We are given: The rabbit is beautiful.
26. Combining: The rabbit is cute and The rabbit is beautiful.
27. Instantiating rule 18 for the rabbit: if The rabbit is cute and The rabbit is beautiful, then The rabbit is furry.
28. Therefore, The rabbit is furry.
29. Instantiating rule 23 for the rabbit: if The rabbit is furry, then The rabbit is small.
30. Therefore, The rabbit is small.

Conclusion:
The rabbit is small.

Final answer: True<|endoftext|>The wolf is rough. The wolf is dull. The wolf is slow. The wolf likes the dog. The snake visits the rabbit. The snake is fierce. The snake is big. The dog is round. The dog is quiet. The dog is smart. The rabbit is cute. The rabbit is beautiful. The rabbit is lovely. Round animals are cute. If something is dull then it chases the dog. If something chases the dog then it is sleepy. If something is rough and dull then it is slow. If something is cute and beautiful then it is furry. If something is fierce and big then it is heavy. All slow animals are lazy. All cute animals are beautiful. All heavy animals are strong. All furry animals are small.

True or false: The rabbit is not small.

Solution:
Context:
Premises:
1. The wolf is rough.
2. The wolf is dull.
3. The wolf is slow.
4. The wolf likes the dog.
5. The snake visits the rabbit.
6. The snake is fierce.
7. The snake is big.
8. The dog is round.
9. The dog is quiet.
10. The dog is smart.
11. The rabbit is cute.
12. The rabbit is beautiful.
13. The rabbit is lovely.
14. Round animals are cute.
15. If something is dull then it chases the dog.
16. If something chases the dog then it is sleepy.
17. If something is rough and dull then it is slow.
18. If something is cute and beautiful then it is furry.
19. If something is fierce and big then it is heavy.
20. All slow animals are lazy.
21. All cute animals are beautiful.
22. All heavy animals are strong.
23. All furry animals are small.

Derivation:
24. We are given: The rabbit is cute.
25. We are given: The rabbit is beautiful.
26. Combining: The rabbit is cute and The rabbit is beautiful.
27. Instantiating rule 18 for the rabbit: if The rabbit is cute and The rabbit is beautiful, then The rabbit is furry.
28. Therefore, The rabbit is furry.
29. Instantiating rule 23 for the rabbit: if The rabbit is furry, then The rabbit is small.
30. Therefore, The rabbit is small.

Conclusion:
The rabbit is small.

Final answer: False<|endoftext|>The wolf is rough. The wolf is dull. The wolf is slow. The wolf likes the dog. The snake visits the rabbit. The snake is fierce. The snake is big. The dog is round. The dog is quiet. The dog is smart. The rabbit is cute. The rabbit is beautiful. The rabbit is lovely. Round animals are cute. If something is dull then it chases the dog. If something chases the dog then it is sleepy. If something is rough and dull then it is slow. If something is cute and beautiful then it is furry. If something is fierce and big then it is heavy. All slow animals are lazy. All cute animals are beautiful. All heavy animals are strong. All furry animals are small.

True or false: The wolf is sleepy.

Solution:
Context:
Premises:
1. The wolf is rough.
2. The wolf is dull.
3. The wolf is slow.
4. The wolf likes the dog.
5. The snake visits the rabbit.
6. The snake is fierce.
7. The snake is big.
8. The dog is round.
9. The dog is quiet.
10. The dog is smart.
11. The rabbit is cute.
12. The rabbit is beautiful.
13. The rabbit is lovely.
14. Round animals are cute.
15. If something is dull then it chases the dog.
16. If something chases the dog then it is sleepy.
17. If something is rough and dull then it is slow.
18. If something is cute and beautiful then it is furry.
19. If something is fierce and big then it is heavy.
20. All slow animals are lazy.
21. All cute animals are beautiful.
22. All heavy animals are strong.
23. All furry animals are small.

Derivation:
24. We are given: The wolf is dull.
25. Instantiating rule 15 for the wolf: if The wolf is dull, then The wolf chases the dog.
26. Therefore, The wolf chases the dog.
27. Instantiating rule 16 for the wolf: if The wolf chases the dog, then The wolf is sleepy.
28. Therefore, The wolf is sleepy.

Conclusion:
The wolf is sleepy.

Final answer: True<|endoftext|>The wolf is rough. The wolf is dull. The wolf is slow. The wolf likes the dog. The snake visits the rabbit. The snake is fierce. The snake is big. The dog is round. The dog is quiet. The dog is smart. The rabbit is cute. The rabbit is beautiful. The rabbit is lovely. Round animals are cute. If something is dull then it chases the dog. If something chases the dog then it is sleepy. If something is rough and dull then it is slow. If something is cute and beautiful then it is furry. If something is fierce and big then it is heavy. All slow animals are lazy. All cute animals are beautiful. All heavy animals are strong. All furry animals are small.

True or false: The wolf is not sleepy.

Solution:
Context:
Premises:
1. The wolf is rough.
2. The wolf is dull.
3. The wolf is slow.
4. The wolf likes the dog.
5. The snake visits the rabbit.
6. The snake is fierce.
7. The snake is big.
8. The dog is round.
9. The dog is quiet.
10. The dog is smart.
11. The rabbit is cute.
12. The rabbit is beautiful.
13. The rabbit is lovely.
14. Round animals are cute.
15. If something is dull then it chases the dog.
16. If something chases the dog then it is sleepy.
17. If something is rough and dull then it is slow.
18. If something is cute and beautiful then it is furry.
19. If something is fierce and big then it is heavy.
20. All slow animals are lazy.
21. All cute animals are beautiful.
22. All heavy animals are strong.
23. All furry animals are small.

Derivation:
24. We are given: The wolf is dull.
25. Instantiating rule 15 for the wolf: if The wolf is dull, then The wolf chases the dog.
26. Therefore, The wolf chases the dog.
27. Instantiating rule 16 for the wolf: if The wolf chases the dog, then The wolf is sleepy.
28. Therefore, The wolf is sleepy.

Conclusion:
The wolf is sleepy.

Final answer: False<|endoftext|>Dave is huge. Dave is high. Dave is strong. Anne is small. Anne is thin. Bob is quiet. Bob is nice. Bob is wealthy. Fiona is sad. Fiona is dull. Fiona is rough. Huge people are quiet. If someone is small and thin then they are little. If someone is sad and dull then they are poor. If someone is quiet and nice then they are smart. All little people are short. All quiet people are nice. All smart people are kind. All poor people are bad.

True or false: Dave is nice.

Solution:
Context:
Premises:
1. Dave is huge.
2. Dave is high.
3. Dave is strong.
4. Anne is small.
5. Anne is thin.
6. Bob is quiet.
7. Bob is nice.
8. Bob is wealthy.
9. Fiona is sad.
10. Fiona is dull.
11. Fiona is rough.
12. Huge people are quiet.
13. If someone is small and thin then they are little.
14. If someone is sad and dull then they are poor.
15. If someone is quiet and nice then they are smart.
16. All little people are short.
17. All quiet people are nice.
18. All smart people are kind.
19. All poor people are bad.

Derivation:
20. We are given: Dave is huge.
21. Instantiating rule 12 for Dave: if Dave is huge, then Dave is quiet.
22. Therefore, Dave is quiet.
23. Instantiating rule 17 for Dave: if Dave is quiet, then Dave is nice.
24. Therefore, Dave is nice.

Conclusion:
Dave is nice.

Final answer: True<|endoftext|>Dave is huge. Dave is high. Dave is strong. Anne is small. Anne is thin. Bob is quiet. Bob is nice. Bob is wealthy. Fiona is sad. Fiona is dull. Fiona is rough. Huge people are quiet. If someone is small and thin then they are little. If someone is sad and dull then they are poor. If someone is quiet and nice then they are smart. All little people are short. All quiet people are nice. All smart people are kind. All poor people are bad.

True or false: Dave is not nice.

Solution:
Context:
Premises:
1. Dave is huge.
2. Dave is high.
3. Dave is strong.
4. Anne is small.
5. Anne is thin.
6. Bob is quiet.
7. Bob is nice.
8. Bob is wealthy.
9. Fiona is sad.
10. Fiona is dull.
11. Fiona is rough.
12. Huge people are quiet.
13. If someone is small and thin then they are little.
14. If someone is sad and dull then they are poor.
15. If someone is quiet and nice then they are smart.
16. All little people are short.
17. All quiet people are nice.
18. All smart people are kind.
19. All poor people are bad.

Derivation:
20. We are given: Dave is huge.
21. Instantiating rule 12 for Dave: if Dave is huge, then Dave is quiet.
22. Therefore, Dave is quiet.
23. Instantiating rule 17 for Dave: if Dave is quiet, then Dave is nice.
24. Therefore, Dave is nice.

Conclusion:
Dave is nice.

Final answer: False<|endoftext|>Fiona is high. Fiona is huge. Fiona is strong. Dave is small. Dave is little. Charlie is quiet. Charlie is wealthy. Charlie is kind. Bob is dull. Bob is sad. Bob is bad. High people are quiet. If someone is small and little then they are short. If someone is dull and sad then they are rough. If someone is quiet and wealthy then they are smart. All short people are thin. All quiet people are wealthy. All smart people are nice. All rough people are poor.

True or false: Fiona is wealthy.

Solution:
Context:
Premises:
1. Fiona is high.
2. Fiona is huge.
3. Fiona is strong.
4. Dave is small.
5. Dave is little.
6. Charlie is quiet.
7. Charlie is wealthy.
8. Charlie is kind.
9. Bob is dull.
10. Bob is sad.
11. Bob is bad.
12. High people are quiet.
13. If someone is small and little then they are short.
14. If someone is dull and sad then they are rough.
15. If someone is quiet and wealthy then they are smart.
16. All short people are thin.
17. All quiet people are wealthy.
18. All smart people are nice.
19. All rough people are poor.

Derivation:
20. We are given: Fiona is high.
21. Instantiating rule 12 for Fiona: if Fiona is high, then Fiona is quiet.
22. Therefore, Fiona is quiet.
23. Instantiating rule 17 for Fiona: if Fiona is quiet, then Fiona is wealthy.
24. Therefore, Fiona is wealthy.

Conclusion:
Fiona is wealthy.

Final answer: True<|endoftext|>
```

## Window 2 (10 documents, 30 pad tokens)
```
The crocodile is slow. The crocodile is sleepy. The crocodile is dull. The crocodile chases the cat. The wolf visits the rabbit. The wolf is heavy. The wolf is big. The cat is kind. The cat is quiet. The cat is nice. The rabbit is lovely. The rabbit is small. The rabbit is beautiful. Kind animals are lovely. If something is sleepy then it likes the cat. If something likes the cat then it is lazy. If something is slow and sleepy then it is dull. If something is lovely and small then it is cute. If something is heavy and big then it is strong. All dull animals are rough. All lovely animals are small. All strong animals are fierce. All cute animals are furry.

True or false: The crocodile is lazy.

Solution:
Context:
Premises:
1. The crocodile is slow.
2. The crocodile is sleepy.
3. The crocodile is dull.
4. The crocodile chases the cat.
5. The wolf visits the rabbit.
6. The wolf is heavy.
7. The wolf is big.
8. The cat is kind.
9. The cat is quiet.
10. The cat is nice.
11. The rabbit is lovely.
12. The rabbit is small.
13. The rabbit is beautiful.
14. Kind animals are lovely.
15. If something is sleepy then it likes the cat.
16. If something likes the cat then it is lazy.
17. If something is slow and sleepy then it is dull.
18. If something is lovely and small then it is cute.
19. If something is heavy and big then it is strong.
20. All dull animals are rough.
21. All lovely animals are small.
22. All strong animals are fierce.
23. All cute animals are furry.

Derivation:
24. We are given: The crocodile is sleepy.
25. Instantiating rule 15 for the crocodile: if The crocodile is sleepy, then The crocodile likes the cat.
26. Therefore, The crocodile likes the cat.
27. Instantiating rule 16 for the crocodile: if The crocodile likes the cat, then The crocodile is lazy.
28. Therefore, The crocodile is lazy.

Conclusion:
The crocodile is lazy.

Final answer: True<|endoftext|>The crocodile is slow. The crocodile is sleepy. The crocodile is dull. The crocodile chases the cat. The wolf visits the rabbit. The wolf is heavy. The wolf is big. The cat is kind. The cat is quiet. The cat is nice. The rabbit is lovely. The rabbit is small. The rabbit is beautiful. Kind animals are lovely. If something is sleepy then it likes the cat. If something likes the cat then it is lazy. If something is slow and sleepy then it is dull. If something is lovely and small then it is cute. If something is heavy and big then it is strong. All dull animals are rough. All lovely animals are small. All strong animals are fierce. All cute animals are furry.

True or false: The crocodile is not lazy.

Solution:
Context:
Premises:
1. The crocodile is slow.
2. The crocodile is sleepy.
3. The crocodile is dull.
4. The crocodile chases the cat.
5. The wolf visits the rabbit.
6. The wolf is heavy.
7. The wolf is big.
8. The cat is kind.
9. The cat is quiet.
10. The cat is nice.
11. The rabbit is lovely.
12. The rabbit is small.
13. The rabbit is beautiful.
14. Kind animals are lovely.
15. If something is sleepy then it likes the cat.
16. If something likes the cat then it is lazy.
17. If something is slow and sleepy then it is dull.
18. If something is lovely and small then it is cute.
19. If something is heavy and big then it is strong.
20. All dull animals are rough.
21. All lovely animals are small.
22. All strong animals are fierce.
23. All cute animals are furry.

Derivation:
24. We are given: The crocodile is sleepy.
25. Instantiating rule 15 for the crocodile: if The crocodile is sleepy, then The crocodile likes the cat.
26. Therefore, The crocodile likes the cat.
27. Instantiating rule 16 for the crocodile: if The crocodile likes the cat, then The crocodile is lazy.
28. Therefore, The crocodile is lazy.

Conclusion:
The crocodile is lazy.

Final answer: False<|endoftext|>Dave is strong. Dave is high. Dave is big. Harry is small. Harry is little. Charlie is nice. Charlie is quiet. Charlie is wealthy. Erin is poor. Erin is sad. Erin is dull. Strong people are nice. If someone is small and little then they are short. If someone is poor and sad then they are rough. If someone is nice and quiet then they are kind. All short people are thin. All nice people are quiet. All kind people are smart. All rough people are bad.

True or false: Dave is quiet.

Solution:
Context:
Premises:
1. Dave is strong.
2. Dave is high.
3. Dave is big.
4. Harry is small.
5. Harry is little.
6. Charlie is nice.
7. Charlie is quiet.
8. Charlie is wealthy.
9. Erin is poor.
10. Erin is sad.
11. Erin is dull.
12. Strong people are nice.
13. If someone is small and little then they are short.
14. If someone is poor and sad then they are rough.
15. If someone is nice and quiet then they are kind.
16. All short people are thin.
17. All nice people are quiet.
18. All kind people are smart.
19. All rough people are bad.

Derivation:
20. We are given: Dave is strong.
21. Instantiating rule 12 for Dave: if Dave is strong, then Dave is nice.
22. Therefore, Dave is nice.
23. Instantiating rule 17 for Dave: if Dave is nice, then Dave is quiet.
24. Therefore, Dave is quiet.

Conclusion:
Dave is quiet.

Final answer: True<|endoftext|>Dave is strong. Dave is high. Dave is big. Harry is small. Harry is little. Charlie is nice. Charlie is quiet. Charlie is wealthy. Erin is poor. Erin is sad. Erin is dull. Strong people are nice. If someone is small and little then they are short. If someone is poor and sad then they are rough. If someone is nice and quiet then they are kind. All short people are thin. All nice people are quiet. All kind people are smart. All rough people are bad.

True or false: Dave is not quiet.

Solution:
Context:
Premises:
1. Dave is strong.
2. Dave is high.
3. Dave is big.
4. Harry is small.
5. Harry is little.
6. Charlie is nice.
7. Charlie is quiet.
8. Charlie is wealthy.
9. Erin is poor.
10. Erin is sad.
11. Erin is dull.
12. Strong people are nice.
13. If someone is small and little then they are short.
14. If someone is poor and sad then they are rough.
15. If someone is nice and quiet then they are kind.
16. All short people are thin.
17. All nice people are quiet.
18. All kind people are smart.
19. All rough people are bad.

Derivation:
20. We are given: Dave is strong.
21. Instantiating rule 12 for Dave: if Dave is strong, then Dave is nice.
22. Therefore, Dave is nice.
23. Instantiating rule 17 for Dave: if Dave is nice, then Dave is quiet.
24. Therefore, Dave is quiet.

Conclusion:
Dave is quiet.

Final answer: False<|endoftext|>Dave is strong. Dave is high. Dave is big. Harry is small. Harry is little. Charlie is nice. Charlie is quiet. Charlie is wealthy. Erin is poor. Erin is sad. Erin is dull. Strong people are nice. If someone is small and little then they are short. If someone is poor and sad then they are rough. If someone is nice and quiet then they are kind. All short people are thin. All nice people are quiet. All kind people are smart. All rough people are bad.

True or false: Harry is thin.

Solution:
Context:
Premises:
1. Dave is strong.
2. Dave is high.
3. Dave is big.
4. Harry is small.
5. Harry is little.
6. Charlie is nice.
7. Charlie is quiet.
8. Charlie is wealthy.
9. Erin is poor.
10. Erin is sad.
11. Erin is dull.
12. Strong people are nice.
13. If someone is small and little then they are short.
14. If someone is poor and sad then they are rough.
15. If someone is nice and quiet then they are kind.
16. All short people are thin.
17. All nice people are quiet.
18. All kind people are smart.
19. All rough people are bad.

Derivation:
20. We are given: Harry is small.
21. We are given: Harry is little.
22. Combining: Harry is small and Harry is little.
23. Instantiating rule 13 for Harry: if Harry is small and Harry is little, then Harry is short.
24. Therefore, Harry is short.
25. Instantiating rule 16 for Harry: if Harry is short, then Harry is thin.
26. Therefore, Harry is thin.

Conclusion:
Harry is thin.

Final answer: True<|endoftext|>Dave is strong. Dave is high. Dave is big. Harry is small. Harry is little. Charlie is nice. Charlie is quiet. Charlie is wealthy. Erin is poor. Erin is sad. Erin is dull. Strong people are nice. If someone is small and little then they are short. If someone is poor and sad then they are rough. If someone is nice and quiet then they are kind. All short people are thin. All nice people are quiet. All kind people are smart. All rough people are bad.

True or false: Harry is not thin.

Solution:
Context:
Premises:
1. Dave is strong.
2. Dave is high.
3. Dave is big.
4. Harry is small.
5. Harry is little.
6. Charlie is nice.
7. Charlie is quiet.
8. Charlie is wealthy.
9. Erin is poor.
10. Erin is sad.
11. Erin is dull.
12. Strong people are nice.
13. If someone is small and little then they are short.
14. If someone is poor and sad then they are rough.
15. If someone is nice and quiet then they are kind.
16. All short people are thin.
17. All nice people are quiet.
18. All kind people are smart.
19. All rough people are bad.

Derivation:
20. We are given: Harry is small.
21. We are given: Harry is little.
22. Combining: Harry is small and Harry is little.
23. Instantiating rule 13 for Harry: if Harry is small and Harry is little, then Harry is short.
24. Therefore, Harry is short.
25. Instantiating rule 16 for Harry: if Harry is short, then Harry is thin.
26. Therefore, Harry is thin.

Conclusion:
Harry is thin.

Final answer: False<|endoftext|>Dave is strong. Dave is high. Dave is big. Harry is small. Harry is little. Charlie is nice. Charlie is quiet. Charlie is wealthy. Erin is poor. Erin is sad. Erin is dull. Strong people are nice. If someone is small and little then they are short. If someone is poor and sad then they are rough. If someone is nice and quiet then they are kind. All short people are thin. All nice people are quiet. All kind people are smart. All rough people are bad.

True or false: Charlie is smart.

Solution:
Context:
Premises:
1. Dave is strong.
2. Dave is high.
3. Dave is big.
4. Harry is small.
5. Harry is little.
6. Charlie is nice.
7. Charlie is quiet.
8. Charlie is wealthy.
9. Erin is poor.
10. Erin is sad.
11. Erin is dull.
12. Strong people are nice.
13. If someone is small and little then they are short.
14. If someone is poor and sad then they are rough.
15. If someone is nice and quiet then they are kind.
16. All short people are thin.
17. All nice people are quiet.
18. All kind people are smart.
19. All rough people are bad.

Derivation:
20. We are given: Charlie is nice.
21. We are given: Charlie is quiet.
22. Combining: Charlie is nice and Charlie is quiet.
23. Instantiating rule 15 for Charlie: if Charlie is nice and Charlie is quiet, then Charlie is kind.
24. Therefore, Charlie is kind.
25. Instantiating rule 18 for Charlie: if Charlie is kind, then Charlie is smart.
26. Therefore, Charlie is smart.

Conclusion:
Charlie is smart.

Final answer: True<|endoftext|>Dave is strong. Dave is high. Dave is big. Harry is small. Harry is little. Charlie is nice. Charlie is quiet. Charlie is wealthy. Erin is poor. Erin is sad. Erin is dull. Strong people are nice. If someone is small and little then they are short. If someone is poor and sad then they are rough. If someone is nice and quiet then they are kind. All short people are thin. All nice people are quiet. All kind people are smart. All rough people are bad.

True or false: Charlie is not smart.

Solution:
Context:
Premises:
1. Dave is strong.
2. Dave is high.
3. Dave is big.
4. Harry is small.
5. Harry is little.
6. Charlie is nice.
7. Charlie is quiet.
8. Charlie is wealthy.
9. Erin is poor.
10. Erin is sad.
11. Erin is dull.
12. Strong people are nice.
13. If someone is small and little then they are short.
14. If someone is poor and sad then they are rough.
15. If someone is nice and quiet then they are kind.
16. All short people are thin.
17. All nice people are quiet.
18. All kind people are smart.
19. All rough people are bad.

Derivation:
20. We are given: Charlie is nice.
21. We are given: Charlie is quiet.
22. Combining: Charlie is nice and Charlie is quiet.
23. Instantiating rule 15 for Charlie: if Charlie is nice and Charlie is quiet, then Charlie is kind.
24. Therefore, Charlie is kind.
25. Instantiating rule 18 for Charlie: if Charlie is kind, then Charlie is smart.
26. Therefore, Charlie is smart.

Conclusion:
Charlie is smart.

Final answer: False<|endoftext|>Dave is strong. Dave is high. Dave is big. Harry is small. Harry is little. Charlie is nice. Charlie is quiet. Charlie is wealthy. Erin is poor. Erin is sad. Erin is dull. Strong people are nice. If someone is small and little then they are short. If someone is poor and sad then they are rough. If someone is nice and quiet then they are kind. All short people are thin. All nice people are quiet. All kind people are smart. All rough people are bad.

True or false: Erin is bad.

Solution:
Context:
Premises:
1. Dave is strong.
2. Dave is high.
3. Dave is big.
4. Harry is small.
5. Harry is little.
6. Charlie is nice.
7. Charlie is quiet.
8. Charlie is wealthy.
9. Erin is poor.
10. Erin is sad.
11. Erin is dull.
12. Strong people are nice.
13. If someone is small and little then they are short.
14. If someone is poor and sad then they are rough.
15. If someone is nice and quiet then they are kind.
16. All short people are thin.
17. All nice people are quiet.
18. All kind people are smart.
19. All rough people are bad.

Derivation:
20. We are given: Erin is poor.
21. We are given: Erin is sad.
22. Combining: Erin is poor and Erin is sad.
23. Instantiating rule 14 for Erin: if Erin is poor and Erin is sad, then Erin is rough.
24. Therefore, Erin is rough.
25. Instantiating rule 19 for Erin: if Erin is rough, then Erin is bad.
26. Therefore, Erin is bad.

Conclusion:
Erin is bad.

Final answer: True<|endoftext|>Dave is strong. Dave is high. Dave is big. Harry is small. Harry is little. Charlie is nice. Charlie is quiet. Charlie is wealthy. Erin is poor. Erin is sad. Erin is dull. Strong people are nice. If someone is small and little then they are short. If someone is poor and sad then they are rough. If someone is nice and quiet then they are kind. All short people are thin. All nice people are quiet. All kind people are smart. All rough people are bad.

True or false: Erin is not bad.

Solution:
Context:
Premises:
1. Dave is strong.
2. Dave is high.
3. Dave is big.
4. Harry is small.
5. Harry is little.
6. Charlie is nice.
7. Charlie is quiet.
8. Charlie is wealthy.
9. Erin is poor.
10. Erin is sad.
11. Erin is dull.
12. Strong people are nice.
13. If someone is small and little then they are short.
14. If someone is poor and sad then they are rough.
15. If someone is nice and quiet then they are kind.
16. All short people are thin.
17. All nice people are quiet.
18. All kind people are smart.
19. All rough people are bad.

Derivation:
20. We are given: Erin is poor.
21. We are given: Erin is sad.
22. Combining: Erin is poor and Erin is sad.
23. Instantiating rule 14 for Erin: if Erin is poor and Erin is sad, then Erin is rough.
24. Therefore, Erin is rough.
25. Instantiating rule 19 for Erin: if Erin is rough, then Erin is bad.
26. Therefore, Erin is bad.

Conclusion:
Erin is bad.

Final answer: False<|endoftext|>
```

## Window 3 (17 documents, 0 pad tokens)
```
The bald eagle chases the lion. The bald eagle eats the lion. The bald eagle is cold. The bald eagle is rough. The bald eagle needs the lion. The dog eats the lion. The dog is rough. The dog needs the bald eagle. The dog needs the mouse. The lion chases the mouse. The lion is cold. The lion needs the dog. The mouse chases the lion. The mouse eats the lion. The mouse needs the bald eagle. The mouse needs the lion. If someone is rough and they chase the bald eagle then they chase the mouse. If someone chases the dog then the dog chases the mouse. All cold people are blue. If someone eats the mouse and the mouse is nice then the mouse eats the bald eagle. If someone is red and they need the mouse then the mouse is blue. If someone chases the bald eagle and the bald eagle is rough then they are cold. If the mouse needs the bald eagle then the bald eagle is rough. If someone needs the lion and the lion needs the mouse then they need the dog.

True or false: The lion is blue.

Solution:
Context:
Premises:
1. The bald eagle chases the lion.
2. The bald eagle eats the lion.
3. The bald eagle is cold.
4. The bald eagle is rough.
5. The bald eagle needs the lion.
6. The dog eats the lion.
7. The dog is rough.
8. The dog needs the bald eagle.
9. The dog needs the mouse.
10. The lion chases the mouse.
11. The lion is cold.
12. The lion needs the dog.
13. The mouse chases the lion.
14. The mouse eats the lion.
15. The mouse needs the bald eagle.
16. The mouse needs the lion.
17. If someone is rough and they chase the bald eagle then they chase the mouse.
18. If someone chases the dog then the dog chases the mouse.
19. All cold people are blue.
20. If someone eats the mouse and the mouse is nice then the mouse eats the bald eagle.
21. If someone is red and they need the mouse then the mouse is blue.
22. If someone chases the bald eagle and the bald eagle is rough then they are cold.
23. If the mouse needs the bald eagle then the bald eagle is rough.
24. If someone needs the lion and the lion needs the mouse then they need the dog.

Derivation:
25. We are given: The lion is cold.
26. Instantiating rule 19 for the lion: if The lion is cold, then The lion is blue.
27. Therefore, The lion is blue.

Conclusion:
The lion is blue.

Final answer: True<|endoftext|>The bald eagle chases the lion. The bald eagle eats the lion. The bald eagle is cold. The bald eagle is rough. The bald eagle needs the lion. The dog eats the lion. The dog is rough. The dog needs the bald eagle. The dog needs the mouse. The lion chases the mouse. The lion is cold. The lion needs the dog. The mouse chases the lion. The mouse eats the lion. The mouse needs the bald eagle. The mouse needs the lion. If someone is rough and they chase the bald eagle then they chase the mouse. If someone chases the dog then the dog chases the mouse. All cold people are blue. If someone eats the mouse and the mouse is nice then the mouse eats the bald eagle. If someone is red and they need the mouse then the mouse is blue. If someone chases the bald eagle and the bald eagle is rough then they are cold. If the mouse needs the bald eagle then the bald eagle is rough. If someone needs the lion and the lion needs the mouse then they need the dog.

True or false: The bald eagle is not blue.

Solution:
Context:
Premises:
1. The bald eagle chases the lion.
2. The bald eagle eats the lion.
3. The bald eagle is cold.
4. The bald eagle is rough.
5. The bald eagle needs the lion.
6. The dog eats the lion.
7. The dog is rough.
8. The dog needs the bald eagle.
9. The dog needs the mouse.
10. The lion chases the mouse.
11. The lion is cold.
12. The lion needs the dog.
13. The mouse chases the lion.
14. The mouse eats the lion.
15. The mouse needs the bald eagle.
16. The mouse needs the lion.
17. If someone is rough and they chase the bald eagle then they chase the mouse.
18. If someone chases the dog then the dog chases the mouse.
19. All cold people are blue.
20. If someone eats the mouse and the mouse is nice then the mouse eats the bald eagle.
21. If someone is red and they need the mouse then the mouse is blue.
22. If someone chases the bald eagle and the bald eagle is rough then they are cold.
23. If the mouse needs the bald eagle then the bald eagle is rough.
24. If someone needs the lion and the lion needs the mouse then they need the dog.

Derivation:
25. We are given: The bald eagle is cold.
26. Instantiating rule 19 for the bald eagle: if The bald eagle is cold, then The bald eagle is blue.
27. Therefore, The bald eagle is blue.

Conclusion:
The bald eagle is blue.

Final answer: False<|endoftext|>Fiona is big. Fiona is cold. Fiona is white. Gary is cold. Gary is nice. Gary is quiet. Gary is white. If Fiona is quiet then Fiona is white. If Gary is kind and Gary is cold then Gary is big. If Gary is white then Gary is kind. Kind things are cold. If Fiona is quiet then Fiona is blue. All blue things are kind.

True or false: Gary is white.

Solution:
Context:
Premises:
1. Fiona is big.
2. Fiona is cold.
3. Fiona is white.
4. Gary is cold.
5. Gary is nice.
6. Gary is quiet.
7. Gary is white.
8. If Fiona is quiet then Fiona is white.
9. If Gary is kind and Gary is cold then Gary is big.
10. If Gary is white then Gary is kind.
11. Kind things are cold.
12. If Fiona is quiet then Fiona is blue.
13. All blue things are kind.

Derivation:
14. We are given: Gary is white.

Conclusion:
Gary is white.

Final answer: True<|endoftext|>Fiona is big. Fiona is cold. Fiona is white. Gary is cold. Gary is nice. Gary is quiet. Gary is white. If Fiona is quiet then Fiona is white. If Gary is kind and Gary is cold then Gary is big. If Gary is white then Gary is kind. Kind things are cold. If Fiona is quiet then Fiona is blue. All blue things are kind.

True or false: Fiona is not big.

Solution:
Context:
Premises:
1. Fiona is big.
2. Fiona is cold.
3. Fiona is white.
4. Gary is cold.
5. Gary is nice.
6. Gary is quiet.
7. Gary is white.
8. If Fiona is quiet then Fiona is white.
9. If Gary is kind and Gary is cold then Gary is big.
10. If Gary is white then Gary is kind.
11. Kind things are cold.
12. If Fiona is quiet then Fiona is blue.
13. All blue things are kind.

Derivation:
14. We are given: Fiona is big.

Conclusion:
Fiona is big.

Final answer: False<|endoftext|>Fiona is big. Fiona is cold. Fiona is white. Gary is cold. Gary is nice. Gary is quiet. Gary is white. If Fiona is quiet then Fiona is white. If Gary is kind and Gary is cold then Gary is big. If Gary is white then Gary is kind. Kind things are cold. If Fiona is quiet then Fiona is blue. All blue things are kind.

True or false: Gary is kind.

Solution:
Context:
Premises:
1. Fiona is big.
2. Fiona is cold.
3. Fiona is white.
4. Gary is cold.
5. Gary is nice.
6. Gary is quiet.
7. Gary is white.
8. If Fiona is quiet then Fiona is white.
9. If Gary is kind and Gary is cold then Gary is big.
10. If Gary is white then Gary is kind.
11. Kind things are cold.
12. If Fiona is quiet then Fiona is blue.
13. All blue things are kind.

Derivation:
14. We are given: Gary is white.
15. Therefore, Gary is kind.

Conclusion:
Gary is kind.

Final answer: True<|endoftext|>Fiona is big. Fiona is cold. Fiona is white. Gary is cold. Gary is nice. Gary is quiet. Gary is white. If Fiona is quiet then Fiona is white. If Gary is kind and Gary is cold then Gary is big. If Gary is white then Gary is kind. Kind things are cold. If Fiona is quiet then Fiona is blue. All blue things are kind.

True or false: Gary is not kind.

Solution:
Context:
Premises:
1. Fiona is big.
2. Fiona is cold.
3. Fiona is white.
4. Gary is cold.
5. Gary is nice.
6. Gary is quiet.
7. Gary is white.
8. If Fiona is quiet then Fiona is white.
9. If Gary is kind and Gary is cold then Gary is big.
10. If Gary is white then Gary is kind.
11. Kind things are cold.
12. If Fiona is quiet then Fiona is blue.
13. All blue things are kind.

Derivation:
14. We are given: Gary is white.
15. Therefore, Gary is kind.

Conclusion:
Gary is kind.

Final answer: False<|endoftext|>The cat needs the cow. The cow needs the cat. If someone needs the cow then they need the cat. If the cow is rough then the cow is nice. If someone sees the cow and the cow likes the cat then the cow is nice.

True or false: The cat needs the cow.

Solution:
Context:
Premises:
1. The cat needs the cow.
2. The cow needs the cat.
3. If someone needs the cow then they need the cat.
4. If the cow is rough then the cow is nice.
5. If someone sees the cow and the cow likes the cat then the cow is nice.

Derivation:
6. We are given: The cat needs the cow.

Conclusion:
The cat needs the cow.

Final answer: True<|endoftext|>The cat needs the cow. The cow needs the cat. If someone needs the cow then they need the cat. If the cow is rough then the cow is nice. If someone sees the cow and the cow likes the cat then the cow is nice.

True or false: The cat does not need the cow.

Solution:
Context:
Premises:
1. The cat needs the cow.
2. The cow needs the cat.
3. If someone needs the cow then they need the cat.
4. If the cow is rough then the cow is nice.
5. If someone sees the cow and the cow likes the cat then the cow is nice.

Derivation:
6. We are given: The cat needs the cow.

Conclusion:
The cat needs the cow.

Final answer: False<|endoftext|>The cat needs the cow. The cow needs the cat. If someone needs the cow then they need the cat. If the cow is rough then the cow is nice. If someone sees the cow and the cow likes the cat then the cow is nice.

True or false: The cat needs the cat.

Solution:
Context:
Premises:
1. The cat needs the cow.
2. The cow needs the cat.
3. If someone needs the cow then they need the cat.
4. If the cow is rough then the cow is nice.
5. If someone sees the cow and the cow likes the cat then the cow is nice.

Derivation:
6. We are given: The cat needs the cow.
7. Instantiating rule 3 for the cat: if The cat needs the cow, then The cat needs the cat.
8. Therefore, The cat needs the cat.

Conclusion:
The cat needs the cat.

Final answer: True<|endoftext|>The cat needs the cow. The cow needs the cat. If someone needs the cow then they need the cat. If the cow is rough then the cow is nice. If someone sees the cow and the cow likes the cat then the cow is nice.

True or false: The cat does not need the cat.

Solution:
Context:
Premises:
1. The cat needs the cow.
2. The cow needs the cat.
3. If someone needs the cow then they need the cat.
4. If the cow is rough then the cow is nice.
5. If someone sees the cow and the cow likes the cat then the cow is nice.

Derivation:
6. We are given: The cat needs the cow.
7. Instantiating rule 3 for the cat: if The cat needs the cow, then The cat needs the cat.
8. Therefore, The cat needs the cat.

Conclusion:
The cat needs the cat.

Final answer: False<|endoftext|>Bob is cold. Bob is nice. Erin is quiet. Erin is white. Harry is big. Harry is smart. Harry is white. All quiet, nice things are white. All quiet things are big. If something is quiet then it is smart. Big things are cold. If Harry is cold and Harry is white then Harry is quiet. If Harry is big and Harry is smart then Harry is cold.

True or false: Erin is quiet.

Solution:
Context:
Premises:
1. Bob is cold.
2. Bob is nice.
3. Erin is quiet.
4. Erin is white.
5. Harry is big.
6. Harry is smart.
7. Harry is white.
8. All quiet, nice things are white.
9. All quiet things are big.
10. If something is quiet then it is smart.
11. Big things are cold.
12. If Harry is cold and Harry is white then Harry is quiet.
13. If Harry is big and Harry is smart then Harry is cold.

Derivation:
14. We are given: Erin is quiet.

Conclusion:
Erin is quiet.

Final answer: True<|endoftext|>Bob is cold. Bob is nice. Erin is quiet. Erin is white. Harry is big. Harry is smart. Harry is white. All quiet, nice things are white. All quiet things are big. If something is quiet then it is smart. Big things are cold. If Harry is cold and Harry is white then Harry is quiet. If Harry is big and Harry is smart then Harry is cold.

True or false: Harry is not big.

Solution:
Context:
Premises:
1. Bob is cold.
2. Bob is nice.
3. Erin is quiet.
4. Erin is white.
5. Harry is big.
6. Harry is smart.
7. Harry is white.
8. All quiet, nice things are white.
9. All quiet things are big.
10. If something is quiet then it is smart.
11. Big things are cold.
12. If Harry is cold and Harry is white then Harry is quiet.
13. If Harry is big and Harry is smart then Harry is cold.

Derivation:
14. We are given: Harry is big.

Conclusion:
Harry is big.

Final answer: False<|endoftext|>Bob is cold. Bob is nice. Erin is quiet. Erin is white. Harry is big. Harry is smart. Harry is white. All quiet, nice things are white. All quiet things are big. If something is quiet then it is smart. Big things are cold. If Harry is cold and Harry is white then Harry is quiet. If Harry is big and Harry is smart then Harry is cold.

True or false: Harry is cold.

Solution:
Context:
Premises:
1. Bob is cold.
2. Bob is nice.
3. Erin is quiet.
4. Erin is white.
5. Harry is big.
6. Harry is smart.
7. Harry is white.
8. All quiet, nice things are white.
9. All quiet things are big.
10. If something is quiet then it is smart.
11. Big things are cold.
12. If Harry is cold and Harry is white then Harry is quiet.
13. If Harry is big and Harry is smart then Harry is cold.

Derivation:
14. We are given: Harry is big.
15. Instantiating rule 11 for Harry: if Harry is big, then Harry is cold.
16. Therefore, Harry is cold.

Conclusion:
Harry is cold.

Final answer: True<|endoftext|>Bob is cold. Bob is nice. Erin is quiet. Erin is white. Harry is big. Harry is smart. Harry is white. All quiet, nice things are white. All quiet things are big. If something is quiet then it is smart. Big things are cold. If Harry is cold and Harry is white then Harry is quiet. If Harry is big and Harry is smart then Harry is cold.

True or false: Harry is not cold.

Solution:
Context:
Premises:
1. Bob is cold.
2. Bob is nice.
3. Erin is quiet.
4. Erin is white.
5. Harry is big.
6. Harry is smart.
7. Harry is white.
8. All quiet, nice things are white.
9. All quiet things are big.
10. If something is quiet then it is smart.
11. Big things are cold.
12. If Harry is cold and Harry is white then Harry is quiet.
13. If Harry is big and Harry is smart then Harry is cold.

Derivation:
14. We are given: Harry is big.
15. Instantiating rule 11 for Harry: if Harry is big, then Harry is cold.
16. Therefore, Harry is cold.

Conclusion:
Harry is cold.

Final answer: False<|endoftext|>Bob is big. Erin is kind. Gary is kind. If someone is blue then they are not smart. Kind people are cold.

True or false: Erin is kind.

Solution:
Context:
Premises:
1. Bob is big.
2. Erin is kind.
3. Gary is kind.
4. If someone is blue then they are not smart.
5. Kind people are cold.

Derivation:
6. We are given: Erin is kind.

Conclusion:
Erin is kind.

Final answer: True<|endoftext|>Bob is big. Erin is kind. Gary is kind. If someone is blue then they are not smart. Kind people are cold.

True or false: Gary is not kind.

Solution:
Context:
Premises:
1. Bob is big.
2. Erin is kind.
3. Gary is kind.
4. If someone is blue then they are not smart.
5. Kind people are cold.

Derivation:
6. We are given: Gary is kind.

Conclusion:
Gary is kind.

Final answer: False<|endoftext|>The bald eagle is cold. If someone is cold then they are young.

True or false: The bald eagle is cold.

Solution:
Context:
Premises:
1. The bald eagle is cold.
2. If someone is cold then they are young.

Derivation:
3. We are given: The bald eagle is cold.

Conclusion:
The bald eagle is cold.

Final answer: True<|endoftext|>
```

## Window 2558 (15 documents, 68 pad tokens)
```
The bald eagle eats the bear. The bald eagle is young. The bald eagle needs the bear. The bear eats the bald eagle. The bear eats the lion. The lion does not eat the bear. The lion is nice. If the bald eagle eats the lion then the bald eagle is not kind. If someone is young then they eat the lion. If someone is round and they need the bald eagle then the bald eagle does not need the bear. If the bear is cold then the bear needs the bald eagle. If someone visits the bear then they need the lion. If the bald eagle eats the lion and the bald eagle is not kind then the bald eagle is cold.

True or false: The bald eagle is not kind.

Solution:
Context:
Premises:
1. The bald eagle eats the bear.
2. The bald eagle is young.
3. The bald eagle needs the bear.
4. The bear eats the bald eagle.
5. The bear eats the lion.
6. The lion does not eat the bear.
7. The lion is nice.
8. If the bald eagle eats the lion then the bald eagle is not kind.
9. If someone is young then they eat the lion.
10. If someone is round and they need the bald eagle then the bald eagle does not need the bear.
11. If the bear is cold then the bear needs the bald eagle.
12. If someone visits the bear then they need the lion.
13. If the bald eagle eats the lion and the bald eagle is not kind then the bald eagle is cold.

Derivation:
14. We are given: The bald eagle is young.
15. Instantiating rule 9 for the bald eagle: if The bald eagle is young, then The bald eagle eats the lion.
16. Therefore, The bald eagle eats the lion.
17. Therefore, The bald eagle is not kind.

Conclusion:
The bald eagle is not kind.

Final answer: True<|endoftext|>The bald eagle eats the bear. The bald eagle is young. The bald eagle needs the bear. The bear eats the bald eagle. The bear eats the lion. The lion does not eat the bear. The lion is nice. If the bald eagle eats the lion then the bald eagle is not kind. If someone is young then they eat the lion. If someone is round and they need the bald eagle then the bald eagle does not need the bear. If the bear is cold then the bear needs the bald eagle. If someone visits the bear then they need the lion. If the bald eagle eats the lion and the bald eagle is not kind then the bald eagle is cold.

True or false: The bald eagle is kind.

Solution:
Context:
Premises:
1. The bald eagle eats the bear.
2. The bald eagle is young.
3. The bald eagle needs the bear.
4. The bear eats the bald eagle.
5. The bear eats the lion.
6. The lion does not eat the bear.
7. The lion is nice.
8. If the bald eagle eats the lion then the bald eagle is not kind.
9. If someone is young then they eat the lion.
10. If someone is round and they need the bald eagle then the bald eagle does not need the bear.
11. If the bear is cold then the bear needs the bald eagle.
12. If someone visits the bear then they need the lion.
13. If the bald eagle eats the lion and the bald eagle is not kind then the bald eagle is cold.

Derivation:
14. We are given: The bald eagle is young.
15. Instantiating rule 9 for the bald eagle: if The bald eagle is young, then The bald eagle eats the lion.
16. Therefore, The bald eagle eats the lion.
17. Therefore, The bald eagle is not kind.

Conclusion:
The bald eagle is not kind.

Final answer: False<|endoftext|>The bear likes the squirrel. The cat likes the squirrel. The squirrel chases the bear. The tiger is blue. If someone likes the squirrel and the squirrel chases the tiger then the squirrel visits the bear. If someone likes the bear then they chase the squirrel. If someone chases the bear then they chase the squirrel. If someone is rough then they chase the squirrel. If someone chases the squirrel then the squirrel is round. If the bear is kind and the bear likes the cat then the bear does not visit the squirrel.

True or false: The squirrel chases the bear.

Solution:
Context:
Premises:
1. The bear likes the squirrel.
2. The cat likes the squirrel.
3. The squirrel chases the bear.
4. The tiger is blue.
5. If someone likes the squirrel and the squirrel chases the tiger then the squirrel visits the bear.
6. If someone likes the bear then they chase the squirrel.
7. If someone chases the bear then they chase the squirrel.
8. If someone is rough then they chase the squirrel.
9. If someone chases the squirrel then the squirrel is round.
10. If the bear is kind and the bear likes the cat then the bear does not visit the squirrel.

Derivation:
11. We are given: The squirrel chases the bear.

Conclusion:
The squirrel chases the bear.

Final answer: True<|endoftext|>The bear likes the squirrel. The cat likes the squirrel. The squirrel chases the bear. The tiger is blue. If someone likes the squirrel and the squirrel chases the tiger then the squirrel visits the bear. If someone likes the bear then they chase the squirrel. If someone chases the bear then they chase the squirrel. If someone is rough then they chase the squirrel. If someone chases the squirrel then the squirrel is round. If the bear is kind and the bear likes the cat then the bear does not visit the squirrel.

True or false: The tiger is not blue.

Solution:
Context:
Premises:
1. The bear likes the squirrel.
2. The cat likes the squirrel.
3. The squirrel chases the bear.
4. The tiger is blue.
5. If someone likes the squirrel and the squirrel chases the tiger then the squirrel visits the bear.
6. If someone likes the bear then they chase the squirrel.
7. If someone chases the bear then they chase the squirrel.
8. If someone is rough then they chase the squirrel.
9. If someone chases the squirrel then the squirrel is round.
10. If the bear is kind and the bear likes the cat then the bear does not visit the squirrel.

Derivation:
11. We are given: The tiger is blue.

Conclusion:
The tiger is blue.

Final answer: False<|endoftext|>The bear likes the squirrel. The cat likes the squirrel. The squirrel chases the bear. The tiger is blue. If someone likes the squirrel and the squirrel chases the tiger then the squirrel visits the bear. If someone likes the bear then they chase the squirrel. If someone chases the bear then they chase the squirrel. If someone is rough then they chase the squirrel. If someone chases the squirrel then the squirrel is round. If the bear is kind and the bear likes the cat then the bear does not visit the squirrel.

True or false: The squirrel chases the squirrel.

Solution:
Context:
Premises:
1. The bear likes the squirrel.
2. The cat likes the squirrel.
3. The squirrel chases the bear.
4. The tiger is blue.
5. If someone likes the squirrel and the squirrel chases the tiger then the squirrel visits the bear.
6. If someone likes the bear then they chase the squirrel.
7. If someone chases the bear then they chase the squirrel.
8. If someone is rough then they chase the squirrel.
9. If someone chases the squirrel then the squirrel is round.
10. If the bear is kind and the bear likes the cat then the bear does not visit the squirrel.

Derivation:
11. We are given: The squirrel chases the bear.
12. Instantiating rule 7 for the squirrel: if The squirrel chases the bear, then The squirrel chases the squirrel.
13. Therefore, The squirrel chases the squirrel.

Conclusion:
The squirrel chases the squirrel.

Final answer: True<|endoftext|>The bear likes the squirrel. The cat likes the squirrel. The squirrel chases the bear. The tiger is blue. If someone likes the squirrel and the squirrel chases the tiger then the squirrel visits the bear. If someone likes the bear then they chase the squirrel. If someone chases the bear then they chase the squirrel. If someone is rough then they chase the squirrel. If someone chases the squirrel then the squirrel is round. If the bear is kind and the bear likes the cat then the bear does not visit the squirrel.

True or false: The squirrel does not chase the squirrel.

Solution:
Context:
Premises:
1. The bear likes the squirrel.
2. The cat likes the squirrel.
3. The squirrel chases the bear.
4. The tiger is blue.
5. If someone likes the squirrel and the squirrel chases the tiger then the squirrel visits the bear.
6. If someone likes the bear then they chase the squirrel.
7. If someone chases the bear then they chase the squirrel.
8. If someone is rough then they chase the squirrel.
9. If someone chases the squirrel then the squirrel is round.
10. If the bear is kind and the bear likes the cat then the bear does not visit the squirrel.

Derivation:
11. We are given: The squirrel chases the bear.
12. Instantiating rule 7 for the squirrel: if The squirrel chases the bear, then The squirrel chases the squirrel.
13. Therefore, The squirrel chases the squirrel.

Conclusion:
The squirrel chases the squirrel.

Final answer: False<|endoftext|>The bear likes the squirrel. The cat likes the squirrel. The squirrel chases the bear. The tiger is blue. If someone likes the squirrel and the squirrel chases the tiger then the squirrel visits the bear. If someone likes the bear then they chase the squirrel. If someone chases the bear then they chase the squirrel. If someone is rough then they chase the squirrel. If someone chases the squirrel then the squirrel is round. If the bear is kind and the bear likes the cat then the bear does not visit the squirrel.

True or false: The squirrel is round.

Solution:
Context:
Premises:
1. The bear likes the squirrel.
2. The cat likes the squirrel.
3. The squirrel chases the bear.
4. The tiger is blue.
5. If someone likes the squirrel and the squirrel chases the tiger then the squirrel visits the bear.
6. If someone likes the bear then they chase the squirrel.
7. If someone chases the bear then they chase the squirrel.
8. If someone is rough then they chase the squirrel.
9. If someone chases the squirrel then the squirrel is round.
10. If the bear is kind and the bear likes the cat then the bear does not visit the squirrel.

Derivation:
11. We are given: The squirrel chases the bear.
12. Instantiating rule 7 for the squirrel: if The squirrel chases the bear, then The squirrel chases the squirrel.
13. Therefore, The squirrel chases the squirrel.
14. Instantiating rule 9 for the squirrel: if The squirrel chases the squirrel, then The squirrel is round.
15. Therefore, The squirrel is round.

Conclusion:
The squirrel is round.

Final answer: True<|endoftext|>The bear likes the squirrel. The cat likes the squirrel. The squirrel chases the bear. The tiger is blue. If someone likes the squirrel and the squirrel chases the tiger then the squirrel visits the bear. If someone likes the bear then they chase the squirrel. If someone chases the bear then they chase the squirrel. If someone is rough then they chase the squirrel. If someone chases the squirrel then the squirrel is round. If the bear is kind and the bear likes the cat then the bear does not visit the squirrel.

True or false: The squirrel is not round.

Solution:
Context:
Premises:
1. The bear likes the squirrel.
2. The cat likes the squirrel.
3. The squirrel chases the bear.
4. The tiger is blue.
5. If someone likes the squirrel and the squirrel chases the tiger then the squirrel visits the bear.
6. If someone likes the bear then they chase the squirrel.
7. If someone chases the bear then they chase the squirrel.
8. If someone is rough then they chase the squirrel.
9. If someone chases the squirrel then the squirrel is round.
10. If the bear is kind and the bear likes the cat then the bear does not visit the squirrel.

Derivation:
11. We are given: The squirrel chases the bear.
12. Instantiating rule 7 for the squirrel: if The squirrel chases the bear, then The squirrel chases the squirrel.
13. Therefore, The squirrel chases the squirrel.
14. Instantiating rule 9 for the squirrel: if The squirrel chases the squirrel, then The squirrel is round.
15. Therefore, The squirrel is round.

Conclusion:
The squirrel is round.

Final answer: False<|endoftext|>The bald eagle eats the cow. The cow needs the bald eagle. The dog needs the bald eagle. If the cow eats the dog then the dog sees the cow. If the cow needs the bald eagle then the cow sees the dog. If something sees the dog then it needs the cow.

True or false: The bald eagle eats the cow.

Solution:
Context:
Premises:
1. The bald eagle eats the cow.
2. The cow needs the bald eagle.
3. The dog needs the bald eagle.
4. If the cow eats the dog then the dog sees the cow.
5. If the cow needs the bald eagle then the cow sees the dog.
6. If something sees the dog then it needs the cow.

Derivation:
7. We are given: The bald eagle eats the cow.

Conclusion:
The bald eagle eats the cow.

Final answer: True<|endoftext|>The bald eagle eats the cow. The cow needs the bald eagle. The dog needs the bald eagle. If the cow eats the dog then the dog sees the cow. If the cow needs the bald eagle then the cow sees the dog. If something sees the dog then it needs the cow.

True or false: The cow does not need the bald eagle.

Solution:
Context:
Premises:
1. The bald eagle eats the cow.
2. The cow needs the bald eagle.
3. The dog needs the bald eagle.
4. If the cow eats the dog then the dog sees the cow.
5. If the cow needs the bald eagle then the cow sees the dog.
6. If something sees the dog then it needs the cow.

Derivation:
7. We are given: The cow needs the bald eagle.

Conclusion:
The cow needs the bald eagle.

Final answer: False<|endoftext|>The bald eagle eats the cow. The cow needs the bald eagle. The dog needs the bald eagle. If the cow eats the dog then the dog sees the cow. If the cow needs the bald eagle then the cow sees the dog. If something sees the dog then it needs the cow.

True or false: The cow sees the dog.

Solution:
Context:
Premises:
1. The bald eagle eats the cow.
2. The cow needs the bald eagle.
3. The dog needs the bald eagle.
4. If the cow eats the dog then the dog sees the cow.
5. If the cow needs the bald eagle then the cow sees the dog.
6. If something sees the dog then it needs the cow.

Derivation:
7. We are given: The cow needs the bald eagle.
8. Therefore, The cow sees the dog.

Conclusion:
The cow sees the dog.

Final answer: True<|endoftext|>The bald eagle eats the cow. The cow needs the bald eagle. The dog needs the bald eagle. If the cow eats the dog then the dog sees the cow. If the cow needs the bald eagle then the cow sees the dog. If something sees the dog then it needs the cow.

True or false: The cow does not see the dog.

Solution:
Context:
Premises:
1. The bald eagle eats the cow.
2. The cow needs the bald eagle.
3. The dog needs the bald eagle.
4. If the cow eats the dog then the dog sees the cow.
5. If the cow needs the bald eagle then the cow sees the dog.
6. If something sees the dog then it needs the cow.

Derivation:
7. We are given: The cow needs the bald eagle.
8. Therefore, The cow sees the dog.

Conclusion:
The cow sees the dog.

Final answer: False<|endoftext|>The bald eagle eats the cow. The cow needs the bald eagle. The dog needs the bald eagle. If the cow eats the dog then the dog sees the cow. If the cow needs the bald eagle then the cow sees the dog. If something sees the dog then it needs the cow.

True or false: The cow needs the cow.

Solution:
Context:
Premises:
1. The bald eagle eats the cow.
2. The cow needs the bald eagle.
3. The dog needs the bald eagle.
4. If the cow eats the dog then the dog sees the cow.
5. If the cow needs the bald eagle then the cow sees the dog.
6. If something sees the dog then it needs the cow.

Derivation:
7. We are given: The cow needs the bald eagle.
8. Therefore, The cow sees the dog.
9. Instantiating rule 6 for the cow: if The cow sees the dog, then The cow needs the cow.
10. Therefore, The cow needs the cow.

Conclusion:
The cow needs the cow.

Final answer: True<|endoftext|>The bald eagle eats the cow. The cow needs the bald eagle. The dog needs the bald eagle. If the cow eats the dog then the dog sees the cow. If the cow needs the bald eagle then the cow sees the dog. If something sees the dog then it needs the cow.

True or false: The cow does not need the cow.

Solution:
Context:
Premises:
1. The bald eagle eats the cow.
2. The cow needs the bald eagle.
3. The dog needs the bald eagle.
4. If the cow eats the dog then the dog sees the cow.
5. If the cow needs the bald eagle then the cow sees the dog.
6. If something sees the dog then it needs the cow.

Derivation:
7. We are given: The cow needs the bald eagle.
8. Therefore, The cow sees the dog.
9. Instantiating rule 6 for the cow: if The cow sees the dog, then The cow needs the cow.
10. Therefore, The cow needs the cow.

Conclusion:
The cow needs the cow.

Final answer: False<|endoftext|>The bear is big. The bear is not kind. The bear is rough. All big people are green. Green, big people are cold.

True or false: The bear is not big.

Solution:
Context:
Premises:
1. The bear is big.
2. The bear is not kind.
3. The bear is rough.
4. All big people are green.
5. Green, big people are cold.

Derivation:
6. We are given: The bear is big.

Conclusion:
The bear is big.

Final answer: False<|endoftext|>
```

## Window 7644 (7 documents, 223 pad tokens)
```
Anne is huge. Anne is strong. Anne is big. Alan is short. Alan is small. Harry is smart. Harry is wealthy. Harry is quiet. Erin is sad. Erin is imperfect. Erin is poor. Huge people are smart. If someone is short and small then they are little. If someone is sad and imperfect then they are dull. If someone is smart and wealthy then they are kind. If someone is little then they are tiny. All tiny people are thin. If someone is smart then they are wealthy. All wealthy people are quiet. If someone is kind then they are nice. All nice people are clever. If someone is dull then they are rough. All rough people are bad.

True or false: Harry is not clever.

Solution:
Context:
Premises:
1. Anne is huge.
2. Anne is strong.
3. Anne is big.
4. Alan is short.
5. Alan is small.
6. Harry is smart.
7. Harry is wealthy.
8. Harry is quiet.
9. Erin is sad.
10. Erin is imperfect.
11. Erin is poor.
12. Huge people are smart.
13. If someone is short and small then they are little.
14. If someone is sad and imperfect then they are dull.
15. If someone is smart and wealthy then they are kind.
16. If someone is little then they are tiny.
17. All tiny people are thin.
18. If someone is smart then they are wealthy.
19. All wealthy people are quiet.
20. If someone is kind then they are nice.
21. All nice people are clever.
22. If someone is dull then they are rough.
23. All rough people are bad.

Derivation:
24. We are given: Harry is smart.
25. We are given: Harry is wealthy.
26. Combining: Harry is smart and Harry is wealthy.
27. Instantiating rule 15 for Harry: if Harry is smart and Harry is wealthy, then Harry is kind.
28. Therefore, Harry is kind.
29. Instantiating rule 20 for Harry: if Harry is kind, then Harry is nice.
30. Therefore, Harry is nice.
31. Instantiating rule 21 for Harry: if Harry is nice, then Harry is clever.
32. Therefore, Harry is clever.

Conclusion:
Harry is clever.

Final answer: False<|endoftext|>Anne is huge. Anne is strong. Anne is big. Alan is short. Alan is small. Harry is smart. Harry is wealthy. Harry is quiet. Erin is sad. Erin is imperfect. Erin is poor. Huge people are smart. If someone is short and small then they are little. If someone is sad and imperfect then they are dull. If someone is smart and wealthy then they are kind. If someone is little then they are tiny. All tiny people are thin. If someone is smart then they are wealthy. All wealthy people are quiet. If someone is kind then they are nice. All nice people are clever. If someone is dull then they are rough. All rough people are bad.

True or false: Erin is bad.

Solution:
Context:
Premises:
1. Anne is huge.
2. Anne is strong.
3. Anne is big.
4. Alan is short.
5. Alan is small.
6. Harry is smart.
7. Harry is wealthy.
8. Harry is quiet.
9. Erin is sad.
10. Erin is imperfect.
11. Erin is poor.
12. Huge people are smart.
13. If someone is short and small then they are little.
14. If someone is sad and imperfect then they are dull.
15. If someone is smart and wealthy then they are kind.
16. If someone is little then they are tiny.
17. All tiny people are thin.
18. If someone is smart then they are wealthy.
19. All wealthy people are quiet.
20. If someone is kind then they are nice.
21. All nice people are clever.
22. If someone is dull then they are rough.
23. All rough people are bad.

Derivation:
24. We are given: Erin is sad.
25. We are given: Erin is imperfect.
26. Combining: Erin is sad and Erin is imperfect.
27. Instantiating rule 14 for Erin: if Erin is sad and Erin is imperfect, then Erin is dull.
28. Therefore, Erin is dull.
29. Instantiating rule 22 for Erin: if Erin is dull, then Erin is rough.
30. Therefore, Erin is rough.
31. Instantiating rule 23 for Erin: if Erin is rough, then Erin is bad.
32. Therefore, Erin is bad.

Conclusion:
Erin is bad.

Final answer: True<|endoftext|>Anne is huge. Anne is strong. Anne is big. Alan is short. Alan is small. Harry is smart. Harry is wealthy. Harry is quiet. Erin is sad. Erin is imperfect. Erin is poor. Huge people are smart. If someone is short and small then they are little. If someone is sad and imperfect then they are dull. If someone is smart and wealthy then they are kind. If someone is little then they are tiny. All tiny people are thin. If someone is smart then they are wealthy. All wealthy people are quiet. If someone is kind then they are nice. All nice people are clever. If someone is dull then they are rough. All rough people are bad.

True or false: Erin is not bad.

Solution:
Context:
Premises:
1. Anne is huge.
2. Anne is strong.
3. Anne is big.
4. Alan is short.
5. Alan is small.
6. Harry is smart.
7. Harry is wealthy.
8. Harry is quiet.
9. Erin is sad.
10. Erin is imperfect.
11. Erin is poor.
12. Huge people are smart.
13. If someone is short and small then they are little.
14. If someone is sad and imperfect then they are dull.
15. If someone is smart and wealthy then they are kind.
16. If someone is little then they are tiny.
17. All tiny people are thin.
18. If someone is smart then they are wealthy.
19. All wealthy people are quiet.
20. If someone is kind then they are nice.
21. All nice people are clever.
22. If someone is dull then they are rough.
23. All rough people are bad.

Derivation:
24. We are given: Erin is sad.
25. We are given: Erin is imperfect.
26. Combining: Erin is sad and Erin is imperfect.
27. Instantiating rule 14 for Erin: if Erin is sad and Erin is imperfect, then Erin is dull.
28. Therefore, Erin is dull.
29. Instantiating rule 22 for Erin: if Erin is dull, then Erin is rough.
30. Therefore, Erin is rough.
31. Instantiating rule 23 for Erin: if Erin is rough, then Erin is bad.
32. Therefore, Erin is bad.

Conclusion:
Erin is bad.

Final answer: False<|endoftext|>The tiger is tired. The tiger is slow. The tiger is rough. The tiger sees the squirrel. The dinosaur attacks the dog. The dinosaur is awful. The dinosaur is obese. The squirrel is smart. The squirrel is nice. The squirrel is kind. The dog is furry. The dog is lovely. The dog is cute. Smart animals are furry. If something is slow then it needs the squirrel. If something needs the squirrel then it is lazy. If something is tired and slow then it is rough. If something is furry and lovely then it is adorable. If something is awful and obese then it is fierce. If something is rough then it is dull. All dull animals are sleepy. If something is furry then it is lovely. All lovely animals are cute. If something is fierce then it is strong. All strong animals are big. If something is adorable then it is small. All small animals are beautiful. All lazy animals are heavy.

True or false: The squirrel is cute.

Solution:
Context:
Premises:
1. The tiger is tired.
2. The tiger is slow.
3. The tiger is rough.
4. The tiger sees the squirrel.
5. The dinosaur attacks the dog.
6. The dinosaur is awful.
7. The dinosaur is obese.
8. The squirrel is smart.
9. The squirrel is nice.
10. The squirrel is kind.
11. The dog is furry.
12. The dog is lovely.
13. The dog is cute.
14. Smart animals are furry.
15. If something is slow then it needs the squirrel.
16. If something needs the squirrel then it is lazy.
17. If something is tired and slow then it is rough.
18. If something is furry and lovely then it is adorable.
19. If something is awful and obese then it is fierce.
20. If something is rough then it is dull.
21. All dull animals are sleepy.
22. If something is furry then it is lovely.
23. All lovely animals are cute.
24. If something is fierce then it is strong.
25. All strong animals are big.
26. If something is adorable then it is small.
27. All small animals are beautiful.
28. All lazy animals are heavy.

Derivation:
29. We are given: The squirrel is smart.
30. Instantiating rule 14 for the squirrel: if The squirrel is smart, then The squirrel is furry.
31. Therefore, The squirrel is furry.
32. Instantiating rule 22 for the squirrel: if The squirrel is furry, then The squirrel is lovely.
33. Therefore, The squirrel is lovely.
34. Instantiating rule 23 for the squirrel: if The squirrel is lovely, then The squirrel is cute.
35. Therefore, The squirrel is cute.

Conclusion:
The squirrel is cute.

Final answer: True<|endoftext|>The tiger is tired. The tiger is slow. The tiger is rough. The tiger sees the squirrel. The dinosaur attacks the dog. The dinosaur is awful. The dinosaur is obese. The squirrel is smart. The squirrel is nice. The squirrel is kind. The dog is furry. The dog is lovely. The dog is cute. Smart animals are furry. If something is slow then it needs the squirrel. If something needs the squirrel then it is lazy. If something is tired and slow then it is rough. If something is furry and lovely then it is adorable. If something is awful and obese then it is fierce. If something is rough then it is dull. All dull animals are sleepy. If something is furry then it is lovely. All lovely animals are cute. If something is fierce then it is strong. All strong animals are big. If something is adorable then it is small. All small animals are beautiful. All lazy animals are heavy.

True or false: The squirrel is not cute.

Solution:
Context:
Premises:
1. The tiger is tired.
2. The tiger is slow.
3. The tiger is rough.
4. The tiger sees the squirrel.
5. The dinosaur attacks the dog.
6. The dinosaur is awful.
7. The dinosaur is obese.
8. The squirrel is smart.
9. The squirrel is nice.
10. The squirrel is kind.
11. The dog is furry.
12. The dog is lovely.
13. The dog is cute.
14. Smart animals are furry.
15. If something is slow then it needs the squirrel.
16. If something needs the squirrel then it is lazy.
17. If something is tired and slow then it is rough.
18. If something is furry and lovely then it is adorable.
19. If something is awful and obese then it is fierce.
20. If something is rough then it is dull.
21. All dull animals are sleepy.
22. If something is furry then it is lovely.
23. All lovely animals are cute.
24. If something is fierce then it is strong.
25. All strong animals are big.
26. If something is adorable then it is small.
27. All small animals are beautiful.
28. All lazy animals are heavy.

Derivation:
29. We are given: The squirrel is smart.
30. Instantiating rule 14 for the squirrel: if The squirrel is smart, then The squirrel is furry.
31. Therefore, The squirrel is furry.
32. Instantiating rule 22 for the squirrel: if The squirrel is furry, then The squirrel is lovely.
33. Therefore, The squirrel is lovely.
34. Instantiating rule 23 for the squirrel: if The squirrel is lovely, then The squirrel is cute.
35. Therefore, The squirrel is cute.

Conclusion:
The squirrel is cute.

Final answer: False<|endoftext|>The tiger is tired. The tiger is slow. The tiger is rough. The tiger sees the squirrel. The dinosaur attacks the dog. The dinosaur is awful. The dinosaur is obese. The squirrel is smart. The squirrel is nice. The squirrel is kind. The dog is furry. The dog is lovely. The dog is cute. Smart animals are furry. If something is slow then it needs the squirrel. If something needs the squirrel then it is lazy. If something is tired and slow then it is rough. If something is furry and lovely then it is adorable. If something is awful and obese then it is fierce. If something is rough then it is dull. All dull animals are sleepy. If something is furry then it is lovely. All lovely animals are cute. If something is fierce then it is strong. All strong animals are big. If something is adorable then it is small. All small animals are beautiful. All lazy animals are heavy.

True or false: The tiger is sleepy.

Solution:
Context:
Premises:
1. The tiger is tired.
2. The tiger is slow.
3. The tiger is rough.
4. The tiger sees the squirrel.
5. The dinosaur attacks the dog.
6. The dinosaur is awful.
7. The dinosaur is obese.
8. The squirrel is smart.
9. The squirrel is nice.
10. The squirrel is kind.
11. The dog is furry.
12. The dog is lovely.
13. The dog is cute.
14. Smart animals are furry.
15. If something is slow then it needs the squirrel.
16. If something needs the squirrel then it is lazy.
17. If something is tired and slow then it is rough.
18. If something is furry and lovely then it is adorable.
19. If something is awful and obese then it is fierce.
20. If something is rough then it is dull.
21. All dull animals are sleepy.
22. If something is furry then it is lovely.
23. All lovely animals are cute.
24. If something is fierce then it is strong.
25. All strong animals are big.
26. If something is adorable then it is small.
27. All small animals are beautiful.
28. All lazy animals are heavy.

Derivation:
29. We are given: The tiger is rough.
30. Instantiating rule 20 for the tiger: if The tiger is rough, then The tiger is dull.
31. Therefore, The tiger is dull.
32. Instantiating rule 21 for the tiger: if The tiger is dull, then The tiger is sleepy.
33. Therefore, The tiger is sleepy.

Conclusion:
The tiger is sleepy.

Final answer: True<|endoftext|>The tiger is tired. The tiger is slow. The tiger is rough. The tiger sees the squirrel. The dinosaur attacks the dog. The dinosaur is awful. The dinosaur is obese. The squirrel is smart. The squirrel is nice. The squirrel is kind. The dog is furry. The dog is lovely. The dog is cute. Smart animals are furry. If something is slow then it needs the squirrel. If something needs the squirrel then it is lazy. If something is tired and slow then it is rough. If something is furry and lovely then it is adorable. If something is awful and obese then it is fierce. If something is rough then it is dull. All dull animals are sleepy. If something is furry then it is lovely. All lovely animals are cute. If something is fierce then it is strong. All strong animals are big. If something is adorable then it is small. All small animals are beautiful. All lazy animals are heavy.

True or false: The tiger is not sleepy.

Solution:
Context:
Premises:
1. The tiger is tired.
2. The tiger is slow.
3. The tiger is rough.
4. The tiger sees the squirrel.
5. The dinosaur attacks the dog.
6. The dinosaur is awful.
7. The dinosaur is obese.
8. The squirrel is smart.
9. The squirrel is nice.
10. The squirrel is kind.
11. The dog is furry.
12. The dog is lovely.
13. The dog is cute.
14. Smart animals are furry.
15. If something is slow then it needs the squirrel.
16. If something needs the squirrel then it is lazy.
17. If something is tired and slow then it is rough.
18. If something is furry and lovely then it is adorable.
19. If something is awful and obese then it is fierce.
20. If something is rough then it is dull.
21. All dull animals are sleepy.
22. If something is furry then it is lovely.
23. All lovely animals are cute.
24. If something is fierce then it is strong.
25. All strong animals are big.
26. If something is adorable then it is small.
27. All small animals are beautiful.
28. All lazy animals are heavy.

Derivation:
29. We are given: The tiger is rough.
30. Instantiating rule 20 for the tiger: if The tiger is rough, then The tiger is dull.
31. Therefore, The tiger is dull.
32. Instantiating rule 21 for the tiger: if The tiger is dull, then The tiger is sleepy.
33. Therefore, The tiger is sleepy.

Conclusion:
The tiger is sleepy.

Final answer: False<|endoftext|>
```

## Window 7962 summary: [{"tokens": 467, "head": "Anne is furry. Anne is smart. Anne is young. Charlie is cold. Charlie is white. ", "tail": "e is nice.  Conclusion: Charlie is nice.  Final answer: True"}, {"tokens": 468, "head": "Anne is furry. Anne is smart. Anne is young. Charlie is cold. Charlie is white. ", "tail": " is nice.  Conclusion: Charlie is nice.  Final answer: False"}, {"tokens": 516, "head": "Anne is furry. Anne is smart. Anne is young. Charlie is cold. Charlie is white. ", "tail": "is green.  Conclusion: Charlie is green.  Final answer: True"}, {"tokens": 517, "head": "Anne is furry. Anne is smart. Anne is young. Charlie is cold. Charlie is white. ", "tail": "s green.  Conclusion: Charlie is green.  Final answer: False"}, {"tokens": 547, "head": "Anne is furry. Anne is smart. Anne is young. Charlie is cold. Charlie is white. ", "tail": "is smart.  Conclusion: Charlie is smart.  Final answer: True"}, {"tokens": 548, "head": "Anne is furry. Anne is smart. Anne is young. Charlie is cold. Charlie is white. ", "tail": "s smart.  Conclusion: Charlie is smart.  Final answer: False"}, {"tokens": 344, "head": "Bob is kind. Charlie is cold. Charlie is kind. Charlie is nice. Charlie is quiet", "tail": "is white.  Conclusion: Charlie is white.  Final answer: True"}, {"tokens": 374, "head": "Bob is kind. Charlie is cold. Charlie is kind. Charlie is nice. Charlie is quiet", "tail": "s rough.  Conclusion: Charlie is rough.  Final answer: False"}, {"tokens": 310, "head": "Anne is furry. Anne is young. Charlie is blue. Charlie is nice. Charlie is young", "tail": "ve is furry.  Conclusion: Dave is furry.  Final answer: True"}]

## Window 8060 summary: [{"tokens": 650, "head": "The crocodile is lazy. The crocodile is dull. The crocodile is sleepy. The croco", "tail": "is small.  Conclusion: The dog is small.  Final answer: True"}, {"tokens": 651, "head": "The crocodile is lazy. The crocodile is dull. The crocodile is sleepy. The croco", "tail": "s small.  Conclusion: The dog is small.  Final answer: False"}, {"tokens": 638, "head": "The crocodile is lazy. The crocodile is dull. The crocodile is sleepy. The croco", "tail": " big.  Conclusion: The crocodile is big.  Final answer: True"}, {"tokens": 639, "head": "The crocodile is lazy. The crocodile is dull. The crocodile is sleepy. The croco", "tail": "big.  Conclusion: The crocodile is big.  Final answer: False"}, {"tokens": 607, "head": "The tiger is tired. The tiger is dull. The tiger is lazy. The tiger sees the rab", "tail": "le.  Conclusion: The rabbit is adorable.  Final answer: True"}, {"tokens": 608, "head": "The tiger is tired. The tiger is dull. The tiger is lazy. The tiger sees the rab", "tail": "e.  Conclusion: The rabbit is adorable.  Final answer: False"}]

## Window 8222 summary: [{"tokens": 412, "head": "Bob is big. Bob is cold. Bob is green. Charlie is big. Charlie is cold. Charlie ", "tail": " is green.  Conclusion: Harry is green.  Final answer: False"}, {"tokens": 460, "head": "Bob is big. Bob is cold. Bob is green. Charlie is big. Charlie is cold. Charlie ", "tail": "rry is kind.  Conclusion: Harry is kind.  Final answer: True"}, {"tokens": 472, "head": "Bob is big. Bob is cold. Bob is green. Charlie is big. Charlie is cold. Charlie ", "tail": " Bob is kind.  Conclusion: Bob is kind.  Final answer: False"}, {"tokens": 277, "head": "The bear eats the mouse. The cat is nice. The mouse sees the bear. The tiger doe", "tail": ".  Conclusion: The bear eats the mouse.  Final answer: False"}, {"tokens": 306, "head": "The bear eats the mouse. The cat is nice. The mouse sees the bear. The tiger doe", "tail": "is round.  Conclusion: The cat is round.  Final answer: True"}, {"tokens": 307, "head": "The bear eats the mouse. The cat is nice. The mouse sees the bear. The tiger doe", "tail": "s round.  Conclusion: The cat is round.  Final answer: False"}, {"tokens": 319, "head": "The bear eats the mouse. The cat is nice. The mouse sees the bear. The tiger doe", "tail": "r.  Conclusion: The cat needs the tiger.  Final answer: True"}, {"tokens": 321, "head": "The bear eats the mouse. The cat is nice. The mouse sees the bear. The tiger doe", "tail": ".  Conclusion: The cat needs the tiger.  Final answer: False"}, {"tokens": 378, "head": "The bear eats the mouse. The cat is nice. The mouse sees the bear. The tiger doe", "tail": "at.  Conclusion: The tiger eats the cat.  Final answer: True"}, {"tokens": 380, "head": "The bear eats the mouse. The cat is nice. The mouse sees the bear. The tiger doe", "tail": "t.  Conclusion: The tiger eats the cat.  Final answer: False"}, {"tokens": 362, "head": "The cow is big. The cow is nice. The cow needs the mouse. The mouse eats the cow", "tail": "cow is big.  Conclusion: The cow is big.  Final answer: True"}]

## Window 9449 summary: [{"tokens": 931, "head": "The crocodile is dull. The crocodile is lazy. The crocodile is reckless. The cro", "tail": ".  Conclusion: The crocodile is boring.  Final answer: False"}, {"tokens": 762, "head": "The bald eagle is lazy. The bald eagle is dull. The bald eagle is sleepy. The ba", "tail": "kind.  Conclusion: The squirrel is kind.  Final answer: True"}, {"tokens": 763, "head": "The bald eagle is lazy. The bald eagle is dull. The bald eagle is sleepy. The ba", "tail": "ind.  Conclusion: The squirrel is kind.  Final answer: False"}, {"tokens": 886, "head": "The bald eagle is lazy. The bald eagle is dull. The bald eagle is sleepy. The ba", "tail": "l.  Conclusion: The bald eagle is awful.  Final answer: True"}, {"tokens": 701, "head": "Alan is high. Alan is huge. Alan is strong. Fiona is tiny. Fiona is thin. Erin i", "tail": "an is heavy.  Conclusion: Alan is heavy.  Final answer: True"}]

## Window 13686 summary: [{"tokens": 514, "head": "Each vumpus is a dumpus. Each yumpus is not liquid. Zumpuses are bitter. Each wu", "tail": " is liquid.  Conclusion: Alex is liquid.  Final answer: True"}, {"tokens": 377, "head": "Each rompus is a yumpus. Grimpuses are not discordant. Each rompus is a lempus. ", "tail": "lly is fast.  Conclusion: Sally is fast.  Final answer: True"}, {"tokens": 301, "head": "Every zumpus is a dumpus. Dumpuses are dull. Rompuses are not slow. Every lempus", "tail": "ot slow.  Conclusion: Polly is not slow.  Final answer: True"}, {"tokens": 318, "head": "Every shumpus is hot. Every tumpus is a numpus. Tumpuses are wooden. Lempuses ar", "tail": "  Conclusion: Stella is not discordant.  Final answer: False"}, {"tokens": 460, "head": "Dumpuses are not rainy. Rompuses are earthy. Each grimpus is a jompus. Brimpuses", "tail": "m is earthy.  Conclusion: Sam is earthy.  Final answer: True"}, {"tokens": 515, "head": "Shumpuses are small. Lorpuses are zumpuses. Every impus is bright. Gorpuses are ", "tail": "t small.  Conclusion: Rex is not small.  Final answer: False"}, {"tokens": 444, "head": "Lorpuses are earthy. Wumpuses are brimpuses. Every dumpus is a grimpus. Wumpuses", "tail": "orange.  Conclusion: Fae is not orange.  Final answer: False"}, {"tokens": 309, "head": "Tumpuses are not opaque. Lempuses are vumpuses. Every tumpus is an impus. Vumpus", "tail": "opaque.  Conclusion: Max is not opaque.  Final answer: False"}, {"tokens": 307, "head": "Each shumpus is temperate. Each gorpus is a jompus. Gorpuses are not moderate. E", "tail": "rate.  Conclusion: Wren is not moderate.  Final answer: True"}, {"tokens": 376, "head": "Every yumpus is spicy. Dumpuses are small. Each yumpus is a tumpus. Vumpuses are", "tail": "y is spicy.  Conclusion: Polly is spicy.  Final answer: True"}]

## Window 14192 summary: [{"tokens": 621, "head": "The lion is lazy. The lion is reckless. The lion is tired. The lion attacks the ", "tail": "s kind.  Conclusion: The mouse is kind.  Final answer: False"}, {"tokens": 725, "head": "The lion is lazy. The lion is reckless. The lion is tired. The lion attacks the ", "tail": "on is big.  Conclusion: The lion is big.  Final answer: True"}, {"tokens": 726, "head": "The lion is lazy. The lion is reckless. The lion is tired. The lion attacks the ", "tail": "n is big.  Conclusion: The lion is big.  Final answer: False"}, {"tokens": 816, "head": "The lion is lazy. The lion is reckless. The lion is tired. The lion attacks the ", "tail": "  Conclusion: The crocodile is reckless.  Final answer: True"}, {"tokens": 817, "head": "The lion is lazy. The lion is reckless. The lion is tired. The lion attacks the ", "tail": " Conclusion: The crocodile is reckless.  Final answer: False"}]

## Window 17048 summary: [{"tokens": 731, "head": "Harry is high. Harry is big. Harry is heavy. Anne is short. Anne is small. Erin ", "tail": "ob is short.  Conclusion: Bob is short.  Final answer: False"}, {"tokens": 701, "head": "Bob is high. Bob is strong. Bob is big. Erin is short. Erin is tiny. Charlie is ", "tail": ", Bob is huge.  Conclusion: Bob is huge.  Final answer: True"}, {"tokens": 702, "head": "Bob is high. Bob is strong. Bob is big. Erin is short. Erin is tiny. Charlie is ", "tail": " Bob is huge.  Conclusion: Bob is huge.  Final answer: False"}, {"tokens": 731, "head": "Bob is high. Bob is strong. Bob is big. Erin is short. Erin is tiny. Charlie is ", "tail": "in is rough.  Conclusion: Erin is rough.  Final answer: True"}, {"tokens": 732, "head": "Bob is high. Bob is strong. Bob is big. Erin is short. Erin is tiny. Charlie is ", "tail": "n is rough.  Conclusion: Erin is rough.  Final answer: False"}]

## Window 23706 summary: [{"tokens": 903, "head": "The wolf is reckless. The wolf is dull. The wolf is slow. The wolf likes the dog", "tail": "sleepy.  Conclusion: The wolf is sleepy.  Final answer: True"}, {"tokens": 904, "head": "The wolf is reckless. The wolf is dull. The wolf is slow. The wolf likes the dog", "tail": "leepy.  Conclusion: The wolf is sleepy.  Final answer: False"}, {"tokens": 701, "head": "Alan is big. Alan is heavy. Alan is strong. Dave is little. Dave is tiny. Gary i", "tail": "Alan is huge.  Conclusion: Alan is huge.  Final answer: True"}, {"tokens": 702, "head": "Alan is big. Alan is heavy. Alan is strong. Dave is little. Dave is tiny. Gary i", "tail": "lan is huge.  Conclusion: Alan is huge.  Final answer: False"}, {"tokens": 730, "head": "Alan is big. Alan is heavy. Alan is strong. Dave is little. Dave is tiny. Gary i", "tail": "Dave is dull.  Conclusion: Dave is dull.  Final answer: True"}]

## Window 23924 summary: [{"tokens": 622, "head": "Anne is strong. Anne is huge. Anne is big. Dave is tiny. Dave is small. Charlie ", "tail": " Dave is bad.  Conclusion: Dave is bad.  Final answer: False"}, {"tokens": 621, "head": "Anne is strong. Anne is huge. Anne is big. Dave is tiny. Dave is small. Charlie ", "tail": "e is huge.  Conclusion: Charlie is huge.  Final answer: True"}, {"tokens": 622, "head": "Anne is strong. Anne is huge. Anne is big. Dave is tiny. Dave is small. Charlie ", "tail": " is huge.  Conclusion: Charlie is huge.  Final answer: False"}, {"tokens": 622, "head": "Anne is strong. Anne is huge. Anne is big. Dave is tiny. Dave is small. Charlie ", "tail": "a is small.  Conclusion: Fiona is small.  Final answer: True"}, {"tokens": 623, "head": "Anne is strong. Anne is huge. Anne is big. Dave is tiny. Dave is small. Charlie ", "tail": " is small.  Conclusion: Fiona is small.  Final answer: False"}, {"tokens": 616, "head": "The leopard is tired. The leopard is lazy. The leopard is sleepy. The leopard ch", "tail": "round.  Conclusion: The rabbit is round.  Final answer: True"}]

## Window 27439 summary: [{"tokens": 621, "head": "Anne is high. Anne is huge. Anne is strong. Dave is tiny. Dave is small. Charlie", "tail": ", Dave is bad.  Conclusion: Dave is bad.  Final answer: True"}, {"tokens": 622, "head": "Anne is high. Anne is huge. Anne is strong. Dave is tiny. Dave is small. Charlie", "tail": " Dave is bad.  Conclusion: Dave is bad.  Final answer: False"}, {"tokens": 621, "head": "Anne is high. Anne is huge. Anne is strong. Dave is tiny. Dave is small. Charlie", "tail": "e is huge.  Conclusion: Charlie is huge.  Final answer: True"}, {"tokens": 622, "head": "Anne is high. Anne is huge. Anne is strong. Dave is tiny. Dave is small. Charlie", "tail": " is huge.  Conclusion: Charlie is huge.  Final answer: False"}, {"tokens": 622, "head": "Anne is high. Anne is huge. Anne is strong. Dave is tiny. Dave is small. Charlie", "tail": "in is small.  Conclusion: Erin is small.  Final answer: True"}, {"tokens": 623, "head": "Anne is high. Anne is huge. Anne is strong. Dave is tiny. Dave is small. Charlie", "tail": "n is small.  Conclusion: Erin is small.  Final answer: False"}]

## Window 27556 summary: [{"tokens": 290, "head": "Each tumpus is a gorpus. Every lorpus is not slow. Every yumpus is a tumpus. Eac", "tail": "lly is slow.  Conclusion: Polly is slow.  Final answer: True"}, {"tokens": 362, "head": "Each tumpus is a wumpus. Tumpuses are grimpuses. Each impus is not rainy. Each d", "tail": "n is rainy.  Conclusion: Wren is rainy.  Final answer: False"}, {"tokens": 360, "head": "Each grimpus is not amenable. Lorpuses are floral. Impuses are gorpuses. Each gr", "tail": ". Fae is fast.  Conclusion: Fae is fast.  Final answer: True"}, {"tokens": 428, "head": "Numpuses are lorpuses. Each zumpus is a tumpus. Every jompus is a brimpus. Every", "tail": "olly is hot.  Conclusion: Polly is hot.  Final answer: False"}, {"tokens": 357, "head": "Each yumpus is luminous. Jompuses are windy. Sterpuses are yumpuses. Vumpuses ar", "tail": "s bitter.  Conclusion: Sally is bitter.  Final answer: False"}, {"tokens": 212, "head": "Every grimpus is opaque. Lempuses are fruity. Grimpuses are lorpuses. Wumpuses a", "tail": " is opaque.  Conclusion: Rex is opaque.  Final answer: False"}, {"tokens": 434, "head": "Impuses are brimpuses. Every impus is feisty. Tumpuses are impuses. Each vumpus ", "tail": " is liquid.  Conclusion: Wren is liquid.  Final answer: True"}, {"tokens": 429, "head": "Each shumpus is a wumpus. Wumpuses are numpuses. Every wumpus is a zumpus. Shump", "tail": "t sweet.  Conclusion: Alex is not sweet.  Final answer: True"}, {"tokens": 413, "head": "Dumpuses are shumpuses. Every gorpus is a lorpus. Each vumpus is fast. Jompuses ", "tail": "x is windy.  Conclusion: Alex is windy.  Final answer: False"}, {"tokens": 280, "head": "Jompuses are tumpuses. Lempuses are not liquid. Each tumpus is a lorpus. Yumpuse", "tail": " is liquid.  Conclusion: Sam is liquid.  Final answer: False"}, {"tokens": 287, "head": "Every jompus is wooden. Numpuses are sweet. Each rompus is not rainy. Tumpuses a", "tail": "en is happy.  Conclusion: Wren is happy.  Final answer: True"}, {"tokens": 210, "head": "Every numpus is not aggressive. Gorpuses are red. Numpuses are impuses. Grimpuse", "tail": "is not red.  Conclusion: Rex is not red.  Final answer: True"}]

## Window 27557 summary: [{"tokens": 358, "head": "Anne is nice. Anne is rough. Anne is white. Charlie is big. Charlie is green. Ch", "tail": "e is nice.  Conclusion: Charlie is nice.  Final answer: True"}, {"tokens": 338, "head": "Anne is nice. Anne is rough. Anne is white. Charlie is big. Charlie is green. Ch", "tail": "n is smart.  Conclusion: Erin is smart.  Final answer: False"}, {"tokens": 367, "head": "Anne is nice. Anne is rough. Anne is white. Charlie is big. Charlie is green. Ch", "tail": "is white.  Conclusion: Charlie is white.  Final answer: True"}, {"tokens": 368, "head": "Anne is nice. Anne is rough. Anne is white. Charlie is big. Charlie is green. Ch", "tail": "s white.  Conclusion: Charlie is white.  Final answer: False"}, {"tokens": 398, "head": "Anne is nice. Anne is rough. Anne is white. Charlie is big. Charlie is green. Ch", "tail": "is smart.  Conclusion: Charlie is smart.  Final answer: True"}, {"tokens": 399, "head": "Anne is nice. Anne is rough. Anne is white. Charlie is big. Charlie is green. Ch", "tail": "s smart.  Conclusion: Charlie is smart.  Final answer: False"}, {"tokens": 307, "head": "Anne is not cold. Anne is nice. Anne is rough. Bob is cold. Bob is quiet. Gary i", "tail": "ry is rough.  Conclusion: Gary is rough.  Final answer: True"}, {"tokens": 308, "head": "Anne is not cold. Anne is nice. Anne is rough. Bob is cold. Bob is quiet. Gary i", "tail": "e is rough.  Conclusion: Anne is rough.  Final answer: False"}, {"tokens": 338, "head": "Anne is not cold. Anne is nice. Anne is rough. Bob is cold. Bob is quiet. Gary i", "tail": "Bob is furry.  Conclusion: Bob is furry.  Final answer: True"}, {"tokens": 339, "head": "Anne is not cold. Anne is nice. Anne is rough. Bob is cold. Bob is quiet. Gary i", "tail": "ob is furry.  Conclusion: Bob is furry.  Final answer: False"}, {"tokens": 387, "head": "Anne is not cold. Anne is nice. Anne is rough. Bob is cold. Bob is quiet. Gary i", "tail": "Bob is rough.  Conclusion: Bob is rough.  Final answer: True"}, {"tokens": 165, "head": "Bob is quiet. Bob is smart. Fiona is cold. Fiona is furry. Fiona is quiet. Fiona", "tail": " is quiet.  Conclusion: Fiona is quiet.  Final answer: False"}]

## Window 27973 summary: [{"tokens": 387, "head": "Yumpuses are gorpuses. Sterpuses are lempuses. Every sterpus is a zumpus. Every ", "tail": "ight.  Conclusion: Stella is not bright.  Final answer: True"}, {"tokens": 533, "head": "Every grimpus is a zumpus. Zumpuses are not bright. Every numpus is a grimpus. E", "tail": "bright.  Conclusion: Wren is not bright.  Final answer: True"}, {"tokens": 440, "head": "Yumpuses are lempuses. Each yumpus is dull. Yumpuses are wumpuses. Each wumpus i", "tail": " Fae is dull.  Conclusion: Fae is dull.  Final answer: False"}, {"tokens": 385, "head": "Brimpuses are metallic. Each jompus is small. Every lempus is a vumpus. Every jo", "tail": "Rex is small.  Conclusion: Rex is small.  Final answer: True"}, {"tokens": 457, "head": "Lempuses are opaque. Every jompus is a tumpus. Every jompus is aggressive. Tumpu", "tail": "paque.  Conclusion: Polly is not opaque.  Final answer: True"}, {"tokens": 371, "head": "Shumpuses are small. Brimpuses are lorpuses. Every brimpus is amenable. Dumpuses", "tail": "is earthy.  Conclusion: Alex is earthy.  Final answer: False"}, {"tokens": 308, "head": "Sterpuses are spicy. Each tumpus is dull. Each yumpus is not cold. Numpuses are ", "tail": "lly is dull.  Conclusion: Polly is dull.  Final answer: True"}, {"tokens": 299, "head": "Each sterpus is dull. Tumpuses are small. Lempuses are tumpuses. Rompuses are no", "tail": "s opaque.  Conclusion: Polly is opaque.  Final answer: False"}, {"tokens": 314, "head": "Every sterpus is a lorpus. Jompuses are brimpuses. Brimpuses are numpuses. Every", "tail": "ot windy.  Conclusion: Fae is not windy.  Final answer: True"}, {"tokens": 454, "head": "Every rompus is not windy. Rompuses are brimpuses. Every shumpus is fruity. Each", "tail": ". Fae is dull.  Conclusion: Fae is dull.  Final answer: True"}]

## Window 27984 summary: [{"tokens": 500, "head": "The bear is cold. The bear is red. The bear likes the mouse. The bear likes the ", "tail": "s young.  Conclusion: The bear is young.  Final answer: True"}, {"tokens": 501, "head": "The bear is cold. The bear is red. The bear likes the mouse. The bear likes the ", "tail": " young.  Conclusion: The bear is young.  Final answer: False"}, {"tokens": 167, "head": "Harry is furry. Harry is nice. Harry is white. All white, young things are kind.", "tail": "y is young.  Conclusion: Harry is young.  Final answer: True"}, {"tokens": 168, "head": "Harry is furry. Harry is nice. Harry is white. All white, young things are kind.", "tail": " is young.  Conclusion: Harry is young.  Final answer: False"}, {"tokens": 226, "head": "Harry is furry. Harry is nice. Harry is white. All white, young things are kind.", "tail": "rry is kind.  Conclusion: Harry is kind.  Final answer: True"}, {"tokens": 227, "head": "Harry is furry. Harry is nice. Harry is white. All white, young things are kind.", "tail": "ry is kind.  Conclusion: Harry is kind.  Final answer: False"}, {"tokens": 274, "head": "Charlie is not big. Charlie is green. Charlie is smart. Gary is big. Gary is gre", "tail": "t young.  Conclusion: Gary is not young.  Final answer: True"}, {"tokens": 272, "head": "Charlie is not big. Charlie is green. Charlie is smart. Gary is big. Gary is gre", "tail": "s green.  Conclusion: Charlie is green.  Final answer: False"}, {"tokens": 302, "head": "Charlie is not big. Charlie is green. Charlie is smart. Gary is big. Gary is gre", "tail": "is round.  Conclusion: Charlie is round.  Final answer: True"}, {"tokens": 303, "head": "Charlie is not big. Charlie is green. Charlie is smart. Gary is big. Gary is gre", "tail": "s round.  Conclusion: Charlie is round.  Final answer: False"}, {"tokens": 333, "head": "Charlie is not big. Charlie is green. Charlie is smart. Gary is big. Gary is gre", "tail": "is rough.  Conclusion: Charlie is rough.  Final answer: True"}, {"tokens": 334, "head": "Charlie is not big. Charlie is green. Charlie is smart. Gary is big. Gary is gre", "tail": "s rough.  Conclusion: Charlie is rough.  Final answer: False"}, {"tokens": 299, "head": "Bob is cold. Bob is red. Bob is rough. Bob is round. Bob is not smart. Erin is c", "tail": "Bob is round.  Conclusion: Bob is round.  Final answer: True"}, {"tokens": 126, "head": "Bob is big. Bob is not young. Erin is rough. If someone is white and rough then ", "tail": "ot young.  Conclusion: Bob is not young.  Final answer: True"}]

## Window 32399 summary: [{"tokens": 533, "head": "The lion chases the mouse. The lion is big. The lion is green. The lion is kind.", "tail": " green.  Conclusion: The mouse is green.  Final answer: True"}, {"tokens": 542, "head": "The lion chases the mouse. The lion is big. The lion is green. The lion is kind.", "tail": "  Conclusion: The lion chases the lion.  Final answer: False"}, {"tokens": 576, "head": "The lion chases the mouse. The lion is big. The lion is green. The lion is kind.", "tail": " Conclusion: The mouse chases the mouse.  Final answer: True"}, {"tokens": 577, "head": "The lion chases the mouse. The lion is big. The lion is green. The lion is kind.", "tail": "Conclusion: The mouse chases the mouse.  Final answer: False"}, {"tokens": 313, "head": "The bald eagle chases the mouse. The mouse chases the bald eagle. The mouse is n", "tail": "usion: The bald eagle chases the mouse.  Final answer: False"}, {"tokens": 359, "head": "The bald eagle chases the mouse. The mouse chases the bald eagle. The mouse is n", "tail": " The bald eagle does not need the mouse.  Final answer: True"}, {"tokens": 357, "head": "The bald eagle chases the mouse. The mouse chases the bald eagle. The mouse is n", "tail": "The bald eagle does not need the mouse.  Final answer: False"}, {"tokens": 364, "head": "The bald eagle chases the mouse. The mouse chases the bald eagle. The mouse is n", "tail": "ce.  Conclusion: The bald eagle is nice.  Final answer: True"}, {"tokens": 365, "head": "The bald eagle chases the mouse. The mouse chases the bald eagle. The mouse is n", "tail": "e.  Conclusion: The bald eagle is nice.  Final answer: False"}, {"tokens": 106, "head": "Dave is big. Dave is quiet. Gary is blue. Red, quiet things are blue. If somethi", "tail": "ary is blue.  Conclusion: Gary is blue.  Final answer: False"}]

## Window 32427 summary: [{"tokens": 506, "head": "Charlie is huge. Charlie is strong. Charlie is big. Fiona is small. Fiona is thi", "tail": "re, Bob is bad.  Conclusion: Bob is bad.  Final answer: True"}, {"tokens": 507, "head": "Charlie is huge. Charlie is strong. Charlie is big. Fiona is small. Fiona is thi", "tail": "e, Bob is bad.  Conclusion: Bob is bad.  Final answer: False"}, {"tokens": 478, "head": "Charlie is big. Charlie is huge. Charlie is high. Bob is little. Bob is thin. Al", "tail": "s quiet.  Conclusion: Charlie is quiet.  Final answer: False"}, {"tokens": 506, "head": "Charlie is big. Charlie is huge. Charlie is high. Bob is little. Bob is thin. Al", "tail": "Bob is small.  Conclusion: Bob is small.  Final answer: True"}, {"tokens": 507, "head": "Charlie is big. Charlie is huge. Charlie is high. Bob is little. Bob is thin. Al", "tail": "ob is small.  Conclusion: Bob is small.  Final answer: False"}, {"tokens": 506, "head": "Charlie is big. Charlie is huge. Charlie is high. Bob is little. Bob is thin. Al", "tail": "an is smart.  Conclusion: Alan is smart.  Final answer: True"}, {"tokens": 507, "head": "Charlie is big. Charlie is huge. Charlie is high. Bob is little. Bob is thin. Al", "tail": "n is smart.  Conclusion: Alan is smart.  Final answer: False"}, {"tokens": 506, "head": "Charlie is big. Charlie is huge. Charlie is high. Bob is little. Bob is thin. Al", "tail": "erfect.  Conclusion: Harry is imperfect.  Final answer: True"}]

## Window 33950 summary: [{"tokens": 904, "head": "The wolf is angry. The wolf is reckless. The wolf is boring. The wolf needs the ", "tail": "s tired.  Conclusion: The wolf is tired.  Final answer: True"}, {"tokens": 905, "head": "The wolf is angry. The wolf is reckless. The wolf is boring. The wolf needs the ", "tail": " tired.  Conclusion: The wolf is tired.  Final answer: False"}, {"tokens": 701, "head": "Charlie is heavy. Charlie is big. Charlie is high. Harry is little. Harry is thi", "tail": " strong.  Conclusion: Charlie is strong.  Final answer: True"}, {"tokens": 702, "head": "Charlie is heavy. Charlie is big. Charlie is high. Harry is little. Harry is thi", "tail": "strong.  Conclusion: Charlie is strong.  Final answer: False"}, {"tokens": 730, "head": "Charlie is heavy. Charlie is big. Charlie is high. Harry is little. Harry is thi", "tail": "rry is dull.  Conclusion: Harry is dull.  Final answer: True"}]

## Window 34086 summary: [{"tokens": 391, "head": "Alan is high. Alan is huge. Alan is big. Charlie is little. Charlie is thin. Dav", "tail": "is small.  Conclusion: Charlie is small.  Final answer: True"}, {"tokens": 392, "head": "Alan is high. Alan is huge. Alan is big. Charlie is little. Charlie is thin. Dav", "tail": "s small.  Conclusion: Charlie is small.  Final answer: False"}, {"tokens": 391, "head": "Alan is high. Alan is huge. Alan is big. Charlie is little. Charlie is thin. Dav", "tail": "Dave is kind.  Conclusion: Dave is kind.  Final answer: True"}, {"tokens": 392, "head": "Alan is high. Alan is huge. Alan is big. Charlie is little. Charlie is thin. Dav", "tail": "ave is kind.  Conclusion: Dave is kind.  Final answer: False"}, {"tokens": 391, "head": "Alan is high. Alan is huge. Alan is big. Charlie is little. Charlie is thin. Dav", "tail": "ne is rough.  Conclusion: Anne is rough.  Final answer: True"}, {"tokens": 392, "head": "Alan is high. Alan is huge. Alan is big. Charlie is little. Charlie is thin. Dav", "tail": "e is rough.  Conclusion: Anne is rough.  Final answer: False"}, {"tokens": 365, "head": "Erin is big. Erin is high. Erin is strong. Fiona is short. Fiona is thin. Anne i", "tail": "n is quiet.  Conclusion: Erin is quiet.  Final answer: False"}, {"tokens": 393, "head": "Erin is big. Erin is high. Erin is strong. Fiona is short. Fiona is thin. Anne i", "tail": "a is small.  Conclusion: Fiona is small.  Final answer: True"}, {"tokens": 394, "head": "Erin is big. Erin is high. Erin is strong. Fiona is short. Fiona is thin. Anne i", "tail": " is small.  Conclusion: Fiona is small.  Final answer: False"}, {"tokens": 392, "head": "Erin is big. Erin is high. Erin is strong. Fiona is short. Fiona is thin. Anne i", "tail": "ne is smart.  Conclusion: Anne is smart.  Final answer: True"}]

## Window 36271 summary: [{"tokens": 329, "head": "Anne is blue. Anne is nice. Anne is red. Anne is rough. Anne is round. Anne is w", "tail": "Anne is nice.  Conclusion: Anne is nice.  Final answer: True"}, {"tokens": 331, "head": "Anne is blue. Anne is nice. Anne is red. Anne is rough. Anne is round. Anne is w", "tail": " Erin is red.  Conclusion: Erin is red.  Final answer: False"}, {"tokens": 360, "head": "Anne is blue. Anne is nice. Anne is red. Anne is rough. Anne is round. Anne is w", "tail": "Bob is young.  Conclusion: Bob is young.  Final answer: True"}, {"tokens": 361, "head": "Anne is blue. Anne is nice. Anne is red. Anne is rough. Anne is round. Anne is w", "tail": "ob is young.  Conclusion: Bob is young.  Final answer: False"}, {"tokens": 188, "head": "Anne is rough. Anne is young. Erin is not young. If something is white and smart", "tail": "ne is rough.  Conclusion: Anne is rough.  Final answer: True"}, {"tokens": 189, "head": "Anne is rough. Anne is young. Erin is not young. If something is white and smart", "tail": "e is rough.  Conclusion: Anne is rough.  Final answer: False"}, {"tokens": 218, "head": "Anne is rough. Anne is young. Erin is not young. If something is white and smart", "tail": "ne is white.  Conclusion: Anne is white.  Final answer: True"}, {"tokens": 219, "head": "Anne is rough. Anne is young. Erin is not young. If something is white and smart", "tail": "e is white.  Conclusion: Anne is white.  Final answer: False"}, {"tokens": 282, "head": "The cat is rough. The cow eats the cat. The cow eats the squirrel. The lion eats", "tail": "is rough.  Conclusion: The cat is rough.  Final answer: True"}, {"tokens": 287, "head": "The cat is rough. The cow eats the cat. The cow eats the squirrel. The lion eats", "tail": "Conclusion: The lion eats the squirrel.  Final answer: False"}, {"tokens": 325, "head": "The cat is rough. The cow eats the cat. The cow eats the squirrel. The lion eats", "tail": "  Conclusion: The squirrel sees the cow.  Final answer: True"}, {"tokens": 327, "head": "The cat is rough. The cow eats the cat. The cow eats the squirrel. The lion eats", "tail": " Conclusion: The squirrel sees the cow.  Final answer: False"}, {"tokens": 93, "head": "Bob is quiet. Charlie is quiet. Erin is young. Harry is nice. All quiet things a", "tail": "Bob is quiet.  Conclusion: Bob is quiet.  Final answer: True"}, {"tokens": 95, "head": "Bob is quiet. Charlie is quiet. Erin is young. Harry is nice. All quiet things a", "tail": "n is young.  Conclusion: Erin is young.  Final answer: False"}, {"tokens": 121, "head": "Bob is quiet. Charlie is quiet. Erin is young. Harry is nice. All quiet things a", "tail": ", Bob is blue.  Conclusion: Bob is blue.  Final answer: True"}, {"tokens": 122, "head": "Bob is quiet. Charlie is quiet. Erin is young. Harry is nice. All quiet things a", "tail": " is blue.  Conclusion: Charlie is blue.  Final answer: False"}, {"tokens": 91, "head": "Bob is cold. All cold people are kind.  True or false: Bob is kind.  Solution: C", "tail": ", Bob is kind.  Conclusion: Bob is kind.  Final answer: True"}, {"tokens": 92, "head": "Bob is cold. All cold people are kind.  True or false: Bob is not kind.  Solutio", "tail": " Bob is kind.  Conclusion: Bob is kind.  Final answer: False"}, {"tokens": 64, "head": "Harry is red. All red people are blue.  True or false: Harry is not red.  Soluti", "tail": "arry is red.  Conclusion: Harry is red.  Final answer: False"}]

## Window 40684 summary: [{"tokens": 391, "head": "Alan is high. Alan is huge. Alan is strong. Dave is thin. Dave is little. Harry ", "tail": "rry is kind.  Conclusion: Harry is kind.  Final answer: True"}, {"tokens": 392, "head": "Alan is high. Alan is huge. Alan is strong. Dave is thin. Dave is little. Harry ", "tail": "ry is kind.  Conclusion: Harry is kind.  Final answer: False"}, {"tokens": 391, "head": "Alan is high. Alan is huge. Alan is strong. Dave is thin. Dave is little. Harry ", "tail": "Gary is dull.  Conclusion: Gary is dull.  Final answer: True"}, {"tokens": 392, "head": "Alan is high. Alan is huge. Alan is strong. Dave is thin. Dave is little. Harry ", "tail": "ary is dull.  Conclusion: Gary is dull.  Final answer: False"}, {"tokens": 471, "head": "The lion is rough. The lion is dull. The lion is sleepy. The lion sees the rabbi", "tail": "vely.  Conclusion: The rabbit is lovely.  Final answer: True"}, {"tokens": 472, "head": "The lion is rough. The lion is dull. The lion is sleepy. The lion sees the rabbi", "tail": "ely.  Conclusion: The rabbit is lovely.  Final answer: False"}, {"tokens": 436, "head": "The lion is rough. The lion is dull. The lion is sleepy. The lion sees the rabbi", "tail": " is lazy.  Conclusion: The lion is lazy.  Final answer: True"}, {"tokens": 437, "head": "The lion is rough. The lion is dull. The lion is sleepy. The lion sees the rabbi", "tail": "is lazy.  Conclusion: The lion is lazy.  Final answer: False"}, {"tokens": 504, "head": "The lion is rough. The lion is dull. The lion is sleepy. The lion sees the rabbi", "tail": " heavy.  Conclusion: The tiger is heavy.  Final answer: True"}]

## Window 41239 summary: [{"tokens": 480, "head": "The tiger is slow. The tiger is sleepy. The tiger is rough. The tiger visits the", "tail": "ely.  Conclusion: The rabbit is lovely.  Final answer: False"}, {"tokens": 444, "head": "The tiger is slow. The tiger is sleepy. The tiger is rough. The tiger visits the", "tail": "is dull.  Conclusion: The tiger is dull.  Final answer: True"}, {"tokens": 445, "head": "The tiger is slow. The tiger is sleepy. The tiger is rough. The tiger visits the", "tail": "s dull.  Conclusion: The tiger is dull.  Final answer: False"}, {"tokens": 527, "head": "The tiger is slow. The tiger is sleepy. The tiger is rough. The tiger visits the", "tail": ".  Conclusion: The bald eagle is strong.  Final answer: True"}, {"tokens": 528, "head": "The tiger is slow. The tiger is sleepy. The tiger is rough. The tiger visits the", "tail": "  Conclusion: The bald eagle is strong.  Final answer: False"}, {"tokens": 512, "head": "The tiger is slow. The tiger is sleepy. The tiger is rough. The tiger visits the", "tail": " small.  Conclusion: The mouse is small.  Final answer: True"}, {"tokens": 513, "head": "The tiger is slow. The tiger is sleepy. The tiger is rough. The tiger visits the", "tail": "small.  Conclusion: The mouse is small.  Final answer: False"}, {"tokens": 482, "head": "The tiger is slow. The tiger is sleepy. The tiger is rough. The tiger visits the", "tail": "is lazy.  Conclusion: The tiger is lazy.  Final answer: True"}]

## Window 41929 summary: [{"tokens": 289, "head": "Anne is cold. Anne is quiet. Anne is smart. Bob is blue. Bob is not cold. Bob is", "tail": "Bob is smart.  Conclusion: Bob is smart.  Final answer: True"}, {"tokens": 290, "head": "Anne is cold. Anne is quiet. Anne is smart. Bob is blue. Bob is not cold. Bob is", "tail": "ob is smart.  Conclusion: Bob is smart.  Final answer: False"}, {"tokens": 320, "head": "Anne is cold. Anne is quiet. Anne is smart. Bob is blue. Bob is not cold. Bob is", "tail": "y is rough.  Conclusion: Harry is rough.  Final answer: True"}, {"tokens": 321, "head": "Anne is cold. Anne is quiet. Anne is smart. Bob is blue. Bob is not cold. Bob is", "tail": " is rough.  Conclusion: Harry is rough.  Final answer: False"}, {"tokens": 351, "head": "Anne is cold. Anne is quiet. Anne is smart. Bob is blue. Bob is not cold. Bob is", "tail": "y is young.  Conclusion: Harry is young.  Final answer: True"}, {"tokens": 352, "head": "Anne is cold. Anne is quiet. Anne is smart. Bob is blue. Bob is not cold. Bob is", "tail": " is young.  Conclusion: Harry is young.  Final answer: False"}, {"tokens": 382, "head": "Anne is cold. Anne is quiet. Anne is smart. Bob is blue. Bob is not cold. Bob is", "tail": "y is round.  Conclusion: Harry is round.  Final answer: True"}, {"tokens": 383, "head": "Anne is cold. Anne is quiet. Anne is smart. Bob is blue. Bob is not cold. Bob is", "tail": " is round.  Conclusion: Harry is round.  Final answer: False"}, {"tokens": 431, "head": "Anne is cold. Anne is quiet. Anne is smart. Bob is blue. Bob is not cold. Bob is", "tail": "y is smart.  Conclusion: Harry is smart.  Final answer: True"}, {"tokens": 432, "head": "Anne is cold. Anne is quiet. Anne is smart. Bob is blue. Bob is not cold. Bob is", "tail": " is smart.  Conclusion: Harry is smart.  Final answer: False"}, {"tokens": 443, "head": "Anne is cold. Anne is quiet. Anne is smart. Bob is blue. Bob is not cold. Bob is", "tail": "ot cold.  Conclusion: Harry is not cold.  Final answer: True"}]

## Window 49398 summary: [{"tokens": 622, "head": "Erin is high. Erin is big. Erin is strong. Anne is thin. Anne is short. Gary is ", "tail": "is short.  Conclusion: Charlie is short.  Final answer: True"}, {"tokens": 623, "head": "Erin is high. Erin is big. Erin is strong. Anne is thin. Anne is short. Gary is ", "tail": "s short.  Conclusion: Charlie is short.  Final answer: False"}, {"tokens": 592, "head": "Gary is high. Gary is strong. Gary is huge. Charlie is small. Charlie is thin. D", "tail": ", Gary is big.  Conclusion: Gary is big.  Final answer: True"}, {"tokens": 593, "head": "Gary is high. Gary is strong. Gary is huge. Charlie is small. Charlie is thin. D", "tail": " Gary is big.  Conclusion: Gary is big.  Final answer: False"}, {"tokens": 621, "head": "Gary is high. Gary is strong. Gary is huge. Charlie is small. Charlie is thin. D", "tail": "lie is sad.  Conclusion: Charlie is sad.  Final answer: True"}, {"tokens": 622, "head": "Gary is high. Gary is strong. Gary is huge. Charlie is small. Charlie is thin. D", "tail": "ie is sad.  Conclusion: Charlie is sad.  Final answer: False"}]

## Window 51583 summary: [{"tokens": 766, "head": "The crocodile is lazy. The crocodile is rough. The crocodile is dull. The crocod", "tail": "is quiet.  Conclusion: The dog is quiet.  Final answer: True"}, {"tokens": 767, "head": "The crocodile is lazy. The crocodile is rough. The crocodile is dull. The crocod", "tail": "s quiet.  Conclusion: The dog is quiet.  Final answer: False"}, {"tokens": 890, "head": "The crocodile is lazy. The crocodile is rough. The crocodile is dull. The crocod", "tail": " big.  Conclusion: The crocodile is big.  Final answer: True"}, {"tokens": 891, "head": "The crocodile is lazy. The crocodile is rough. The crocodile is dull. The crocod", "tail": "big.  Conclusion: The crocodile is big.  Final answer: False"}, {"tokens": 754, "head": "The tiger is slow. The tiger is dull. The tiger is sleepy. The tiger sees the ca", "tail": "is quiet.  Conclusion: The cat is quiet.  Final answer: True"}]

## Window 55111 summary: [{"tokens": 493, "head": "The crocodile is slow. The crocodile is sleepy. The crocodile is lazy. The croco", "tail": "gh.  Conclusion: The crocodile is rough.  Final answer: True"}, {"tokens": 494, "head": "The crocodile is slow. The crocodile is sleepy. The crocodile is lazy. The croco", "tail": "h.  Conclusion: The crocodile is rough.  Final answer: False"}, {"tokens": 473, "head": "The wolf is slow. The wolf is dull. The wolf is rough. The wolf likes the dog. T", "tail": " lovely.  Conclusion: The dog is lovely.  Final answer: True"}, {"tokens": 474, "head": "The wolf is slow. The wolf is dull. The wolf is rough. The wolf likes the dog. T", "tail": "lovely.  Conclusion: The dog is lovely.  Final answer: False"}, {"tokens": 438, "head": "The wolf is slow. The wolf is dull. The wolf is rough. The wolf likes the dog. T", "tail": "sleepy.  Conclusion: The wolf is sleepy.  Final answer: True"}, {"tokens": 439, "head": "The wolf is slow. The wolf is dull. The wolf is rough. The wolf likes the dog. T", "tail": "leepy.  Conclusion: The wolf is sleepy.  Final answer: False"}, {"tokens": 506, "head": "The wolf is slow. The wolf is dull. The wolf is rough. The wolf likes the dog. T", "tail": " heavy.  Conclusion: The tiger is heavy.  Final answer: True"}, {"tokens": 507, "head": "The wolf is slow. The wolf is dull. The wolf is rough. The wolf likes the dog. T", "tail": "heavy.  Conclusion: The tiger is heavy.  Final answer: False"}]

## Window 55112 summary: [{"tokens": 621, "head": "Dave is big. Dave is strong. Dave is high. Erin is thin. Erin is small. Charlie ", "tail": " strong.  Conclusion: Charlie is strong.  Final answer: True"}, {"tokens": 622, "head": "Dave is big. Dave is strong. Dave is high. Erin is thin. Erin is small. Charlie ", "tail": "strong.  Conclusion: Charlie is strong.  Final answer: False"}, {"tokens": 621, "head": "Dave is big. Dave is strong. Dave is high. Erin is thin. Erin is small. Charlie ", "tail": "ne is small.  Conclusion: Anne is small.  Final answer: True"}, {"tokens": 622, "head": "Dave is big. Dave is strong. Dave is high. Erin is thin. Erin is small. Charlie ", "tail": "e is small.  Conclusion: Anne is small.  Final answer: False"}, {"tokens": 614, "head": "The wolf is rough. The wolf is slow. The wolf is reckless. The wolf needs the sq", "tail": "und.  Conclusion: The squirrel is round.  Final answer: True"}, {"tokens": 615, "head": "The wolf is rough. The wolf is slow. The wolf is reckless. The wolf needs the sq", "tail": "nd.  Conclusion: The squirrel is round.  Final answer: False"}]
