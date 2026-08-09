# Decoded-batch audit examples (document-preserving docpack loader)

## Window 0 (10 documents, 39 pad tokens)
```
The bear chases the tiger. The cat likes the tiger. The dog likes the cat. The tiger needs the cat. If something is blue then it likes the tiger. If something likes the cat then it likes the tiger. If the tiger likes the dog then the dog is red. If something chases the tiger and the tiger needs the cat then the cat needs the bear. If something needs the bear then the bear likes the cat. If something is round then it likes the cat.

True or false: The bear likes the cat.

Solution:
Context:
Constants:
bear = the bear
cat = the cat
dog = the dog
tiger = the tiger

Predicates:
Blue(x) = x is blue
Chases(x,y) = x chases y
Likes(x,y) = x likes y
Needs(x,y) = x needs y
Red(x) = x is red
Round(x) = x is round

Premises:
1. Chases(bear,tiger)
2. Likes(cat,tiger)
3. Likes(dog,cat)
4. Needs(tiger,cat)
5. Ax(Blue(x) -> Likes(x,tiger))
6. Ax(Likes(x,cat) -> Likes(x,tiger))
7. Likes(tiger,dog) -> Red(dog)
8. Ax(Chases(x,tiger) & Needs(tiger,cat) -> Needs(cat,bear))
9. Ax(Needs(x,bear) -> Likes(bear,cat))
10. Ax(Round(x) -> Likes(x,cat))

Derivation:
11. Chases(bear,tiger) ; R,1
12. Needs(tiger,cat) ; R,4
13. Chases(bear,tiger) & Needs(tiger,cat) ; ∧I,11,12
14. Chases(bear,tiger) & Needs(tiger,cat) -> Needs(cat,bear) ; AE,8
15. Needs(cat,bear) ; ->E,14,13
16. Needs(cat,bear) -> Likes(bear,cat) ; AE,9
17. Likes(bear,cat) ; ->E,16,15

Conclusion:
Likes(bear,cat)

Final answer: True<|endoftext|>The bear chases the tiger. The cat likes the tiger. The dog likes the cat. The tiger needs the cat. If something is blue then it likes the tiger. If something likes the cat then it likes the tiger. If the tiger likes the dog then the dog is red. If something chases the tiger and the tiger needs the cat then the cat needs the bear. If something needs the bear then the bear likes the cat. If something is round then it likes the cat.

True or false: The bear does not like the cat.

Solution:
Context:
Constants:
bear = the bear
cat = the cat
dog = the dog
tiger = the tiger

Predicates:
Blue(x) = x is blue
Chases(x,y) = x chases y
Likes(x,y) = x likes y
Needs(x,y) = x needs y
Red(x) = x is red
Round(x) = x is round

Premises:
1. Chases(bear,tiger)
2. Likes(cat,tiger)
3. Likes(dog,cat)
4. Needs(tiger,cat)
5. Ax(Blue(x) -> Likes(x,tiger))
6. Ax(Likes(x,cat) -> Likes(x,tiger))
7. Likes(tiger,dog) -> Red(dog)
8. Ax(Chases(x,tiger) & Needs(tiger,cat) -> Needs(cat,bear))
9. Ax(Needs(x,bear) -> Likes(bear,cat))
10. Ax(Round(x) -> Likes(x,cat))

Derivation:
11. Chases(bear,tiger) ; R,1
12. Needs(tiger,cat) ; R,4
13. Chases(bear,tiger) & Needs(tiger,cat) ; ∧I,11,12
14. Chases(bear,tiger) & Needs(tiger,cat) -> Needs(cat,bear) ; AE,8
15. Needs(cat,bear) ; ->E,14,13
16. Needs(cat,bear) -> Likes(bear,cat) ; AE,9
17. Likes(bear,cat) ; ->E,16,15

Conclusion:
Likes(bear,cat)

Final answer: False<|endoftext|>The bear chases the tiger. The cat likes the tiger. The dog likes the cat. The tiger needs the cat. If something is blue then it likes the tiger. If something likes the cat then it likes the tiger. If the tiger likes the dog then the dog is red. If something chases the tiger and the tiger needs the cat then the cat needs the bear. If something needs the bear then the bear likes the cat. If something is round then it likes the cat.

True or false: The bear likes the tiger.

Solution:
Context:
Constants:
bear = the bear
cat = the cat
dog = the dog
tiger = the tiger

Predicates:
Blue(x) = x is blue
Chases(x,y) = x chases y
Likes(x,y) = x likes y
Needs(x,y) = x needs y
Red(x) = x is red
Round(x) = x is round

Premises:
1. Chases(bear,tiger)
2. Likes(cat,tiger)
3. Likes(dog,cat)
4. Needs(tiger,cat)
5. Ax(Blue(x) -> Likes(x,tiger))
6. Ax(Likes(x,cat) -> Likes(x,tiger))
7. Likes(tiger,dog) -> Red(dog)
8. Ax(Chases(x,tiger) & Needs(tiger,cat) -> Needs(cat,bear))
9. Ax(Needs(x,bear) -> Likes(bear,cat))
10. Ax(Round(x) -> Likes(x,cat))

Derivation:
11. Chases(bear,tiger) ; R,1
12. Needs(tiger,cat) ; R,4
13. Chases(bear,tiger) & Needs(tiger,cat) ; ∧I,11,12
14. Chases(bear,tiger) & Needs(tiger,cat) -> Needs(cat,bear) ; AE,8
15. Needs(cat,bear) ; ->E,14,13
16. Needs(cat,bear) -> Likes(bear,cat) ; AE,9
17. Likes(bear,cat) ; ->E,16,15
18. Likes(bear,cat) -> Likes(bear,tiger) ; AE,6
19. Likes(bear,tiger) ; ->E,18,17

Conclusion:
Likes(bear,tiger)

Final answer: True<|endoftext|>The bear chases the tiger. The cat likes the tiger. The dog likes the cat. The tiger needs the cat. If something is blue then it likes the tiger. If something likes the cat then it likes the tiger. If the tiger likes the dog then the dog is red. If something chases the tiger and the tiger needs the cat then the cat needs the bear. If something needs the bear then the bear likes the cat. If something is round then it likes the cat.

True or false: The bear does not like the tiger.

Solution:
Context:
Constants:
bear = the bear
cat = the cat
dog = the dog
tiger = the tiger

Predicates:
Blue(x) = x is blue
Chases(x,y) = x chases y
Likes(x,y) = x likes y
Needs(x,y) = x needs y
Red(x) = x is red
Round(x) = x is round

Premises:
1. Chases(bear,tiger)
2. Likes(cat,tiger)
3. Likes(dog,cat)
4. Needs(tiger,cat)
5. Ax(Blue(x) -> Likes(x,tiger))
6. Ax(Likes(x,cat) -> Likes(x,tiger))
7. Likes(tiger,dog) -> Red(dog)
8. Ax(Chases(x,tiger) & Needs(tiger,cat) -> Needs(cat,bear))
9. Ax(Needs(x,bear) -> Likes(bear,cat))
10. Ax(Round(x) -> Likes(x,cat))

Derivation:
11. Chases(bear,tiger) ; R,1
12. Needs(tiger,cat) ; R,4
13. Chases(bear,tiger) & Needs(tiger,cat) ; ∧I,11,12
14. Chases(bear,tiger) & Needs(tiger,cat) -> Needs(cat,bear) ; AE,8
15. Needs(cat,bear) ; ->E,14,13
16. Needs(cat,bear) -> Likes(bear,cat) ; AE,9
17. Likes(bear,cat) ; ->E,16,15
18. Likes(bear,cat) -> Likes(bear,tiger) ; AE,6
19. Likes(bear,tiger) ; ->E,18,17

Conclusion:
Likes(bear,tiger)

Final answer: False<|endoftext|>Anne is cold. Gary is cold. Gary is quiet. If someone is quiet then they are young. All young people are nice. If Gary is nice and Gary is cold then Gary is blue.

True or false: Gary is nice.

Solution:
Context:
Constants:
anne = Anne
gary = Gary

Predicates:
Blue(x) = x is blue
Cold(x) = x is cold
Nice(x) = x is nice
Quiet(x) = x is quiet
Young(x) = x is young

Premises:
1. Cold(anne)
2. Cold(gary)
3. Quiet(gary)
4. Ax(Quiet(x) -> Young(x))
5. Ax(Young(x) -> Nice(x))
6. Nice(gary) & Cold(gary) -> Blue(gary)

Derivation:
7. Quiet(gary) ; R,3
8. Quiet(gary) -> Young(gary) ; AE,4
9. Young(gary) ; ->E,8,7
10. Young(gary) -> Nice(gary) ; AE,5
11. Nice(gary) ; ->E,10,9

Conclusion:
Nice(gary)

Final answer: True<|endoftext|>Anne is cold. Gary is cold. Gary is quiet. If someone is quiet then they are young. All young people are nice. If Gary is nice and Gary is cold then Gary is blue.

True or false: Gary is not nice.

Solution:
Context:
Constants:
anne = Anne
gary = Gary

Predicates:
Blue(x) = x is blue
Cold(x) = x is cold
Nice(x) = x is nice
Quiet(x) = x is quiet
Young(x) = x is young

Premises:
1. Cold(anne)
2. Cold(gary)
3. Quiet(gary)
4. Ax(Quiet(x) -> Young(x))
5. Ax(Young(x) -> Nice(x))
6. Nice(gary) & Cold(gary) -> Blue(gary)

Derivation:
7. Quiet(gary) ; R,3
8. Quiet(gary) -> Young(gary) ; AE,4
9. Young(gary) ; ->E,8,7
10. Young(gary) -> Nice(gary) ; AE,5
11. Nice(gary) ; ->E,10,9

Conclusion:
Nice(gary)

Final answer: False<|endoftext|>Anne is cold. Gary is cold. Gary is quiet. If someone is quiet then they are young. All young people are nice. If Gary is nice and Gary is cold then Gary is blue.

True or false: Gary is blue.

Solution:
Context:
Constants:
anne = Anne
gary = Gary

Predicates:
Blue(x) = x is blue
Cold(x) = x is cold
Nice(x) = x is nice
Quiet(x) = x is quiet
Young(x) = x is young

Premises:
1. Cold(anne)
2. Cold(gary)
3. Quiet(gary)
4. Ax(Quiet(x) -> Young(x))
5. Ax(Young(x) -> Nice(x))
6. Nice(gary) & Cold(gary) -> Blue(gary)

Derivation:
7. Quiet(gary) ; R,3
8. Quiet(gary) -> Young(gary) ; AE,4
9. Young(gary) ; ->E,8,7
10. Young(gary) -> Nice(gary) ; AE,5
11. Nice(gary) ; ->E,10,9
12. Cold(gary) ; R,2
13. Nice(gary) & Cold(gary) ; ∧I,11,12
14. Blue(gary) ; ->E,6,13

Conclusion:
Blue(gary)

Final answer: True<|endoftext|>Anne is cold. Gary is cold. Gary is quiet. If someone is quiet then they are young. All young people are nice. If Gary is nice and Gary is cold then Gary is blue.

True or false: Gary is not blue.

Solution:
Context:
Constants:
anne = Anne
gary = Gary

Predicates:
Blue(x) = x is blue
Cold(x) = x is cold
Nice(x) = x is nice
Quiet(x) = x is quiet
Young(x) = x is young

Premises:
1. Cold(anne)
2. Cold(gary)
3. Quiet(gary)
4. Ax(Quiet(x) -> Young(x))
5. Ax(Young(x) -> Nice(x))
6. Nice(gary) & Cold(gary) -> Blue(gary)

Derivation:
7. Quiet(gary) ; R,3
8. Quiet(gary) -> Young(gary) ; AE,4
9. Young(gary) ; ->E,8,7
10. Young(gary) -> Nice(gary) ; AE,5
11. Nice(gary) ; ->E,10,9
12. Cold(gary) ; R,2
13. Nice(gary) & Cold(gary) ; ∧I,11,12
14. Blue(gary) ; ->E,6,13

Conclusion:
Blue(gary)

Final answer: False<|endoftext|>The dog is green. The dog likes the mouse. The dog likes the tiger. The mouse needs the dog. The tiger likes the dog. The tiger needs the dog. The tiger needs the mouse. If someone likes the mouse then they need the tiger. If the dog needs the mouse then the dog visits the mouse. If someone visits the tiger and they visit the dog then the tiger needs the mouse. If someone needs the mouse and they like the tiger then the tiger likes the mouse. If someone needs the tiger then they visit the mouse. If someone is green and they visit the mouse then they need the mouse. If someone likes the mouse and the mouse likes the tiger then the tiger visits the mouse. If someone likes the dog then they visit the mouse.

True or false: The dog likes the tiger.

Solution:
Context:
Constants:
dog = the dog
mouse = the mouse
tiger = the tiger

Predicates:
Green(x) = x is green
Likes(x,y) = x likes y
Needs(x,y) = x needs y
Visits(x,y) = x visits y

Premises:
1. Green(dog)
2. Likes(dog,mouse)
3. Likes(dog,tiger)
4. Needs(mouse,dog)
5. Likes(tiger,dog)
6. Needs(tiger,dog)
7. Needs(tiger,mouse)
8. Ax(Likes(x,mouse) -> Needs(x,tiger))
9. Needs(dog,mouse) -> Visits(dog,mouse)
10. Ax(Visits(x,tiger) & Visits(x,dog) -> Needs(tiger,mouse))
11. Ax(Needs(x,mouse) & Likes(x,tiger) -> Likes(tiger,mouse))
12. Ax(Needs(x,tiger) -> Visits(x,mouse))
13. Ax(Green(x) & Visits(x,mouse) -> Needs(x,mouse))
14. Ax(Likes(x,mouse) & Likes(mouse,tiger) -> Visits(tiger,mouse))
15. Ax(Likes(x,dog) -> Visits(x,mouse))

Derivation:
16. Likes(dog,tiger) ; R,3

Conclusion:
Likes(dog,tiger)

Final answer: True<|endoftext|>The dog is green. The dog likes the mouse. The dog likes the tiger. The mouse needs the dog. The tiger likes the dog. The tiger needs the dog. The tiger needs the mouse. If someone likes the mouse then they need the tiger. If the dog needs the mouse then the dog visits the mouse. If someone visits the tiger and they visit the dog then the tiger needs the mouse. If someone needs the mouse and they like the tiger then the tiger likes the mouse. If someone needs the tiger then they visit the mouse. If someone is green and they visit the mouse then they need the mouse. If someone likes the mouse and the mouse likes the tiger then the tiger visits the mouse. If someone likes the dog then they visit the mouse.

True or false: The tiger does not need the mouse.

Solution:
Context:
Constants:
dog = the dog
mouse = the mouse
tiger = the tiger

Predicates:
Green(x) = x is green
Likes(x,y) = x likes y
Needs(x,y) = x needs y
Visits(x,y) = x visits y

Premises:
1. Green(dog)
2. Likes(dog,mouse)
3. Likes(dog,tiger)
4. Needs(mouse,dog)
5. Likes(tiger,dog)
6. Needs(tiger,dog)
7. Needs(tiger,mouse)
8. Ax(Likes(x,mouse) -> Needs(x,tiger))
9. Needs(dog,mouse) -> Visits(dog,mouse)
10. Ax(Visits(x,tiger) & Visits(x,dog) -> Needs(tiger,mouse))
11. Ax(Needs(x,mouse) & Likes(x,tiger) -> Likes(tiger,mouse))
12. Ax(Needs(x,tiger) -> Visits(x,mouse))
13. Ax(Green(x) & Visits(x,mouse) -> Needs(x,mouse))
14. Ax(Likes(x,mouse) & Likes(mouse,tiger) -> Visits(tiger,mouse))
15. Ax(Likes(x,dog) -> Visits(x,mouse))

Derivation:
16. Needs(tiger,mouse) ; R,7

Conclusion:
Needs(tiger,mouse)

Final answer: False<|endoftext|>
```

## Window 1 (4 documents, 581 pad tokens)
```
The lion is tired. The lion is dull. The lion is lazy. The lion chases the dog. The bear needs the rabbit. The bear is awful. The bear is strong. The dog is round. The dog is quiet. The dog is nice. The rabbit is lovely. The rabbit is beautiful. The rabbit is adorable. Round animals are lovely. If something is dull then it visits the dog. If something visits the dog then it is rough. If something is tired and dull then it is lazy. If something is lovely and beautiful then it is small. If something is awful and strong then it is heavy. If something is lazy then it is sleepy. All sleepy animals are slow. If something is lovely then it is beautiful. All beautiful animals are adorable. If something is heavy then it is big. All big animals are obese. If something is small then it is cute. All cute animals are furry. All rough animals are fierce.

True or false: The lion is slow.

Solution:
Context:
Constants:
bear = the bear
dog = the dog
lion = the lion
rabbit = the rabbit

Predicates:
Adorable(x) = x is adorable
Awful(x) = x is awful
Beautiful(x) = x is beautiful
Big(x) = x is big
Chases(x,y) = x chases y
Cute(x) = x is cute
Dull(x) = x is dull
Fierce(x) = x is fierce
Furry(x) = x is furry
Heavy(x) = x is heavy
Lazy(x) = x is lazy
Lovely(x) = x is lovely
Needs(x,y) = x needs y
Nice(x) = x is nice
Obese(x) = x is obese
Quiet(x) = x is quiet
Rough(x) = x is rough
Round(x) = x is round
Sleepy(x) = x is sleepy
Slow(x) = x is slow
Small(x) = x is small
Strong(x) = x is strong
Tired(x) = x is tired
Visits(x,y) = x visits y

Premises:
1. Tired(lion)
2. Dull(lion)
3. Lazy(lion)
4. Chases(lion,dog)
5. Needs(bear,rabbit)
6. Awful(bear)
7. Strong(bear)
8. Round(dog)
9. Quiet(dog)
10. Nice(dog)
11. Lovely(rabbit)
12. Beautiful(rabbit)
13. Adorable(rabbit)
14. Ax(Round(x) -> Lovely(x))
15. Ax(Dull(x) -> Visits(x,dog))
16. Ax(Visits(x,dog) -> Rough(x))
17. Ax(Tired(x) & Dull(x) -> Lazy(x))
18. Ax(Lovely(x) & Beautiful(x) -> Small(x))
19. Ax(Awful(x) & Strong(x) -> Heavy(x))
20. Ax(Lazy(x) -> Sleepy(x))
21. Ax(Sleepy(x) -> Slow(x))
22. Ax(Lovely(x) -> Beautiful(x))
23. Ax(Beautiful(x) -> Adorable(x))
24. Ax(Heavy(x) -> Big(x))
25. Ax(Big(x) -> Obese(x))
26. Ax(Small(x) -> Cute(x))
27. Ax(Cute(x) -> Furry(x))
28. Ax(Rough(x) -> Fierce(x))

Derivation:
29. Lazy(lion) ; R,3
30. Lazy(lion) -> Sleepy(lion) ; AE,20
31. Sleepy(lion) ; ->E,30,29
32. Sleepy(lion) -> Slow(lion) ; AE,21
33. Slow(lion) ; ->E,32,31

Conclusion:
Slow(lion)

Final answer: True<|endoftext|>The lion is tired. The lion is dull. The lion is lazy. The lion chases the dog. The bear needs the rabbit. The bear is awful. The bear is strong. The dog is round. The dog is quiet. The dog is nice. The rabbit is lovely. The rabbit is beautiful. The rabbit is adorable. Round animals are lovely. If something is dull then it visits the dog. If something visits the dog then it is rough. If something is tired and dull then it is lazy. If something is lovely and beautiful then it is small. If something is awful and strong then it is heavy. If something is lazy then it is sleepy. All sleepy animals are slow. If something is lovely then it is beautiful. All beautiful animals are adorable. If something is heavy then it is big. All big animals are obese. If something is small then it is cute. All cute animals are furry. All rough animals are fierce.

True or false: The lion is not slow.

Solution:
Context:
Constants:
bear = the bear
dog = the dog
lion = the lion
rabbit = the rabbit

Predicates:
Adorable(x) = x is adorable
Awful(x) = x is awful
Beautiful(x) = x is beautiful
Big(x) = x is big
Chases(x,y) = x chases y
Cute(x) = x is cute
Dull(x) = x is dull
Fierce(x) = x is fierce
Furry(x) = x is furry
Heavy(x) = x is heavy
Lazy(x) = x is lazy
Lovely(x) = x is lovely
Needs(x,y) = x needs y
Nice(x) = x is nice
Obese(x) = x is obese
Quiet(x) = x is quiet
Rough(x) = x is rough
Round(x) = x is round
Sleepy(x) = x is sleepy
Slow(x) = x is slow
Small(x) = x is small
Strong(x) = x is strong
Tired(x) = x is tired
Visits(x,y) = x visits y

Premises:
1. Tired(lion)
2. Dull(lion)
3. Lazy(lion)
4. Chases(lion,dog)
5. Needs(bear,rabbit)
6. Awful(bear)
7. Strong(bear)
8. Round(dog)
9. Quiet(dog)
10. Nice(dog)
11. Lovely(rabbit)
12. Beautiful(rabbit)
13. Adorable(rabbit)
14. Ax(Round(x) -> Lovely(x))
15. Ax(Dull(x) -> Visits(x,dog))
16. Ax(Visits(x,dog) -> Rough(x))
17. Ax(Tired(x) & Dull(x) -> Lazy(x))
18. Ax(Lovely(x) & Beautiful(x) -> Small(x))
19. Ax(Awful(x) & Strong(x) -> Heavy(x))
20. Ax(Lazy(x) -> Sleepy(x))
21. Ax(Sleepy(x) -> Slow(x))
22. Ax(Lovely(x) -> Beautiful(x))
23. Ax(Beautiful(x) -> Adorable(x))
24. Ax(Heavy(x) -> Big(x))
25. Ax(Big(x) -> Obese(x))
26. Ax(Small(x) -> Cute(x))
27. Ax(Cute(x) -> Furry(x))
28. Ax(Rough(x) -> Fierce(x))

Derivation:
29. Lazy(lion) ; R,3
30. Lazy(lion) -> Sleepy(lion) ; AE,20
31. Sleepy(lion) ; ->E,30,29
32. Sleepy(lion) -> Slow(lion) ; AE,21
33. Slow(lion) ; ->E,32,31

Conclusion:
Slow(lion)

Final answer: False<|endoftext|>The lion is tired. The lion is dull. The lion is lazy. The lion chases the dog. The bear needs the rabbit. The bear is awful. The bear is strong. The dog is round. The dog is quiet. The dog is nice. The rabbit is lovely. The rabbit is beautiful. The rabbit is adorable. Round animals are lovely. If something is dull then it visits the dog. If something visits the dog then it is rough. If something is tired and dull then it is lazy. If something is lovely and beautiful then it is small. If something is awful and strong then it is heavy. If something is lazy then it is sleepy. All sleepy animals are slow. If something is lovely then it is beautiful. All beautiful animals are adorable. If something is heavy then it is big. All big animals are obese. If something is small then it is cute. All cute animals are furry. All rough animals are fierce.

True or false: The bear is obese.

Solution:
Context:
Constants:
bear = the bear
dog = the dog
lion = the lion
rabbit = the rabbit

Predicates:
Adorable(x) = x is adorable
Awful(x) = x is awful
Beautiful(x) = x is beautiful
Big(x) = x is big
Chases(x,y) = x chases y
Cute(x) = x is cute
Dull(x) = x is dull
Fierce(x) = x is fierce
Furry(x) = x is furry
Heavy(x) = x is heavy
Lazy(x) = x is lazy
Lovely(x) = x is lovely
Needs(x,y) = x needs y
Nice(x) = x is nice
Obese(x) = x is obese
Quiet(x) = x is quiet
Rough(x) = x is rough
Round(x) = x is round
Sleepy(x) = x is sleepy
Slow(x) = x is slow
Small(x) = x is small
Strong(x) = x is strong
Tired(x) = x is tired
Visits(x,y) = x visits y

Premises:
1. Tired(lion)
2. Dull(lion)
3. Lazy(lion)
4. Chases(lion,dog)
5. Needs(bear,rabbit)
6. Awful(bear)
7. Strong(bear)
8. Round(dog)
9. Quiet(dog)
10. Nice(dog)
11. Lovely(rabbit)
12. Beautiful(rabbit)
13. Adorable(rabbit)
14. Ax(Round(x) -> Lovely(x))
15. Ax(Dull(x) -> Visits(x,dog))
16. Ax(Visits(x,dog) -> Rough(x))
17. Ax(Tired(x) & Dull(x) -> Lazy(x))
18. Ax(Lovely(x) & Beautiful(x) -> Small(x))
19. Ax(Awful(x) & Strong(x) -> Heavy(x))
20. Ax(Lazy(x) -> Sleepy(x))
21. Ax(Sleepy(x) -> Slow(x))
22. Ax(Lovely(x) -> Beautiful(x))
23. Ax(Beautiful(x) -> Adorable(x))
24. Ax(Heavy(x) -> Big(x))
25. Ax(Big(x) -> Obese(x))
26. Ax(Small(x) -> Cute(x))
27. Ax(Cute(x) -> Furry(x))
28. Ax(Rough(x) -> Fierce(x))

Derivation:
29. Awful(bear) ; R,6
30. Strong(bear) ; R,7
31. Awful(bear) & Strong(bear) ; ∧I,29,30
32. Awful(bear) & Strong(bear) -> Heavy(bear) ; AE,19
33. Heavy(bear) ; ->E,32,31
34. Heavy(bear) -> Big(bear) ; AE,24
35. Big(bear) ; ->E,34,33
36. Big(bear) -> Obese(bear) ; AE,25
37. Obese(bear) ; ->E,36,35

Conclusion:
Obese(bear)

Final answer: True<|endoftext|>The lion is tired. The lion is dull. The lion is lazy. The lion chases the dog. The bear needs the rabbit. The bear is awful. The bear is strong. The dog is round. The dog is quiet. The dog is nice. The rabbit is lovely. The rabbit is beautiful. The rabbit is adorable. Round animals are lovely. If something is dull then it visits the dog. If something visits the dog then it is rough. If something is tired and dull then it is lazy. If something is lovely and beautiful then it is small. If something is awful and strong then it is heavy. If something is lazy then it is sleepy. All sleepy animals are slow. If something is lovely then it is beautiful. All beautiful animals are adorable. If something is heavy then it is big. All big animals are obese. If something is small then it is cute. All cute animals are furry. All rough animals are fierce.

True or false: The bear is not obese.

Solution:
Context:
Constants:
bear = the bear
dog = the dog
lion = the lion
rabbit = the rabbit

Predicates:
Adorable(x) = x is adorable
Awful(x) = x is awful
Beautiful(x) = x is beautiful
Big(x) = x is big
Chases(x,y) = x chases y
Cute(x) = x is cute
Dull(x) = x is dull
Fierce(x) = x is fierce
Furry(x) = x is furry
Heavy(x) = x is heavy
Lazy(x) = x is lazy
Lovely(x) = x is lovely
Needs(x,y) = x needs y
Nice(x) = x is nice
Obese(x) = x is obese
Quiet(x) = x is quiet
Rough(x) = x is rough
Round(x) = x is round
Sleepy(x) = x is sleepy
Slow(x) = x is slow
Small(x) = x is small
Strong(x) = x is strong
Tired(x) = x is tired
Visits(x,y) = x visits y

Premises:
1. Tired(lion)
2. Dull(lion)
3. Lazy(lion)
4. Chases(lion,dog)
5. Needs(bear,rabbit)
6. Awful(bear)
7. Strong(bear)
8. Round(dog)
9. Quiet(dog)
10. Nice(dog)
11. Lovely(rabbit)
12. Beautiful(rabbit)
13. Adorable(rabbit)
14. Ax(Round(x) -> Lovely(x))
15. Ax(Dull(x) -> Visits(x,dog))
16. Ax(Visits(x,dog) -> Rough(x))
17. Ax(Tired(x) & Dull(x) -> Lazy(x))
18. Ax(Lovely(x) & Beautiful(x) -> Small(x))
19. Ax(Awful(x) & Strong(x) -> Heavy(x))
20. Ax(Lazy(x) -> Sleepy(x))
21. Ax(Sleepy(x) -> Slow(x))
22. Ax(Lovely(x) -> Beautiful(x))
23. Ax(Beautiful(x) -> Adorable(x))
24. Ax(Heavy(x) -> Big(x))
25. Ax(Big(x) -> Obese(x))
26. Ax(Small(x) -> Cute(x))
27. Ax(Cute(x) -> Furry(x))
28. Ax(Rough(x) -> Fierce(x))

Derivation:
29. Awful(bear) ; R,6
30. Strong(bear) ; R,7
31. Awful(bear) & Strong(bear) ; ∧I,29,30
32. Awful(bear) & Strong(bear) -> Heavy(bear) ; AE,19
33. Heavy(bear) ; ->E,32,31
34. Heavy(bear) -> Big(bear) ; AE,24
35. Big(bear) ; ->E,34,33
36. Big(bear) -> Obese(bear) ; AE,25
37. Obese(bear) ; ->E,36,35

Conclusion:
Obese(bear)

Final answer: False<|endoftext|>
```

## Window 2 (4 documents, 667 pad tokens)
```
Alan is big. Alan is huge. Alan is high. Bob is small. Bob is little. Fiona is kind. Fiona is quiet. Fiona is nice. Anne is bad. Anne is rough. Anne is imperfect. Big people are kind. If someone is small and little then they are thin. If someone is bad and rough then they are poor. If someone is kind and quiet then they are smart. If someone is thin then they are short. If someone is short then they are tiny. All tiny people are imperfect. If someone is kind then they are quiet. If someone is quiet then they are nice. All nice people are strong. If someone is smart then they are wealthy. If someone is wealthy then they are clever. All clever people are huge. If someone is poor then they are sad. If someone is sad then they are dull. All dull people are little.

True or false: Alan is strong.

Solution:
Context:
Constants:
alan = Alan
anne = Anne
bob = Bob
fiona = Fiona

Predicates:
Bad(x) = x is bad
Big(x) = x is big
Clever(x) = x is clever
Dull(x) = x is dull
High(x) = x is high
Huge(x) = x is huge
Imperfect(x) = x is imperfect
Kind(x) = x is kind
Little(x) = x is little
Nice(x) = x is nice
Poor(x) = x is poor
Quiet(x) = x is quiet
Rough(x) = x is rough
Sad(x) = x is sad
Short(x) = x is short
Small(x) = x is small
Smart(x) = x is smart
Strong(x) = x is strong
Thin(x) = x is thin
Tiny(x) = x is tiny
Wealthy(x) = x is wealthy

Premises:
1. Big(alan)
2. Huge(alan)
3. High(alan)
4. Small(bob)
5. Little(bob)
6. Kind(fiona)
7. Quiet(fiona)
8. Nice(fiona)
9. Bad(anne)
10. Rough(anne)
11. Imperfect(anne)
12. Ax(Big(x) -> Kind(x))
13. Ax(Small(x) & Little(x) -> Thin(x))
14. Ax(Bad(x) & Rough(x) -> Poor(x))
15. Ax(Kind(x) & Quiet(x) -> Smart(x))
16. Ax(Thin(x) -> Short(x))
17. Ax(Short(x) -> Tiny(x))
18. Ax(Tiny(x) -> Imperfect(x))
19. Ax(Kind(x) -> Quiet(x))
20. Ax(Quiet(x) -> Nice(x))
21. Ax(Nice(x) -> Strong(x))
22. Ax(Smart(x) -> Wealthy(x))
23. Ax(Wealthy(x) -> Clever(x))
24. Ax(Clever(x) -> Huge(x))
25. Ax(Poor(x) -> Sad(x))
26. Ax(Sad(x) -> Dull(x))
27. Ax(Dull(x) -> Little(x))

Derivation:
28. Big(alan) ; R,1
29. Big(alan) -> Kind(alan) ; AE,12
30. Kind(alan) ; ->E,29,28
31. Kind(alan) -> Quiet(alan) ; AE,19
32. Quiet(alan) ; ->E,31,30
33. Quiet(alan) -> Nice(alan) ; AE,20
34. Nice(alan) ; ->E,33,32
35. Nice(alan) -> Strong(alan) ; AE,21
36. Strong(alan) ; ->E,35,34

Conclusion:
Strong(alan)

Final answer: True<|endoftext|>Alan is big. Alan is huge. Alan is high. Bob is small. Bob is little. Fiona is kind. Fiona is quiet. Fiona is nice. Anne is bad. Anne is rough. Anne is imperfect. Big people are kind. If someone is small and little then they are thin. If someone is bad and rough then they are poor. If someone is kind and quiet then they are smart. If someone is thin then they are short. If someone is short then they are tiny. All tiny people are imperfect. If someone is kind then they are quiet. If someone is quiet then they are nice. All nice people are strong. If someone is smart then they are wealthy. If someone is wealthy then they are clever. All clever people are huge. If someone is poor then they are sad. If someone is sad then they are dull. All dull people are little.

True or false: Alan is not strong.

Solution:
Context:
Constants:
alan = Alan
anne = Anne
bob = Bob
fiona = Fiona

Predicates:
Bad(x) = x is bad
Big(x) = x is big
Clever(x) = x is clever
Dull(x) = x is dull
High(x) = x is high
Huge(x) = x is huge
Imperfect(x) = x is imperfect
Kind(x) = x is kind
Little(x) = x is little
Nice(x) = x is nice
Poor(x) = x is poor
Quiet(x) = x is quiet
Rough(x) = x is rough
Sad(x) = x is sad
Short(x) = x is short
Small(x) = x is small
Smart(x) = x is smart
Strong(x) = x is strong
Thin(x) = x is thin
Tiny(x) = x is tiny
Wealthy(x) = x is wealthy

Premises:
1. Big(alan)
2. Huge(alan)
3. High(alan)
4. Small(bob)
5. Little(bob)
6. Kind(fiona)
7. Quiet(fiona)
8. Nice(fiona)
9. Bad(anne)
10. Rough(anne)
11. Imperfect(anne)
12. Ax(Big(x) -> Kind(x))
13. Ax(Small(x) & Little(x) -> Thin(x))
14. Ax(Bad(x) & Rough(x) -> Poor(x))
15. Ax(Kind(x) & Quiet(x) -> Smart(x))
16. Ax(Thin(x) -> Short(x))
17. Ax(Short(x) -> Tiny(x))
18. Ax(Tiny(x) -> Imperfect(x))
19. Ax(Kind(x) -> Quiet(x))
20. Ax(Quiet(x) -> Nice(x))
21. Ax(Nice(x) -> Strong(x))
22. Ax(Smart(x) -> Wealthy(x))
23. Ax(Wealthy(x) -> Clever(x))
24. Ax(Clever(x) -> Huge(x))
25. Ax(Poor(x) -> Sad(x))
26. Ax(Sad(x) -> Dull(x))
27. Ax(Dull(x) -> Little(x))

Derivation:
28. Big(alan) ; R,1
29. Big(alan) -> Kind(alan) ; AE,12
30. Kind(alan) ; ->E,29,28
31. Kind(alan) -> Quiet(alan) ; AE,19
32. Quiet(alan) ; ->E,31,30
33. Quiet(alan) -> Nice(alan) ; AE,20
34. Nice(alan) ; ->E,33,32
35. Nice(alan) -> Strong(alan) ; AE,21
36. Strong(alan) ; ->E,35,34

Conclusion:
Strong(alan)

Final answer: False<|endoftext|>Alan is big. Alan is huge. Alan is high. Bob is small. Bob is little. Fiona is kind. Fiona is quiet. Fiona is nice. Anne is bad. Anne is rough. Anne is imperfect. Big people are kind. If someone is small and little then they are thin. If someone is bad and rough then they are poor. If someone is kind and quiet then they are smart. If someone is thin then they are short. If someone is short then they are tiny. All tiny people are imperfect. If someone is kind then they are quiet. If someone is quiet then they are nice. All nice people are strong. If someone is smart then they are wealthy. If someone is wealthy then they are clever. All clever people are huge. If someone is poor then they are sad. If someone is sad then they are dull. All dull people are little.

True or false: Bob is imperfect.

Solution:
Context:
Constants:
alan = Alan
anne = Anne
bob = Bob
fiona = Fiona

Predicates:
Bad(x) = x is bad
Big(x) = x is big
Clever(x) = x is clever
Dull(x) = x is dull
High(x) = x is high
Huge(x) = x is huge
Imperfect(x) = x is imperfect
Kind(x) = x is kind
Little(x) = x is little
Nice(x) = x is nice
Poor(x) = x is poor
Quiet(x) = x is quiet
Rough(x) = x is rough
Sad(x) = x is sad
Short(x) = x is short
Small(x) = x is small
Smart(x) = x is smart
Strong(x) = x is strong
Thin(x) = x is thin
Tiny(x) = x is tiny
Wealthy(x) = x is wealthy

Premises:
1. Big(alan)
2. Huge(alan)
3. High(alan)
4. Small(bob)
5. Little(bob)
6. Kind(fiona)
7. Quiet(fiona)
8. Nice(fiona)
9. Bad(anne)
10. Rough(anne)
11. Imperfect(anne)
12. Ax(Big(x) -> Kind(x))
13. Ax(Small(x) & Little(x) -> Thin(x))
14. Ax(Bad(x) & Rough(x) -> Poor(x))
15. Ax(Kind(x) & Quiet(x) -> Smart(x))
16. Ax(Thin(x) -> Short(x))
17. Ax(Short(x) -> Tiny(x))
18. Ax(Tiny(x) -> Imperfect(x))
19. Ax(Kind(x) -> Quiet(x))
20. Ax(Quiet(x) -> Nice(x))
21. Ax(Nice(x) -> Strong(x))
22. Ax(Smart(x) -> Wealthy(x))
23. Ax(Wealthy(x) -> Clever(x))
24. Ax(Clever(x) -> Huge(x))
25. Ax(Poor(x) -> Sad(x))
26. Ax(Sad(x) -> Dull(x))
27. Ax(Dull(x) -> Little(x))

Derivation:
28. Small(bob) ; R,4
29. Little(bob) ; R,5
30. Small(bob) & Little(bob) ; ∧I,28,29
31. Small(bob) & Little(bob) -> Thin(bob) ; AE,13
32. Thin(bob) ; ->E,31,30
33. Thin(bob) -> Short(bob) ; AE,16
34. Short(bob) ; ->E,33,32
35. Short(bob) -> Tiny(bob) ; AE,17
36. Tiny(bob) ; ->E,35,34
37. Tiny(bob) -> Imperfect(bob) ; AE,18
38. Imperfect(bob) ; ->E,37,36

Conclusion:
Imperfect(bob)

Final answer: True<|endoftext|>Alan is big. Alan is huge. Alan is high. Bob is small. Bob is little. Fiona is kind. Fiona is quiet. Fiona is nice. Anne is bad. Anne is rough. Anne is imperfect. Big people are kind. If someone is small and little then they are thin. If someone is bad and rough then they are poor. If someone is kind and quiet then they are smart. If someone is thin then they are short. If someone is short then they are tiny. All tiny people are imperfect. If someone is kind then they are quiet. If someone is quiet then they are nice. All nice people are strong. If someone is smart then they are wealthy. If someone is wealthy then they are clever. All clever people are huge. If someone is poor then they are sad. If someone is sad then they are dull. All dull people are little.

True or false: Bob is not imperfect.

Solution:
Context:
Constants:
alan = Alan
anne = Anne
bob = Bob
fiona = Fiona

Predicates:
Bad(x) = x is bad
Big(x) = x is big
Clever(x) = x is clever
Dull(x) = x is dull
High(x) = x is high
Huge(x) = x is huge
Imperfect(x) = x is imperfect
Kind(x) = x is kind
Little(x) = x is little
Nice(x) = x is nice
Poor(x) = x is poor
Quiet(x) = x is quiet
Rough(x) = x is rough
Sad(x) = x is sad
Short(x) = x is short
Small(x) = x is small
Smart(x) = x is smart
Strong(x) = x is strong
Thin(x) = x is thin
Tiny(x) = x is tiny
Wealthy(x) = x is wealthy

Premises:
1. Big(alan)
2. Huge(alan)
3. High(alan)
4. Small(bob)
5. Little(bob)
6. Kind(fiona)
7. Quiet(fiona)
8. Nice(fiona)
9. Bad(anne)
10. Rough(anne)
11. Imperfect(anne)
12. Ax(Big(x) -> Kind(x))
13. Ax(Small(x) & Little(x) -> Thin(x))
14. Ax(Bad(x) & Rough(x) -> Poor(x))
15. Ax(Kind(x) & Quiet(x) -> Smart(x))
16. Ax(Thin(x) -> Short(x))
17. Ax(Short(x) -> Tiny(x))
18. Ax(Tiny(x) -> Imperfect(x))
19. Ax(Kind(x) -> Quiet(x))
20. Ax(Quiet(x) -> Nice(x))
21. Ax(Nice(x) -> Strong(x))
22. Ax(Smart(x) -> Wealthy(x))
23. Ax(Wealthy(x) -> Clever(x))
24. Ax(Clever(x) -> Huge(x))
25. Ax(Poor(x) -> Sad(x))
26. Ax(Sad(x) -> Dull(x))
27. Ax(Dull(x) -> Little(x))

Derivation:
28. Small(bob) ; R,4
29. Little(bob) ; R,5
30. Small(bob) & Little(bob) ; ∧I,28,29
31. Small(bob) & Little(bob) -> Thin(bob) ; AE,13
32. Thin(bob) ; ->E,31,30
33. Thin(bob) -> Short(bob) ; AE,16
34. Short(bob) ; ->E,33,32
35. Short(bob) -> Tiny(bob) ; AE,17
36. Tiny(bob) ; ->E,35,34
37. Tiny(bob) -> Imperfect(bob) ; AE,18
38. Imperfect(bob) ; ->E,37,36

Conclusion:
Imperfect(bob)

Final answer: False<|endoftext|>
```

## Window 3 (5 documents, 521 pad tokens)
```
The bald eagle is rough. The bald eagle is lazy. The bald eagle is sleepy. The bald eagle likes the dog. The snake attacks the cat. The snake is big. The snake is awful. The dog is nice. The dog is quiet. The dog is kind. The cat is small. The cat is lovely. The cat is furry. Nice animals are small. If something is lazy then it sees the dog. If something sees the dog then it is slow. If something is rough and lazy then it is sleepy. If something is small and lovely then it is cute. If something is big and awful then it is strong. All sleepy animals are dull. All small animals are lovely. All strong animals are fierce. All cute animals are beautiful.

True or false: The dog is not lovely.

Solution:
Context:
Constants:
bald_eagle = the bald eagle
cat = the cat
dog = the dog
snake = the snake

Predicates:
Attacks(x,y) = x attacks y
Awful(x) = x is awful
Beautiful(x) = x is beautiful
Big(x) = x is big
Cute(x) = x is cute
Dull(x) = x is dull
Fierce(x) = x is fierce
Furry(x) = x is furry
Kind(x) = x is kind
Lazy(x) = x is lazy
Likes(x,y) = x likes y
Lovely(x) = x is lovely
Nice(x) = x is nice
Quiet(x) = x is quiet
Rough(x) = x is rough
Sees(x,y) = x sees y
Sleepy(x) = x is sleepy
Slow(x) = x is slow
Small(x) = x is small
Strong(x) = x is strong

Premises:
1. Rough(bald_eagle)
2. Lazy(bald_eagle)
3. Sleepy(bald_eagle)
4. Likes(bald_eagle,dog)
5. Attacks(snake,cat)
6. Big(snake)
7. Awful(snake)
8. Nice(dog)
9. Quiet(dog)
10. Kind(dog)
11. Small(cat)
12. Lovely(cat)
13. Furry(cat)
14. Ax(Nice(x) -> Small(x))
15. Ax(Lazy(x) -> Sees(x,dog))
16. Ax(Sees(x,dog) -> Slow(x))
17. Ax(Rough(x) & Lazy(x) -> Sleepy(x))
18. Ax(Small(x) & Lovely(x) -> Cute(x))
19. Ax(Big(x) & Awful(x) -> Strong(x))
20. Ax(Sleepy(x) -> Dull(x))
21. Ax(Small(x) -> Lovely(x))
22. Ax(Strong(x) -> Fierce(x))
23. Ax(Cute(x) -> Beautiful(x))

Derivation:
24. Nice(dog) ; R,8
25. Nice(dog) -> Small(dog) ; AE,14
26. Small(dog) ; ->E,25,24
27. Small(dog) -> Lovely(dog) ; AE,21
28. Lovely(dog) ; ->E,27,26

Conclusion:
Lovely(dog)

Final answer: False<|endoftext|>The bald eagle is rough. The bald eagle is lazy. The bald eagle is sleepy. The bald eagle likes the dog. The snake attacks the cat. The snake is big. The snake is awful. The dog is nice. The dog is quiet. The dog is kind. The cat is small. The cat is lovely. The cat is furry. Nice animals are small. If something is lazy then it sees the dog. If something sees the dog then it is slow. If something is rough and lazy then it is sleepy. If something is small and lovely then it is cute. If something is big and awful then it is strong. All sleepy animals are dull. All small animals are lovely. All strong animals are fierce. All cute animals are beautiful.

True or false: The bald eagle is dull.

Solution:
Context:
Constants:
bald_eagle = the bald eagle
cat = the cat
dog = the dog
snake = the snake

Predicates:
Attacks(x,y) = x attacks y
Awful(x) = x is awful
Beautiful(x) = x is beautiful
Big(x) = x is big
Cute(x) = x is cute
Dull(x) = x is dull
Fierce(x) = x is fierce
Furry(x) = x is furry
Kind(x) = x is kind
Lazy(x) = x is lazy
Likes(x,y) = x likes y
Lovely(x) = x is lovely
Nice(x) = x is nice
Quiet(x) = x is quiet
Rough(x) = x is rough
Sees(x,y) = x sees y
Sleepy(x) = x is sleepy
Slow(x) = x is slow
Small(x) = x is small
Strong(x) = x is strong

Premises:
1. Rough(bald_eagle)
2. Lazy(bald_eagle)
3. Sleepy(bald_eagle)
4. Likes(bald_eagle,dog)
5. Attacks(snake,cat)
6. Big(snake)
7. Awful(snake)
8. Nice(dog)
9. Quiet(dog)
10. Kind(dog)
11. Small(cat)
12. Lovely(cat)
13. Furry(cat)
14. Ax(Nice(x) -> Small(x))
15. Ax(Lazy(x) -> Sees(x,dog))
16. Ax(Sees(x,dog) -> Slow(x))
17. Ax(Rough(x) & Lazy(x) -> Sleepy(x))
18. Ax(Small(x) & Lovely(x) -> Cute(x))
19. Ax(Big(x) & Awful(x) -> Strong(x))
20. Ax(Sleepy(x) -> Dull(x))
21. Ax(Small(x) -> Lovely(x))
22. Ax(Strong(x) -> Fierce(x))
23. Ax(Cute(x) -> Beautiful(x))

Derivation:
24. Sleepy(bald_eagle) ; R,3
25. Sleepy(bald_eagle) -> Dull(bald_eagle) ; AE,20
26. Dull(bald_eagle) ; ->E,25,24

Conclusion:
Dull(bald_eagle)

Final answer: True<|endoftext|>The bald eagle is rough. The bald eagle is lazy. The bald eagle is sleepy. The bald eagle likes the dog. The snake attacks the cat. The snake is big. The snake is awful. The dog is nice. The dog is quiet. The dog is kind. The cat is small. The cat is lovely. The cat is furry. Nice animals are small. If something is lazy then it sees the dog. If something sees the dog then it is slow. If something is rough and lazy then it is sleepy. If something is small and lovely then it is cute. If something is big and awful then it is strong. All sleepy animals are dull. All small animals are lovely. All strong animals are fierce. All cute animals are beautiful.

True or false: The bald eagle is not dull.

Solution:
Context:
Constants:
bald_eagle = the bald eagle
cat = the cat
dog = the dog
snake = the snake

Predicates:
Attacks(x,y) = x attacks y
Awful(x) = x is awful
Beautiful(x) = x is beautiful
Big(x) = x is big
Cute(x) = x is cute
Dull(x) = x is dull
Fierce(x) = x is fierce
Furry(x) = x is furry
Kind(x) = x is kind
Lazy(x) = x is lazy
Likes(x,y) = x likes y
Lovely(x) = x is lovely
Nice(x) = x is nice
Quiet(x) = x is quiet
Rough(x) = x is rough
Sees(x,y) = x sees y
Sleepy(x) = x is sleepy
Slow(x) = x is slow
Small(x) = x is small
Strong(x) = x is strong

Premises:
1. Rough(bald_eagle)
2. Lazy(bald_eagle)
3. Sleepy(bald_eagle)
4. Likes(bald_eagle,dog)
5. Attacks(snake,cat)
6. Big(snake)
7. Awful(snake)
8. Nice(dog)
9. Quiet(dog)
10. Kind(dog)
11. Small(cat)
12. Lovely(cat)
13. Furry(cat)
14. Ax(Nice(x) -> Small(x))
15. Ax(Lazy(x) -> Sees(x,dog))
16. Ax(Sees(x,dog) -> Slow(x))
17. Ax(Rough(x) & Lazy(x) -> Sleepy(x))
18. Ax(Small(x) & Lovely(x) -> Cute(x))
19. Ax(Big(x) & Awful(x) -> Strong(x))
20. Ax(Sleepy(x) -> Dull(x))
21. Ax(Small(x) -> Lovely(x))
22. Ax(Strong(x) -> Fierce(x))
23. Ax(Cute(x) -> Beautiful(x))

Derivation:
24. Sleepy(bald_eagle) ; R,3
25. Sleepy(bald_eagle) -> Dull(bald_eagle) ; AE,20
26. Dull(bald_eagle) ; ->E,25,24

Conclusion:
Dull(bald_eagle)

Final answer: False<|endoftext|>The bald eagle is rough. The bald eagle is lazy. The bald eagle is sleepy. The bald eagle likes the dog. The snake attacks the cat. The snake is big. The snake is awful. The dog is nice. The dog is quiet. The dog is kind. The cat is small. The cat is lovely. The cat is furry. Nice animals are small. If something is lazy then it sees the dog. If something sees the dog then it is slow. If something is rough and lazy then it is sleepy. If something is small and lovely then it is cute. If something is big and awful then it is strong. All sleepy animals are dull. All small animals are lovely. All strong animals are fierce. All cute animals are beautiful.

True or false: The snake is fierce.

Solution:
Context:
Constants:
bald_eagle = the bald eagle
cat = the cat
dog = the dog
snake = the snake

Predicates:
Attacks(x,y) = x attacks y
Awful(x) = x is awful
Beautiful(x) = x is beautiful
Big(x) = x is big
Cute(x) = x is cute
Dull(x) = x is dull
Fierce(x) = x is fierce
Furry(x) = x is furry
Kind(x) = x is kind
Lazy(x) = x is lazy
Likes(x,y) = x likes y
Lovely(x) = x is lovely
Nice(x) = x is nice
Quiet(x) = x is quiet
Rough(x) = x is rough
Sees(x,y) = x sees y
Sleepy(x) = x is sleepy
Slow(x) = x is slow
Small(x) = x is small
Strong(x) = x is strong

Premises:
1. Rough(bald_eagle)
2. Lazy(bald_eagle)
3. Sleepy(bald_eagle)
4. Likes(bald_eagle,dog)
5. Attacks(snake,cat)
6. Big(snake)
7. Awful(snake)
8. Nice(dog)
9. Quiet(dog)
10. Kind(dog)
11. Small(cat)
12. Lovely(cat)
13. Furry(cat)
14. Ax(Nice(x) -> Small(x))
15. Ax(Lazy(x) -> Sees(x,dog))
16. Ax(Sees(x,dog) -> Slow(x))
17. Ax(Rough(x) & Lazy(x) -> Sleepy(x))
18. Ax(Small(x) & Lovely(x) -> Cute(x))
19. Ax(Big(x) & Awful(x) -> Strong(x))
20. Ax(Sleepy(x) -> Dull(x))
21. Ax(Small(x) -> Lovely(x))
22. Ax(Strong(x) -> Fierce(x))
23. Ax(Cute(x) -> Beautiful(x))

Derivation:
24. Big(snake) ; R,6
25. Awful(snake) ; R,7
26. Big(snake) & Awful(snake) ; ∧I,24,25
27. Big(snake) & Awful(snake) -> Strong(snake) ; AE,19
28. Strong(snake) ; ->E,27,26
29. Strong(snake) -> Fierce(snake) ; AE,22
30. Fierce(snake) ; ->E,29,28

Conclusion:
Fierce(snake)

Final answer: True<|endoftext|>The bald eagle is rough. The bald eagle is lazy. The bald eagle is sleepy. The bald eagle likes the dog. The snake attacks the cat. The snake is big. The snake is awful. The dog is nice. The dog is quiet. The dog is kind. The cat is small. The cat is lovely. The cat is furry. Nice animals are small. If something is lazy then it sees the dog. If something sees the dog then it is slow. If something is rough and lazy then it is sleepy. If something is small and lovely then it is cute. If something is big and awful then it is strong. All sleepy animals are dull. All small animals are lovely. All strong animals are fierce. All cute animals are beautiful.

True or false: The snake is not fierce.

Solution:
Context:
Constants:
bald_eagle = the bald eagle
cat = the cat
dog = the dog
snake = the snake

Predicates:
Attacks(x,y) = x attacks y
Awful(x) = x is awful
Beautiful(x) = x is beautiful
Big(x) = x is big
Cute(x) = x is cute
Dull(x) = x is dull
Fierce(x) = x is fierce
Furry(x) = x is furry
Kind(x) = x is kind
Lazy(x) = x is lazy
Likes(x,y) = x likes y
Lovely(x) = x is lovely
Nice(x) = x is nice
Quiet(x) = x is quiet
Rough(x) = x is rough
Sees(x,y) = x sees y
Sleepy(x) = x is sleepy
Slow(x) = x is slow
Small(x) = x is small
Strong(x) = x is strong

Premises:
1. Rough(bald_eagle)
2. Lazy(bald_eagle)
3. Sleepy(bald_eagle)
4. Likes(bald_eagle,dog)
5. Attacks(snake,cat)
6. Big(snake)
7. Awful(snake)
8. Nice(dog)
9. Quiet(dog)
10. Kind(dog)
11. Small(cat)
12. Lovely(cat)
13. Furry(cat)
14. Ax(Nice(x) -> Small(x))
15. Ax(Lazy(x) -> Sees(x,dog))
16. Ax(Sees(x,dog) -> Slow(x))
17. Ax(Rough(x) & Lazy(x) -> Sleepy(x))
18. Ax(Small(x) & Lovely(x) -> Cute(x))
19. Ax(Big(x) & Awful(x) -> Strong(x))
20. Ax(Sleepy(x) -> Dull(x))
21. Ax(Small(x) -> Lovely(x))
22. Ax(Strong(x) -> Fierce(x))
23. Ax(Cute(x) -> Beautiful(x))

Derivation:
24. Big(snake) ; R,6
25. Awful(snake) ; R,7
26. Big(snake) & Awful(snake) ; ∧I,24,25
27. Big(snake) & Awful(snake) -> Strong(snake) ; AE,19
28. Strong(snake) ; ->E,27,26
29. Strong(snake) -> Fierce(snake) ; AE,22
30. Fierce(snake) ; ->E,29,28

Conclusion:
Fierce(snake)

Final answer: False<|endoftext|>
```

## Window 3824 (10 documents, 17 pad tokens)
```
Anne is furry. Anne is green. Anne is kind. Erin is big. Erin is blue. Erin is furry. Erin is green. Erin is kind. Erin is quiet. Erin is round. Fiona is big. Fiona is green. Fiona is kind. Fiona is quiet. Fiona is round. Kind people are quiet. If someone is round then they are kind. All kind, big people are blue. If someone is green then they are furry. All green, blue people are furry. If someone is green and quiet then they are big.

True or false: Anne is not blue.

Solution:
Context:
Constants:
anne = Anne
erin = Erin
fiona = Fiona

Predicates:
Big(x) = x is big
Blue(x) = x is blue
Furry(x) = x is furry
Green(x) = x is green
Kind(x) = x is kind
Quiet(x) = x is quiet
Round(x) = x is round

Premises:
1. Furry(anne)
2. Green(anne)
3. Kind(anne)
4. Big(erin)
5. Blue(erin)
6. Furry(erin)
7. Green(erin)
8. Kind(erin)
9. Quiet(erin)
10. Round(erin)
11. Big(fiona)
12. Green(fiona)
13. Kind(fiona)
14. Quiet(fiona)
15. Round(fiona)
16. Ax(Kind(x) -> Quiet(x))
17. Ax(Round(x) -> Kind(x))
18. Ax(Kind(x) & Big(x) -> Blue(x))
19. Ax(Green(x) -> Furry(x))
20. Ax(Green(x) & Blue(x) -> Furry(x))
21. Ax(Green(x) & Quiet(x) -> Big(x))

Derivation:
22. Kind(anne) ; R,3
23. Green(anne) ; R,2
24. Kind(anne) -> Quiet(anne) ; AE,16
25. Quiet(anne) ; ->E,24,22
26. Green(anne) & Quiet(anne) ; ∧I,23,25
27. Green(anne) & Quiet(anne) -> Big(anne) ; AE,21
28. Big(anne) ; ->E,27,26
29. Kind(anne) & Big(anne) ; ∧I,22,28
30. Kind(anne) & Big(anne) -> Blue(anne) ; AE,18
31. Blue(anne) ; ->E,30,29

Conclusion:
Blue(anne)

Final answer: False<|endoftext|>Anne is cold. Anne is furry. Charlie is cold. Charlie is green. Erin is green. Erin is smart. Erin is white. If something is green and furry then it is white. Furry, smart things are big. Big things are smart. If something is furry and green then it is big. If something is cold then it is furry. If something is green and white then it is furry.

True or false: Erin is smart.

Solution:
Context:
Constants:
anne = Anne
charlie = Charlie
erin = Erin

Predicates:
Big(x) = x is big
Cold(x) = x is cold
Furry(x) = x is furry
Green(x) = x is green
Smart(x) = x is smart
White(x) = x is white

Premises:
1. Cold(anne)
2. Furry(anne)
3. Cold(charlie)
4. Green(charlie)
5. Green(erin)
6. Smart(erin)
7. White(erin)
8. Ax(Green(x) & Furry(x) -> White(x))
9. Ax(Furry(x) & Smart(x) -> Big(x))
10. Ax(Big(x) -> Smart(x))
11. Ax(Furry(x) & Green(x) -> Big(x))
12. Ax(Cold(x) -> Furry(x))
13. Ax(Green(x) & White(x) -> Furry(x))

Derivation:
14. Smart(erin) ; R,6

Conclusion:
Smart(erin)

Final answer: True<|endoftext|>Anne is cold. Anne is furry. Charlie is cold. Charlie is green. Erin is green. Erin is smart. Erin is white. If something is green and furry then it is white. Furry, smart things are big. Big things are smart. If something is furry and green then it is big. If something is cold then it is furry. If something is green and white then it is furry.

True or false: Erin is not smart.

Solution:
Context:
Constants:
anne = Anne
charlie = Charlie
erin = Erin

Predicates:
Big(x) = x is big
Cold(x) = x is cold
Furry(x) = x is furry
Green(x) = x is green
Smart(x) = x is smart
White(x) = x is white

Premises:
1. Cold(anne)
2. Furry(anne)
3. Cold(charlie)
4. Green(charlie)
5. Green(erin)
6. Smart(erin)
7. White(erin)
8. Ax(Green(x) & Furry(x) -> White(x))
9. Ax(Furry(x) & Smart(x) -> Big(x))
10. Ax(Big(x) -> Smart(x))
11. Ax(Furry(x) & Green(x) -> Big(x))
12. Ax(Cold(x) -> Furry(x))
13. Ax(Green(x) & White(x) -> Furry(x))

Derivation:
14. Smart(erin) ; R,6

Conclusion:
Smart(erin)

Final answer: False<|endoftext|>Anne is cold. Anne is furry. Charlie is cold. Charlie is green. Erin is green. Erin is smart. Erin is white. If something is green and furry then it is white. Furry, smart things are big. Big things are smart. If something is furry and green then it is big. If something is cold then it is furry. If something is green and white then it is furry.

True or false: Erin is furry.

Solution:
Context:
Constants:
anne = Anne
charlie = Charlie
erin = Erin

Predicates:
Big(x) = x is big
Cold(x) = x is cold
Furry(x) = x is furry
Green(x) = x is green
Smart(x) = x is smart
White(x) = x is white

Premises:
1. Cold(anne)
2. Furry(anne)
3. Cold(charlie)
4. Green(charlie)
5. Green(erin)
6. Smart(erin)
7. White(erin)
8. Ax(Green(x) & Furry(x) -> White(x))
9. Ax(Furry(x) & Smart(x) -> Big(x))
10. Ax(Big(x) -> Smart(x))
11. Ax(Furry(x) & Green(x) -> Big(x))
12. Ax(Cold(x) -> Furry(x))
13. Ax(Green(x) & White(x) -> Furry(x))

Derivation:
14. Green(erin) ; R,5
15. White(erin) ; R,7
16. Green(erin) & White(erin) ; ∧I,14,15
17. Green(erin) & White(erin) -> Furry(erin) ; AE,13
18. Furry(erin) ; ->E,17,16

Conclusion:
Furry(erin)

Final answer: True<|endoftext|>Anne is cold. Anne is furry. Charlie is cold. Charlie is green. Erin is green. Erin is smart. Erin is white. If something is green and furry then it is white. Furry, smart things are big. Big things are smart. If something is furry and green then it is big. If something is cold then it is furry. If something is green and white then it is furry.

True or false: Erin is not furry.

Solution:
Context:
Constants:
anne = Anne
charlie = Charlie
erin = Erin

Predicates:
Big(x) = x is big
Cold(x) = x is cold
Furry(x) = x is furry
Green(x) = x is green
Smart(x) = x is smart
White(x) = x is white

Premises:
1. Cold(anne)
2. Furry(anne)
3. Cold(charlie)
4. Green(charlie)
5. Green(erin)
6. Smart(erin)
7. White(erin)
8. Ax(Green(x) & Furry(x) -> White(x))
9. Ax(Furry(x) & Smart(x) -> Big(x))
10. Ax(Big(x) -> Smart(x))
11. Ax(Furry(x) & Green(x) -> Big(x))
12. Ax(Cold(x) -> Furry(x))
13. Ax(Green(x) & White(x) -> Furry(x))

Derivation:
14. Green(erin) ; R,5
15. White(erin) ; R,7
16. Green(erin) & White(erin) ; ∧I,14,15
17. Green(erin) & White(erin) -> Furry(erin) ; AE,13
18. Furry(erin) ; ->E,17,16

Conclusion:
Furry(erin)

Final answer: False<|endoftext|>Anne is cold. Anne is furry. Charlie is cold. Charlie is green. Erin is green. Erin is smart. Erin is white. If something is green and furry then it is white. Furry, smart things are big. Big things are smart. If something is furry and green then it is big. If something is cold then it is furry. If something is green and white then it is furry.

True or false: Charlie is white.

Solution:
Context:
Constants:
anne = Anne
charlie = Charlie
erin = Erin

Predicates:
Big(x) = x is big
Cold(x) = x is cold
Furry(x) = x is furry
Green(x) = x is green
Smart(x) = x is smart
White(x) = x is white

Premises:
1. Cold(anne)
2. Furry(anne)
3. Cold(charlie)
4. Green(charlie)
5. Green(erin)
6. Smart(erin)
7. White(erin)
8. Ax(Green(x) & Furry(x) -> White(x))
9. Ax(Furry(x) & Smart(x) -> Big(x))
10. Ax(Big(x) -> Smart(x))
11. Ax(Furry(x) & Green(x) -> Big(x))
12. Ax(Cold(x) -> Furry(x))
13. Ax(Green(x) & White(x) -> Furry(x))

Derivation:
14. Green(charlie) ; R,4
15. Cold(charlie) ; R,3
16. Cold(charlie) -> Furry(charlie) ; AE,12
17. Furry(charlie) ; ->E,16,15
18. Green(charlie) & Furry(charlie) ; ∧I,14,17
19. Green(charlie) & Furry(charlie) -> White(charlie) ; AE,8
20. White(charlie) ; ->E,19,18

Conclusion:
White(charlie)

Final answer: True<|endoftext|>Anne is cold. Anne is furry. Charlie is cold. Charlie is green. Erin is green. Erin is smart. Erin is white. If something is green and furry then it is white. Furry, smart things are big. Big things are smart. If something is furry and green then it is big. If something is cold then it is furry. If something is green and white then it is furry.

True or false: Charlie is not big.

Solution:
Context:
Constants:
anne = Anne
charlie = Charlie
erin = Erin

Predicates:
Big(x) = x is big
Cold(x) = x is cold
Furry(x) = x is furry
Green(x) = x is green
Smart(x) = x is smart
White(x) = x is white

Premises:
1. Cold(anne)
2. Furry(anne)
3. Cold(charlie)
4. Green(charlie)
5. Green(erin)
6. Smart(erin)
7. White(erin)
8. Ax(Green(x) & Furry(x) -> White(x))
9. Ax(Furry(x) & Smart(x) -> Big(x))
10. Ax(Big(x) -> Smart(x))
11. Ax(Furry(x) & Green(x) -> Big(x))
12. Ax(Cold(x) -> Furry(x))
13. Ax(Green(x) & White(x) -> Furry(x))

Derivation:
14. Cold(charlie) ; R,3
15. Cold(charlie) -> Furry(charlie) ; AE,12
16. Furry(charlie) ; ->E,15,14
17. Green(charlie) ; R,4
18. Furry(charlie) & Green(charlie) ; ∧I,16,17
19. Furry(charlie) & Green(charlie) -> Big(charlie) ; AE,11
20. Big(charlie) ; ->E,19,18

Conclusion:
Big(charlie)

Final answer: False<|endoftext|>Anne is cold. Anne is furry. Charlie is cold. Charlie is green. Erin is green. Erin is smart. Erin is white. If something is green and furry then it is white. Furry, smart things are big. Big things are smart. If something is furry and green then it is big. If something is cold then it is furry. If something is green and white then it is furry.

True or false: Charlie is smart.

Solution:
Context:
Constants:
anne = Anne
charlie = Charlie
erin = Erin

Predicates:
Big(x) = x is big
Cold(x) = x is cold
Furry(x) = x is furry
Green(x) = x is green
Smart(x) = x is smart
White(x) = x is white

Premises:
1. Cold(anne)
2. Furry(anne)
3. Cold(charlie)
4. Green(charlie)
5. Green(erin)
6. Smart(erin)
7. White(erin)
8. Ax(Green(x) & Furry(x) -> White(x))
9. Ax(Furry(x) & Smart(x) -> Big(x))
10. Ax(Big(x) -> Smart(x))
11. Ax(Furry(x) & Green(x) -> Big(x))
12. Ax(Cold(x) -> Furry(x))
13. Ax(Green(x) & White(x) -> Furry(x))

Derivation:
14. Cold(charlie) ; R,3
15. Cold(charlie) -> Furry(charlie) ; AE,12
16. Furry(charlie) ; ->E,15,14
17. Green(charlie) ; R,4
18. Furry(charlie) & Green(charlie) ; ∧I,16,17
19. Furry(charlie) & Green(charlie) -> Big(charlie) ; AE,11
20. Big(charlie) ; ->E,19,18
21. Big(charlie) -> Smart(charlie) ; AE,10
22. Smart(charlie) ; ->E,21,20

Conclusion:
Smart(charlie)

Final answer: True<|endoftext|>Anne is cold. Anne is furry. Charlie is cold. Charlie is green. Erin is green. Erin is smart. Erin is white. If something is green and furry then it is white. Furry, smart things are big. Big things are smart. If something is furry and green then it is big. If something is cold then it is furry. If something is green and white then it is furry.

True or false: Charlie is not smart.

Solution:
Context:
Constants:
anne = Anne
charlie = Charlie
erin = Erin

Predicates:
Big(x) = x is big
Cold(x) = x is cold
Furry(x) = x is furry
Green(x) = x is green
Smart(x) = x is smart
White(x) = x is white

Premises:
1. Cold(anne)
2. Furry(anne)
3. Cold(charlie)
4. Green(charlie)
5. Green(erin)
6. Smart(erin)
7. White(erin)
8. Ax(Green(x) & Furry(x) -> White(x))
9. Ax(Furry(x) & Smart(x) -> Big(x))
10. Ax(Big(x) -> Smart(x))
11. Ax(Furry(x) & Green(x) -> Big(x))
12. Ax(Cold(x) -> Furry(x))
13. Ax(Green(x) & White(x) -> Furry(x))

Derivation:
14. Cold(charlie) ; R,3
15. Cold(charlie) -> Furry(charlie) ; AE,12
16. Furry(charlie) ; ->E,15,14
17. Green(charlie) ; R,4
18. Furry(charlie) & Green(charlie) ; ∧I,16,17
19. Furry(charlie) & Green(charlie) -> Big(charlie) ; AE,11
20. Big(charlie) ; ->E,19,18
21. Big(charlie) -> Smart(charlie) ; AE,10
22. Smart(charlie) ; ->E,21,20

Conclusion:
Smart(charlie)

Final answer: False<|endoftext|>The cat is young. If something is red and not young then it is not cold. All young things are not cold. If something is blue then it is green. Green things are blue. If something is red then it is not blue. If something is young and not cold then it is blue.

True or false: The cat is not young.

Solution:
Context:
Constants:
cat = the cat

Predicates:
Blue(x) = x is blue
Cold(x) = x is cold
Green(x) = x is green
Red(x) = x is red
Young(x) = x is young

Premises:
1. Young(cat)
2. Ax(Red(x) & ~Young(x) -> ~Cold(x))
3. Ax(Young(x) -> ~Cold(x))
4. Ax(Blue(x) -> Green(x))
5. Ax(Green(x) -> Blue(x))
6. Ax(Red(x) -> ~Blue(x))
7. Ax(Young(x) & ~Cold(x) -> Blue(x))

Derivation:
8. Young(cat) ; R,1

Conclusion:
Young(cat)

Final answer: False<|endoftext|>
```

## Window 8060 (3 documents, 343 pad tokens)
```
The wolf is reckless. The wolf is slow. The wolf is rough. The wolf sees the rabbit. The tiger needs the cat. The tiger is big. The tiger is heavy. The rabbit is quiet. The rabbit is kind. The rabbit is round. The cat is adorable. The cat is lovely. The cat is small. Quiet animals are adorable. If something is slow then it chases the rabbit. If something chases the rabbit then it is tired. If something is reckless and slow then it is rough. If something is adorable and lovely then it is cute. If something is big and heavy then it is strong. If something is rough then it is dull. If something is dull then it is boring. If something is boring then it is heavy. All heavy animals are big. If something is adorable then it is lovely. If something is lovely then it is small. If something is small then it is round. If something is round then it is smart. All small animals are nice. If something is strong then it is obese. All obese animals are fierce. If something is fierce then it is slow. All slow animals are angry. If something is cute then it is furry. All furry animals are funny. If something is funny then it is reckless. All reckless animals are beautiful. If something is tired then it is awful. If something is awful then it is lazy. All lazy animals are sleepy.

True or false: The cat is not beautiful.

Solution:
Context:
Constants:
cat = the cat
rabbit = the rabbit
tiger = the tiger
wolf = the wolf

Predicates:
Adorable(x) = x is adorable
Angry(x) = x is angry
Awful(x) = x is awful
Beautiful(x) = x is beautiful
Big(x) = x is big
Boring(x) = x is boring
Chases(x,y) = x chases y
Cute(x) = x is cute
Dull(x) = x is dull
Fierce(x) = x is fierce
Funny(x) = x is funny
Furry(x) = x is furry
Heavy(x) = x is heavy
Kind(x) = x is kind
Lazy(x) = x is lazy
Lovely(x) = x is lovely
Needs(x,y) = x needs y
Nice(x) = x is nice
Obese(x) = x is obese
Quiet(x) = x is quiet
Reckless(x) = x is reckless
Rough(x) = x is rough
Round(x) = x is round
Sees(x,y) = x sees y
Sleepy(x) = x is sleepy
Slow(x) = x is slow
Small(x) = x is small
Smart(x) = x is smart
Strong(x) = x is strong
Tired(x) = x is tired

Premises:
1. Reckless(wolf)
2. Slow(wolf)
3. Rough(wolf)
4. Sees(wolf,rabbit)
5. Needs(tiger,cat)
6. Big(tiger)
7. Heavy(tiger)
8. Quiet(rabbit)
9. Kind(rabbit)
10. Round(rabbit)
11. Adorable(cat)
12. Lovely(cat)
13. Small(cat)
14. Ax(Quiet(x) -> Adorable(x))
15. Ax(Slow(x) -> Chases(x,rabbit))
16. Ax(Chases(x,rabbit) -> Tired(x))
17. Ax(Reckless(x) & Slow(x) -> Rough(x))
18. Ax(Adorable(x) & Lovely(x) -> Cute(x))
19. Ax(Big(x) & Heavy(x) -> Strong(x))
20. Ax(Rough(x) -> Dull(x))
21. Ax(Dull(x) -> Boring(x))
22. Ax(Boring(x) -> Heavy(x))
23. Ax(Heavy(x) -> Big(x))
24. Ax(Adorable(x) -> Lovely(x))
25. Ax(Lovely(x) -> Small(x))
26. Ax(Small(x) -> Round(x))
27. Ax(Round(x) -> Smart(x))
28. Ax(Small(x) -> Nice(x))
29. Ax(Strong(x) -> Obese(x))
30. Ax(Obese(x) -> Fierce(x))
31. Ax(Fierce(x) -> Slow(x))
32. Ax(Slow(x) -> Angry(x))
33. Ax(Cute(x) -> Furry(x))
34. Ax(Furry(x) -> Funny(x))
35. Ax(Funny(x) -> Reckless(x))
36. Ax(Reckless(x) -> Beautiful(x))
37. Ax(Tired(x) -> Awful(x))
38. Ax(Awful(x) -> Lazy(x))
39. Ax(Lazy(x) -> Sleepy(x))

Derivation:
40. Adorable(cat) ; R,11
41. Lovely(cat) ; R,12
42. Adorable(cat) & Lovely(cat) ; ∧I,40,41
43. Adorable(cat) & Lovely(cat) -> Cute(cat) ; AE,18
44. Cute(cat) ; ->E,43,42
45. Cute(cat) -> Furry(cat) ; AE,33
46. Furry(cat) ; ->E,45,44
47. Furry(cat) -> Funny(cat) ; AE,34
48. Funny(cat) ; ->E,47,46
49. Funny(cat) -> Reckless(cat) ; AE,35
50. Reckless(cat) ; ->E,49,48
51. Reckless(cat) -> Beautiful(cat) ; AE,36
52. Beautiful(cat) ; ->E,51,50

Conclusion:
Beautiful(cat)

Final answer: False<|endoftext|>The wolf is reckless. The wolf is slow. The wolf is rough. The wolf sees the rabbit. The tiger needs the cat. The tiger is big. The tiger is heavy. The rabbit is quiet. The rabbit is kind. The rabbit is round. The cat is adorable. The cat is lovely. The cat is small. Quiet animals are adorable. If something is slow then it chases the rabbit. If something chases the rabbit then it is tired. If something is reckless and slow then it is rough. If something is adorable and lovely then it is cute. If something is big and heavy then it is strong. If something is rough then it is dull. If something is dull then it is boring. If something is boring then it is heavy. All heavy animals are big. If something is adorable then it is lovely. If something is lovely then it is small. If something is small then it is round. If something is round then it is smart. All small animals are nice. If something is strong then it is obese. All obese animals are fierce. If something is fierce then it is slow. All slow animals are angry. If something is cute then it is furry. All furry animals are funny. If something is funny then it is reckless. All reckless animals are beautiful. If something is tired then it is awful. If something is awful then it is lazy. All lazy animals are sleepy.

True or false: The wolf is sleepy.

Solution:
Context:
Constants:
cat = the cat
rabbit = the rabbit
tiger = the tiger
wolf = the wolf

Predicates:
Adorable(x) = x is adorable
Angry(x) = x is angry
Awful(x) = x is awful
Beautiful(x) = x is beautiful
Big(x) = x is big
Boring(x) = x is boring
Chases(x,y) = x chases y
Cute(x) = x is cute
Dull(x) = x is dull
Fierce(x) = x is fierce
Funny(x) = x is funny
Furry(x) = x is furry
Heavy(x) = x is heavy
Kind(x) = x is kind
Lazy(x) = x is lazy
Lovely(x) = x is lovely
Needs(x,y) = x needs y
Nice(x) = x is nice
Obese(x) = x is obese
Quiet(x) = x is quiet
Reckless(x) = x is reckless
Rough(x) = x is rough
Round(x) = x is round
Sees(x,y) = x sees y
Sleepy(x) = x is sleepy
Slow(x) = x is slow
Small(x) = x is small
Smart(x) = x is smart
Strong(x) = x is strong
Tired(x) = x is tired

Premises:
1. Reckless(wolf)
2. Slow(wolf)
3. Rough(wolf)
4. Sees(wolf,rabbit)
5. Needs(tiger,cat)
6. Big(tiger)
7. Heavy(tiger)
8. Quiet(rabbit)
9. Kind(rabbit)
10. Round(rabbit)
11. Adorable(cat)
12. Lovely(cat)
13. Small(cat)
14. Ax(Quiet(x) -> Adorable(x))
15. Ax(Slow(x) -> Chases(x,rabbit))
16. Ax(Chases(x,rabbit) -> Tired(x))
17. Ax(Reckless(x) & Slow(x) -> Rough(x))
18. Ax(Adorable(x) & Lovely(x) -> Cute(x))
19. Ax(Big(x) & Heavy(x) -> Strong(x))
20. Ax(Rough(x) -> Dull(x))
21. Ax(Dull(x) -> Boring(x))
22. Ax(Boring(x) -> Heavy(x))
23. Ax(Heavy(x) -> Big(x))
24. Ax(Adorable(x) -> Lovely(x))
25. Ax(Lovely(x) -> Small(x))
26. Ax(Small(x) -> Round(x))
27. Ax(Round(x) -> Smart(x))
28. Ax(Small(x) -> Nice(x))
29. Ax(Strong(x) -> Obese(x))
30. Ax(Obese(x) -> Fierce(x))
31. Ax(Fierce(x) -> Slow(x))
32. Ax(Slow(x) -> Angry(x))
33. Ax(Cute(x) -> Furry(x))
34. Ax(Furry(x) -> Funny(x))
35. Ax(Funny(x) -> Reckless(x))
36. Ax(Reckless(x) -> Beautiful(x))
37. Ax(Tired(x) -> Awful(x))
38. Ax(Awful(x) -> Lazy(x))
39. Ax(Lazy(x) -> Sleepy(x))

Derivation:
40. Slow(wolf) ; R,2
41. Slow(wolf) -> Chases(wolf,rabbit) ; AE,15
42. Chases(wolf,rabbit) ; ->E,41,40
43. Chases(wolf,rabbit) -> Tired(wolf) ; AE,16
44. Tired(wolf) ; ->E,43,42
45. Tired(wolf) -> Awful(wolf) ; AE,37
46. Awful(wolf) ; ->E,45,44
47. Awful(wolf) -> Lazy(wolf) ; AE,38
48. Lazy(wolf) ; ->E,47,46
49. Lazy(wolf) -> Sleepy(wolf) ; AE,39
50. Sleepy(wolf) ; ->E,49,48

Conclusion:
Sleepy(wolf)

Final answer: True<|endoftext|>The wolf is reckless. The wolf is slow. The wolf is rough. The wolf sees the rabbit. The tiger needs the cat. The tiger is big. The tiger is heavy. The rabbit is quiet. The rabbit is kind. The rabbit is round. The cat is adorable. The cat is lovely. The cat is small. Quiet animals are adorable. If something is slow then it chases the rabbit. If something chases the rabbit then it is tired. If something is reckless and slow then it is rough. If something is adorable and lovely then it is cute. If something is big and heavy then it is strong. If something is rough then it is dull. If something is dull then it is boring. If something is boring then it is heavy. All heavy animals are big. If something is adorable then it is lovely. If something is lovely then it is small. If something is small then it is round. If something is round then it is smart. All small animals are nice. If something is strong then it is obese. All obese animals are fierce. If something is fierce then it is slow. All slow animals are angry. If something is cute then it is furry. All furry animals are funny. If something is funny then it is reckless. All reckless animals are beautiful. If something is tired then it is awful. If something is awful then it is lazy. All lazy animals are sleepy.

True or false: The wolf is not sleepy.

Solution:
Context:
Constants:
cat = the cat
rabbit = the rabbit
tiger = the tiger
wolf = the wolf

Predicates:
Adorable(x) = x is adorable
Angry(x) = x is angry
Awful(x) = x is awful
Beautiful(x) = x is beautiful
Big(x) = x is big
Boring(x) = x is boring
Chases(x,y) = x chases y
Cute(x) = x is cute
Dull(x) = x is dull
Fierce(x) = x is fierce
Funny(x) = x is funny
Furry(x) = x is furry
Heavy(x) = x is heavy
Kind(x) = x is kind
Lazy(x) = x is lazy
Lovely(x) = x is lovely
Needs(x,y) = x needs y
Nice(x) = x is nice
Obese(x) = x is obese
Quiet(x) = x is quiet
Reckless(x) = x is reckless
Rough(x) = x is rough
Round(x) = x is round
Sees(x,y) = x sees y
Sleepy(x) = x is sleepy
Slow(x) = x is slow
Small(x) = x is small
Smart(x) = x is smart
Strong(x) = x is strong
Tired(x) = x is tired

Premises:
1. Reckless(wolf)
2. Slow(wolf)
3. Rough(wolf)
4. Sees(wolf,rabbit)
5. Needs(tiger,cat)
6. Big(tiger)
7. Heavy(tiger)
8. Quiet(rabbit)
9. Kind(rabbit)
10. Round(rabbit)
11. Adorable(cat)
12. Lovely(cat)
13. Small(cat)
14. Ax(Quiet(x) -> Adorable(x))
15. Ax(Slow(x) -> Chases(x,rabbit))
16. Ax(Chases(x,rabbit) -> Tired(x))
17. Ax(Reckless(x) & Slow(x) -> Rough(x))
18. Ax(Adorable(x) & Lovely(x) -> Cute(x))
19. Ax(Big(x) & Heavy(x) -> Strong(x))
20. Ax(Rough(x) -> Dull(x))
21. Ax(Dull(x) -> Boring(x))
22. Ax(Boring(x) -> Heavy(x))
23. Ax(Heavy(x) -> Big(x))
24. Ax(Adorable(x) -> Lovely(x))
25. Ax(Lovely(x) -> Small(x))
26. Ax(Small(x) -> Round(x))
27. Ax(Round(x) -> Smart(x))
28. Ax(Small(x) -> Nice(x))
29. Ax(Strong(x) -> Obese(x))
30. Ax(Obese(x) -> Fierce(x))
31. Ax(Fierce(x) -> Slow(x))
32. Ax(Slow(x) -> Angry(x))
33. Ax(Cute(x) -> Furry(x))
34. Ax(Furry(x) -> Funny(x))
35. Ax(Funny(x) -> Reckless(x))
36. Ax(Reckless(x) -> Beautiful(x))
37. Ax(Tired(x) -> Awful(x))
38. Ax(Awful(x) -> Lazy(x))
39. Ax(Lazy(x) -> Sleepy(x))

Derivation:
40. Slow(wolf) ; R,2
41. Slow(wolf) -> Chases(wolf,rabbit) ; AE,15
42. Chases(wolf,rabbit) ; ->E,41,40
43. Chases(wolf,rabbit) -> Tired(wolf) ; AE,16
44. Tired(wolf) ; ->E,43,42
45. Tired(wolf) -> Awful(wolf) ; AE,37
46. Awful(wolf) ; ->E,45,44
47. Awful(wolf) -> Lazy(wolf) ; AE,38
48. Lazy(wolf) ; ->E,47,46
49. Lazy(wolf) -> Sleepy(wolf) ; AE,39
50. Sleepy(wolf) ; ->E,49,48

Conclusion:
Sleepy(wolf)

Final answer: False<|endoftext|>
```

## Window 8222 summary: [{"tokens": 200, "head": "The dog is cold. The dog is kind. The dog is red. The dog is round. The dog is y", "tail": "ound(dog) ; R,4  Conclusion: Round(dog)  Final answer: False"}, {"tokens": 489, "head": "The bear chases the mouse. The bear is blue. The bear is cold. The bear is red. ", "tail": "ung(bear) ; R,6  Conclusion: Young(bear)  Final answer: True"}, {"tokens": 497, "head": "The bear chases the mouse. The bear is blue. The bear is cold. The bear is red. ", "tail": " ; R,12  Conclusion: Visits(mouse,bear)  Final answer: False"}, {"tokens": 171, "head": "Fiona is round. If someone is cold and not round then they are white. Smart, qui", "tail": "d(fiona) ; R,1  Conclusion: Round(fiona)  Final answer: True"}, {"tokens": 172, "head": "Fiona is round. If someone is cold and not round then they are white. Smart, qui", "tail": "(fiona) ; R,1  Conclusion: Round(fiona)  Final answer: False"}, {"tokens": 262, "head": "The rabbit is blue. If the rabbit is green and the rabbit is not red then the ra", "tail": "(rabbit) ; R,1  Conclusion: Blue(rabbit)  Final answer: True"}, {"tokens": 263, "head": "The rabbit is blue. If the rabbit is green and the rabbit is not red then the ra", "tail": "rabbit) ; R,1  Conclusion: Blue(rabbit)  Final answer: False"}, {"tokens": 238, "head": "Gary is big. Gary is cold. Gary is kind. Gary is red. Gary is rough. Gary is whi", "tail": "ung(gary) ; R,7  Conclusion: Young(gary)  Final answer: True"}, {"tokens": 240, "head": "Gary is big. Gary is cold. Gary is kind. Gary is red. Gary is rough. Gary is whi", "tail": "gh(gary) ; R,5  Conclusion: Rough(gary)  Final answer: False"}, {"tokens": 301, "head": "Bob is blue. Bob is kind. Bob is nice. Bob is red. Bob is rough. Bob is round. B", "tail": "Blue(erin) ; R,8  Conclusion: Blue(erin)  Final answer: True"}, {"tokens": 303, "head": "Bob is blue. Bob is kind. Bob is nice. Bob is red. Bob is rough. Bob is round. B", "tail": "Red(erin) ; R,11  Conclusion: Red(erin)  Final answer: False"}, {"tokens": 333, "head": "Anne is red. Charlie is red. Erin is red. If Anne is red and Anne is smart then ", "tail": ". Red(erin) ; R,3  Conclusion: Red(erin)  Final answer: True"}, {"tokens": 334, "head": "Anne is red. Charlie is red. Erin is red. If Anne is red and Anne is smart then ", "tail": " Red(anne) ; R,1  Conclusion: Red(anne)  Final answer: False"}, {"tokens": 225, "head": "Bob is not big. Bob is blue. Bob is green. Bob is not kind. Fiona is blue. Fiona", "tail": "n(fiona) ; R,6  Conclusion: Green(fiona)  Final answer: True"}]

## Window 9449 summary: [{"tokens": 1011, "head": "Harry is heavy. Harry is big. Harry is strong. Dave is tiny. Dave is thin. Gary ", "tail": ") ; ->E,43,42  Conclusion: Strong(gary)  Final answer: False"}, {"tokens": 1011, "head": "Harry is heavy. Harry is big. Harry is strong. Dave is tiny. Dave is thin. Gary ", "tail": "rin) ; ->E,43,42  Conclusion: Tiny(erin)  Final answer: True"}, {"tokens": 1012, "head": "Harry is heavy. Harry is big. Harry is strong. Dave is tiny. Dave is thin. Gary ", "tail": "in) ; ->E,43,42  Conclusion: Tiny(erin)  Final answer: False"}, {"tokens": 965, "head": "Harry is heavy. Harry is huge. Harry is strong. Gary is small. Gary is thin. Ala", "tail": "rry) ; ->E,41,40  Conclusion: Big(harry)  Final answer: True"}]

## Window 21580 summary: [{"tokens": 292, "head": "Charlie is not green. Dave is rough. Gary is round. Blue, quiet people are green", "tail": "nd(gary) ; R,3  Conclusion: Round(gary)  Final answer: False"}, {"tokens": 330, "head": "Gary is big. Gary is not green. Gary is kind. Gary is not quiet. Gary is rough. ", "tail": ". Big(gary) ; R,1  Conclusion: Big(gary)  Final answer: True"}, {"tokens": 331, "head": "Gary is big. Gary is not green. Gary is kind. Gary is not quiet. Gary is rough. ", "tail": "nd(gary) ; R,6  Conclusion: Round(gary)  Final answer: False"}, {"tokens": 287, "head": "Anne is furry. Anne is kind. Anne is nice. Anne is smart. Anne is young. Charlie", "tail": "art(gary) ; R,9  Conclusion: Smart(gary)  Final answer: True"}, {"tokens": 288, "head": "Anne is furry. Anne is kind. Anne is nice. Anne is smart. Anne is young. Charlie", "tail": "ice(anne) ; R,3  Conclusion: Nice(anne)  Final answer: False"}, {"tokens": 168, "head": "The cat is young. The cow sees the squirrel. The squirrel is cold. If something ", "tail": "Young(cat) ; R,1  Conclusion: Young(cat)  Final answer: True"}, {"tokens": 179, "head": "The cat is young. The cow sees the squirrel. The squirrel is cold. If something ", "tail": ") ; R,2  Conclusion: Sees(cow,squirrel)  Final answer: False"}, {"tokens": 266, "head": "The cat chases the cow. The cow is young. The squirrel needs the tiger. The tige", "tail": "Young(cow) ; R,2  Conclusion: Young(cow)  Final answer: True"}, {"tokens": 267, "head": "The cat chases the cow. The cow is young. The squirrel needs the tiger. The tige", "tail": "oung(cow) ; R,2  Conclusion: Young(cow)  Final answer: False"}, {"tokens": 215, "head": "Anne is blue. Anne is nice. Anne is quiet. Anne is red. Fiona is blue. Fiona is ", "tail": "ce(fiona) ; R,6  Conclusion: Nice(fiona)  Final answer: True"}, {"tokens": 217, "head": "Anne is blue. Anne is nice. Anne is quiet. Anne is red. Fiona is blue. Fiona is ", "tail": "(fiona) ; R,7  Conclusion: Rough(fiona)  Final answer: False"}, {"tokens": 213, "head": "Dave is smart. Harry is not big. All smart people are cold. If Dave is cold and ", "tail": "ig(harry) ; R,2  Conclusion: ~Big(harry)  Final answer: True"}, {"tokens": 211, "head": "Dave is smart. Harry is not big. All smart people are cold. If Dave is cold and ", "tail": "rt(dave) ; R,1  Conclusion: Smart(dave)  Final answer: False"}, {"tokens": 460, "head": "The lion chases the mouse. The lion chases the tiger. The lion is cold. The lion", "tail": "Cold(lion) ; R,3  Conclusion: Cold(lion)  Final answer: True"}, {"tokens": 360, "head": "The cow is round. The rabbit does not like the tiger. The rabbit needs the squir", "tail": "rrel) ; R,8  Conclusion: Round(squirrel)  Final answer: True"}]

## Window 22267 summary: [{"tokens": 600, "head": "Charlie is blue. Dave is blue. Dave is red. Dave is young. Fiona is big. Fiona i", "tail": "ry) ; ->E,27,26  Conclusion: ~Red(gary)  Final answer: False"}, {"tokens": 626, "head": "The bear is blue. The bear is rough. The bear is young. The bear needs the lion.", "tail": "ugh(bear) ; R,2  Conclusion: Rough(bear)  Final answer: True"}, {"tokens": 626, "head": "The bear is blue. The bear is rough. The bear is young. The bear needs the lion.", "tail": "lue(bear) ; R,1  Conclusion: Blue(bear)  Final answer: False"}, {"tokens": 667, "head": "The bear is blue. The bear is rough. The bear is young. The bear needs the lion.", "tail": " ->E,25,24  Conclusion: Needs(bear,bear)  Final answer: True"}, {"tokens": 676, "head": "The bear is blue. The bear is rough. The bear is young. The bear needs the lion.", "tail": ",25,24  Conclusion: Sees(squirrel,bear)  Final answer: False"}, {"tokens": 762, "head": "The bear is blue. The bear is rough. The bear is young. The bear needs the lion.", "tail": ",28  Conclusion: Sees(squirrel,squirrel)  Final answer: True"}]

## Window 23691 summary: [{"tokens": 572, "head": "Charlie is furry. Charlie is green. Erin is white. Fiona is big. Fiona is green.", "tail": "in) ; ->E,26,25  Conclusion: Quiet(erin)  Final answer: True"}, {"tokens": 573, "head": "Charlie is furry. Charlie is green. Erin is white. Fiona is big. Fiona is green.", "tail": "n) ; ->E,26,25  Conclusion: Quiet(erin)  Final answer: False"}, {"tokens": 563, "head": "The bald eagle eats the dog. The bald eagle likes the bear. The bald eagle sees ", "tail": "; R,3  Conclusion: Sees(bald_eagle,bear)  Final answer: True"}, {"tokens": 563, "head": "The bald eagle eats the dog. The bald eagle likes the bear. The bald eagle sees ", "tail": "R,2  Conclusion: Likes(bald_eagle,bear)  Final answer: False"}, {"tokens": 606, "head": "The bald eagle eats the dog. The bald eagle likes the bear. The bald eagle sees ", "tail": "->E,19,18  Conclusion: Eats(bear,rabbit)  Final answer: True"}, {"tokens": 608, "head": "The bald eagle eats the dog. The bald eagle likes the bear. The bald eagle sees ", "tail": ">E,19,18  Conclusion: Eats(bear,rabbit)  Final answer: False"}, {"tokens": 431, "head": "Bob is big. Bob is green. Dave is big. Erin is nice. Fiona is big. Fiona is nice", "tail": "na) ; ->E,19,18  Conclusion: Blue(fiona)  Final answer: True"}]

## Window 23706 summary: [{"tokens": 483, "head": "The cat chases the lion. The cat chases the mouse. The cat chases the rabbit. Th", "tail": "on) ; R,1  Conclusion: Chases(cat,lion)  Final answer: False"}, {"tokens": 353, "head": "Dave is big. Dave is blue. Dave is green. Dave is kind. Dave is not round. Dave ", "tail": "ung(dave) ; R,7  Conclusion: Young(dave)  Final answer: True"}, {"tokens": 354, "head": "Dave is big. Dave is blue. Dave is green. Dave is kind. Dave is not round. Dave ", "tail": " Big(dave) ; R,1  Conclusion: Big(dave)  Final answer: False"}, {"tokens": 239, "head": "Anne is big. Anne is cold. Anne is quiet. Anne is red. Bob is red. Bob is smart.", "tail": "et(anne) ; R,3  Conclusion: Quiet(anne)  Final answer: False"}, {"tokens": 421, "head": "Charlie is not green. Charlie is kind. Charlie is not nice. Charlie is not red. ", "tail": "y(harry) ; R,7  Conclusion: Furry(harry)  Final answer: True"}, {"tokens": 420, "head": "Charlie is not green. Charlie is kind. Charlie is not nice. Charlie is not red. ", "tail": "d(harry) ; R,9  Conclusion: Kind(harry)  Final answer: False"}, {"tokens": 299, "head": "The cat does not chase the dog. The cat is not nice. The cat is red. The cat see", "tail": "~Nice(cat) ; R,2  Conclusion: ~Nice(cat)  Final answer: True"}, {"tokens": 308, "head": "The cat does not chase the dog. The cat is not nice. The cat is red. The cat see", "tail": "at) ; R,10  Conclusion: Visits(dog,cat)  Final answer: False"}, {"tokens": 314, "head": "Erin is blue. Erin is nice. Erin is red. Erin is rough. Erin is not round. Erin ", "tail": "ce(harry) ; R,8  Conclusion: Nice(harry)  Final answer: True"}, {"tokens": 315, "head": "Erin is blue. Erin is nice. Erin is red. Erin is rough. Erin is not round. Erin ", "tail": "ed(harry) ; R,9  Conclusion: Red(harry)  Final answer: False"}, {"tokens": 265, "head": "Anne is kind. Anne is not red. Anne is smart. Anne is young. Dave is round. Harr", "tail": "h(harry) ; R,9  Conclusion: Rough(harry)  Final answer: True"}, {"tokens": 265, "head": "Anne is kind. Anne is not red. Anne is smart. Anne is young. Dave is round. Harr", "tail": "rt(anne) ; R,3  Conclusion: Smart(anne)  Final answer: False"}]

## Window 32399 summary: [{"tokens": 338, "head": "Anne is quiet. Fiona is not green. If Anne is quiet and Anne is green then Anne ", "tail": "e) ; ->E,12,11  Conclusion: Green(anne)  Final answer: False"}, {"tokens": 377, "head": "Anne is quiet. Fiona is not green. If Anne is quiet and Anne is green then Anne ", "tail": "anne) ; ->E,3,14  Conclusion: ~Red(anne)  Final answer: True"}, {"tokens": 376, "head": "Anne is quiet. Fiona is not green. If Anne is quiet and Anne is green then Anne ", "tail": "nne) ; ->E,3,14  Conclusion: ~Red(anne)  Final answer: False"}, {"tokens": 308, "head": "Charlie is round. Gary is cold. Gary is quiet. Quiet, round people are red. Quie", "tail": "arlie) ; R,1  Conclusion: Round(charlie)  Final answer: True"}, {"tokens": 309, "head": "Charlie is round. Gary is cold. Gary is quiet. Quiet, round people are red. Quie", "tail": "rlie) ; R,1  Conclusion: Round(charlie)  Final answer: False"}, {"tokens": 343, "head": "Charlie is round. Gary is cold. Gary is quiet. Quiet, round people are red. Quie", "tail": "ary) ; ->E,13,12  Conclusion: Kind(gary)  Final answer: True"}, {"tokens": 344, "head": "Charlie is round. Gary is cold. Gary is quiet. Quiet, round people are red. Quie", "tail": "ry) ; ->E,13,12  Conclusion: Kind(gary)  Final answer: False"}, {"tokens": 415, "head": "Charlie is round. Gary is cold. Gary is quiet. Quiet, round people are red. Quie", "tail": "gary) ; ->E,17,16  Conclusion: Red(gary)  Final answer: True"}, {"tokens": 416, "head": "Charlie is round. Gary is cold. Gary is quiet. Quiet, round people are red. Quie", "tail": "ary) ; ->E,17,16  Conclusion: Red(gary)  Final answer: False"}, {"tokens": 449, "head": "Charlie is round. Gary is cold. Gary is quiet. Quiet, round people are red. Quie", "tail": "gary) ; ->E,19,18  Conclusion: Big(gary)  Final answer: True"}, {"tokens": 394, "head": "Anne is smart. Dave is young. Gary is not cold. Gary is not rough. Harry is roun", "tail": "d(harry) ; R,5  Conclusion: Round(harry)  Final answer: True"}]

## Window 33950 summary: [{"tokens": 600, "head": "Dave is huge. Dave is strong. Dave is high. Gary is short. Gary is little. Bob i", "tail": "ary) ; ->E,25,24  Conclusion: Thin(gary)  Final answer: True"}, {"tokens": 601, "head": "Dave is huge. Dave is strong. Dave is high. Gary is short. Gary is little. Bob i", "tail": "ry) ; ->E,25,24  Conclusion: Thin(gary)  Final answer: False"}, {"tokens": 603, "head": "Dave is huge. Dave is strong. Dave is high. Gary is short. Gary is little. Bob i", "tail": "bob) ; ->E,25,24  Conclusion: Quiet(bob)  Final answer: True"}, {"tokens": 604, "head": "Dave is huge. Dave is strong. Dave is high. Gary is short. Gary is little. Bob i", "tail": "ob) ; ->E,25,24  Conclusion: Quiet(bob)  Final answer: False"}, {"tokens": 602, "head": "Dave is huge. Dave is strong. Dave is high. Gary is short. Gary is little. Bob i", "tail": " ; ->E,25,24  Conclusion: Rough(charlie)  Final answer: True"}, {"tokens": 603, "head": "Dave is huge. Dave is strong. Dave is high. Gary is short. Gary is little. Bob i", "tail": "; ->E,25,24  Conclusion: Rough(charlie)  Final answer: False"}]

## Window 34086 summary: [{"tokens": 857, "head": "The leopard is slow. The leopard is dull. The leopard is tired. The leopard sees", "tail": "cat) ; ->E,34,33  Conclusion: Furry(cat)  Final answer: True"}, {"tokens": 858, "head": "The leopard is slow. The leopard is dull. The leopard is tired. The leopard sees", "tail": "at) ; ->E,34,33  Conclusion: Furry(cat)  Final answer: False"}, {"tokens": 835, "head": "The leopard is slow. The leopard is dull. The leopard is tired. The leopard sees", "tail": "; ->E,32,31  Conclusion: Sleepy(leopard)  Final answer: True"}, {"tokens": 836, "head": "The leopard is slow. The leopard is dull. The leopard is tired. The leopard sees", "tail": " ->E,32,31  Conclusion: Sleepy(leopard)  Final answer: False"}, {"tokens": 710, "head": "Anne is high. Anne is huge. Anne is strong. Gary is short. Gary is small. Erin i", "tail": "e) ; ->E,29,28  Conclusion: Clever(anne)  Final answer: True"}]

## Window 36271 summary: [{"tokens": 1099, "head": "The bald eagle is rough. The bald eagle is lazy. The bald eagle is tired. The ba", "tail": "(cat) ; ->E,41,40  Conclusion: Kind(cat)  Final answer: True"}, {"tokens": 1100, "head": "The bald eagle is rough. The bald eagle is lazy. The bald eagle is tired. The ba", "tail": "cat) ; ->E,41,40  Conclusion: Kind(cat)  Final answer: False"}, {"tokens": 1248, "head": "The bald eagle is rough. The bald eagle is lazy. The bald eagle is tired. The ba", "tail": "->E,47,46  Conclusion: Heavy(bald_eagle)  Final answer: True"}]

## Window 39248 summary: [{"tokens": 562, "head": "The cow sees the rabbit. The cow sees the squirrel. The rabbit chases the cow. T", "tail": "E,15,18  Conclusion: Needs(cow,squirrel)  Final answer: True"}, {"tokens": 564, "head": "The cow sees the rabbit. The cow sees the squirrel. The rabbit chases the cow. T", "tail": ",15,18  Conclusion: Needs(cow,squirrel)  Final answer: False"}, {"tokens": 646, "head": "The cow sees the rabbit. The cow sees the squirrel. The rabbit chases the cow. T", "tail": "(cow) ; ->E,22,21  Conclusion: Cold(cow)  Final answer: True"}, {"tokens": 647, "head": "The cow sees the rabbit. The cow sees the squirrel. The rabbit chases the cow. T", "tail": "cow) ; ->E,22,21  Conclusion: Cold(cow)  Final answer: False"}, {"tokens": 421, "head": "Anne is cold. Anne is kind. Anne is smart. Charlie is big. Charlie is cold. Char", "tail": "Cold(anne) ; R,1  Conclusion: Cold(anne)  Final answer: True"}, {"tokens": 423, "head": "Anne is cold. Anne is kind. Anne is smart. Charlie is big. Charlie is cold. Char", "tail": "g(harry) ; R,13  Conclusion: Big(harry)  Final answer: False"}, {"tokens": 457, "head": "Anne is cold. Anne is kind. Anne is smart. Charlie is big. Charlie is cold. Char", "tail": "y) ; ->E,23,22  Conclusion: Green(harry)  Final answer: True"}, {"tokens": 301, "head": "Charlie is blue. Fiona is rough. Harry is rough. If Charlie is blue then Charlie", "tail": ") ; ->E,4,10  Conclusion: Rough(charlie)  Final answer: True"}]

## Window 39249 summary: [{"tokens": 702, "head": "The leopard is rough. The leopard is lazy. The leopard is sleepy. The leopard li", "tail": ") ; ->E,27,26  Conclusion: Dull(leopard)  Final answer: True"}, {"tokens": 703, "head": "The leopard is rough. The leopard is lazy. The leopard is sleepy. The leopard li", "tail": " ; ->E,27,26  Conclusion: Dull(leopard)  Final answer: False"}, {"tokens": 600, "head": "Bob is huge. Bob is high. Bob is big. Charlie is short. Charlie is little. Harry", "tail": ") ; ->E,25,24  Conclusion: Thin(charlie)  Final answer: True"}, {"tokens": 601, "head": "Bob is huge. Bob is high. Bob is big. Charlie is short. Charlie is little. Harry", "tail": " ; ->E,25,24  Conclusion: Thin(charlie)  Final answer: False"}, {"tokens": 603, "head": "Bob is huge. Bob is high. Bob is big. Charlie is short. Charlie is little. Harry", "tail": "y) ; ->E,25,24  Conclusion: Quiet(harry)  Final answer: True"}, {"tokens": 604, "head": "Bob is huge. Bob is high. Bob is big. Charlie is short. Charlie is little. Harry", "tail": ") ; ->E,25,24  Conclusion: Quiet(harry)  Final answer: False"}]

## Window 40684 summary: [{"tokens": 695, "head": "The wolf is lazy. The wolf is sleepy. The wolf is rough. The wolf visits the squ", "tail": " ->E,27,26  Conclusion: Lovely(squirrel)  Final answer: True"}, {"tokens": 696, "head": "The wolf is lazy. The wolf is sleepy. The wolf is rough. The wolf visits the squ", "tail": "->E,27,26  Conclusion: Lovely(squirrel)  Final answer: False"}, {"tokens": 662, "head": "The wolf is lazy. The wolf is sleepy. The wolf is rough. The wolf visits the squ", "tail": "olf) ; ->E,25,24  Conclusion: Dull(wolf)  Final answer: True"}, {"tokens": 663, "head": "The wolf is lazy. The wolf is sleepy. The wolf is rough. The wolf visits the squ", "tail": "lf) ; ->E,25,24  Conclusion: Dull(wolf)  Final answer: False"}, {"tokens": 739, "head": "The wolf is lazy. The wolf is sleepy. The wolf is rough. The wolf visits the squ", "tail": "r) ; ->E,29,28  Conclusion: Fierce(bear)  Final answer: True"}, {"tokens": 565, "head": "Harry is big. Harry is huge. Harry is high. Erin is short. Erin is thin. Anne is", "tail": " ; ->E,23,22  Conclusion: Wealthy(harry)  Final answer: True"}]

## Window 41239 summary: [{"tokens": 751, "head": "Charlie is huge. Charlie is big. Charlie is high. Gary is tiny. Gary is thin. An", "tail": "nne) ; ->E,31,30  Conclusion: Nice(anne)  Final answer: True"}, {"tokens": 752, "head": "Charlie is huge. Charlie is big. Charlie is high. Gary is tiny. Gary is thin. An", "tail": "ne) ; ->E,31,30  Conclusion: Nice(anne)  Final answer: False"}, {"tokens": 755, "head": "Charlie is huge. Charlie is big. Charlie is high. Gary is tiny. Gary is thin. An", "tail": "rin) ; ->E,31,30  Conclusion: Dull(erin)  Final answer: True"}, {"tokens": 756, "head": "Charlie is huge. Charlie is big. Charlie is high. Gary is tiny. Gary is thin. An", "tail": "in) ; ->E,31,30  Conclusion: Dull(erin)  Final answer: False"}, {"tokens": 714, "head": "Erin is high. Erin is big. Erin is huge. Gary is little. Gary is short. Bob is k", "tail": "rin) ; ->E,29,28  Conclusion: Nice(erin)  Final answer: True"}]

## Window 49398 summary: [{"tokens": 341, "head": "The bald eagle is nice. The cat sees the dog. The dog is big. If someone likes t", "tail": "0. Big(dog) ; R,3  Conclusion: Big(dog)  Final answer: False"}, {"tokens": 391, "head": "The bald eagle is nice. The cat sees the dog. The dog is big. If someone likes t", "tail": ",11,10  Conclusion: Sees(dog,bald_eagle)  Final answer: True"}, {"tokens": 393, "head": "The bald eagle is nice. The cat sees the dog. The dog is big. If someone likes t", "tail": "11,10  Conclusion: Sees(dog,bald_eagle)  Final answer: False"}, {"tokens": 356, "head": "The bald eagle needs the bear. The bear sees the bald eagle. If something is blu", "tail": " R,1  Conclusion: Needs(bald_eagle,bear)  Final answer: True"}, {"tokens": 358, "head": "The bald eagle needs the bear. The bear sees the bald eagle. If something is blu", "tail": "R,1  Conclusion: Needs(bald_eagle,bear)  Final answer: False"}, {"tokens": 401, "head": "The bald eagle needs the bear. The bear sees the bald eagle. If something is blu", "tail": "10,9  Conclusion: Needs(bear,bald_eagle)  Final answer: True"}, {"tokens": 403, "head": "The bald eagle needs the bear. The bear sees the bald eagle. If something is blu", "tail": "0,9  Conclusion: Needs(bear,bald_eagle)  Final answer: False"}, {"tokens": 307, "head": "Anne is big. Charlie is red. Erin is big. Gary is furry. If Erin is furry then E", "tail": "rry(gary) ; R,4  Conclusion: Furry(gary)  Final answer: True"}, {"tokens": 308, "head": "Anne is big. Charlie is red. Erin is big. Gary is furry. If Erin is furry then E", "tail": "ry(gary) ; R,4  Conclusion: Furry(gary)  Final answer: False"}, {"tokens": 343, "head": "Anne is big. Charlie is red. Erin is big. Gary is furry. If Erin is furry then E", "tail": "; ->E,12,11  Conclusion: ~Green(charlie)  Final answer: True"}, {"tokens": 342, "head": "Anne is big. Charlie is red. Erin is big. Gary is furry. If Erin is furry then E", "tail": " ->E,12,11  Conclusion: ~Green(charlie)  Final answer: False"}, {"tokens": 143, "head": "Bob is white. Charlie is young. Harry is red. If someone is white then they are ", "tail": "arlie) ; R,2  Conclusion: Young(charlie)  Final answer: True"}]

## Window 50682 summary: [{"tokens": 754, "head": "Alan is big. Alan is strong. Alan is high. Erin is short. Erin is thin. Bob is s", "tail": "(bob) ; ->E,31,30  Conclusion: Kind(bob)  Final answer: True"}, {"tokens": 755, "head": "Alan is big. Alan is strong. Alan is high. Erin is short. Erin is thin. Bob is s", "tail": "bob) ; ->E,31,30  Conclusion: Kind(bob)  Final answer: False"}, {"tokens": 755, "head": "Alan is big. Alan is strong. Alan is high. Erin is short. Erin is thin. Bob is s", "tail": "; ->E,31,30  Conclusion: Imperfect(dave)  Final answer: True"}, {"tokens": 756, "head": "Alan is big. Alan is strong. Alan is high. Erin is short. Erin is thin. Bob is s", "tail": " ->E,31,30  Conclusion: Imperfect(dave)  Final answer: False"}, {"tokens": 721, "head": "Gary is strong. Gary is big. Gary is huge. Erin is tiny. Erin is thin. Charlie i", "tail": "ry) ; ->E,29,28  Conclusion: Smart(gary)  Final answer: True"}]

## Window 55985 summary: [{"tokens": 862, "head": "The cow visits the lion. The lion eats the rabbit. The lion is young. The lion n", "tail": "->E,21,33  Conclusion: Visits(lion,cow)  Final answer: False"}, {"tokens": 894, "head": "The cow visits the lion. The lion eats the rabbit. The lion is young. The lion n", "tail": "lion) ; ->E,35,34  Conclusion: Red(lion)  Final answer: True"}, {"tokens": 895, "head": "The cow visits the lion. The lion eats the rabbit. The lion is young. The lion n", "tail": "ion) ; ->E,35,34  Conclusion: Red(lion)  Final answer: False"}, {"tokens": 506, "head": "Bob is blue. Bob is cold. Bob is green. Bob is red. Bob is rough. Bob is round. ", "tail": "in) ; ->E,26,25  Conclusion: Green(erin)  Final answer: True"}, {"tokens": 508, "head": "Bob is blue. Bob is cold. Bob is green. Bob is red. Bob is rough. Bob is round. ", "tail": ") ; ->E,26,25  Conclusion: Green(harry)  Final answer: False"}, {"tokens": 380, "head": "Charlie is green. Charlie is round. Erin is furry. Erin is nice. Fiona is nice. ", "tail": "(fiona) ; R,7  Conclusion: Round(fiona)  Final answer: False"}]

## Window 58067 summary: [{"tokens": 883, "head": "Fiona is high. Fiona is strong. Fiona is huge. Erin is little. Erin is thin. Bob", "tail": "b) ; ->E,37,36  Conclusion: Strong(bob)  Final answer: False"}, {"tokens": 883, "head": "Fiona is high. Fiona is strong. Fiona is huge. Erin is little. Erin is thin. Bob", "tail": "ry) ; ->E,37,36  Conclusion: Thin(harry)  Final answer: True"}, {"tokens": 884, "head": "Fiona is high. Fiona is strong. Fiona is huge. Erin is little. Erin is thin. Bob", "tail": "y) ; ->E,37,36  Conclusion: Thin(harry)  Final answer: False"}, {"tokens": 905, "head": "The wolf is rough. The wolf is sleepy. The wolf is tired. The wolf likes the mou", "tail": "e(mouse) ; R,10  Conclusion: Nice(mouse)  Final answer: True"}]

## Window 60620 summary: [{"tokens": 886, "head": "Bob is high. Bob is strong. Bob is big. Harry is short. Harry is thin. Anne is w", "tail": "an) ; ->E,37,36  Conclusion: Thin(alan)  Final answer: False"}, {"tokens": 843, "head": "Bob is strong. Bob is big. Bob is huge. Harry is small. Harry is tiny. Anne is w", "tail": "(bob) ; ->E,35,34  Conclusion: High(bob)  Final answer: True"}, {"tokens": 844, "head": "Bob is strong. Bob is big. Bob is huge. Harry is small. Harry is tiny. Anne is w", "tail": "bob) ; ->E,35,34  Conclusion: High(bob)  Final answer: False"}, {"tokens": 882, "head": "Bob is strong. Bob is big. Bob is huge. Harry is small. Harry is tiny. Anne is w", "tail": " ->E,37,36  Conclusion: Imperfect(harry)  Final answer: True"}]

## Window 68094 summary: [{"tokens": 755, "head": "Fiona is strong. Fiona is huge. Fiona is big. Harry is small. Harry is thin. Cha", "tail": "y) ; ->E,31,30  Conclusion: Tiny(harry)  Final answer: False"}, {"tokens": 760, "head": "Fiona is strong. Fiona is huge. Fiona is big. Harry is small. Harry is thin. Cha", "tail": ") ; ->E,31,30  Conclusion: Nice(charlie)  Final answer: True"}, {"tokens": 761, "head": "Fiona is strong. Fiona is huge. Fiona is big. Harry is small. Harry is thin. Cha", "tail": " ; ->E,31,30  Conclusion: Nice(charlie)  Final answer: False"}, {"tokens": 761, "head": "Fiona is strong. Fiona is huge. Fiona is big. Harry is small. Harry is thin. Cha", "tail": "lan) ; ->E,31,30  Conclusion: Poor(alan)  Final answer: True"}, {"tokens": 762, "head": "Fiona is strong. Fiona is huge. Fiona is big. Harry is small. Harry is thin. Cha", "tail": "an) ; ->E,31,30  Conclusion: Poor(alan)  Final answer: False"}]

## Window 73180 summary: [{"tokens": 382, "head": "Anne is kind. Charlie is furry. Fiona is young. Gary is young. Kind people are w", "tail": "; ->E,16,15  Conclusion: Smart(charlie)  Final answer: False"}, {"tokens": 416, "head": "Anne is kind. Charlie is furry. Fiona is young. Gary is young. Kind people are w", "tail": ") ; ->E,18,17  Conclusion: Kind(charlie)  Final answer: True"}, {"tokens": 417, "head": "Anne is kind. Charlie is furry. Fiona is young. Gary is young. Kind people are w", "tail": " ; ->E,18,17  Conclusion: Kind(charlie)  Final answer: False"}, {"tokens": 397, "head": "The cat is red. The cat is rough. The mouse chases the cat. The mouse eats the c", "tail": ",cat) ; R,4  Conclusion: Eats(mouse,cat)  Final answer: True"}, {"tokens": 392, "head": "The cat is red. The cat is rough. The mouse chases the cat. The mouse eats the c", "tail": "ough(cat) ; R,2  Conclusion: Rough(cat)  Final answer: False"}, {"tokens": 438, "head": "The cat is red. The cat is rough. The mouse chases the cat. The mouse eats the c", "tail": "; ->E,15,14  Conclusion: Eats(cat,mouse)  Final answer: True"}, {"tokens": 440, "head": "The cat is red. The cat is rough. The mouse chases the cat. The mouse eats the c", "tail": " ->E,15,14  Conclusion: Eats(cat,mouse)  Final answer: False"}, {"tokens": 469, "head": "The cat is red. The cat is rough. The mouse chases the cat. The mouse eats the c", "tail": "e) ; ->E,17,16  Conclusion: Rough(mouse)  Final answer: True"}, {"tokens": 470, "head": "The cat is red. The cat is rough. The mouse chases the cat. The mouse eats the c", "tail": ") ; ->E,17,16  Conclusion: Rough(mouse)  Final answer: False"}, {"tokens": 253, "head": "The tiger is young. If the tiger is young then the tiger is rough. If the tiger ", "tail": "ger) ; ->E,2,8  Conclusion: Rough(tiger)  Final answer: True"}]

## Window 73498 summary: [{"tokens": 603, "head": "Dave is strong. Dave is huge. Dave is big. Alan is small. Alan is thin. Anne is ", "tail": "nne) ; ->E,25,24  Conclusion: Nice(anne)  Final answer: True"}, {"tokens": 604, "head": "Dave is strong. Dave is huge. Dave is big. Alan is small. Alan is thin. Anne is ", "tail": "ne) ; ->E,25,24  Conclusion: Nice(anne)  Final answer: False"}, {"tokens": 604, "head": "Dave is strong. Dave is huge. Dave is big. Alan is small. Alan is thin. Anne is ", "tail": ") ; ->E,25,24  Conclusion: Dull(charlie)  Final answer: True"}, {"tokens": 605, "head": "Dave is strong. Dave is huge. Dave is big. Alan is small. Alan is thin. Anne is ", "tail": " ; ->E,25,24  Conclusion: Dull(charlie)  Final answer: False"}, {"tokens": 710, "head": "The leopard is slow. The leopard is dull. The leopard is lazy. The leopard likes", "tail": "; ->E,27,26  Conclusion: Small(squirrel)  Final answer: True"}, {"tokens": 711, "head": "The leopard is slow. The leopard is dull. The leopard is lazy. The leopard likes", "tail": " ->E,27,26  Conclusion: Small(squirrel)  Final answer: False"}]

## Window 76519 summary: [{"tokens": 1090, "head": "The tiger is lazy. The tiger is reckless. The tiger is angry. The tiger chases t", "tail": ") ; ->E,41,40  Conclusion: Kind(rabbit)  Final answer: False"}, {"tokens": 1199, "head": "The tiger is lazy. The tiger is reckless. The tiger is angry. The tiger chases t", "tail": "r) ; ->E,47,46  Conclusion: Awful(tiger)  Final answer: True"}, {"tokens": 1200, "head": "The tiger is lazy. The tiger is reckless. The tiger is angry. The tiger chases t", "tail": ") ; ->E,47,46  Conclusion: Awful(tiger)  Final answer: False"}]

## Window 77734 summary: [{"tokens": 254, "head": "The bald eagle is not blue. The bald eagle is cold. The bald eagle is nice. The ", "tail": "gle) ; R,3  Conclusion: Nice(bald_eagle)  Final answer: True"}, {"tokens": 255, "head": "The bald eagle is not blue. The bald eagle is cold. The bald eagle is nice. The ", "tail": "le) ; R,3  Conclusion: Nice(bald_eagle)  Final answer: False"}, {"tokens": 322, "head": "The dog eats the mouse. The dog is nice. The dog is red. The dog sees the squirr", "tail": ". Nice(dog) ; R,2  Conclusion: Nice(dog)  Final answer: True"}, {"tokens": 331, "head": "The dog eats the mouse. The dog is nice. The dog is red. The dog sees the squirr", "tail": "use) ; R,1  Conclusion: Eats(dog,mouse)  Final answer: False"}, {"tokens": 196, "head": "The cow eats the mouse. The mouse eats the cow. If something chases the mouse th", "tail": "ouse) ; R,1  Conclusion: Eats(cow,mouse)  Final answer: True"}, {"tokens": 198, "head": "The cow eats the mouse. The mouse eats the cow. If something chases the mouse th", "tail": "use) ; R,1  Conclusion: Eats(cow,mouse)  Final answer: False"}, {"tokens": 462, "head": "Anne is blue. Anne is green. Anne is kind. Anne is quiet. Erin is blue. Erin is ", "tail": "Kind(anne) ; R,3  Conclusion: Kind(anne)  Final answer: True"}, {"tokens": 466, "head": "Anne is blue. Anne is green. Anne is kind. Anne is quiet. Erin is blue. Erin is ", "tail": "y(gary) ; R,10  Conclusion: Furry(gary)  Final answer: False"}, {"tokens": 527, "head": "The dog is nice. The dog is round. The dog likes the rabbit. The dog needs the t", "tail": "n,dog) ; R,8  Conclusion: Sees(lion,dog)  Final answer: True"}, {"tokens": 527, "head": "The dog is nice. The dog is round. The dog likes the rabbit. The dog needs the t", "tail": "g) ; R,9  Conclusion: Likes(rabbit,dog)  Final answer: False"}, {"tokens": 288, "head": "Charlie is white. All big people are rough. All red, white people are round. All", "tail": "arlie) ; R,1  Conclusion: White(charlie)  Final answer: True"}, {"tokens": 222, "head": "The lion is big. The lion is cold. The lion is not kind. The lion is nice. The l", "tail": "ung(lion) ; R,5  Conclusion: Young(lion)  Final answer: True"}]

## Window 78495 summary: [{"tokens": 601, "head": "Alan is huge. Alan is big. Alan is high. Dave is thin. Dave is little. Gary is q", "tail": "e) ; ->E,25,24  Conclusion: Short(dave)  Final answer: False"}, {"tokens": 600, "head": "Alan is huge. Alan is big. Alan is high. Dave is thin. Dave is little. Gary is q", "tail": "ary) ; ->E,25,24  Conclusion: Nice(gary)  Final answer: True"}, {"tokens": 601, "head": "Alan is huge. Alan is big. Alan is high. Dave is thin. Dave is little. Gary is q", "tail": "ry) ; ->E,25,24  Conclusion: Nice(gary)  Final answer: False"}, {"tokens": 604, "head": "Alan is huge. Alan is big. Alan is high. Dave is thin. Dave is little. Gary is q", "tail": "d(bob) ; ->E,25,24  Conclusion: Sad(bob)  Final answer: True"}, {"tokens": 605, "head": "Alan is huge. Alan is big. Alan is high. Dave is thin. Dave is little. Gary is q", "tail": "(bob) ; ->E,25,24  Conclusion: Sad(bob)  Final answer: False"}, {"tokens": 564, "head": "Alan is huge. Alan is strong. Alan is high. Gary is short. Gary is small. Anne i", "tail": ") ; ->E,23,22  Conclusion: Wealthy(alan)  Final answer: True"}]

## Window 78496 summary: [{"tokens": 757, "head": "Fiona is high. Fiona is strong. Fiona is big. Harry is short. Harry is tiny. Ala", "tail": "n) ; ->E,31,30  Conclusion: Quiet(alan)  Final answer: False"}, {"tokens": 757, "head": "Fiona is high. Fiona is strong. Fiona is big. Harry is short. Harry is tiny. Ala", "tail": "(bob) ; ->E,31,30  Conclusion: Dull(bob)  Final answer: True"}, {"tokens": 758, "head": "Fiona is high. Fiona is strong. Fiona is big. Harry is short. Harry is tiny. Ala", "tail": "bob) ; ->E,31,30  Conclusion: Dull(bob)  Final answer: False"}, {"tokens": 866, "head": "The leopard is sleepy. The leopard is tired. The leopard is dull. The leopard se", "tail": "og) ; ->E,34,33  Conclusion: Lovely(dog)  Final answer: True"}, {"tokens": 867, "head": "The leopard is sleepy. The leopard is tired. The leopard is dull. The leopard se", "tail": "g) ; ->E,34,33  Conclusion: Lovely(dog)  Final answer: False"}]
