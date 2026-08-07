# Real-data paired logic/NL corpus - audit

Tokenizer for token counts: **qwen2.5**

## Dedup vs held-out splits (ProofWriter OWA test incl. birds-electricity, PARARULE-Plus test, FOLIO v0 all)

```json
{
  "ngram": 10,
  "threshold": 0.8,
  "test_problems_exact": 112455,
  "test_theories_ngram": 13756,
  "stats": {
    "pararule": {
      "kept": 212352,
      "exact_dropped": 0,
      "near_dup_dropped": 0
    },
    "prontoqa": {
      "kept": 21000,
      "exact_dropped": 0,
      "near_dup_dropped": 0
    },
    "proofwriter": {
      "kept": 217810,
      "exact_dropped": 0,
      "near_dup_dropped": 34
    }
  }
}
```

## Source: pararule

- examples scanned: 424658, converted+validated: 212352 (50.0%)
- drop reasons: `{"not_derivable_explicit": 212306}`
- kept after dedup: **212352** paired docs
- answers: {'True': 106176, 'False': 106176}
- depth distribution: {0: 5892, 1: 11794, 2: 53122, 3: 53172, 4: 47158, 5: 41214}
- formal doc tokens: total 189,646,344; mean 893; p50 883; p90 1202; max 1354
- NL doc tokens: total 132,569,134; mean 624; p50 620; p90 861; max 972

### pararule random example 1 (id=NonNegationRule-D2-920::NonNegationRule-D2-9207, depth=2)

FORMAL DOC:
```
Bob is big. Bob is huge. Bob is high. Harry is short. Harry is little. Fiona is kind. Fiona is wealthy. Fiona is smart. Alan is bad. Alan is poor. Alan is rough. Big people are kind. If someone is short and little then they are thin. If someone is bad and poor then they are dull. If someone is kind and wealthy then they are quiet. All thin people are small. All kind people are wealthy. All quiet people are nice. All dull people are sad.

True or false: Alan is sad.

Solution:
Context:
Constants:
alan = Alan
bob = Bob
fiona = Fiona
harry = Harry

Predicates:
Bad(x) = x is bad
Big(x) = x is big
Dull(x) = x is dull
High(x) = x is high
Huge(x) = x is huge
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
Thin(x) = x is thin
Wealthy(x) = x is wealthy

Premises:
1. Big(bob)
2. Huge(bob)
3. High(bob)
4. Short(harry)
5. Little(harry)
6. Kind(fiona)
7. Wealthy(fiona)
8. Smart(fiona)
9. Bad(alan)
10. Poor(alan)
11. Rough(alan)
12. Ax(Big(x) -> Kind(x))
13. Ax(Short(x) & Little(x) -> Thin(x))
14. Ax(Bad(x) & Poor(x) -> Dull(x))
15. Ax(Kind(x) & Wealthy(x) -> Quiet(x))
16. Ax(Thin(x) -> Small(x))
17. Ax(Kind(x) -> Wealthy(x))
18. Ax(Quiet(x) -> Nice(x))
19. Ax(Dull(x) -> Sad(x))

Derivation:
20. Bad(alan) ; R,9
21. Poor(alan) ; R,10
22. Bad(alan) & Poor(alan) ; ∧I,20,21
23. Bad(alan) & Poor(alan) -> Dull(alan) ; AE,14
24. Dull(alan) ; ->E,23,22
25. Dull(alan) -> Sad(alan) ; AE,19
26. Sad(alan) ; ->E,25,24

Conclusion:
Sad(alan)

Final answer: True
```
NL DOC:
```
Bob is big. Bob is huge. Bob is high. Harry is short. Harry is little. Fiona is kind. Fiona is wealthy. Fiona is smart. Alan is bad. Alan is poor. Alan is rough. Big people are kind. If someone is short and little then they are thin. If someone is bad and poor then they are dull. If someone is kind and wealthy then they are quiet. All thin people are small. All kind people are wealthy. All quiet people are nice. All dull people are sad.

True or false: Alan is sad.

Solution:
Context:
Premises:
1. Bob is big.
2. Bob is huge.
3. Bob is high.
4. Harry is short.
5. Harry is little.
6. Fiona is kind.
7. Fiona is wealthy.
8. Fiona is smart.
9. Alan is bad.
10. Alan is poor.
11. Alan is rough.
12. Big people are kind.
13. If someone is short and little then they are thin.
14. If someone is bad and poor then they are dull.
15. If someone is kind and wealthy then they are quiet.
16. All thin people are small.
17. All kind people are wealthy.
18. All quiet people are nice.
19. All dull people are sad.

Derivation:
20. We are given: Alan is bad.
21. We are given: Alan is poor.
22. Combining: Alan is bad and Alan is poor.
23. Instantiating rule 14 for Alan: if Alan is bad and Alan is poor, then Alan is dull.
24. Therefore, Alan is dull.
25. Instantiating rule 19 for Alan: if Alan is dull, then Alan is sad.
26. Therefore, Alan is sad.

Conclusion:
Alan is sad.

Final answer: True
```

### pararule random example 2 (id=NonNegationRule-Animal-D5-2611::NonNegationRule-Animal-D5-26116, depth=5)

FORMAL DOC:
```
The snake is angry. The snake is tired. The snake is reckless. The snake likes the mouse. The dinosaur attacks the rabbit. The dinosaur is obese. The dinosaur is fierce. The mouse is kind. The mouse is smart. The mouse is round. The rabbit is beautiful. The rabbit is cute. The rabbit is small. Kind animals are beautiful. If something is tired then it chases the mouse. If something chases the mouse then it is lazy. If something is angry and tired then it is reckless. If something is beautiful and cute then it is funny. If something is obese and fierce then it is strong. If something is reckless then it is slow. If something is slow then it is dull. If something is dull then it is fierce. All fierce animals are obese. If something is beautiful then it is cute. If something is cute then it is small. If something is small then it is round. If something is round then it is quiet. All small animals are nice. If something is strong then it is big. All big animals are heavy. If something is heavy then it is tired. All tired animals are rough. If something is funny then it is adorable. All adorable animals are lovely. If something is lovely then it is angry. All angry animals are furry. If something is lazy then it is awful. If something is awful then it is sleepy. All sleepy animals are boring.

True or false: The dinosaur is not rough.

Solution:
Context:
Constants:
dinosaur = the dinosaur
mouse = the mouse
rabbit = the rabbit
snake = the snake

Predicates:
Adorable(x) = x is adorable
Angry(x) = x is angry
Attacks(x,y) = x attacks y
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
Likes(x,y) = x likes y
Lovely(x) = x is lovely
Nice(x) = x is nice
Obese(x) = x is obese
Quiet(x) = x is quiet
Reckless(x) = x is reckless
Rough(x) = x is rough
Round(x) = x is round
Sleepy(x) = x is sleepy
Slow(x) = x is slow
Small(x) = x is small
Smart(x) = x is smart
Strong(x) = x is strong
Tired(x) = x is tired

Premises:
1. Angry(snake)
2. Tired(snake)
3. Reckless(snake)
4. Likes(snake,mouse)
5. Attacks(dinosaur,rabbit)
6. Obese(dinosaur)
7. Fierce(dinosaur)
8. Kind(mouse)
9. Smart(mouse)
10. Round(mouse)
11. Beautiful(rabbit)
12. Cute(rabbit)
13. Small(rabbit)
14. Ax(Kind(x) -> Beautiful(x))
15. Ax(Tired(x) -> Chases(x,mouse))
16. Ax(Chases(x,mouse) -> Lazy(x))
17. Ax(Angry(x) & Tired(x) -> Reckless(x))
18. Ax(Beautiful(x) & Cute(x) -> Funny(x))
19. Ax(Obese(x) & Fierce(x) -> Strong(x))
20. Ax(Reckless(x) -> Slow(x))
21. Ax(Slow(x) -> Dull(x))
22. Ax(Dull(x) -> Fierce(x))
23. Ax(Fierce(x) -> Obese(x))
24. Ax(Beautiful(x) -> Cute(x))
25. Ax(Cute(x) -> Small(x))
26. Ax(Small(x) -> Round(x))
27. Ax(Round(x) -> Quiet(x))
28. Ax(Small(x) -> Nice(x))
29. Ax(Strong(x) -> Big(x))
30. Ax(Big(x) -> Heavy(x))
31. Ax(Heavy(x) -> Tired(x))
32. Ax(Tired(x) -> Rough(x))
33. Ax(Funny(x) -> Adorable(x))
34. Ax(Adorable(x) -> Lovely(x))
35. Ax(Lovely(x) -> Angry(x))
36. Ax(Angry(x) -> Furry(x))
37. Ax(Lazy(x) -> Awful(x))
38. Ax(Awful(x) -> Sleepy(x))
39. Ax(Sleepy(x) -> Boring(x))

Derivation:
40. Obese(dinosaur) ; R,6
41. Fierce(dinosaur) ; R,7
42. Obese(dinosaur) & Fierce(dinosaur) ; ∧I,40,41
43. Obese(dinosaur) & Fierce(dinosaur) -> Strong(dinosaur) ; AE,19
44. Strong(dinosaur) ; ->E,43,42
45. Strong(dinosaur) -> Big(dinosaur) ; AE,29
46. Big(dinosaur) ; ->E,45,44
47. Big(dinosaur) -> Heavy(dinosaur) ; AE,30
48. Heavy(dinosaur) ; ->E,47,46
49. Heavy(dinosaur) -> Tired(dinosaur) ; AE,31
50. Tired(dinosaur) ; ->E,49,48
51. Tired(dinosaur) -> Rough(dinosaur) ; AE,32
52. Rough(dinosaur) ; ->E,51,50

Conclusion:
Rough(dinosaur)

Final answer: False
```
NL DOC:
```
The snake is angry. The snake is tired. The snake is reckless. The snake likes the mouse. The dinosaur attacks the rabbit. The dinosaur is obese. The dinosaur is fierce. The mouse is kind. The mouse is smart. The mouse is round. The rabbit is beautiful. The rabbit is cute. The rabbit is small. Kind animals are beautiful. If something is tired then it chases the mouse. If something chases the mouse then it is lazy. If something is angry and tired then it is reckless. If something is beautiful and cute then it is funny. If something is obese and fierce then it is strong. If something is reckless then it is slow. If something is slow then it is dull. If something is dull then it is fierce. All fierce animals are obese. If something is beautiful then it is cute. If something is cute then it is small. If something is small then it is round. If something is round then it is quiet. All small animals are nice. If something is strong then it is big. All big animals are heavy. If something is heavy then it is tired. All tired animals are rough. If something is funny then it is adorable. All adorable animals are lovely. If something is lovely then it is angry. All angry animals are furry. If something is lazy then it is awful. If something is awful then it is sleepy. All sleepy animals are boring.

True or false: The dinosaur is not rough.

Solution:
Context:
Premises:
1. The snake is angry.
2. The snake is tired.
3. The snake is reckless.
4. The snake likes the mouse.
5. The dinosaur attacks the rabbit.
6. The dinosaur is obese.
7. The dinosaur is fierce.
8. The mouse is kind.
9. The mouse is smart.
10. The mouse is round.
11. The rabbit is beautiful.
12. The rabbit is cute.
13. The rabbit is small.
14. Kind animals are beautiful.
15. If something is tired then it chases the mouse.
16. If something chases the mouse then it is lazy.
17. If something is angry and tired then it is reckless.
18. If something is beautiful and cute then it is funny.
19. If something is obese and fierce then it is strong.
20. If something is reckless then it is slow.
21. If something is slow then it is dull.
22. If something is dull then it is fierce.
23. All fierce animals are obese.
24. If something is beautiful then it is cute.
25. If something is cute then it is small.
26. If something is small then it is round.
27. If something is round then it is quiet.
28. All small animals are nice.
29. If something is strong then it is big.
30. All big animals are heavy.
31. If something is heavy then it is tired.
32. All tired animals are rough.
33. If something is funny then it is adorable.
34. All adorable animals are lovely.
35. If something is lovely then it is angry.
36. All angry animals are furry.
37. If something is lazy then it is awful.
38. If something is awful then it is sleepy.
39. All sleepy animals are boring.

Derivation:
40. We are given: The dinosaur is obese.
41. We are given: The dinosaur is fierce.
42. Combining: The dinosaur is obese and The dinosaur is fierce.
43. Instantiating rule 19 for the dinosaur: if The dinosaur is obese and The dinosaur is fierce, then The dinosaur is strong.
44. Therefore, The dinosaur is strong.
45. Instantiating rule 29 for the dinosaur: if The dinosaur is strong, then The dinosaur is big.
46. Therefore, The dinosaur is big.
47. Instantiating rule 30 for the dinosaur: if The dinosaur is big, then The dinosaur is heavy.
48. Therefore, The dinosaur is heavy.
49. Instantiating rule 31 for the dinosaur: if The dinosaur is heavy, then The dinosaur is tired.
50. Therefore, The dinosaur is tired.
51. Instantiating rule 32 for the dinosaur: if The dinosaur is tired, then The dinosaur is rough.
52. Therefore, The dinosaur is rough.

Conclusion:
The dinosaur is rough.

Final answer: False
```

### pararule random example 3 (id=NonNegationRule-Animal-D2-2674::NonNegationRule-Animal-D2-26748, depth=2)

FORMAL DOC:
```
The snake is rough. The snake is slow. The snake is sleepy. The snake sees the rabbit. The leopard likes the dog. The leopard is awful. The leopard is heavy. The rabbit is nice. The rabbit is smart. The rabbit is round. The dog is small. The dog is cute. The dog is beautiful. Nice animals are small. If something is slow then it needs the rabbit. If something needs the rabbit then it is dull. If something is rough and slow then it is sleepy. If something is small and cute then it is furry. If something is awful and heavy then it is big. All sleepy animals are lazy. All small animals are cute. All big animals are strong. All furry animals are lovely.

True or false: The dog is not lovely.

Solution:
Context:
Constants:
dog = the dog
leopard = the leopard
rabbit = the rabbit
snake = the snake

Predicates:
Awful(x) = x is awful
Beautiful(x) = x is beautiful
Big(x) = x is big
Cute(x) = x is cute
Dull(x) = x is dull
Furry(x) = x is furry
Heavy(x) = x is heavy
Lazy(x) = x is lazy
Likes(x,y) = x likes y
Lovely(x) = x is lovely
Needs(x,y) = x needs y
Nice(x) = x is nice
Rough(x) = x is rough
Round(x) = x is round
Sees(x,y) = x sees y
Sleepy(x) = x is sleepy
Slow(x) = x is slow
Small(x) = x is small
Smart(x) = x is smart
Strong(x) = x is strong

Premises:
1. Rough(snake)
2. Slow(snake)
3. Sleepy(snake)
4. Sees(snake,rabbit)
5. Likes(leopard,dog)
6. Awful(leopard)
7. Heavy(leopard)
8. Nice(rabbit)
9. Smart(rabbit)
10. Round(rabbit)
11. Small(dog)
12. Cute(dog)
13. Beautiful(dog)
14. Ax(Nice(x) -> Small(x))
15. Ax(Slow(x) -> Needs(x,rabbit))
16. Ax(Needs(x,rabbit) -> Dull(x))
17. Ax(Rough(x) & Slow(x) -> Sleepy(x))
18. Ax(Small(x) & Cute(x) -> Furry(x))
19. Ax(Awful(x) & Heavy(x) -> Big(x))
20. Ax(Sleepy(x) -> Lazy(x))
21. Ax(Small(x) -> Cute(x))
22. Ax(Big(x) -> Strong(x))
23. Ax(Furry(x) -> Lovely(x))

Derivation:
24. Small(dog) ; R,11
25. Cute(dog) ; R,12
26. Small(dog) & Cute(dog) ; ∧I,24,25
27. Small(dog) & Cute(dog) -> Furry(dog) ; AE,18
28. Furry(dog) ; ->E,27,26
29. Furry(dog) -> Lovely(dog) ; AE,23
30. Lovely(dog) ; ->E,29,28

Conclusion:
Lovely(dog)

Final answer: False
```
NL DOC:
```
The snake is rough. The snake is slow. The snake is sleepy. The snake sees the rabbit. The leopard likes the dog. The leopard is awful. The leopard is heavy. The rabbit is nice. The rabbit is smart. The rabbit is round. The dog is small. The dog is cute. The dog is beautiful. Nice animals are small. If something is slow then it needs the rabbit. If something needs the rabbit then it is dull. If something is rough and slow then it is sleepy. If something is small and cute then it is furry. If something is awful and heavy then it is big. All sleepy animals are lazy. All small animals are cute. All big animals are strong. All furry animals are lovely.

True or false: The dog is not lovely.

Solution:
Context:
Premises:
1. The snake is rough.
2. The snake is slow.
3. The snake is sleepy.
4. The snake sees the rabbit.
5. The leopard likes the dog.
6. The leopard is awful.
7. The leopard is heavy.
8. The rabbit is nice.
9. The rabbit is smart.
10. The rabbit is round.
11. The dog is small.
12. The dog is cute.
13. The dog is beautiful.
14. Nice animals are small.
15. If something is slow then it needs the rabbit.
16. If something needs the rabbit then it is dull.
17. If something is rough and slow then it is sleepy.
18. If something is small and cute then it is furry.
19. If something is awful and heavy then it is big.
20. All sleepy animals are lazy.
21. All small animals are cute.
22. All big animals are strong.
23. All furry animals are lovely.

Derivation:
24. We are given: The dog is small.
25. We are given: The dog is cute.
26. Combining: The dog is small and The dog is cute.
27. Instantiating rule 18 for the dog: if The dog is small and The dog is cute, then The dog is furry.
28. Therefore, The dog is furry.
29. Instantiating rule 23 for the dog: if The dog is furry, then The dog is lovely.
30. Therefore, The dog is lovely.

Conclusion:
The dog is lovely.

Final answer: False
```

### pararule random example 4 (id=NonNegationRule-D2-1936::NonNegationRule-D2-19367, depth=2)

FORMAL DOC:
```
Erin is strong. Erin is huge. Erin is high. Gary is little. Gary is short. Anne is kind. Anne is quiet. Anne is nice. Dave is poor. Dave is rough. Dave is dull. Strong people are kind. If someone is little and short then they are small. If someone is poor and rough then they are sad. If someone is kind and quiet then they are wealthy. All small people are thin. All kind people are quiet. All wealthy people are smart. All sad people are bad.

True or false: Dave is bad.

Solution:
Context:
Constants:
anne = Anne
dave = Dave
erin = Erin
gary = Gary

Predicates:
Bad(x) = x is bad
Dull(x) = x is dull
High(x) = x is high
Huge(x) = x is huge
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
Wealthy(x) = x is wealthy

Premises:
1. Strong(erin)
2. Huge(erin)
3. High(erin)
4. Little(gary)
5. Short(gary)
6. Kind(anne)
7. Quiet(anne)
8. Nice(anne)
9. Poor(dave)
10. Rough(dave)
11. Dull(dave)
12. Ax(Strong(x) -> Kind(x))
13. Ax(Little(x) & Short(x) -> Small(x))
14. Ax(Poor(x) & Rough(x) -> Sad(x))
15. Ax(Kind(x) & Quiet(x) -> Wealthy(x))
16. Ax(Small(x) -> Thin(x))
17. Ax(Kind(x) -> Quiet(x))
18. Ax(Wealthy(x) -> Smart(x))
19. Ax(Sad(x) -> Bad(x))

Derivation:
20. Poor(dave) ; R,9
21. Rough(dave) ; R,10
22. Poor(dave) & Rough(dave) ; ∧I,20,21
23. Poor(dave) & Rough(dave) -> Sad(dave) ; AE,14
24. Sad(dave) ; ->E,23,22
25. Sad(dave) -> Bad(dave) ; AE,19
26. Bad(dave) ; ->E,25,24

Conclusion:
Bad(dave)

Final answer: True
```
NL DOC:
```
Erin is strong. Erin is huge. Erin is high. Gary is little. Gary is short. Anne is kind. Anne is quiet. Anne is nice. Dave is poor. Dave is rough. Dave is dull. Strong people are kind. If someone is little and short then they are small. If someone is poor and rough then they are sad. If someone is kind and quiet then they are wealthy. All small people are thin. All kind people are quiet. All wealthy people are smart. All sad people are bad.

True or false: Dave is bad.

Solution:
Context:
Premises:
1. Erin is strong.
2. Erin is huge.
3. Erin is high.
4. Gary is little.
5. Gary is short.
6. Anne is kind.
7. Anne is quiet.
8. Anne is nice.
9. Dave is poor.
10. Dave is rough.
11. Dave is dull.
12. Strong people are kind.
13. If someone is little and short then they are small.
14. If someone is poor and rough then they are sad.
15. If someone is kind and quiet then they are wealthy.
16. All small people are thin.
17. All kind people are quiet.
18. All wealthy people are smart.
19. All sad people are bad.

Derivation:
20. We are given: Dave is poor.
21. We are given: Dave is rough.
22. Combining: Dave is poor and Dave is rough.
23. Instantiating rule 14 for Dave: if Dave is poor and Dave is rough, then Dave is sad.
24. Therefore, Dave is sad.
25. Instantiating rule 19 for Dave: if Dave is sad, then Dave is bad.
26. Therefore, Dave is bad.

Conclusion:
Dave is bad.

Final answer: True
```

### pararule random example 5 (id=NonNegationRule-D5-2417::NonNegationRule-D5-24177, depth=5)

FORMAL DOC:
```
Gary is heavy. Gary is strong. Gary is big. Alan is little. Alan is small. Dave is quiet. Dave is kind. Dave is nice. Harry is dull. Harry is rough. Harry is poor. Heavy people are quiet. If someone is little and small then they are tiny. If someone is dull and rough then they are bad. If someone is quiet and kind then they are smart. If someone is tiny then they are short. If someone is short then they are thin. If someone is thin then they are poor. All poor people are rough. If someone is quiet then they are kind. If someone is kind then they are nice. If someone is nice then they are huge. All huge people are high. If someone is smart then they are wealthy. If someone is wealthy then they are clever. If someone is clever then they are strong. All strong people are big. If someone is bad then they are sad. If someone is sad then they are imperfect. All imperfect people are small. All small people are little.

True or false: Harry is little.

Solution:
Context:
Constants:
alan = Alan
dave = Dave
gary = Gary
harry = Harry

Predicates:
Bad(x) = x is bad
Big(x) = x is big
Clever(x) = x is clever
Dull(x) = x is dull
Heavy(x) = x is heavy
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
1. Heavy(gary)
2. Strong(gary)
3. Big(gary)
4. Little(alan)
5. Small(alan)
6. Quiet(dave)
7. Kind(dave)
8. Nice(dave)
9. Dull(harry)
10. Rough(harry)
11. Poor(harry)
12. Ax(Heavy(x) -> Quiet(x))
13. Ax(Little(x) & Small(x) -> Tiny(x))
14. Ax(Dull(x) & Rough(x) -> Bad(x))
15. Ax(Quiet(x) & Kind(x) -> Smart(x))
16. Ax(Tiny(x) -> Short(x))
17. Ax(Short(x) -> Thin(x))
18. Ax(Thin(x) -> Poor(x))
19. Ax(Poor(x) -> Rough(x))
20. Ax(Quiet(x) -> Kind(x))
21. Ax(Kind(x) -> Nice(x))
22. Ax(Nice(x) -> Huge(x))
23. Ax(Huge(x) -> High(x))
24. Ax(Smart(x) -> Wealthy(x))
25. Ax(Wealthy(x) -> Clever(x))
26. Ax(Clever(x) -> Strong(x))
27. Ax(Strong(x) -> Big(x))
28. Ax(Bad(x) -> Sad(x))
29. Ax(Sad(x) -> Imperfect(x))
30. Ax(Imperfect(x) -> Small(x))
31. Ax(Small(x) -> Little(x))

Derivation:
32. Dull(harry) ; R,9
33. Rough(harry) ; R,10
34. Dull(harry) & Rough(harry) ; ∧I,32,33
35. Dull(harry) & Rough(harry) -> Bad(harry) ; AE,14
36. Bad(harry) ; ->E,35,34
37. Bad(harry) -> Sad(harry) ; AE,28
38. Sad(harry) ; ->E,37,36
39. Sad(harry) -> Imperfect(harry) ; AE,29
40. Imperfect(harry) ; ->E,39,38
41. Imperfect(harry) -> Small(harry) ; AE,30
42. Small(harry) ; ->E,41,40
43. Small(harry) -> Little(harry) ; AE,31
44. Little(harry) ; ->E,43,42

Conclusion:
Little(harry)

Final answer: True
```
NL DOC:
```
Gary is heavy. Gary is strong. Gary is big. Alan is little. Alan is small. Dave is quiet. Dave is kind. Dave is nice. Harry is dull. Harry is rough. Harry is poor. Heavy people are quiet. If someone is little and small then they are tiny. If someone is dull and rough then they are bad. If someone is quiet and kind then they are smart. If someone is tiny then they are short. If someone is short then they are thin. If someone is thin then they are poor. All poor people are rough. If someone is quiet then they are kind. If someone is kind then they are nice. If someone is nice then they are huge. All huge people are high. If someone is smart then they are wealthy. If someone is wealthy then they are clever. If someone is clever then they are strong. All strong people are big. If someone is bad then they are sad. If someone is sad then they are imperfect. All imperfect people are small. All small people are little.

True or false: Harry is little.

Solution:
Context:
Premises:
1. Gary is heavy.
2. Gary is strong.
3. Gary is big.
4. Alan is little.
5. Alan is small.
6. Dave is quiet.
7. Dave is kind.
8. Dave is nice.
9. Harry is dull.
10. Harry is rough.
11. Harry is poor.
12. Heavy people are quiet.
13. If someone is little and small then they are tiny.
14. If someone is dull and rough then they are bad.
15. If someone is quiet and kind then they are smart.
16. If someone is tiny then they are short.
17. If someone is short then they are thin.
18. If someone is thin then they are poor.
19. All poor people are rough.
20. If someone is quiet then they are kind.
21. If someone is kind then they are nice.
22. If someone is nice then they are huge.
23. All huge people are high.
24. If someone is smart then they are wealthy.
25. If someone is wealthy then they are clever.
26. If someone is clever then they are strong.
27. All strong people are big.
28. If someone is bad then they are sad.
29. If someone is sad then they are imperfect.
30. All imperfect people are small.
31. All small people are little.

Derivation:
32. We are given: Harry is dull.
33. We are given: Harry is rough.
34. Combining: Harry is dull and Harry is rough.
35. Instantiating rule 14 for Harry: if Harry is dull and Harry is rough, then Harry is bad.
36. Therefore, Harry is bad.
37. Instantiating rule 28 for Harry: if Harry is bad, then Harry is sad.
38. Therefore, Harry is sad.
39. Instantiating rule 29 for Harry: if Harry is sad, then Harry is imperfect.
40. Therefore, Harry is imperfect.
41. Instantiating rule 30 for Harry: if Harry is imperfect, then Harry is small.
42. Therefore, Harry is small.
43. Instantiating rule 31 for Harry: if Harry is small, then Harry is little.
44. Therefore, Harry is little.

Conclusion:
Harry is little.

Final answer: True
```

## Source: prontoqa

- examples scanned: 21000, converted+validated: 21000 (100.0%)
- drop reasons: `{}`
- kept after dedup: **21000** paired docs
- answers: {'False': 7560, 'True': 13440}
- depth distribution: {0: 3000, 1: 6000, 2: 3000, 3: 3000, 4: 3000, 5: 3000}
- formal doc tokens: total 15,751,429; mean 750; p50 737; p90 1041; max 1086
- NL doc tokens: total 8,759,234; mean 417; p50 409; p90 582; max 630

### prontoqa random example 1 (id=1hop_0shot_random_seed3407::example580, depth=1)

FORMAL DOC:
```
Every grimpus is a rompus. Impuses are tumpuses. Yumpuses are not nervous. Lorpuses are melodic. Yumpuses are jompuses. Dumpuses are hot. Every impus is a lorpus. Impuses are not fruity. Sterpuses are dull. Sterpuses are dumpuses. Every grimpus is a sterpus. Each grimpus is slow. Rompuses are blue. Vumpuses are not dull. Every sterpus is an impus. Wren is a yumpus. Wren is a sterpus.

True or false: Wren is dull.

Solution:
Context:
Constants:
wren = Wren

Predicates:
Blue(x) = x is blue
Dull(x) = x is dull
Dumpus(x) = x is a dumpus
Fruity(x) = x is fruity
Grimpus(x) = x is a grimpus
Hot(x) = x is hot
Impus(x) = x is a impus
Jompus(x) = x is a jompus
Lorpus(x) = x is a lorpus
Melodic(x) = x is melodic
Nervous(x) = x is nervous
Rompus(x) = x is a rompus
Slow(x) = x is slow
Sterpus(x) = x is a sterpus
Tumpus(x) = x is a tumpus
Vumpus(x) = x is a vumpus
Yumpus(x) = x is a yumpus

Premises:
1. Ax(Grimpus(x) -> Rompus(x))
2. Ax(Impus(x) -> Tumpus(x))
3. Ax(Yumpus(x) -> ~Nervous(x))
4. Ax(Lorpus(x) -> Melodic(x))
5. Ax(Yumpus(x) -> Jompus(x))
6. Ax(Dumpus(x) -> Hot(x))
7. Ax(Impus(x) -> Lorpus(x))
8. Ax(Impus(x) -> ~Fruity(x))
9. Ax(Sterpus(x) -> Dull(x))
10. Ax(Sterpus(x) -> Dumpus(x))
11. Ax(Grimpus(x) -> Sterpus(x))
12. Ax(Grimpus(x) -> Slow(x))
13. Ax(Rompus(x) -> Blue(x))
14. Ax(Vumpus(x) -> ~Dull(x))
15. Ax(Sterpus(x) -> Impus(x))
16. Yumpus(wren)
17. Sterpus(wren)

Derivation:
18. Sterpus(wren) ; R,17
19. Sterpus(wren) -> Dull(wren) ; AE,9
20. Dull(wren) ; ->E,19,18

Conclusion:
Dull(wren)

Final answer: True
```
NL DOC:
```
Every grimpus is a rompus. Impuses are tumpuses. Yumpuses are not nervous. Lorpuses are melodic. Yumpuses are jompuses. Dumpuses are hot. Every impus is a lorpus. Impuses are not fruity. Sterpuses are dull. Sterpuses are dumpuses. Every grimpus is a sterpus. Each grimpus is slow. Rompuses are blue. Vumpuses are not dull. Every sterpus is an impus. Wren is a yumpus. Wren is a sterpus.

True or false: Wren is dull.

Solution:
Context:
Premises:
1. Every grimpus is a rompus.
2. Impuses are tumpuses.
3. Yumpuses are not nervous.
4. Lorpuses are melodic.
5. Yumpuses are jompuses.
6. Dumpuses are hot.
7. Every impus is a lorpus.
8. Impuses are not fruity.
9. Sterpuses are dull.
10. Sterpuses are dumpuses.
11. Every grimpus is a sterpus.
12. Each grimpus is slow.
13. Rompuses are blue.
14. Vumpuses are not dull.
15. Every sterpus is an impus.
16. Wren is a yumpus.
17. Wren is a sterpus.

Derivation:
18. Wren is a sterpus.
19. Sterpuses are dull.
20. Wren is dull.

Conclusion:
Wren is dull.

Final answer: True
```

### prontoqa random example 2 (id=5hop_0shot_random_seed3407::example1773, depth=5)

FORMAL DOC:
```
Dumpuses are muffled. Jompuses are brimpuses. Each gorpus is a rompus. Every gorpus is a lempus. Every numpus is an impus. Gorpuses are not opaque. Tumpuses are grimpuses. Every tumpus is a gorpus. Every numpus is a jompus. Each grimpus is orange. Each numpus is floral. Each brimpus is not feisty. Every dumpus is a sterpus. Wumpuses are small. Rompuses are not dull. Each jompus is sweet. Wumpuses are vumpuses. Zumpuses are opaque. Each jompus is a tumpus. Every impus is angry. Each vumpus is moderate. Wumpuses are numpuses. Each tumpus is luminous. Wren is a wumpus. Wren is a dumpus.

True or false: Wren is opaque.

Solution:
Context:
Constants:
wren = Wren

Predicates:
Angry(x) = x is angry
Brimpus(x) = x is a brimpus
Dull(x) = x is dull
Dumpus(x) = x is a dumpus
Feisty(x) = x is feisty
Floral(x) = x is floral
Gorpus(x) = x is a gorpus
Grimpus(x) = x is a grimpus
Impus(x) = x is a impus
Jompus(x) = x is a jompus
Lempus(x) = x is a lempus
Luminous(x) = x is luminous
Moderate(x) = x is moderate
Muffled(x) = x is muffled
Numpus(x) = x is a numpus
Opaque(x) = x is opaque
Orange(x) = x is orange
Rompus(x) = x is a rompus
Small(x) = x is small
Sterpus(x) = x is a sterpus
Sweet(x) = x is sweet
Tumpus(x) = x is a tumpus
Vumpus(x) = x is a vumpus
Wumpus(x) = x is a wumpus
Zumpus(x) = x is a zumpus

Premises:
1. Ax(Dumpus(x) -> Muffled(x))
2. Ax(Jompus(x) -> Brimpus(x))
3. Ax(Gorpus(x) -> Rompus(x))
4. Ax(Gorpus(x) -> Lempus(x))
5. Ax(Numpus(x) -> Impus(x))
6. Ax(Gorpus(x) -> ~Opaque(x))
7. Ax(Tumpus(x) -> Grimpus(x))
8. Ax(Tumpus(x) -> Gorpus(x))
9. Ax(Numpus(x) -> Jompus(x))
10. Ax(Grimpus(x) -> Orange(x))
11. Ax(Numpus(x) -> Floral(x))
12. Ax(Brimpus(x) -> ~Feisty(x))
13. Ax(Dumpus(x) -> Sterpus(x))
14. Ax(Wumpus(x) -> Small(x))
15. Ax(Rompus(x) -> ~Dull(x))
16. Ax(Jompus(x) -> Sweet(x))
17. Ax(Wumpus(x) -> Vumpus(x))
18. Ax(Zumpus(x) -> Opaque(x))
19. Ax(Jompus(x) -> Tumpus(x))
20. Ax(Impus(x) -> Angry(x))
21. Ax(Vumpus(x) -> Moderate(x))
22. Ax(Wumpus(x) -> Numpus(x))
23. Ax(Tumpus(x) -> Luminous(x))
24. Wumpus(wren)
25. Dumpus(wren)

Derivation:
26. Wumpus(wren) ; R,24
27. Wumpus(wren) -> Numpus(wren) ; AE,22
28. Numpus(wren) ; ->E,27,26
29. Numpus(wren) -> Jompus(wren) ; AE,9
30. Jompus(wren) ; ->E,29,28
31. Jompus(wren) -> Tumpus(wren) ; AE,19
32. Tumpus(wren) ; ->E,31,30
33. Tumpus(wren) -> Gorpus(wren) ; AE,8
34. Gorpus(wren) ; ->E,33,32
35. Gorpus(wren) -> ~Opaque(wren) ; AE,6
36. ~Opaque(wren) ; ->E,35,34

Conclusion:
~Opaque(wren)

Final answer: False
```
NL DOC:
```
Dumpuses are muffled. Jompuses are brimpuses. Each gorpus is a rompus. Every gorpus is a lempus. Every numpus is an impus. Gorpuses are not opaque. Tumpuses are grimpuses. Every tumpus is a gorpus. Every numpus is a jompus. Each grimpus is orange. Each numpus is floral. Each brimpus is not feisty. Every dumpus is a sterpus. Wumpuses are small. Rompuses are not dull. Each jompus is sweet. Wumpuses are vumpuses. Zumpuses are opaque. Each jompus is a tumpus. Every impus is angry. Each vumpus is moderate. Wumpuses are numpuses. Each tumpus is luminous. Wren is a wumpus. Wren is a dumpus.

True or false: Wren is opaque.

Solution:
Context:
Premises:
1. Dumpuses are muffled.
2. Jompuses are brimpuses.
3. Each gorpus is a rompus.
4. Every gorpus is a lempus.
5. Every numpus is an impus.
6. Gorpuses are not opaque.
7. Tumpuses are grimpuses.
8. Every tumpus is a gorpus.
9. Every numpus is a jompus.
10. Each grimpus is orange.
11. Each numpus is floral.
12. Each brimpus is not feisty.
13. Every dumpus is a sterpus.
14. Wumpuses are small.
15. Rompuses are not dull.
16. Each jompus is sweet.
17. Wumpuses are vumpuses.
18. Zumpuses are opaque.
19. Each jompus is a tumpus.
20. Every impus is angry.
21. Each vumpus is moderate.
22. Wumpuses are numpuses.
23. Each tumpus is luminous.
24. Wren is a wumpus.
25. Wren is a dumpus.

Derivation:
26. Wren is a wumpus.
27. Wumpuses are numpuses.
28. Wren is a numpus.
29. Every numpus is a jompus.
30. Wren is a jompus.
31. Each jompus is a tumpus.
32. Wren is a tumpus.
33. Every tumpus is a gorpus.
34. Wren is a gorpus.
35. Gorpuses are not opaque.
36. Wren is not opaque.

Conclusion:
Wren is not opaque.

Final answer: False
```

### prontoqa random example 3 (id=2hop_AndIntro_0shot_random_seed3407::example243, depth=1)

FORMAL DOC:
```
Sweet lempuses are lorpuses. Opaque brimpuses are gorpuses. Bright rompuses are zumpuses. Each snowy yumpus is a tumpus. Each brown vumpus is a yumpus. Melodic vumpuses are dumpuses. Fast yumpuses are grimpuses. Floral brimpuses are lempuses. Temperate lempuses are vumpuses. Sally is a rompus. Sally is floral. Sally is temperate. Sally is a brimpus. Sally is opaque.

Prove: Sally is a temperate lempus.

Solution:
Context:
Constants:
sally = Sally

Predicates:
Bright(x) = x is bright
Brimpus(x) = x is a brimpus
Brown(x) = x is brown
Dumpus(x) = x is a dumpus
Fast(x) = x is fast
Floral(x) = x is floral
Gorpus(x) = x is a gorpus
Grimpus(x) = x is a grimpus
Lempus(x) = x is a lempus
Lorpus(x) = x is a lorpus
Melodic(x) = x is melodic
Opaque(x) = x is opaque
Rompus(x) = x is a rompus
Snowy(x) = x is snowy
Sweet(x) = x is sweet
Temperate(x) = x is temperate
Tumpus(x) = x is a tumpus
Vumpus(x) = x is a vumpus
Yumpus(x) = x is a yumpus
Zumpus(x) = x is a zumpus

Premises:
1. Ax(Sweet(x) & Lempus(x) -> Lorpus(x))
2. Ax(Opaque(x) & Brimpus(x) -> Gorpus(x))
3. Ax(Bright(x) & Rompus(x) -> Zumpus(x))
4. Ax(Snowy(x) & Yumpus(x) -> Tumpus(x))
5. Ax(Brown(x) & Vumpus(x) -> Yumpus(x))
6. Ax(Melodic(x) & Vumpus(x) -> Dumpus(x))
7. Ax(Fast(x) & Yumpus(x) -> Grimpus(x))
8. Ax(Floral(x) & Brimpus(x) -> Lempus(x))
9. Ax(Temperate(x) & Lempus(x) -> Vumpus(x))
10. Rompus(sally)
11. Floral(sally)
12. Temperate(sally)
13. Brimpus(sally)
14. Opaque(sally)

Derivation:
15. Brimpus(sally) ; R,13
16. Floral(sally) ; R,11
17. Floral(sally) & Brimpus(sally) ; ∧I,16,15
18. Floral(sally) & Brimpus(sally) -> Lempus(sally) ; AE,8
19. Lempus(sally) ; ->E,18,17
20. Temperate(sally) ; R,12
21. Temperate(sally) & Lempus(sally) ; ∧I,20,19

Conclusion:
Temperate(sally) & Lempus(sally)

Final answer: True
```
NL DOC:
```
Sweet lempuses are lorpuses. Opaque brimpuses are gorpuses. Bright rompuses are zumpuses. Each snowy yumpus is a tumpus. Each brown vumpus is a yumpus. Melodic vumpuses are dumpuses. Fast yumpuses are grimpuses. Floral brimpuses are lempuses. Temperate lempuses are vumpuses. Sally is a rompus. Sally is floral. Sally is temperate. Sally is a brimpus. Sally is opaque.

Prove: Sally is a temperate lempus.

Solution:
Context:
Premises:
1. Sweet lempuses are lorpuses.
2. Opaque brimpuses are gorpuses.
3. Bright rompuses are zumpuses.
4. Each snowy yumpus is a tumpus.
5. Each brown vumpus is a yumpus.
6. Melodic vumpuses are dumpuses.
7. Fast yumpuses are grimpuses.
8. Floral brimpuses are lempuses.
9. Temperate lempuses are vumpuses.
10. Sally is a rompus.
11. Sally is floral.
12. Sally is temperate.
13. Sally is a brimpus.
14. Sally is opaque.

Derivation:
15. Sally is a brimpus.
16. Sally is floral.
17. Sally is a floral brimpus.
18. Floral brimpuses are lempuses.
19. Sally is a lempus.
20. Sally is temperate.
21. Sally is a temperate lempus.

Conclusion:
Sally is a temperate lempus.

Final answer: True
```

### prontoqa random example 4 (id=1hop_AndElim_0shot_random_seed3407::example734, depth=0)

FORMAL DOC:
```
Every grimpus is a gorpus and a shumpus. Each shumpus is an impus and a brimpus. Wumpuses are zumpuses and dumpuses. Tumpuses are windy sterpuses. Every rompus is a vumpus and a jompus. Each jompus is a wumpus and a grimpus. Vumpuses are yumpuses and numpuses. Wren is a brimpus and an impus. Wren is a grimpus and a shumpus.

Prove: Wren is a grimpus.

Solution:
Context:
Constants:
wren = Wren

Predicates:
Brimpus(x) = x is a brimpus
Dumpus(x) = x is a dumpus
Gorpus(x) = x is a gorpus
Grimpus(x) = x is a grimpus
Impus(x) = x is a impus
Jompus(x) = x is a jompus
Numpus(x) = x is a numpus
Rompus(x) = x is a rompus
Shumpus(x) = x is a shumpus
Sterpus(x) = x is a sterpus
Tumpus(x) = x is a tumpus
Vumpus(x) = x is a vumpus
Windy(x) = x is windy
Wumpus(x) = x is a wumpus
Yumpus(x) = x is a yumpus
Zumpus(x) = x is a zumpus

Premises:
1. Ax(Grimpus(x) -> Gorpus(x) & Shumpus(x))
2. Ax(Shumpus(x) -> Impus(x) & Brimpus(x))
3. Ax(Wumpus(x) -> Zumpus(x) & Dumpus(x))
4. Ax(Tumpus(x) -> Windy(x) & Sterpus(x))
5. Ax(Rompus(x) -> Vumpus(x) & Jompus(x))
6. Ax(Jompus(x) -> Wumpus(x) & Grimpus(x))
7. Ax(Vumpus(x) -> Yumpus(x) & Numpus(x))
8. Brimpus(wren) & Impus(wren)
9. Grimpus(wren) & Shumpus(wren)

Derivation:
10. Grimpus(wren) & Shumpus(wren) ; R,9
11. Grimpus(wren) ; ∧E,10

Conclusion:
Grimpus(wren)

Final answer: True
```
NL DOC:
```
Every grimpus is a gorpus and a shumpus. Each shumpus is an impus and a brimpus. Wumpuses are zumpuses and dumpuses. Tumpuses are windy sterpuses. Every rompus is a vumpus and a jompus. Each jompus is a wumpus and a grimpus. Vumpuses are yumpuses and numpuses. Wren is a brimpus and an impus. Wren is a grimpus and a shumpus.

Prove: Wren is a grimpus.

Solution:
Context:
Premises:
1. Every grimpus is a gorpus and a shumpus.
2. Each shumpus is an impus and a brimpus.
3. Wumpuses are zumpuses and dumpuses.
4. Tumpuses are windy sterpuses.
5. Every rompus is a vumpus and a jompus.
6. Each jompus is a wumpus and a grimpus.
7. Vumpuses are yumpuses and numpuses.
8. Wren is a brimpus and an impus.
9. Wren is a grimpus and a shumpus.

Derivation:
10. Wren is a grimpus and a shumpus.
11. Wren is a grimpus.

Conclusion:
Wren is a grimpus.

Final answer: True
```

### prontoqa random example 5 (id=4hop_0shot_random_seed3407::example2786, depth=4)

FORMAL DOC:
```
Each dumpus is not red. Grimpuses are not melodic. Every sterpus is a lorpus. Every impus is not liquid. Every rompus is a dumpus. Yumpuses are sterpuses. Every impus is a wumpus. Rompuses are fast. Sterpuses are not cold. Wumpuses are small. Sterpuses are impuses. Every yumpus is bright. Vumpuses are liquid. Brimpuses are fruity. Lorpuses are rainy. Rompuses are yumpuses. Yumpuses are grimpuses. Every impus is a tumpus. Every brimpus is a shumpus. Sam is a brimpus. Sam is a rompus.

True or false: Sam is liquid.

Solution:
Context:
Constants:
sam = Sam

Predicates:
Bright(x) = x is bright
Brimpus(x) = x is a brimpus
Cold(x) = x is cold
Dumpus(x) = x is a dumpus
Fast(x) = x is fast
Fruity(x) = x is fruity
Grimpus(x) = x is a grimpus
Impus(x) = x is a impus
Liquid(x) = x is liquid
Lorpus(x) = x is a lorpus
Melodic(x) = x is melodic
Rainy(x) = x is rainy
Red(x) = x is red
Rompus(x) = x is a rompus
Shumpus(x) = x is a shumpus
Small(x) = x is small
Sterpus(x) = x is a sterpus
Tumpus(x) = x is a tumpus
Vumpus(x) = x is a vumpus
Wumpus(x) = x is a wumpus
Yumpus(x) = x is a yumpus

Premises:
1. Ax(Dumpus(x) -> ~Red(x))
2. Ax(Grimpus(x) -> ~Melodic(x))
3. Ax(Sterpus(x) -> Lorpus(x))
4. Ax(Impus(x) -> ~Liquid(x))
5. Ax(Rompus(x) -> Dumpus(x))
6. Ax(Yumpus(x) -> Sterpus(x))
7. Ax(Impus(x) -> Wumpus(x))
8. Ax(Rompus(x) -> Fast(x))
9. Ax(Sterpus(x) -> ~Cold(x))
10. Ax(Wumpus(x) -> Small(x))
11. Ax(Sterpus(x) -> Impus(x))
12. Ax(Yumpus(x) -> Bright(x))
13. Ax(Vumpus(x) -> Liquid(x))
14. Ax(Brimpus(x) -> Fruity(x))
15. Ax(Lorpus(x) -> Rainy(x))
16. Ax(Rompus(x) -> Yumpus(x))
17. Ax(Yumpus(x) -> Grimpus(x))
18. Ax(Impus(x) -> Tumpus(x))
19. Ax(Brimpus(x) -> Shumpus(x))
20. Brimpus(sam)
21. Rompus(sam)

Derivation:
22. Rompus(sam) ; R,21
23. Rompus(sam) -> Yumpus(sam) ; AE,16
24. Yumpus(sam) ; ->E,23,22
25. Yumpus(sam) -> Sterpus(sam) ; AE,6
26. Sterpus(sam) ; ->E,25,24
27. Sterpus(sam) -> Impus(sam) ; AE,11
28. Impus(sam) ; ->E,27,26
29. Impus(sam) -> ~Liquid(sam) ; AE,4
30. ~Liquid(sam) ; ->E,29,28

Conclusion:
~Liquid(sam)

Final answer: False
```
NL DOC:
```
Each dumpus is not red. Grimpuses are not melodic. Every sterpus is a lorpus. Every impus is not liquid. Every rompus is a dumpus. Yumpuses are sterpuses. Every impus is a wumpus. Rompuses are fast. Sterpuses are not cold. Wumpuses are small. Sterpuses are impuses. Every yumpus is bright. Vumpuses are liquid. Brimpuses are fruity. Lorpuses are rainy. Rompuses are yumpuses. Yumpuses are grimpuses. Every impus is a tumpus. Every brimpus is a shumpus. Sam is a brimpus. Sam is a rompus.

True or false: Sam is liquid.

Solution:
Context:
Premises:
1. Each dumpus is not red.
2. Grimpuses are not melodic.
3. Every sterpus is a lorpus.
4. Every impus is not liquid.
5. Every rompus is a dumpus.
6. Yumpuses are sterpuses.
7. Every impus is a wumpus.
8. Rompuses are fast.
9. Sterpuses are not cold.
10. Wumpuses are small.
11. Sterpuses are impuses.
12. Every yumpus is bright.
13. Vumpuses are liquid.
14. Brimpuses are fruity.
15. Lorpuses are rainy.
16. Rompuses are yumpuses.
17. Yumpuses are grimpuses.
18. Every impus is a tumpus.
19. Every brimpus is a shumpus.
20. Sam is a brimpus.
21. Sam is a rompus.

Derivation:
22. Sam is a rompus.
23. Rompuses are yumpuses.
24. Sam is a yumpus.
25. Yumpuses are sterpuses.
26. Sam is a sterpus.
27. Sterpuses are impuses.
28. Sam is an impus.
29. Every impus is not liquid.
30. Sam is not liquid.

Conclusion:
Sam is not liquid.

Final answer: False
```

## Source: proofwriter

- examples scanned: 399640, converted+validated: 217844 (54.5%)
- drop reasons: `{"skip_strategy_inv-random": 45319, "skip_strategy_inv-rconc": 45579, "skip_strategy_random": 66319, "skip_strategy_rconc": 24579}`
- kept after dedup: **217810** paired docs
- answers: {'True': 108905, 'False': 108905}
- depth distribution: {0: 96972, 1: 53800, 2: 33150, 3: 18676, 4: 7606, 5: 7606}
- formal doc tokens: total 93,487,019; mean 429; p50 413; p90 651; max 1166
- NL doc tokens: total 72,048,446; mean 331; p50 313; p90 529; max 957

### proofwriter random example 1 (id=RelNoneg-OWA-D1-339::Q2, depth=0)

FORMAL DOC:
```
The bald eagle chases the cow. The bald eagle is big. The bald eagle is rough. The cat chases the cow. The cat is cold. The cat is kind. The cat is young. The cat needs the cow. The cow is kind. The tiger chases the bald eagle. The tiger chases the cow. The tiger is young. If something needs the tiger then the tiger eats the bald eagle. If the bald eagle is rough and the bald eagle chases the cow then the bald eagle needs the cow. If something needs the cat and the cat eats the bald eagle then the bald eagle is cold. If something is young then it needs the tiger. If something needs the tiger then the tiger chases the bald eagle. If something eats the bald eagle then it needs the cat.

True or false: The cat is not kind.

Solution:
Context:
Constants:
bald_eagle = the bald eagle
cat = the cat
cow = the cow
tiger = the tiger

Predicates:
Big(x) = x is big
Chases(x,y) = x chases y
Cold(x) = x is cold
Eats(x,y) = x eats y
Kind(x) = x is kind
Needs(x,y) = x needs y
Rough(x) = x is rough
Young(x) = x is young

Premises:
1. Chases(bald_eagle,cow)
2. Big(bald_eagle)
3. Rough(bald_eagle)
4. Chases(cat,cow)
5. Cold(cat)
6. Kind(cat)
7. Young(cat)
8. Needs(cat,cow)
9. Kind(cow)
10. Chases(tiger,bald_eagle)
11. Chases(tiger,cow)
12. Young(tiger)
13. Ax(Needs(x,tiger) -> Eats(tiger,bald_eagle))
14. Rough(bald_eagle) & Chases(bald_eagle,cow) -> Needs(bald_eagle,cow)
15. Ax(Needs(x,cat) & Eats(cat,bald_eagle) -> Cold(bald_eagle))
16. Ax(Young(x) -> Needs(x,tiger))
17. Ax(Needs(x,tiger) -> Chases(tiger,bald_eagle))
18. Ax(Eats(x,bald_eagle) -> Needs(x,cat))

Derivation:
19. Kind(cat) ; R,6

Conclusion:
Kind(cat)

Final answer: False
```
NL DOC:
```
The bald eagle chases the cow. The bald eagle is big. The bald eagle is rough. The cat chases the cow. The cat is cold. The cat is kind. The cat is young. The cat needs the cow. The cow is kind. The tiger chases the bald eagle. The tiger chases the cow. The tiger is young. If something needs the tiger then the tiger eats the bald eagle. If the bald eagle is rough and the bald eagle chases the cow then the bald eagle needs the cow. If something needs the cat and the cat eats the bald eagle then the bald eagle is cold. If something is young then it needs the tiger. If something needs the tiger then the tiger chases the bald eagle. If something eats the bald eagle then it needs the cat.

True or false: The cat is not kind.

Solution:
Context:
Premises:
1. The bald eagle chases the cow.
2. The bald eagle is big.
3. The bald eagle is rough.
4. The cat chases the cow.
5. The cat is cold.
6. The cat is kind.
7. The cat is young.
8. The cat needs the cow.
9. The cow is kind.
10. The tiger chases the bald eagle.
11. The tiger chases the cow.
12. The tiger is young.
13. If something needs the tiger then the tiger eats the bald eagle.
14. If the bald eagle is rough and the bald eagle chases the cow then the bald eagle needs the cow.
15. If something needs the cat and the cat eats the bald eagle then the bald eagle is cold.
16. If something is young then it needs the tiger.
17. If something needs the tiger then the tiger chases the bald eagle.
18. If something eats the bald eagle then it needs the cat.

Derivation:
19. We are given: The cat is kind.

Conclusion:
The cat is kind.

Final answer: False
```

### proofwriter random example 2 (id=AttNoneg-OWA-D0-1962::Q1, depth=0)

FORMAL DOC:
```
Erin is big. Erin is cold. Erin is furry. Erin is kind. Erin is rough. Erin is white. Erin is young. All big things are young.

True or false: Erin is furry.

Solution:
Context:
Constants:
erin = Erin

Predicates:
Big(x) = x is big
Cold(x) = x is cold
Furry(x) = x is furry
Kind(x) = x is kind
Rough(x) = x is rough
White(x) = x is white
Young(x) = x is young

Premises:
1. Big(erin)
2. Cold(erin)
3. Furry(erin)
4. Kind(erin)
5. Rough(erin)
6. White(erin)
7. Young(erin)
8. Ax(Big(x) -> Young(x))

Derivation:
9. Furry(erin) ; R,3

Conclusion:
Furry(erin)

Final answer: True
```
NL DOC:
```
Erin is big. Erin is cold. Erin is furry. Erin is kind. Erin is rough. Erin is white. Erin is young. All big things are young.

True or false: Erin is furry.

Solution:
Context:
Premises:
1. Erin is big.
2. Erin is cold.
3. Erin is furry.
4. Erin is kind.
5. Erin is rough.
6. Erin is white.
7. Erin is young.
8. All big things are young.

Derivation:
9. We are given: Erin is furry.

Conclusion:
Erin is furry.

Final answer: True
```

### proofwriter random example 3 (id=AttNeg-OWA-D3-611::Q7, depth=3)

FORMAL DOC:
```
Bob is cold. Bob is quiet. Bob is round. Charlie is quiet. Erin is quiet. Erin is young. Harry is not young. Rough people are cold. All young, quiet people are cold. Quiet, blue people are round. Green people are rough. If someone is quiet then they are rough. Cold, rough people are not green.

True or false: Charlie is not green.

Solution:
Context:
Constants:
bob = Bob
charlie = Charlie
erin = Erin
harry = Harry

Predicates:
Blue(x) = x is blue
Cold(x) = x is cold
Green(x) = x is green
Quiet(x) = x is quiet
Rough(x) = x is rough
Round(x) = x is round
Young(x) = x is young

Premises:
1. Cold(bob)
2. Quiet(bob)
3. Round(bob)
4. Quiet(charlie)
5. Quiet(erin)
6. Young(erin)
7. ~Young(harry)
8. Ax(Rough(x) -> Cold(x))
9. Ax(Young(x) & Quiet(x) -> Cold(x))
10. Ax(Quiet(x) & Blue(x) -> Round(x))
11. Ax(Green(x) -> Rough(x))
12. Ax(Quiet(x) -> Rough(x))
13. Ax(Cold(x) & Rough(x) -> ~Green(x))

Derivation:
14. Quiet(charlie) ; R,4
15. Quiet(charlie) -> Rough(charlie) ; AE,12
16. Rough(charlie) ; ->E,15,14
17. Rough(charlie) -> Cold(charlie) ; AE,8
18. Cold(charlie) ; ->E,17,16
19. Cold(charlie) & Rough(charlie) ; ∧I,18,16
20. Cold(charlie) & Rough(charlie) -> ~Green(charlie) ; AE,13
21. ~Green(charlie) ; ->E,20,19

Conclusion:
~Green(charlie)

Final answer: True
```
NL DOC:
```
Bob is cold. Bob is quiet. Bob is round. Charlie is quiet. Erin is quiet. Erin is young. Harry is not young. Rough people are cold. All young, quiet people are cold. Quiet, blue people are round. Green people are rough. If someone is quiet then they are rough. Cold, rough people are not green.

True or false: Charlie is not green.

Solution:
Context:
Premises:
1. Bob is cold.
2. Bob is quiet.
3. Bob is round.
4. Charlie is quiet.
5. Erin is quiet.
6. Erin is young.
7. Harry is not young.
8. Rough people are cold.
9. All young, quiet people are cold.
10. Quiet, blue people are round.
11. Green people are rough.
12. If someone is quiet then they are rough.
13. Cold, rough people are not green.

Derivation:
14. We are given: Charlie is quiet.
15. Instantiating rule 12 for Charlie: if Charlie is quiet, then Charlie is rough.
16. Therefore, Charlie is rough.
17. Instantiating rule 8 for Charlie: if Charlie is rough, then Charlie is cold.
18. Therefore, Charlie is cold.
19. Combining: Charlie is cold and Charlie is rough.
20. Instantiating rule 13 for Charlie: if Charlie is cold and Charlie is rough, then Charlie is not green.
21. Therefore, Charlie is not green.

Conclusion:
Charlie is not green.

Final answer: True
```

### proofwriter random example 4 (id=RelNoneg-OWA-D0-1446::Q2, depth=0)

FORMAL DOC:
```
The dog chases the tiger. The dog is big. The dog is red. The dog is round. The dog needs the tiger. The dog visits the tiger. The tiger chases the dog. The tiger is big. The tiger is green. The tiger is red. The tiger needs the dog. The tiger visits the dog. If someone needs the dog and they are round then the dog chases the tiger. If someone chases the tiger then they visit the tiger. If someone is cold then they need the dog. If someone visits the dog then the dog is round. If someone chases the dog then the dog needs the tiger. If someone chases the tiger and the tiger visits the dog then they chase the dog.

True or false: The tiger does not visit the dog.

Solution:
Context:
Constants:
dog = the dog
tiger = the tiger

Predicates:
Big(x) = x is big
Chases(x,y) = x chases y
Cold(x) = x is cold
Green(x) = x is green
Needs(x,y) = x needs y
Red(x) = x is red
Round(x) = x is round
Visits(x,y) = x visits y

Premises:
1. Chases(dog,tiger)
2. Big(dog)
3. Red(dog)
4. Round(dog)
5. Needs(dog,tiger)
6. Visits(dog,tiger)
7. Chases(tiger,dog)
8. Big(tiger)
9. Green(tiger)
10. Red(tiger)
11. Needs(tiger,dog)
12. Visits(tiger,dog)
13. Ax(Needs(x,dog) & Round(x) -> Chases(dog,tiger))
14. Ax(Chases(x,tiger) -> Visits(x,tiger))
15. Ax(Cold(x) -> Needs(x,dog))
16. Ax(Visits(x,dog) -> Round(dog))
17. Ax(Chases(x,dog) -> Needs(dog,tiger))
18. Ax(Chases(x,tiger) & Visits(tiger,dog) -> Chases(x,dog))

Derivation:
19. Visits(tiger,dog) ; R,12

Conclusion:
Visits(tiger,dog)

Final answer: False
```
NL DOC:
```
The dog chases the tiger. The dog is big. The dog is red. The dog is round. The dog needs the tiger. The dog visits the tiger. The tiger chases the dog. The tiger is big. The tiger is green. The tiger is red. The tiger needs the dog. The tiger visits the dog. If someone needs the dog and they are round then the dog chases the tiger. If someone chases the tiger then they visit the tiger. If someone is cold then they need the dog. If someone visits the dog then the dog is round. If someone chases the dog then the dog needs the tiger. If someone chases the tiger and the tiger visits the dog then they chase the dog.

True or false: The tiger does not visit the dog.

Solution:
Context:
Premises:
1. The dog chases the tiger.
2. The dog is big.
3. The dog is red.
4. The dog is round.
5. The dog needs the tiger.
6. The dog visits the tiger.
7. The tiger chases the dog.
8. The tiger is big.
9. The tiger is green.
10. The tiger is red.
11. The tiger needs the dog.
12. The tiger visits the dog.
13. If someone needs the dog and they are round then the dog chases the tiger.
14. If someone chases the tiger then they visit the tiger.
15. If someone is cold then they need the dog.
16. If someone visits the dog then the dog is round.
17. If someone chases the dog then the dog needs the tiger.
18. If someone chases the tiger and the tiger visits the dog then they chase the dog.

Derivation:
19. We are given: The tiger visits the dog.

Conclusion:
The tiger visits the dog.

Final answer: False
```

### proofwriter random example 5 (id=RelNoneg-OWA-D2-1282::Q2, depth=0)

FORMAL DOC:
```
The dog is big. The dog is cold. The dog is nice. The dog is red. The dog likes the rabbit. The dog needs the rabbit. The rabbit chases the dog. The rabbit is big. The rabbit is cold. The rabbit is red. The rabbit likes the dog. The rabbit needs the dog. If something is big then it likes the dog. If the dog is young and the dog chases the rabbit then the rabbit needs the dog. If the rabbit needs the dog then the rabbit likes the dog. If something likes the dog then it chases the dog. If something needs the dog then it chases the dog. If something is young then it likes the dog.

True or false: The rabbit does not need the dog.

Solution:
Context:
Constants:
dog = the dog
rabbit = the rabbit

Predicates:
Big(x) = x is big
Chases(x,y) = x chases y
Cold(x) = x is cold
Likes(x,y) = x likes y
Needs(x,y) = x needs y
Nice(x) = x is nice
Red(x) = x is red
Young(x) = x is young

Premises:
1. Big(dog)
2. Cold(dog)
3. Nice(dog)
4. Red(dog)
5. Likes(dog,rabbit)
6. Needs(dog,rabbit)
7. Chases(rabbit,dog)
8. Big(rabbit)
9. Cold(rabbit)
10. Red(rabbit)
11. Likes(rabbit,dog)
12. Needs(rabbit,dog)
13. Ax(Big(x) -> Likes(x,dog))
14. Young(dog) & Chases(dog,rabbit) -> Needs(rabbit,dog)
15. Needs(rabbit,dog) -> Likes(rabbit,dog)
16. Ax(Likes(x,dog) -> Chases(x,dog))
17. Ax(Needs(x,dog) -> Chases(x,dog))
18. Ax(Young(x) -> Likes(x,dog))

Derivation:
19. Needs(rabbit,dog) ; R,12

Conclusion:
Needs(rabbit,dog)

Final answer: False
```
NL DOC:
```
The dog is big. The dog is cold. The dog is nice. The dog is red. The dog likes the rabbit. The dog needs the rabbit. The rabbit chases the dog. The rabbit is big. The rabbit is cold. The rabbit is red. The rabbit likes the dog. The rabbit needs the dog. If something is big then it likes the dog. If the dog is young and the dog chases the rabbit then the rabbit needs the dog. If the rabbit needs the dog then the rabbit likes the dog. If something likes the dog then it chases the dog. If something needs the dog then it chases the dog. If something is young then it likes the dog.

True or false: The rabbit does not need the dog.

Solution:
Context:
Premises:
1. The dog is big.
2. The dog is cold.
3. The dog is nice.
4. The dog is red.
5. The dog likes the rabbit.
6. The dog needs the rabbit.
7. The rabbit chases the dog.
8. The rabbit is big.
9. The rabbit is cold.
10. The rabbit is red.
11. The rabbit likes the dog.
12. The rabbit needs the dog.
13. If something is big then it likes the dog.
14. If the dog is young and the dog chases the rabbit then the rabbit needs the dog.
15. If the rabbit needs the dog then the rabbit likes the dog.
16. If something likes the dog then it chases the dog.
17. If something needs the dog then it chases the dog.
18. If something is young then it likes the dog.

Derivation:
19. We are given: The rabbit needs the dog.

Conclusion:
The rabbit needs the dog.

Final answer: False
```

## Totals

- paired docs: 451,162
- formal tokens: 298,884,792
- NL tokens: 213,376,814
- combined (formal+NL): 512,261,606
