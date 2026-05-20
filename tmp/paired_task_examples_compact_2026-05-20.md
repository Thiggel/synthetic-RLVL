# official_igsm
answer: 9
validity: format=1.0, internal=1.0, grounded=1.0, citation_free_grounded=1.0
metadata: {'dataset_family': 'official_igsm', 'depth': 2, 'official_n_op': 2, 'official_problem_text': " The number of each Tool Backpack's Diary equals 12 times as much as the sum of each Briefcase Backpack's Colored Paper and each Commuter Backpack's Construction Paper. The number of each Briefcase Backpack's Construction Paper equals 11 more than each Briefcase Backpack's Diary. The number of each Commuter Backpack's Construction Paper equals 1 more than each Briefcase Backpack's Colored Paper. The number of each Briefcase Backpack's Diary equals 21. The number of each Tool Backpack's Colored Paper equals 19 times as much as the difference of each Briefcase Backpack's Construction Paper and each Briefcase Backpack's Diary. The number of each Briefcase Backpack's Colored Paper equals 10. The number of each Commuter Backpack's Colored Paper equals 3 more than each Briefcase Backpack's Construction Paper. How many Construction Paper does Briefcase Backpack have?", 'official_solution_text': " Define Briefcase Backpack's Diary as e; so e = 21. Define Briefcase Backpack's Construction Paper as W; so W = 11 + e = 11 + 21 = 9.", 'official_answer_text': '9', 'equation_chains': [{'original_var': 'e', 'var': 'v_e', 'expr': '21', 'result': 21, 'official_text': 'e = 21'}, {'original_var': 'W', 'var': 'v_W', 'expr': '11 + v_e', 'result': 9, 'official_text': 'W = 11 + e = 11 + 21 = 9'}], 'gold_answer': '9', 'modulus': 23, 'logic_trace_valid': True}
```text
<question>
1. The number of each Tool Backpack's Diary equals 12 times as much as the sum of each Briefcase Backpack's Colored Paper and each Commuter Backpack's Construction Paper.
2. The number of each Briefcase Backpack's Construction Paper equals 11 more than each Briefcase Backpack's Diary.
3. The number of each Commuter Backpack's Construction Paper equals 1 more than each Briefcase Backpack's Colored Paper.
4. The number of each Briefcase Backpack's Diary equals 21.
5. The number of each Tool Backpack's Colored Paper equals 19 times as much as the difference of each Briefcase Backpack's Construction Paper and each Briefcase Backpack's Diary.
6. The number of each Briefcase Backpack's Colored Paper equals 10.
7. The number of each Commuter Backpack's Colored Paper equals 3 more than each Briefcase Backpack's Construction Paper.
How many Construction Paper does Briefcase Backpack have?
</question>

<formal>
<constants>
v_W = official iGSM variable W
v_e = official iGSM variable e
</constants>
<predicates>

</predicates>
<premises>
v_e = 21
v_W = 11 + v_e
</premises>
<proof>
v_e = 21 ; R,1
v_W = 11 + v_e ; R,2
v_W = 11 + 21 ; =E,3,4
v_W = 9 ; MOD23,5
</proof>
<conclusion>
v_W = 9
</conclusion>
</formal>
<answer>
9
</answer>
```

# maze_navigation
answer: granite
validity: format=1.0, internal=1.0, grounded=1.0, citation_free_grounded=1.0
metadata: {'dataset_family': 'maze_navigation', 'task_structure': 'keyed_constrained_graph', 'depth': 3, 'start': 'lantern', 'gold_path': ['lantern', 'indigo', 'pearl', 'granite'], 'key_path': ['yellow', 'purple', 'white', 'green'], 'blocked_edges': [{'step': 0, 'from_room': 'lantern', 'required_key': 'white', 'to_room': 'silver'}, {'step': 0, 'from_room': 'lantern', 'required_key': 'green', 'to_room': 'timber'}, {'step': 0, 'from_room': 'lantern', 'required_key': 'orange', 'to_room': 'aurora'}, {'step': 0, 'from_room': 'lantern', 'required_key': 'blue', 'to_room': 'ochre'}, {'step': 1, 'from_room': 'indigo', 'required_key': 'red', 'to_room': 'heather'}, {'step': 1, 'from_room': 'indigo', 'required_key': 'orange', 'to_room': 'keystone'}, {'step': 1, 'from_room': 'indigo', 'required_key': 'white', 'to_room': 'willow'}, {'step': 1, 'from_room': 'indigo', 'required_key': 'yellow', 'to_room': 'citadel'}, {'step': 2, 'from_room': 'pearl', 'required_key': 'black', 'to_room': 'ruby'}, {'step': 2, 'from_room': 'pearl', 'required_key': 'blue', 'to_room': 'umber'}, {'step': 2, 'from_room': 'pearl', 'required_key': 'green', 'to_room': 'prairie'}, {'step': 2, 'from_room': 'pearl', 'required_key': 'silver', 'to_room': 'estate'}], 'treasure_rooms': ['laurel', 'nectar', 'forest', 'granite', 'linen'], 'unreachable_treasure_rooms': ['forest', 'linen', 'nectar', 'laurel'], 'requires_key_tracking': True, 'solution_rule_for_all_treasures': True, 'gold_answer': 'granite', 'logic_trace_valid': True}
```text
<question>
1. The explorer starts in room lantern.
2. The explorer initially holds the yellow key.
3. There is a door from lantern to silver that requires the white key.
4. If the explorer is in lantern after 0 moves, has the white key, and the matching door leads to silver, then silver is reachable after 1 moves.
5. There is a door from lantern to aurora that requires the orange key.
6. If the explorer is in lantern after 0 moves, has the orange key, and the matching door leads to aurora, then aurora is reachable after 1 moves.
7. There is a door from lantern to indigo that requires the yellow key.
8. If the explorer is in lantern after 0 moves, has the yellow key, and the matching door leads to indigo, then indigo is reachable after 1 moves.
9. Room indigo contains the purple key.
10. If the explorer reaches indigo after 1 moves and indigo contains the purple key, then the explorer has the purple key after 1 moves.
11. There is a door from lantern to timber that requires the green key.
12. If the explorer is in lantern after 0 moves, has the green key, and the matching door leads to timber, then timber is reachable after 1 moves.
13. There is a door from lantern to ochre that requires the blue key.
14. If the explorer is in lantern after 0 moves, has the blue key, and the matching door leads to ochre, then ochre is reachable after 1 moves.
15. There is a door from indigo to heather that requires the red key.
16. If the explorer is in indigo after 1 moves, has the red key, and the matching door leads to heather, then heather is reachable after 2 moves.
17. There is a door from indigo to citadel that requires the yellow key.
18. If the explorer is in indigo after 1 moves, has the yellow key, and the matching door leads to citadel, then citadel is reachable after 2 moves.
19. There is a door from indigo to pearl that requires the purple key.
20. If the explorer is in indigo after 1 moves, has the purple key, and the matching door leads to pearl, then pearl is reachable after 2 moves.
21. Room pearl contains the white key.
22. If the explorer reaches pearl after 2 moves and pearl contains the white key, then the explorer has the white key after 2 moves.
23. There is a door from indigo to keystone that requires the orange key.
24. If the explorer is in indigo after 1 moves, has the orange key, and the matching door leads to keystone, then keystone is reachable after 2 moves.
25. There is a door from indigo to willow that requires the white key.
26. If the explorer is in indigo after 1 moves, has the white key, and the matching door leads to willow, then willow is reachable after 2 moves.
27. There is a door from pearl to prairie that requires the green key.
28. If the explorer is in pearl after 2 moves, has the green key, and the matching door leads to prairie, then prairie is reachable after 3 moves.
29. There is a door from pearl to estate that requires the silver key.
30. If the explorer is in pearl after 2 moves, has the silver key, and the matching door leads to estate, then estate is reachable after 3 moves.
31. There is a door from pearl to umber that requires the blue key.
32. If the explorer is in pearl after 2 moves, has the blue key, and the matching door leads to umber, then umber is reachable after 3 moves.
33. There is a door from pearl to granite that requires the white key.
34. If the explorer is in pearl after 2 moves, has the white key, and the matching door leads to granite, then granite is reachable after 3 moves.
35. Room granite contains the green key.
36. If the explorer reaches granite after 3 moves and granite contains the green key, then the explorer has the green key after 3 moves.
37. There is a door from pearl to ruby that requires the black key.
38. If the explorer is in pearl after 2 moves, has the black key, and the matching door leads to ruby, then ruby is reachable after 3 moves.
39. Room laurel contains a marked treasure.
40. Room nectar contains a marked treasure.
41. Room forest contains a marked treasure.
42. Room granite contains a marked treasure.
43. Room linen contains a marked treasure.
44. If room laurel is reachable after exactly 3 key-constrained moves and contains a treasure, then the treasure in laurel is found.
45. If room nectar is reachable after exactly 3 key-constrained moves and contains a treasure, then the treasure in nectar is found.
46. If room forest is reachable after exactly 3 key-constrained moves and contains a treasure, then the treasure in forest is found.
47. If room granite is reachable after exactly 3 key-constrained moves and contains a treasure, then the treasure in granite is found.
48. If room linen is reachable after exactly 3 key-constrained moves and contains a treasure, then the treasure in linen is found.
The rooms form a locked maze. The explorer may use only doors whose key they currently hold, and entering a room may reveal the next key. Which marked treasure room is reachable after exactly 3 moves?
</question>

<formal>
<constants>
aurora = maze room aurora
citadel = maze room citadel
estate = maze room estate
forest = maze room forest
granite = maze room granite
heather = maze room heather
indigo = maze room indigo
keystone = maze room keystone
lantern = maze room lantern
laurel = maze room laurel
linen = maze room linen
nectar = maze room nectar
ochre = maze room ochre
pearl = maze room pearl
prairie = maze room prairie
ruby = maze room ruby
silver = maze room silver
timber = maze room timber
umber = maze room umber
willow = maze room willow
black = maze key black
blue = maze key blue
green = maze key green
orange = maze key orange
purple = maze key purple
red = maze key red
silver = maze key silver
white = maze key white
yellow = maze key yellow
</constants>
<predicates>
AtN(x): the explorer can be at room x after N moves
HaveN(x): the explorer has key x after N moves
Door(x,y,z): there is a door from room x to room z requiring key y
Finds(x,y): room x contains key y
Treasure(x): room x contains a marked treasure
Found(x): the reachable marked treasure is in room x
</predicates>
<premises>
At0(lantern)
Have0(yellow)
Door(lantern,white,silver)
At0(lantern) & Have0(white) & Door(lantern,white,silver) -> At1(silver)
Door(lantern,orange,aurora)
At0(lantern) & Have0(orange) & Door(lantern,orange,aurora) -> At1(aurora)
Door(lantern,yellow,indigo)
At0(lantern) & Have0(yellow) & Door(lantern,yellow,indigo) -> At1(indigo)
Finds(indigo,purple)
At1(indigo) & Finds(indigo,purple) -> Have1(purple)
Door(lantern,green,timber)
At0(lantern) & Have0(green) & Door(lantern,green,timber) -> At1(timber)
Door(lantern,blue,ochre)
At0(lantern) & Have0(blue) & Door(lantern,blue,ochre) -> At1(ochre)
Door(indigo,red,heather)
At1(indigo) & Have1(red) & Door(indigo,red,heather) -> At2(heather)
Door(indigo,yellow,citadel)
At1(indigo) & Have1(yellow) & Door(indigo,yellow,citadel) -> At2(citadel)
Door(indigo,purple,pearl)
At1(indigo) & Have1(purple) & Door(indigo,purple,pearl) -> At2(pearl)
Finds(pearl,white)
At2(pearl) & Finds(pearl,white) -> Have2(white)
Door(indigo,orange,keystone)
At1(indigo) & Have1(orange) & Door(indigo,orange,keystone) -> At2(keystone)
Door(indigo,white,willow)
At1(indigo) & Have1(white) & Door(indigo,white,willow) -> At2(willow)
Door(pearl,green,prairie)
At2(pearl) & Have2(green) & Door(pearl,green,prairie) -> At3(prairie)
Door(pearl,silver,estate)
At2(pearl) & Have2(silver) & Door(pearl,silver,estate) -> At3(estate)
Door(pearl,blue,umber)
At2(pearl) & Have2(blue) & Door(pearl,blue,umber) -> At3(umber)
Door(pearl,white,granite)
At2(pearl) & Have2(white) & Door(pearl,white,granite) -> At3(granite)
Finds(granite,green)
At3(granite) & Finds(granite,green) -> Have3(green)
Door(pearl,black,ruby)
At2(pearl) & Have2(black) & Door(pearl,black,ruby) -> At3(ruby)
Treasure(laurel)
Treasure(nectar)
Treasure(forest)
Treasure(granite)
Treasure(linen)
At3(laurel) & Treasure(laurel) -> Found(laurel)
At3(nectar) & Treasure(nectar) -> Found(nectar)
At3(forest) & Treasure(forest) -> Found(forest)
At3(granite) & Treasure(granite) -> Found(granite)
At3(linen) & Treasure(linen) -> Found(linen)
</premises>
<proof>
At0(lantern) ; R,1
Have0(yellow) ; R,2
Door(lantern,yellow,indigo) ; R,7
At0(lantern) & Have0(yellow) ; ∧I,49,50
At0(lantern) & Have0(yellow) & Door(lantern,yellow,indigo) ; ∧I,52,51
At1(indigo) ; ->E,8,53
Finds(indigo,purple) ; R,9
At1(indigo) & Finds(indigo,purple) ; ∧I,54,55
Have1(purple) ; ->E,10,56
Door(indigo,purple,pearl) ; R,19
At1(indigo) & Have1(purple) ; ∧I,54,57
At1(indigo) & Have1(purple) & Door(indigo,purple,pearl) ; ∧I,59,58
At2(pearl) ; ->E,20,60
Finds(pearl,white) ; R,21
At2(pearl) & Finds(pearl,white) ; ∧I,61,62
Have2(white) ; ->E,22,63
Door(pearl,white,granite) ; R,33
At2(pearl) & Have2(white) ; ∧I,61,64
At2(pearl) & Have2(white) & Door(pearl,white,granite) ; ∧I,66,65
At3(granite) ; ->E,34,67
Finds(granite,green) ; R,35
At3(granite) & Finds(granite,green) ; ∧I,68,69
Have3(green) ; ->E,36,70
Treasure(granite) ; R,42
At3(granite) & Treasure(granite) ; ∧I,68,72
Found(granite) ; ->E,47,73
</proof>
<conclusion>
Found(granite)
</conclusion>
</formal>
<answer>
granite
</answer>
```

# attribute_constraints
answer: orange-black-blue-green
validity: format=1.0, internal=1.0, grounded=1.0, citation_free_grounded=1.0
metadata: {'dataset_family': 'attribute_constraints', 'task_structure': 'multi_input_slot_constraint_dag', 'depth': 4, 'slot_count': 4, 'base_slot_count': 2, 'palette': ['red', 'blue', 'green', 'yellow', 'white', 'black', 'orange'], 'slots': [{'slot': 'slot_0', 'value': 'orange'}, {'slot': 'slot_1', 'value': 'black'}, {'slot': 'slot_2', 'value': 'blue'}, {'slot': 'slot_3', 'value': 'green'}], 'constraints': [{'target_index': 2, 'dep_a': 0, 'dep_b': 1, 'slot_a': 'slot_0', 'value_a': 'orange', 'slot_b': 'slot_1', 'value_b': 'black', 'target_slot': 'slot_2', 'target_value': 'blue'}, {'target_index': 3, 'dep_a': 1, 'dep_b': 2, 'slot_a': 'slot_1', 'value_a': 'black', 'slot_b': 'slot_2', 'value_b': 'blue', 'target_slot': 'slot_3', 'target_value': 'green'}], 'decoy_constraints': [{'target_index': 2, 'slot_a': 'slot_0', 'value_a': 'green', 'slot_b': 'slot_1', 'value_b': 'blue', 'target_slot': 'slot_2', 'target_value': 'yellow'}, {'target_index': 2, 'slot_a': 'slot_0', 'value_a': 'red', 'slot_b': 'slot_1', 'value_b': 'black', 'target_slot': 'slot_2', 'target_value': 'white'}, {'target_index': 2, 'slot_a': 'slot_0', 'value_a': 'orange', 'slot_b': 'slot_1', 'value_b': 'red', 'target_slot': 'slot_2', 'target_value': 'orange'}, {'target_index': 2, 'slot_a': 'slot_0', 'value_a': 'yellow', 'slot_b': 'slot_1', 'value_b': 'green', 'target_slot': 'slot_2', 'target_value': 'yellow'}, {'target_index': 3, 'slot_a': 'slot_1', 'value_a': 'yellow', 'slot_b': 'slot_2', 'value_b': 'white', 'target_slot': 'slot_3', 'target_value': 'black'}, {'target_index': 3, 'slot_a': 'slot_1', 'value_a': 'black', 'slot_b': 'slot_2', 'value_b': 'green', 'target_slot': 'slot_3', 'target_value': 'yellow'}, {'target_index': 3, 'slot_a': 'slot_1', 'value_a': 'white', 'slot_b': 'slot_2', 'value_b': 'green', 'target_slot': 'slot_3', 'target_value': 'orange'}, {'target_index': 3, 'slot_a': 'slot_1', 'value_a': 'black', 'slot_b': 'slot_2', 'value_b': 'orange', 'target_slot': 'slot_3', 'target_value': 'blue'}], 'gold_answer': 'orange-black-blue-green', 'logic_trace_valid': True, 'grounded_validity_supported': True}
```text
<question>
1. slot_0 has value orange.
2. slot_1 has value black.
3. The joint constraint says: if slot_0 is orange and slot_1 is black, then slot_2 is blue.
4. If both prerequisite slot values hold and the matching joint constraint is present, then slot_2 has blue.
5. A decoy joint constraint says: if slot_0 is green and slot_1 is blue, then slot_2 is yellow.
6. If the decoy prerequisite values held and the decoy constraint applied, then slot_2 would be yellow.
7. A decoy joint constraint says: if slot_0 is red and slot_1 is black, then slot_2 is white.
8. If the decoy prerequisite values held and the decoy constraint applied, then slot_2 would be white.
9. A decoy joint constraint says: if slot_0 is orange and slot_1 is red, then slot_2 is orange.
10. If the decoy prerequisite values held and the decoy constraint applied, then slot_2 would be orange.
11. A decoy joint constraint says: if slot_0 is yellow and slot_1 is green, then slot_2 is yellow.
12. If the decoy prerequisite values held and the decoy constraint applied, then slot_2 would be yellow.
13. The joint constraint says: if slot_1 is black and slot_2 is blue, then slot_3 is green.
14. If both prerequisite slot values hold and the matching joint constraint is present, then slot_3 has green.
15. A decoy joint constraint says: if slot_1 is yellow and slot_2 is white, then slot_3 is black.
16. If the decoy prerequisite values held and the decoy constraint applied, then slot_3 would be black.
17. A decoy joint constraint says: if slot_1 is black and slot_2 is green, then slot_3 is yellow.
18. If the decoy prerequisite values held and the decoy constraint applied, then slot_3 would be yellow.
19. A decoy joint constraint says: if slot_1 is white and slot_2 is green, then slot_3 is orange.
20. If the decoy prerequisite values held and the decoy constraint applied, then slot_3 would be orange.
21. A decoy joint constraint says: if slot_1 is black and slot_2 is orange, then slot_3 is blue.
22. If the decoy prerequisite values held and the decoy constraint applied, then slot_3 would be blue.
Starting from the given slot values, apply the joint constraints. Which values fill all slots?
</question>

<formal>
<constants>
slot_0 = attribute slot 0
slot_1 = attribute slot 1
slot_2 = attribute slot 2
slot_3 = attribute slot 3
black = attribute value black
blue = attribute value blue
green = attribute value green
orange = attribute value orange
red = attribute value red
white = attribute value white
yellow = attribute value yellow
</constants>
<predicates>
Value(x,y): slot x has value y
Constraint(x,y,z,w,u,v): values y and w at slots x and z jointly force value v at slot u
</predicates>
<premises>
Value(slot_0,orange)
Value(slot_1,black)
Constraint(slot_0,orange,slot_1,black,slot_2,blue)
Value(slot_0,orange) & Value(slot_1,black) & Constraint(slot_0,orange,slot_1,black,slot_2,blue) -> Value(slot_2,blue)
Constraint(slot_0,green,slot_1,blue,slot_2,yellow)
Value(slot_0,green) & Value(slot_1,blue) & Constraint(slot_0,green,slot_1,blue,slot_2,yellow) -> Value(slot_2,yellow)
Constraint(slot_0,red,slot_1,black,slot_2,white)
Value(slot_0,red) & Value(slot_1,black) & Constraint(slot_0,red,slot_1,black,slot_2,white) -> Value(slot_2,white)
Constraint(slot_0,orange,slot_1,red,slot_2,orange)
Value(slot_0,orange) & Value(slot_1,red) & Constraint(slot_0,orange,slot_1,red,slot_2,orange) -> Value(slot_2,orange)
Constraint(slot_0,yellow,slot_1,green,slot_2,yellow)
Value(slot_0,yellow) & Value(slot_1,green) & Constraint(slot_0,yellow,slot_1,green,slot_2,yellow) -> Value(slot_2,yellow)
Constraint(slot_1,black,slot_2,blue,slot_3,green)
Value(slot_1,black) & Value(slot_2,blue) & Constraint(slot_1,black,slot_2,blue,slot_3,green) -> Value(slot_3,green)
Constraint(slot_1,yellow,slot_2,white,slot_3,black)
Value(slot_1,yellow) & Value(slot_2,white) & Constraint(slot_1,yellow,slot_2,white,slot_3,black) -> Value(slot_3,black)
Constraint(slot_1,black,slot_2,green,slot_3,yellow)
Value(slot_1,black) & Value(slot_2,green) & Constraint(slot_1,black,slot_2,green,slot_3,yellow) -> Value(slot_3,yellow)
Constraint(slot_1,white,slot_2,green,slot_3,orange)
Value(slot_1,white) & Value(slot_2,green) & Constraint(slot_1,white,slot_2,green,slot_3,orange) -> Value(slot_3,orange)
Constraint(slot_1,black,slot_2,orange,slot_3,blue)
Value(slot_1,black) & Value(slot_2,orange) & Constraint(slot_1,black,slot_2,orange,slot_3,blue) -> Value(slot_3,blue)
</premises>
<proof>
Value(slot_0,orange) ; R,1
Value(slot_1,black) ; R,2
Constraint(slot_0,orange,slot_1,black,slot_2,blue) ; R,3
Value(slot_0,orange) & Value(slot_1,black) ; ∧I,23,24
Value(slot_0,orange) & Value(slot_1,black) & Constraint(slot_0,orange,slot_1,black,slot_2,blue) ; ∧I,26,25
Value(slot_2,blue) ; ->E,4,27
Constraint(slot_1,black,slot_2,blue,slot_3,green) ; R,13
Value(slot_1,black) & Value(slot_2,blue) ; ∧I,24,28
Value(slot_1,black) & Value(slot_2,blue) & Constraint(slot_1,black,slot_2,blue,slot_3,green) ; ∧I,30,29
Value(slot_3,green) ; ->E,14,31
Value(slot_0,orange) & Value(slot_1,black) ; ∧I,23,24
Value(slot_0,orange) & Value(slot_1,black) & Value(slot_2,blue) ; ∧I,33,28
Value(slot_0,orange) & Value(slot_1,black) & Value(slot_2,blue) & Value(slot_3,green) ; ∧I,34,32
</proof>
<conclusion>
Value(slot_0,orange) & Value(slot_1,black) & Value(slot_2,blue) & Value(slot_3,green)
</conclusion>
</formal>
<answer>
orange-black-blue-green
</answer>
```

