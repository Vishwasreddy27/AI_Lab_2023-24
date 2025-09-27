# Ex.No: 12 Planning –  Monkey Banana Problem
### DATE: 27.09.2025                                                                           
### REGISTER NUMBER : 212222060111
### AIM: 
To find the sequence of plan for Monkey Banana problem using PDDL Editor.
###  Algorithm:
Step 1:  Start the program <br> 
Step 2 : Create a domain for Monkey Banana Problem. <br> 
Step 3:  Create a domain by specifying predicates. <br> 
Step 4: Specify the actions GOTO, CLIMB, PUSH-BOX, GET-KNIFE, GRAB-BANANAS in Monkey Banana problem.<br>  
Step 5:   Define a problem for Monkey Banana problem.<br> 
Step 6:  Obtain the plan for given problem.<br> 
Step 7: Stop the program.<br> 
### Program:
```
(define (domain monkey)
 (:requirements :strips)
 (:constants monkey box knife bananas glass waterfountain)
 (:predicates (location ?x)
 (on-floor)
 (at ?m ?x)
 (hasknife)
 (onbox ?x)
 (hasbananas)
 (hasglass)
 (haswater))
 ;; movement and climbing
 (:action GO-TO
 :parameters (?x ?y)
 :precondition (and (location ?x) (location ?y) (on-floor) (at monkey ?y))
 :effect (and (at monkey ?x) (not (at monkey ?y))))
 (:action CLIMB
 :parameters (?x)
 :precondition (and (location ?x) (at box ?x) (at monkey ?x))
 :effect (and (onbox ?x) (not (on-floor))))
 (:action PUSH-BOX
 :parameters (?x ?y)
 :precondition (and (location ?x) (location ?y) (at box ?y) (at monkey ?y)
 (on-floor))
 :effect (and (at monkey ?x) (not (at monkey ?y))
 (at box ?x) (not (at box ?y))))
 ;; getting bananas
