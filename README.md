# Hierarchy of Rabbits

## Rule-based Agonistic Action Detection

All rules can be found in the file -> Behaviour Detection/BehaviorDetector.py
Players are animals involved in an unknown event (A and B) and also positionally closed unknown animal P. <br>
All of the rules are cross-checked with a matching counter move and validated robustly. For more details see the code file.

### Rules

---
1. Absolute velocity of animal A is larger than or equal to 20 (t-10 to t-0 and also t0 to t+10) -> Initiator A 
2. Absolute velocity of animal B is larger than or equal to 20 (t-10 to t-0 and also t0 to t+10) -> Initiator B
3. Absolute difference in velocity between animal A and B is less than 5 and sum of their absolute velocity is larger than or equal to 20 (t0 to t+10) -> 
    If absolute velocity A > absolute velocity B then Initiator A else Initiator B.
4. Absolute difference in velocity between animal A (or B) and P is less than 5 and initial distance between animals is less than 50 -> Initiator P
5. Absolute difference in velocity between animal A (or B) and P is less than 5 and initial distance between animals is less than 75 and sum of their absolute velocity is larger than 10 (t0 to t+10) and animal P == (A or B) 
    -> Initiator P
6. Absolute difference in velocity between animal A (or B) and P is less than 5 and initial distance between animals is less than 80 and animal P == (A or B) 
    -> Initiator P
7. Absolute difference in velocity between animal A (or B) and P is less than 5 and sum of their absolute velocity is larger than 15 (t0 to t+10) and animal P == (A or B) 
    -> Initiator P
8. Initial distance between animals (A or B) and P is less than 100 and distance to the third animal is more than 100 -> Initiator P
