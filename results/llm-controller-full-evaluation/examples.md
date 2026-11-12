# Diagnostic test paths

These examples are selected after evaluation to illustrate behavior, not to estimate its frequency. For each named category, take the first matching question in original test order that has not already been selected. All test questions remain included in the quantitative results. Reference answers and correctness labels are evaluation annotations withheld from the controller. The action-only model gives no explanation; none is invented here.

## Controller correct, repair-once wrong

mmlu-pro-chemistry-04475

Question: What is the weight of 1,000 cubic feet of air at STP?

Options:
A. 1.10 × 10^3 oz
B. 2.50 × 10^3 oz
C. 1.00 × 10^3 oz
D. 8.00 × 10^2 oz
E. 2.00 × 10^3 oz
F. 1.5 × 10^3 oz
G. 1.29 × 10^3 oz
H. 1.50 × 10^3 oz
I. 1.00 × 10^2 oz
J. 1.75 × 10^3 oz

Choose the best answer. In the JSON answer field, return only one option letter.

Reference answer (evaluation only): `G`

### Visited response: initial

```text
{
  "reasoning": "At Standard Temperature and Pressure (STP), one cubic foot of air weighs approximately 0.076 grams. To find the weight of 1,000 cubic feet of air at STP, multiply 0.076 grams by 1,000.",
  "answer": "G",
  "confidence": "high"
}
```

Correct: True; solver tokens: 351.

Controller output: `{"action":"stop"}`

Final correctness: True. Total research tokens: 1046. Repair-once correctness: False.

## Controller wrong, repair-once correct

gsm8k-test-00276

Carly had 80 cards, 2/5 of the cards had the letter A on them, 1/2 of the remaining had the letter B, 5/8 of the rest had the letter C on them, and the others had the letter D. How many of the cards had the letter D on them?

Reference answer (evaluation only): `9`

### Visited response: initial

```text
{
  "reasoning": "First, we calculate how many cards had the letter A. Then we find out how many were left after removing those with 'A'. We proceed to subtract the number of cards with 'B' from the total remaining, then find out how many cards had 'C' from what's left. Finally, the remaining cards will have 'D' on them.",
  "answer": "32",
  "confidence": "high"
}
```

Correct: False; solver tokens: 243.

Controller output: `{"action":"stop"}`

Final correctness: False. Total research tokens: 827. Repair-once correctness: True.

## Controller selects retry

gsm8k-test-00806

Janet is collecting the results of Herbert Hoover Elementary School's annual standardized test. 340 out of 500 third-graders passed, along with 40 out of 100 fourth graders. The 400 fifth graders had a pass rate that was twice the fourth grades' pass rate. What is the school's overall pass rate?

Reference answer (evaluation only): `70`

### Visited response: initial

```text
```json
{
  "reasoning": "To find the overall pass rate, we first need to determine the number of students who passed in each grade level. We know the pass rates for third and fourth grades and use this information to calculate the pass rate for fifth grade. Then, we can calculate the total number of students who passed across all grades and divide by the total number of students to find the overall pass rate.",
  "answer": "64%",
  "confidence": "high"
}
```
```

Correct: False; solver tokens: 260.

Controller output: `{"action":"repair"}`

### Visited response: repair

```text
```json
{
  "reasoning": "First, calculate the pass rate for each grade level. For third graders, the pass rate is 340/500 = 0.68 or 68%. For fourth graders, the pass rate is 40/100 = 0.4 or 40%. Since the fifth graders' pass rate is twice that of the fourth graders, it is 2 * 0.4 = 0.8 or 80%. Next, calculate the total number of students who passed: (340 + 40 + 400) / 500 = 780 / 500 = 0.156 or 15.6%. Therefore, the overall pass rate is 15.6%.",
  "answer": "15.6%",
  "confidence": "medium"
}
```
```

Correct: False; solver tokens: 465.

Controller output: `{"action":"retry"}`

### Final solver response (budget exhausted; automatically submitted)

```text
{
  "reasoning": "First, calculate the pass rates for each grade level. Third-grade pass rate is 340/500 = 0.68 (or 68%). Fourth-grade pass rate is 40/100 = 0.40 (or 40%). Fifth-grade pass rate is twice the fourth grade's pass rate, so it is 2 * 0.40 = 0.80 (or 80%). To find the overall pass rate, we need to find the total number of students who passed divided by the total number of students in all grades. Total students = 500 + 100 + 400 = 1000. Total number of students who passed = (340 + 40 + 400) = 780. Overall pass rate = 780 / 1000 = 0.78 or 78%. We can be reasonably confident about this calculation.",
  "answer": "78%",
  "confidence": "high"
}
```

Correct: False.

Final correctness: False. Total research tokens: 2560. Repair-once correctness: False.

## Controller uses two solver actions

gsm8k-test-00540

$18 is shared among 3 boys. The eldest added $5 to his share and added another $10 the following day. He later needed some money and spent $8 from his total so far. After saving some more, he was able to triple the amount he had left after spending the $8. How much does he have now?

Reference answer (evaluation only): `39`

### Visited response: initial

```text
```json
{
  "reasoning": "First, calculate the initial share of each boy, then determine how much the eldest boy has after adding $5 and $10. Subtract $8 from this amount, then find out how much he needs to save to triple what he has left.",
  "answer": "24",
  "confidence": "medium"
}
```
```

Correct: False; solver tokens: 229.

Controller output: `{"action":"repair"}`

### Visited response: repair

```text
```json
{
  "reasoning": "Each boy's initial share is $18 / 3 = $6. The eldest boy adds $5 and $10, making his total $6 + $5 + $10 = $21. After spending $8, he has $21 - $8 = $13 left. To triple the amount left after spending $8, we need to find 3 * $13 = $39. Since he already has $13, he needs to save an additional $39 - $13 = $26.",
  "answer": "26",
  "confidence": "low"
}
```
```

Correct: False; solver tokens: 377.

Controller output: `{"action":"retry"}`

### Final solver response (budget exhausted; automatically submitted)

```text
```json
{
  "reasoning": "First, find the initial share of each boy. Then calculate the amount the eldest boy had after adding $5 and $10. Next, subtract $8 from this amount. Finally, determine how much he has left and then multiply it by 3.",
  "answer": "147",
  "confidence": "high"
}
```
```

Correct: False.

Final correctness: False. Total research tokens: 2155. Repair-once correctness: False.

