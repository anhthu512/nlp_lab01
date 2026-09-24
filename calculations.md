# Part B: Hand Calculations

Complete these calculations by hand before using Python to verify them. This file intentionally leaves answers blank for you to complete. Follow the formulas specified in each W1 exercise: normalized TF from Section 4.2 and unsmoothed IDF for Part B Exercise 3.

## Exercise 1: Count vectors

Corpus:

- D1 = `cat eats fish`
- D2 = `dog eats fish`
- D3 = `cat likes fish`

Use the required vocabulary order `[cat, dog, eats, fish, likes]`.

- D1 count vector: `[____, ____, ____, ____, ____]`
- D2 count vector: `[____, ____, ____, ____, ____]`
- D3 count vector: `[____, ____, ____, ____, ____]`

## Exercise 2: TF

For D1, state the document length and calculate TF for each vocabulary term using the normalized-TF formula in W1 Section 4.2.

- Document length: `__________`
- TF(D1): `[____, ____, ____, ____, ____]`
- Formula used: `____________________________________________`

## Exercise 3: DF and IDF

For each term, record DF and calculate IDF using the unsmoothed formula `log(N / df(t))` specified in Part B Exercise 3.

| term | DF | IDF calculation | IDF |
|---|---:|---|---:|
| cat |  |  |  |
| dog |  |  |  |
| eats |  |  |  |
| fish |  |  |  |
| likes |  |  |  |

## Exercise 4: TF-IDF

Calculate TF-IDF for D1 = `cat eats fish` and the three terms `cat`, `eats`, and `fish`, using the unsmoothed IDF formula specified in Part B.

- TF-IDF(cat, D1): `_______________________________________`
- TF-IDF(eats, D1): `______________________________________`
- TF-IDF(fish, D1): `______________________________________`
- Why is fish's IDF/TF-IDF zero here? `______________________________`

## Exercise 5: Cosine similarity

Use the vectors `x = [1, 1, 1]` and `y = [1, 1, 0]` from W1. Show the dot product, each vector's L2 norm, cosine similarity, and explain why it is not simply the fraction of shared terms.

- Dot product: `________________________`
- `||x||2`: `____________________________`
- `||y||2`: `____________________________`
- Cosine similarity: `___________________`
- Explanation in my own words: `________________________________________`

## Exercise 6: Prediction before running an experiment

For the query `medical image classification`, make the four predictions requested in W1 before running the relevant experiment. Do not fill these in from later retrieval output.

- Predicted highest-similarity document: `______________________________`
- Predicted lowest-similarity document: `_______________________________`
- Term(s) expected to have low IDF: `__________________________________`
- Would count-only ranking change? Why do I predict that? `______________`

## Verification record

Only after completing the hand calculations, run:

```powershell
.\.venv\Scripts\python.exe .\lab01\implementation.py
```

Record any discrepancy and identify whether it came from tokenization, vocabulary ordering, TF, IDF smoothing, normalization, or arithmetic. Do not change code merely to force agreement with a reference library.
