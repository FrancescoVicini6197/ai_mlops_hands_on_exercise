# Task 1 report: Exploratory data analysis

| Field | Value |
| --- | --- |
| Student | Francesco Vicini |
| Dataset | census_income.csv |
| Task | 1 |
| Tag / commit | task1 / [short hash] |
| How to run | `pip install pandas numpy matplotlib seaborn scikit-learn ucimlrepo`, then run all cells of "part1_task1.py". No secrets needed. |
| Timebox | [3h; finished] |

## 1. Summary

I cleaned and explored the 1994 US Census income data (48,842 rows, 15 columns) and trained a logistic regression to predict whether a person earns more than 50K a year. Cleaning removed 52 duplicates and turned 6,456 missing values into an "Unknown" category instead of dropping 7.4% of the rows. The target is imbalanced (23.9% earn >50K), so I used a stratified split and class weights. The model reaches AUC 0.90 and finds 84% of high earners, against a baseline that finds none. The main limitation is precision: when the model predicts >50K it is right only 56% of the time, and rare categories (for example small countries) get unstable coefficients.

## 2. Goal and success criteria

Records from the 1994 US Census.

**Goal.** Predict whether a person earns more than 50K a year.

| Criterion | How it is measured | Target | Result |
| --- | --- | --- | --- |
| Clean data | Duplicates, missing values and impossible values left after cleaning | 0 | 0 |
| Better than guessing | Accuracy vs always predicting "<=50K" | > 0.761 | 0.802 |
| Finds high earners | Recall on class >50K | >= 0.70 | 0.844 |
| Separates the classes | ROC AUC on test set | >= 0.85 | 0.902 |

**Evidence:** `part1_task1.py`, variables "n_dup", "missing", section "check impossibile values" and the model cell "regression model".

## 3. Approach and key decisions

| # | Finding | Decision | Alternative considered | Where in the code |
| --- | --- | --- | --- | --- |
| D1 | Missing values are coded two ways: NaN and "?". 6,456 values in workclass, occupation and native-country, touching 7.4% of rows | Replace both with "Unknown" | `dropna()`: rejected, it loses 7.4% of the data | cell "check single values anomalies" and variable "rows_unknown"|
| D2 | The target has 4 labels: "<=50K", "<=50K.", ">50K", ">50K." | Remove the final "." and map to 0/1 | Keep 4 classes: wrong, they mean the same thing | cell "check single values anomalies" |
| D3 | 29 duplicate rows at load, 23 more after cleaning the text | Drop duplicates twice, before and after cleaning | Drop only once: misses 23 rows | variable "n_dup" and cell "new duplicates elimination" |
| D4 | Only 23.9% earn >50K | Stratified train/test split and `class_weight="balanced"` | Judge the model on accuracy only: a model that always says "<=50K" already gets 76% | model cell |
| D5 | capital-gain is 0 for 91.7% of people and goes up to 99,999 | Apply `log1p` to capital-gain and capital-loss | Raw values: the model did not converge | model cell |
| D6 | education is a copy of education-num; fnlwgt has no relation with income or other variables (low Spearman coefficients) | Drop both from the model | Keep them: adds noise and duplicate information | cell: model |
| D7 | The model needs numbers, but 7 columns are text | `pd.get_dummies` with `drop_first=True` (84 columns) | Label encoding (1, 2, 3...): invents an order that does not exist | model cell |

**Evidence:** `part1_task1.py`, cells listed in the last column; printed outputs of `isna().sum()`, `value_counts()` and `describe()`.

## 4. Results

| Result | Value | Baseline or reference |
| --- | --- | --- |
| Accuracy | 0.802 | 0.761 (always "<=50K") |
| Recall on >50K | 0.844 (1,972 of 2,336 found) | 0.000 (always "<=50K") |
| Precision on >50K | 0.557 (1,570 false alarms) | n/a |
| ROC AUC | 0.902 | 0.5 (random guess) |
| Strongest numeric link with income | education-num, Spearman 0.33 | fnlwgt, Spearman -0.01 |

Main findings from the exploration:

| Finding | Number | Meaning for later tasks |
| --- | --- | --- |
| Married people earn more | 44.6% of married-civ-spouse earn >50K vs 4.6% of never-married | marital-status and relationship are key features |
| Education matters | 72.6% of doctorates vs 15.9% of high-school graduates earn >50K | education-num is the strongest numeric feature |
| Gap between men and women | 30.4% of men vs 10.9% of women earn >50K | the model will learn this gap (see section 6) |
| High earners are older and work more | mean age 44.3 vs 36.9; hours per week 45.5 vs 38.8 | age and hours-per-week are useful features |

![](grafici/01_distribuzione_numeriche.png)
*capital-gain is 0 for 91.7% of people and very skewed (skewness 11.9), which is why it is log-transformed.*

![](grafici/02_conteggi_categoriche.png)
*Most people work in the private sector (69.4%) and were born in the United States.*

![](grafici/03_numeriche_vs_income.png)
*People earning >50K are older on average (44.3 vs 36.9 years) and work more hours (45.5 vs 38.8).*

![](grafici/04_categoriche_vs_income.png)
*44.6% of married people earn >50K, against 4.6% of people who never married.*

![](grafici/05_correlazioni.png)
*education-num has the strongest correlation with income (0.33); fnlwgt has none (-0.01).*

**Evidence:** `part1_task1.py`, plotting cells and model cell; figures in `grafici/`.

## 5. Problems and how you handled them

| Problem | Cause | What you did | Status |
| --- | --- | --- | --- |
| Missing values still present after replacing "?" | The dataset uses both "?" and NaN | Added `.fillna("Unknown")`, checked that `isna().sum()` is 0 | solved |
| Target had 4 labels instead of 2 | Labels from the original test set end with "." | `str.rstrip(".")` before mapping to 0/1 | solved |
| New duplicates appeared after cleaning | Spaces and different spellings hid them | Second `drop_duplicates()` | solved |
| Highly skewed values for capital-gain and capital-loss | values up to 99,999 | `log1p` on capital-gain and capital-loss | solved |


**Evidence:** `part1_task1.py`, cells "valori mancanti", "check single values anomalies" and cell model.

## 6. Limitations and trade-offs

| Area | Limitation or trade-off | Impact | Mitigation or next step |
| --- | --- | --- | --- |
| model | Class weights raise recall (0.84) but lower precision (0.56) | 1,570 people wrongly predicted as >50K out of 9,758 | Tune the decision threshold (now fixed at 0.5) |
| model | Rare categories get extreme coefficients, e.g. native-country Ireland (odds ratio 4.0) or Columbia (0.22) from very few people | These coefficients are not reliable | Group rare countries into "Other" |
| data | Data from 1994; the 50K threshold is not adjusted for inflation | Results do not describe today's incomes | Use only for learning, not for real decisions |
| data | Strong gaps by sex (30.4% vs 10.9%) and race in the data | The model reproduces these gaps | Check errors by group before any real use |
| model | One train/test split only | The result could depend on the split | Use cross-validation |

**Evidence:** model cell (classification report, confusion matrix, coefficients); cell with % >50K per category.

## 7. Next steps

| Priority | Next step | Why | Effort |
| --- | --- | --- | --- |
| 1 | Tune the decision threshold | Choose the balance between precision and recall | S |
| 2 | 5-fold cross-validation | Check that AUC 0.90 does not depend on one split | S |
| 3 | Compare with a tree model (random forest) | May capture non-linear effects | M |

## 8. Use of AI assistants

| Tool | What you used it for | Where (files or sections) | How you checked the output |
| --- | --- | --- | --- |
| Claude (Anthropic) | Syntax of functions (`to_numeric`, `nunique`, `between`, `hist`, `groupby`, `savefig`) | Cleaning and plotting cells | Ran each cell and checked the printed output |
| Claude (Anthropic) | Debugging my code | Exploration and plotting | Re-ran the cells |
| Claude (Anthropic) | Draft of the plot cells and logistic regression cell| Model cell | Ran it, compared metrics with the baseline |
| Claude (Anthropic) | First draft of this report | `reports/task1.md` | Checked every number against my notebook output and rewrote it in my own words |

## Checklist

- [ ] Task tag pushed and named in the table at the top
- [ ] Runs from a fresh clone with the commands above
- [ ] No secrets, API keys or large data files committed
- [ ] Every section has an Evidence line
- [ ] Every figure has a caption with a number
- [ ] Guidance blocks and placeholders removed
- [ ] `AI_LOG.md` up to date