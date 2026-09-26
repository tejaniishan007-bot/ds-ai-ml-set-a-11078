# Data Science & AI/ML — Set A

## Student Information

- **Student Name:** ISHAN TEJANI
- **Student ID:** 11078
- **Set:** A

## Project Objective

This project implements the technical work required for the Data Science & AI/ML Set A examination. The work covers dataset generation, data auditing and cleaning, statistical analysis, feature engineering, predictive modelling, clustering, and evaluation.

## Dataset Generation

The dataset is generated using the supplied generator with:

- Random seed: `404`
- Initial number of records: `300`
- Numeric variables: `visits`, `recency`, `engagement`, `spend`
- Group variable: `G1` / `G2`
- Target variable: `response`
- Five duplicate rows are appended to the generated data.

Run the generator from the repository root:

```bash
python src/generate_data.py
```

The generated dataset is saved as:

```text
data/raw/set_A.csv
```

The generator used in `src/generate_data.py` should remain unchanged from the supplied examination generator.

---

## Data Dictionary

| Column | Description |
|---|---|
| `record_id` | Unique record identifier |
| `visits` | Numeric visit-related feature |
| `recency` | Numeric recency feature |
| `engagement` | Numeric engagement feature |
| `spend` | Numeric spending feature |
| `group` | Categorical group, `G1` or `G2` |
| `response` | Binary target variable |

---

## Data Cleaning

The dataset is audited before modelling.

The cleaning workflow includes:

1. Checking the dataset shape.
2. Checking exact duplicate rows.
3. Checking missing values.
4. Checking target counts.
5. Removing exact duplicate rows.
6. Verifying that the cleaned dataset contains 300 unique records.
7. Verifying that `record_id` is unique after deduplication.

The target column is:

```text
response
```

The identifier column:

```text
record_id
```

is not used as a predictive feature.

---

## Statistical Analysis

### M1 — Descriptive Statistics

For observed `engagement` values, the analysis reports:

- Number of observed values
- Mean
- Median
- Sample standard deviation using `ddof=1`

A histogram of observed engagement is also generated with a title and axis labels.

Missing engagement values are not imputed for the descriptive-statistics calculation.

### M2 — Statistical Inference

The G1 and G2 engagement values are compared using a two-sided Welch t-test with:

```text
alpha = 0.05
```

The analysis reports:

- G1 and G2 sample counts
- Group means
- Welch t-statistic
- p-value
- 95% confidence interval for the difference in group means

The result is interpreted as statistical evidence about an observed difference in group means and is not treated as evidence of causation.

### M3 — Linear Algebra

For complete observations of `visits` and `engagement`, the notebook calculates:

- Sample covariance matrix
- Eigenvalues
- Eigenvectors
- Principal direction
- Variance share explained by the first principal direction

---

## Data Preprocessing & Feature Engineering

The preprocessing workflow includes:

- Exact duplicate removal
- Train/test partitioning
- Numeric missing-value handling
- Categorical encoding
- Feature scaling
- Feature engineering

The engineered feature used in the modelling workflow is:

```text
engagement_per_visit
```

The preprocessing fitted on the training/fit partition is reused for validation/test data without refitting.

`record_id` and the target `response` are excluded from predictive features to prevent leakage.

---

## Supervised Learning

The supervised-learning section uses a fixed train/test partition and compares the classifier against a majority-class baseline.

The classification model is:

```text
LogisticRegression(max_iter=1000)
```

The evaluation includes:

- Accuracy
- Precision for class 1
- Recall for class 1
- F1-score
- Confusion matrix
- Per-record predictions
- Predicted probability for class 1

A prediction threshold of:

```text
0.5
```

is used for converting class-1 probabilities into predicted classes.

### Error Interpretation

- **False positive:** the model predicts `response = 1` when the actual response is `0`.
- **False negative:** the model predicts `response = 0` when the actual response is `1`.

Accuracy is considered together with precision, recall and F1 because the operational consequences of false positives and false negatives can differ.

---

## Unsupervised Learning

K-Means clustering is evaluated using candidate values:

```text
k = 2, 3, 4
```

The clustering configuration uses:

```text
n_init = 10
random_state = 42
```

The silhouette score is used to compare the candidate cluster counts.

The clustering process excludes:

- `record_id`
- target `response`
- group one-hot columns

Cluster IDs are arbitrary labels. Cluster names are assigned descriptively from the observed feature profiles, and practical actions are proposed for each segment.

The target label is not used to choose the clusters.


## References

- Supplied examination dataset generator.
- Python documentation and package documentation used during implementation.
- Any additional external references used in the project should be listed here with their URLs.

---

## Declaration

> All work is my own except where explicitly cited or referenced.

