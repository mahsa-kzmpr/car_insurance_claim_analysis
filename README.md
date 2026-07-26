# Car Insurance Claim Prediction and Analysis with Python

## Project Overview

This project analyzes a car insurance dataset and develops classification models to predict `OUTCOME`, a binary target indicating whether a policyholder recorded an insurance claim.

The workflow covers data inspection, preprocessing, model evaluation, feature-importance analysis, SVM configuration search, and supplementary analysis. The main goal is to build a reproducible classification pipeline and examine which customer and driving-related variables are most strongly associated with claim outcomes.

## Project Objectives

- Review the dataset structure, data types, missing values, and target distribution.
- Remove the identifier column and examine continuous variables for outliers.
- Compare numerical characteristics between policyholders with and without recorded claims.
- Convert categorical variables using One-Hot Encoding.
- Train and compare a Decision Tree classifier and `LinearSVC`.
- Identify the most influential variables in the Decision Tree model.
- Evaluate several `SVC` configurations using `GridSearchCV`.
- Compare the trained models with a simple baseline classifier.
- Examine claim rates across driving-experience groups.

## Dataset Overview

The original dataset contains **10,000 records and 19 columns**. Each row represents one policyholder and includes demographic, financial, vehicle, and driving-history information.

The target variable is:

- `OUTCOME = 0`: no insurance claim recorded
- `OUTCOME = 1`: insurance claim recorded

Initial data-quality checks showed:

- `CREDIT_SCORE`: 982 missing values, equal to 9.82% of the data
- `ANNUAL_MILEAGE`: 957 missing values, equal to 9.57% of the data
- Original target distribution:
  - Class 0: 6,867 records, or 68.67%
  - Class 1: 3,133 records, or 31.33%

## Analytical Workflow

### 1. Initial Data Review

The dataset was loaded into a pandas DataFrame and inspected for dimensions, column types, missing values, and unique values. The `ID` column was removed because it serves only as a record identifier and does not provide meaningful predictive information.

### 2. Outlier Detection and Removal

Outliers were examined in the continuous numerical variables `CREDIT_SCORE` and `ANNUAL_MILEAGE` using boxplots and the IQR method.

A total of 26 rows were removed:

- 9 outliers from `CREDIT_SCORE`
- 17 outliers from `ANNUAL_MILEAGE`

The cleaned dataset contained 9,974 records.

Count variables such as speeding violations and previous accidents were retained because high values may represent meaningful driving-risk information rather than data errors.

### 3. Comparison of Claim Groups

Selected numerical variables were compared between `OUTCOME = 0` and `OUTCOME = 1`.

The descriptive analysis showed that policyholders with recorded claims generally had:

- Lower credit scores
- Higher annual mileage

Differences were also observed in speeding violations, DUI records, and previous accidents. These comparisons are descriptive and should not be interpreted as causal relationships.

### 4. Class Balancing

The target variable was imbalanced, so random undersampling was used to reduce the majority class to the size of the minority class.

After balancing:

- Class 0: 3,119 records
- Class 1: 3,119 records
- Total: 6,238 records

This workflow follows the assignment order. In a production project, balancing should generally be applied only to the training set so the test set remains closer to the original real-world distribution.

### 5. Feature Encoding

Categorical variables were transformed using One-Hot Encoding. This converted the dataset into a fully numerical format suitable for machine-learning algorithms.

### 6. Train-Test Split and Missing-Value Treatment

The data was divided into:

- 80% training data
- 20% test data

A stratified split was used to preserve the class distribution in both subsets.

Missing numerical values were filled with the median. To prevent data leakage, the imputer was fitted only on the training data and then applied to the test data.

### 7. Model Development

Two classification models were trained:

- Decision Tree Classifier
- `LinearSVC`

Features were standardized before fitting `LinearSVC` because SVM-based models are sensitive to differences in feature scale.

### 8. Feature Importance

Decision Tree feature importance was calculated for the encoded variables. One-Hot encoded columns were then grouped back into their original variables to provide a clearer interpretation.

### 9. SVC Configuration Search

`GridSearchCV` was used to evaluate several `SVC` configurations, including:

- Linear and RBF kernels
- Different values of `C`
- Different `gamma` values for the RBF kernel

The best cross-validation configuration used a linear kernel with `C = 0.1`.

The initial model used `LinearSVC`, while the GridSearch evaluated `SVC`. These are related but not identical implementations, so the GridSearch result should not be interpreted as a direct before-and-after tuning result for the exact same estimator.

### 10. Supplementary Analysis

Two additional analyses were included:

- Comparison with a Dummy Classifier baseline
- Claim-rate analysis across driving-experience groups

## Model Performance

| Model | Test Accuracy |
|---|---:|
| Dummy Baseline | 0.5000 |
| Decision Tree | 0.7588 |
| `LinearSVC` | **0.8141** |
| Best `SVC` from GridSearchCV | 0.8069 |

`LinearSVC` achieved the highest test accuracy in this analysis.

The best `SVC` configuration reached a mean cross-validation accuracy of **0.8289**, but its test accuracy was slightly lower than the `LinearSVC` result. Because the two results come from different SVM implementations, this comparison should be treated as a comparison between tested models rather than a pure tuning improvement.

## Key Findings

### Model Comparison

`LinearSVC` produced the strongest test-set performance with an accuracy of 81.41%. It correctly classified 1,016 of the 1,248 test observations.

The Decision Tree achieved 75.88% accuracy. Although its predictive performance was lower, it provided a clearer view of feature importance.

### Most Important Variables

The most influential original variables in the Decision Tree model were:

| Rank | Variable | Importance |
|---:|---|---:|
| 1 | `DRIVING_EXPERIENCE` | 0.3233 |
| 2 | `CREDIT_SCORE` | 0.1741 |
| 3 | `VEHICLE_OWNERSHIP` | 0.0877 |
| 4 | `ANNUAL_MILEAGE` | 0.0748 |
| 5 | `POSTAL_CODE` | 0.0659 |

`POSTAL_CODE` should be interpreted carefully. Although it is stored as a number, it represents a geographic category rather than a continuous numerical measurement.

### Claim Rate by Driving Experience

Claim rates declined substantially as driving experience increased:

| Driving Experience | Claim Rate |
|---|---:|
| 0–9 years | 62.75% |
| 10–19 years | 23.81% |
| 20–29 years | 5.11% |
| 30+ years | 1.91% |

This pattern indicates a strong association between driving experience and claim outcome within this dataset. It does not, by itself, establish a causal relationship.

## Methodological Notes

- Undersampling creates balanced classes but removes observations from the majority class.
- In this assignment, balancing was performed before the train-test split to follow the required workflow. In a production project, balancing should generally be applied only to the training data.
- Accuracy values were calculated on a balanced test set and should not be interpreted as expected real-world claim accuracy without further validation.
- `POSTAL_CODE` was retained in its original numerical form, although it is conceptually a categorical geographic variable.
- Results may vary with a different random split, sampling strategy, estimator, or hyperparameter search space.

## Tools and Libraries

- Python
- Jupyter Notebook
- pandas
- NumPy
- Matplotlib
- scikit-learn

## Repository Structure

```text
.
├── README.md
├── car_insurance_claim_analysis.ipynb
└── data/
    └── Car_Insurance_Claim.csv
```

## Running the Notebook

Install the required libraries:

```bash
pip install pandas numpy matplotlib scikit-learn
```

Place the dataset at:

```text
data/Car_Insurance_Claim.csv
```

Then open the notebook and run the cells in order.

## Author

**Mahsa Kazempour**
