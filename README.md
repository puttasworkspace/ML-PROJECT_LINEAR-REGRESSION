# Polynomial Regression: Model Selection & Validation

**Machine Learning Assignment - 1**

**Author:** Puttaraja beedinalmath 

**Repository:** [ML-PROJECT_LINEAR-REGRESSION](https://github.com/puttasworkspace/ML-PROJECT_LINEAR-REGRESSION.git)

## 📌 Executive Summary

This repository contains the data, code, and final report for a two-phase polynomial regression experiment. The project focuses on predicting target variables using polynomial feature expansion paired with regularized linear estimators (Ridge, Lasso, and Elastic Net).

The experiments emphasize rigorous model selection, hyperparameter tuning (degree, alpha, l1_ratio), and validation across multiple train-test splits to ensure optimal generalization without overfitting.

### Headline Results

* **Phase 1:** **Lasso (Degree 5)** emerged as the strongest overall candidate, achieving a validation RMSE of 0.5864. After refitting on the combined training and validation data, it scored a Test RMSE of 0.5682, MAE of 0.4592, and an $R^2$ of 0.9629.

* **Phase 2:** The requested 80/20 split candidate is **Ridge (Degree 12)** with $\alpha=3.1623$. It achieved a Cross-Validation (CV) RMSE of 0.5119, Test RMSE of 0.4947, MAE of 0.3838, and an impressive Test $R^2$ of 0.9932.

## 📂 Project Structure & Data Files

The project relies on two sets of variables (var1 for Phase 1, var2 for Phase 2). The following files are included in this project:

* **Phase 1 Data:**

  * `IMT2024091_train_var1.csv`: Training dataset for Phase 1.

  * `IMT2024091_test_var1.csv`: Held-out external test set for Phase 1 final predictions.

* **Phase 2 Data:**

  * `IMT2024091_train_var2.csv`: Training dataset for Phase 2.

  * `IMT2024091_test_var2.csv`: Held-out external test set for Phase 2 final predictions.

* **Documentation:**

  * `final_polynomial_regression_phase1_phase2_report.pdf`: Detailed project report containing plots, evaluation rationales, and metric tables.

## 🔬 Methodology & Phases

### Phase 1: Degree Selection

* **Objective:** Evaluate the effect of polynomial degrees (1-7) on Ridge, Lasso, and Elastic Net estimators.

* **Selection Rule:** Lowest validation RMSE within each model family.

* **Key Findings:** Increasing the degree from 1 to 5 sharply reduces validation error. Lasso reaches its minimum at degree 5 (0.5864). Higher complexities (degrees 6-7) offered no validation benefit, confirming that extra polynomial terms raise variance without improving generalization.

### Phase 2: Regularization and Split Comparison

* **Objective:** Broaden the degree search space (1-20), tune regularization strengths ($\alpha$), and compare performances across different train-test splits (90/10, 85/15, 80/20, 75/25, and 70/30).

* **Technique:** Used `GridSearchCV` with 5-fold cross-validation.

* **Key Findings:** Ridge consistently selected degree 12 for the 90/10, 85/15, and 80/20 settings. For the 80/20 split, Ridge (Degree 12, $\alpha=3.1623$) showed strong held-out fit ($R^2$ 0.9932). While Elastic Net technically won the CV RMSE tightly on the 80/20 split, Ridge Degree 12 was chosen for its stability and strong test R² across the larger-training splits.

## ⚙️ Techniques Used

1. **Polynomial Feature Expansion:** Controls the order of nonlinear terms available to the linear estimator. Captures curvature but requires tuning to prevent overfitting.

2. **Regularization:**

   * *Ridge (L2):* Shrinks coefficient magnitudes.

   * *Lasso (L1):* Shrinks coefficients and can set some to exactly zero (feature selection).

   * *Elastic Net:* A hybrid of L1 and L2 penalties.

3. **Cross-Validation:** Used CV RMSE to compare degrees and penalty settings.

## 📊 Evaluation Metrics

* **RMSE (Root Mean Squared Error) & MAE (Mean Absolute Error):** Expressed in the response variable's units; lower is better. RMSE heavily penalizes larger errors compared to MAE.

* $R^2$ **(R-squared):** Measure of fit relative to predicting the target mean; higher is better (approaching 1.0).

## 🚀 How to Run

1. Clone the repository:

   ```
   git clone https://github.com/puttasworkspace/ML-PROJECT_LINEAR-REGRESSION.git
   cd ML-PROJECT_LINEAR-REGRESSION
   
   ```

2. Ensure you have the required Python libraries installed (`pandas`, `numpy`, `scikit-learn`, `matplotlib`).

3. Place the dataset files (`IMT2024091_train_var1.csv`, `IMT2024091_test_var1.csv`, `IMT2024091_train_var2.csv`, `IMT2024091_test_var2.csv`) in the root directory.

4. Run the respective Jupyter Notebooks or Python scripts for Phase 1 and Phase 2 to reproduce the model selection and generate the final predictions (`IMT2024091_pred_var1.csv` and `IMT2024091_pred_var2.csv`).
