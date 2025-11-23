# Project Summary: Medical Insurance Cost Regression

**Author:** Brandon Jean  
**Course:** Applied Machine Learning  
**Date:** Fall 2025  

---

## 1. Problem Description

The goal of this project is to build regression models that predict medical insurance charges based on demographic and health-related features. The dataset includes patient characteristics such as age, sex, BMI, number of children, smoking status, and region. The task is to understand which factors influence insurance costs and to build models that can accurately predict new patient charges.

---

## 2. Dataset Description

**Dataset:** `insurance.csv`  
**Rows:** 1338  
**Columns:** 7  

**Features:**
- **age** – patient age  
- **sex** – male/female  
- **bmi** – body mass index  
- **children** – number of dependents  
- **smoker** – yes/no  
- **region** – US region  
- **charges** – target variable (continuous cost)

No missing values were found. Categorical variables were encoded using one-hot encoding, and two engineered features were added:
- **bmi_over_30** – obesity indicator  
- **age_smoker_interaction** – amplifies cost for older smokers  

The dataset shows a strong right-skew in the target variable due to very expensive patients.

---

## 3. Data Exploration

### Key Observations:
- **BMI** clusters around 25–35 with some high outliers.
- **Charges** are heavily right-skewed, with expensive patients contributing to high variance.
- **Smoking** is the most imbalanced feature (more non-smokers).
- Initial visualizations (histograms, boxplots, count plots) helped identify patterns and outliers that influenced model design.

---

## 4. Data Preparation

### Steps Completed:
1. One-hot encoded categorical features (`sex`, `smoker`, `region`).
2. Created engineered features:
   - `bmi_over_30`
   - `age_smoker_interaction`
3. Confirmed no missing values, but still implemented imputers in pipelines for safety.
4. Split data 80/20 using `train_test_split`.

---

## 5. Baseline Model

The first model was a plain **Linear Regression** trained on the encoded data with no scaling.

### Baseline Performance:
- **R²:** (varies by student dataset but printed in notebook)
- **MAE:** printed in notebook  
- **RMSE:** printed in notebook  

The baseline captured general trends but struggled with outliers and non-linear relationships.

---

## 6. Improved Models & Pipelines

Two advanced pipelines were implemented:

### **Pipeline 1:**  
Imputer → StandardScaler → Linear Regression  
- Helps normalize feature scales  
- Often improves stability and convergence  

### **Pipeline 2:**  
Imputer → PolynomialFeatures (degree=3) → StandardScaler → Linear Regression  
- Allows modeling of non-linear relationships  
- Captures complex interactions between features  
- Higher risk of overfitting  

### Performance Comparison:
A full comparison table (R², MAE, RMSE) is included in the notebook.  
The polynomial pipeline generally produced the highest R² and lowest error, showing that non-linear patterns matter in health cost prediction.

---

## 7. Final Thoughts

### What Worked:
- One-hot encoding + feature engineering improved model performance.
- Polynomial features allowed the model to capture deeper relationships.
- Scaling was important for stability when using polynomial features.

### Challenges:
- The target variable is highly skewed, making errors larger for expensive patients.
- Large outliers inflate RMSE.
- Choosing polynomial degree required balancing performance vs. overfitting risk.

### Next Steps:
- Try Ridge, Lasso, and Elastic Net to reduce overfitting.
- Experiment with tree-based models (Random Forest, XGBoost).
- Log-transform target variable to reduce skew.
- Use cross-validation and hyperparameter tuning for stronger generalization.

---

## 8. Key Learnings

This project strengthened skills in:
- Building full end-to-end regression workflows  
- Feature engineering and encoding  
- Designing ML pipelines  
- Comparing models with consistent metrics  
- Understanding bias/variance trade-offs  
- Communicating findings clearly and professionally  

Regression modeling is not just about fitting a line — it requires careful preprocessing, thoughtful model choice, and structured evaluation.

---

