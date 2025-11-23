# Peer Review: Albert Kabore – Medical Insurance Cost Regression

**Reviewer:** Brandon
**Date:** November 23, 2025  

**Notebook Reviewed:** [Albert Kabore – Final Project: Regression Analysis - Medical Insurance Costs](https://github.com/albertokabore/ml-07-kabore/blob/main/Albertkabore_Regression.ipynb)

---

## 1. Clarity & Organization

Albert’s notebook is very well-structured and easy to follow. The sections clearly follow the project outline: data import, exploration, feature engineering, modeling, pipelines, and final reflections. Each section has code followed by a reflection, which makes it simple to understand both *what* was done and *why*. The use of clear print statements and plot titles also helps readability.

**Suggestions:**
- Consider moving some of the printed “justification” text into Markdown cells instead of `print()` so the narrative reads more like a report and less like raw code output.
- There seems to be a stray quote before the markdown for section 3.2 (`"### 3.2 Define X and y`); cleaning that up would make the notebook look more polished.
- You might add a short bullet-point summary at the end of each major section (especially Sections 2 and 5) to reinforce the key takeaways before moving on.

---

## 2. Feature Selection & Justification

Feature selection is thoughtful and grounded in both correlation analysis and domain logic. Albert correctly identified `smoker_yes` as the strongest predictor and added sensible engineered features like `age_squared`, `bmi_age_interaction`, `is_overweight`, and `has_children`. The justification section is explicit and explains why each feature matters for insurance charges, which shows good understanding.

**Suggestions:**
- Since the correlation bar plot shows some signal for region and sex, it might be interesting to briefly test one model that also includes those variables, even if they’re weaker predictors, just to see if they add marginal improvement.
- You could add a short comment on potential multicollinearity between `age` and `age_squared` and between `bmi` and `bmi_age_interaction`, and mention that regularization (Ridge/Lasso) might help if this becomes an issue.
- A quick summary table showing basic statistics (mean, std) for selected features before modeling would make the feature selection section even stronger.

---

## 3. Model Performance & Comparisons

Albert used appropriate metrics for regression (R², MAE, RMSE) and clearly reported results for the baseline linear regression and both pipelines. The performance comparison table and bar charts make it very easy to see how much the polynomial pipeline improves the model. The jump from ~0.78 R² to ~0.87 R² and the reduction in MAE/RMSE are well highlighted and interpreted correctly.

**Suggestions:**
- The residual plots for the baseline model are great; you might add similar residual or error distribution plots for the polynomial model to confirm that the error structure improved and to watch for overfitting.
- Consider adding a brief comment on what an MAE of ~$2,700–$4,100 means in practical terms (e.g., “on average, predictions are off by about X% of the mean charge”) to give more business context.
- A short note on potential overfitting with polynomial degree 2 (and how you might test degrees 1, 2, 3 with cross-validation) would make the comparison discussion even more complete.

---

## 4. Reflection Quality

The reflections are detailed, honest, and show strong understanding of both the data and the modeling choices. Albert clearly explains why polynomial features helped, acknowledges the skewness in the target, and ties the strongest predictors (smoking, age, BMI) back to real-world intuition. The “Next Steps” section is excellent and reads like a legit roadmap for a second iteration of the project.

**Suggestions:**
- In the reflections, you could explicitly mention any limitations of the current approach (e.g., no cross-validation yet, no regularization, still some pattern in residuals) to show a clear awareness of what’s missing.
- The “Business Applications” ideas at the end are very strong; consider tying at least one back to specific model outputs (e.g., how the model could be used for risk segmentation thresholds).
- You might briefly discuss interpretability vs. complexity trade-offs (simple linear model vs. polynomial model vs. future tree-based models) to round out the reflection on model choice.

---

## Overall Impression

This is a **well-executed and professional project**. The notebook is clean, the modeling pipeline is solid, feature engineering is thoughtful, and the comparison between baseline and polynomial models is clearly explained. With minor polish on organization (Markdown vs. print, tiny formatting cleanup) and some extra discussion on overfitting and business interpretation, this would be an excellent portfolio piece.
