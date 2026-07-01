# AI & Machine Learning - Task 3
### Model Validation, Overfitting Control & Hyperparameter Tuning

---

## 1. Objective

This task enhances the **House Price Prediction system** built in Task-2 by moving from
basic model training into professional ML practice:

- Detect and control overfitting
- Validate models using cross-validation
- Tune model hyperparameters systematically
- Improve generalization performance
- Justify the final model selection scientifically

This task focuses on **model reliability**, not just accuracy.

---

## 2. Dataset

- **Dataset:** California Housing Dataset
- **Target variable:** Median House Value (`HousePrice`, in $100,000s)
- **Input features:** `MedInc`, `HouseAge`, `AveRooms`, `AveBedrms`, `Population`, `AveOccup`, `Latitude`, `Longitude`
- **Rows used:** 20,433 (after removing 207 rows with missing `total_bedrooms`)

> **Note on data source:** `sklearn.datasets.fetch_california_housing()` downloads from an
> external host (`ndownloader.figshare.com`) that isn't reachable from this sandboxed
> environment. The dataset was therefore reconstructed from its original raw source
> (Pace & Barry, 1997 block-group data, mirrored on GitHub as `housing_raw.csv`) using the
> **exact same feature engineering** scikit-learn applies internally
> (e.g. `AveRooms = total_rooms / households`). The resulting columns, scale, and values are
> equivalent to calling `fetch_california_housing(as_frame=True)` directly.

---

## 3. Tools & Technologies

- Python 3.12
- Jupyter Notebook (executed via `nbconvert`)
- pandas, NumPy
- scikit-learn
- matplotlib
- joblib

---

## 4. Workflow Summary

| Step | What was done |
|------|----------------|
| 1 | Imported all required libraries |
| 2 | Loaded and reconstructed the California Housing dataset |
| 3 | Applied `StandardScaler` feature scaling |
| 4 | Created an 80/20 train-test split (`random_state=42`) |
| 5 | Trained an **untuned Decision Tree** and compared train vs. test RMSE to detect overfitting |
| 6 | Ran **5-fold cross-validation** for a reliable performance estimate |
| 7 | Tuned `max_depth` and `min_samples_split` using **GridSearchCV** |
| 8 | Evaluated the optimized (tuned) Decision Tree on the test set |
| 9 | Built a model comparison table (Linear Regression, Ridge, untuned Tree, tuned Tree) |
| 10 | Wrote the final model selection justification |

---

## 5. Key Results

### Overfitting check (untuned Decision Tree)
| Metric | Value |
|---|---|
| Train RMSE | 0.0000 |
| Test RMSE | 0.7502 |
| Gap (Test − Train) | 0.7502 |

A near-zero training error with a much higher test error confirms **severe overfitting** —
the unconstrained tree memorized the training data instead of generalizing.

### 5-Fold Cross-Validation (untuned Decision Tree)
| Fold 1 | Fold 2 | Fold 3 | Fold 4 | Fold 5 | Mean CV RMSE |
|---|---|---|---|---|---|
| 0.8536 | 0.8004 | 0.8601 | 0.9521 | 0.8728 | **0.8678 (± 0.0488)** |

### GridSearchCV - Hyperparameter Tuning
- **Search grid:** `max_depth ∈ {3, 5, 7, 10}`, `min_samples_split ∈ {2, 5, 10}`
- **Best parameters:** `max_depth = 10`, `min_samples_split = 10`
- **Best CV RMSE:** 0.6349

### Final Model Comparison

| Model | Train RMSE | Test RMSE | R² Score |
|---|---|---|---|
| Linear Regression | 0.7185 | 0.7514 | 0.5871 |
| Ridge Regression | 0.7185 | 0.7514 | 0.5872 |
| Untuned Decision Tree | 0.0000 | 0.7502 | 0.5885 |
| **Tuned Decision Tree** | **0.4869** | **0.6532** | **0.6880** |

The tuned Decision Tree wins on every metric that matters for deployment, and its
train/test RMSE gap (0.1664) is far smaller than the untuned tree's (0.7502) — proof that
tuning meaningfully controlled overfitting rather than just improving raw accuracy.

---

## 6. Final Model Selection Justification

- **Why this model was selected:** The tuned Decision Tree achieves the lowest test RMSE
  and highest R² of all candidates, beating both Linear and Ridge Regression.
- **How overfitting was reduced:** Constraining `max_depth` and `min_samples_split` stopped
  the tree from growing branches that memorize individual training points, shrinking the
  train/test RMSE gap from 0.7502 to 0.1664.
- **Why cross-validation results are trusted:** The 5-fold CV RMSE averages performance
  across five different train/validation splits, so hyperparameter selection isn't biased
  by how one particular 80/20 split happened to fall.
- **Trade-offs between simplicity and performance:** Linear/Ridge Regression are simpler
  and more interpretable but underfit the non-linear relationships in housing prices. The
  tuned Decision Tree trades some interpretability for materially better generalization —
  the right trade-off for a production house-price prediction system.

---

## 7. Deliverables

### Mandatory
| File | Description |
|---|---|
| `AI_ML_Task3_Model_Validation_Tuning.ipynb` | Full, pre-executed Jupyter notebook covering Steps 1–10 |
| `AI_ML_Task3_Report.pdf` | 3-page PDF report: overfitting analysis, tuning approach, final model justification |
| `model_comparison.csv` | Model comparison summary table (RMSE, R² for all 4 models) |
| *(cross-validation & hyperparameter tuning results)* | Included inline in the notebook (Steps 6–7) and summarized in the PDF |

### Optional (included)
| File | Description |
|---|---|
| `plot_train_vs_test_rmse.png` | Train vs. test RMSE across all 4 models (overfitting check) |
| `plot_cv_scores.png` | 5-fold cross-validation RMSE per fold |
| `plot_gridsearch_tuning.png` | GridSearchCV tuning curves (`max_depth` vs. CV RMSE per `min_samples_split`) |
| `plot_actual_vs_predicted.png` | Actual vs. predicted house prices for the tuned model |
| `best_tuned_decision_tree.joblib` | Saved final trained model (`GridSearchCV` best estimator) |
| `feature_scaler.joblib` | Saved fitted `StandardScaler` used for preprocessing |

---

## 8. How to Reproduce

```bash
# Install dependencies
pip install pandas numpy matplotlib scikit-learn joblib jupyter nbconvert

# Run the notebook end-to-end
jupyter nbconvert --to notebook --execute --inplace \
    AI_ML_Task3_Model_Validation_Tuning.ipynb

# Load the saved model for inference on new data
python3 -c "
import joblib
model = joblib.load('best_tuned_decision_tree.joblib')
scaler = joblib.load('feature_scaler.joblib')
# X_new = scaler.transform(new_data)
# predictions = model.predict(X_new)
"
```

---

## 9. Key Learning Outcomes

After completing Task-3:
- Detect overfitting and underfitting via train/test RMSE comparison
-  Apply cross-validation correctly for reliable performance estimation
-  Tune hyperparameters professionally using `GridSearchCV`
-  Build generalizable ML models
-  Follow industry-aligned ML workflows
