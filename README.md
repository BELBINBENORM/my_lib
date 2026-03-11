## 🏆 Evaluate Regression ML
**An AutoML-lite Marathon Engine for Scikit-Learn Regressors**

Developed by **BELBIN BENO RM**

**Evaluate Regression** is a high-speed model selection framework designed to automate the evaluation of over 30+ regression models simultaneously. It spans a massive catalog including Linear, Robust, Bayesian, Ensemble, and Neural Network architectures while ensuring system stability through active resource management.

---

## ✨ Key Features

* **🏎️ Model Marathon:** Automatically fits and evaluates a broad spectrum of estimators—from standard OLS to specialized GLMs and Decompositions.
* **🛡️ Active Resource Guarding:** Features an "Active Kill" system that terminates worker processes exceeding a 15-minute time limit or a 10GB RAM threshold.
* **💾 Performance Persistence:** Automatically persists every successful model as a `.joblib` file tagged with metric-based filenames: `Model_TrainRMSE_ValRMSE.joblib`.
* **📊 Live Leaderboard:** Maintains a real-time `score_df` (pandas DataFrame) sorted by **Validation RMSE** for instant performance comparison.
* **🛠️ Lifecycle Management:** Integrated helpers for model inspection, interactive directory cleanup, and zipping results for deployment.

---

## 🚀 Basic Usage

### Notebook Installation (Kaggle / Colab / Jupyter)

```python
!pip install -q git+https://github.com/BELBINBENORM/evaluate-regression-ml.git
```
---
### 🏃 Execution Example

```python
from Evaluate_Regression import EvaluateRegression

# 1. Initialize the engine [cite: 10]
eval_reg = EvaluateRegression()

# 2. Run the Marathon [cite: 10, 11]
# Fits and saves models not already present in your directory [cite: 13, 17]
eval_reg.evaluate(X_train, X_val, y_train, y_val)

# 3. View the Leaderboard [cite: 11, 16]
df = eval_reg.score()
print(df.head())

# 4. Inspect a specific model [cite: 11, 33]
eval_reg.inspection("RandomForest")
```
---
## 📊 Evaluation Output

When calling `eval_reg.score()`, you get a detailed snapshot of your experiment, automatically sorted by the lowest **Val_RMSE**:

| Model | Group | Train_RMSE | Val_RMSE | File |
| :--- | :--- | :--- | :--- | :--- |
| **HistGradientBoosting** | Ensemble | 1.2405 | 1.5621 | HistGradientBoosting_1.2405_1.5621.joblib |
| **RandomForest** | Ensemble | 0.8920 | 1.6110 | RandomForest_0.892_1.611.joblib |
| **BayesianRidge** | Bayesian | 1.8500 | 1.9210 | BayesianRidge_1.85_1.921.joblib |

---

## 💡 Why use Evaluate Regression?

* **Independent Execution:** Each model is trained in an independent worker process using `multiprocessing`. If a complex model like `GaussianProcessRegressor` exceeds memory limits, it is terminated without crashing your notebook.
* **Smart Resumption:** The engine intelligently scans your folder for existing `.joblib` files and skips models that have already been evaluated.
* **Flexibility:** You can easily exclude slow models using `set_ignore_list(['SVR', 'GaussianProcessRegressor'])` to optimize your marathon run.

---

📬 Contact
Author: BELBIN BENO RM

Email: belbin.datascientist@gmail.com

GitHub: BELBINBENORM
