# my_lib
## select_regression.py — README

Overview
- A compact utility that runs a "model marathon" over a broad catalog of scikit-learn regressors, saves trained models to .joblib files named by their train/validation RMSE, and provides simple inspection, cleanup and zipping helpers.
- Main entrypoint: EvaluateRegression — a class that organizes the catalog, runs training/evaluation, and manages discovered saved models.

Key features
- Large built-in catalog of regression estimators (linear, robust, Bayesian, tree/ensemble, SVM, neighbors, GLM, neural net, decompositions, baseline).
- Automatic saving of trained models to filename pattern: ModelName_{Train_RMSE}_{Val_RMSE}.joblib
- Maintains a pandas DataFrame (score_df) of discovered models (scanned from current directory).
- Helpers:
  - set_ignore_list(models_to_ignore) ��� skip specific catalog items
  - refresh_score_df() — scan current dir for .joblib files matching the naming pattern
  - evaluate(X_train, X_val, y_train, y_val) — fit catalog models (skips ones already present or ignored), compute train/val RMSE and save models
  - inspection(model_name_or_file) — load & print model class and params
  - cleanup_models() — prompt & delete all .joblib files
  - zip_models(zip_name="zipped_model") — package all discovered .joblib files into a zip

Requirements
- Python 3.8+
- pandas
- scikit-learn (version compatible with the used estimators and metrics)
- joblib
- (standard library modules used: os, re, gc, shutil, warnings)

Notes about scikit-learn compatibility
- The script imports many estimators — availability depends on the scikit-learn version. Some estimators or metrics names (e.g., root_mean_squared_error) may require scikit-learn >= a specific version. If you hit import errors, upgrade scikit-learn: pip install -U scikit-learn
- Warnings from sklearn ConvergenceWarning and UserWarning are suppressed for cleaner output.

Filename pattern and scanning
- Saved model filenames follow the regex: (.+)_([0-9\.eE\+\-]+)_([0-9\.eE\+\-]+)\.joblib
  - i.e., ModelName_trainRMSE_valRMSE.joblib
- refresh_score_df() scans the current working directory for files matching that pattern, extracts model name, train RMSE and val RMSE, maps model to a group (catalog grouping), and builds a DataFrame sorted by Val_RMSE.

API / Method summary

- EvaluateRegression()
  - Initializes the model catalog and an empty score_df.

- set_ignore_list(models_to_ignore)
  - models_to_ignore: list of strings matching keys in the catalog
  - Stores an ignore list used by evaluate() to skip models.

- refresh_score_df() -> pandas.DataFrame
  - Scans current dir for .joblib files matching the file-name pattern and returns the DataFrame with columns: Model, Group, Train_RMSE, Val_RMSE, File.

- evaluate(X_train, X_val, y_train, y_val)
  - Trains each catalog model (unless ignored or already present in score_df), computes train/val RMSE, saves model to .joblib using the naming convention.
  - Uses sklearn estimators' fit and predict methods.
  - Exceptions during training are caught; garbage collection runs in finally.

- inspection(model_name_or_file)
  - Accepts either a model name (e.g., "RandomForest") present in score_df or a direct joblib filename. Loads the saved model and prints its class and parameters.

- cleanup_models()
  - Prompts the user to confirm and deletes all .joblib files in the current directory. Afterwards refreshes score_df.

- zip_models(zip_name="zipped_model")
  - Copies discovered .joblib files to a temporary directory and creates zip_name.zip in the current directory. Replaces existing zip if present.

Basic usage example
- Example: train a catalog over already-split train/val sets (X_train, X_val, y_train, y_val):

  from select_regression import EvaluateRegression

  evalr = EvaluateRegression()

  ### Optionally skip some slow/undesired models
  evalr.set_ignore_list(['GaussianProcessRegressor', 'KernelRidge'])

  ### If you already have saved .joblib files in the cwd, update the score DF
  evalr.refresh_score_df()

  ### Run the marathon (this fits & saves models not already present)
  evalr.evaluate(X_train, X_val, y_train, y_val)

  ### Re-scan to load results into the DataFrame
  df = evalr.refresh_score_df()
  print(df.head())

  # Inspect a model by name (if present in df) or full filename
  evalr.inspection('RandomForest')
  ### or
  evalr.inspection('RandomForest_1.2345_1.5678.joblib')

  ### Zip up discovered models
  evalr.zip_models('my_models_archive')

  ### Clean up (interactive prompt)
  evalr.cleanup_models()

Practical tips & caveats
- Working directory matters: the class scans and writes files to the current working directory. Run from a dedicated folder to avoid cluttering unrelated files.
- Some models (GaussianProcessRegressor, SVMs, etc.) can be slow or memory-heavy on larger datasets — consider adding them to the ignore list for large-scale marathons.
- cleanup_models() permanently deletes .joblib files — use cautiously.
- The script attempts to handle exceptions per-model so the marathon continues if one estimator fails.
- If you want cross-validation or hyperparameter tuning, extend evaluate() to wrap models in GridSearchCV / RandomizedSearchCV before fit.


