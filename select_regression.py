import os
import re
import gc
import joblib
import shutil
import time
import psutil
import multiprocessing
import pandas as pd
import warnings
from sklearn.metrics import root_mean_squared_error

# Ignore specific sklearn warnings for a cleaner console
from sklearn.exceptions import ConvergenceWarning
warnings.filterwarnings("ignore", category=ConvergenceWarning)
warnings.filterwarnings("ignore", category=UserWarning)

# All required imports
from sklearn.linear_model import (
    LinearRegression, Ridge, Lasso, ElasticNet, SGDRegressor, 
    Lars, LassoLars, LassoLarsIC, OrthogonalMatchingPursuit, 
    HuberRegressor, RANSACRegressor, TheilSenRegressor, 
    PoissonRegressor, GammaRegressor, TweedieRegressor, 
    QuantileRegressor, PassiveAggressiveRegressor,
    BayesianRidge, ARDRegression
)
from sklearn.tree import DecisionTreeRegressor, ExtraTreeRegressor
from sklearn.ensemble import (
    RandomForestRegressor, ExtraTreesRegressor, BaggingRegressor,
    HistGradientBoostingRegressor, GradientBoostingRegressor, AdaBoostRegressor
)
from sklearn.neighbors import KNeighborsRegressor, RadiusNeighborsRegressor
from sklearn.svm import SVR, NuSVR, LinearSVR
from sklearn.neural_network import MLPRegressor
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.kernel_ridge import KernelRidge
from sklearn.isotonic import IsotonicRegression
from sklearn.dummy import DummyRegressor
from sklearn.cross_decomposition import CCA, PLSRegression, PLSCanonical

def train_worker(model, X, y, return_dict):
    """
    Independent worker function. 
    Required to be outside the class for multiprocessing pickling.
    """
    try:
        model.fit(X, y)
        return_dict['model'] = model
        return_dict['success'] = True
    except Exception as e:
        return_dict['success'] = False
        return_dict['error'] = str(e)
        
class EvaluateRegression:
    def __init__(self):
        # Full Catalog (No CV models)
        self.catalog = {
            # Linear Models
            'LinearRegression': (LinearRegression(), 'Linear'),
            'Ridge': (Ridge(), 'Linear'),
            'Lasso': (Lasso(), 'Linear'),
            'ElasticNet': (ElasticNet(), 'Linear'),
            'SGDRegressor': (SGDRegressor(), 'Linear'),
            'Lars': (Lars(), 'Linear'),
            'LassoLars': (LassoLars(), 'Linear'),
            'LassoLarsIC': (LassoLarsIC(), 'Linear'),
            'OrthogonalMatchingPursuit': (OrthogonalMatchingPursuit(), 'Linear'),
            'PassiveAggressiveRegressor': (PassiveAggressiveRegressor(), 'Linear'),
            
            # Robust Models
            'HuberRegressor': (HuberRegressor(), 'Robust'),
            'RANSACRegressor': (RANSACRegressor(), 'Robust'),
            'TheilSenRegressor': (TheilSenRegressor(), 'Robust'),
            
            # Bayesian & Gaussian
            'BayesianRidge': (BayesianRidge(), 'Bayesian'),
            'ARDRegression': (ARDRegression(), 'Bayesian'),
            'GaussianProcessRegressor': (GaussianProcessRegressor(), 'Gaussian'),
            
            # Trees & Ensembles
            'DecisionTree': (DecisionTreeRegressor(), 'Tree'),
            'ExtraTree': (ExtraTreeRegressor(), 'Tree'),
            'RandomForest': (RandomForestRegressor(), 'Ensemble'),
            'ExtraTrees': (ExtraTreesRegressor(), 'Ensemble'),
            'BaggingRegressor': (BaggingRegressor(), 'Ensemble'),
            'GradientBoosting': (GradientBoostingRegressor(), 'Ensemble'),
            'HistGradientBoosting': (HistGradientBoostingRegressor(), 'Ensemble'),
            'AdaBoost': (AdaBoostRegressor(), 'Ensemble'),
            
            # Neighbors & SVM
            'KNeighbors': (KNeighborsRegressor(), 'Neighbors'),
            'RadiusNeighbors': (RadiusNeighborsRegressor(), 'Neighbors'),
            'SVR': (SVR(), 'SVM'),
            'NuSVR': (NuSVR(), 'SVM'),
            'LinearSVR': (LinearSVR(), 'SVM'),
            
            # GLM & Specialized
            'PoissonRegressor': (PoissonRegressor(), 'GLM'),
            'GammaRegressor': (GammaRegressor(), 'GLM'),
            'TweedieRegressor': (TweedieRegressor(), 'GLM'),
            'QuantileRegressor': (QuantileRegressor(), 'Linear'),
            'KernelRidge': (KernelRidge(), 'Kernel'),
            'IsotonicRegression': (IsotonicRegression(), 'Specialized'),
            'MLPRegressor': (MLPRegressor(), 'Neural Network'),
            
            # Decompositions & Baseline
            'CCA': (CCA(), 'Decomposition'),
            'PLSRegression': (PLSRegression(), 'Decomposition'),
            'PLSCanonical': (PLSCanonical(), 'Decomposition'),
            'DummyRegressor': (DummyRegressor(), 'Baseline')
        }
        self.columns = ['Model', 'Group', 'Train_RMSE', 'Val_RMSE', 'File']
        self.score_df = pd.DataFrame(columns=self.columns)
        self.ignore_list = []

    def set_ignore_list(self, models_to_ignore):
        """Sets a list of model names to skip during the marathon."""
        if isinstance(models_to_ignore, list):
            self.ignore_list = models_to_ignore
            print(f"🚫 Ignore list updated: {self.ignore_list}")
        else:
            print("⚠️ Please provide a list of strings.")

    def refresh_score_df(self):
        """Scans directory for .joblib files and updates self.score_df"""
        data = []
        pattern = r"(.+)_([0-9\.eE\+\-]+)_([0-9\.eE\+\-]+)\.joblib"
        
        for file in os.listdir('.'):
            match = re.search(pattern, file)
            if match:
                name, t_rmse, v_rmse = match.groups()
                group = self.catalog.get(name, (None, "Unknown"))[1]
                data.append({
                    'Model': name, 'Group': group, 
                    'Train_RMSE': float(t_rmse), 'Val_RMSE': float(v_rmse), 
                    'File': file
                })
        
        self.score_df = pd.DataFrame(data)
        if not self.score_df.empty:
            self.score_df = self.score_df.sort_values('Val_RMSE').reset_index(drop=True)
        else:
            self.score_df = pd.DataFrame(columns=self.columns)

    def evaluate(self, X_train, X_val, y_train, y_val):
        """Runs training loop with Active Kill for Time and RAM Guarding."""
        # Ensure the scoreboard is fresh before starting
        self.refresh_score_df()
        print(f"🚀 Marathon Started. Current Progress: {len(self.score_df)} models found.\n")
        
        RAM_LIMIT_GB = 10.0
        TIME_LIMIT_SEC = 15 * 60  # 900 seconds
    
        for name, (model, group) in self.catalog.items():
            # 1. Skip if Ignored or Already Done
            if name in self.ignore_list:
                print(f"🚫 {name:30} [Ignored by User  ]")
                continue
    
            if name in self.score_df['Model'].values:
                print(f"⏩ {name:30} [Already Evaluated]")
                continue
    
            # 2. RAM Guard
            current_ram = psutil.virtual_memory().used / (1024**3)
            if current_ram > RAM_LIMIT_GB:
                print(f"⚠️ {name:30} [Ignored High RAM ] ({current_ram:.1f}GB)")
                continue
    
            start_time = time.time()
            print(f"⏸️ {name:30} [", end="", flush=True)
    
            # 3. Active Time Killing via Multiprocessing
            manager = multiprocessing.Manager()
            return_dict = manager.dict()
            
            p = multiprocessing.Process(target=train_worker, args=(model, X_train, y_train, return_dict))
            p.start()
            p.join(timeout=TIME_LIMIT_SEC)
    
            if p.is_alive():
                p.terminate()  # STOP MODEL IMMEDIATELY
                p.join()
                print(f"❌ Ignored High Run Time ]")
                manager.shutdown()
                continue
    
            if not return_dict.get('success', False):
                err = return_dict.get('error', 'Unknown Error')
                print(f"❌ Fit Failed: {str(err)[:25]} ]")
                manager.shutdown()
                continue
    
            # 4. Success: Retrieve and Score
            fitted_model = return_dict['model']
    
            try:
                t_p = fitted_model.predict(X_train)
                t_rmse = round(root_mean_squared_error(y_train, t_p), 4)
                del t_p
    
                v_p = fitted_model.predict(X_val)
                v_rmse = round(root_mean_squared_error(y_val, v_p), 4)
                del v_p
    
                fname = f"{name}_{t_rmse}_{v_rmse}.joblib"
                joblib.dump(fitted_model, fname)
    
                # Update your results DataFrame
                new_row = {"Model": name, "Group": group, "Train_RMSE": t_rmse, "Val_RMSE": v_rmse, "File": fname}
                self.score_df = pd.concat([self.score_df, pd.DataFrame([new_row])], ignore_index=True)
                
                elapsed = round(time.time() - start_time, 1)
                print(f" ✅ Complected ] ({elapsed}s)")
    
            except Exception as e:
                print(f" ❌ Scoring Error: {str(e)[:20]} ]")
            
            finally:
                manager.shutdown()
                gc.collect()
    
    def inspection(self, model_name_or_file):
        """Inspects a model by name from the DF or by specific filename."""
        target = model_name_or_file
        if not target.endswith('.joblib'):
            # Try to find it in score_df
            row = self.score_df[self.score_df['Model'] == target]
            if not row.empty:
                target = row.iloc[0]['File']
        
        if os.path.exists(target):
            m = joblib.load(target)
            print(f"\n--- 🔍 Inspecting {target} ---")
            print(f"Class: {type(m).__name__}")
            print(f"Params: {m.get_params()}")
        else:
            print("Model file not found.")

    def cleanup_models(self):
        """Deletes all .joblib files and clears the score_df."""
        confirm = input("⚠️ Are you sure you want to delete ALL saved models? (y/n): ")
        if confirm.lower() == 'y':
            files_to_remove = [f for f in os.listdir('.') if f.endswith('.joblib')]
            for f in files_to_remove:
                os.remove(f)
            self.refresh_score_df()
            print(f"🧹 Cleaned up {len(files_to_remove)} files.")
            
    def zip_models(self, zip_name="zipped_model"):
        """
        Creates a zip file containing all .joblib files in the current directory.
        Replaces the file if it already exists.
        """
        self.refresh_score_df()
        if self.score_df.empty:
            print("⚠️ No models found to zip.")
            return

        # Define path
        zip_filename = f"{zip_name}.zip"
        
        # Remove existing zip if it exists
        if os.path.exists(zip_filename):
            os.remove(zip_filename)
        
        # Create a temporary directory to gather files 
        # (This prevents zipping other unrelated files in the folder)
        temp_dir = "temp_models_to_zip"
        os.makedirs(temp_dir, exist_ok=True)
        
        try:
            for file in self.score_df['File']:
                shutil.copy(file, os.path.join(temp_dir, file))
            
            # Create the zip from the temp directory
            shutil.make_archive(zip_name, 'zip', temp_dir)
            
            full_path = os.path.abspath(zip_filename)
            print(f"📦 Successfully zipped {len(self.score_df)} models.")
            print(f"📍 Path: {full_path}")
            
        finally:
            # Clean up the temp directory
            shutil.rmtree(temp_dir)
      
