!pip install -q kagglehub

import os
os.environ["KAGGLE_API_TOKEN"] = "KGAT_065b4fd74d80ac46715dc543dd2f2676"

import kagglehub

path = kagglehub.competition_download(
    "house-prices-advanced-regression-techniques"
)

print(path)

"""
House Price Prediction - Production-Grade ML Pipeline
========================================================
Solves the Kaggle House Prices Advanced Regression Techniques competition.
Implements modular data processing, feature engineering, and model evaluation.

Features:
- Dynamic dataset fetching via kagglehub
- Systematic EDA and missing value handling
- Feature extraction (square footage, bedrooms, bathrooms)
- Feature scaling and categorical encoding
- Multiple model comparison (Linear Regression, Ridge, Lasso)
- Comprehensive evaluation metrics (RMSE, R²)
"""

import os

os.environ["KAGGLE_API_TOKEN"] = "KGAT_065b4fd74d80ac46715dc543dd2f2676"

import pandas as pd
import numpy as np
import warnings
from pathlib import Path
from typing import Tuple, Dict, Any

import kagglehub
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error

warnings.filterwarnings('ignore')


# =====================================================================
# 1. DATA LOADING
# =====================================================================

def fetch_dataset() -> Path:
    """
    Fetch the House Prices dataset from Kaggle using kagglehub.

    Returns:
        Path: Directory path containing train.csv and test.csv
    """
    print("📥 Fetching House Prices dataset from Kaggle...")
    path = kagglehub.competition_download('house-prices-advanced-regression-techniques')
    print(f"✓ Dataset downloaded to: {path}\n")
    return Path(path)


def load_data(dataset_path: Path) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Load training and testing datasets.

    Args:
        dataset_path: Path to the directory containing train.csv and test.csv

    Returns:
        Tuple of (train_df, test_df)
    """
    train_path = dataset_path / 'train.csv'
    test_path = dataset_path / 'test.csv'

    train_df = pd.read_csv(train_path)
    test_df = pd.read_csv(test_path)

    print(f"✓ Train set shape: {train_df.shape}")
    print(f"✓ Test set shape: {test_df.shape}")
    return train_df, test_df


# =====================================================================
# 2. EXPLORATORY DATA ANALYSIS (EDA)
# =====================================================================

def perform_eda(train_df: pd.DataFrame) -> None:
    """
    Perform exploratory data analysis on the training dataset.

    Args:
        train_df: Training dataset
    """
    print("\n" + "="*70)
    print("EXPLORATORY DATA ANALYSIS (EDA)")
    print("="*70)

    print(f"\n📊 Dataset Info:")
    print(f"  - Total rows: {len(train_df)}")
    print(f"  - Total columns: {len(train_df.columns)}")
    print(f"  - Memory usage: {train_df.memory_usage().sum() / 1024**2:.2f} MB")

    print(f"\n📈 Target Variable (SalePrice) Statistics:")
    print(train_df['SalePrice'].describe().to_string())

    print(f"\n❌ Missing Values (Top 15):")
    missing = train_df.isnull().sum().sort_values(ascending=False).head(15)
    if len(missing) > 0:
        for col, count in missing.items():
            pct = (count / len(train_df)) * 100
            print(f"  {col:20} {count:5} ({pct:5.1f}%)")
    else:
        print("  No missing values")

    print(f"\n🔍 Data Types:")
    print(train_df.dtypes.value_counts())


# =====================================================================
# 3. FEATURE ENGINEERING & PREPROCESSING
# =====================================================================

def handle_missing_values(train_df: pd.DataFrame, test_df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Handle missing values systematically.
    - Numeric columns: forward fill or median imputation
    - Categorical columns: fill with mode or 'Unknown'

    Args:
        train_df: Training dataset
        test_df: Test dataset

    Returns:
        Tuple of (train_df, test_df) with missing values handled
    """
    print("\n" + "="*70)
    print("HANDLING MISSING VALUES")
    print("="*70)

    # Strategy: Fill numeric with median, categorical with mode
    for col in train_df.columns:
        if train_df[col].isnull().sum() > 0:
            if train_df[col].dtype in ['float64', 'int64']:
                # Numeric: use median
                median_val = train_df[col].median()
                train_df[col].fillna(median_val, inplace=True)
                test_df[col].fillna(median_val, inplace=True)
                print(f"✓ {col:20} (numeric) → filled with median: {median_val:.2f}")
            else:
                # Categorical: use mode or 'Unknown'
                mode_val = train_df[col].mode()[0] if len(train_df[col].mode()) > 0 else 'Unknown'
                train_df[col].fillna(mode_val, inplace=True)
                test_df[col].fillna(mode_val, inplace=True)
                print(f"✓ {col:20} (categorical) → filled with: {mode_val}")

    print(f"\n✓ No more missing values in train: {train_df.isnull().sum().sum()}")
    print(f"✓ No more missing values in test: {test_df.isnull().sum().sum()}")
    return train_df, test_df


def extract_key_features(train_df: pd.DataFrame, test_df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Extract and engineer key features related to:
    - Square footage
    - Number of bedrooms
    - Number of bathrooms

    Args:
        train_df: Training dataset
        test_df: Test dataset

    Returns:
        Tuple of (train_df, test_df) with engineered features
    """
    print("\n" + "="*70)
    print("FEATURE EXTRACTION & ENGINEERING")
    print("="*70)

    # Feature mapping based on typical house price dataset structure
    feature_mapping = {
        'GrLivArea': 'total_sqft',           # Total square footage
        'TotalBsmtSF': 'basement_sqft',      # Basement square footage
        '1stFlrSF': 'first_floor_sqft',      # First floor square footage
        '2ndFlrSF': 'second_floor_sqft',     # Second floor square footage
        'BedroomAbvGr': 'bedrooms',          # Bedrooms above ground
        'FullBath': 'full_bathrooms',        # Full bathrooms
        'HalfBath': 'half_bathrooms',        # Half bathrooms
    }

    # Extract and rename features
    for original, new in feature_mapping.items():
        if original in train_df.columns:
            train_df[new] = train_df[original]
            test_df[new] = test_df[original]
            print(f"✓ Extracted: {original:15} → {new}")

    # Engineer composite features
    if 'total_sqft' in train_df.columns:
        train_df['sqft_per_bedroom'] = train_df['total_sqft'] / (train_df['bedrooms'] + 1)
        test_df['sqft_per_bedroom'] = test_df['total_sqft'] / (test_df['bedrooms'] + 1)
        print(f"✓ Engineered: sqft_per_bedroom (total_sqft / bedrooms)")

    if 'full_bathrooms' in train_df.columns and 'half_bathrooms' in train_df.columns:
        train_df['total_bathrooms'] = train_df['full_bathrooms'] + 0.5 * train_df['half_bathrooms']
        test_df['total_bathrooms'] = test_df['full_bathrooms'] + 0.5 * test_df['half_bathrooms']
        print(f"✓ Engineered: total_bathrooms (full + 0.5 * half)")

    return train_df, test_df


def prepare_features(train_df: pd.DataFrame, test_df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame, list, list]:
    """
    Prepare features for modeling:
    - Separate numeric and categorical features
    - Encode categorical variables (one-hot encoding)
    - Select relevant features

    Args:
        train_df: Training dataset
        test_df: Test dataset

    Returns:
        Tuple of (X_train, X_test, numeric_cols, categorical_cols)
    """
    print("\n" + "="*70)
    print("FEATURE PREPARATION")
    print("="*70)

    # Identify feature types
    numeric_cols = train_df.select_dtypes(include=['int64', 'float64']).columns.tolist()
    categorical_cols = train_df.select_dtypes(include=['object']).columns.tolist()

    # Remove target variable if present
    if 'SalePrice' in numeric_cols:
        numeric_cols.remove('SalePrice')
    if 'Id' in numeric_cols:
        numeric_cols.remove('Id')

    print(f"✓ Numeric features: {len(numeric_cols)}")
    print(f"✓ Categorical features: {len(categorical_cols)}")

    # One-hot encode categorical variables
    if categorical_cols:
        print(f"\n🔀 Encoding categorical features (one-hot encoding)...")
        train_df = pd.get_dummies(train_df, columns=categorical_cols, drop_first=True)
        test_df = pd.get_dummies(test_df, columns=categorical_cols, drop_first=True)

        # Align test set with train set columns
        train_cols = set(train_df.columns)
        test_cols = set(test_df.columns)

        # Add missing columns to test set
        for col in train_cols - test_cols:
            test_df[col] = 0

        # Remove extra columns from test set
        test_df = test_df[train_df.columns]

        print(f"✓ Encoded categorical features")

    # Prepare feature matrices (exclude ID and target)
    feature_cols = [col for col in train_df.columns if col not in ['Id', 'SalePrice']]
    X_train = train_df[feature_cols].copy()
    X_test = test_df[feature_cols].copy()

    print(f"\n✓ Final feature matrix shape: {X_train.shape}")

    return X_train, X_test, numeric_cols, categorical_cols


# =====================================================================
# 4. FEATURE SCALING
# =====================================================================

def scale_features(X_train: pd.DataFrame, X_test: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame, StandardScaler]:
    """
    Standardize features using StandardScaler.

    Args:
        X_train: Training features
        X_test: Test features

    Returns:
        Tuple of (X_train_scaled, X_test_scaled, scaler)
    """
    print("\n" + "="*70)
    print("FEATURE SCALING")
    print("="*70)

    scaler = StandardScaler()
    X_train_scaled = pd.DataFrame(
        scaler.fit_transform(X_train),
        columns=X_train.columns,
        index=X_train.index
    )
    X_test_scaled = pd.DataFrame(
        scaler.transform(X_test),
        columns=X_test.columns,
        index=X_test.index
    )

    print(f"✓ Features scaled using StandardScaler")
    print(f"  - Mean of scaled train features: {X_train_scaled.mean().mean():.6f}")
    print(f"  - Std of scaled train features: {X_train_scaled.std().mean():.6f}")

    return X_train_scaled, X_test_scaled, scaler


# =====================================================================
# 5. MODEL TRAINING & EVALUATION
# =====================================================================

def train_models(X_train: pd.DataFrame, y_train: pd.Series) -> Dict[str, Any]:
    """
    Train multiple regression models:
    - Linear Regression (baseline)
    - Ridge Regression (L2 regularization)
    - Lasso Regression (L1 regularization)

    Args:
        X_train: Training features
        y_train: Training target

    Returns:
        Dictionary of trained models
    """
    print("\n" + "="*70)
    print("MODEL TRAINING")
    print("="*70)

    models = {}

    # Linear Regression (Baseline)
    print("\n🔧 Training Linear Regression...")
    lr_model = LinearRegression()
    lr_model.fit(X_train, y_train)
    models['Linear Regression'] = lr_model
    print("✓ Linear Regression trained")

    # Ridge Regression (L2 Regularization)
    print("🔧 Training Ridge Regression (alpha=10)...")
    ridge_model = Ridge(alpha=10.0)
    ridge_model.fit(X_train, y_train)
    models['Ridge'] = ridge_model
    print("✓ Ridge Regression trained")

    # Lasso Regression (L1 Regularization)
    print("🔧 Training Lasso Regression (alpha=1000)...")
    lasso_model = Lasso(alpha=1000.0, max_iter=2000)
    lasso_model.fit(X_train, y_train)
    models['Lasso'] = lasso_model
    print("✓ Lasso Regression trained")

    return models


def evaluate_models(models: Dict[str, Any], X_train: pd.DataFrame, X_test: pd.DataFrame,
                    y_train: pd.Series, y_test: pd.Series) -> Dict[str, Dict[str, float]]:
    """
    Evaluate models using RMSE, MAE, and R² metrics on train and test sets.

    Args:
        models: Dictionary of trained models
        X_train: Training features
        X_test: Test features
        y_train: Training target
        y_test: Test target

    Returns:
        Dictionary of metrics for each model
    """
    print("\n" + "="*70)
    print("MODEL EVALUATION")
    print("="*70)

    results = {}

    for model_name, model in models.items():
        print(f"\n📊 {model_name}")
        print("-" * 70)

        # Training predictions
        y_train_pred = model.predict(X_train)
        train_rmse = np.sqrt(mean_squared_error(y_train, y_train_pred))
        train_mae = mean_absolute_error(y_train, y_train_pred)
        train_r2 = r2_score(y_train, y_train_pred)

        # Test predictions
        y_test_pred = model.predict(X_test)
        test_rmse = np.sqrt(mean_squared_error(y_test, y_test_pred))
        test_mae = mean_absolute_error(y_test, y_test_pred)
        test_r2 = r2_score(y_test, y_test_pred)

        results[model_name] = {
            'train_rmse': train_rmse,
            'test_rmse': test_rmse,
            'train_mae': train_mae,
            'test_mae': test_mae,
            'train_r2': train_r2,
            'test_r2': test_r2
        }

        # Print metrics
        print(f"  Train RMSE:  ${train_rmse:,.2f}")
        print(f"  Test RMSE:   ${test_rmse:,.2f}")
        print(f"  Train MAE:   ${train_mae:,.2f}")
        print(f"  Test MAE:    ${test_mae:,.2f}")
        print(f"  Train R²:    {train_r2:.4f}")
        print(f"  Test R²:     {test_r2:.4f}")

    return results


def print_summary(results: Dict[str, Dict[str, float]]) -> None:
    """
    Print a summary comparison of all models.

    Args:
        results: Dictionary of metrics for each model
    """
    print("\n" + "="*70)
    print("MODEL COMPARISON SUMMARY")
    print("="*70)

    # Create summary dataframe
    summary_data = []
    for model_name, metrics in results.items():
        summary_data.append({
            'Model': model_name,
            'Test RMSE': f"${metrics['test_rmse']:,.2f}",
            'Test MAE': f"${metrics['test_mae']:,.2f}",
            'Test R²': f"{metrics['test_r2']:.4f}",
            'Overfit Risk': f"{metrics['train_r2'] - metrics['test_r2']:.4f}"
        })

    summary_df = pd.DataFrame(summary_data)
    print("\n" + summary_df.to_string(index=False))

    # Find best model
    best_model = min(results.items(), key=lambda x: x[1]['test_rmse'])
    print(f"\n🏆 Best Model: {best_model[0]} (Test RMSE: ${best_model[1]['test_rmse']:,.2f})")


# =====================================================================
# 6. MAIN PIPELINE
# =====================================================================

def main():
    """
    Execute the complete house price prediction pipeline.
    """
    print("\n" + "="*70)
    print("🏠 HOUSE PRICE PREDICTION - ML PIPELINE")
    print("="*70)

    try:
        # Step 1: Fetch and Load Data
        dataset_path = fetch_dataset()
        train_df, test_df = load_data(dataset_path)

        # Step 2: EDA
        perform_eda(train_df)

        # Step 3: Missing Value Handling
        train_df, test_df = handle_missing_values(train_df, test_df)

        # Step 4: Feature Engineering
        train_df, test_df = extract_key_features(train_df, test_df)
        X_train, X_test, numeric_cols, cat_cols = prepare_features(train_df, test_df)

        # Step 5: Feature Scaling
        X_train_scaled, X_test_scaled, scaler = scale_features(X_train, X_test)

        # Extract target variable
        y_train = train_df['SalePrice'].copy()

        # Step 6: Train-Test Split (from scaled training data)
        X_tr, X_val, y_tr, y_val = train_test_split(
            X_train_scaled, y_train, test_size=0.2, random_state=42
        )

        # Step 7: Model Training
        models = train_models(X_tr, y_tr)

        # Step 8: Model Evaluation
        results = evaluate_models(models, X_tr, X_val, y_tr, y_val)

        # Step 9: Summary
        print_summary(results)

        print("\n" + "="*70)
        print("✓ PIPELINE COMPLETED SUCCESSFULLY")
        print("="*70)

    except Exception as e:
        print(f"\n❌ Error occurred: {str(e)}")
        raise


if __name__ == "__main__":
    main()

import matplotlib.pyplot as plt
import seaborn as sns

# =====================================================================
# 1. MODIFIED MAIN PIPELINE (To return our variables out of local scope)
# =====================================================================
def run_pipeline_and_return_assets():
    print("\n" + "="*70)
    print("🏠 HOUSE PRICE PREDICTION - ML PIPELINE")
    print("="*70)

    try:
        # Step 1: Fetch and Load Data
        dataset_path = fetch_dataset()
        train_df, test_df = load_data(dataset_path)

        # Step 2: EDA
        perform_eda(train_df)

        # Step 3: Missing Value Handling
        train_df, test_df = handle_missing_values(train_df, test_df)

        # Step 4: Feature Engineering
        train_df, test_df = extract_key_features(train_df, test_df)
        X_train, X_test, numeric_cols, cat_cols = prepare_features(train_df, test_df)

        # Step 5: Feature Scaling
        X_train_scaled, X_test_scaled, scaler = scale_features(X_train, X_test)

        # Extract target variable
        y_train = train_df['SalePrice'].copy()

        # Step 6: Train-Test Split (from scaled training data)
        X_tr, X_val, y_tr, y_val = train_test_split(
            X_train_scaled, y_train, test_size=0.2, random_state=42
        )

        # Step 7: Model Training
        models = train_models(X_tr, y_tr)

        # Step 8: Model Evaluation
        results = evaluate_models(models, X_tr, X_val, y_tr, y_val)

        # Step 9: Summary
        print_summary(results)

        print("\n" + "="*70)
        print("✓ PIPELINE COMPLETED SUCCESSFULLY")
        print("="*70)

        # KEY FIX: Return these three objects so our notebook cell can use them!
        return models, X_val, y_val

    except Exception as e:
        print(f"\n❌ Error occurred: {str(e)}")
        raise

# =====================================================================
# 2. EXECUTE THE PIPELINE AND CAPTURE VARIABLES GLOBALly
# =====================================================================
# This line calls the function and unpacks the returned items into global scope
notebook_models, notebook_X_val, notebook_y_val = run_pipeline_and_return_assets()

# =====================================================================
# 3. GENERATE AND SAVE PLOT FOR LINKEDIN
# =====================================================================
print("\n🎨 Generating model visualization plot...")

# Grab the best performing model from the dictionary
best_lasso_model = notebook_models['Lasso']
y_val_pred = best_lasso_model.predict(notebook_X_val)

plt.figure(figsize=(9, 6))
# Plot actual vs predicted values
sns.scatterplot(x=notebook_y_val, y=y_val_pred, alpha=0.6, color='#6a0dad', edgecolor='w', s=50)

# Add a perfect prediction reference line (y = x)
line_min = min(notebook_y_val.min(), y_val_pred.min())
line_max = max(notebook_y_val.max(), y_val_pred.max())
plt.plot([line_min, line_max], [line_min, line_max], color='red', linestyle='--', lw=2, label='Perfect Prediction')

plt.title('Lasso Regression: Predicted vs. Actual House Prices', fontsize=14, fontweight='bold', pad=15)
plt.xlabel('Actual Sale Price ($)', fontsize=12)
plt.ylabel('Predicted Sale Price ($)', fontsize=12)
plt.gca().xaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f"${x:,.0f}"))
plt.gca().yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f"${x:,.0f}"))
plt.grid(True, linestyle=':', alpha=0.6)
plt.legend()
plt.tight_layout()

# Save the plot locally to upload on LinkedIn
plt.savefig('model_performance.png', dpi=300)
plt.show()

print("✓ Plot saved successfully as 'model_performance.png'!")
