# 🏠 House Price Prediction: Production-Grade ML Pipeline

An end-to-end Machine Learning pipeline designed to predict residential home prices using the Kaggle **House Prices: Advanced Regression Techniques** dataset. Built with production-grade practices, featuring robust data engineering, systematic missing value handling, feature scaling, and multi-model regularization techniques.

## 🚀 Key Features
* **Automated Data Ingestion:** Dynamic dataset fetching using `kagglehub`.
* **Robust Data Engineering:** Automated imputation strategies (Median for continuous, Mode for categorical).
* **Feature Engineering:** Extracted high-impact domain features like `sqft_per_bedroom` and `total_bathrooms`.
* **Regularization Analysis:** Implemented and evaluated **Linear Regression**, **Ridge (L2)**, and **Lasso (L1)** models to address high-dimensionality and multi-collinearity.

## 📊 Pipeline Architecture & Results
The final dataset shape reached **253 features** post-one-hot encoding. Due to high dimensionality, the baseline Linear Regression suffered heavily from overfitting. Regularization via Lasso provided the most stable generalization.

### Model Performance Metrics

| Model | Validation RMSE | Validation MAE | Validation R² | Overfit Risk |
| :--- | :---: | :---: | :---: | :---: |
| **Linear Regression** | \$51,364.99 | \$20,263.19 | 0.6560 | 0.2799 |
| **Ridge (L2, α=10.0)** | \$36,082.81 | \$19,673.26 | 0.8303 | 0.0991 |
| **Lasso (L1, α=1000)** | **\$31,058.23** | **\$18,187.55** | **0.8742** | **0.0135** |

🏆 **Best Performer:** **Lasso Regression** reached an R² score of **87.42%** with near-zero overfitting risk. Visual outputs are saved in the `outputs/` folder.

## 🛠️ Tech Stack
* **Language:** Python 3.x
* **Libraries:** Pandas, NumPy, Scikit-Learn
* **Data Source:** Kaggle API via `kagglehub`
