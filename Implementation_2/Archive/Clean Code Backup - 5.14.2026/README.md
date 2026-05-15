# Blood Glucose Prediction with Regularized Gradient Descent

A machine learning project implementing custom gradient descent optimization algorithms to predict blood glucose levels from diabetes patient data. Compares baseline MSE, L1 (Lasso), L2 (Ridge), and Elastic Net regularization techniques.

## Project Overview

This project builds predictive models for blood glucose levels using custom implementations of gradient descent with various regularization methods. The models use pre-meal blood glucose measurements to predict the next measurement and are evaluated using MSE, RMSE, and MAPE metrics.

## Files & Data Processing

### Data Files
- **`data/diabetes.zip`**: Source compressed diabetes dataset
- **`data/extracted_data/`**: Individual extracted patient CSV files from the compressed archive (tab-separated format with Date, Time, Code, Value columns)
- **`data/transformed_data/`**: Patient-specific transformed data (70 patient files for rows 1-70 of extracted data)
- **`data/final_data/final_cleaned_data.csv`**: Consolidated dataset ready for modeling with all 70 patients combined

### Code Files

#### `extract_data.py`
Decompresses and extracts the diabetes dataset:
- Reads from `data/diabetes/diabetes-data.tar.Z`
- Extracts individual patient records and saves as CSV files to `data/extracted_data/`
- Transforms the first 70 files (patient data) by adding PATIENT_ID column
- Saves transformed data to `data/transformed_data/` with `_transformed.csv` suffix

#### `data_analysis.py`
Consolidates transformed patient data:
- Reads all transformed data files from `data/transformed_data/`
- Concatenates all 70 patient files row-wise
- Saves consolidated dataset to `data/final_data/final_cleaned_data.csv`

#### `model_preglucose.py`
Implements custom gradient descent optimization and model evaluation:
- Loads `data/final_data/final_cleaned_data.csv`
- Processes data:
  - Creates datetime from Date and Time columns
  - Pivots data by Code (medical codes) as columns
  - Fill missing activity or glucose measurements by the mean code measurement (per-patient) 
- Creates target variable:
  - Uses pre-meal blood glucose codes (58, 60, 62, 64) - pre-meal glucose measurments
  - Averages pre-meal measurements as feature
  - Shifts by -1 to create next-measurement target for each patient (i.e., for current patient indicators, predict next period pre-glucose)
- Feature engineering:
  - Drops pre-meal glucose codes and mean/target columns (i.e., prevent data leakage)
  - Fills remaining missing values with mode (i.e., across all patients)
  - Scales features using MinMaxScaler (fit on training data)
  - Adds bias term
- Trains four models:
  1. **Base Model**: MSE only (no regularization)
  2. **L1 Model**: MSE + L1 regularization (lam=0.1) with soft-thresholding
  3. **L2 Model**: MSE + L2 regularization (rho=0.1)
  4. **Elastic Net**: MSE + L1 (lam=0.1) + L2 (rho=0.1) regularization
- All models use 80/20 train-test split (no shuffle) with learning rate alpha=0.01
- Generates output files in `output/` directory

### Output Files
- **`output/mse_curves.png`**: 2×2 plot showing MSE convergence curves for all four models
- **`output/residual_plots.png`**: 4×2 residual plots (predictions vs. residuals) for train/test of each model
- **`output/qq_plots.png`**: 4×2 Q-Q plots checking normality of residuals
- **`output/metrics.csv`**: Performance metrics (MSE, RMSE, MAPE) for each model on train/test data
- **`output/opt_coefficients.csv`**: Optimized coefficients for each model across all features

## Blood Glucose Measurement Codes

Medical codes in dataset:
- **58**: Pre-breakfast blood glucose
- **59**: Post-breakfast blood glucose
- **60**: Pre-lunch blood glucose
- **61**: Post-lunch blood glucose
- **62**: Pre-supper blood glucose
- **63**: Post-supper blood glucose
- **64**: Pre-snack blood glucose

## Algorithm Details

### Gradient Descent Implementation
- **Objective Functions**:
  - MSE: `(1/2) * mean((X @ w - y)^2)` $\frac{1}{2} \sum_{i=1}^{n}((\mathbf{X} \mathbf{w} - \mathbf{y})^2)$
  - L1: MSE + `lam * ||w[1:]||_1` $\lambda \lVert \mathbf{w} \rVert_1$
  - L2: MSE + `(1/2) * rho * sum(w[1:]^2)` $\frac{1}{2} \rho \lVert \mathbf{w} \rVert_2^2$
  - Elastic Net: MSE + L1 + L2
- **Soft-Thresholding**: Applied for L1/Elastic Net regularization
- **Bias Exclusion**: Regularization applied only to feature coefficients, not bias term
- **Convergence**: Stops when `||w_new - w|| < 1e-4` or reaches max 1000 iterations

### Data Pipeline
1. Uncompress and extract diabetes patient records
2. Add patient identifiers and transform to standard format
3. Consolidate all patient data into single file
4. Pivot codes as features, create target variable (next pre-meal glucose)
5. Handle missing values via patient-level mean imputation
6. MinMax scale features using training data statistics
7. Split into 80% train / 20% test (preserves temporal order)
8. Train models and evaluate metrics

## Dependencies

- pandas
- numpy
- matplotlib
- scikit-learn
- statsmodels

## Usage

```bash
# Step 1: Extract and process raw data
python extract_data.py
python data_analysis.py

# Step 2: Train models and generate results
python model_preglucose.py
```

## Author

Chris Ognibene  
May 2026

## Notes

This is an educational project for CPE608, implementing regularized regression optimization algorithms from scratch to understand the mechanics of gradient descent and various regularization techniques.
