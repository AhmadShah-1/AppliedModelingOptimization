# final_project_CPE608

A machine learning project for predicting blood glucose levels using gradient descent optimization with various regularization techniques.

## Project Overview

This project implements custom gradient descent optimization algorithms to build predictive models for blood glucose levels from diabetes patient data. The implementation compares three modeling approaches:
- **Baseline Model**: Standard MSE optimization
- **L1 Regularization (Lasso)**: Feature selection through L1 penalty
- **L2 Regularization (Ridge)**: Coefficient shrinkage through L2 penalty

## Files

### Data Processing
- **`extract_data.py`**: Extracts and decompresses the diabetes dataset from `diabetes-data.tar.Z`, creates individual patient CSV files, and transforms them for further analysis.
- **`data_analysis.py`**: Consolidates transformed patient data files into a single cleaned dataset (`final_cleaned_data.csv`) for model training.

### Model Training
- **`model.py`**: Core implementation of:
  - Custom gradient descent optimizer with support for MSE, L1, and L2 regularization
  - Data preprocessing (MinMax scaling, train-test split)
  - Model training and evaluation
  - Comparison with scikit-learn implementations (Ridge and Lasso)

### Visualizations
- **`plot1mse.png`**: MSE convergence curve for baseline model
- **`plot2l1reg.png`**: MSE convergence curve with L1 regularization
- **`plot2l2reg.png`**: MSE convergence curve with L2 regularization

## Key Features

### Gradient Descent Implementation
- Supports multiple objective function types (MSE, L1, L2)
- Configurable learning rate, convergence tolerance, and maximum iterations
- Soft-thresholding for L1 regularization
- Bias term exclusion from regularization

### Data Pipeline
1. Extract compressed diabetes patient records
2. Transform time-series medical codes into features
3. Create target variable (next blood glucose measurement)
4. Scale features using MinMaxScaler
5. Split into train/test sets (80/20 split)

### Model Evaluation
- Training and testing error metrics
- Coefficient visualization across regularization methods
- Validation against scikit-learn implementations

## Usage

```bash
# Step 1: Extract and process data
python extract_data.py
python data_analysis.py

# Step 2: Train models and generate results
python model.py
```

## Data

The project uses diabetes patient data containing:
- **Codes 58-63**: Blood glucose measurements (pre/post meals)
  - 58: Pre-breakfast
  - 59: Post-breakfast
  - 60: Pre-lunch
  - 61: Post-lunch
  - 62: Pre-supper
  - 63: Post-supper

## Dependencies

- pandas
- numpy
- matplotlib
- scikit-learn

## Author

Chris Ognibene  
Date: May 2026

## Notes

This is a personal educational project for CPE608, implementing optimization algorithms from scratch to understand the mechanics of regularized regression.
