import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os
from pathlib import Path
from sklearn.preprocessing import MinMaxScaler, StandardScaler

# import final_data from directory final_data
base_path = Path("final_project_CPE608").parent
data_files_path = base_path / "final_data" / "final_cleaned_data.csv"

# store predictions and results from all optimization runs 
all_results_df = pd.DataFrame()

final_data = pd.read_csv(data_files_path, header=0)


def objFunc(x, w, y, obFuncType='mse', lam=0, rho=0):
    """
    Parameters
    ----------
    x : ndarray (data)
    w : ndarray (coefficient)
    y : ndarray (target)
    obFuncType : str, optional
        The default is 'mse'.
    lam : L1 regularization parameter
    rho : L2 regularization parameter
        
    Returns
    -------
    fx : ndarray
    mse
    """
    if obFuncType == 'mse':
        fx = (1/2) * np.mean((x @ w - y)**2)
    elif obFuncType == 'l1':
        fx = (1/2) * np.mean((x @ w - y)**2) + lam * l1norm(w[1:])
    elif obFuncType == 'l2':
        fx = (1/2) * np.mean((x @ w - y)**2) + (1/2) * rho * np.sum(w[1:]**2)
    elif obFuncType == 'elasticNet':
        fx = (1/2) * np.mean((x @ w - y)**2) + lam * l1norm(w[1:]) + (1/2) * rho * np.sum(w[1:]**2)
    return fx


def l1norm(u):
    """
    Computes the L1 norm of a vector
    """
    return np.linalg.norm(u, ord=1)

    
def gradFunc(X, w, y, obFuncType = 'mse', lam=0, rho=0):
    n = len(y)
    
    if obFuncType == 'mse': # gradient with no regularization
        gradf = 1/n * X.T @ (X @ w - y)
        return gradf, None
    
    elif obFuncType == 'l1': # gradient with L1 regularization
        soft_thresh = {}
        for i, wt in enumerate(w):
            if wt > lam:
                soft_thresh[i] = wt - lam 
            elif wt < -lam:
                soft_thresh[i] = wt + lam
            else:
                soft_thresh[i] = 0
        soft_thresh[0] = 0 # bias term is not regularized
        gradf = 1/n * X.T @ (X @ w - y)
        return gradf, pd.Series(soft_thresh).values
    
    elif obFuncType == 'l2': # gradient with L2 regularization
        reg_term = rho * w
        reg_term[0] = 0 # exclude bias term from regularization
        gradf =  1/n * X.T @ (X @ w - y) + reg_term
        return gradf, None
    
    elif obFuncType == 'elasticNet': # gradient with both L1 and L2 regularization
        l2_reg_term = rho * w
        gradf =  1/n * X.T @ (X @ w - y) + l2_reg_term
        return gradf, None
    
def gradient_descent(X, w, y, obFuncType='mse', lam=0, rho=0, alpha=0.01, tol=1e-4, max_iter=1000):
    """
    Parameters
    ----------
    X : ndarray (training observations)
    w : ndarray (coefficient)
    y : ndarray (target)
    obFuncType : str, optional
        The default is 'mse'.
    lam : float, optional
        Regularization parameter. The default is 0.
    alpha : float, optional
        learning rate. The default is 0.0001.
    tol : float, optional
        Convergence threshold. The default is 0.0001.
    max_iter : int, optional
        max number of iterations. The default is 1000.

    Returns
    -------
    w : ndarray
        vector of optimal coefficients
    mse_vals : ndarray
        vector of mse values per iteration
    iter_vals : ndarray
        vector of iteration numbers

    """
    
    i = 1 # keep track of iterations 
    
    mse_vals = [] # store mse values per iteration
    iter_vals = [] # store iteration numbers
    
    mse_vals.append(objFunc(X, w, y, obFuncType, lam, rho)) # store initial MSE
    iter_vals.append(i) # store initial iteration number 
    
    w_new = np.zeros(len(w))
    
    while (i <= max_iter):
        # evaluate the gradient at current weights 
        gradient, soft_thresh = gradFunc(X, w, y, obFuncType, lam, rho)
        
        if obFuncType == 'l1':
            z = w - alpha * gradient
            w_new = np.sign(z) * np.maximum(np.abs(z) - alpha * lam, 0)
            w_new[0] = z[0]  # don't regularize bias
        elif obFuncType == 'elasticNet':
                z = w - alpha * gradient
                w_new = np.sign(z) * np.maximum(np.abs(z) - alpha * lam, 0)
                w_new[0] = z[0]  # don't regularize bias
                
        else:
            step_size = alpha * gradient
            w_new = w - step_size
            

        if np.linalg.norm(w_new - w) < tol:
            break
        
        w = w_new
        mse_vals.append(objFunc(X, w, y, obFuncType, lam, rho))
        
        i += 1 # increment to next iteration
        iter_vals.append(i)

    return w, mse_vals, iter_vals

# compute the target variable
# average of the columns 
    #58 = Pre-breakfast blood glucose measurement
    #59 = Post-breakfast blood glucose measurement
    #60 = Pre-lunch blood glucose measurement
    #61 = Post-lunch blood glucose measurement
    #62 = Pre-supper blood glucose measurement
    #63 = Post-supper blood glucose measurement
    #64 = Pre-snack blood glucose measurement

all_glucose_cols = np.arange(58, 65).astype(str)
pre_glucose_cols = np.arange(58, 65, 2).astype(str)

final_data['datetime'] = pd.to_datetime(final_data['Date'] + ' ' + final_data['Time'], errors='coerce')
#final_data['datetime'].unique() #14741 unique observations
final_data = pd.pivot_table(final_data, index=['PATIENT_ID', 'datetime'], columns='Code', values='Value')

# fill missing recordings with the mean of the code
final_data = final_data.fillna(final_data.mean())

# create target variable
final_data.columns = final_data.columns.astype(str)
#final_data['mean'] = final_data[['58', '59', '60', '61', '62', '63']].mean(axis=1)
final_data['mean'] = final_data[pre_glucose_cols].mean(axis=1)
final_data['target'] = final_data['mean'].shift(-1) 
final_data = final_data.dropna()

# create the training and target columns 
final_data = final_data.sort_values(by='datetime', ascending=True)
#X = final_data.drop(['58', '59', '60', '61', '62', '63', 'mean', 'target'], axis=1)

X = final_data.drop(np.insert(pre_glucose_cols, -1, ['mean', 'target']), axis=1)
y = final_data['target']

# split data into train-test split
from sklearn.model_selection import train_test_split
test_pct = .2
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = test_pct, shuffle=False)

X_train = pd.DataFrame(X_train, columns=X.columns)
X_test = pd.DataFrame(X_test, columns=X.columns)

# scale the data (X only)
scaler = MinMaxScaler()
#X_train = pd.DataFrame(scaler.fit_transform(X_train), columns=X_train.columns)
X_train = pd.DataFrame(scaler.fit_transform(X_train), columns=X_train.columns)

X_train = np.c_[np.ones(len(X_train)), X_train]
X_train = pd.DataFrame(X_train, columns=np.insert(X.columns, 0, 'bias')) # add a bias term


# initialize weight vector 
w0 = np.repeat(1, len(X_train.columns)) # first term is bias intercept, remaining terms are coefficients to codes

# Optimize core MSE function 
w_opt, mse_vec, iter_nums = gradient_descent(X_train.values, w0, y_train.values, alpha=0.01) # run gradient descent on our training data 

base_train_results = pd.DataFrame({'y_pred' : X_train @ w_opt, 'y_act' : y_train.values})
base_train_results['error'] = (base_train_results['y_pred'] - base_train_results['y_act'])
base_train_results['abs error'] = np.abs((base_train_results['y_pred'] - base_train_results['y_act'])/base_train_results['y_act'])


# scale test data set but with respect to the training data
X_test = pd.DataFrame(X_test, columns=X.columns)
X_test = pd.DataFrame(scaler.transform(X_test), columns=X.columns)
X_test = np.c_[np.ones(len(X_test)), X_test]
X_test = pd.DataFrame(X_test, columns=np.insert(X.columns, 0, 'bias'))

base_test_results = pd.DataFrame({'y_pred' : X_test @ w_opt, 'y_act' : y_test.values})
base_test_results['error'] = (base_test_results['y_pred'] - base_test_results['y_act'])
base_test_results['abs error'] = np.abs((base_test_results['y_pred'] - base_test_results['y_act'])/base_test_results['y_act'])


# Optimize MSE function with Lasso Regularization (L1)
w_opt_l1, mse_vec_l1, iter_nums_l1 = gradient_descent(X_train.values, w0, y_train.values, alpha=0.01, obFuncType = 'l1', lam=0.1)

l1_train_results = pd.DataFrame({'y_pred' : X_train @ w_opt_l1, 'y_act' : y_train.values})
l1_train_results['error'] = (l1_train_results['y_pred'] - l1_train_results['y_act'])
l1_train_results['abs error'] = np.abs((l1_train_results['y_pred'] - l1_train_results['y_act'])/l1_train_results['y_act'])


l1_test_results = pd.DataFrame({'y_pred' : X_test @ w_opt_l1, 'y_act' : y_test.values})
l1_test_results['error'] = (l1_test_results['y_pred'] - l1_test_results['y_act'])
l1_test_results['abs error'] = np.abs((l1_test_results['y_pred'] - l1_test_results['y_act'])/l1_test_results['y_act'])


# set output dir
base_dir = Path("Christopher").parent
if Path(base_dir / "output_preGlucoseModel").is_dir():
    print("output_preGlucoseModel output directory exists")
else:
    Path(base_dir / "output_preGlucoseModel").mkdir(parents=True, exist_ok=True)
output_dir = base_dir / 'output_preGlucoseModel'

# Optimize MSE function with Ridge Regularization (L2)
w_opt_l2, mse_vec_l2, iter_nums_l2 = gradient_descent(X_train.values, w0, y_train.values, alpha=0.01, obFuncType = 'l2', rho=0.1)

l2_train_results = pd.DataFrame({'y_pred' : X_train @ w_opt_l2, 'y_act' : y_train.values})
l2_train_results['error'] = (l2_train_results['y_pred'] - l2_train_results['y_act'])
l2_train_results['abs error'] = np.abs((l2_train_results['y_pred'] - l2_train_results['y_act'])/l2_train_results['y_act'])


l2_test_results = pd.DataFrame({'y_pred' : X_test @ w_opt_l2, 'y_act' : y_test.values})
l2_test_results['error'] = (l2_test_results['y_pred'] - l2_test_results['y_act'])
l2_test_results['abs error'] = np.abs((l2_test_results['y_pred'] - l2_test_results['y_act'])/l2_test_results['y_act'])



# Optimize MSE function with Elastic Net Regularization
w_opt_elastic, mse_vec_elastic, iter_nums_elastic = gradient_descent(X_train.values, w0, y_train.values, alpha=0.01, obFuncType = 'elasticNet', lam=0.1, rho=0.1)

elastic_train_results = pd.DataFrame({'y_pred' : X_train @ w_opt_elastic, 'y_act' : y_train.values})
elastic_train_results['error'] = (elastic_train_results['y_pred'] - elastic_train_results['y_act'])
elastic_train_results['abs error'] = np.abs((elastic_train_results['y_pred'] - elastic_train_results['y_act'])/elastic_train_results['y_act'])

elastic_test_results = pd.DataFrame({'y_pred' : X_test @ w_opt_elastic, 'y_act' : y_test.values})
elastic_test_results['error'] = (elastic_test_results['y_pred'] - elastic_test_results['y_act'])
elastic_test_results['abs error'] = np.abs((elastic_test_results['y_pred'] - elastic_test_results['y_act'])/elastic_test_results['y_act'])


# plot MSE curves
fig, ax = plt.subplots(2,2, figsize=(10,10))

ax[0][0].plot(iter_nums, mse_vec) # base unconstrained model
ax[0][1].plot(iter_nums_l1, mse_vec_l1) # L1 regularization
ax[1][0].plot(iter_nums_l2, mse_vec_l2) # L2 regularization 
ax[1][1].plot(iter_nums_elastic, mse_vec_elastic) #elastic net

# plot customizations
ax[0][0].set_title('Base Model: MSE')
ax[0][1].set_title('L1 Model: MSE')
ax[1][0].set_title('L2 Model: MSE')
ax[1][1].set_title('Elastic Net Model: MSE')

ax[0][0].set_xlabel('Iteration'); ax[0][0].set_ylabel('MSE')
ax[0][1].set_xlabel('Iteration'); ax[0][1].set_ylabel('MSE')
ax[1][0].set_xlabel('Iteration'); ax[1][0].set_ylabel('MSE')
ax[1][1].set_xlabel('Iteration'); ax[1][1].set_ylabel('MSE')

plt.suptitle('MSE Cost Curves') # overall plot title 
fig.savefig(output_dir / 'mse_curves.png')

# MSE results
mse_res_agg = pd.Series([(base_train_results['error']**2).mean(), (base_test_results['error']**2).mean(), 
                         (l1_train_results['error']**2).mean(), (l1_test_results['error']**2).mean(), 
                         (l2_train_results['error']**2).mean(), (l2_test_results['error']**2).mean(), 
                         (elastic_train_results['error']**2).mean(), (elastic_test_results['error']**2).mean()])

mse_res_agg.columns = ['Base Train MSE', 'Base Test MSE', 
                       'L1 Train MSE', 'L1 Test MSE', 
                       'L2 Train MSE', 'L2 Test MSE', 
                       'Elastic Net Train MSE', 'Elastic Net Test MSE']

# RMSE results
rmse_res_agg = np.sqrt(mse_res_agg)
rmse_res_agg.columns = ['Base Train RMSE', 'Base Test RMSE', 
                        'L1 Train RMSE', 'L1 Test RMSE', 
                        'L2 Train RMSE', 'L2 Test RMSE', 
                        'Elastic Net Train RMSE', 'Elastic Net Test RMSE']

# MAPE results
mape_res_agg = pd.Series([base_train_results['abs error'].mean(), base_test_results['abs error'].mean(), 
                         l1_train_results['abs error'].mean(), l1_test_results['abs error'].mean(), 
                         l2_train_results['abs error'].mean(), l2_test_results['abs error'].mean(), 
                         elastic_train_results['abs error'].mean(), elastic_test_results['abs error'].mean()])
mape_res_agg.columns = ['Base Train MAPE', 'Base Test MAPE', 
                        'L1 Train MAPE', 'L1 Test MAPE', 
                        'L2 Train MAPE', 'L2 Test MAPE', 
                        'Elastic Net Train MAPE', 'Elastic Net Test MAPE']


# combine metrics
metrics_df = pd.concat([mse_res_agg, rmse_res_agg, mape_res_agg], axis=1)
metrics_df.index = ['Base Model (Train)', 'Base Model (Test)', 
                    'L1 Model (Train)', 'L1 Model (Test)', 
                    'L2 Model (Train)', 'L2 Model (Test)', 
                    'Elastic Net Model (Train)', 'Elastic Net Model (Test)'] 
metrics_df.columns = ['MSE', 'RMSE', 'MAPE']
metrics_df.to_csv(output_dir / 'metrics.csv')

# compile coefficients for each model 
coeff_df = pd.DataFrame(np.concatenate(
    (w_opt.reshape(-1,1), w_opt_l1.reshape(-1,1), w_opt_l2.reshape(-1,1), w_opt_elastic.reshape(-1,1)), axis=1))
coeff_df.index = X_train.columns
coeff_df.columns = ['Base Model', 'L1 Model', 'L2 Model', 'Elastic Net Model']
coeff_df.to_csv(output_dir / 'opt_coefficients.csv')

################ CHECKS ONLY ########################################
# check against sklearn ridge regression
from sklearn.linear_model import Ridge
y = y_train
X = X_train
clf = Ridge(alpha=.01)
clf.fit(X, y)
Ridge()

print(f"Sklearn ridge coefficients: {clf.coef_}")

# check against sklearn lasso regression
from sklearn.linear_model import Lasso
y = y_train
X = X_train
clf = Lasso(alpha=.01)
clf.fit(X, y)
Lasso()
print(f"Sklearn lasso coefficients: {clf.coef_}")

#######################Diagnostic Plots##############################

base_train = base_train_results[['y_pred', 'error']]
base_test = base_test_results[['y_pred', 'error']]
l1_train = l1_train_results[['y_pred', 'error']]
l1_test = l1_test_results[['y_pred', 'error']]
l2_train = l2_train_results[['y_pred', 'error']]
l2_test = l2_test_results[['y_pred', 'error']]
elastic_train = elastic_train_results[['y_pred', 'error']]
elastic_test = elastic_test_results[['y_pred', 'error']]

datasets = [base_train, base_test, l1_train, l1_test, l2_train, l2_test, elastic_train, elastic_test]
plot_titles = ['Base Model (Train)', 'Base Model (Test)',
                'L1 Model (Train)', 'L1 Model (Test)', 
                'L2 Model (Train)', 'L2 Model (Test)',
                'Elastic Net Model (Train)', 'Elastic Net Model (Test)']


# Create residual plots 
fig, ax = plt.subplots(4,2, figsize=(10,10))
dummy_array = np.zeros((4,2))

k = 0

for i, j in np.argwhere(dummy_array == 0):
    ax[i][j].scatter('y_pred', 'error', data=datasets[k])
    ax[i][j].set_title(plot_titles[k])
    ax[i][j].set_xlabel('Predictions')
    ax[i][j].set_ylabel('Residuals')
    k += 1
fig.suptitle("Residual Plots")
fig.tight_layout()
fig.savefig(output_dir / 'residual_plots.png')


# QQ Plots
import statsmodels.api as sm

fig, ax = plt.subplots(4,2, figsize=(10,10))
k = 0

for i, j in np.argwhere(dummy_array == 0):
    sm.qqplot(datasets[k]['error'], line='45', fit=True, ax=ax[i, j])
    ax[i][j].set_title(plot_titles[k])
    k += 1
fig.suptitle("Normal Q-Q Plots")
fig.tight_layout()
fig.savefig(output_dir / 'qq_plots.png')

# Multi-collinearity check
# Quick check for Multicollinearity (VIF)
from statsmodels.stats.outliers_influence import variance_inflation_factor
vif_data = pd.DataFrame()
vif_data["feature"] = X_train.columns[1:]
vif_data["VIF"] = [variance_inflation_factor(X_train.iloc[:, 1:].values, i) for i in range(len(X_train.columns[1:]))]
vif_data

# look at the p-values of coefficients in the final models 
# look at the R2 values of the final models 
# p-values: t-stat --> apply the inverse CDF --> p-value

# Cook's Distance  (DRAFT ONLY)
# References: https://www.statology.org/cooks-distance-python/
# Leverage: https://en.wikipedia.org/wiki/Leverage_(statistics)

#hat_matrix = X_train @ np.linalg.inv(X_train.T @ X_train) @ X_train.T # has perfect multi-collinearity
#hat_matrix = X_train @ np.linalg.pinv(X_train.T @ X_train) @ X_train.T # has perfect multi-collinearity
##H = X_with_const @ np.linalg.inv(X_with_const.T @ X_with_const) @ X_with_const.T

#H = (X_train @ np.linalg.pinv(X_train.T @ X_train)).T @ X_train
#leverage = np.diagonal(H)

def cooks_distance(resid, p, mse, lev):
    return (resid**2 / (p * mse)) * (lev/ (1-lev)**2)

#cooks_dist = cooks_distance(base_train_results['error'], len(X_train.columns[1:]),
#                            (base_train_results['error']**2).mean(), leverage[0])

# Partial Regression Plots
from statsmodels.graphics.regressionplots import plot_partregress_grid
