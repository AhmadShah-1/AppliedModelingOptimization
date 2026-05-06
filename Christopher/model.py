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


def objFunc(x, w, y, obFuncType='mse', lam=0):
    """
    Parameters
    ----------
    x : ndarray (data)
    w : ndarray (coefficient)
    y : ndarray (target)
    obFuncType : str, optional
        The default is 'mse'.
        
    Returns
    -------
    fx : ndarray
    mse
    """
    if obFuncType == 'mse':
        #fx = np.mean((x @ w - y)**2)
        fx = (1/2) * np.mean((x @ w - y)**2)
    elif obFuncType == 'l1':
        #fx = np.mean((x @ w - y)**2) + lam * l1norm(w)
        fx = (1/2) * np.mean((x @ w - y)**2) + lam * l1norm(w[1:])
    elif obFuncType == 'l2':
        #fx = np.mean((x @ w - y)**2) + lam * l2norm(w)**2
        #fx = np.mean((x @ w - y)**2) + lam * np.sum(w**2)
        #fx = np.mean((x @ w - y)**2) + (1/2) * lam * np.sum(w**2)
        fx = (1/2) * np.mean((x @ w - y)**2) + (1/2) * lam * np.sum(w[1:]**2)
        
    return fx


def l1norm(u):
    """
    Computes the L1 norm of a vector
    """
    return np.linalg.norm(u, ord=1)

    
def gradFunc(X, w, y, obFuncType = 'mse', lam=0):
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
        #reg_term = 2 * lam * w
        reg_term = lam * w
        reg_term[0] = 0 # exclude bias term from regularization
        gradf =  1/n * X.T @ (X @ w - y) + reg_term
        return gradf, None
    
def gradient_descent(X, w, y, obFuncType='mse', lam=0, alpha=0.01, tol=1e-4, max_iter=1000):
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
    
    mse_vals.append(objFunc(X, w, y, obFuncType, lam)) # store initial MSE
    iter_vals.append(i) # store initial iteration number 
    
    w_new = np.zeros(len(w))
    
    while (i <= max_iter):
        # evaluate the gradient at current weights 
        gradient, soft_thresh = gradFunc(X, w, y, obFuncType, lam)
        
        
        if obFuncType == 'l1':
            z = w - alpha * gradient
            w_new = np.sign(z) * np.maximum(np.abs(z) - alpha * lam, 0)
            w_new[0] = z[0]  # don't regularize bias
        else:
            step_size = alpha * gradient
            w_new = w - step_size
            

        #if obFuncType != 'l1':
        #    w_new = w - step_size
        #else:
        #    #w_new = soft_thresh * (w + step_size)
        #    w_new = w + soft_thresh * step_size
        
        
            
        #if np.linalg.norm(step_size) < tol: # old criteria
        #if np.linalg.norm(gradient) < tol: # new criteria
        if np.linalg.norm(w_new - w) < tol: # new criteria
            break
        
        w = w_new
        mse_vals.append(objFunc(X, w, y, obFuncType, lam))
        
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

# create date time index 
#final_data.reset_index(inplace=True)

#from datetime import datetime, date

final_data['datetime'] = pd.to_datetime(final_data['Date'] + ' ' + final_data['Time'], errors='coerce')
#final_data['datetime'].unique() #14741 unique observations
final_data = pd.pivot_table(final_data, index=['PATIENT_ID', 'datetime'], columns='Code', values='Value')

# fill missing recordings with the mean of the code
final_data = final_data.fillna(final_data.mean())

# create target variable
final_data.columns = final_data.columns.astype(str)
final_data['mean'] = final_data[['58', '59', '60', '61', '62', '63']].mean(axis=1)
final_data['target'] = final_data['mean'].shift(-1) #final_data['mean'].pct_change()
final_data = final_data.dropna()

# add a bias term
#final_data = pd.concat([pd.Series(np.ones(final_data.shape[0]), index=final_data.index), final_data], axis=1)

# create the training and target columns 
final_data = final_data.sort_values(by='datetime', ascending=True)
X = final_data.drop(['58', '59', '60', '61', '62', '63', 'target'], axis=1)
#X.rename(columns={0 : 'bias'}, inplace=True)
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
X_train = pd.DataFrame(X_train, columns=np.insert(X.columns, 0, 'bias'))
#X_train.rename(columns={0 : 'bias'}, inplace=True)


# initialize weight vector 
w0 = np.repeat(1, len(X_train.columns)) # first term is bias intercept, remaining terms are coefficients to codes


# Optimize core MSE function 
w_opt, mse_vec, iter_nums = gradient_descent(X_train.values, w0, y_train.values, alpha=0.01) # run gradient descent on our training data 

plt.figure(figsize=(8,4))
plt.title('MSE vs. Iteration')
plt.xlabel('Iteration'); plt.ylabel('MSE')
plt.plot(iter_nums, mse_vec)
plt.savefig('plot1mse.png')

train_results = pd.DataFrame({'y_pred' : X_train @ w_opt, 
                              'y_act' : y_train.values})
train_results['error'] = (train_results['y_pred'] - train_results['y_act'])**2
print(f"Training Error (OG Model): {train_results['error'].mean()}")


# scale test data set but with respect to the training data
X_test = pd.DataFrame(X_test, columns=X.columns)
X_test = pd.DataFrame(scaler.transform(X_test), columns=X.columns)
X_test = np.c_[np.ones(len(X_test)), X_test]
X_test = pd.DataFrame(X_test, columns=np.insert(X.columns, 0, 'bias'))


test_results = pd.DataFrame({'y_pred' : X_test @ w_opt, 
                              'y_act' : y_test.values})
test_results['error'] = (test_results['y_pred'] - test_results['y_act'])**2
test_results['error'].mean() 
print(f"Testing Error (OG Model): {test_results['error'].mean()}")


# Optimize MSE function with Lasso Regularization (L1)
w_opt_l1, mse_vec_l1, iter_nums_l1 = gradient_descent(X_train.values, w0, y_train.values, alpha=0.01, obFuncType = 'l1', lam=0.1)

plt.figure(figsize=(8,4))
plt.title('MSE vs. Iteration (with L1 Regularization)')
plt.xlabel('Iteration'); plt.ylabel('MSE')
plt.plot(iter_nums_l1, mse_vec_l1)
plt.savefig('plot2l1reg.png')

train_results = pd.DataFrame({'y_pred' : X_train @ w_opt_l1, 
                              'y_act' : y_train.values})
train_results['error'] = (train_results['y_pred'] - train_results['y_act'])**2
print(f"Training Error (L1 Reg): {train_results['error'].mean()}")


# scale test data set but with respect to the training data
test_results = pd.DataFrame({'y_pred' : X_test @ w_opt_l1, 
                              'y_act' : y_test.values})
test_results['error'] = (test_results['y_pred'] - test_results['y_act'])**2
test_results['error'].mean() 
print(f"Testing Error (L1 Reg): {test_results['error'].mean()}")



# Optimize MSE function with Ridge Regularization (L2)
w_opt_l2, mse_vec_l2, iter_nums_l2 = gradient_descent(X_train.values, w0, y_train.values, alpha=0.01, obFuncType = 'l2', lam=0.1)

plt.figure(figsize=(8,4))
plt.title('MSE vs. Iteration (with L2 Regularization)')
plt.xlabel('Iteration'); plt.ylabel('MSE')
plt.plot(iter_nums_l2, mse_vec_l2)
plt.savefig('plot2l2reg.png')

#print("No reg:", w_opt)
#print("L2 reg:", w_opt_l2)

train_results = pd.DataFrame({'y_pred' : X_train @ w_opt_l2, 
                              'y_act' : y_train.values})
train_results['error'] = (train_results['y_pred'] - train_results['y_act'])**2
train_results['error'].mean()
print(f"Training Error (L2 Reg Model): {train_results['error'].mean()}")


# scale test data set but with respect to the training data
test_results = pd.DataFrame({'y_pred' : X_test @ w_opt_l2, 
                              'y_act' : y_test.values})
test_results['error'] = (test_results['y_pred'] - test_results['y_act'])**2
test_results['error'].mean() 
print(f"Testing Error (L2 Reg Model): {test_results['error'].mean()}")


################ Print coefficients ################################
print(f"OG Model Coefficients: {w_opt}")
print(f"L1 Regularization Model Coefficients: {w_opt_l1}")
print(f"L2 Regularization Model Coefficients: {w_opt_l2}")


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


