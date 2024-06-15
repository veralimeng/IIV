import pandas as pd
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
from itertools import combinations
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import random
import time
# from twitter.scraper import Scraper
print('test 1')
from typing import List, Any
import pyarrow
import os
import statsmodels.api as sm
import statsmodels.formula.api as smf
from statsmodels.sandbox.regression.gmm import IV2SLS
from linearmodels.panel import PanelOLS
print('test 2')
from statsmodels.stats.outliers_influence import variance_inflation_factor
from tqdm import tqdm

print('test 3')
import zipfile
import pandas as pd
from tqdm import tqdm
from joblib import Parallel, delayed
from numba import njit
import numpy as np
import requests
import os

# 假设 zip_file_url 是你的ZIP文件的URL
zip_file_url = "https://github.com/veralimeng/IIV/raw/main/funding_twitter_df_copy2.csv.zip"
csv_file_name = "funding_twitter_df_copy2.csv"
local_zip_path = "/tmp/funding_twitter_df_copy2.csv.zip"

# 下载ZIP文件到本地
response = requests.get(zip_file_url)
with open(local_zip_path, 'wb') as f:
    f.write(response.content)

print("ZIP文件下载完成")

# 打开ZIP文件
with zipfile.ZipFile(local_zip_path, 'r') as zip_ref:
    # 解压特定的CSV文件到内存
    with zip_ref.open(csv_file_name) as csv_file:
        # 使用Pandas读取CSV文件
        funding_twitter_df_copy2 = pd.read_csv(csv_file)
        
print("数据加载完成")


def ols_reg(funding_twitter_iiv_avg_df):
    # Set the multi-index for panel data
    funding_twitter_reg_df = funding_twitter_iiv_avg_df.copy()
    
    # funding_twitter_iiv_avg_df = funding_twitter_iiv_avg_df.dropna(subset=['IIV2_TE', 'age_year', 'num_rounds_acc'])
    
    # funding_twitter_reg_df = funding_twitter_reg_df.dropna(subset=['IIV2_TE', 'age_year', 'num_rounds_acc'])
    
    # Add a dummy time variable
    funding_twitter_reg_df['dummy_time'] = range(len(funding_twitter_reg_df))
    
    # Use for creating X_tilde but not in this model
    company_uuid_dummies = pd.get_dummies(funding_twitter_reg_df['company_uuid'], prefix='company_uuid', drop_first=True)
    company_uuid_dummies = company_uuid_dummies.astype(int)
    
    # Create an interaction term between 'company_uuid' and 'market'
    funding_twitter_reg_df['year_market'] = funding_twitter_reg_df['announced_on_year'].astype(str) + "_" + funding_twitter_reg_df['market'].astype(str)
    
    # Convert 'has_Twitter' to numeric
    funding_twitter_reg_df['has_Twitter'] = funding_twitter_reg_df['has_Twitter'].astype(int)
    
    funding_twitter_reg_df = funding_twitter_reg_df.set_index(['company_uuid', 'dummy_time'])
    
    # Create dummy variables for 'announced_on_year', dropping the first category to avoid multicollinearity
    year_dummies = pd.get_dummies(funding_twitter_reg_df['announced_on_year'], prefix='announced_on_year', drop_first=True)
    year_dummies = year_dummies.astype(int)
    
    # # Create dummy variables for 'year_market' and 'announced_on_year', dropping the first category to avoid multicollinearity
    year_market_dummies = pd.get_dummies(funding_twitter_reg_df['year_market'], prefix='year_market', drop_first = True)  # Limit to first 10 to reduce multicollinearity
    year_market_dummies = year_market_dummies.astype(int)
    
    # Concatenate the dummy variables with the original DataFrame
    funding_twitter_reg_df = pd.concat([funding_twitter_reg_df, year_market_dummies, year_dummies], axis=1)
    
    # Define the dependent and independent variables
    y = funding_twitter_reg_df['log_lead_funding']
    X = funding_twitter_reg_df[['Twitter_engagement', 'age_year', 'num_rounds_acc']]
    # X = funding_twitter_reg_df[['Twitter_engagement', 'has_Twitter']]
    
    X = pd.concat([X, year_market_dummies, year_dummies], axis=1)
    
    # # Fit the PanelOLS model with entity and time effects
    model = PanelOLS(y, X, entity_effects=True, time_effects=False, drop_absorbed=True, check_rank=False).fit()
    
    parameter_Twitter_engagement = model.params['Twitter_engagement']
    print(f"The parameter for Twitter_engagement is: {parameter_Twitter_engagement}")
    
    # parameter_has_Twitter = model.params['has_Twitter']
    # print(f"The parameter for has_Twitter is: {parameter_has_Twitter}")

    return model


print('Function ols_reg 加载完成')


def iv_2sls_reg(funding_twitter_iiv_avg_df, IIV = 'IIV1_TE'):

    # Assuming funding_twitter_reg_df and other preprocessing steps have been done as described
    funding_twitter_reg_df = funding_twitter_iiv_avg_df.copy()
    
    # Drop NaN values for relevant columns
    funding_twitter_reg_df = funding_twitter_reg_df.dropna(subset=['Twitter_engagement', 'IIV1_TE', 'log_lead_funding', 'IIV2_TE', 'age_year', 'num_rounds_acc'])
    
    # Add a dummy time variable
    funding_twitter_reg_df['dummy_time'] = range(len(funding_twitter_reg_df))
    funding_twitter_reg_df = funding_twitter_reg_df.set_index(['company_uuid', 'dummy_time'])
    
    # Create an interaction term between 'company_uuid' and 'market'
    funding_twitter_reg_df['year_market'] = funding_twitter_reg_df['announced_on_year'].astype(str) + "_" + funding_twitter_reg_df['market'].astype(str)
    
    # Create dummy variables for 'year_market' and 'announced_on_year', dropping the first category to avoid multicollinearity
    year_market_dummies = pd.get_dummies(funding_twitter_reg_df['year_market'], prefix='year_market', drop_first=True)
    year_dummies = pd.get_dummies(funding_twitter_reg_df['announced_on_year'], prefix='announced_on_year', drop_first=True)
    
    # Convert to numeric types
    year_market_dummies = year_market_dummies.astype(int)
    year_dummies = year_dummies.astype(int)
    
    # Concatenate the dummy variables with the original DataFrame
    funding_twitter_reg_df = pd.concat([funding_twitter_reg_df, year_market_dummies, year_dummies], axis=1)
    
    # Define the dependent, endogenous, exogenous variables, and instrument
    y = funding_twitter_reg_df['log_lead_funding']
    X = funding_twitter_reg_df[['Twitter_engagement']]  # Endogenous variable
    Z = funding_twitter_reg_df[[IIV]]  # Instrument
    W = funding_twitter_reg_df[['age_year', 'num_rounds_acc']]  # Other covariates
    W = pd.concat([W, year_market_dummies, year_dummies], axis=1)
    
    # Combine Z (instrument) with covariates for the first stage regression
    W_first_stage = pd.concat([W, Z], axis=1)
    
    # First Stage: Regress Twitter_engagement on instrument and covariates with fixed effects
    first_stage = PanelOLS(X, W_first_stage, entity_effects=True, check_rank=False, drop_absorbed=True).fit()
    funding_twitter_reg_df['Twitter_engagement_hat'] = first_stage.predict()
    
    # Combine predicted values with covariates for the second stage regression
    W_second_stage = pd.concat([W, funding_twitter_reg_df[['Twitter_engagement_hat']]], axis=1)
    
    # Second Stage: Regress log_lead_funding on predicted values from the first stage, covariates, and fixed effects
    second_stage = PanelOLS(y, W_second_stage, entity_effects=True, check_rank=False, drop_absorbed=True).fit()
    
    # Extract the parameter for Twitter_engagement_hat
    parameter_Twitter_engagement = second_stage.params['Twitter_engagement_hat']
    print(f"The parameter for Twitter_engagement is: {parameter_Twitter_engagement}")
    
    print(second_stage.summary)

    # # Regress Y on fixed effects and covariates to get residuals
    # model_y = PanelOLS(y, W, entity_effects=True, check_rank=False, drop_absorbed=True).fit()
    # y_residuals = model_y.resids
    
    # # Regress X on fixed effects and covariates to get residuals
    # model_x = PanelOLS(X, W, entity_effects=True, check_rank=False, drop_absorbed=True).fit()
    # X_residuals = model_x.resids
    
    # # Regress Z on fixed effects and covariates to get residuals
    # model_z = PanelOLS(Z, W, entity_effects=True, check_rank=False, drop_absorbed=True).fit()
    # Z_residuals = model_z.resids
    
    # # Calculate the covariances using the residuals
    # cov_Z_Y = np.cov(Z_residuals, y_residuals, rowvar=False)[0, 1]
    # cov_X_Z = np.cov(X_residuals, Z_residuals, rowvar=False)[0, 1]
     
    # # Calculate the 2SLS estimate
    # beta_2sls = cov_Z_Y / cov_X_Z
    # print(f"The estimate for Twitter_engagement using covariance is: {beta_2sls}")

    return X, y, W, funding_twitter_reg_df


print('Function iv_2sls_reg 加载完成')


# Define the function to calculate the average Twitter Engagement for a given year and IIV list
def add_iiv_avg_twitter_engagement(df, iiv_list_column, iiv_column_name):
    def avg_twitter_engagement_for_year(row):
        year = row['announced_on_year']
        iiv_list = row[iiv_list_column]
        filtered_df = df[(df['company_uuid'].isin(iiv_list)) & (df['announced_on_year'] == year)]
        
        # Filter out NaN values from the Twitter_engagement column
        filtered_df = filtered_df[filtered_df['Twitter_engagement'].notna()]
        
        if not filtered_df.empty:
            return filtered_df['Twitter_engagement'].mean()
        else:
            return float('nan')  # Return NaN if filtered_df is empty

    # Apply the function to each row
    df[iiv_column_name] = df.apply(avg_twitter_engagement_for_year, axis=1)
    
    return df


# In[15]:


def find_IIV1(row, df):
    
    year_filter = df['announced_on_year'] == row['announced_on_year']
    investor_filter = df['all_investor_names'].apply(lambda x: bool(x.intersection(row['all_investor_names'])) if pd.notna(x) and pd.notna(row['all_investor_names']) else False)
    funding_filter = df['funding_round_types'].apply(lambda x: bool(x.isdisjoint(row['funding_round_types'])) if pd.notna(x) and pd.notna(row['funding_round_types']) else False)
    
    iiv1 = df[year_filter & investor_filter & funding_filter]

    if iiv1.empty:
        return []
    else:
        return list(iiv1['company_uuid'])

def find_IIV2(row, df):

    year_filter = df['announced_on_year'] == row['announced_on_year']
    investor_filter = df['all_investor_names'].apply(lambda x: bool(x.isdisjoint(row['all_investor_names'])) if pd.notna(x) and pd.notna(row['all_investor_names']) else False)
    funding_filter = df['funding_round_types'].apply(lambda x: bool(x.intersection(row['funding_round_types'])) if pd.notna(x) and pd.notna(row['funding_round_types']) else False)
    
    iiv2 = df[year_filter & investor_filter & funding_filter]
    
    if iiv2.empty:
        return []
    else:
        return list(iiv2['company_uuid'])


def apply_IIV_functions(df, n_jobs=100):
    
    new_df = df.copy()
    
    def apply_find_IIV1(row):
        return find_IIV1(row, new_df)
    
    def apply_find_IIV2(row):
        return find_IIV2(row, new_df)

    # Applying IIV1 in parallel with limited jobs
    new_df['IIV1'] = Parallel(n_jobs=n_jobs)(delayed(apply_find_IIV1)(row) for _, row in tqdm(new_df.iterrows(), total=new_df.shape[0], desc="Calculating IIV1"))
    print('IIV1 finish')

    # Applying IIV2 in parallel with limited jobs
    new_df['IIV2'] = Parallel(n_jobs=n_jobs)(delayed(apply_find_IIV2)(row) for _, row in tqdm(new_df.iterrows(), total=new_df.shape[0], desc="Calculating IIV2"))
    print('IIV2 finish')
    
    return new_df


print('Function apply_IIV_functions 加载完成')



funding_twitter_reg_df = funding_twitter_df_copy2.copy()

funding_twitter_reg_df = funding_twitter_reg_df[['announced_on_year', 'company_name', 'region',
                                                 'company_uuid', 'funding_round_types', 'all_investor_names', 
                                                 'num_rounds_year', 'num_rounds_acc', 'category_group_list',
                                                 'Twitter_name', 'employee_count', 'user_id', 'Twitter_engagement', 
                                                 'log_lead_funding', 'age', 'has_Twitter', 'market', 'age_year']]

print('数据funding_twitter_reg_df已复制')

def convert_to_set(x):
    if isinstance(x, list):
        return set(x)
    return np.nan

funding_twitter_reg_df['all_investor_names'] = funding_twitter_reg_df['all_investor_names'].apply(convert_to_set)
funding_twitter_reg_df['funding_round_types'] = funding_twitter_reg_df['funding_round_types'].apply(convert_to_set)
funding_twitter_reg_df['category_group_list'] = funding_twitter_reg_df['category_group_list'].apply(convert_to_set)


print('Prepare to calculate IIV.')

# Find the IIV1 and IIV2
funding_twitter_iiv_df = apply_IIV_functions(funding_twitter_reg_df)


# Create a copy of the original DataFrame
funding_twitter_iiv_avg_df = funding_twitter_iiv_df.copy()

# Apply the function for IIV1 and IIV2
funding_twitter_iiv_avg_df = add_iiv_avg_twitter_engagement(funding_twitter_iiv_avg_df, 'IIV1', 'IIV1_TE')
print('IIV1 finish')
funding_twitter_iiv_avg_df = add_iiv_avg_twitter_engagement(funding_twitter_iiv_avg_df, 'IIV2', 'IIV2_TE')
print('IIV2 finish')


print('OLS+FE model summary', ols_reg(funding_twitter_iiv_avg_df))


# In[38]:


X, y, W, funding_twitter_reg_df = iv_2sls_reg(funding_twitter_iiv_avg_df, IIV = 'IIV1_TE')
X, y, W, funding_twitter_reg_df = iv_2sls_reg(funding_twitter_iiv_avg_df, IIV = 'IIV2_TE')


# In[44]:


def filter_company_uuid_dummies(company_uuid_dummies):

    # Extract company_uuid part from column names
    company_uuid_counts = company_uuid_dummies.sum(axis=0)
    
    # Identify columns with more than one observation
    valid_columns = company_uuid_counts[company_uuid_counts > 1].index
    
    # Filter the dummy variables to include only valid company_uuid columns
    valid_company_uuid_dummies = company_uuid_dummies[valid_columns]
    
    return valid_company_uuid_dummies


# In[45]:


def get_W(data, focal_variable, fixed_effect_dummies):
    
    # Extract the variables used in the model
    model_variables = model.params.index.tolist()
    
    # Exclude the focal variable
    model_variables = [var for var in model_variables if var != focal_variable]
    
    # Extract the relevant columns from the original data
    W = data[model_variables]

    # Create dummy variables for company_uuid and filter them
    filtered_company_uuid_dummies = filter_company_uuid_dummies(fixed_effect_dummies)
    
    # # Append the filtered company_uuid dummies to W
    # W = W.reset_index(drop=True).join(filtered_company_uuid_dummies.reset_index(drop=True))
    
    # Use all company_uuid dummies
    W = W.reset_index(drop=True).join(fixed_effect_dummies.reset_index(drop=True))
    
    return W


# In[46]:


def compute_tilde(X, Y, W):
    # Ensure X, Y, and W are numpy arrays
    X = np.array(X) if not isinstance(X, pd.DataFrame) else X.values
    Y = np.array(Y).reshape(-1, 1) if not isinstance(Y, pd.DataFrame) else Y.values.reshape(-1, 1)
    W = W if isinstance(W, pd.DataFrame) else pd.DataFrame(W)
    
    if X.ndim == 1:
        X = X.reshape(-1, 1)  # Ensure X is a 2D array with one column
    
    # Compute expectations
    n, n_w = W.shape
    if X.ndim == 1:
        n_x = 1  # Ensure X is a 2D array with one column
    else:
        n_x = X.shape[1]
    
    # Compute expectations using vectorized operations
    E_WtW = (W.T @ W) / n
    E_WtX = (W.T @ X) / n
    E_WtY = (W.T @ Y) / n
        
    # Compute (E(W'W))^{-1}
    E_WtW_inv = np.linalg.inv(E_WtW)

    # Compute the adjustment terms
    W_E_WtW_inv_E_WtX = (W @ E_WtW_inv).values @ E_WtX.values
    W_E_WtW_inv_E_WtY = (W @ E_WtW_inv).values @ E_WtY.values
    
    # Compute X_tilde and Y_tilde
    X_tilde = X - W_E_WtW_inv_E_WtX
    Y_tilde = Y - W_E_WtW_inv_E_WtY
    
    return X_tilde, Y_tilde


# In[47]:


X_tilde, Y_tilde = compute_tilde(X, y, W)
cov_Z2_Y = np.cov(funding_twitter_reg_df['IIV2_TE'], Y_tilde.flatten())[0, 1]
cov_Z2_X = np.cov(funding_twitter_reg_df['IIV2_TE'], X_tilde.flatten())[0, 1]
cov_Z1_Y = np.cov(funding_twitter_reg_df['IIV1_TE'], Y_tilde.flatten())[0, 1]
cov_Z1_X = np.cov(funding_twitter_reg_df['IIV1_TE'], X_tilde.flatten())[0, 1]

print('The lower bound for beta is: ', (cov_Z2_Y - cov_Z1_Y) / (cov_Z2_X - cov_Z1_X))
print('cov_Z1_X is: ', cov_Z1_X)
print('cov_Z1_Y is: ', cov_Z1_Y)
print('cov_Z2_X is: ', cov_Z2_X)
print('cov_Z2_Y is: ', cov_Z2_Y)

