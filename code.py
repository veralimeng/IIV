import pandas as pd
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
from itertools import combinations
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import os
import statsmodels.api as sm
from linearmodels.panel import PanelOLS
from tqdm import tqdm
from joblib import Parallel, delayed
from numba import njit

# Load the data
file_path = '/Users/vera/Desktop/experiment/crunchbase_export-4-10-2019.xlsx'
df = pd.read_excel(file_path, sheet_name='Rounds')

selected_columns = ['company_name', 'country_code', 'state_code', 'region', 'city', 'funding_round_type', 'announced_on', 'raised_amount_usd', 'investor_names', 'company_uuid']
filtered_df = df[selected_columns].copy()

# Find the company_uuids where either raised_amount_usd or region is missing
company_uuids_with_missing_data = filtered_df[filtered_df['raised_amount_usd'].isnull()]['company_uuid'].unique()

# Drop rows where company_uuid is in the list of company_uuids with missing raised_amount_usd or region
cleaned_df = filtered_df[~filtered_df['company_uuid'].isin(company_uuids_with_missing_data)]

# Add announced year column
cleaned_df['announced_on_year'] = pd.to_datetime(cleaned_df['announced_on']).dt.year

# Sum the 'raised_amount_usd' by the same company in same year and same round.
yearly_funding = cleaned_df.groupby(['company_name', 'region', 'funding_round_type', 'announced_on_year', 'announced_on', 'investor_names', 'company_uuid'])['raised_amount_usd'].sum()

# Reset index to convert Series to DataFrame
yearly_funding_df = yearly_funding.reset_index()
yearly_funding_df['investor_names'] = yearly_funding_df['investor_names'].astype(str)
grouped = yearly_funding_df.groupby(['announced_on_year', 'company_name', 'region', 'company_uuid'])

# Aggregate the data
yearly_funding_agg_df = grouped.agg(
    total_raised_amount_usd=pd.NamedAgg(column='raised_amount_usd', aggfunc='sum'),
    funding_round_types=pd.NamedAgg(column='funding_round_type', aggfunc=lambda x: list(x.unique())),
    all_investor_names=pd.NamedAgg(column='investor_names', aggfunc=lambda x: ', '.join(x.unique())),
    num_rounds_year=pd.NamedAgg(column='announced_on_year', aggfunc='size')
).reset_index()

# Compute the accumulated 'num_rounds_year' as the accumulated number until the year.
yearly_funding_agg_df['num_rounds_acc'] = yearly_funding_agg_df.groupby('company_uuid')['num_rounds_year'].cumsum()

# Change the investors to a list
def process_investor_names(investor_names_str):
    investors = investor_names_str.replace('Lead - ', '').split(',')
    unique_investors = list(set([investor.strip() for investor in investors]))
    return unique_investors

yearly_funding_agg_df['all_investor_names'] = yearly_funding_agg_df['all_investor_names'].apply(process_investor_names)
yearly_funding_agg_df = yearly_funding_agg_df.sort_values(by=['company_name', 'announced_on_year'])

# Load the company data
file_path = '/Users/vera/Desktop/experiment/crunchbase_export-4-10-2019.xlsx'
company_df = pd.read_excel(file_path, sheet_name='Funded Companies')

# Extract Twitter name from URL
def extract_twitter_name(url):
    if pd.isna(url) or 'twitter.com/' not in url:
        return None
    parts = url.split('twitter.com/')
    twitter_name = parts[1].split('?')[0]
    return twitter_name

company_df['Twitter_name'] = company_df['twitter_url'].apply(extract_twitter_name)
company_tweet_account_df = company_df[['category_group_list', 'Twitter_name', 'uuid', 'employee_count', 'founded_on', 'closed_on']]

# Split the string into a list of industries
def split_categories(categories):
    if pd.isna(categories):
        return []
    return [category.strip() for category in categories.split('|')]

company_tweet_account_df['category_group_list'] = company_tweet_account_df['category_group_list'].apply(split_categories)

# Link the funding dataframe with companies' Twitter name by company uuid
yearly_funding_twitter_account_df = pd.merge(yearly_funding_agg_df, company_tweet_account_df, left_on='company_uuid', right_on='uuid', how='inner')
yearly_funding_twitter_account_df = yearly_funding_twitter_account_df.drop(columns=['uuid'])

# Load the Twitter data
combined_tweet_df = pd.read_csv('/Users/vera/Desktop/experiment/combined_tweet_df.csv')

# Get the user file
user = pd.read_csv('/Users/vera/Desktop/experiment/user.csv')

# Function to clean strings: lowercase, remove spaces, and punctuation
def clean_string(s):
    return s.str.lower().str.replace(r'\s+', '', regex=True).str.replace(r'[^\w\s]', '', regex=True)

# Apply the function to Twitter_name and company_name in yearly_funding_twitter_account_df
yearly_funding_twitter_account_df['Twitter_name'] = clean_string(yearly_funding_twitter_account_df['Twitter_name'])
yearly_funding_twitter_account_df['company_name'] = clean_string(yearly_funding_twitter_account_df['company_name'])

# Apply the function to Twitter_name and company_name in user
user['Twitter_name'] = clean_string(user['Twitter_name'])
user['company_name'] = clean_string(user['company_name'])

matched_engage_df = pd.merge(yearly_funding_twitter_account_df, user, 
                             left_on=['Twitter_name', 'company_name'], 
                             right_on=['Twitter_name', 'company_name'], 
                             how='left')

# Perform a left merge to include all rows from funding_twitter_engage_df
merged_df = matched_engage_df.merge(
    combined_tweet_df, 
    left_on=['user_id', 'announced_on_year'], 
    right_on=['user_id', 'year'],
    how='outer'
)

# Function to fill NaN values within each group
def fill_na_in_group(group):
    for column in columns_to_fill:
        group[column] = group[column].fillna(method='ffill').fillna(method='bfill')
    return group

# Columns to fill
columns_to_fill = ['company_uuid', 'company_name', 'region', 'category_group_list', 'employee_count', 'founded_on', 'closed_on', 'Twitter_name']

# Filter out rows where 'user_id' is NaN
non_empty_user_id_df = merged_df.dropna(subset=['user_id'])

# Group by 'user_id' and apply the fill_na_in_group function
filled_df = non_empty_user_id_df.groupby('user_id', group_keys=False).apply(fill_na_in_group)

# Merge the filled data back to the original dataframe to include rows with NaN 'user_id'
merged_na_df = pd.concat([filled_df, merged_df[merged_df['user_id'].isna()]])

# Filling NaN values in specified columns with 0 where 'Twitter_name' is not None or not NaN
columns_to_fill = ['Twitter_interactions', 'Tweets_posted', 'total_raised_amount_usd', 'num_rounds_year']

condition = merged_na_df['Twitter_name'].notna()
merged_na_df.loc[condition, columns_to_fill] = merged_na_df.loc[condition, columns_to_fill].fillna(0)

# Fill NaN values in 'announced_on_year' with the corresponding values from 'year'
merged_na_df['announced_on_year'] = merged_na_df['announced_on_year'].fillna(merged_na_df['year'])

# Sort by 'company_uuid' and 'announced_on_year'
merged_na_df = merged_na_df.sort_values(by=['company_uuid', 'announced_on_year'])

# Compute the accumulated 'num_rounds_year' as the accumulated number until the year.
merged_na_df['num_rounds_acc'] = merged_na_df.groupby('company_uuid')['num_rounds_year'].cumsum()
funding_twitter_engage_df = merged_na_df.copy()

def compute_first_principal_component(df, column_names):
    # Extract the specified columns
    X = df[column_names]

    # Standardize the data
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Initialize PCA and compute the first principal component
    pca = PCA(n_components=1)
    principal_component = pca.fit_transform(X_scaled)[:, 0]

    # Return the first principal component
    return principal_component

funding_twitter_engage_df_copy = funding_twitter_engage_df.copy()
funding_twitter_engage_df_copy = funding_twitter_engage_df_copy.dropna(subset=['Twitter_interactions', 'Tweets_posted'])

column_names = ['Twitter_interactions', 'Tweets_posted']
funding_twitter_engage_df_copy['Twitter_engagement'] = compute_first_principal_component(funding_twitter_engage_df_copy, column_names)

# Merge the dataframes
funding_twitter_engage_df = funding_twitter_engage_df.merge(
    funding_twitter_engage_df_copy[['company_uuid', 'announced_on_year', 'Twitter_engagement']],
    on=['company_uuid', 'announced_on_year'],
    how='left'
)

def compute_log_lead_funding(row):
    # Find the matching row in yearly_funding_agg_df
    match = yearly_funding_agg_df[
        (yearly_funding_agg_df['company_uuid'] == row['company_uuid']) &
        (yearly_funding_agg_df['announced_on_year'] == row['announced_on_year'] + 1)
    ]

    # If match is found, compute log_lead_funding
    if not match.empty:
        return np.log(match['total_raised_amount_usd'].iloc[0] + 1)
    else:
        return 0

funding_twitter_df = funding_twitter_engage_df.copy()

# Compute Y (Log lead funding)
funding_twitter_df['log_lead_funding'] = funding_twitter_df.apply(compute_log_lead_funding, axis=1)

# Function to safely convert dates, setting invalid dates to NaT
def safe_to_datetime(date_series):
    return pd.to_datetime(date_series, errors='coerce')

# Convert 'closed_on' and 'founded_on' to datetime, coercing errors to NaT
funding_twitter_df['closed_on'] = safe_to_datetime(funding_twitter_df['closed_on'])
funding_twitter_df['founded_on'] = safe_to_datetime(funding_twitter_df['founded_on'])

# Reference date
reference_date = pd.to_datetime('2019-04-09')

# Function to calculate the age in years as a decimal
def calculate_age(founded_on, closed_on, reference_date):
    if pd.isna(founded_on):
        return np.nan
    if pd.isna(closed_on):
        delta = reference_date.to_pydatetime() - founded_on.to_pydatetime()
    else:
        delta = closed_on.to_pydatetime() - founded_on.to_pydatetime()
    return delta.days / 365.25

# Calculate 'age'
funding_twitter_df['age'] = funding_twitter_df.apply(
    lambda row: calculate_age(row['founded_on'], row['closed_on'], reference_date),
    axis=1
)

# Add 'has_Twitter' column based on whether 'Twitter_name' is not None or NaN
funding_twitter_df['has_Twitter'] = funding_twitter_df['Twitter_name'].notna()

# Category mapping
category_mapping = {
    'internet services': 'IT Services',
    'information technology': 'IT Services',
    'administrative services': 'IT Services',
    'data and analytics': 'IT Services',
    'professional services': 'IT Services',
    'privacy and security': 'IT Services',
    'software': 'Software',
    'platforms': 'Software',
    'apps': 'Software',
    'sales and marketing': 'Software',
    'artificial intelligence': 'Software',
    'design': 'Software',
    'education': 'Software',
    'media and entertainment': 'Software',
    'financial services': 'Software',
    'community and lifestyle': 'Software',
    'video': 'Software',
    'advertising': 'Software',
    'mobile': 'Communication Equipment',
    'messaging and telecommunications': 'Communication Equipment',
    'hardware': 'Technology Hardware, Storage & Peripherals',
    'consumer electronics': 'Electronic Equipment, Instruments & Components',
    'science and engineering': 'IT Services',
    'technology hardware': 'Technology Hardware, Storage & Peripherals',
    'electronic equipment': 'Electronic Equipment, Instruments & Components'
}

def map_relevant_categories(categories):
    if not categories:  # Check if the list is empty
        return []
    filtered_categories = [category for category in categories if category in category_mapping]
    mapped_list = [category_mapping[category] for category in filtered_categories]
    return mapped_list

# Apply the function to the 'category_group_list' column
funding_twitter_df['mapped_category_group_list'] = funding_twitter_df['category_group_list'].apply(map_relevant_categories)

# Remove rows with empty lists before exploding
funding_twitter_df_copy = funding_twitter_df[funding_twitter_df['mapped_category_group_list'].apply(len) > 0]

# Expand the DataFrame so each category gets its own row
expanded_df = funding_twitter_df_copy.explode('mapped_category_group_list').reset_index(drop=True)

# Set the random seed
np.random.seed(6)

# Function to determine the most frequent category per company_uuid
def most_frequent_category(categories):
    if categories.empty:
        return np.nan
    mode = categories.mode()
    if len(mode) > 1:  # Handle ties by randomly selecting one of the modes
        return np.random.choice(mode)
    else:
        return mode.iloc[0]

# Group by 'company_uuid' and apply the function to determine the most frequent category
market_df = expanded_df.groupby('company_uuid')['mapped_category_group_list'].apply(most_frequent_category).reset_index()

# Rename the columns
market_df.columns = ['company_uuid', 'market']

# Merge the 'market' back into the original DataFrame
funding_twitter_df_copy = funding_twitter_df.merge(market_df, on='company_uuid', how='outer')

funding_twitter_df_copy2 = funding_twitter_df_copy.copy()

# Filter out the rows where 'announced_on_year' is 2019
funding_twitter_df_copy2 = funding_twitter_df_copy2[funding_twitter_df_copy2['announced_on_year'] != 2019]

# Create the 'age_year' column by converting 'age' to integers where 'age' is not NaN
funding_twitter_df_copy2['age_year'] = funding_twitter_df_copy2['age'].apply(lambda x: int(x) if pd.notna(x) else np.nan)

# Find the IIV1 and IIV2
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

def apply_IIV_functions(df, n_jobs=40):
    new_df = df.copy()
    def apply_find_IIV1(row):
        return find_IIV1(row, new_df)
    def apply_find_IIV2(row):
        return find_IIV2(row, new_df)
    new_df['IIV1'] = Parallel(n_jobs=n_jobs)(delayed(apply_find_IIV1)(row) for _, row in tqdm(new_df.iterrows(), total=new_df.shape[0], desc="Calculating IIV1"))
    new_df['IIV2'] = Parallel(n_jobs=n_jobs)(delayed(apply_find_IIV2)(row) for _, row in tqdm(new_df.iterrows(), total=new_df.shape[0], desc="Calculating IIV2"))
    return new_df

funding_twitter_reg_df = funding_twitter_df_copy2.copy()

funding_twitter_reg_df = funding_twitter_reg_df[['announced_on_year', 'company_name', 'region',
                                                 'company_uuid', 'funding_round_types', 'all_investor_names', 
                                                 'num_rounds_year', 'num_rounds_acc', 'category_group_list',
                                                 'Twitter_name', 'employee_count', 'user_id', 'Twitter_engagement', 
                                                 'log_lead_funding', 'age', 'has_Twitter', 'market', 'age_year']]

def convert_to_set(x):
    if isinstance(x, list):
        return set(x)
    return np.nan

funding_twitter_reg_df['all_investor_names'] = funding_twitter_reg_df['all_investor_names'].apply(convert_to_set)
funding_twitter_reg_df['funding_round_types'] = funding_twitter_reg_df['funding_round_types'].apply(convert_to_set)
funding_twitter_reg_df['category_group_list'] = funding_twitter_reg_df['category_group_list'].apply(convert_to_set)

# Find the IIV1 and IIV2
funding_twitter_iiv_df = apply_IIV_functions(funding_twitter_reg_df)
funding_twitter_iiv_df.to_csv('/Users/vera/Desktop/experiment/result/iiv_experiment1.csv', index=False)
