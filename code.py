#!/usr/bin/env python
# coding: utf-8

# In[1]:


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
from twitter.scraper import Scraper
from typing import List, Any
import pyarrow
import os
import statsmodels.api as sm
import statsmodels.formula.api as smf
from statsmodels.sandbox.regression.gmm import IV2SLS
from linearmodels.panel import PanelOLS
from statsmodels.stats.outliers_influence import variance_inflation_factor
from tqdm import tqdm


# # Company data from Crunchbase
# #### Twitter_name: Twitter user name, like @andpizza
# #### company_name: Twitter account name, like &pizza

# In[2]:


# Load the data
file_path = '/Users/vera/Desktop/experiment/crunchbase_export-4-10-2019.xlsx'
df = pd.read_excel(file_path, sheet_name = 'Rounds')


# In[3]:


selected_columns = ['company_name', 'country_code', 'state_code', 'region', 'city', 'funding_round_type', 'announced_on', 'raised_amount_usd', 'investor_names', 'company_uuid']
filtered_df = df[selected_columns].copy()

# Counting the missing values for each column in filtered_df
missing_values_count = filtered_df.isna().sum()
missing_values_table = pd.DataFrame(missing_values_count, columns = ['Missing Values'])
missing_values_table


# In[5]:


# Find the company_uuids where either raised_amount_usd or region is missing
company_uuids_with_missing_data = filtered_df[
    filtered_df['raised_amount_usd'].isnull()
]['company_uuid'].unique()

# Drop rows where company_uuid is in the list of company_uuids with missing raised_amount_usd or region
cleaned_df = filtered_df[~filtered_df['company_uuid'].isin(company_uuids_with_missing_data)]


# In[7]:


# not choosing data
cleaned_df


# In[8]:


# Counting the missing values for each column in cleaned_df
missing_values_count = cleaned_df.isna().sum()
missing_values_table = pd.DataFrame(missing_values_count, columns = ['Missing Values'])
missing_values_table


# In[9]:


# Add announced year column
cleaned_df['announced_on_year'] = pd.to_datetime(cleaned_df['announced_on']).dt.year

# Sum the 'raised_amount_usd' by the same company in same year and same round.
# yearly_funding = cleaned_df.groupby(['company_name', 'company_category_list', 'funding_round_type', 'announced_on_year', 'investor_names', 'company_uuid'])['raised_amount_usd'].sum()
# yearly_funding = cleaned_df.groupby(['company_name', 'country_code', 'state_code', 'region', 'city', 'company_category_list', 'funding_round_type', 'announced_on_year', 'investor_names', 'company_uuid'])['raised_amount_usd'].sum()
yearly_funding = cleaned_df.groupby(['company_name', 'region', 'funding_round_type', 'announced_on_year', 'announced_on', 'investor_names', 'company_uuid'])['raised_amount_usd'].sum()

# Reset index to convert Series to DataFrame
yearly_funding_df = yearly_funding.reset_index()
yearly_funding_df.head()


# In[10]:


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
    # Remove 'Lead - ' and split by ','
    investors = investor_names_str.replace('Lead - ', '').split(',')
    
    # Remove leading/trailing whitespace and duplicates, converting to a list
    unique_investors = list(set([investor.strip() for investor in investors]))

    return unique_investors

def parse_category_list(s):
    # Remove curly braces and split the string by comma
    categories = s.strip("{}").split(",")
    # Remove extra quotes and strip whitespace
    categories = [cat.replace('"', '').replace("'", "").strip() for cat in categories]
    return categories

# # Apply parse function to each value in the 'company_category_list' column
# yearly_funding_agg_df['company_category_list'] = yearly_funding_agg_df['company_category_list'].apply(parse_category_list)

# Apply the function to the 'all_investor_names' column
yearly_funding_agg_df['all_investor_names'] = yearly_funding_agg_df['all_investor_names'].apply(process_investor_names)
yearly_funding_agg_df = yearly_funding_agg_df.sort_values(by=['company_name', 'announced_on_year'])
yearly_funding_agg_df.head(5)


# In[ ]:





# ## Company Twitter name data

# In[11]:


# Load the data
file_path = '/Users/vera/Desktop/experiment/crunchbase_export-4-10-2019.xlsx'
company_df = pd.read_excel(file_path, sheet_name = 'Funded Companies')


# In[12]:


company_df.head()


# In[13]:


# Function to extract Twitter name from URL
def extract_twitter_name(url):
    if pd.isna(url) or 'twitter.com/' not in url:
        return None
    parts = url.split('twitter.com/')
    twitter_name = parts[1].split('?')[0]  # Splitting to remove any query parameters
    return twitter_name

# Apply the function to the 'twitter_url' column
company_df['Twitter_name'] = company_df['twitter_url'].apply(extract_twitter_name)

# Display the DataFrame with the new column
company_tweet_account_df = company_df[['category_group_list', 'Twitter_name', 'uuid', 'employee_count', 'founded_on', 'closed_on']]

# Function to split the string into a list of industries
def split_categories(categories):
    if pd.isna(categories):
        return []
    return [category.strip() for category in categories.split('|')]

# Apply the function to the 'category_group_list' column
company_tweet_account_df['category_group_list'] = company_tweet_account_df['category_group_list'].apply(split_categories)

company_tweet_account_df


# In[14]:


# Link the funding dataframe with companies' Twitter name by company uuid
yearly_funding_twitter_account_df = pd.merge(yearly_funding_agg_df, company_tweet_account_df, left_on='company_uuid', right_on='uuid', how='inner')
yearly_funding_twitter_account_df = yearly_funding_twitter_account_df.drop(columns=['uuid'])

# Drop the companies which are closed
yearly_funding_twitter_account_df.head(5)


# In[ ]:





# # Twitter data
# #### Tweets posted: the total number of Tweets posted in a year.
# #### Twitter interactions: the total number of reposts, comments, and likes in a year.
# #### To get the "Retweets and comments" total as displayed on the Twitter clients, simply add retweet_count and quote_count. (Twitter developer: https://developer.twitter.com/en/docs/twitter-api/metrics)
# 

# In[17]:


combined_tweet_df = pd.read_csv('/Users/vera/Desktop/experiment/combined_tweet_df'.csv)


# In[18]:


# # added
# combined_tweet_df_copy = combined_tweet_df.copy()
# column_names = ['Twitter_interactions', 'Tweets_posted']
# combined_tweet_df_copy['Twitter_engagement'] = compute_first_principal_component(combined_tweet_df_copy, column_names)


# In[19]:


# get the user file
user = pd.read_csv('/Users/vera/Desktop/experiment/user.csv')


# In[138]:


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


# In[60]:


# orginal matching
# funding_twitter_engage_df = pd.merge(filtered_combined_tweet_df, matched_engage_df, 
#                                      left_on=['user_id', 'year'],
#                                      right_on=['user_id', 'announced_on_year'],
#                                      how='right')
# funding_twitter_engage_df.head(10)


# In[525]:


# Perform a left merge to include all rows from funding_twitter_engage_df
merged_df = matched_engage_df.merge(
    filtered_combined_tweet_df, 
    left_on=['user_id', 'announced_on_year'], 
    right_on=['user_id', 'year'],
    how='outer'
)


# In[526]:


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


# In[527]:


# Compute the accumulated 'num_rounds_year' as the accumulated number until the year.
merged_na_df['num_rounds_acc'] = merged_na_df.groupby('company_uuid')['num_rounds_year'].cumsum()
funding_twitter_engage_df = merged_na_df.copy()
funding_twitter_engage_df.head()


# In[21]:


# Rapidfuzzy matching

# from rapidfuzz import process, fuzz
# from tqdm import tqdm
# import pandas as pd
# import re

# # Clean the strings by removing special characters and extra spaces
# def clean_string(s):
#     if isinstance(s, str):
#         return re.sub(r'\s+', ' ', re.sub(r'[^a-zA-Z0-9\s]', '', s)).strip().lower()
#     return ""

# # Apply cleaning to Twitter_name and company_name
# yearly_funding_twitter_account_df['Twitter_name'] = yearly_funding_twitter_account_df['Twitter_name'].apply(clean_string)
# yearly_funding_twitter_account_df['company_name'] = yearly_funding_twitter_account_df['company_name'].apply(clean_string)
# user['Twitter_name'] = user['Twitter_name'].apply(clean_string)
# user['company_name'] = user['company_name'].apply(clean_string)
# # combined_tweet_df['user_id'] = combined_tweet_df['user_id'].astype(str)

# def rapidfuzz_merge(df1, df2, keys1, keys2, threshold=80):
#     matches = []
#     choices = df2[keys2[0]].tolist()  # Accessing the Twitter_name column
#     choices_company = df2[keys2[1]].tolist()  # Accessing the company_name column
    
#     for index, row in tqdm(df1.iterrows(), total=df1.shape[0], desc=f'Matching {keys1}'):
#         if pd.notnull(row[keys1[0]]) and pd.notnull(row[keys1[1]]):
#             match_twitter = process.extractOne(row[keys1[0]], choices, scorer=fuzz.ratio)
#             match_company = process.extractOne(row[keys1[1]], choices_company, scorer=fuzz.ratio)
#             if match_twitter and match_twitter[1] >= threshold and match_company and match_company[1] >= threshold:
#                 matches.append((index, choices.index(match_twitter[0]), choices_company.index(match_company[0]), match_twitter[1], match_company[1]))
    
#     return matches

# # Perform fuzzy matching on 'Twitter_name' and 'company_name' for 'yearly_funding_twitter_account_df'
# matches_both = rapidfuzz_merge(yearly_funding_twitter_account_df, user, ['Twitter_name', 'company_name'], ['Twitter_name', 'company_name'], threshold=95)

# # Create DataFrame with matches
# matched_both_df = pd.DataFrame(matches_both, columns=['index_df1', 'index_df2_twitter', 'index_df2_company', 'similarity_twitter', 'similarity_company'])

# # Merge the fuzzy matched results back to the original DataFrames
# merged_both_df = yearly_funding_twitter_account_df.iloc[matched_both_df['index_df1']].reset_index(drop=True).merge(
#     user.iloc[matched_both_df['index_df2_twitter']].reset_index(drop=True),
#     left_index=True, right_index=True,
#     suffixes=('_yearly_funding', '_user'),
#     how='left'
# )

# # # Perform fuzzy matching on the new merged DataFrame and 'combined_tweet_df'
# # matches_combined_tweet = rapidfuzz_merge(merged_both_df, combined_tweet_df, ['user_id_yearly_funding'], ['user_id'], threshold=95)

# # # Create DataFrame with matches
# # matched_combined_tweet_df = pd.DataFrame(matches_combined_tweet, columns=['index_df1', 'index_df2', 'similarity'])


# In[22]:


# combined_tweet_df['user_id'] = combined_tweet_df['user_id'].astype(str)
# merged_both_df['user_id'] = merged_both_df['user_id'].astype(str)

# # Merge the fuzzy matched results with 'combined_tweet_df' using left join to retain all information from 'yearly_funding_twitter_account_df'
# final_merged_df = pd.merge(
#     merged_both_df,
#     combined_tweet_df,
#     left_on=['user_id', 'announced_on_year'],
#     right_on=['user_id', 'year'],
#     how='left'
# )


# In[23]:


# funding_twitter_engage_df = pd.merge(
#     yearly_funding_twitter_account_df,
#     final_merged_df[['company_uuid', 'announced_on_year', 'Twitter_interactions', 'Tweets_posted']],
#     on=['company_uuid', 'announced_on_year'],
#     how='left'
# )


# In[24]:


# # Assuming you want to save the DataFrame as a CSV file named 'funding_twitter_engage.csv'
# funding_twitter_engage_df.to_csv('/Users/vera/Desktop/funding_twitter_engage.csv', index=False)


# # Other variable construction

# ## Twitter engagement

# In[528]:


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


# In[544]:


# # Update 'Twitter_interactions' and 'Tweets_posted' if 'Twitter_name' is not NaN or None
# condition = funding_twitter_engage_df['Twitter_name'].notna() & funding_twitter_engage_df['Twitter_interactions'].isna() & funding_twitter_engage_df['Tweets_posted'].isna()
# funding_twitter_engage_df.loc[condition, ['Twitter_interactions', 'Tweets_posted']] = 0

funding_twitter_engage_df_copy = funding_twitter_engage_df.copy()
funding_twitter_engage_df_copy = funding_twitter_engage_df_copy.dropna(subset=['Twitter_interactions', 'Tweets_posted'])

column_names = ['Twitter_interactions', 'Tweets_posted']
funding_twitter_engage_df_copy['Twitter_engagement'] = compute_first_principal_component(funding_twitter_engage_df_copy, column_names)

# Next use funding_twitter_engage_df to compute Y then merge.


# In[553]:


# Merge the dataframes
funding_twitter_engage_df = funding_twitter_engage_df.merge(
    funding_twitter_engage_df_copy[['company_uuid', 'announced_on_year', 'Twitter_engagement']],
    on=['company_uuid', 'announced_on_year'],
    how='left'
)


# In[23]:


# # change Twitter name and company name to lowercase
# yearly_funding_twitter_account_df['Twitter_name'] = yearly_funding_twitter_account_df['Twitter_name'].str.lower()
# yearly_funding_twitter_account_df['company_name'] = yearly_funding_twitter_account_df['company_name'].str.lower()
# user['Twitter_name'] = user['Twitter_name'].str.lower()
# user['company_name'] = user['company_name'].str.lower()

# matched_engage_df = pd.merge(yearly_funding_twitter_account_df, user, 
#                              left_on=['Twitter_name', 'company_name'], 
#                              right_on=['Twitter_name', 'company_name'], 
#                              how='inner')
# funding_twitter_engage_df = pd.merge(combined_tweet_df, matched_engage_df, 
#                                      left_on=['user_id', 'year'],
#                                      right_on=['user_id', 'announced_on_year'],
#                                      how='inner')
# funding_twitter_engage_df.head()


# In[ ]:





# ## Log lead funding

# In[555]:


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


# In[556]:


funding_twitter_df = funding_twitter_engage_df.copy()

# Compute Y (Log lead funding)
funding_twitter_df['log_lead_funding'] = funding_twitter_df.apply(compute_log_lead_funding, axis=1)
# funding_twitter_df = funding_twitter_df[funding_twitter_df['announced_on_year'] < 2019]


# In[557]:


funding_twitter_df


# In[ ]:





# ## Age

# In[558]:


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

funding_twitter_df.head()


# ## Has Twitter

# In[559]:


# Add 'has_Twitter' column based on whether 'Twitter_name' is not None or NaN
funding_twitter_df['has_Twitter'] = funding_twitter_df['Twitter_name'].notna()

# # Convert if accumulated funding rounds >= 10 to 10
# funding_twitter_df['num_rounds_acc'] = funding_twitter_df['num_rounds_acc'].clip(upper=10)

funding_twitter_df.head()


# In[535]:


# # Splitting string values and converting to list for 'funding_round_types' column
# funding_twitter_df['funding_round_types'] = funding_twitter_df['funding_round_types'].apply(lambda x: x.split(',') if isinstance(x, str) else [])

# # Splitting string values and converting to list for 'all_investor_names' column
# funding_twitter_df['all_investor_names'] = funding_twitter_df['all_investor_names'].apply(lambda x: x.split(',') if isinstance(x, str) else [])

# # Splitting string values and converting to list for 'category_group_list' column
# funding_twitter_df['category_group_list'] = funding_twitter_df['category_group_list'].apply(lambda x: x.split(',') if isinstance(x, str) else [])


# In[536]:


# # Explode the lists into separate rows
# exploded_df = funding_twitter_df.explode('category_group_list')

# # Get all unique categories
# unique_categories = exploded_df['category_group_list'].unique()

# # Convert to a list
# unique_categories_list = unique_categories.tolist()


# ## Market time

# In[560]:


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
    # 'navigation and mapping': 'Software',
    'manufacturing': 'Technology Hardware, Storage & Peripherals',
    # 'natural resources': 'Technology Hardware, Storage & Peripherals',
    'consumer electronics': 'Electronic Equipment, Instruments & Components',
    # 'biotechnology': 'Electronic Equipment, Instruments & Components',
    # 'energy': 'Electronic Equipment, Instruments & Components',
    # 'agriculture and farming': 'Electronic Equipment, Instruments & Components',
    'science and engineering': 'IT Services',
    'technology hardware': 'Technology Hardware, Storage & Peripherals',
    'electronic equipment': 'Electronic Equipment, Instruments & Components'
}

def map_relevant_categories(categories):
    if not categories:  # Check if the list is empty
        return []
    # Filter categories to include only those in the category_mapping
    filtered_categories = [category for category in categories if category in category_mapping]
    # Map the filtered categories
    mapped_list = [category_mapping[category] for category in filtered_categories]
    return mapped_list
    
# Apply the function to the 'category_group_list' column
funding_twitter_df['mapped_category_group_list'] = funding_twitter_df['category_group_list'].apply(map_relevant_categories)
funding_twitter_df.head()


# In[575]:


funding_twitter_df_copy = funding_twitter_df.copy()

# Remove rows with empty lists before exploding
funding_twitter_df_copy = funding_twitter_df_copy[funding_twitter_df_copy['mapped_category_group_list'].apply(len) > 0]

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
funding_twitter_df_copy.head()


# In[576]:


funding_twitter_df_copy2 = funding_twitter_df_copy.copy()
len(funding_twitter_df_copy2)


# In[577]:


# Filter out the rows where 'announced_on_year' is 2019
funding_twitter_df_copy2 = funding_twitter_df_copy2[funding_twitter_df_copy2['announced_on_year'] != 2019]

# funding_twitter_df_copy2 = funding_twitter_df_copy2.dropna(subset=['region'])
# funding_twitter_df_copy2 = funding_twitter_df_copy2.dropna(subset=['all_investor_names'])

# Create the 'age_year' column by converting 'age' to integers where 'age' is not NaN
funding_twitter_df_copy2['age_year'] = funding_twitter_df_copy2['age'].apply(lambda x: int(x) if pd.notna(x) else np.nan)

# # Drop the last row of the 'closed' company
# # Filter to include only rows where 'closed_on' is not NaN
# filtered_df = funding_twitter_df_copy2[~funding_twitter_df_copy2['closed_on'].isna()]

# # Identify the last row for each 'company_uuid'
# last_rows = filtered_df.groupby('company_uuid')['closed_on'].idxmax()

# # Drop these rows from the original DataFrame
# funding_twitter_df_copy2 = funding_twitter_df_copy2.drop(last_rows)


# funding_twitter_df_copy2 = funding_twitter_df_copy2.dropna(subset=['company_name'])
# funding_twitter_df_copy2 = funding_twitter_df_copy2.dropna(subset=['market'])
# funding_twitter_df_copy2 = funding_twitter_df_copy2.dropna(subset=['Twitter_engagement'])
# funding_twitter_df_copy2 = funding_twitter_df_copy2.dropna(subset=['age'])


# In[578]:


len(funding_twitter_df_copy2)


# In[ ]:


# now every company has a 'market' type


# In[ ]:





# # Constructing IIV

# In[43]:


# Potential network
## age, age of the year
## employee_count
## use both all data or the has Twitter data to test.


# ## Test 1
# #### X network: sharing investors
# #### Y network: same numbers of accumulated rounds

# In[515]:


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


# In[517]:


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


# In[518]:


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


# In[666]:


# # Define the safe_intersection function
# def safe_intersection(list1, list2):
#     result1 = pd.isna(list1)
#     result2 = pd.isna(list2)
#     if isinstance(result1, np.ndarray):
#         result1 = result1.all()
#     if isinstance(result2, np.ndarray):
#         result2 = result2.all()

#     if result1 or result2:
#         return False
#     else:
#         return set(list1).intersection(set(list2))

# # Define the safe_non_equality function
# def safe_isdisjoint(list1, list2):
#     result1 = pd.isna(list1)
#     result2 = pd.isna(list2)
#     if isinstance(result1, np.ndarray):
#         result1 = result1.all()
#     if isinstance(result2, np.ndarray):
#         result2 = result2.all()

#     if result1 or result2:
#         return False
#     else:
#         return set(list1).isdisjoint(set(list2))

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

    # Applying IIV1 in parallel with limited jobs
    new_df['IIV1'] = Parallel(n_jobs=n_jobs)(delayed(apply_find_IIV1)(row) for _, row in tqdm(new_df.iterrows(), total=new_df.shape[0], desc="Calculating IIV1"))
    print('IIV1 finish')

    # Applying IIV2 in parallel with limited jobs
    new_df['IIV2'] = Parallel(n_jobs=n_jobs)(delayed(apply_find_IIV2)(row) for _, row in tqdm(new_df.iterrows(), total=new_df.shape[0], desc="Calculating IIV2"))
    print('IIV2 finish')
    
    return new_df



# In[661]:


from tqdm import tqdm
from joblib import Parallel, delayed
from numba import njit


# In[667]:


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
funding_twitter_iiv_df.head()


# In[ ]:


funding_twitter_iiv_df.to_csv('/Users/vera/Desktop/experiment/result/iiv_experiment1.csv', index=False)



















