#!/usr/bin/env python
# coding: utf-8

# In[131]:


import streamlit as st
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.impute import SimpleImputer
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA


# In[132]:


# Load the dataset (replace 'your_dataset.csv' with the actual file name)
try:
    data= pd.read_csv('World_development_mesurement (1).csv')
except FileNotFoundError:
    st.error("Dataset not found. Please upload 'your_dataset.csv' to the app's directory.")
    st.stop()


# In[134]:


# Impute missing values in numerical columns
numerical_cols = ['Birth Rate','CO2 Emissions','Days to Start Business','Energy Usage','Health Exp % GDP','Hours to do Tax','Infant Mortality Rate','Internet Usage','Lending Interest','Life Expectancy Female','Life Expectancy Male','Mobile Phone Usage','Number of Records','Population 0-14','Population 15-64','Population 65+','Population Total','Population Urban']
imputer_numeric = SimpleImputer(strategy='mean')
data[numerical_cols] = imputer_numeric.fit_transform(data[numerical_cols])

categorical_cols = ['Country']
imputer_categorical = SimpleImputer(strategy='most_frequent')
data[categorical_cols] = imputer_categorical.fit_transform(data[categorical_cols])


# In[135]:


data.info()


# In[140]:


def clean_and_convert(data, col_name):
    # Convert to string and remove non-numeric characters
    data[col_name] = data[col_name].astype(str).str.replace('[^0-9.]', '', regex=True)

    # Remove leading/trailing whitespace
    data[col_name] = data[col_name].str.strip()

    # Convert to numeric, handling errors
    try:
        data[col_name] = pd.to_numeric(data[col_name], errors='coerce')
    except ValueError as e:
        print(f"Error converting {col_name}: {e}")
        print(data[col_name].head(20))

    # Handle missing values (e.g., using mean imputation)
    if data[col_name].isnull().sum() > 0:
        imputer_numeric = SimpleImputer(strategy='mean')
        data[numerical_cols] = imputer_numeric.fit_transform(data[numerical_cols])

    return data

# Assuming your DataFrame is 'df'
currency_cols = ['GDP', 'Health Exp/Capita', 'Tourism Inbound', 'Tourism Outbound']

for col in currency_cols:
    data= clean_and_convert(data, col)
# Assuming 'data' is your DataFrame
data['Country'] = data['Country'].astype('object')

# Print the data types after conversion
print(data.dtypes)
print(data.info())


# In[138]:


# Drop columns with excessive missing values
data.drop(['Business Tax Rate','Ease of Business', ], axis=1, inplace=True)


# In[143]:


currency_cols = ['GDP', 'Health Exp/Capita', 'Tourism Inbound', 'Tourism Outbound']

for col in currency_cols:
        imputer_numeric = SimpleImputer(strategy='mean')
        data[currency_cols] = imputer_numeric.fit_transform(data[currency_cols])


# Assuming 'data' is your DataFrame
data['Country'] = data['Country'].astype('object')

    


# In[144]:


data.info()


# In[145]:


def identify_outliers(data, column_name):

    z_scores = np.abs((data[column_name] - data[column_name].mean()) / data[column_name].std())
    outliers = data[z_scores > 3].index
    return outliers

def handle_outliers(data, column_name, method='trimming'):

    if method == 'trimming':
        outliers = identify_outliers(data, column_name)
        data.drop(outliers, inplace=True)  # Trimming outliers directly from 'data'
    elif method == 'capping':
        threshold = data[column_name].quantile(0.95)
        data[column_name] = np.where(data[column_name] > threshold, threshold, data[column_name])
    elif method == 'winsorization':
        lower_percentile = 0.05
        upper_percentile = 0.95
        lower_bound = data[column_name].quantile(lower_percentile)
        upper_bound = data[column_name].quantile(upper_percentile)
        data[column_name] = np.clip(data[column_name], lower_bound, upper_bound)

    return data

# Identify and handle outliers in all numerical columns
numerical_cols = data.select_dtypes(include=np.number).columns



# In[146]:


data_new=data.copy()


# In[147]:


# Scale numerical columns
scaler = StandardScaler()
data_new[numerical_cols] = scaler.fit_transform(data_new[numerical_cols])


# In[148]:


# Encode categorical columns
label_encoder = LabelEncoder()
for col in categorical_cols:
    data_new[col] = label_encoder.fit_transform(data_new[col])


# In[149]:


data_new.head()


# In[158]:
data_new['Country'] = data_new['Country'].astype('object')

data_new.info()


# In[159]:


# PCA
from sklearn.decomposition import PCA
pca = PCA(n_components=2)  # Reduce to 2 principal components
principal_components = pca.fit_transform(data_new)


# In[166]:





# In[169]:


from sklearn.cluster import KMeans

import warnings
i=3
try:
    kmeans = KMeans(n_clusters=i, init='k-means++', random_state=42)
    kmeans.fit(data_new)
    # ... (rest of your code)
except AttributeError:
    warnings.warn("Error with threadpoolctl. Skipping version check.")


# In[170]:


st.title("Global_development_Analysis")


# In[171]:


# Select a country
country = st.selectbox("Select a Country", data['Country'].unique())


# In[172]:


# Filter data for the selected country
country_data = data[data['Country'] == country]


# In[173]:


# Visualize key metrics
import warnings

# To suppress all warnings:
warnings.filterwarnings("ignore")
# Visualize key metrics
fig, ax = plt.subplots(figsize=(10, 6))
sns.lineplot(x=country_data.index, y='GDP', data=country_data, label='GDP')
sns.lineplot(x=country_data.index, y='Tourism Inbound', data=country_data, label='Tourism Inbound')
sns.lineplot(x=country_data.index, y='Tourism Outbound', data=country_data, label='Tourism Outbound')
plt.xlabel('Index')  # Assuming index represents time or another relevant dimension
plt.ylabel('Value')
plt.title(f'Key Metrics for {country}')
st.pyplot(fig)


# In[174]:


# Compare with global trends
global_avg_gdp = data['GDP'].mean()
global_avg_inbound = data['Tourism Inbound'].mean()
global_avg_outbound = data['Tourism Outbound'].mean()


# In[175]:


st.write(f"**Global Averages:**")
st.write(f"- GDP: {global_avg_gdp:.2f}")
st.write(f"- Tourism Inbound: {global_avg_inbound:.2f}")
st.write(f"- Tourism Outbound: {global_avg_outbound:.2f}")


# In[176]:


st.write(f"**{country} Comparison:**")
st.write(f"- GDP: {country_data['GDP'].mean():.2f} (vs. Global Avg: {global_avg_gdp:.2f})")
st.write(f"- Tourism Inbound: {country_data['Tourism Inbound'].mean():.2f} (vs. Global Avg: {global_avg_inbound:.2f})")
st.write(f"- Tourism Outbound: {country_data['Tourism Outbound'].mean():.2f} (vs. Global Avg: {global_avg_outbound:.2f})")


# In[ ]:




# In[ ]:





# In[ ]:




