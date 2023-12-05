#!/usr/bin/env python
# coding: utf-8

# In[7]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')

from pandas.plotting import scatter_matrix
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn import metrics
import warnings
warnings.filterwarnings('ignore')


# In[8]:


df = pd.read_csv('https://raw.githubusercontent.com/dsrscientist/Data-Science-ML-Capstone-Projects/master/avocado.csv')
df


# In[115]:


df.head()


# In[116]:


df.tail()


# In[117]:


df.info()


# In[118]:


df.shape


# In[119]:


df.isnull().sum()


# In[120]:


sns.heatmap(df.isnull())


# In[121]:


df.columns.tolist()


# In[122]:


sns.set()
plt.figure.figsize=(6,6)
sns.distplot(df['AveragePrice'])
plt.title('Average Price')
plt.show


# In[123]:


new_df= df.fillna(method="ffill")
new_df


# In[124]:


new_df.isnull().sum()


# In[125]:


sns.heatmap(new_df.isnull())


# In[126]:


sns.set()
plt.figure.figsize=(6,6)
sns.distplot(new_df['AveragePrice'])
plt.title('Average Price')
plt.show


# In[127]:


new_df['type'].value_counts()


# In[128]:


plt.figure(figsize=(3,3))
sns.countplot(x='type',data= new_df)
plt.title('type')
plt.show()


# In[129]:


new_df['year'].value_counts()


# In[130]:


plt.figure(figsize=(4,4))
sns.countplot(x='year',data= new_df)
plt.title('Year')
plt.show()


# In[131]:


new_df.replace({'year':{'2015.0':1,'2016.0':0}},inplace=True)
new_df.replace({'type':{'conventional':1}},inplace=True)


# In[132]:


X=new_df.drop(columns='AveragePrice',axis=1)
Y=new_df['AveragePrice']


# In[133]:


print(X)


# In[134]:


print(Y)


# In[135]:


X_train, X_test, Y_train, Y_test = train_test_split(X,Y,test_size=0.2,random_state=2)


# In[136]:


print(X.shape, X_train.shape, X_test.shape)


# In[137]:


regressor=LinearRegression()


# In[138]:


regressor.fit(X_train, Y_train)


# In[ ]:




