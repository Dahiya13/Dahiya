#!/usr/bin/env python
# coding: utf-8

# In[3]:


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


# In[4]:


df = pd.read_csv('https://raw.githubusercontent.com/dsrscientist/Data-Science-ML-Capstone-Projects/master/baseball.csv')
df


# In[5]:


df.head()


# In[6]:


df.tail()


# In[7]:


df.shape


# In[8]:


df.info()


# In[9]:


df.isnull().sum()


# In[10]:


sns.heatmap(df.isnull())


# In[11]:


df.dtypes


# In[12]:


sns.set()
plt.figure.figsize=(6,6)
sns.distplot(df['W'])
plt.title('WINS')
plt.show


# In[13]:


sns.set()
plt.figure.figsize=(6,6)
sns.distplot(df['R'])
plt.title('RUNS')
plt.show


# In[14]:


sns.set()
plt.figure.figsize=(6,6)
sns.distplot(df['RA'])
plt.title('RUNS AVERAGE')
plt.show


# In[15]:


sns.set()
plt.figure.figsize=(6,6)
sns.distplot(df['ER'])
plt.title('EARNED RUN')
plt.show


# In[16]:


sns.set()
plt.figure.figsize=(6,6)
sns.distplot(df['ERA'])
plt.title('ER AVERAGE')
plt.show


# In[17]:


df['W'].value_counts()


# In[18]:


plt.figure(figsize=(6,6))
sns.countplot(x='W',data= df)
plt.title('WINS')
plt.show()


# In[19]:


df['R'].value_counts()


# In[20]:


plt.figure(figsize=(6,6))
sns.countplot(x='R',data= df)
plt.title('RUNS')
plt.show()


# In[21]:


df['ER'].value_counts()


# In[25]:


plt.figure(figsize=(12,6))
sns.countplot(x='ER',data= df)
plt.title('EARNED RUNS')
plt.show()


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:




