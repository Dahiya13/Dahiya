#!/usr/bin/env python
# coding: utf-8

# In[1]:


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


# In[2]:


df = pd.read_csv('https://raw.githubusercontent.com/shantanu1109/IBM-HR-Analytics-Employee-Attrition-and-Performance-Prediction/main/DATASET/IBM-HR-Analytics-Employee-Attrition-and-Performance-Revised.csv')
df


# In[3]:


df.shape


# In[4]:


df.info()


# In[5]:


df.head()


# In[6]:


df.tail()


# In[7]:


df.isnull().sum()


# In[8]:


sns.heatmap(df.isnull())


# In[9]:


sns.set()
plt.figure.figsize=(6,6)
sns.distplot(df['Age'])
plt.title('Age')
plt.show


# In[12]:


sns.set()
plt.figure.figsize=(6,6)
sns.distplot(df['HourlyRate'])
plt.title('Daily Rate')
plt.show


# In[14]:


sns.set()
plt.figure.figsize=(6,6)
sns.distplot(df['MonthlyIncome'])
plt.title('Monthly Income')
plt.show


# In[15]:


sns.set()
plt.figure.figsize=(6,6)
sns.distplot(df['MonthlyRate'])
plt.title('Monthly Rate')
plt.show


# In[16]:


sns.set()
plt.figure.figsize=(6,6)
sns.distplot(df['PercentSalaryHike'])
plt.title('Percent Salary Hike')
plt.show


# In[18]:


sns.set()
plt.figure.figsize=(6,6)
sns.distplot(df['YearsAtCompany'])
plt.title('Years At Company')
plt.show


# In[21]:


df['Attrition'].value_counts()


# In[23]:


plt.figure(figsize=(6,6))
sns.countplot(x='Attrition',data= df)
plt.title('Attrition')
plt.show()


# In[24]:


df['BusinessTravel'].value_counts()


# In[25]:


plt.figure(figsize=(6,6))
sns.countplot(x='BusinessTravel',data= df)
plt.title('BusinessTravel')
plt.show()


# In[26]:


df['Department'].value_counts()


# In[27]:


plt.figure(figsize=(6,6))
sns.countplot(x='Department',data= df)
plt.title('Department')
plt.show()


# In[28]:


df['Education'].value_counts()


# In[33]:


plt.figure(figsize=(6,8))
sns.countplot(x='Education',data= df)
plt.title('Education')
plt.show()


# In[30]:


df['EducationField'].value_counts()


# In[31]:


plt.figure(figsize=(6,6))
sns.countplot(x='EducationField',data= df)
plt.title('Education Field')
plt.show()


# In[37]:


df['EnvironmentSatisfaction'].value_counts()


# In[38]:


plt.figure(figsize=(6,6))
sns.countplot(x='EnvironmentSatisfaction',data= df)
plt.title('Environment Satisfaction')
plt.show()


# In[39]:


df['Gender'].value_counts()


# In[40]:


plt.figure(figsize=(6,6))
sns.countplot(x='Gender',data= df)
plt.title('Gender')
plt.show()


# In[41]:


df['PerformanceRating'].value_counts()


# In[42]:


plt.figure(figsize=(6,6))
sns.countplot(x='PerformanceRating',data= df)
plt.title('PerformanceRating')
plt.show()


# In[43]:


df['RelationshipSatisfaction'].value_counts()


# In[45]:


plt.figure(figsize=(6,6))
sns.countplot(x='RelationshipSatisfaction',data=df)
plt.title('Relationship Satisfaction')
plt.show()


# In[46]:


sns.set()
plt.figure.figsize=(6,6)
sns.distplot(df['YearsInCurrentRole'])
plt.title('Years In Current Role')
plt.show


# In[48]:


sns.set()
plt.figure.figsize=(6,6)
sns.distplot(df['YearsAtCompany'])
plt.title('Years At Company')
plt.show


# In[49]:


sns.set()
plt.figure.figsize=(6,6)
sns.distplot(df['YearsSinceLastPromotion'])
plt.title('Years Since Last Promotion')
plt.show


# In[50]:


sns.set()
plt.figure.figsize=(6,6)
sns.distplot(df['YearsWithCurrManager'])
plt.title('Years With Current Manager')
plt.show


# In[ ]:




