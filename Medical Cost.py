#!/usr/bin/env python
# coding: utf-8

# In[7]:


import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn import metrics
import csv
import warnings
warnings.filterwarnings('ignore')


# In[10]:


Medical_cost=pd.read_csv('https://raw.githubusercontent.com/dsrscientist/dataset4/main/medical_cost_insurance.csv')
Medical_cost


# In[12]:


Medical_cost.head


# In[13]:


Medical_cost.shape


# In[14]:


Medical_cost.columns.tolist()


# In[15]:


Medical_cost.dtypes


# In[16]:


Medical_cost.isnull().sum().sum()


# In[17]:


Medical_cost.info()


# In[18]:


sns.heatmap(Medical_cost.isnull())


# In[19]:


sns.set()
plt.figure.figsize=(6,6)
sns.distplot(Medical_cost['age'])
plt.title('Age Distribution')
plt.show


# In[21]:


Medical_cost['sex'].value_counts()


# In[20]:


plt.figure(figsize=(6,6))
sns.countplot(x='sex',data= Medical_cost)
plt.title('Sex Distribution')
plt.show()


# In[24]:


plt.figure(figsize=(6,6))
sns.distplot(Medical_cost['bmi'])
plt.title('BMI Distribution')
plt.show()


# In[25]:


plt.figure(figsize=(6,6))
sns.countplot(x='children',data= Medical_cost)
plt.title('Children')
plt.show()


# In[26]:


Medical_cost['children'].value_counts()


# In[27]:


Medical_cost['smoker'].value_counts()


# In[28]:


plt.figure(figsize=(6,6))
sns.countplot(x='smoker',data= Medical_cost)
plt.title('Smoker')
plt.show()


# In[29]:


Medical_cost['region'].value_counts()


# In[30]:


plt.figure(figsize=(6,6))
sns.countplot(x='region',data= Medical_cost)
plt.title('Region')
plt.show()


# In[32]:


plt.figure(figsize=(6,6))
sns.distplot(Medical_cost['charges'])
plt.title('Charges Distribution')
plt.show()


# In[36]:


Medical_cost.replace({'sex':{'male':1,'female':0}},inplace=True)
Medical_cost.replace({'smoker':{'yes':1,'no':0}},inplace=True)


# In[38]:


Medical_cost.replace({'region':{'southeast':0,'southwest':1,'northeast':2,'northwest':3}},inplace=True)


# In[39]:


X=Medical_cost.drop(columns='charges',axis=1)
Y=Medical_cost['charges']


# In[40]:


print(X)


# In[41]:


print(Y)


# In[42]:


X_train, X_test, Y_train, Y_test = train_test_split(X,Y,test_size=0.2,random_state=2)


# In[43]:


print(X.shape, X_train.shape, X_test.shape)


# In[45]:


regressor=LinearRegression()


# In[46]:


regressor.fit(X_train, Y_train)


# In[47]:


trainingprediction=regressor.predict(X_train)


# In[49]:


rtrain= metrics.r2_score(Y_train, trainingprediction)
print('R squared vale:',rtrain)


# In[50]:


testprediction=regressor.predict(X_test)


# In[51]:


rtest= metrics.r2_score(Y_test, testprediction)
print('R squared vale:',rtest)


# In[58]:


inputdata=[19,1,27.990,1,1,1]


# In[56]:


inputarray=np.asarray(inputdata)
inputreshaped=inputarray.reshape(1,-1)
prediction=regressor.predict(inputreshaped)
print(prediction)
print('The insurance cost is',prediction[0])


# In[ ]:




