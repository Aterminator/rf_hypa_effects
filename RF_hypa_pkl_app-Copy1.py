#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import pickle
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder


# In[3]:streamli


df = pd.read_csv('insurance_data-Copy1.csv')


# In[4]:


df.shape


# In[5]:


df.head()


# In[6]:


df.drop(columns=['index','PatientID'],axis=1,inplace=True)


# In[7]:


df.isnull().sum()


# In[8]:


df.dropna(axis=0, how='any', inplace=True)


# In[9]:


df.isnull().sum()


# In[10]:


df['smoker'].replace({'Yes':1,'No':0},inplace=True)


# In[11]:


X = df.drop(columns=['smoker'],axis=1)
y = df.smoker


# In[12]:


X.shape,y.shape


# In[13]:


X.head()


# In[14]:


X_num = X.select_dtypes(include='number')
X_cat = X.select_dtypes(include='object')


# In[15]:


scaler = MinMaxScaler()
X_num_scaled = scaler.fit_transform(X_num)


# In[16]:


X_num_scaled = pd.DataFrame(X_num_scaled, columns=X_num.columns, index=X_num.index)
X_num_scaled.head()


# In[17]:


X_cat_encoded = pd.get_dummies(X_cat, drop_first=False, dtype=int)
X_cat_encoded.head()


# In[18]:


X = pd.concat([X_num_scaled, X_cat_encoded], axis=1)


# In[19]:


X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.2,random_state=42)


# In[20]:


split_data = {
    'X_train': X_train,
    'X_test': X_test,
    'y_train': y_train,
    'y_test': y_test
}


# In[21]:


with open('RF_hypa.pkl', 'wb') as file:
    pickle.dump(split_data, file)


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:




