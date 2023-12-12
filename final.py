#!/usr/bin/env python
# coding: utf-8

# In[33]:


import streamlit as st
import pandas as pd
import numpy as np
from sklearn.metrics import accuracy_score, roc_auc_score, auc
import pickle
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_curve, auc


# In[15]:


#Load pickle model:
with open('RF_hypa.pkl','rb') as file:
    clf = pickle.load(file)


# In[16]:


X_train = clf['X_train']
X_test = clf['X_test']
y_train = clf['y_train']
y_test = clf['y_test']


# In[17]:


st.write("""

# Hyperparameters Viz. for Random Forest Classifier

This app helps to visualization the effects of **Hyperparameters** on the RF model!

""")


# In[18]:


st.sidebar.header('User Input Parameters')


# In[19]:


st.sidebar.header('Hyperparameters')
n_estimators = st.sidebar.slider('Number of Trees (n_estimators)', min_value=1, max_value=500, value=25,step=1)
max_depth = st.sidebar.slider('Max Depth', min_value=1, max_value=50, value=5,step=1)
criterion = st.sidebar.selectbox('Criterion',['gini','entropy'],index=0)
max_features = st.sidebar.selectbox('Max Features', ['sqrt','log2',None],index=0)
max_samples = st.sidebar.slider('Max Samples', min_value=1, max_value=300, value=5,step=1)
bootstrap = st.sidebar.checkbox('Bootstrap', value=True)
oob_score = st.sidebar.checkbox('Out-of-Bag Score', value=False)
verbose = st.sidebar.selectbox('Verbose', [0, 1, 2], index=1)


# In[20]:


rf = RandomForestClassifier(
    n_estimators=n_estimators,
    max_depth=max_depth,
    criterion=criterion,
    max_features = max_features,
    max_samples = max_samples,
    bootstrap=bootstrap,
    oob_score=oob_score,
    verbose=verbose,
    random_state=42 
)

# In[26]:
rf.fit(X_train,y_train)

y_prob = rf.predict_proba(X_test)[:, 1]

y_pred = (y_prob > 0.5).astype(int)

accuracy = accuracy_score(y_test, (y_prob > 0.5).astype(int))
# accuracy = accuracy_score(y_test,y_prob)

st.write(f'Accuracy: {accuracy:.2f}')


# Calculating the ROC curve
fpr, tpr, thresholds = roc_curve(y_test, y_prob)


# In[28]:


roc_auc = auc(fpr, tpr)


# In[32]:


fig, ax = plt.subplots(figsize=(8, 6))
ax.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (AUC = {roc_auc:.2f})')
ax.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--', label='Random')
ax.set_xlabel('False Positive Rate')
ax.set_ylabel('True Positive Rate')
ax.set_title('Receiver Operating Characteristic (ROC) Curve')
ax.legend(loc='lower right')

# Display the plot in Streamlit
st.pyplot(fig)


# In[ ]:




