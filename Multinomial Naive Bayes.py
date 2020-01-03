#!/usr/bin/env python
# coding: utf-8

# In[1]:


# Importing Libraries
# To import necessary libraries for this task, execute the following import statements:

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')


# In[9]:


#Import scikit-learn dataset library
from sklearn import datasets

#Load dataset
wine = datasets.load_wine()


# In[10]:


# Exploring datasets

# print the names of the 13 features
print("Features: ", wine.feature_names)

# print the label type of wine(class_0, class_1, class_2)
print("Labels: ", wine.target_names)


# In[11]:


# print data(feature)shape

wine.data.shape


# In[13]:


# print the wine data features (top 5 records)

print(wine.data[0:5])


# In[14]:


# print the wine labels (0:Class_0, 1:class_2, 2:class_2)

print(wine.target)


# In[18]:


# Import train_test_split function
from sklearn.model_selection import train_test_split

# Split dataset into training set and test set
X_train, X_test, y_train, y_test = train_test_split(wine.data, wine.target, test_size=0.3,random_state=109) # 70% training and 30% test


# In[19]:


# import Gaussian Naive Bayes import GaussianNB
from sklearn.naive_bayes import GaussianNB

#Create a Gaussian Classifier 
gnb = GaussianNB()

# Train the model using the training sets
gnb.fit(X_train, y_train)

# Predict the response for the test dataset
y_pred = gnb.predict(X_test)


# In[20]:


#Import scikit-learn metrics module for accuracy calculation
from sklearn import metrics

# Model Accuracy, how often is the classifier correct?
print("Accuracy:",metrics.accuracy_score(y_test, y_pred))


# In[ ]:




