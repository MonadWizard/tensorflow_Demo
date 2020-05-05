#!/usr/bin/env python
# coding: utf-8

# # TensorFlow Classification

# ## Data
# 
# https://archive.ics.uci.edu/ml/datasets/pima+indians+diabetes
# 
# 1. Title: Pima Indians Diabetes Database
# 
# 2. Sources:
#    (a) Original owners: National Institute of Diabetes and Digestive and
#                         Kidney Diseases
#    (b) Donor of database: Vincent Sigillito (vgs@aplcen.apl.jhu.edu)
#                           Research Center, RMI Group Leader
#                           Applied Physics Laboratory
#                           The Johns Hopkins University
#                           Johns Hopkins Road
#                           Laurel, MD 20707
#                           (301) 953-6231
#    (c) Date received: 9 May 1990
# 
# 3. Past Usage:
#     1. Smith,~J.~W., Everhart,~J.~E., Dickson,~W.~C., Knowler,~W.~C., \&
#        Johannes,~R.~S. (1988). Using the ADAP learning algorithm to forecast
#        the onset of diabetes mellitus.  In {\it Proceedings of the Symposium
#        on Computer Applications and Medical Care} (pp. 261--265).  IEEE
#        Computer Society Press.
# 
#        The diagnostic, binary-valued variable investigated is whether the
#        patient shows signs of diabetes according to World Health Organization
#        criteria (i.e., if the 2 hour post-load plasma glucose was at least 
#        200 mg/dl at any survey  examination or if found during routine medical
#        care).   The population lives near Phoenix, Arizona, USA.
# 
#        Results: Their ADAP algorithm makes a real-valued prediction between
#        0 and 1.  This was transformed into a binary decision using a cutoff of 
#        0.448.  Using 576 training instances, the sensitivity and specificity
#        of their algorithm was 76% on the remaining 192 instances.
# 
# 4. Relevant Information:
#       Several constraints were placed on the selection of these instances from
#       a larger database.  In particular, all patients here are females at
#       least 21 years old of Pima Indian heritage.  ADAP is an adaptive learning
#       routine that generates and executes digital analogs of perceptron-like
#       devices.  It is a unique algorithm; see the paper for details.
# 
# 5. Number of Instances: 768
# 
# 6. Number of Attributes: 8 plus class 
# 
#     7. For Each Attribute: (all numeric-valued)
#        1. Number of times pregnant
#        2. Plasma glucose concentration a 2 hours in an oral glucose tolerance test
#        3. Diastolic blood pressure (mm Hg)
#        4. Triceps skin fold thickness (mm)
#        5. 2-Hour serum insulin (mu U/ml)
#        6. Body mass index (weight in kg/(height in m)^2)
#        7. Diabetes pedigree function
#        8. Age (years)
#        9. Class variable (0 or 1)
# 
# 8. Missing Attribute Values: Yes
# 
# 9. Class Distribution: (class value 1 is interpreted as "tested positive for
#    diabetes")
# 
#    Class Value  Number of instances
#    0            500
#    1            268
# 
# 10. Brief statistical analysis:
# 
#         Attribute number:    Mean:   Standard Deviation:
#         1.                     3.8     3.4
#         2.                   120.9    32.0
#         3.                    69.1    19.4
#         4.                    20.5    16.0
#         5.                    79.8   115.2
#         6.                    32.0     7.9
#         7.                     0.5     0.3
#         8.                    33.2    11.8

# In[202]:


import pandas as pd


# In[203]:


diabetes = pd.read_csv('pima-indians-diabetes.csv')


# In[204]:


diabetes.head()


# In[205]:


diabetes.columns


# ### Clean the Data

# In[206]:


cols_to_norm = ['Number_pregnant', 'Glucose_concentration', 'Blood_pressure', 'Triceps',
       'Insulin', 'BMI', 'Pedigree']


# In[207]:


diabetes[cols_to_norm] = diabetes[cols_to_norm].apply(lambda x: (x - x.min()) / (x.max() - x.min()))


# In[250]:


diabetes.head()


# ### Feature Columns

# In[209]:


diabetes.columns 


# In[210]:


import tensorflow as tf


# ### Continuous Features
# 
# * Number of times pregnant
# * Plasma glucose concentration a 2 hours in an oral glucose tolerance test
# * Diastolic blood pressure (mm Hg)
# * Triceps skin fold thickness (mm)
# * 2-Hour serum insulin (mu U/ml)
# * Body mass index (weight in kg/(height in m)^2)
# * Diabetes pedigree function

# In[211]:


num_preg = tf.feature_column.numeric_column('Number_pregnant')
plasma_gluc = tf.feature_column.numeric_column('Glucose_concentration')
dias_press = tf.feature_column.numeric_column('Blood_pressure')
tricep = tf.feature_column.numeric_column('Triceps')
insulin = tf.feature_column.numeric_column('Insulin')
bmi = tf.feature_column.numeric_column('BMI')
diabetes_pedigree = tf.feature_column.numeric_column('Pedigree')
age = tf.feature_column.numeric_column('Age')


# ### Categorical Features

# If you know the set of all possible feature values of a column and there are only a few of them, you can use categorical_column_with_vocabulary_list. If you don't know the set of possible values in advance you can use categorical_column_with_hash_bucket

# In[212]:


assigned_group = tf.feature_column.categorical_column_with_vocabulary_list('Group',['A','B','C','D'])
# Alternative
# assigned_group = tf.feature_column.categorical_column_with_hash_bucket('Group', hash_bucket_size=10)


# ### Converting Continuous to Categorical

# In[213]:


import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')


# In[214]:


diabetes['Age'].hist(bins=20)


# In[215]:


age_buckets = tf.feature_column.bucketized_column(age, boundaries=[20,30,40,50,60,70,80])


# ### Putting them together

# In[226]:


feat_cols = [num_preg ,plasma_gluc,dias_press ,tricep ,insulin,bmi,diabetes_pedigree ,assigned_group, age_buckets]


# ### Train Test Split

# In[227]:


diabetes.head()


# In[252]:


diabetes.info()


# In[254]:


x_data = diabetes.drop('Class',axis=1)


# In[255]:


labels = diabetes['Class']


# In[253]:


from sklearn.model_selection import train_test_split


# In[258]:


X_train, X_test, y_train, y_test = train_test_split(x_data,labels,test_size=0.33, random_state=101)


# ### Input Function

# In[259]:


input_func = tf.estimator.inputs.pandas_input_fn(x=X_train,y=y_train,batch_size=10,num_epochs=1000,shuffle=True)


# ### Creating the Model

# In[260]:


model = tf.estimator.LinearClassifier(feature_columns=feat_cols,n_classes=2)


# In[261]:


model.train(input_fn=input_func,steps=1000)


# In[262]:


# Useful link ofr your own data
# https://stackoverflow.com/questions/44664285/what-are-the-contraints-for-tensorflow-scope-names


# ## Evaluation

# In[289]:


eval_input_func = tf.estimator.inputs.pandas_input_fn(
      x=X_test,
      y=y_test,
      batch_size=10,
      num_epochs=1,
      shuffle=False)


# In[281]:


results = model.evaluate(eval_input_func)


# In[290]:


results


# ## Predictions

# In[293]:


pred_input_func = tf.estimator.inputs.pandas_input_fn(
      x=X_test,
      batch_size=10,
      num_epochs=1,
      shuffle=False)


# In[304]:


# Predictions is a generator! 
predictions = model.predict(pred_input_func)


# In[305]:


list(predictions)


# # DNN Classifier

# In[307]:


dnn_model = tf.estimator.DNNClassifier(hidden_units=[10,10,10],feature_columns=feat_cols,n_classes=2)


# In[309]:


# UH OH! AN ERROR. Check out the video to see why and how to fix.
# https://github.com/tensorflow/tensorflow/blob/master/tensorflow/python/feature_column/feature_column.py
dnn_model.train(input_fn=input_func,steps=1000)


# In[316]:


embedded_group_column = tf.feature_column.embedding_column(assigned_group, dimension=4)


# In[317]:


feat_cols = [num_preg ,plasma_gluc,dias_press ,tricep ,insulin,bmi,diabetes_pedigree ,embedded_group_column, age_buckets]


# In[318]:


input_func = tf.estimator.inputs.pandas_input_fn(x=X_train,y=y_train,batch_size=10,num_epochs=1000,shuffle=True)


# In[320]:


dnn_model = tf.estimator.DNNClassifier(hidden_units=[10,10,10],feature_columns=feat_cols,n_classes=2)


# In[323]:


dnn_model.train(input_fn=input_func,steps=1000)


# In[324]:


eval_input_func = tf.estimator.inputs.pandas_input_fn(
      x=X_test,
      y=y_test,
      batch_size=10,
      num_epochs=1,
      shuffle=False)


# In[325]:


dnn_model.evaluate(eval_input_func)


# # Great Job!
