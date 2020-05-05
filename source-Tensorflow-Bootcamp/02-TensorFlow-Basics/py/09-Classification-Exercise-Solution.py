#!/usr/bin/env python
# coding: utf-8

# # Classification Exercise - Solutions

# We'll be working with some California Census Data, we'll be trying to use various features of an individual to predict what class of income they belogn in (>50k or <=50k). 
# 
# Here is some information about the data:

# <table>
# <thead>
# <tr>
# <th>Column Name</th>
# <th>Type</th>
# <th>Description</th>
# </tr>
# </thead>
# <tbody>
# <tr>
# <td>age</td>
# <td>Continuous</td>
# <td>The age of the individual</td>
# </tr>
# <tr>
# <td>workclass</td>
# <td>Categorical</td>
# <td>The type of employer the  individual has (government,  military, private, etc.).</td>
# </tr>
# <tr>
# <td>fnlwgt</td>
# <td>Continuous</td>
# <td>The number of people the census  takers believe that observation  represents (sample weight). This  variable will not be used.</td>
# </tr>
# <tr>
# <td>education</td>
# <td>Categorical</td>
# <td>The highest level of education  achieved for that individual.</td>
# </tr>
# <tr>
# <td>education_num</td>
# <td>Continuous</td>
# <td>The highest level of education in  numerical form.</td>
# </tr>
# <tr>
# <td>marital_status</td>
# <td>Categorical</td>
# <td>Marital status of the individual.</td>
# </tr>
# <tr>
# <td>occupation</td>
# <td>Categorical</td>
# <td>The occupation of the individual.</td>
# </tr>
# <tr>
# <td>relationship</td>
# <td>Categorical</td>
# <td>Wife, Own-child, Husband,  Not-in-family, Other-relative,  Unmarried.</td>
# </tr>
# <tr>
# <td>race</td>
# <td>Categorical</td>
# <td>White, Asian-Pac-Islander,  Amer-Indian-Eskimo, Other, Black.</td>
# </tr>
# <tr>
# <td>gender</td>
# <td>Categorical</td>
# <td>Female, Male.</td>
# </tr>
# <tr>
# <td>capital_gain</td>
# <td>Continuous</td>
# <td>Capital gains recorded.</td>
# </tr>
# <tr>
# <td>capital_loss</td>
# <td>Continuous</td>
# <td>Capital Losses recorded.</td>
# </tr>
# <tr>
# <td>hours_per_week</td>
# <td>Continuous</td>
# <td>Hours worked per week.</td>
# </tr>
# <tr>
# <td>native_country</td>
# <td>Categorical</td>
# <td>Country of origin of the  individual.</td>
# </tr>
# <tr>
# <td>income</td>
# <td>Categorical</td>
# <td>"&gt;50K" or "&lt;=50K", meaning  whether the person makes more  than \$50,000 annually.</td>
# </tr>
# </tbody>
# </table>

# ## Follow the Directions in Bold. If you get stuck, check out the solutions lecture.

# ### THE DATA

# ** Read in the census_data.csv data with pandas**

# In[3]:


import pandas as pd


# In[4]:


census = pd.read_csv("census_data.csv")


# In[5]:


census.head()


# ** TensorFlow won't be able to understand strings as labels, you'll need to use pandas .apply() method to apply a custom function that converts them to 0s and 1s. This might be hard if you aren't very familiar with pandas, so feel free to take a peek at the solutions for this part.**
# 
# ** Convert the Label column to 0s and 1s instead of strings.**

# In[6]:


census['income_bracket'].unique()


# In[7]:


def label_fix(label):
    if label==' <=50K':
        return 0
    else:
        return 1


# In[8]:


census['income_bracket'] = census['income_bracket'].apply(label_fix)


# In[9]:


# Cool Alternative
# lambda label:int(label==' <=50k')

# census['income_bracket'].apply(lambda label: int(label==' <=50K'))


# ### Perform a Train Test Split on the Data

# In[10]:


from sklearn.model_selection import train_test_split


# In[11]:


x_data = census.drop('income_bracket',axis=1)
y_labels = census['income_bracket']
X_train, X_test, y_train, y_test = train_test_split(x_data,y_labels,test_size=0.3,random_state=101)


# ### Create the Feature Columns for tf.esitmator
# 
# ** Take note of categorical vs continuous values! **

# In[13]:


census.columns


# ** Import Tensorflow **

# In[14]:


import tensorflow as tf


# ** Create the tf.feature_columns for the categorical values. Use vocabulary lists or just use hash buckets. **

# In[15]:


gender = tf.feature_column.categorical_column_with_vocabulary_list("gender", ["Female", "Male"])
occupation = tf.feature_column.categorical_column_with_hash_bucket("occupation", hash_bucket_size=1000)
marital_status = tf.feature_column.categorical_column_with_hash_bucket("marital_status", hash_bucket_size=1000)
relationship = tf.feature_column.categorical_column_with_hash_bucket("relationship", hash_bucket_size=1000)
education = tf.feature_column.categorical_column_with_hash_bucket("education", hash_bucket_size=1000)
workclass = tf.feature_column.categorical_column_with_hash_bucket("workclass", hash_bucket_size=1000)
native_country = tf.feature_column.categorical_column_with_hash_bucket("native_country", hash_bucket_size=1000)


# ** Create the continuous feature_columns for the continuous values using numeric_column **

# In[18]:


age = tf.feature_column.numeric_column("age")
education_num = tf.feature_column.numeric_column("education_num")
capital_gain = tf.feature_column.numeric_column("capital_gain")
capital_loss = tf.feature_column.numeric_column("capital_loss")
hours_per_week = tf.feature_column.numeric_column("hours_per_week")


# ** Put all these variables into a single list with the variable name feat_cols **

# In[19]:


feat_cols = [gender,occupation,marital_status,relationship,education,workclass,native_country,
            age,education_num,capital_gain,capital_loss,hours_per_week]


# ### Create Input Function
# 
# ** Batch_size is up to you. But do make sure to shuffle!**

# In[20]:


input_func=tf.estimator.inputs.pandas_input_fn(x=X_train,y=y_train,batch_size=100,num_epochs=None,shuffle=True)


# #### Create your model with tf.estimator
# 
# **Create a LinearClassifier.(If you want to use a DNNClassifier, keep in mind you'll need to create embedded columns out of the cateogrical feature that use strings, check out the previous lecture on this for more info.)**

# In[21]:


model = tf.estimator.LinearClassifier(feature_columns=feat_cols)


# ** Train your model on the data, for at least 5000 steps. **

# In[22]:


model.train(input_fn=input_func,steps=5000)


# ### Evaluation
# 
# ** Create a prediction input function. Remember to only supprt X_test data and keep shuffle=False. **

# In[23]:


pred_fn = tf.estimator.inputs.pandas_input_fn(x=X_test,batch_size=len(X_test),shuffle=False)


# ** Use model.predict() and pass in your input function. This will produce a generator of predictions, which you can then transform into a list, with list() **

# In[25]:


predictions = list(model.predict(input_fn=pred_fn))


# ** Each item in your list will look like this: **

# In[27]:


predictions[0]


# ** Create a list of only the class_ids key values from the prediction list of dictionaries, these are the predictions you will use to compare against the real y_test values. **

# In[28]:


final_preds = []
for pred in predictions:
    final_preds.append(pred['class_ids'][0])


# In[30]:


final_preds[:10]


# ** Import classification_report from sklearn.metrics and then see if you can figure out how to use it to easily get a full report of your model's performance on the test data. **

# In[31]:


from sklearn.metrics import classification_report


# In[32]:


print(classification_report(y_test,final_preds))


# # Great Job!
