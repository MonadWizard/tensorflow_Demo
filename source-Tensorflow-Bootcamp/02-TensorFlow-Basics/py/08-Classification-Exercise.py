#!/usr/bin/env python
# coding: utf-8

# # Classification Exercise

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





# In[4]:





# In[5]:





# ** TensorFlow won't be able to understand strings as labels, you'll need to use pandas .apply() method to apply a custom function that converts them to 0s and 1s. This might be hard if you aren't very familiar with pandas, so feel free to take a peek at the solutions for this part.**
# 
# ** Convert the Label column to 0s and 1s instead of strings.**

# In[6]:





# In[7]:





# In[8]:





# In[9]:





# ### Perform a Train Test Split on the Data

# In[10]:





# In[11]:





# ### Create the Feature Columns for tf.esitmator
# 
# ** Take note of categorical vs continuous values! **

# In[13]:





# ** Import Tensorflow **

# In[14]:





# ** Create the tf.feature_columns for the categorical values. Use vocabulary lists or just use hash buckets. **

# In[15]:





# ** Create the continuous feature_columns for the continuous values using numeric_column **

# In[18]:





# ** Put all these variables into a single list with the variable name feat_cols **

# In[19]:





# ### Create Input Function
# 
# ** Batch_size is up to you. But do make sure to shuffle!**

# In[20]:





# #### Create your model with tf.estimator
# 
# **Create a LinearClassifier.(If you want to use a DNNClassifier, keep in mind you'll need to create embedded columns out of the cateogrical feature that use strings, check out the previous lecture on this for more info.)**

# In[21]:





# ** Train your model on the data, for at least 5000 steps. **

# In[22]:





# ### Evaluation
# 
# ** Create a prediction input function. Remember to only supprt X_test data and keep shuffle=False. **

# In[23]:





# ** Use model.predict() and pass in your input function. This will produce a generator of predictions, which you can then transform into a list, with list() **

# In[25]:





# ** Each item in your list will look like this: **

# In[27]:





# ** Create a list of only the class_ids key values from the prediction list of dictionaries, these are the predictions you will use to compare against the real y_test values. **

# In[28]:





# In[30]:





# ** Import classification_report from sklearn.metrics and then see if you can figure out how to use it to easily get a full report of your model's performance on the test data. **

# In[31]:





# In[32]:





# # Great Job!
