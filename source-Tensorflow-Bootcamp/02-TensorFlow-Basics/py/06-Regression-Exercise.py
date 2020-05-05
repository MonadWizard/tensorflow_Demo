#!/usr/bin/env python
# coding: utf-8

# # Regression Exercise 
# 
# California Housing Data
# 
# This data set contains information about all the block groups in California from the 1990 Census. In this sample a block group on average includes 1425.5 individuals living in a geographically compact area. 
# 
# The task is to aproximate the median house value of each block from the values of the rest of the variables. 
# 
#  It has been obtained from the LIACC repository. The original page where the data set can be found is: http://www.liaad.up.pt/~ltorgo/Regression/DataSets.html.
#  

# The Features:
#  
# * housingMedianAge: continuous. 
# * totalRooms: continuous. 
# * totalBedrooms: continuous. 
# * population: continuous. 
# * households: continuous. 
# * medianIncome: continuous. 
# * medianHouseValue: continuous. 

# ## The Data

# ** Import the cal_housing_clean.csv file with pandas. Separate it into a training (70%) and testing set(30%).**

# In[125]:





# In[126]:





# In[127]:





# In[128]:





# In[129]:





# In[130]:





# In[131]:





# In[132]:





# ### Scale the Feature Data
# 
# ** Use sklearn preprocessing to create a MinMaxScaler for the feature data. Fit this scaler only to the training data. Then use it to transform X_test and X_train. Then use the scaled X_test and X_train along with pd.Dataframe to re-create two dataframes of scaled data.**

# In[133]:





# In[134]:





# In[135]:





# In[136]:





# In[137]:





# ### Create Feature Columns
# 
# ** Create the necessary tf.feature_column objects for the estimator. They should all be trated as continuous numeric_columns. **

# In[138]:





# In[139]:





# In[140]:





# In[141]:





# ** Create the input function for the estimator object. (play around with batch_size and num_epochs)**

# In[142]:





# ** Create the estimator model. Use a DNNRegressor. Play around with the hidden units! **

# In[143]:





# ##### ** Train the model for ~1,000 steps. (Later come back to this and train it for more and check for improvement) **

# In[144]:





# ** Create a prediction input function and then use the .predict method off your estimator model to create a list or predictions on your test data. **

# In[153]:





# In[154]:





# In[155]:





# ** Calculate the RMSE. You should be able to get around 100,000 RMSE (remember that this is in the same units as the label.) Do this manually or use [sklearn.metrics](http://scikit-learn.org/stable/modules/generated/sklearn.metrics.mean_squared_error.html) **

# In[156]:





# In[157]:





# In[158]:





# # Great Job!
