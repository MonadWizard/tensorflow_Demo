#!/usr/bin/env python
# coding: utf-8

# # Regression Exercise - Solutions
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

# ** Import the cal_housing.csv file with pandas. Separate it into a training (70%) and testing set(30%).**

# In[125]:


import pandas as pd


# In[126]:


housing = pd.read_csv('cal_housing_clean.csv')


# In[127]:


housing.head()


# In[128]:


housing.describe().transpose()


# In[129]:


x_data = housing.drop(['medianHouseValue'],axis=1)


# In[130]:


y_val = housing['medianHouseValue']


# In[131]:


from sklearn.model_selection import train_test_split


# In[132]:


X_train, X_test, y_train, y_test = train_test_split(x_data,y_val,test_size=0.3,random_state=101)


# ### Scale the Feature Data
# 
# ** Use sklearn preprocessing to create a MinMaxScaler for the feature data. Fit this scaler only to the training data. Then use it to transform X_test and X_train. Then use the scaled X_test and X_train along with pd.Dataframe to re-create two dataframes of scaled data.**

# In[133]:


from sklearn.preprocessing import MinMaxScaler


# In[134]:


scaler = MinMaxScaler()


# In[135]:


scaler.fit(X_train)


# In[136]:


X_train = pd.DataFrame(data=scaler.transform(X_train),columns = X_train.columns,index=X_train.index)


# In[137]:


X_test = pd.DataFrame(data=scaler.transform(X_test),columns = X_test.columns,index=X_test.index)


# ### Create Feature Columns
# 
# ** Create the necessary tf.feature_column objects for the estimator. They should all be trated as continuous numeric_columns. **

# In[138]:


housing.columns


# In[139]:


import tensorflow as tf


# In[140]:


age = tf.feature_column.numeric_column('housingMedianAge')
rooms = tf.feature_column.numeric_column('totalRooms')
bedrooms = tf.feature_column.numeric_column('totalBedrooms')
pop = tf.feature_column.numeric_column('population')
households = tf.feature_column.numeric_column('households')
income = tf.feature_column.numeric_column('medianIncome')


# In[141]:


feat_cols = [ age,rooms,bedrooms,pop,households,income]


# ** Create the input function for the estimator object. (play around with batch_size and num_epochs)**

# In[142]:


input_func = tf.estimator.inputs.pandas_input_fn(x=X_train,y=y_train ,batch_size=10,num_epochs=1000,
                                            shuffle=True)


# ** Create the estimator model. Use a DNNRegressor. Play around with the hidden units! **

# In[143]:


model = tf.estimator.DNNRegressor(hidden_units=[6,6,6],feature_columns=feat_cols)


# ##### ** Train the model for ~1,000 steps. (Later come back to this and train it for more and check for improvement) **

# In[144]:


model.train(input_fn=input_func,steps=25000)


# ** Create a prediction input function and then use the .predict method off your estimator model to create a list or predictions on your test data. **

# In[153]:


predict_input_func = tf.estimator.inputs.pandas_input_fn(
      x=X_test,
      batch_size=10,
      num_epochs=1,
      shuffle=False)


# In[154]:


pred_gen = model.predict(predict_input_func)


# In[155]:


predictions = list(pred_gen)


# ** Calculate the RMSE. Do this manually or use [sklearn.metrics](http://scikit-learn.org/stable/modules/generated/sklearn.metrics.mean_squared_error.html) **

# In[156]:


final_preds = []
for pred in predictions:
    final_preds.append(pred['predictions'])


# In[157]:


from sklearn.metrics import mean_squared_error


# In[158]:


mean_squared_error(y_test,final_preds)**0.5


# # Great Job!
