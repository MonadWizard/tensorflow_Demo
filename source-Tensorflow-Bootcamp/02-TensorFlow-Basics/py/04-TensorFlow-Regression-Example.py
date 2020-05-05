#!/usr/bin/env python
# coding: utf-8

# # TensorFlow Regression Example

# ## Creating Data

# In[1]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')


# In[2]:


# 1 Million Points
x_data = np.linspace(0.0,10.0,1000000)


# In[3]:


noise = np.random.randn(len(x_data))


# In[4]:


# y = mx + b + noise_levels
b = 5

y_true =  (0.5 * x_data ) + 5 + noise


# In[5]:


my_data = pd.concat([pd.DataFrame(data=x_data,columns=['X Data']),pd.DataFrame(data=y_true,columns=['Y'])],axis=1)


# In[6]:


my_data.head()


# In[7]:


my_data.sample(n=250).plot(kind='scatter',x='X Data',y='Y')


# # TensorFlow
# ## Batch Size
# 
# We will take the data in batches (1,000,000 points is a lot to pass in at once)

# In[8]:


import tensorflow as tf


# In[9]:


# Random 10 points to grab
batch_size = 8


# ** Variables **

# In[10]:


m = tf.Variable(0.5)
b = tf.Variable(1.0)


# ** Placeholders **

# In[11]:


xph = tf.placeholder(tf.float32,[batch_size])
yph = tf.placeholder(tf.float32,[batch_size])


# ** Graph **

# In[12]:


y_model = m*xph + b


# ** Loss Function **

# In[13]:


error = tf.reduce_sum(tf.square(yph-y_model))


# ** Optimizer **

# In[14]:


optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.001)
train = optimizer.minimize(error)


# ** Initialize Variables **

# In[15]:


init = tf.global_variables_initializer()


# ### Session

# In[16]:


with tf.Session() as sess:
    
    sess.run(init)
    
    batches = 1000
    
    for i in range(batches):
        
        rand_ind = np.random.randint(len(x_data),size=batch_size)
        
        feed = {xph:x_data[rand_ind],yph:y_true[rand_ind]}
        
        sess.run(train,feed_dict=feed)
        
    model_m,model_b = sess.run([m,b])


# In[17]:


model_m


# In[18]:


model_b


# ### Results

# In[19]:


y_hat = x_data * model_m + model_b


# In[20]:


my_data.sample(n=250).plot(kind='scatter',x='X Data',y='Y')
plt.plot(x_data,y_hat,'r')


# ## tf.estimator API
# 
# Much simpler API for basic tasks like regression! We'll talk about more abstractions like TF-Slim later on.

# In[32]:


feat_cols = [tf.feature_column.numeric_column('x',shape=[1])]


# In[33]:


estimator = tf.estimator.LinearRegressor(feature_columns=feat_cols)


# ### Train Test Split
# 
# We haven't actually performed a train test split yet! So let's do that on our data now and perform a more realistic version of a Regression Task

# In[34]:


from sklearn.model_selection import train_test_split


# In[35]:


x_train, x_eval, y_train, y_eval = train_test_split(x_data,y_true,test_size=0.3, random_state = 101)


# In[36]:


print(x_train.shape)
print(y_train.shape)

print(x_eval.shape)
print(y_eval.shape)


# ### Set up Estimator Inputs

# In[37]:


# Can also do .pandas_input_fn
input_func = tf.estimator.inputs.numpy_input_fn({'x':x_train},y_train,batch_size=4,num_epochs=None,shuffle=True)


# In[38]:


train_input_func = tf.estimator.inputs.numpy_input_fn({'x':x_train},y_train,batch_size=4,num_epochs=1000,shuffle=False)


# In[39]:


eval_input_func = tf.estimator.inputs.numpy_input_fn({'x':x_eval},y_eval,batch_size=4,num_epochs=1000,shuffle=False)


# ### Train the Estimator

# In[40]:


estimator.train(input_fn=input_func,steps=1000)


# ### Evaluation

# In[41]:


train_metrics = estimator.evaluate(input_fn=train_input_func,steps=1000)


# In[53]:


eval_metrics = estimator.evaluate(input_fn=eval_input_func,steps=1000)


# In[54]:


print("train metrics: {}".format(train_metrics))
print("eval metrics: {}".format(eval_metrics))


# ### Predictions

# In[68]:


input_fn_predict = tf.estimator.inputs.numpy_input_fn({'x':np.linspace(0,10,10)},shuffle=False)


# In[73]:


list(estimator.predict(input_fn=input_fn_predict))


# In[80]:


predictions = []# np.array([])
for x in estimator.predict(input_fn=input_fn_predict):
    predictions.append(x['predictions'])


# In[81]:


predictions


# In[82]:


my_data.sample(n=250).plot(kind='scatter',x='X Data',y='Y')
plt.plot(np.linspace(0,10,10),predictions,'r')


# # Great Job!
