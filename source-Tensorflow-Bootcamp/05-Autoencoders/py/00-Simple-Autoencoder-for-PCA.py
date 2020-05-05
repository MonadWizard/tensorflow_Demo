#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')


# ## Create Some Data

# In[2]:


from sklearn.datasets import make_blobs


# In[3]:


data = make_blobs(n_samples=100, n_features=3,centers=2,random_state=101)


# In[4]:


data


# ## Scale the Data

# In[18]:


from sklearn.preprocessing import MinMaxScaler


# In[19]:


scaler = MinMaxScaler()
scaled_data = scaler.fit_transform(data[0])


# In[20]:


# data[0] 
data_x = scaled_data[:,0]
data_y = scaled_data[:,1]
data_z = scaled_data[:,2]


# In[21]:


from mpl_toolkits.mplot3d import Axes3D


# In[22]:


fig = plt.figure()
ax = fig.add_subplot(111,projection='3d')


# In[37]:


fig = plt.figure()
ax = fig.add_subplot(111,projection='3d')
ax.scatter(data_x,data_y,data_z,c=data[1])


# # The Linear Autoencoder

# In[24]:


import tensorflow as tf
from tensorflow.contrib.layers import fully_connected


# In[25]:


num_inputs = 3  # 3 dimensional input
num_hidden = 2  # 2 dimensional representation 
num_outputs = num_inputs # Must be true for an autoencoder!

learning_rate = 0.01


# ### Placeholder
# 
# Notice there is no real label here, just X.

# In[26]:


X = tf.placeholder(tf.float32, shape=[None, num_inputs])


# ### Layers
# 
# Using the fully_connected layers API, we **do not** provide an activation function!

# In[27]:


hidden = fully_connected(X, num_hidden, activation_fn=None)
outputs = fully_connected(hidden, num_outputs, activation_fn=None)


# ### Loss Function

# In[28]:


loss = tf.reduce_mean(tf.square(outputs - X))  # MSE


# ### Optimizer

# In[29]:


optimizer = tf.train.AdamOptimizer(learning_rate)
train  = optimizer.minimize( loss)


# ### Init

# In[30]:


init = tf.global_variables_initializer()


# ## Running the Session

# In[35]:


num_steps = 1000

with tf.Session() as sess:
    sess.run(init)
    
    for iteration in range(num_steps):
        sess.run(train,feed_dict={X: scaled_data})

        
    # Now ask for the hidden layer output (the 2 dimensional output)
    output_2d = hidden.eval(feed_dict={X: scaled_data})


# In[36]:


output_2d.shape


# In[42]:


plt.scatter(output_2d[:,0],output_2d[:,1],c=data[1])


# # Great Job!
