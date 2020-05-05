#!/usr/bin/env python
# coding: utf-8

# # Linear Autoencoder for PCA - EXERCISE SOLUTIONS
# 
# ** Follow the bold instructions below to reduce a 30 dimensional data set for classification into a 2-dimensional dataset! Then use the color classes to see if you still kept the same level of class separation in the dimensionality reduction**

# ## The Data
# 
# ** Import numpy, matplotlib, and pandas**

# In[1]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')


# ** Use pandas to read in the csv file called anonymized_data.csv . It contains 500 rows and 30 columns of anonymized data along with 1 last column with a classification label, where the columns have been renamed to 4 letter codes.**

# In[2]:


df = pd.read_csv('anonymized_data.csv')


# In[3]:


df.head()


# In[4]:


df.info()


# ## Scale the Data
# 
# ** Use scikit learn to scale the data with a MinMaxScaler. Remember not to scale the Label column, just the data. Save this scaled data as a new variable called scaled_data. **

# In[5]:


from sklearn.preprocessing import MinMaxScaler


# In[6]:


scaler = MinMaxScaler()
scaled_data = scaler.fit_transform(df.drop('Label',axis=1))


# # The Linear Autoencoder

# ** Import tensorflow and import fully_connected layers from tensorflow.contrib.layers. **

# In[7]:


import tensorflow as tf
from tensorflow.contrib.layers import fully_connected


# ** Fill out the number of inputs to fit the dimensions of the data set and set the hidden number of units to be 2. Also set the number of outputs to match the number of inputs. Also choose a learning_rate value.**

# In[8]:


num_inputs = 30  # 3 dimensional input
num_hidden = 2  # 2 dimensional representation 
num_outputs = num_inputs # Must be true for an autoencoder!

learning_rate = 0.01


# ### Placeholder
# 
# ** Create a placeholder fot the data called X.**

# In[9]:


X = tf.placeholder(tf.float32, shape=[None, num_inputs])


# ### Layers
# 
# ** Create the hidden layer and the output layers using the fully_connected function. Remember that to perform PCA there is no activation function.**

# In[10]:


hidden = fully_connected(X, num_hidden, activation_fn=None)
outputs = fully_connected(hidden, num_outputs, activation_fn=None)


# ### Loss Function
# 
# ** Create a Mean Squared Error loss function. **

# In[11]:


loss = tf.reduce_mean(tf.square(outputs - X))  # MSE


# ### Optimizer

# ** Create an AdamOptimizer designed to minimize the previous loss function. **

# In[12]:


optimizer = tf.train.AdamOptimizer(learning_rate)
train  = optimizer.minimize( loss)


# ### Init
# 
# ** Create an instance of a global variable intializer. **

# In[13]:


init = tf.global_variables_initializer()


# ## Running the Session
# 
# ** Now create a Tensorflow session that runs the optimizer for at least 1000 steps. (You can also use epochs if you prefer, where 1 epoch is defined by one single run through the entire dataset. **

# In[17]:


num_steps = 1000

with tf.Session() as sess:
    sess.run(init)
    
    for iteration in range(num_steps):
        sess.run(train,feed_dict={X: scaled_data})


# ** Now create a session that runs the scaled data through the hidden layer. (You could have also done this in the last step after all the training steps. **

# In[18]:


with tf.Session() as sess:
    sess.run(init)
        
    # Now ask for the hidden layer output (the 2 dimensional output)
    output_2d = hidden.eval(feed_dict={X: scaled_data})


# ** Confirm that your output is now 2 dimensional along the previous axis of 30 features. **

# In[21]:


output_2d.shape


# ** Now plot out the reduced dimensional representation of the data. Do you still have clear separation of classes even with the reduction in dimensions? Hint: You definitely should, the classes should still be clearly seperable, even when reduced to 2 dimensions. **

# In[22]:


plt.scatter(output_2d[:,0],output_2d[:,1],c=df['Label'])


# # Great Job!

# In[ ]:




