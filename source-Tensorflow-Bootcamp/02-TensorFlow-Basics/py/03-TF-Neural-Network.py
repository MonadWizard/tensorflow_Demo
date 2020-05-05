#!/usr/bin/env python
# coding: utf-8

# # First Neurons

# In[6]:


import numpy as np
import tensorflow as tf


# ** Set Random Seeds for same results **

# In[7]:


np.random.seed(101)
tf.set_random_seed(101)


# ** Data Setup **

# Setting Up some Random Data for Demonstration Purposes

# In[8]:


rand_a = np.random.uniform(0,100,(5,5))
rand_a


# In[9]:


rand_b = np.random.uniform(0,100,(5,1))
rand_b


# In[10]:


# CONFIRM SAME  RANDOM NUMBERS (EXECUTE SEED IN SAME CELL!) Watch video for explanation
np.random.seed(101)
rand_a = np.random.uniform(0,100,(5,5))
rand_b = np.random.uniform(0,100,(5,1))


# ### Placeholders

# In[11]:


a = tf.placeholder(tf.float32)
b = tf.placeholder(tf.float32)


# ### Operations

# In[13]:


add_op = a+b # tf.add(a,b)
mult_op = a*b #tf.multiply(a,b)


# ### Running Sessions  to create Graphs with Feed Dictionaries

# In[17]:


with tf.Session() as sess:
    add_result = sess.run(add_op,feed_dict={a:rand_a,b:rand_b})
    print(add_result)
    
    print('\n')
    
    mult_result = sess.run(mult_op,feed_dict={a:rand_a,b:rand_b})
    print(mult_result)


# ________________________
# 
# ________________________

# ## Example Neural Network

# In[18]:


n_features = 10
n_dense_neurons = 3


# In[19]:


# Placeholder for x
x = tf.placeholder(tf.float32,(None,n_features))


# In[21]:


# Variables for w and b
b = tf.Variable(tf.zeros([n_dense_neurons]))

W = tf.Variable(tf.random_normal([n_features,n_dense_neurons]))


# ** Operation Activation Function **

# In[22]:


xW = tf.matmul(x,W)


# In[23]:


z = tf.add(xW,b)


# In[24]:


# tf.nn.relu() or tf.tanh()
a = tf.sigmoid(z)


# ** Variable Intializer! **

# In[25]:


init = tf.global_variables_initializer()


# In[27]:


with tf.Session() as sess:
    sess.run(init)
    
    layer_out = sess.run(a,feed_dict={x : np.random.random([1,n_features])})


# In[28]:


print(layer_out)


# We still need to finish off this process with optimization! Let's learn how to do this next.
# 
# _____

# ## Full Network Example
# 
# Let's work on a regression example, we are trying to solve a very simple equation:
# 
# y = mx + b
# 
# y will be the y_labels and x is the x_data. We are trying to figure out the slope and the intercept for the line that best fits our data!

# ### Artifical Data (Some Made Up Regression Data)

# In[37]:


x_data = np.linspace(0,10,10) + np.random.uniform(-1.5,1.5,10)


# In[38]:


x_data


# In[39]:


y_label = np.linspace(0,10,10) + np.random.uniform(-1.5,1.5,10)


# In[40]:


plt.plot(x_data,y_label,'*')


# ** Variables **

# In[79]:


np.random.rand(2)


# In[80]:


m = tf.Variable(0.39)
b = tf.Variable(0.2)


# ### Cost Function

# In[81]:


error = 0

for x,y in zip(x_data,y_label):
    
    y_hat = m*x + b  #Our predicted value
    
    error += (y-y_hat)**2 # The cost we want to minimize (we'll need to use an optimization function for the minimization!)


# ### Optimizer

# In[82]:


optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.001)
train = optimizer.minimize(error)


# ### Initialize Variables

# In[83]:


init = tf.global_variables_initializer()


# ### Create Session and Run!

# In[84]:


with tf.Session() as sess:
    
    sess.run(init)
    
    epochs = 100
    
    for i in range(epochs):
        
        sess.run(train)
        

    # Fetch Back Results
    final_slope , final_intercept = sess.run([m,b])


# In[85]:


final_slope


# In[86]:


final_intercept


# ### Evaluate Results

# In[88]:


x_test = np.linspace(-1,11,10)
y_pred_plot = final_slope*x_test + final_intercept

plt.plot(x_test,y_pred_plot,'r')

plt.plot(x_data,y_label,'*')


# # Great Job!
