#!/usr/bin/env python
# coding: utf-8

# # MNIST Data Set - Basic Approach

# ### Get the MNIST Data

# In[1]:


import tensorflow as tf


# In[2]:


from tensorflow.examples.tutorials.mnist import input_data


# In[3]:


mnist = input_data.read_data_sets("MNIST_data/",one_hot=True)


# ** Alternative sources of the data just in case: **
# 
# * http://yann.lecun.com/exdb/mnist/
# * https://github.com/mrgloom/MNIST-dataset-in-different-formats

# In[4]:


type(mnist)


# In[8]:


mnist.train.images


# In[9]:


mnist.train.num_examples


# In[10]:


mnist.test.num_examples


# In[11]:


mnist.validation.num_examples


# ### Visualizing the Data

# In[12]:


import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')


# In[20]:


mnist.train.images[1].shape


# In[22]:


plt.imshow(mnist.train.images[1].reshape(28,28))


# In[24]:


plt.imshow(mnist.train.images[1].reshape(28,28),cmap='gist_gray')


# In[25]:


mnist.train.images[1].max()


# In[44]:


plt.imshow(mnist.train.images[1].reshape(784,1))


# In[45]:


plt.imshow(mnist.train.images[1].reshape(784,1),cmap='gist_gray',aspect=0.02)


# ## Create the Model

# In[48]:


x = tf.placeholder(tf.float32,shape=[None,784])


# In[49]:


# 10 because 0-9 possible numbers
W = tf.Variable(tf.zeros([784,10]))


# In[50]:


b = tf.Variable(tf.zeros([10]))


# In[51]:


# Create the Graph
y = tf.matmul(x,W) + b 


# Loss and Optimizer

# In[52]:


y_true = tf.placeholder(tf.float32,[None,10])


# In[53]:


# Cross Entropy


# In[55]:


cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y_true, logits=y))


# In[69]:


optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.5)


# In[70]:


train = optimizer.minimize(cross_entropy)


# ### Create Session

# In[71]:


init = tf.global_variables_initializer()


# In[72]:


with tf.Session() as sess:
    sess.run(init)
    
    # Train the model for 1000 steps on the training set
    # Using built in batch feeder from mnist for convenience
    
    for step in range(1000):
        
        batch_x , batch_y = mnist.train.next_batch(100)
        
        sess.run(train,feed_dict={x:batch_x,y_true:batch_y})
        
    # Test the Train Model
    matches = tf.equal(tf.argmax(y,1),tf.argmax(y_true,1))
    
    acc = tf.reduce_mean(tf.cast(matches,tf.float32))
    
    print(sess.run(acc,feed_dict={x:mnist.test.images,y_true:mnist.test.labels}))


# While this may seem pretty good, we can actually do much better, the best models can get above 99% accuracy.
# 
# How do they do this? By using other models, such as convolutional neural networks!
