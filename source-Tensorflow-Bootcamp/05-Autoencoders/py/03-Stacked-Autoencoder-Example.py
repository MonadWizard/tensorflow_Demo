#!/usr/bin/env python
# coding: utf-8

# # Stacked Autoencoder

# In[1]:


import numpy as np
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')


# In[2]:


import tensorflow as tf


# In[3]:


from tensorflow.examples.tutorials.mnist import input_data


# In[4]:


mnist = input_data.read_data_sets("../03-Convolutional-Neural-Networks/MNIST_data/",one_hot=True)


# In[5]:


tf.reset_default_graph() 


# ## Parameters

# In[6]:


num_inputs = 784 # 28*28
neurons_hid1 = 392
neurons_hid2 = 196
neurons_hid3 = neurons_hid1 # Decoder Begins
num_outputs = num_inputs

learning_rate = 0.01


# ** Activation function **

# In[7]:


actf = tf.nn.relu


# ### Placeholder

# In[8]:


X = tf.placeholder(tf.float32, shape=[None, num_inputs])


# ### Weights
# 
# Initializer capable of adapting its scale to the shape of weights tensors.
# 
# With `distribution="normal"`, samples are drawn from a truncated normal
# distribution centered on zero, with `stddev = sqrt(scale / n)`
# where n is:
#   - number of input units in the weight tensor, if mode = "fan_in"
#   - number of output units, if mode = "fan_out"
#   - average of the numbers of input and output units, if mode = "fan_avg"
# 
# With `distribution="uniform"`, samples are drawn from a uniform distribution
# within [-limit, limit], with `limit = sqrt(3 * scale / n)`.
# 

# In[9]:


initializer = tf.variance_scaling_initializer()


# In[10]:


w1 = tf.Variable(initializer([num_inputs, neurons_hid1]), dtype=tf.float32)
w2 = tf.Variable(initializer([neurons_hid1, neurons_hid2]), dtype=tf.float32)
w3 = tf.Variable(initializer([neurons_hid2, neurons_hid3]), dtype=tf.float32)
w4 = tf.Variable(initializer([neurons_hid3, num_outputs]), dtype=tf.float32)


# ### Biases

# In[11]:


b1 = tf.Variable(tf.zeros(neurons_hid1))
b2 = tf.Variable(tf.zeros(neurons_hid2))
b3 = tf.Variable(tf.zeros(neurons_hid3))
b4 = tf.Variable(tf.zeros(num_outputs))


# ### Activation Function and Layers

# In[12]:


act_func = tf.nn.relu


# In[13]:


hid_layer1 = act_func(tf.matmul(X, w1) + b1)
hid_layer2 = act_func(tf.matmul(hid_layer1, w2) + b2)
hid_layer3 = act_func(tf.matmul(hid_layer2, w3) + b3)
output_layer = tf.matmul(hid_layer3, w4) + b4


# ### Loss Function

# In[14]:


loss = tf.reduce_mean(tf.square(output_layer - X))


# ### Optimizer

# In[15]:


#tf.train.RMSPropOptimizer
optimizer = tf.train.AdamOptimizer(learning_rate)


# In[16]:


train = optimizer.minimize(loss)


# ### Intialize Variables

# In[17]:


init = tf.global_variables_initializer()


# In[18]:


saver = tf.train.Saver() 


# In[19]:


num_epochs = 5
batch_size = 150

with tf.Session() as sess:
    sess.run(init)
    
    # Epoch == Entire Training Set
    for epoch in range(num_epochs):
        
        num_batches = mnist.train.num_examples // batch_size
        
        # 150 batch size
        for iteration in range(num_batches):
            
            X_batch, y_batch = mnist.train.next_batch(batch_size)
            sess.run(train, feed_dict={X: X_batch})
            
        training_loss = loss.eval(feed_dict={X: X_batch})   
        
        print("Epoch {} Complete. Training Loss: {}".format(epoch,training_loss))
     
    saver.save(sess, "./stacked_autoencoder.ckpt")      


# ## Test Autoencoder output on Test Data

# In[20]:


num_test_images = 10

with tf.Session() as sess:
    
    saver.restore(sess,"./stacked_autoencoder.ckpt")
    
    results = output_layer.eval(feed_dict={X:mnist.test.images[:num_test_images]})


# In[21]:


# Compare original images with their reconstructions
f, a = plt.subplots(2, 10, figsize=(20, 4))
for i in range(num_test_images):
    a[0][i].imshow(np.reshape(mnist.test.images[i], (28, 28)))
    a[1][i].imshow(np.reshape(results[i], (28, 28)))


# # Great Job!
