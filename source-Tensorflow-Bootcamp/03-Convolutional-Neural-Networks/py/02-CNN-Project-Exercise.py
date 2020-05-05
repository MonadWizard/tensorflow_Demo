#!/usr/bin/env python
# coding: utf-8

# ## CNN-Project-Exercise
# We'll be using the CIFAR-10 dataset, which is very famous dataset for image recognition! 
# 
# The CIFAR-10 dataset consists of 60000 32x32 colour images in 10 classes, with 6000 images per class. There are 50000 training images and 10000 test images. 
# 
# The dataset is divided into five training batches and one test batch, each with 10000 images. The test batch contains exactly 1000 randomly-selected images from each class. The training batches contain the remaining images in random order, but some training batches may contain more images from one class than another. Between them, the training batches contain exactly 5000 images from each class. 
# 
# ### Follow the Instructions in Bold, if you get stuck somewhere, view the solutions video! Most of the challenge with this project is actually dealing with the data and its dimensions, not from setting up the CNN itself!

# ## Step 0: Get the Data
# 
# ** *Note: If you have trouble with this just watch the solutions video. This doesn't really have anything to do with the exercise, its more about setting up your data. Please make sure to watch the solutions video before posting any QA questions.* **

# ** Download the data for CIFAR from here: https://www.cs.toronto.edu/~kriz/cifar.html **
# 
# **Specifically the CIFAR-10 python version link: https://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz **
# 
# ** Remember the directory you save the file in! **

# In[1]:


# Put file path as a string here
CIFAR_DIR = ''


# The archive contains the files data_batch_1, data_batch_2, ..., data_batch_5, as well as test_batch. Each of these files is a Python "pickled" object produced with cPickle. 
# 
# ** Load the Data. Use the Code Below to load the data: **

# In[2]:


def unpickle(file):
    import pickle
    with open(file, 'rb') as fo:
        cifar_dict = pickle.load(fo, encoding='bytes')
    return cifar_dict


# In[3]:


dirs = ['batches.meta','data_batch_1','data_batch_2','data_batch_3','data_batch_4','data_batch_5','test_batch']


# In[4]:


all_data = [0,1,2,3,4,5,6]


# In[5]:


for i,direc in zip(all_data,dirs):
    all_data[i] = unpickle(CIFAR_DIR+direc)


# In[6]:


batch_meta = all_data[0]
data_batch1 = all_data[1]
data_batch2 = all_data[2]
data_batch3 = all_data[3]
data_batch4 = all_data[4]
data_batch5 = all_data[5]
test_batch = all_data[6]


# In[7]:


batch_meta


# ** Why the 'b's in front of the string? **
# Bytes literals are always prefixed with 'b' or 'B'; they produce an instance of the bytes type instead of the str type. They may only contain ASCII characters; bytes with a numeric value of 128 or greater must be expressed with escapes.
# 
# https://stackoverflow.com/questions/6269765/what-does-the-b-character-do-in-front-of-a-string-literal

# In[8]:


data_batch1.keys()


# Loaded in this way, each of the batch files contains a dictionary with the following elements:
# * data -- a 10000x3072 numpy array of uint8s. Each row of the array stores a 32x32 colour image. The first 1024 entries contain the red channel values, the next 1024 the green, and the final 1024 the blue. The image is stored in row-major order, so that the first 32 entries of the array are the red channel values of the first row of the image.
# * labels -- a list of 10000 numbers in the range 0-9. The number at index i indicates the label of the ith image in the array data.
# 
# The dataset contains another file, called batches.meta. It too contains a Python dictionary object. It has the following entries:
# 
# * label_names -- a 10-element list which gives meaningful names to the numeric labels in the labels array described above. For example, label_names[0] == "airplane", label_names[1] == "automobile", etc.

# ### Display a single image using matplotlib.
# 
# ** Grab a single image from data_batch1 and display it with plt.imshow(). You'll need to reshape and transpose the numpy array inside the X = data_batch[b'data'] dictionary entry.**
# 
# ** It should end up looking like this: **
# 
#     # Array of all images reshaped and formatted for viewing
#     X = X.reshape(10000, 3, 32, 32).transpose(0,2,3,1).astype("uint8")

# In[1]:


import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
import numpy as np


# In[ ]:


# Put the code here that transforms the X array!


# In[14]:


plt.imshow(X[0])


# In[15]:


plt.imshow(X[1])


# In[16]:


plt.imshow(X[4])


# # Helper Functions for Dealing With Data.
# 
# ** Use the provided code below to help with dealing with grabbing the next batch once you've gotten ready to create the Graph Session. Can you break down how it works? **

# In[17]:


def one_hot_encode(vec, vals=10):
    '''
    For use to one-hot encode the 10- possible labels
    '''
    n = len(vec)
    out = np.zeros((n, vals))
    out[range(n), vec] = 1
    return out


# In[18]:


class CifarHelper():
    
    def __init__(self):
        self.i = 0
        
        # Grabs a list of all the data batches for training
        self.all_train_batches = [data_batch1,data_batch2,data_batch3,data_batch4,data_batch5]
        # Grabs a list of all the test batches (really just one batch)
        self.test_batch = [test_batch]
        
        # Intialize some empty variables for later on
        self.training_images = None
        self.training_labels = None
        
        self.test_images = None
        self.test_labels = None
    
    def set_up_images(self):
        
        print("Setting Up Training Images and Labels")
        
        # Vertically stacks the training images
        self.training_images = np.vstack([d[b"data"] for d in self.all_train_batches])
        train_len = len(self.training_images)
        
        # Reshapes and normalizes training images
        self.training_images = self.training_images.reshape(train_len,3,32,32).transpose(0,2,3,1)/255
        # One hot Encodes the training labels (e.g. [0,0,0,1,0,0,0,0,0,0])
        self.training_labels = one_hot_encode(np.hstack([d[b"labels"] for d in self.all_train_batches]), 10)
        
        print("Setting Up Test Images and Labels")
        
        # Vertically stacks the test images
        self.test_images = np.vstack([d[b"data"] for d in self.test_batch])
        test_len = len(self.test_images)
        
        # Reshapes and normalizes test images
        self.test_images = self.test_images.reshape(test_len,3,32,32).transpose(0,2,3,1)/255
        # One hot Encodes the test labels (e.g. [0,0,0,1,0,0,0,0,0,0])
        self.test_labels = one_hot_encode(np.hstack([d[b"labels"] for d in self.test_batch]), 10)

        
    def next_batch(self, batch_size):
        # Note that the 100 dimension in the reshape call is set by an assumed batch size of 100
        x = self.training_images[self.i:self.i+batch_size].reshape(100,32,32,3)
        y = self.training_labels[self.i:self.i+batch_size]
        self.i = (self.i + batch_size) % len(self.training_images)
        return x, y


# ** How to use the above code: **

# In[19]:


# Before Your tf.Session run these two lines
ch = CifarHelper()
ch.set_up_images()

# During your session to grab the next batch use this line
# (Just like we did for mnist.train.next_batch)
# batch = ch.next_batch(100)


# ## Creating the Model
# 
# ** Import tensorflow **

# In[20]:





# ** Create 2 placeholders, x and y_true. Their shapes should be: **
# 
# * x shape = [None,32,32,3]
# * y_true shape = [None,10]
# 

# In[21]:





# ** Create one more placeholder called hold_prob. No need for shape here. This placeholder will just hold a single probability for the dropout. **

# In[22]:





# ### Helper Functions
# 
# ** Grab the helper functions from MNIST with CNN (or recreate them here yourself for a hard challenge!). You'll need: **
# 
# * init_weights
# * init_bias
# * conv2d
# * max_pool_2by2
# * convolutional_layer
# * normal_full_layer

# In[23]:





# ### Create the Layers
# 
# ** Create a convolutional layer and a pooling layer as we did for MNIST. **
# ** Its up to you what the 2d size of the convolution should be, but the last two digits need to be 3 and 32 because of the 3 color channels and 32 pixels. So for example you could use:**
# 
#         convo_1 = convolutional_layer(x,shape=[4,4,3,32])

# In[24]:





# ** Create the next convolutional and pooling layers.  The last two dimensions of the convo_2 layer should be 32,64 **

# In[25]:





# ** Now create a flattened layer by reshaping the pooling layer into [-1,8 \* 8 \* 64] or [-1,4096] **

# In[26]:


8*8*64


# In[27]:





# ** Create a new full layer using the normal_full_layer function and passing in your flattend convolutional 2 layer with size=1024. (You could also choose to reduce this to something like 512)**

# In[28]:





# ** Now create the dropout layer with tf.nn.dropout, remember to pass in your hold_prob placeholder. **

# In[29]:





# ** Finally set the output to y_pred by passing in the dropout layer into the normal_full_layer function. The size should be 10 because of the 10 possible labels**

# In[30]:





# ### Loss Function
# 
# ** Create a cross_entropy loss function **

# In[31]:





# ### Optimizer
# ** Create the optimizer using an Adam Optimizer. **

# In[32]:





# ** Create a variable to intialize all the global tf variables. **

# In[33]:





# ## Graph Session
# 
# ** Perform the training and test print outs in a Tf session and run your model! **

# In[34]:




