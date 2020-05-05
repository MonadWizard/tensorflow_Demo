#!/usr/bin/env python
# coding: utf-8

# # Deep Nets with TF Abstractions
# 
# Let's explore a few of the various abstractions that TensorFlow offers. You can check out the tf.contrib documentation for more options.

# # The Data

# To compare these various abstractions we'll use a dataset easily available from the SciKit Learn library. The data is comprised of the results of a chemical analysis of wines grown in the same region in Italy by three different cultivators. There are thirteen different
# measurements taken for different constituents found in the three types of wine. We will use the various TF Abstractions to classify the wine to one of the 3 possible labels.
# 
# First let's show you how to get the data:

# In[1]:


from sklearn.datasets import load_wine


# In[2]:


wine_data = load_wine()


# In[3]:


type(wine_data)


# In[4]:


wine_data.keys()


# In[5]:


print(wine_data.DESCR)


# In[6]:


feat_data = wine_data['data']


# In[7]:


labels = wine_data['target']


# ### Train Test Split

# In[8]:


from sklearn.model_selection import train_test_split


# In[60]:


X_train, X_test, y_train, y_test = train_test_split(feat_data,
                                                    labels,
                                                    test_size=0.3,
                                                   random_state=101)


# ### Scale the Data

# In[61]:


from sklearn.preprocessing import MinMaxScaler


# In[62]:


scaler = MinMaxScaler()


# In[63]:


scaled_x_train = scaler.fit_transform(X_train)


# In[64]:


scaled_x_test = scaler.transform(X_test)


# # Abstractions

# ## Estimator API

# In[14]:


import tensorflow as tf


# In[260]:


from tensorflow import estimator 


# In[261]:


X_train.shape


# In[262]:


feat_cols = [tf.feature_column.numeric_column("x", shape=[13])]


# In[263]:


deep_model = estimator.DNNClassifier(hidden_units=[13,13,13],
                            feature_columns=feat_cols,
                            n_classes=3,
                            optimizer=tf.train.GradientDescentOptimizer(learning_rate=0.01) )


# In[264]:


input_fn = estimator.inputs.numpy_input_fn(x={'x':scaled_x_train},y=y_train,shuffle=True,batch_size=10,num_epochs=5)


# In[265]:


deep_model.train(input_fn=input_fn,steps=500)


# In[266]:


input_fn_eval = estimator.inputs.numpy_input_fn(x={'x':scaled_x_test},shuffle=False)


# In[267]:


preds = list(deep_model.predict(input_fn=input_fn_eval))


# In[268]:


predictions = [p['class_ids'][0] for p in preds]


# In[269]:


from sklearn.metrics import confusion_matrix,classification_report


# In[270]:


print(classification_report(y_test,predictions))


# ____________
# ______________

# # TensorFlow Keras

# ### Create the Model

# In[728]:


from tensorflow.contrib.keras import models


# In[729]:


dnn_keras_model = models.Sequential()


# ### Add Layers to the model

# In[730]:


from tensorflow.contrib.keras import layers


# In[731]:


dnn_keras_model.add(layers.Dense(units=13,input_dim=13,activation='relu'))


# In[732]:


dnn_keras_model.add(layers.Dense(units=13,activation='relu'))
dnn_keras_model.add(layers.Dense(units=13,activation='relu'))


# In[733]:


dnn_keras_model.add(layers.Dense(units=3,activation='softmax'))


# ### Compile the Model

# In[734]:


from tensorflow.contrib.keras import losses,optimizers,metrics


# In[735]:


# explore these
# losses.


# In[736]:


#optimizers.


# In[744]:


losses.sparse_categorical_crossentropy


# In[738]:


dnn_keras_model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])


# ### Train Model

# In[741]:


dnn_keras_model.fit(scaled_x_train,y_train,epochs=50)


# In[742]:


predictions = dnn_keras_model.predict_classes(scaled_x_test)


# In[743]:


print(classification_report(predictions,y_test))


# # Layers API
# 
# https://www.tensorflow.org/tutorials/layers

# ## Formating Data

# In[1]:


import pandas as pd
from sklearn.datasets import load_wine
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler


# In[2]:


wine_data = load_wine()
feat_data = wine_data['data']
labels = wine_data['target']


# In[23]:


X_train, X_test, y_train, y_test = train_test_split(feat_data,
                                                    labels,
                                                    test_size=0.3,
                                                   random_state=101)


# In[4]:


scaler = MinMaxScaler()
scaled_x_train = scaler.fit_transform(X_train)
scaled_x_test = scaler.transform(X_test)
# ONE HOT ENCODED
onehot_y_train = pd.get_dummies(y_train).as_matrix()
one_hot_y_test = pd.get_dummies(y_test).as_matrix()


# ### Parameters

# In[5]:


num_feat = 13
num_hidden1 = 13
num_hidden2 = 13
num_outputs = 3
learning_rate = 0.01


# In[6]:


import tensorflow as tf
from tensorflow.contrib.layers import fully_connected


# ### Placeholder

# In[7]:


X = tf.placeholder(tf.float32,shape=[None,num_feat])
y_true = tf.placeholder(tf.float32,shape=[None,3])


# ### Activation Function

# In[8]:


actf = tf.nn.relu


# ### Create Layers

# In[9]:


hidden1 = fully_connected(X,num_hidden1,activation_fn=actf)


# In[10]:


hidden2 = fully_connected(hidden1,num_hidden2,activation_fn=actf)


# In[11]:


output = fully_connected(hidden2,num_outputs)


# ### Loss Function

# In[12]:


loss = tf.losses.softmax_cross_entropy(onehot_labels=y_true, logits=output)


# ### Optimizer

# In[13]:


optimizer = tf.train.AdamOptimizer(learning_rate)
train = optimizer.minimize(loss)


# ### Init

# In[14]:


init = tf.global_variables_initializer()


# In[21]:


training_steps = 1000
with tf.Session() as sess:
    sess.run(init)
    
    for i in range(training_steps):
        sess.run(train,feed_dict={X:scaled_x_train,y_true:y_train})
        
    # Get Predictions
    logits = output.eval(feed_dict={X:scaled_x_test})
    
    preds = tf.argmax(logits,axis=1)
    
    results = preds.eval()


# In[25]:


from sklearn.metrics import confusion_matrix,classification_report
print(classification_report(results,y_test))


# In[ ]:





# In[ ]:




