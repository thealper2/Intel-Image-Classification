#!/usr/bin/env python
# coding: utf-8

# In[8]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
from os import listdir
from os.path import isfile, join
from sklearn.preprocessing import LabelBinarizer
from keras.preprocessing import image
from tensorflow.keras.utils import img_to_array, array_to_img
from keras.optimizers import Adam
from PIL import Image
from keras.models import Sequential
from keras.layers import BatchNormalization
from keras.layers import Conv2D, MaxPooling2D, Activation, Flatten, Dropout, Dense, LeakyReLU
from sklearn.model_selection import train_test_split
import pickle


# In[11]:


root_dir = "seg_train/seg_train"
image_list = []
label_list = []

for directory in os.listdir(root_dir):
    directory_path = join(root_dir, directory)
    if os.path.isdir(directory_path):
        for file_name in os.listdir(directory_path):
            file_path = join(directory_path, file_name)
            if isfile(file_path):
                image = Image.open(file_path)
                image = image.resize((150, 150))
                image = img_to_array(image)
                image_list.append(image)
                label_list.append(directory)


# In[12]:


df_labels = pd.DataFrame(label_list).value_counts()
df_labels


# In[13]:


np.array(image_list).shape


# In[14]:


X_train, X_test, y_train, y_test = train_test_split(image_list, label_list, test_size=0.2, random_state=4242) 


# In[15]:


# Normalization
X_train = np.array(X_train, dtype=np.float16) / 225.0
X_test = np.array(X_test, dtype=np.float16) / 225.0


# In[16]:


lb = LabelBinarizer()
y_train = lb.fit_transform(y_train)
y_test = lb.fit_transform(y_test)
print(lb.classes_)


# In[17]:


pickle.dump(lb, open("lb.pkl", "wb"))


# In[18]:


X_train, X_valid, y_train, y_valid = train_test_split(X_train, y_train, test_size=0.2)


# In[19]:


model = Sequential([
    Conv2D(16, kernel_size=(3,3), input_shape=(150, 150, 3)),
    BatchNormalization(),
    LeakyReLU(),
    
    Conv2D(32, kernel_size=(3,3)),
    BatchNormalization(),
    LeakyReLU(),
    MaxPooling2D(5,5),
    
    Flatten(),
    
    Dense(32),
    Dropout(rate=0.2),
    BatchNormalization(),
    LeakyReLU(),
    
    Dense(16),
    Dropout(rate=0.2),
    BatchNormalization(),
    LeakyReLU(1),
    
    Dense(6, activation="softmax")
])


# In[20]:


model.summary()


# In[21]:


model.compile(loss="categorical_crossentropy", optimizer=Adam(0.0005), metrics=["accuracy"])


# In[22]:


history = model.fit(X_train, y_train, batch_size=128, epochs=10, validation_data=(X_valid, y_valid))


# In[23]:


model.save("intel_image.h5")


# In[24]:


plt.figure()
plt.plot(history.history["accuracy"])
plt.plot(history.history["val_accuracy"])
plt.title("Model Accuracy")
plt.ylabel("Accuracy")
plt.xlabel("Epochs")
plt.legend(["train", "val"])
plt.show()


# In[25]:


plt.figure()
plt.plot(history.history["loss"])
plt.plot(history.history["val_loss"])
plt.title("Model Loss")
plt.ylabel("Loss")
plt.xlabel("Epochs")
plt.show()


# In[26]:


scores = model.evaluate(X_test, y_test)
acc = scores[1] * 100
print("Test Accuracy:", acc)


# In[27]:


y_pred = model.predict(X_test)


# In[28]:


labels = lb.classes_
print("Originally : ",labels[np.argmax(y_test[44])])
print("Predicted : ",labels[np.argmax(y_pred[44])])


# In[29]:


count = 0
for i in range(1, 100):
    if labels[np.argmax(y_test[i])] == labels[np.argmax(y_pred[i])]:
        count += 1
        
print(count , "%")


# In[ ]:




