# -*- coding: utf-8 -*-
"""Categorizing_articles.ipynb

"""

import os 
import re
import json
import pickle
import datetime 
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, classification_report

from tensorflow.keras import Input,Sequential
from tensorflow.keras.utils import plot_model
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.callbacks import TensorBoard,EarlyStopping
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.layers import LSTM,Dense,Dropout,Embedding,Bidirectional

# Step 1) Data Loading
df = pd.read_csv('https://raw.githubusercontent.com/susanli2016/PyCon-Canada-2019-NLP-Tutorial/master/bbc-text.csv',
                 delimiter=',')

# Step 2) Data Inspection
df.info()
df.describe().T
df.head()

# Explore a few articles from the dataset
print(df['text'][10])
print(df['text'][50])
print(df['text'][100])

# The text are clean

print(len(df['text'][100]))
print(len(df['text'][10]))
print(len(df['text'][50]))

# The length of each article is different from each other

# Backup the data to save time of loading data from database
text = df['text']
category = df['category']

text_backup = text.copy()
category_backup = category.copy()

# Step 4) Features Selection
# There are no feature that need to be selected. Only 2 features available in this dataset.
# Step 5) Pre-Processing

#Tokenization used in this dataset helps breaking the raw text into small words.
#It helps in interpreting the meaning of the text by analyzing the sequence of the words

# The tokenized dataset will take 50000 most common words. 
# oov_token will put a OOV value for words that are not available in word_index

vocab_size = 50000
oov_token = '<OOV'
tokenizer = Tokenizer(num_words=vocab_size,oov_token=oov_token)
tokenizer.fit_on_texts(text)  # Go through the text and create a dictionary
word_index = tokenizer.word_index

print(dict(list(word_index.items())[0:10]))

# Converting tokens to list of sequence
text_int = tokenizer.texts_to_sequences(text)

# use median for truncating & padding 

# Use padding to make the sequences as the same size which is 341. 
# All sequences has to be in size of 341

max_len = np.median([len(text_int[i]) for i in range(len(text_int))])
print(max_len)

# If the sequence is less than the max_len, then padding is applied at 'post' or at the last sequence.
# The padded value used at the end of sequence is median
padded_text = pad_sequences(text_int,maxlen=int(max_len),padding='post',
                              truncating='post')

# For y features, convert the categorical data to numerical value 
ohe = OneHotEncoder(sparse=False)
category = ohe.fit_transform(np.expand_dims(category,axis=-1))

X_train,X_test,y_train,y_test = train_test_split(padded_text,category,
                                                 test_size=0.3,
                                                 random_state=123)

# Model Development

# Embedding is used in the model to convert 50000 (vocab_size) to 200 (out_dims)
# Bidireectional LSTM is used as input flows in both directions, and it's capable of utilizing information from both sides
# Use sofmax for the activation function of the output layer as it can handle multiple classes.

input_shape = np.shape(X_train)[1:]
out_dim = 200

model = Sequential()
model.add(Input(shape=(input_shape)))
model.add(Embedding(vocab_size,out_dim))
model.add(Bidirectional(LSTM(128,return_sequences=True)))
model.add(Dropout(0.2))
model.add(Bidirectional(LSTM(128)))
model.add(Dropout(0.2))
model.add(Dense(5,activation='softmax'))
model.summary()

plot_model(model,show_shapes=(True))

# Used as categorical_crossentropy as loss function as there are more than two output labels
model.compile(optimizer='adam',loss='categorical_crossentropy',
              metrics=['acc'])

#Callback
log_dir = os.path.join("logs_text", datetime.datetime.now().strftime("%Y%m%d-%H%M%S"))
tensorboard_callback = TensorBoard(log_dir=log_dir, histogram_freq=1)

# Use EarlyStopping to prevent overfitting
early_callback = EarlyStopping(monitor='val_loss',patience=6)

hist = model.fit(X_train,y_train,
                 epochs=5,
                 validation_data=(X_test,y_test),
                 callbacks=[tensorboard_callback,early_callback])

# Model evaluation

print(hist.history.keys())

plt.figure()
plt.plot(hist.history['loss'])
plt.plot(hist.history['val_loss'])
plt.xlabel('epoch')
plt.legend(['Training Loss', 'Validation Loss'])
plt.show()

plt.figure()
plt.plot(hist.history['acc'])
plt.plot(hist.history['val_acc'])
plt.xlabel('epoch')
plt.legend(['Training Acc', 'Validation Acc'])
plt.show()

print(model.evaluate(X_test,y_test))

#Model Analysis
pred_y = model.predict(X_test)
pred_y = np.argmax(pred_y,axis=1)
true_y = np.argmax(y_test,axis=1)

#print classification report for the predicted and validation data
cr = classification_report(true_y,pred_y)
print(cr)

# Commented out IPython magic to ensure Python compatibility.
# Load the TensorBoard notebook extension
# %load_ext tensorboard
# %tensorboard --logdir logs_text

# Model Saving
TOKENIZER_SAVE_PATH = os.path.join(os.getcwd(),'sample_data',
                                   'tokenizer.json')              
token_json = tokenizer.to_json()

with open(TOKENIZER_SAVE_PATH,'w') as file:
  json.dump(token_json,file)

# OHE 
OHE_SAVE_PATH = os.path.join(os.getcwd(),'sample_data',
                             'ohe.pkl')
with open(OHE_SAVE_PATH,'wb') as file:
  pickle.dump(ohe,file)

# Model
MODEL_SAVE_PATH = os.path.join(os.getcwd(),'sample_data','model.h5')
model.save(MODEL_SAVE_PATH)