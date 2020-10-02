# -*- coding: utf-8 -*-
"""
Created on Wed Sep 30 18:30:02 2020

@author: VK
"""


# Problem Statement : Creating a Dataset and training an Artificial Neural Network with Keras
## Drug vaccination between age 13-100 years
# People age>65 observed some side effects
# People age<65 no side effects

import keras
from keras.models import Sequential
from keras.layers import Dense
from keras.optimizers import Adam


import numpy as np
from random import randint
from sklearn.preprocessing import MinMaxScaler

train_sample = []
train_label = []

for i in range(1000):
    younger_ages = randint(13,64)
    train_sample.append(younger_ages)
    train_label.append(0)
    
    older_ages = randint(65,100)
    train_sample.append(older_ages)
    train_label.append(1)

train_sample = np.array(train_sample)
train_label = np.array(train_label)


scalar = MinMaxScaler(feature_range=(0, 1))
scalar_train_sample = scalar.fit_transform(train_sample.reshape(-1,1))

model = Sequential([Dense(16, input_dim=1, activation='relu'), Dense(32, activation='relu'), Dense(2, activation='softmax')])

model.summary()

model.compile(optimizer=Adam(learning_rate=0.001), loss='sparse_categorical_crossentropy', metrics=['accuracy'])

model.fit(scalar_train_sample, train_label, batch_size=10, epochs=10)



test_sample = []
test_label = []

for i in range(500):
    younger_ages = randint(13,64)
    test_sample.append(younger_ages)
    test_label.append(0)
    
    older_ages = randint(65,100)
    test_sample.append(older_ages)
    test_label.append(1)

test_sample = np.array(test_sample)
test_label = np.array(test_label)

scalar = MinMaxScaler(feature_range=(0, 1))
scalar_test_sample = scalar.fit_transform(test_sample.reshape(-1,1))

test_sample_output = model.predict_classes(scalar_test_sample, batch_size=10)


from sklearn.metrics import confusion_matrix, accuracy_score
predicted_values = confusion_matrix(test_label, test_sample_output)

score = accuracy_score(test_label, test_sample_output)














