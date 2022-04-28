#!/usr/bin/env python
# -*- coding: utf-8 -*-
#----------------------------------------------------------------------------
# Created By  : Arjun Thangaraju
# ---------------------------------------------------------------------------
# Setup
import os
os.environ["CUDA_VISIBLE_DEVICES"] = '-1'
import tensorflow as tf
if tf.test.gpu_device_name():
    print('GPU found')
else:
    print("No GPU found")

import tensorflow as tf
from tensorflow import keras
tf.test.gpu_device_name()
# import libraries
import seaborn as sns
import plotly.express as px
import plotly.offline as pyo

from sklearn.preprocessing import LabelEncoder

from keras.layers import Conv2D, Conv1D, MaxPooling2D, MaxPooling1D, Flatten, BatchNormalization, Dense
from keras.utils.np_utils import to_categorical
from keras.models import Sequential
from keras.callbacks import CSVLogger, ModelCheckpoint
from keras.models import load_model
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import pickle
import time

from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report


# Track Elapsed Time
start_time = time.time()

# Load Data
ids2018_day1 = pd.read_csv('data/data_in_csv/02-14-2018.csv')
ids2018_day2 = pd.read_csv('data/data_in_csv/02-15-2018.csv')

network_data = pd.concat([ids2018_day1, ids2018_day2])
network_data.reset_index(inplace=True, drop=True)
# Exploratory Data Analysis
# check the shape of data
network_data.shape
# check the number of rows and columns
print('Number of Rows (Samples): %s' % str((network_data.shape[0])))
print('Number of Columns (Features): %s' % str((network_data.shape[1])))

network_data.head(4)
# check the columns in data
network_data.columns
# check the number of columns
print('Total columns in our data: %s' % str(len(network_data.columns)))
network_data.info()
# check the number of values for labels
print(network_data['Label'].value_counts())
# Data Visualization
# make a plot number of labels
sns.set(rc={'figure.figsize':(12, 6)})
plt.xlabel('Attack Type')
sns.set_theme()
ax = sns.countplot(x='Label', data=network_data)
ax.set(xlabel='Attack Type', ylabel='Number of Attacks')
plt.show()

# Scatterplot
# pyo.init_notebook_mode()
fig = px.scatter(x = network_data["Bwd Pkts/s"][:100000],
                 y=network_data["Fwd Seg Size Min"][:100000])
# fig
plt.show()
# %%timeValueError: cannot reindex from a duplicate axis
sns.set(rc={'figure.figsize':(12, 6)})
sns.scatterplot(x=network_data['Bwd Pkts/s'][:50000], y=network_data['Fwd Seg Size Min'][:50000],
                hue='Label', data=network_data)
plt.show()
# check the dtype of timestamp column
(network_data['Timestamp'].dtype)

# Date Preprocessing
# check for some null or missing values in our dataset
network_data.isna().sum().to_numpy()
# drop null or missing columns
cleaned_data = network_data.dropna()
cleaned_data.isna().sum().to_numpy()

# Label Encoding
# encode the column labels
label_encoder = LabelEncoder()
cleaned_data['Label'] = label_encoder.fit_transform(cleaned_data['Label'])
cleaned_data['Label'].unique()

# Shuffle Train Dataset
cleaned_data = shuffle(cleaned_data)

# Split train and test dataset
train, test = train_test_split(cleaned_data, test_size=0.2)

cleaned_data = train

# check for encoded labels
cleaned_data['Label'].value_counts()

# """
# Shaping data for CNN
# make 3 seperate datasets for 3 feature labels
data_1 = cleaned_data[cleaned_data['Label'] == 0]
data_2 = cleaned_data[cleaned_data['Label'] == 1]
data_3 = cleaned_data[cleaned_data['Label'] == 2]
data_4 = cleaned_data[cleaned_data['Label'] == 3]
data_5 = cleaned_data[cleaned_data['Label'] == 4]

# Data Augmentation
from sklearn.utils import resample

data_1_resample = resample(data_1, n_samples=20000,
                           random_state=123, replace=True)
data_2_resample = resample(data_2, n_samples=20000,
                           random_state=123, replace=True)
data_3_resample = resample(data_3, n_samples=20000,
                           random_state=123, replace=True)
data_4_resample = resample(data_4, n_samples=20000,
                           random_state=123, replace=True)
data_5_resample = resample(data_5, n_samples=20000,
                           random_state=123, replace=True)

train_dataset = pd.concat([data_1_resample, data_2_resample, data_3_resample, data_4_resample, data_5_resample])
train_dataset.head(2)

# viewing the distribution of intrusion attacks in our dataset
plt.figure(figsize=(10, 8))
circle = plt.Circle((0, 0), 0.7, color='white')
plt.title('Intrusion Attack Type Distribution')
plt.pie(train_dataset['Label'].value_counts(), labels=['0-Benign', '1-FTP-BruteForce', '2-SSH-BruteForce', '3-Dos-GoldenEye', '4-Dos-SLowloris'], colors=['orange', 'green', 'blue', 'magenta', 'cyan'])
p = plt.gcf()
p.gca().add_artist(circle)
plt.show()

# Making X & Y Variables (CNN)

# test_dataset = train_dataset.sample(frac=0.15)
test_dataset = test
target_train = train_dataset['Label']
target_test = test_dataset['Label']
target_train.unique(), target_test.unique()
yy_test = target_test


y_train = to_categorical(target_train, num_classes=5)
y_test = to_categorical(target_test, num_classes=5)

train_dataset = train_dataset.drop(columns = ["Timestamp", "Protocol","PSH Flag Cnt","Init Fwd Win Byts","Flow Byts/s","Flow Pkts/s", "Label"], axis=1)
test_dataset = test_dataset.drop(columns = ["Timestamp", "Protocol","PSH Flag Cnt","Init Fwd Win Byts","Flow Byts/s","Flow Pkts/s", "Label"], axis=1)

# making train & test splits
X_train = train_dataset.iloc[:, :-1].values
X_test = test_dataset.iloc[:, :-1].values
X_test

print(X_train.shape)
print(X_test.shape)
print(y_train.shape)
print(y_test.shape)

# reshape the data for CNN
X_train = X_train.reshape(len(X_train), X_train.shape[1], 1)
X_test = X_test.reshape(len(X_test), X_test.shape[1], 1)
X_train.shape, X_test.shape

# making the deep learning function
def model():
    model = Sequential()
    model.add(Conv1D(filters=64, kernel_size=6, activation='relu',
                     padding='same', input_shape=(72, 1)))
    model.add(BatchNormalization())

    # adding a pooling layer
    model.add(MaxPooling1D(pool_size=(3), strides=2, padding='same'))

    model.add(Conv1D(filters=64, kernel_size=6, activation='relu',
                     padding='same', input_shape=(72, 1)))
    model.add(BatchNormalization())
    model.add(MaxPooling1D(pool_size=(3), strides=2, padding='same'))

    model.add(Conv1D(filters=64, kernel_size=6, activation='relu',
                     padding='same', input_shape=(72, 1)))
    model.add(BatchNormalization())
    model.add(MaxPooling1D(pool_size=(3), strides=2, padding='same'))
    model.add(Flatten())
    model.add(Dense(64, activation='relu'))
    model.add(Dense(64, activation='relu'))
    model.add(Dense(5, activation='softmax'))

    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    return model


def model_train(model, X_train, y_train, X_test, y_test):
    logger = CSVLogger('logs.csv', append=True)
    # his = model.fit(X_train, y_train, epochs=120, batch_size=256,
    #           validation_data=(X_test, y_test), callbacks=[logger])

    # Train the model
    his = model.fit(
        X_train,
        y_train,
        epochs=20,
        batch_size=64,
        shuffle=True,
        validation_split=0.15,
        callbacks=[logger,
            keras.callbacks.EarlyStopping(monitor="val_loss", patience=50, mode="min")
        ],
    )

    # check the model performance on test data
    scores = model.evaluate(X_test, y_test)
    print("%s: %.2f%%" % (model.metrics_names[1], scores[1] * 100))

    # check history of model
    history = his.history
    history.keys()

    epochs = range(1, len(history['loss']) + 1)
    acc = history['accuracy']
    loss = history['loss']
    val_acc = history['val_accuracy']
    val_loss = history['val_loss']

     # Save model
    print("\nSaving Trained Model")
    model.save("saved_models/cnn_model_baseline_ids2018_combined.h5")

    # visualize training and val accuracy
    plt.figure(figsize=(10, 5))
    plt.title('Training and Validation Accuracy (CNN)')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.plot(epochs, acc, label='accuracy')
    plt.plot(epochs, val_acc, label='val_acc')
    plt.legend()

    # visualize train and val loss
    plt.figure(figsize=(10, 5))
    plt.title('Training and Validation Loss(CNN)')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.plot(epochs, loss, label='loss', color='g')
    plt.plot(epochs, val_loss, label='val_loss', color='r')
    plt.legend()
    plt.show()

train_model = 0

if(train_model == 1):
    model = model()
    model_train(model, X_train, y_train, X_test, y_test)
    model.summary()

if(train_model == 0):
    print("\nLoading Trained Model")
    model = load_model("saved_models/cnn_model_baseline_ids2018_combined.h5")
    model.summary()

from sklearn.metrics import f1_score, precision_score, recall_score, confusion_matrix
y_pred1 = model.predict(X_test)
y_pred = np.argmax(y_pred1, axis=1)
y_pred = np.transpose(y_pred, axes=None)
# Print f1, precision, and recall scores

_, accuracy = model.evaluate(X_test, y_test)
print("\nEvaluation Metrics")
print("---------------------")
print("Accuracy = {} %" .format(accuracy*100))
print("Precision = {} %" .format(precision_score(yy_test, yy_test, average="macro")*100))
print("Recall = {} %" .format(recall_score(yy_test, y_pred, average="macro")*100))
print("F1 Score = {} %" .format(f1_score(yy_test, y_pred, average="macro")*100))
print("\n--- %s seconds ---" % (time.time() - start_time))

# Re-assign Class Names
my_dict = {0:'0-Benign', 1:'1-FTP-BruteForce', 2:'2-SSH-BruteForce', 3:'3-Dos-GoldenEye', 4:'4-Dos-SLowloris'}
yy_test= [my_dict[i] for i in yy_test]
y_pred= [my_dict[i] for i in y_pred]

# Print Classification Report
# target_names = ['Benign', 'Dos-GoldenEye', 'Dos-SLowloris', 'BF-FTP', 'BF-SSH'] # target values
print(classification_report(yy_test, y_pred))
# """