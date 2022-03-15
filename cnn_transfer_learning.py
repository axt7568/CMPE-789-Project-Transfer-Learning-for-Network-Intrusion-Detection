# Setup
# import os
# os.environ["CUDA_VISIBLE_DEVICES"] = '-1'
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



# Track Elapsed Time
start_time = time.time()

# Load Data
network_data = pd.read_csv('C:/Users/Arjun/PycharmProjects/CMPE-789/data/data_in_csv/02-15-2018.csv')

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
# %%time
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
cleaned_data['Label']= label_encoder.fit_transform(cleaned_data['Label'])
cleaned_data['Label'].unique()

# check for encoded labels
cleaned_data['Label'].value_counts()

# Shaping data for CNN
# make 3 seperate datasets for 3 feature labels
data_1 = cleaned_data[cleaned_data['Label'] == 0]
data_2 = cleaned_data[cleaned_data['Label'] == 1]
data_3 = cleaned_data[cleaned_data['Label'] == 2]

# make benign feature
y_1 = np.zeros(data_1.shape[0])
y_benign = pd.DataFrame(y_1)

# make bruteforce feature
y_2 = np.ones(data_2.shape[0])
y_bf = pd.DataFrame(y_2)

# make bruteforceSSH feature
y_3 = np.full(data_3.shape[0], 2)
y_ssh = pd.DataFrame(y_3)

# merging the original dataframe
X = pd.concat([data_1, data_2, data_3], sort=True)
y = pd.concat([y_benign, y_bf, y_ssh], sort=True)

y_1, y_2, y_3

print(X.shape)
print(y.shape)

# checking if there are some null values in data
X.isnull().sum().to_numpy()

# Data Augmentation
from sklearn.utils import resample

data_1_resample = resample(data_1, n_samples=20000,
                           random_state=123, replace=True)
data_2_resample = resample(data_2, n_samples=20000,
                           random_state=123, replace=True)
data_3_resample = resample(data_3, n_samples=20000,
                           random_state=123, replace=True)

train_dataset = pd.concat([data_1_resample, data_2_resample, data_3_resample])
train_dataset.head(2)

# viewing the distribution of intrusion attacks in our dataset
plt.figure(figsize=(10, 8))
circle = plt.Circle((0, 0), 0.7, color='white')
plt.title('Intrusion Attack Type Distribution')
plt.pie(train_dataset['Label'].value_counts(), labels=['Benign', 'DoS attacks-GoldenEye', 'DoS attacks-Slowloris'], colors=['blue', 'magenta', 'cyan'])
p = plt.gcf()
p.gca().add_artist(circle)
plt.show()

# Making X & Y Variables (CNN)

test_dataset = train_dataset.sample(frac=0.15)
target_train = train_dataset['Label']
target_test = test_dataset['Label']
target_train.unique(), target_test.unique()
yy_test = target_test


y_train = to_categorical(target_train, num_classes=3)
y_test = to_categorical(target_test, num_classes=3)

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

# Load Trained Model
print("\nLoading Trained Model")
model = load_model("C:/Users/Arjun/PycharmProjects/CMPE-789/models/results/saved_models/cnn_model_baseline_3_14_10_59_pm.h5")
model.summary()

# Transfer Learning
# Feature Extraction
# Freeze the convolutional base
# Freeze all the layers
for layer in model.layers[:6]:
    layer.trainable = False

# Check the trainable status of the individual layers
for layer in model.layers:
    print(layer, layer.trainable)

# Compiler model
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
model.summary()

logger = CSVLogger('logs_transfer_learning.csv', append=True)


# Train the model
his = model.fit(
    X_train,
    y_train,
    epochs=5,
    batch_size=512,
    shuffle=True,
    validation_split=0.15,
    callbacks=[logger,
        keras.callbacks.EarlyStopping(monitor="val_loss", patience=5, mode="min")
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
model.save("C:/Users/Arjun/PycharmProjects/CMPE-789/models/results/saved_models/transfer_learning_model_3_15_12_19_pm.h5")

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

from sklearn.metrics import f1_score, precision_score, recall_score, confusion_matrix
y_pred1 = model.predict(X_test)
y_pred = np.argmax(y_pred1, axis=1)
y_pred = np.transpose(y_pred, axes=None)
# Print f1, precision, and recall scores

_, accuracy = model.evaluate(X_test, y_test)
print("\nEvaluation Metrics")
print("---------------------")
print("Accuracy = {} %" .format(accuracy*100))
print("Precision = {} %" .format(precision_score(yy_test, y_pred, average="macro")*100))
print("Recall = {} %" .format(recall_score(yy_test, y_pred, average="macro")*100))
print("F1 Score = {} %" .format(f1_score(yy_test, y_pred, average="macro")*100))
print("\n--- %s seconds ---" % (time.time() - start_time))




















