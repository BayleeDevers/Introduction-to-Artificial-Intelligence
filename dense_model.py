import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from keras.models import Sequential
from keras.layers import Dense
from keras.optimizers import SGD
from keras.utils import to_categorical
from sklearn.preprocessing import StandardScaler
from sklearn.dummy import DummyClassifier
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score

# Import complete data set
df = pd.read_csv('data/bitstampUSD_1-hour_data_2016-01-01_to_2020-11-11_filled_gaps.csv')
print(df.shape)

data = df.iloc[:, 1:8]
print(data)

x = data
opening = df.iloc[:, 1:2].values
Target = []
Target.append(0)
for i in range(len(opening) - 1):
    if opening[i] > opening[i + 1]:
        Target.append(1)
    else:
        Target.append(0)
y = Target
total_intervals, num_features = x.shape
print(num_features)
num_classes = len(np.unique(y))
print(num_classes)
y = to_categorical(y)

# Split into training and test sets
train_len = int(total_intervals * 0.8)
x_train, x_test = x[:train_len], x[train_len:]
y_train, y_test = y[:train_len], y[train_len:]
scaler = StandardScaler()
x_train = scaler.fit_transform(x_train)
x_test = scaler.transform(x_test)

# Create dense model
model = Sequential()

model.add(Dense(128, input_dim=num_features, activation = 'relu'))
model.add(Dense(64, activation = 'relu'))
model.add(Dense(32, activation = 'relu'))
model.add(Dense(16, activation = 'relu'))
model.add(Dense(8, activation = 'relu'))
model.add(Dense(4, activation = 'relu'))
model.add(Dense(num_classes, activation = 'softmax'))

model.compile(loss = 'categorical_crossentropy', optimizer = SGD(learning_rate = 0.001), metrics = ['accuracy'])

history = model.fit(x_train, y_train, epochs = 100)

# Evaluate performance
loss, accuracy = model.evaluate(x_test, y_test)

# Plot loss over time
plt.plot(history.history['loss'])
plt.xlabel('epochs')
plt.ylabel('loss')
plt.show()

# Plot accuracy over time
plt.plot(history.history['accuracy'])
plt.xlabel('epochs')
plt.ylabel('accuracy')
plt.show()

# Compare with DummyClassifier
dummy_classifiers = ['stratified', 'most_frequent', 'prior', 'uniform']
for i in dummy_classifiers:
        dummy_classifier = DummyClassifier(strategy = i).fit(x_train, y_train)
        s = dummy_classifier.score(x_train, y_train)
        print('Score for', i ,'classifier is:', s)

# Create confusion matrix
vals = model.predict(x_test)
for i in vals:
    for j in range(len(i)):
        if i[j]  == max(i):
            i[j] = 1
        else:
            i[j] = 0
y_pred = []

for i in vals:
    y_pred.append(np.argmax(i))

y_true = Target[len(x_train):]

model = confusion_matrix(y_true, y_pred)
print(model)

# Calculate accuracy, precision and recall
print('Accuracy:', accuracy_score(y_true, y_pred))
print('Precision:', precision_score(y_true, y_pred))
print('Recall:', recall_score(y_true, y_pred))
