import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '1' 

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from keras.models import Sequential
from keras.layers import Dense, LSTM
from keras.optimizers import SGD
from keras.utils import to_categorical
from keras.callbacks import EarlyStopping
from keras.losses import BinaryCrossentropy
from sklearn.dummy import DummyClassifier
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score

from preprocessing import full_pipeline, stack_windows

data = pd.read_csv('data/cache.csv')
data.fillna(method='ffill', inplace=True)

scaled_data = full_pipeline.fit_transform(data)

# Split into training and testing sets
total_intervals, n_features = scaled_data.shape
train_len = int(total_intervals * 0.8)
train, test = scaled_data[:train_len], scaled_data[train_len:]

window_len = 15
units_ahead = 0 # 0 is the minimum valid value, representing the next unit in the future

X_train, y_train_pct_change = stack_windows(train, window_len, units_ahead, True)
X_test, y_test_pct_change = stack_windows(test, window_len, units_ahead, True)

y_train = to_categorical(y_train_pct_change > 0)
y_test = to_categorical(y_test_pct_change > 0)

# Create and fit the LSTM model
model = Sequential()
model.add(LSTM(128, activation='relu', input_shape=(window_len, n_features)))
model.add(Dense(2, activation = 'softmax'))

opt = SGD(learning_rate=0.01, momentum=0.0)

model.compile(loss='binary_crossentropy', optimizer=opt)

es = EarlyStopping(monitor='val_loss', mode='min', patience=10, verbose=1)
history = model.fit(X_train, y_train, epochs=100, batch_size=100, verbose=1,
                    validation_data=(X_test, y_test), callbacks=[es])

y_pred = model.predict(X_test)

# loss, accuracy = model.evaluate(X_test, y_test)
# print(loss, accuracy)

# Plot loss over time
fig1, ax1 = plt.subplots()

baseline = BinaryCrossentropy()(y_test, np.ones_like(y_test)*0.5)
ax1.plot(history.history['loss'], label='Train Loss')
ax1.plot(history.history['val_loss'], label='Validation Loss')
ax1.axhline(baseline, color='k', ls='--', label='Naive Loss')
ax1.set_xlabel('epochs')
ax1.set_ylabel('loss')
ax1.legend()

# Compare with DummyClassifier
dummy_classifiers = ['stratified', 'most_frequent', 'prior', 'uniform']
for i in dummy_classifiers:
        dummy_classifier = DummyClassifier(strategy = i).fit(X_train, y_train)
        s = dummy_classifier.score(X_test, y_test)
        print('Score for', i ,'classifier is:', s)



y_pred = np.argmax(y_pred, axis=1)
y_test = np.argmax(y_test, axis=1)

print(y_pred)
print(y_test)

# Calculate accuracy, precision and recall
print(confusion_matrix(y_test, y_pred))
print('Accuracy:', accuracy_score(y_test, y_pred))
print('Precision:', precision_score(y_test, y_pred))
print('Recall:', recall_score(y_test, y_pred))


price_scaler = full_pipeline.named_transformers_['price_pipeline'].named_steps['scaler']
y_test_pct_change = price_scaler.inverse_transform(y_test_pct_change[:, np.newaxis])

test_days = len(y_train) + np.arange(len(y_test))

trading = np.cumprod(y_test_pct_change.flatten() * y_pred + 1)
hold = np.cumprod(y_test_pct_change.flatten() + 1)

# Plot comparison of buy & hold vs trading
fig4, ax4 = plt.subplots()
ax4.plot(test_days, hold, 'k-', label='Hold')
ax4.plot(test_days, trading, 'r-', label='Trading')
ax4.legend()

plt.show()
