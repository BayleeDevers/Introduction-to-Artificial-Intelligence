import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '1' 

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from keras.models import Sequential
from keras.layers import Dense, LSTM, Dropout
from keras.optimizers import Adam, SGD
from keras.callbacks import EarlyStopping
from sklearn.metrics import mean_squared_error

from preprocessing import full_pipeline, stack_windows
from attention import Attention

data = pd.read_csv('data/cache.csv')
data.fillna(method='ffill', inplace=True)

scaled_data = full_pipeline.fit_transform(data)

# Split into training and testing sets
total_intervals, n_features = scaled_data.shape
train_len = int(total_intervals * 0.8)
train, test = scaled_data[:train_len], scaled_data[train_len:]

window_len = 15
units_ahead = 0 # 0 is the minimum valid value, representing the next unit in the future

X_train, y_train = stack_windows(train, window_len, units_ahead, True)
X_test, y_test = stack_windows(test, window_len, units_ahead, True)

# print(X_train, y_train)

# Create and fit the LSTM model
model = Sequential()
model.add(LSTM(128, activation='relu', input_shape=(window_len, n_features), return_sequences=True))
model.add(Attention(64))
model.add(Dropout(0.2))
model.add(Dense(1))

opt = SGD(learning_rate=0.01, momentum=0.0)
model.compile(loss='mean_squared_error', optimizer=opt)

es = EarlyStopping(monitor='val_loss', mode='min', patience=10, verbose=1)
history = model.fit(X_train, y_train, epochs=100, batch_size=100, verbose=1,
                    validation_data=(X_test, y_test), callbacks=[es])


y_pred = model.predict(X_test)
price_scaler = full_pipeline.named_transformers_['price_pipeline'].named_steps['scaler']
y_naive = price_scaler.transform(np.zeros(len(y_test))[:, np.newaxis])

baseline = mean_squared_error(y_test, y_naive)
scaled_mse = mean_squared_error(y_test, y_pred)

y_naive = price_scaler.inverse_transform(y_naive)
y_train = price_scaler.inverse_transform(y_train[:, np.newaxis])
y_test = price_scaler.inverse_transform(y_test[:, np.newaxis])
y_pred = price_scaler.inverse_transform(y_pred)
print(y_naive)
print(y_train)
print(y_test)
print(y_pred)

print('------------------------------------------')
print(f'strategy | {"scaled mse":>7} | rmse')
print(f'LSTM + A | {scaled_mse:>7.4e} | {mean_squared_error(y_test, y_pred, squared=False):.4e}')
print(f'Naive    | {baseline:>7.4e} | {mean_squared_error(y_test, y_naive, squared=False):.4e}')

# Plot loss over time
fig1, ax1 = plt.subplots()
ax1.plot(history.history['loss'], label='Train Loss')
ax1.plot(history.history['val_loss'], label='Validation Loss')
ax1.axhline(baseline, color='k', ls='--', label='Naive Loss')
ax1.set_xlabel('epochs')
ax1.set_ylabel('loss')
ax1.set_yscale('log')
ax1.legend()

test_days = len(y_train) + np.arange(len(y_test))

# Plot timeseries of true vs predicted prices
fig2, ax2 = plt.subplots()
ax2.plot(y_train, 'k-', label='Training Set')
ax2.plot(test_days, y_test.flatten(), 'k-', alpha=0.5, label='Test Set Labels')
ax2.plot(test_days, y_pred.flatten(), 'r--', label='Test Set Predictions')
ax2.plot(test_days, y_naive.flatten(), 'b--', label='Test Set Naive Predictions')
ax2.legend()

# Plot histogram of true vs predicted prices
fig3, ax3 = plt.subplots()
ax3.hist(y_test.flatten(), color='k', alpha=0.7, label='Test Set Labels', bins=np.linspace(-0.02, 0.02, 200))
ax3.hist(y_pred.flatten(), color='r', alpha=0.7, label='Test Set Predictions', bins=np.linspace(-0.02, 0.02, 200))
ax3.axvline(0, color='b', linestyle='--', alpha=0.7, label='Test Set Naive Predictions')
ax3.axvline(np.mean(y_train), color='m', linestyle='--', alpha=0.7, label='Train Set Mean')
ax3.legend()
ax3.set_xlabel('% Change')

trading = np.cumprod(y_test.flatten() * (y_pred.flatten() > 0) + 1)
hold = np.cumprod(y_test.flatten() + 1)

# Plot comparison of buy & hold vs trading
fig4, ax4 = plt.subplots()
ax4.plot(test_days, hold, 'k-', label='Hold')
ax4.plot(test_days, trading, 'r-', label='Trading')
ax4.legend()

plt.show()
