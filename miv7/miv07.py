import numpy as np
import matplotlib.pyplot as plt
from keras import layers

from keras.models import Sequential
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

a = np.loadtxt('danet.txt')

x = a[0:40, [1, 2, 3]]
y = a[0:40, [0]]
# task 1
X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.3)

# task 2
c = np.hstack([x])
v = np.linalg.inv(c.T @ c) @ c.T @ y

plt.plot(v[0] * x[:, 0] + v[1] * x[:, 1] + v[2] * x[:, 2], 'b-')  # ar model
plt.plot(y, 'r-')  # today
plt.plot(x[:, 0], 'y-')  # 1 step back
plt.show()

# simple model
e_train = (y_train - np.reshape(X_train[:, 0], (28, 1)))
error_mean_train_simple = e_train.T @ e_train / len(e_train)

y_train = np.reshape(y_train, (28, 1))
e_train = (y_train - (np.reshape(X_train[:, 0], (28, 1)) * v[0] + np.reshape(X_train[:, 1], (28, 1)) * v[1] + np.reshape(X_train[:, 2], (28, 1)) * v[2]))
error_mean_train = e_train.T @ e_train / len(e_train)

y_test = np.reshape(y_test, (12, 1))
e_test = (y_test - (np.reshape(X_test[:, 0], (12, 1)) * v[0] + np.reshape(X_test[:, 1], (12, 1)) * v[1] + np.reshape(
    X_test[:, 2], (12, 1)) * v[2]))
error_mean_test = e_test.T @ e_test / len(e_test)

# task3
X_train = StandardScaler().fit_transform(X_train)
y_train = StandardScaler().fit_transform(y_train)

X_test = StandardScaler().fit_transform(X_test)
y_test = StandardScaler().fit_transform(y_test)
model = Sequential([
    layers.LSTM(100, input_shape=(3, 1), return_sequences=True),
    layers.Dropout(0.2),
    layers.LSTM(50),
    layers.Dense(1)
])
model.compile(loss='mean_squared_error', optimizer='adam')
model.fit(X_train, y_train, epochs=100, batch_size=1, verbose=2, validation_data=(X_test, y_test))
y1 = model.predict(x)

test_acc = model.evaluate(X_test, y_test, verbose=0)
print('simple model:', error_mean_train_simple)
print('recurrent nn, error prediction', test_acc)
print('ar model, prediction on train: ', error_mean_train, ' prediction on test:', error_mean_test)

