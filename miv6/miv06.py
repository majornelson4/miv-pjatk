from keras import Sequential, layers
from keras.datasets import cifar10
import matplotlib.pyplot as plt
import numpy as np
from sklearn.model_selection import train_test_split

(X_train, y_train), (X_test, y_test) = cifar10.load_data()

# task1
X = np.concatenate((X_train, X_test))
Y = np.concatenate((y_train, y_test))
X_train, X_test, y_train, y_test = train_test_split(X, Y, train_size=0.3)  # add random_state=1 for constant values.
# normalizing inputs
X_train = X_train.astype('float32') / 255
X_test = X_test.astype('float32') / 255
y_train = y_train.reshape(-1, )
y_test = y_test.reshape(-1, )

vehicles_only = [0, 1, 8, 9]
y_train = np.array(list(map(int, [i in vehicles_only for i in y_train])))
y_test = np.array(list(map(int, [i in vehicles_only for i in y_test])))

# task2
def animal_vehicle(X, y, index):
    classes_task2 = ["animal", "vehicle"]
    plt.imshow(X[index])
    plt.xlabel(classes_task2[y[index]])
    plt.show()


# animal_vehicle(X_train, y_train, 7)

# task3,4
model = Sequential([
    layers.Conv2D(32, (3, 3), activation='relu', input_shape=(32, 32, 3)),
    layers.MaxPooling2D((2, 2)),
    layers.Flatten(),
    layers.Dense(64, activation='relu'),
    layers.Dense(2, activation='softmax')  # cifar10 has 10 classes
])

model2 = Sequential([
    layers.Conv2D(128, (3, 3), activation='relu', input_shape=(32, 32, 3)),
    layers.MaxPooling2D((2, 2)),
    layers.Conv2D(64, (3, 3), activation='relu', input_shape=(32, 32, 3)),
    layers.MaxPooling2D((2, 2)),
    layers.Flatten(),
    layers.Dense(64, activation='relu'),
    layers.Dense(2, activation='softmax')
])

model3 = Sequential([
    layers.Conv2D(128, (3, 3), activation='relu', input_shape=(32, 32, 3)),
    layers.MaxPooling2D((2, 2)),
    layers.Conv2D(64, (3, 3), activation='relu', input_shape=(32, 32, 3)),
    layers.MaxPooling2D((2, 2)),
    layers.Conv2D(32, (3, 3), activation='relu', input_shape=(32, 32, 3)),
    layers.MaxPooling2D((2, 2)),
    layers.Flatten(),
    layers.Dense(64, activation='relu'),
    layers.Dense(2, activation='softmax')
])

model.summary()
model.compile(optimizer='rmsprop',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])
model.fit(X_train, y_train, epochs=5, batch_size=64)
test_loss, test_accuracy = model.evaluate(X_test, y_test)

model2.summary()
model2.compile(optimizer='rmsprop',
               loss='sparse_categorical_crossentropy',
               metrics=['accuracy'])
model2.fit(X_train, y_train, epochs=10, batch_size=64)
test_loss2, test_accuracy2 = model2.evaluate(X_test, y_test)

model3.summary()
model3.compile(optimizer='rmsprop',
               loss='sparse_categorical_crossentropy',
               metrics=['accuracy'])
model3.fit(X_train, y_train, epochs=10, batch_size=100)
test_loss3, test_accuracy3 = model3.evaluate(X_test, y_test)

print("model1 1 conv layer: (test accuracy)= %f " % (test_accuracy * 100))
print("model2 2 conv layers: (test accuracy)= %f " % (test_accuracy2 * 100))
print("model3 3 conv layers: (test accuracy)= %f " % (test_accuracy3 * 100))
