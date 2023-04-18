import numpy as np
from matplotlib import pyplot as plt
from sklearn.model_selection import train_test_split

data = np.loadtxt('dane04.txt')

x = data[:,[0]]
y = data[:,[1]]
plt.plot(x, y, 'b*') # original

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3)

# least squares method with Fisher observtion matrix
c_model1 = np.hstack([x, np.ones(x.shape)])
v_model1 = np.linalg.inv(c_model1.T @ c_model1) @ c_model1.T @ y
model_response = x * v_model1[0] + v_model1[1]
plt.plot(x, model_response, 'r-')

#verifying quality of the model 1 using training and testing sets

# approach: mean squared error
e_train = (y_train - (x_train * v_model1[0] + v_model1[1]))
error_mean_train_model1 = e_train.T @ e_train / len(e_train)
e_test = (y_test - (x_test * v_model1[0] + v_model1[1]))
error_mean_test_model1 = e_test.T @ e_test / len(e_test)

# proposing more complex model, this's polynomial regression
c_model2 = np.hstack([x**5, x**4, x**3, x**2, x, np.ones(x.shape)])
v_model2 = np.linalg.pinv(c_model2.T @ c_model2) @ c_model2.T @ y
model_response = x**5 * v_model2[0] + x**4 * v_model2[1] + x**3 * v_model2[2] + x**2 * v_model2[3] + x * v_model2[4] + v_model2[5]
plt.plot(x, model_response, 'g-')

#verifying quality of the model 2 using training and testing sets

e_train = (y_train - (x_train**5 * v_model2[0] + x_train**4 * v_model2[1] + x_train**3 * v_model2[2] + x_train**2 * v_model2[3] + x_train * v_model2[4] + v_model2[5]))
error_mean_train_model2 = e_train.T @ e_train / len(e_train)
e_test = (y_test - (x_test**5 * v_model2[0] + x_test**4 * v_model2[1] + x_test**3 * v_model2[2] + x_test**2 * v_model2[3] + x_test * v_model2[4] + v_model2[5]))
error_mean_test_model2 = e_test.T @ e_test / len(e_test)

print("Model1: train:", str(error_mean_train_model1), "test:", str(error_mean_test_model1))
print("Model2: train:", str(error_mean_train_model2), "test:", str(error_mean_test_model2))

plt.show()
