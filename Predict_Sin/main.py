import math
import numpy as np
import matplotlib.pyplot as plt
from kmeans import KMeans


def generateSinPairs(N, valueRange):
    X = np.zeros((N, 1))
    Y = np.zeros((N, 1))
    for i in range(N):
        X[i][0] = (np.random.ranf() * valueRange - (0.5 * valueRange)) * math.pi
        Y[i][0] = math.sin(X[i][0])
        pass
    return X, Y


def predict(x, mu, precision_rate, w):

    A = np.zeros((1, mu.shape[0]))
    # write Radial Function e^ -sigma*(|x-mu|^2)
    for i in range(mu.shape[0]):
        # distance between point x and mu
        dist = np.linalg.norm(x - mu[i])
        # square distance, multiply by lambda
        exponent = (- precision_rate[i]) * dist * dist
        A[0][i] = math.exp(exponent)
        pass
    return np.dot(A, w)  # f(x)=∑(k=1) W * e−λ∥x−μ∥2


def get_loss(X, Y, precision_rate, mu, w):
    mse = 0
    count = 0
    for i in range(X.shape[0]):
        predicted = predict(X[i], mu, precision_rate, w)
        err = Y[i] - predicted
        mse += err * err
        count += 1
        pass
    res = mse / count
    return res


def regress_w(X, Y, mu, lambda_val): # Compute Matrix Weight (W)
    A = np.zeros((X.shape[0], mu.shape[0]))

    for i in range(X.shape[0]):
        for j in range(mu.shape[0]):
            # create vectors for x and mu
            _x = np.transpose([X[i]])
            _mu = np.transpose([mu[j]])
            # Take the distance between point x and mu
            dist = np.linalg.norm(_x - _mu)
            # square distance, multiply by lambda
            exponent = (-lambda_val[j]) * dist * dist
            A[i][j] = math.exp(exponent)
            pass
        pass

    # Compute Pseudo Inverse Matrix
    # Compute (Transpose(A)*A)
    transpose = np.transpose(A)
    ATA = np.dot(transpose, A)

    # Compute (ATA)^-1 (inverse of ATA)
    pseudoInv = np.linalg.inv(ATA)

    # Compute Desire Output Vector
    res = np.dot(transpose, Y)

    # Compute D.PseudoInverse Matrix (W)
    res = np.dot(pseudoInv, res)
    return res


def descend_lambda(x, y, mu, lambda_val, w, eta):
    # Predict the value of f(x)
    # 'eta' is the learning rate
    f = predict(x, mu, lambda_val, w)
    coefficient = - eta * (y - f)
    learning_vector = np.ones(mu.shape)
    for i in range(mu.shape[0]):
        dist = (x - mu[i]) ** 2
        exponent = -lambda_val[i] * dist
        learning_vector[i] = dist * w[i] * math.exp(exponent)
        pass
    new_lambda = lambda_val.copy()
    new_lambda += coefficient * learning_vector
    #print(new_lambda)
    return new_lambda


N = 100
(train_X, train_Y) = generateSinPairs(N, 3)

plt.subplot(111)
plt.scatter(train_X, train_Y)
plt.show()

# This block initializes our dataset and uses K-means to find the vector mu.
train_N = 100
# Error calculated from 100 values
test_N = 20
# Set K proportional to log size of training set
K = int(2 * math.log(train_N))

# Generate relevant (x, y) pairs
(train_X, train_Y) = generateSinPairs(train_N, 3)
(test_X, test_Y) = generateSinPairs(test_N, 3)

# Run K-means on training set.
print("Running K-means...")
print(K)
KMeans = KMeans(K, train_X)

mu = KMeans.assign_means(10, 15, verbose=False)
#print(mu)

# Start off lambda as 1 for all values
lambda_val = np.ones(mu.shape) * 1

# Set the learning rate and number of epochs
eta = 0.01
num_epochs = 20

# Set up
E_in_vector = []
E_out_vector = []
epoch_X = np.arange(0, num_epochs, 1)

print("Training model...")
for epoch in range(num_epochs):
    #     Regress to find w
    w = regress_w(train_X, train_Y, mu, lambda_val)
    #     use gradient decent through one epoch to find lambda
    for i in range(train_X.shape[0]):
        lambda_val = descend_lambda(train_X[i], train_Y[i], mu, lambda_val, w, eta)
        pass

    #   Get the error within the training set
    E_in = get_loss(train_X, train_Y, lambda_val, mu, w)
    E_in_vector.append(float(E_in))

    # Get the error outside the training set
    E_out = get_loss(test_X, test_Y, lambda_val, mu, w)
    E_out_vector.append(float(E_out))

    print('MSE for epoch {} was {}, In-sample error was {}'.format(epoch, E_out, E_in))
    pass

plt.subplot(111)
plt.plot(epoch_X, E_in_vector, c=[0, 0, 1], label="in-sample error")
plt.plot(epoch_X, E_out_vector, c=[0, 1, 0], label="out-of-sample error")
plt.legend(loc='best')
plt.show()

plot_X = np.arange(-4, 4, 0.1)
plot_Y = np.zeros(plot_X.shape)
plot_sin = np.zeros(plot_X.shape)
plt.subplot(111)
for i in range(plot_X.shape[0]):
    plot_Y[i] = predict(plot_X[i], mu, lambda_val, w)
    plot_sin[i] = math.sin(plot_X[i])
    pass

plt.plot(plot_X, plot_Y, c=[0, 0, 1], label="f(x)")
plt.plot(plot_X, plot_sin, c=[1, 0, 0], label="sin(x)")
plt.legend(loc="best")
plt.show()
