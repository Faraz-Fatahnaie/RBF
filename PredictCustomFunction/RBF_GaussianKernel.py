import numpy as np
import math
import matplotlib.pyplot as plt


def generateSinPairs(valueRange, N=100):
    X = np.zeros((N, 1))
    Y = np.zeros((N, 1))
    for i in range(N):
        X[i][0] = (np.random.ranf() * valueRange - (0.5 * valueRange)) * math.pi
        # Y[i][0] = X[i][0] * X[i][0] * abs(math.sin(X[i][0]*math.pi))
        Y[i][0] = math.sin(X[i][0])
        pass
    return X, Y


def generateFunctionPairs(valueRange, N=50):
    X = np.zeros((N, 1))
    Y = np.zeros((N, 1))
    for i in range(N):
        X[i][0] = (np.random.ranf() * valueRange - (0.5 * valueRange)) * math.pi
        Y[i][0] = X[i][0] * X[i][0] * abs(math.sin(X[i][0]*math.pi))
        pass
    return X, Y


class Kmeans:
    def __init__(self, k, data, no_iter=100, optimized=True):
        self.k = k
        total_loss = []
        # Random initialization
        cluster_centers = np.random.random([k, data.shape[-1]])
        # Calculate Distance of each point to every cluster centers.
        dist_vec = np.zeros([k, data.shape[0]])
        cur = 0
        while cur < no_iter:
            for idx, _center in enumerate(cluster_centers):
                dist_vec[idx] = np.sum(np.square(np.subtract(np.broadcast_to(_center, data.shape), data)),axis=1)

            # Determine Argmin center
            labels = np.argmin(dist_vec, axis=0)
            loss = 0
            for idx in range(k):
                # Check Cluster Balance
                if data[labels == idx].shape[0] < 2:  # Degrees should be greater than 0
                    cluster_centers = np.random.random([k, data.shape[-1]])
                    cur = -1
                    break

                # Calculate Loss J
                loss += np.sum(dist_vec[idx][labels == idx])

                # Update cluster centers
                cluster_centers[idx] = np.average(data[labels == idx], axis=0)  # dim 784

            if cur >= 0:
                print('Iterations {}\t loss {} '.format(cur, loss))
                total_loss.append(loss)
            if optimized and cur > 1 and (total_loss[-1] == total_loss[-2]):
                break
            cur += 1

        self.centers, self.labels, self.total_loss = cluster_centers, labels, total_loss

    def predict(self, data):
        dist_vec = np.zeros([self.k, data.shape[0]])
        for idx, _center in enumerate(self.centers):
            dist_vec[idx] = np.sum(np.square(np.subtract(np.broadcast_to(_center, data.shape), data)), axis=1)
        return np.argmin(dist_vec, axis=0)


class RBFNetwork:
    def __init__(self, no_basis, no_labels, learning_rate=0.01):
        self.no_basis = no_basis
        self.no_labels = no_labels
        self.mu, self.sigma = np.zeros([0]), np.zeros([0])
        self.weights = np.random.normal(0, 0.05, [no_basis + 1, no_labels])  # including bias 1
        self.output = np.zeros([0])
        self.loss = 0

    def train(self, data, labels):
        self._back_propagate(data, labels)
        self.loss = self.get_loss(data, labels)
        print("RBF Loss(MSE): {:.4f}".format(self.loss))

    def predict(self, data):
        self._propagate(data)
        return self.output

    def get_loss(self, data, labels):
        # Return: MSE loss
        return np.mean(np.square(self.predict(data) - labels))

    def test(self, data, labels, threshold=None):
        if threshold:
            return np.mean((self.predict(data) > threshold).astype(np.int16) == labels)
        return self.get_loss(data, labels)

    def _propagate(self, data):
        # initial propagate
        if not self.mu.any():
            # Calculate mu and sigma using Kmeans
            self.mu, self.sigma = self._calculate_param(self.no_basis, data)
        self.basis = np.zeros([data.shape[0], self.no_basis + 1])
        self.basis[:, 0] = np.ones([data.shape[0]])  # for Bias
        # apply RBF kernels
        for i in range(data.shape[0]):
            self.basis[i, 1:] = self._gaussian_kernels(data[i], self.mu, self.sigma)
        # output
        self.output = np.dot(self.basis, self.weights)

    def _back_propagate(self, data, labels):
        self._propagate(data)
        # Calculate Weight Matrix Using Pseudo Inverse
        self.weights = np.dot(np.dot(np.linalg.pinv(np.dot(self.basis.T, self.basis)), self.basis.T), labels)

    def _calculate_param(self, k, data):
        # Calculate means & variances using K-Means Clustering
        print("Kmeans Training")
        kmeans = Kmeans(k, data)
        cluster_means, cluster_vars = [], []
        for i in range(k):
            cluster_elements = data[kmeans.predict(data) == i]
            cluster_means.append(np.mean(cluster_elements, axis=0))
            if data.ndim == 1 or data.shape[-1] == 1:  # univariate
                cluster_vars.append(np.var(cluster_elements))
            else:  # multivariate
                cluster_vars.append(np.linalg.pinv(np.cov(cluster_elements.T)))
        print("Kmeans Training Complete")
        return np.array(cluster_means), np.array(cluster_vars)

    def _gaussian_kernels(self, x, mu, sigma):
        if mu.ndim == 1:
            hidden_nodes = np.zeros(mu.shape[0])
            for j in range(mu.shape[0]):  # (k,dim)
                return np.exp(-np.divide(np.square(x - mu[j]), 2 * np.square(sigma[j])))
            return hidden_nodes
        else:
            hidden_nodes = np.zeros(mu.shape[0])
            for j in range(mu.shape[0]):  # (k,dim)
                # (X-u).T*Sigma*(X-u)
                hidden_nodes[j] = np.exp(-np.divide(np.dot(np.dot((x - mu[j]).T, sigma[j]), x - mu[j]), 2))
            return hidden_nodes


(train_X, train_Y) = generateFunctionPairs(2, N=300)
data = np.concatenate((train_X, train_Y), axis=-1)
plt.subplot(111)
plt.scatter(train_X, train_Y)
plt.show()

rbf = RBFNetwork(no_basis=10, no_labels=1)
rbf.train(train_X, train_Y)
pred = rbf.predict(train_X)

plt.scatter(train_X, train_Y, c=[0, 0, 1], label="custom_function(x)")
plt.scatter(train_X, pred, c=[1, 0, 0], label="RBF_prediction(x)")
plt.legend(loc="best")
plt.show()
