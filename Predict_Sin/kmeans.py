import math
import numpy as np
import matplotlib.pyplot as plt


#  Lloyd's algorithm: Implmentation.
# Assume feature engineering has been done.
class KMeans:

    def __init__(self, k, X):
        self.pointMatrix = X
        self.k = int(k)
        pass

    def assign_means(self, num_trials, num_iterations, verbose=True):
        if verbose:
            print("evaluting means for k = {}...".format(self.k))
            plt.subplot(111)
            pass
        best_error = float("inf")  # Best error is positive infinity
        best_mu = []
        best_meanSet = []
        pointMatrix = self.pointMatrix
        k = self.k
        col = [0, 0, 1]

        for trial in range(num_trials):
            self.mu = self.init_mu()
            errors = []

            for i in range(num_iterations):
                # iteratively update the clusters and cluster means nunm_iterations times
                self.meanSet = self.update_clusters()
                self.mu = self.update_cluster_points()
                errors.append(self.calculate_total_error())
                pass

            # calculate error of current mu model
            model_error = self.calculate_total_error()

            if verbose:
                print("MSE for trial {} : {}".format(trial, model_error))
                #             plot convergence over iterations
                plt.plot(np.arange(0, num_iterations, 1), errors.copy(), c=col.copy())
                #             reset errors, iterate color
                col[1] += (1 / num_trials) * 0.9
                col[2] -= (1 / num_trials) * 0.9
                pass

            # always take the best model w/ minimum error
            if (model_error < best_error):
                best_error = model_error
                best_mu = self.mu
                best_meanSet = self.meanSet
            pass
        pass
        if verbose:
            plt.show()
            pass
        print("Saved best mu with mean squared error: {}".format(best_error))
        return best_mu

    def init_mu(self):
        k = self.k
        X = self.pointMatrix
        mu = np.zeros((k, X.shape[1]))
        #         print("generating mu of {} from x of {}".format(mu.shape, X.shape))
        for i in range(X.shape[1]):
            for j in range(k):
                index = int(np.random.ranf() * X.shape[0])
                mu[j] = X[index]
                pass
            pass
        return mu

    def update_clusters(self):
        pointMatrix = self.pointMatrix
        mu = self.mu
        k = self.k
        meanSet = [[] for i in range(k)]
        #   iterate over points
        for i in range(pointMatrix.shape[0]):
            minIndex = 0
            minDistance = np.linalg.norm(pointMatrix[i] - mu[minIndex])
            #     iterate over mu (mean points)
            for j in range(k):
                dist = np.linalg.norm(pointMatrix[i] - mu[j])
                #       pick j with the minimum distance from i
                if (dist < minDistance):
                    minDistance = dist
                    minIndex = j
                    pass
                pass
            #   Add point i to mu[j]'s cluster'
            meanSet[minIndex].append(pointMatrix[i])
            pass
        return meanSet

    def update_cluster_points(self):
        meanSet = self.meanSet
        mu = self.mu
        k = self.k
        #   iterate over mu
        for i in range(k):
            set_sum = np.zeros(mu[i].shape)
            #     iterate over mu[i]'s cluster'
            for j in range(len(meanSet[i])):
                # sum up all the positions of each point
                set_sum += meanSet[i][j]
                pass
            # update mu to the average of each point in mu's cluster
            if len(meanSet[i]) != 0:
                mu[i] = set_sum / len(meanSet[i])
                pass
            pass
        return mu

    def calculate_total_error(self):
        meanSet = self.meanSet
        mu = self.mu
        mserror = 0
        N = 0
        for i in range(self.k):
            for j in range(len(meanSet[i])):
                error = np.linalg.norm(meanSet[i][j] - mu[i])
                mserror += error * error
                N += 1
                pass
            pass
        return mserror / N

    pass
