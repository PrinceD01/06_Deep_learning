import numpy as np
from scipy.spatial.distance import cdist
from math import sqrt, pi
from numpy import exp
from scipy.special import erf
from scipy import optimize


def disagreement(x):
    return 0.5*(1.0-(erf(x*(1.0/sqrt(2.0))))**2)


def d_disagreement(x):
    return -(sqrt(2.0/pi))*erf(x*(1.0/sqrt(2.0)))*exp(-0.5*x**2)


def jointError(x):
    return 0.25*(1.0-erf(x*(1.0/sqrt(2.0))))**2


def d_jointError(x):
    return -(1.0/sqrt(2*pi))*exp(-0.5*x**2)*(1.0-erf(x*(1.0/sqrt(2.0))))


class dalc:
    def __init__(self, B=1.0, C=1.0, kernel="linear", gamma=1):
        self.B = B
        self.C = C
        self.kernel = kernel
        self.gamma = gamma

    def getKernel(self, X1, X2):
        if self.kernel == "linear":
            return np.dot(X1, X2.T)
        elif self.kernel == "rbf":
            return np.exp(-self.gamma*cdist(X1, X2, 'sqeuclidean'))
        else:
            raise Exception("Unknown kernel.")

    def decision_function(self, X):
        return np.dot(self.getKernel(self.X1, X).T, self.alpha_vector)

    def predict(self, X):
        pred = self.decision_function(X)
        yPred = np.full(X.shape[0], self.positiveLabel)
        yPred[pred < 0] = self.negativeLabel
        return yPred

    def fit(self, source_X, source_y, target_X):
        # Assuming there is two labels in y. Convert them in -1 and 1 labels.
        labels = sorted(np.unique(source_y))
        self.negativeLabel, self.positiveLabel = labels[0], labels[1]
        newY = np.ones(source_X.shape[0])  # Set all labels at 1
        newY[source_y == labels[0]] = -1  # except the smallest label in y at -1.
        self.X1 = np.vstack((source_X, target_X))
        label = np.hstack((newY, np.zeros(len(target_X))))
        self.kernel_matrix = self.getKernel(self.X1, self.X1)
        self.target_mask = np.array(label == 0, dtype=int)
        self.source_mask = np.array(label != 0, dtype=int)
        self.margin_factor = ((label + self.target_mask) /
                              np.sqrt(np.diag(self.kernel_matrix)))
        self.alpha_vector, _, _ = optimize.fmin_l_bfgs_b(
                                      func=self.loss_grad, x0=label/len(label))

    def loss_grad(self, alpha_vector):
        ker = np.dot(self.kernel_matrix, alpha_vector)
        margin_vector = ker * self.margin_factor
        l_source = (jointError(margin_vector) * self.source_mask).sum()
        l_target = (disagreement(margin_vector) * self.target_mask).sum()
        d_source = (d_jointError(margin_vector) * self.margin_factor *
                    self.source_mask).dot(self.kernel_matrix)
        d_target = (d_disagreement(margin_vector) * self.margin_factor *
                    self.target_mask).dot(self.kernel_matrix)
        KL = np.dot(ker, alpha_vector) / 2
        loss = l_source/self.C + l_target/self.B + KL/(self.B*self.C)
        grad = d_source/self.C + d_target/self.B + ker/(self.B*self.C)
        return loss, grad
