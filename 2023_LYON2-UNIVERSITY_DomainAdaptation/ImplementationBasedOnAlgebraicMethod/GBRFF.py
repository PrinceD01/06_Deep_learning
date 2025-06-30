import numpy as np
from scipy import optimize
from scipy.optimize import fminbound


class gbrff(object):
    def __init__(self, gamma=0.1, Lambda=0, T=100, randomState=np.random):
        self.T = T
        self.randomState = randomState
        self.Lambda = Lambda
        self.gamma = gamma

    def loss_grad(self, omega):
        dots = np.dot(omega, self.XT) - self.b
        dott = np.dot(omega, self.XtT) - self.b
        self.yTildePred = np.cos(dots)
        self.yTildePred_t = np.cos(dott)
        v0 = np.exp(self.yTildeN*self.yTildePred)
        return ((1/self.n_s)*np.sum(v0) + self.Lambda*(
                                    (1/self.n_s)*np.sum(self.yTildePred)-
                                    (1/self.n_t)*np.sum(self.yTildePred_t))**2,
                (1/self.n_s)*(self.yTilde*v0*np.sin(dots)).dot(
                    self.X) + self.Lambda*2*(
                                (1/self.n_s)*np.sum(self.yTildePred) -
                                (1/self.n_t)*np.sum(self.yTildePred_t)) * (
                                (1/self.n_s)*np.sin(dots).dot(self.X) -
                                (1/self.n_t)*np.sin(dott).dot(self.Xt)))

    def fit(self, y, Xs, Xt):
        # Assuming there is two labels in y. Convert them in -1 and 1 labels.
        labels = sorted(np.unique(y))
        self.negativeLabel, self.positiveLabel = labels[0], labels[1]
        newY = np.ones(Xs.shape[0])  # Set all labels at 1
        newY[y == labels[0]] = -1  # except the smallest label in y at -1.
        y = newY
        self.n_s, d = Xs.shape
        self.n_t = Xt.shape[0]
        meanY = np.mean(y)
        self.initPred = 0.5*np.log((1+meanY)/(1-meanY))
        curPred = np.full(self.n_s, self.initPred)
        pi2 = np.pi*2
        self.omegas = np.empty((self.T, d))
        self.alphas = np.empty(self.T)
        self.xts = np.empty(self.T)
        inits = self.randomState.randn(self.T, d)*(2*self.gamma)**0.5
        self.X = Xs
        self.Xt = Xt
        self.XT = Xs.T
        self.XtT = Xt.T
        for t in range(self.T):
            init = inits[t]
            wx_s = init.dot(self.XT)
            wx_t = init.dot(self.XtT)
            w = np.exp(-y*curPred)
            self.yTilde = y*w
            self.yTildeN = -self.yTilde
            self.b = pi2*fminbound(lambda n: (1/self.n_s)*np.sum(np.exp(
                       self.yTildeN*np.cos(pi2*n - wx_s))) + self.Lambda*(
                                 (1/self.n_s)*np.sum(np.cos(pi2*n - wx_s)) -
                                 (1/self.n_t)*np.sum(np.cos(pi2*n - wx_t)))**2,
                                   -0.5, 0.5, xtol=1e-2)
            self.xts[t] = self.b
            self.omegas[t], _, _ = optimize.fmin_l_bfgs_b(
                                      func=self.loss_grad, x0=init, maxiter=10)
            vi = (y*self.yTildePred).dot(w)
            vj = np.sum(w)
            alpha = 0.5*np.log((vj+vi)/(vj-vi))
            self.alphas[t] = alpha
            curPred = self.initPred+(self.alphas[:t]/np.sum(np.abs(
                        self.alphas[0:t]))).dot(np.cos(
                            self.xts[:t, None]-self.omegas[:t,:].dot(self.XT)))

    def predict(self, X):
        pred = self.initPred+(self.alphas/np.sum(np.abs(self.alphas))).dot(
                                np.cos(self.xts[:, None]-self.omegas.dot(X.T)))
        # Then convert back the labels -1 and 1 to the labels given in fit
        yPred = np.full(X.shape[0], self.positiveLabel)
        yPred[pred < 0] = self.negativeLabel
        return yPred
