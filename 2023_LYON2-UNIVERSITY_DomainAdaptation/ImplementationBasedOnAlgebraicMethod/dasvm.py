import numpy as np
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import KFold
from sklearn.svm import SVC
import math
import time


def listP(dic):  # Create grid of parameters given parameters ranges
    params = list(dic.keys())
    listParam = [{params[0]: value} for value in dic[params[0]]]
    for i in range(1, len(params)):
        newListParam = []
        currentParamName = params[i]
        currentParamRange = dic[currentParamName]
        for previousParam in listParam:
            for value in currentParamRange:
                newParam = previousParam.copy()
                newParam[currentParamName] = value
                newListParam.append(newParam)
        listParam = newListParam.copy()
    return listParam


class dasvm(object):
    # Parameter tuning:
    #  - C (of SVM) and sigma (of Gaussian kernel) tuned using source only
    #  - Cstar, rho and gamma tuned via reverse validation
    #  - tau fixed to 0.5, beta fixed to 3e-2
    #  - the parameter Cstar must be smaller than tau*C
    def __init__(self):
        self.beta = 3e-2
        self.tau = 0.5

    def fit(self, Xs, ys, Xt):
        # Initialize C and sigma as the best fitting the source
        parametersSVM = {'C': [1, 0.1, 10, 0.01, 100],
                         'gamma': [2**i/Xs.shape[1]
                                   for i in [0, -1, 1, -2, 2]]}
        np.random.seed(1)  # for deterministic GridSearchCV folds construction
        grid = GridSearchCV(estimator=SVC(kernel="rbf", random_state=1),
                            param_grid=parametersSVM)
        grid.fit(Xs, ys)
        self.C = grid.best_params_["C"]
        self.sigma = grid.best_params_["gamma"]
        self.CstarMax = self.tau*self.C
        self.listParams = listP({"rho": [math.ceil(Xt.shape[0]/v)
                                         for v in [100, 50, 20]],
                                 "gamma": [2, 3, 5],
                                 "Cstar": [self.CstarMax/v
                                           for v in [1, 5, 10]]})
        self.reverseValidation(Xs, ys, Xt)
        self.fitReverseValidation(self.listParams[self.bestIdx], Xs, ys, Xt)

    def reverseValidation(self, Xs, ys, Xt):
        perfs = []
        labels = sorted(np.unique(ys))
        for i, p in enumerate(self.listParams):
            t1 = time.time()
            self.fitReverseValidation(p, Xs, ys, Xt)
            reverseLabels = self.predict(Xt)
            self.fitReverseValidation(p, Xt, reverseLabels, Xs, labels)
            pred = self.predict(Xs)
            perf = 100*float(sum(pred == ys)) / len(pred)
            perfs.append(perf)
            t2 = time.time()
            print("{:5.2f}".format(perf), p, i+1, "/", len(self.listParams),
                  "in {:6.2f}s".format(t2-t1))
        self.bestIdx = np.argmax(perfs)

    def fitReverseValidation(self, params, Xs, ys, Xt, labels=None):
        # Hackish piece of code.Sometimes, during the reverse validation, the
        # set of reverse labels contain only one class; But as it is required
        # to have at least two classes to fit the SVM, just swap a label
        if labels is not None:
            if labels[0] not in ys:
                ys[0] = labels[0]
            elif labels[1] not in ys:
                ys[0] = labels[1]
        self.Cstar = params["Cstar"]
        self.rho = params["rho"]
        self.gamma = params["gamma"]
        self.ceil = math.ceil(self.beta*Xt.shape[0])
        JsemilabeledSets = []
        for _ in range(self.gamma):
            JsemilabeledSets.append([np.array([]).astype(np.int),  # idx up
                                     np.array([]).astype(np.int)])  # idx low
        idxsSource = np.arange(Xs.shape[0])  # idx source remaining
        i = 0
        lambdda = 0
        delta = 0
        Si = np.array([])
        while True:
            # First, build the training set from source and target semilabeled.
            # Initialize training set with the remaining source examples
            # having a weight Ci that depends on the iteration
            Ci = max(((self.Cstar-self.C)/self.gamma**2)*i**2+self.C,
                     self.Cstar)  # weight of remaining source samples
            sampleWeights = np.array([Ci] * idxsSource.shape[0])
            trainingX = Xs[idxsSource]
            trainingy = ys[idxsSource]
            # Append to the training set the semilabeled target samples with
            # their corresponding weight Cstaru that depends on which subset
            # they are between u=1 (small confidence i.e. small value of
            # Cstaru) and u=gamma (large confidence i.e. large value of Cstaru)
            prevHup = np.array([], dtype=np.int)
            prevHlow = np.array([], dtype=np.int)
            for u in range(1, self.gamma+1):
                Cstaru = ((self.CstarMax-self.Cstar) /
                          (self.gamma-1)**2)*(u-1)**2+self.Cstar
                Hup = JsemilabeledSets[u-1][0]
                prevHup = np.hstack((prevHup, Hup))
                nbUp = Hup.shape[0]
                if nbUp > 0:
                    sampleWeights = np.hstack((sampleWeights, [Cstaru]*nbUp))
                    trainingX = np.vstack((trainingX, Xt[Hup]))
                    trainingy = np.hstack((trainingy, [+1]*nbUp))
                Hlow = JsemilabeledSets[u-1][1]
                prevHlow = np.hstack((prevHlow, Hlow))
                nbLow = Hlow.shape[0]
                if nbLow > 0:
                    sampleWeights = np.hstack((sampleWeights, [Cstaru]*nbLow))
                    trainingX = np.vstack((trainingX, Xt[Hlow]))
                    trainingy = np.hstack((trainingy, [-1]*nbLow))
            # Train the SVM at the current iteration
            try:
                svm = SVC(kernel="rbf", C=self.C, gamma=self.sigma,
                          random_state=1)
                svm.fit(trainingX, trainingy, sample_weight=sampleWeights)
                self.svm = svm
            except:
                # if there is an error during the fit, stop and use the SVM
                # model from the previous iteration. A possible error is when
                # there are samples from only one class.
                self.finishedWithError = True
                break
            # Test convergence and stop if necessary here. In this case, the
            # SVM model just trained is the final model for the target domain.
            if (i > 0 and idxsSource.shape[0] == 0 and
               lambdda + delta <= self.ceil and
               Si.shape[0] <= self.ceil):
                self.finishedWithError = False
                break
            # If we reach here, we have not converged yet.
            # Predict label of target examples and compute new semilabeled sets
            ytpred = self.svm.decision_function(Xt).reshape(-1)
            ytpredPos = np.where(ytpred >= 0)[0]
            ytpredNeg = np.where(ytpred < 0)[0]
            Hup = ytpredPos[ytpred[ytpredPos] <= 1]
            Hlow = ytpredNeg[ytpred[ytpredNeg] >= -1]
            # Remove from Hup and Hlow the target examples that were already
            # added to the training set in the previous iterations.
            if prevHup.shape[0] > 0:
                Hup = np.array([v for v in Hup if v not in prevHup],
                               dtype=np.int)
                Hlow = np.array([v for v in Hlow if v not in prevHup],
                                dtype=np.int)
            if prevHlow.shape[0] > 0:
                Hup = np.array([v for v in Hup if v not in prevHlow],
                               dtype=np.int)
                Hlow = np.array([v for v in Hlow if v not in prevHlow],
                                dtype=np.int)
            # Sort to have the closest to the margin at the begining
            Hup = Hup[np.argsort(ytpred[Hup])[::-1]]
            Hlow = Hlow[np.argsort(ytpred[Hlow])]
            lambdda = min(self.rho, Hup.shape[0])
            delta = min(self.rho, Hlow.shape[0])
            # Select the first closest to the margin for both sets
            Hup = Hup[:lambdda]
            Hlow = Hlow[:delta]
            # Compute the set Si containing the target example that were
            # previously added to the training set with a certain semilabel but
            # that have a prediction by the current predictor different than
            # this semilabel
            Si = np.hstack((np.intersect1d(ytpredPos, prevHlow),
                            np.intersect1d(ytpredNeg, prevHup)))
            # Recompute the sets of semilabeled target examples
            nextJsemilabeledSets = []
            for k in range(1, self.gamma+1):
                if k == 1:
                    Jik = [Hup, Hlow]
                elif k >= 2 and k <= self.gamma-1:
                    Jik = JsemilabeledSets[k-2]
                    Jik[0] = np.array([v for v in Jik[0] if v not in Si],
                                      dtype=np.int)
                    Jik[1] = np.array([v for v in Jik[1] if v not in Si],
                                      dtype=np.int)
                else:  # k == gamma
                    prev = JsemilabeledSets[k-2]
                    last = JsemilabeledSets[k-1]
                    Jik = [np.union1d(prev[0], last[0]),
                           np.union1d(prev[1], last[1])]
                    Jik[0] = np.array([v for v in Jik[0] if v not in Si],
                                      dtype=np.int)
                    Jik[1] = np.array([v for v in Jik[1] if v not in Si],
                                      dtype=np.int)
                nextJsemilabeledSets.append(Jik)
            JsemilabeledSets = nextJsemilabeledSets
            # Remove the source examples that are the farthest from the
            # decision boundary
            yspred = self.svm.decision_function(Xs).reshape(-1)
            Qup = np.where(yspred >= 0)[0]
            Qup = np.intersect1d(Qup, idxsSource)
            Qup = Qup[np.argsort(yspred[Qup])[::-1]]
            Qlow = np.where(yspred < 0)[0]
            Qlow = np.intersect1d(Qlow, idxsSource)
            Qlow = Qlow[np.argsort(yspred[Qlow])]
            # if none of the remaining unabeled target fall into margin
            if lambdda + delta == 0:
                nu = min(self.rho, Qup.shape[0])
                kappa = min(self.rho, Qlow.shape[0])
            else:
                nu = min(lambdda, Qup.shape[0])
                kappa = min(delta, Qlow.shape[0])
            Qup = Qup[:nu]
            Qlow = Qlow[:kappa]
            if nu > 0:
                idxsSource = np.array([v for v in idxsSource if v not in Qup],
                                      dtype=np.int)
            if kappa > 0:
                idxsSource = np.array([v for v in idxsSource if v not in Qlow],
                                      dtype=np.int)
            i += 1

    def predict(self, X):
        return self.svm.predict(X)

    def decision_function(self, X):
        return self.svm.decision_function(X)
