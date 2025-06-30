import gzip
import math
import os
import pickle
import random
import sys
import time
import warnings

import numpy as np
from sklearn.metrics import accuracy_score
from sklearn.datasets import make_moons
from sklearn.datasets import load_svmlight_files

from dasvm import dasvm
from GBRFF_mean import gbrff_da_mean
from GBRFF_variance import gbrff_da_variance
from GBRFF_meanvariance import gbrff_da_meanvariance
#from GBRFF2 import gbrff as gbrff2
#from GBRFF_batch import gbrff as gbrff_batch
from dalc import disagreement, d_disagreement, jointError, d_jointError, dalc


def loadMoons(seed):
    #n_source = 300
    n_source = 300
    #n_target = 300
    n_target = 300
    #n_test = 1000
    n_test = 1000
    Xs, ys = make_moons(n_source, noise=0.05, random_state=seed)
    ys[ys == 0] = -1
    Xt, _ = make_moons(n_target, noise=0.05, random_state=seed)
    Xtest, ytest = make_moons(n_test, noise=0.05, random_state=seed)
    ytest[ytest == 0] = -1
    trans = -np.mean(Xs, axis=0)
    Xs = 2 * (Xs + trans)
    Xt = 2 * (Xt + trans)
    Xtest = 2 * (Xtest + trans)
    data = {}
    for degree in list(np.arange(0,360,36)): # pi/5 radians jumps
        theta = -degree*math.pi/180
        rotation = np.array([[math.cos(theta), math.sin(theta)],
                            [-math.sin(theta), math.cos(theta)]])
        Xt_degree = np.dot(Xt, rotation.T)
        Xtest_degree = np.dot(Xtest, rotation.T)
        data[str(degree)] = (Xs, ys, Xt_degree, Xtest_degree, ytest)
    return data


def loadAmazonReview():
    data = {}
    for da in ["books", "dvd", "electronics", "kitchen"]:
        train = "amazonreview/" + da + "_train.svmlight"
        test = "amazonreview/" + da + "_test.svmlight"
        Xtrain, ytrain, Xtest, ytest = load_svmlight_files([train, test])
        Xtrain, Xtest = (np.array(X.todense()) for X in (Xtrain, Xtest))
        ytrain, ytest = (np.array(y, dtype=int)
                         for y in (ytrain, ytest))
        summ = np.sum(Xtrain, axis=1)[:, None]
        summ[summ == 0] = 1
        TF = Xtrain / summ
        DF = np.sum(Xtrain > 0, axis=0)
        Xtrain = TF*np.log(Xtrain.shape[0]/(DF+1))
        summ = np.sum(Xtest, axis=1)[:, None]
        summ[summ == 0] = 1
        TF = Xtest / summ
        DF = np.sum(Xtest > 0, axis=0)
        Xtest = TF*np.log(Xtest.shape[0]/(DF+1))
        data[da] = (Xtrain, Xtest, ytrain, ytest)
    return data


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


def fitClf(da, algo, p, Xs_train, ys_train, Xt_train, Xt_test, seed,
           labels=None):
    # Hackish piece of code. Sometimes, during the reverse validation, the set
    # of reverse labels contains only one class. But as it is required to have
    # at least two classes to fit some classifiers, just swap one label.
    if labels is not None:
        if labels[0] not in ys_train:
            ys_train[0] = labels[0]
        elif labels[1] not in ys_train:
            ys_train[0] = labels[1]
    if algo == "DASVM":
        clf = dasvm()
        clf.fit(Xs_train, ys_train, Xt_train)
    elif algo == "GBRFF":
        clf = gbrff_da_mean(T=20, randomState=np.random.RandomState(1),
                    Lambda_0=p["Lambda_0"], Lambda_b = 0, Lambda_omega = 0, gamma=1/Xs_train.shape[1])
        clf.fit(ys_train, Xs_train, Xt_train)
    elif algo == "GBRFF_DA_mean":
        print(p)
        clf = gbrff_da_mean(T=20, randomState=np.random.RandomState(1),
                    Lambda_0=p["Lambda_0"], Lambda_b=p["Lambda_b"], Lambda_omega=p["Lambda_omega"], gamma=p["gamma"])
        clf.fit(ys_train, Xs_train, Xt_train)
    elif algo == "GBRFF_DA_variance":
        print(p)
        clf = gbrff_da_variance(T=20, randomState=np.random.RandomState(1),
                    Lambda_0=p["Lambda_0"], Lambda_b=p["Lambda_b"], Lambda_omega=p["Lambda_omega"], gamma=p["gamma"])
        clf.fit(ys_train, Xs_train, Xt_train)
    elif algo == "GBRFF_DA_meanvariance40":
        print(p)
        clf = gbrff_da_meanvariance(T=40, randomState=np.random.RandomState(1),
                    Lambda_0=p["Lambda_0"], Lambda_b=p["Lambda_b"], Lambda_omega=p["Lambda_omega"], gamma=p["gamma"])
        clf.fit(ys_train, Xs_train, Xt_train)
    elif algo == "GBRFF_DA_meanvariance":
        print(p)
        clf = gbrff_da_meanvariance(T=20, randomState=np.random.RandomState(1),
                    Lambda_0=p["Lambda_0"], Lambda_b=p["Lambda_b"], Lambda_omega=p["Lambda_omega"], gamma=p["gamma"])
        clf.fit(ys_train, Xs_train, Xt_train)
    #elif algo == "GBRFF_DA_meanvariancesqrt":
    #    print(p)
    #    clf = gbrff_da_meanvariancesqrt(T=20, randomState=np.random.RandomState(1),
    #                Lambda=p["Lambda"], gamma=p["gamma"])
    #    clf.fit(ys_train, Xs_train, Xt_train)
    #elif algo == "GBRFF_DA":
    #    print(p)
    #    clf = gbrff(T=20, randomState=np.random.RandomState(1),
    #                Lambda=p["Lambda"], gamma=1/Xs_train.shape[1])
    #    clf.fit(ys_train, Xs_train, Xt_train)
    #elif algo == "GBRFF2" or algo == "GBRFF2_DA":
    #    clf = gbrff2(T=20, randomState=np.random.RandomState(1),
    #                 Lambda=p["Lambda"], gamma=1/Xs_train.shape[1])
    #    clf.fit(ys_train, Xs_train, Xt_train)
    #elif algo == "GBRFF_Batch" or algo == "GBRFF_Batch_DA":
    #    clf = gbrff_batch(T=20, randomState=np.random.RandomState(1),
    #                 Lambda=p["Lambda"], gamma=1/Xs_train.shape[1])
    #    clf.fit(ys_train, Xs_train, Xt_train)
    elif algo == "DALC":
        clf = dalc(B=p["B"], C=p["C"], kernel="rbf", gamma=1/Xs_train.shape[1])
        clf.fit(Xs_train, ys_train, Xt_train)
    return clf.predict(Xt_test)


def reverseValidation(da, algo, Xs_train, ys_train, Xt_train, seed):
    print("Debut reverse validation")
    perfs = []
    labels = sorted(np.unique(ys_train))
    for i, p in enumerate(listParams[algo]):
        t1 = time.time()
        pred = fitClf(da, algo, p, Xs_train, ys_train, Xt_train,
                      Xt_train, seed)
        pred = fitClf(da, algo, p, Xt_train, pred, Xs_train, Xs_train, seed,
                      labels)
        perf = 100*accuracy_score(ys_train, pred)
        perfs.append(perf)
        t2 = time.time()
        print("{:5.2f}".format(perf), p, i+1, "/", len(listParams[algo]),
              "in {:6.2f}s".format(t2-t1))
    bestIdx = np.argmax(perfs)
    print("Best with", listParams[algo][bestIdx])
    return listParams[algo][bestIdx]


def fitClfReverseValidation(da, algo, Xs_train, ys_train, Xt_train, Xt_test,
                            seed):
    bestParams = reverseValidation(da, algo, Xs_train, ys_train, Xt_train,
                                   seed)
    return fitClf(da, algo, bestParams, Xs_train, ys_train, Xt_train, Xt_test,
                  seed), bestParams


warnings.filterwarnings("ignore")
if not os.path.exists("results"):
    try:
        os.makedirs("results")
    except:
        pass
#listParams = {# "DASVM": [{"None": [None]}],
#              "GBRFF": listP({"Lambda": [0]}),
#              "GBRFF_DA": listP({"Lambda": [0.01, 0.1, 1, 10, 100,1000, 10000,
#                                            100000]}),
#              "GBRFF2": listP({"Lambda": [0]}),
#              "GBRFF2_DA": listP({"Lambda": [0.01, 0.1, 1, 10, 100,1000, 10000,
#                                             100000]}),
#              "GBRFF3_DA": listP({"Lambda": [0.01, 0.1, 1, 10, 100,1000, 10000,
#                                             100000]})}

# liste des paramètres pour la cross-validation

listParams = { #"DASVM": [{"None": [None]}],
              "DALC": listP({"B": [0.01, 0.1, 1, 10, 100],
                             "C": [0.01, 0.1, 1, 10, 100]}),
              "GBRFF": listP({"Lambda_0": [0, 2**-5, 2**-4, 2**-3, 2**-2]}),#,
              #"GBRFF_DA": listP({"Lambda": [1e-7, 1e-6, 1e-5, 1e-4, 1e-3, 1e-2, 2**-5, 0.05, 2**-4, 0.1, 2**-3, 2**-2, 0.5, 1, 10, 100,  1e3, 1e4, 1e5, 1e6, 1e7, 1e8, 1e9, 1e10, 1e11, 1e12]})
              #"GBRFF_DA": listP({"Lambda": [0, 2**-5, 0.05, 2**-4, 0.1, 2**-3, 2**-2, 0.5, 1, 10, 100,  1e3, 1e4, 1e5]})
              #"GBRFF_DA_mean": listP({"Lambda_0": [0, 2**-5, 2**-4, 2**-3, 2**-2], 
              #                   "gamma": [2**-3, 2**-2, 2**-1, 1, 2],
              #                   "Lambda_b": [0, 0.01, 0.1, 1, 10, 100],
              #                   "Lambda_omega": [0, 0.01, 0.1, 1, 10, 100]})
              "GBRFF_DA_mean": listP({"Lambda_0": [0, 2**-5, 2**-4, 2**-3, 2**-2], 
                                 "gamma": [2**-3, 2**-2, 2**-1, 1, 2],
                                 "Lambda_b": [32],
                                 "Lambda_omega": [12]}),
              "GBRFF_DA_variance": listP({"Lambda_0": [0, 2**-5, 2**-4, 2**-3, 2**-2], 
                                 "gamma": [2**-3, 2**-2, 2**-1, 1, 2],
                                 "Lambda_b": [32],
                                 "Lambda_omega": [12]}),
              "GBRFF_DA_meanvariance": listP({"Lambda_0": [0, 2**-5, 2**-4, 2**-3, 2**-2], 
                                 "gamma": [2**-3, 2**-2, 2**-1, 1, 2],
                                 "Lambda_b": [32],
                                 "Lambda_omega": [12]}),
              ##"GBRFF_DA_meanvariance40": listP({"Lambda_0": [0, 2**-5, 2**-4, 2**-3, 2**-2], 
              #                   "gamma": [2**-3, 2**-2, 2**-1, 1, 2],
              #                   "Lambda_b": [0.001],
              #                   "Lambda_omega": [1000]})
              ##"GBRFF_DA_variance": listP({"Lambda": [0, 2**-5, 2**-4, 2**-3, 2**-2], 
                                 #"gamma": [2**-3, 2**-2, 2**-1, 1, 2]}),
              ##"GBRFF_DA_meanvariance": listP({"Lambda": [0, 2**-5, 2**-4, 2**-3, 2**-2], 
                                 #"gamma": [2**-3, 2**-2, 2**-1, 1, 2]})
              #"GBRFF_DA_meanvariancesqrt": listP({"Lambda": [0, 2**-5, 2**-4, 2**-3, 2**-2], 
              #                   "gamma": [2**-3, 2**-2, 2**-1, 1, 2]})
              #"GBRFF_DA": listP({"Lambda": [0, 2**-5, 2**-4, 2**-3, 2**-2]})
              #"GBRFF2": listP({"Lambda": [0]}),
              #"GBRFF2_DA": listP({"Lambda": [0.05, 0.1, 0.5, 1, 10, 100, 1e3, 1e4, 1e5, 1e6]}),
              #"GBRFF_Batch": listP({"Lambda": [0]}),
              #"GBRFF_Batch_DA": listP({"Lambda": [0.05, 0.1, 0.5, 1, 10, 100, 1e3, 1e4, 1e5, 1e6]})
              }

list_accuracy = {'GBRFF' : [], 'GBRFF_DA_mean' : [], 'GBRFF_DA_variance' : [], 'GBRFF_DA_meanvariance' : [], 'GBRFF_DA_meanvariance40' : [], 'DALC' : []}

seed = 1
if len(sys.argv) == 2:
    seed = int(sys.argv[1])
print("seed", seed)
np.random.seed(seed)
random.seed(seed)
r = {}
datasets = {"moons": loadMoons(seed)}#,
#            "amazonreview": loadAmazonReview()}

for da in datasets.keys():
    r[da] = {}
    print("\nDomain adaptation benchmark:", da)
    domains = list(datasets[da].keys())
    print("Domains:", domains)
    for source in domains:
        if da == "moons" and source != "0":
            continue
        for target in domains:
            if source == target and da != "moons":
                continue
            pair = str.upper(source[0])+"->"+str.upper(target[0])
            if da == "amazonreview":
                Xs_train, _, ys_train, _ = datasets[da][source]
                Xt_train, Xt_test, _, yt_test = datasets[da][target]
            elif da == "moons":
                Xs_train, ys_train, Xt_train, Xt_test, yt_test = datasets[
                                                                    da][target]
                pair = target
            r[da][pair] = {}
            for algo in listParams.keys():
                startTime = time.time()
                if len(listParams[algo]) > 1:  # reverse validation
                    print("Reverse Validation")
                    pred, tmp_params = fitClfReverseValidation(da, algo, Xs_train,
                                                   ys_train, Xt_train, Xt_test,
                                                   seed)
                    print("End Reverse Validation")
                else:  # no reverse validation
                    pred = fitClf(da, algo, listParams[algo][0], Xs_train,
                                  ys_train, Xt_train, Xt_test, seed)
                accuracy = 100 * accuracy_score(yt_test, pred)
                list_accuracy[algo] += [accuracy]
                elapsed = time.time() - startTime
                r[da][pair][algo] = (accuracy, elapsed)
                print(da, pair, algo, "{:5.2f}".format(accuracy),
                      "{:6.2f}s".format(elapsed))
    with gzip.open("./results/res" + str(seed) + ".pklz", "wb") as f:
        pickle.dump(r, f)

#%% to show accuracy in a polar plot

import matplotlib
import matplotlib.pyplot as plt
matplotlib.rcParams.update({'font.size': 22})
        

theta = list(np.arange(0,2*np.pi,np.pi/5))
fig = plt.figure(figsize=(20,20))
ax = fig.add_subplot(111, polar=True)
line1, = ax.plot(theta, list_accuracy['GBRFF'], label='GBRFF')
line2, = ax.plot(theta, list_accuracy['GBRFF_DA_mean'], label='GBRFF_DA_mean')
line3, = ax.plot(theta, list_accuracy['GBRFF_DA_variance'], label='GBRFF_DA_variance')
line4, = ax.plot(theta, list_accuracy['GBRFF_DA_meanvariance'], label='GBRFF_DA_meanvariance')
line5, = ax.plot(theta, list_accuracy['DALC'], label='DALC')

ax.set_rmax(100)
ax.set_rticks([20, 40, 60, 80])  # Less radial ticks
ax.set_rlabel_position(-22.5)  # Move radial labels away from plotted line
ax.grid(True)

leg = ax.legend(loc='center left', bbox_to_anchor=(0.1, 0), fancybox=True, shadow=True)
leg.get_frame().set_alpha(0.4)
#ax.set_title("36 12", va='bottom')

lines = [line1, line2, line3, line4, line5]#, line6]
lined = dict()
for legline, origline in zip(leg.get_lines(), lines):
    legline.set_picker(5)  # tolerance
    lined[legline] = origline


def onpick(event): # to make plots disappear if you click on the corresponding legend
    legline = event.artist
    origline = lined[legline]
    vis = not origline.get_visible()
    origline.set_visible(vis)
    if vis:
        legline.set_alpha(1.0)
    else:
        legline.set_alpha(0.2)
    fig.canvas.draw()

fig.canvas.mpl_connect('pick_event', onpick)

box = ax.get_position()
ax.set_position([box.x0, box.y0, box.width * 0.8, box.height])

plt.show()
