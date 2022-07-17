"""
variables to solve

anfisobj.rules:
        self.memClass = copy.deepcopy(memFunction)
        self.memFuncs = self.memClass.MFList
        self.memFuncsByVariable = [[x for x in range(len(self.memFuncs[z]))] for z in range(len(self.memFuncs))]
        self.rules = np.array(list(itertools.product(*self.memFuncsByVariable)))

anfisobj.consequents:
        self.consequents = np.empty(self.Y.ndim * len(self.rules) * (self.X.shape[1] + 1))
        self.consequents.fill(0)

"""
import itertools
import numpy as np
from membership import mfDerivs
import membership.membershipfunction
import copy
import pandas as pd
import os

class PREDICTION:
    def __init__(self, X, Y, memFunction, consequents):
        self.X = np.array(copy.copy(X))
        self.Y = np.array(copy.copy(Y))
        self.XLen = len(self.X)
        self.memClass = copy.deepcopy(memFunction)
        self.memFuncs = self.memClass.MFList
        self.memFuncsByVariable = [[x for x in range(len(self.memFuncs[z]))] for z in range(len(self.memFuncs))]
        self.rules = np.array(list(itertools.product(*self.memFuncsByVariable)))
        self.consequents = consequents


    def prediction(self):
        self.fitted = []
        self.resid = []
        self.fittedValues = np.round_(predict(self, self.X))
        for xs in self.fittedValues:
            for x in xs:
                self.fitted.append(x)
        self.residuals = self.Y - self.fittedValues[:, 0]
        for y in self.residuals:
            self.resid.append(y)

        return self.fitted, self.resid

    def plotResults(self):
        import matplotlib.pyplot as plt
        plt.plot(range(len(self.fitted)), self.fitted, 'r', label='trained')
        plt.plot(range(len(self.Y)), self.Y, 'b', label='original')
        plt.legend(loc='upper left')
        plt.show()

    def plotError(self):
        import matplotlib.pyplot as plt
        # plt.scatter(range(len(self.residuals)), self.residuals)
        plt.errorbar(range(len(self.fitted)), self.fitted, fmt='o', color='b', yerr=self.resid, capsize=2, linestyle='None')
        plt.show()


def forwardHalfPass(ANFISObj, Xs):
    layerFour = np.empty(0, )
    wSum = []

    for pattern in range(len(Xs[:, 0])):
        # layer one
        # membership functions
        layerOne = ANFISObj.memClass.evaluateMF(Xs[pattern, :])

        # layer two
        miAlloc = [[layerOne[x][ANFISObj.rules[row][x]] for x in range(len(ANFISObj.rules[0]))] for row in
                   range(len(ANFISObj.rules))]
        layerTwo = np.array([np.product(x) for x in miAlloc]).T

        # layer three
        wSum.append(np.sum(layerTwo))
        layerThree = layerTwo / wSum[pattern]

        # prep for layer four (bit of a hack)
        rowHolder = np.concatenate([x * np.append(Xs[pattern, :], 1) for x in layerThree])
        layerFour = np.append(layerFour, rowHolder)

    layerFour = np.array(np.array_split(layerFour, pattern + 1))

    return layerFour


def predict(ANFISObj, varsToTest):
    layerFour = forwardHalfPass(ANFISObj, varsToTest)

    # layer five
    layerFive = np.dot(layerFour, ANFISObj.consequents)
    print(layerFive)

    return layerFive
