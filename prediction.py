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
        self.fittedValues = predict(self, self.X)
        self.residuals = self.Y - self.fittedValues[:, 0]

        return self.fittedValues, self.residuals

    def plotResults(self):
        import matplotlib.pyplot as plt
        plt.plot(range(len(self.fittedValues)), self.fittedValues, 'r', label='trained')
        plt.plot(range(len(self.Y)), self.Y, 'b', label='original')
        plt.legend(loc='upper left')
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
    print("layerFive", layerFive)

    return layerFive


consequents = [[-3.835804820182112], [460.5980554919805], [-132.9793790801906],[-284.7300195743627],
               [-167.41357508428607], [310.4560294633015], [264.1486783401019], [63.87287797980972],
               [-271.0164033646234], [17.60359047421069], [-490.1372826642983], [187.2030834125831]]

mf = [
      [['gaussmf', {'mean': 0., 'sigma': 1.}], ['gaussmf', {'mean': -1., 'sigma': 2.}]],
      [['gaussmf', {'mean': 0., 'sigma': 1.}], ['gaussmf', {'mean': -1., 'sigma': 2.}]]
      ]
mfc = membership.membershipfunction.MemFuncs(mf)

path = os.path.dirname(os.path.abspath(__file__))
df_test = pd.read_csv(os.path.join(path, 'kdd_test.csv'), low_memory=False)
# df_test = pd.read_excel(os.path.join(path, 'kdd_test.xlsx'))
x_test = df_test.iloc[:120,2:4]
y_test = df_test.iloc[:120,5]

pred = PREDICTION(x_test,y_test,mfc, consequents)
# print(pred)
pred.plotResults()