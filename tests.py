import anfis
import membership.mfDerivs
import membership.membershipfunction
import numpy as np
import pandas as pd
import os
from datetime import datetime
# record time
start_time = datetime.now()

# ts = np.loadtxt("trainingSet.txt", usecols=[1,2,3])#numpy.loadtxt('c:\\Python_fiddling\\myProject\\MF\\trainingSet.txt',usecols=[1,2,3])
# x = ts[:,0:2]
# print(x1)
# y = ts[:,2]
# print(y1)

# importing kdd cup 99 dataset
path = os.path.dirname(os.path.abspath(__file__))
df = pd.read_csv(os.path.join(path, 'kddcup99_csv3.csv'), low_memory=False)
data = df.values
x = data[:1200, [37,39]]
# x = data[:1200, [25, 27, 37,39]]
x_int = np.array(x, dtype=int)
y = data[:1200, 42]

df_test = pd.read_csv(os.path.join(path, 'kdd_test.csv'), low_memory=False)
# data_test = df.values
x_test = df_test.iloc[:120,2:4]
y_test = df_test.iloc[:120,5]
# print(x_test)
# print(y_test)

mf = [
      [['gaussmf', {'mean': 0., 'sigma': 1.}], ['gaussmf', {'mean': -1., 'sigma': 2.}]],
      [['gaussmf', {'mean': 0., 'sigma': 1.}], ['gaussmf', {'mean': -1., 'sigma': 2.}]]
      ]
# mf = [
#       [['gaussmf', {'mean': 0., 'sigma': 1.}], ['gaussmf', {'mean': -1., 'sigma': 2.}], ['gaussmf', {'mean': 2., 'sigma': 3.}]],
#       [['gaussmf', {'mean': 0., 'sigma': 1.}], ['gaussmf', {'mean': -1., 'sigma': 2.}], ['gaussmf', {'mean': 2., 'sigma': 3.}]],
#       [['gaussmf', {'mean': 0., 'sigma': 1.}], ['gaussmf', {'mean': -1., 'sigma': 2.}], ['gaussmf', {'mean': 2., 'sigma': 3.}]],
#       [['gaussmf', {'mean': 0., 'sigma': 1.}], ['gaussmf', {'mean': -1., 'sigma': 2.}], ['gaussmf', {'mean': 2., 'sigma': 3.}]]
#       ]

mfc = membership.membershipfunction.MemFuncs(mf)

anf = anfis.ANFIS(x, y, x_test,y_test, mfc)
anf.trainHybridJangOffLine(epochs=10)
# print(round(anf.consequents[-1][0], 6))
# print(round(anf.consequents[-2][0], 6))
# print(round(anf.fittedValues[9][0], 6))
# if round(anf.consequents[-1][0], 6) == -5.275538 and round(anf.consequents[-2][0], 6) == -1.990703 and round(
#         anf.fittedValues[9][0], 6) == 0.002249:
#     print('test is good')

print("Plotting errors")
anf.plotErrors()
print("Plotting results")
anf.plotResults()
# anf.plotMF(x_int,0)

end_time = datetime.now()
print('Duration: {}'.format(end_time - start_time))
