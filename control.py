import anfis
import prediction
import membership.mfDerivs
import membership.membershipfunction
import numpy as np
import pandas as pd
import os
from datetime import datetime

"""
Create a control panel to control either training or doing prediction
"""

# input training dataset
path = os.path.dirname(os.path.abspath(__file__))
df = pd.read_csv(os.path.join(path, 'kddcup99_csv3.csv'), low_memory=False)
x = df.iloc[:120, [25, 27, 37, 39]]
y = df.iloc[:120, 42]

# input testing dataset
df_test = pd.read_csv(os.path.join(path, 'kdd_test.csv'), low_memory=False)
x_test = df_test.iloc[:120, 2:4]
y_test = df_test.iloc[:120, 5]

# input membership functions (gaussian) (4features + 3mfs)
mf = [
    [['gaussmf', {'mean': 0., 'sigma': 1.}], ['gaussmf', {'mean': -1., 'sigma': 2.}], ['gaussmf', {'mean': 2., 'sigma': 3.}]],
    [['gaussmf', {'mean': 0., 'sigma': 1.}], ['gaussmf', {'mean': -1., 'sigma': 2.}], ['gaussmf', {'mean': 2., 'sigma': 3.}]],
    [['gaussmf', {'mean': 0., 'sigma': 1.}], ['gaussmf', {'mean': -1., 'sigma': 2.}], ['gaussmf', {'mean': 2., 'sigma': 3.}]],
    [['gaussmf', {'mean': 0., 'sigma': 1.}], ['gaussmf', {'mean': -1., 'sigma': 2.}], ['gaussmf', {'mean': 2., 'sigma': 3.}]]
]
mfc = membership.membershipfunction.MemFuncs(mf)

# input consequents
consequents = [[-3.835804820182112], [460.5980554919805], [-132.9793790801906], [-284.7300195743627],
            [-167.41357508428607], [310.4560294633015], [264.1486783401019], [63.87287797980972],
            [-271.0164033646234], [17.60359047421069], [-490.1372826642983], [187.2030834125831]]

control = "t"
# count time
start_time = datetime.now()
if control.lower() == "t":
    anf = anfis.ANFIS(x, y, x_test, y_test, mfc)
    anf.trainHybridJangOffLine(epochs=10)
elif control.lower() == "p":
    mf = [
        [['gaussmf', {'mean': 0., 'sigma': 1.}], ['gaussmf', {'mean': -1., 'sigma': 2.}]],
        [['gaussmf', {'mean': 0., 'sigma': 1.}], ['gaussmf', {'mean': -1., 'sigma': 2.}]]
    ]
    mfc = membership.membershipfunction.MemFuncs(mf)

    pred = prediction.PREDICTION(x_test, y_test, mfc, consequents)
    pred.prediction()
    pred.plotResults()
    pred.plotError()

end_time = datetime.now()
print('Duration: {}'.format(end_time - start_time))
