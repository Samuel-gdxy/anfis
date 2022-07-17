import anfis
import prediction
import membership.mfDerivs
import membership.membershipfunction
import numpy as np
import pandas as pd
import os
from datetime import datetime
import re

"""
Create a control panel to control either training or doing prediction
0: activate training module
1: activate prediction module
"""
control = 1
# count time
start_time = datetime.now()


# input training dataset
def input_training_data(file_name: str, data_size: int, features, target: int):
    path = os.path.dirname(os.path.abspath(__file__))
    df_back = pd.read_csv(os.path.join(path, file_name), low_memory=False)
    x = df_back.iloc[:data_size, features]
    y = df_back.iloc[:data_size, target]

    return x, y


# get parameters from back attack dataset
def back_attack_training():
    # input initial membership functions (gaussian) (4features + 3mfs)
    mf = [
        [['gaussmf', {'mean': 0., 'sigma': 1.}], ['gaussmf', {'mean': -1., 'sigma': 2.}],
         ['gaussmf', {'mean': 2., 'sigma': 3.}]],
        [['gaussmf', {'mean': 0., 'sigma': 1.}], ['gaussmf', {'mean': -1., 'sigma': 2.}],
         ['gaussmf', {'mean': 2., 'sigma': 3.}]],
        [['gaussmf', {'mean': 0., 'sigma': 1.}], ['gaussmf', {'mean': -1., 'sigma': 2.}],
         ['gaussmf', {'mean': 2., 'sigma': 3.}]],
        [['gaussmf', {'mean': 0., 'sigma': 1.}], ['gaussmf', {'mean': -1., 'sigma': 2.}],
         ['gaussmf', {'mean': 2., 'sigma': 3.}]]
    ]
    mfc = membership.membershipfunction.MemFuncs(mf)

    x_back, y_back = input_training_data("back_attack.csv", 200, [24, 26, 37, 39], 42)
    anf_back = anfis.ANFIS(x_back, y_back, x_back, y_back, mfc)
    anf_back.trainHybridJangOffLine(epochs=2)
    # record trained parameters
    np.save('back_consequents.npy', anf_back.consequents_final)
    np.save('back_mfs.npy', anf_back.memClass_final)


if control == 0:
    # training for back attack
    back_attack_training()

elif control == 1:
    f1 = np.load('back_consequents.npy', allow_pickle=True)
    f2 = np.load('back_mfs.npy', allow_pickle=True)
    # input testing dataset
    x_test, y_test = input_training_data("back_attack.csv", 250, [24, 26, 37, 39], 42)

    # input consequents
    consequents = f1

    # input membership functions
    mf = f2
    mfc = membership.membershipfunction.MemFuncs(mf)

    # activate module
    pred = prediction.PREDICTION(x_test, y_test, mfc, consequents)
    pred.prediction()
    # plot result
    pred.plotResults()
    # plot error
    pred.plotError()

end_time = datetime.now()
print('Duration: {}'.format(end_time - start_time))
