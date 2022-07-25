import anfis
import prediction
import membership.mfDerivs
import membership.membershipfunction
import numpy as np
import pandas as pd
import os
from datetime import datetime
from sklearn import preprocessing

"""
Create a control panel to control either training or doing prediction
0: activate training module
1: activate prediction module
"""


def control(control):
    # input training dataset
    def input_training_data(file_name: str, data_size, features, target: int):
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

        x_back, y_back = input_training_data("back_attack.csv", 4000, [24, 26, 37, 39], 42)
        anf_back = anfis.ANFIS(x_back, y_back, x_back, y_back, mfc)
        anf_back.trainHybridJangOffLine(epochs=10)
        # record trained parameters
        np.save('back_consequents.npy', anf_back.consequents_final)
        np.save('back_mfs.npy', anf_back.memClass_final)

    # get parameters from smurf attack dataset
    def smurf_attack_training():
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

        x_back, y_back = input_training_data("smurf_attack.csv", 4000, [24, 26, 37, 39], 42)
        anf_back = anfis.ANFIS(x_back, y_back, x_back, y_back, mfc)
        anf_back.trainHybridJangOffLine(epochs=10)
        # record trained parameters
        np.save('smurf_consequents.npy', anf_back.consequents_final)
        np.save('smurf_mfs.npy', anf_back.memClass_final)

    # get parameters from neptune attack dataset
    def neptune_attack_training():
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

        x_back, y_back = input_training_data("neptune_attack.csv", 4000, [24, 26, 37, 39], 42)
        anf_back = anfis.ANFIS(x_back, y_back, x_back, y_back, mfc)
        anf_back.trainHybridJangOffLine(epochs=10)
        # record trained parameters
        np.save('neptune_consequents.npy', anf_back.consequents_final)
        np.save('neptune_mfs.npy', anf_back.memClass_final)

    # predict back attack using trained parameters
    def back_attack_testing(testing_dataset, num):
        f1 = np.load(f'./parameters/back_consequents{num}.npy', allow_pickle=True)
        f2 = np.load(f'./parameters/back_mfs{num}.npy', allow_pickle=True)
        # input testing dataset
        x_test, y_test = testing_dataset

        # input consequents
        consequents = f1

        # input membership functions
        mf = f2
        mfc = membership.membershipfunction.MemFuncs(mf)

        # activate module
        pred = prediction.PREDICTION(x_test, y_test, mfc, consequents)
        pred.prediction()
        # plot error
        # pred.plotError()
        return pred.fitted

    # predict smurf attack using trained parameters
    def smurf_attack_testing(testing_dataset, num):
        f1 = np.load(f'./parameters/smurf_consequents{num}.npy', allow_pickle=True)
        f2 = np.load(f'./parameters/smurf_mfs{num}.npy', allow_pickle=True)
        # input testing dataset
        x_test, y_test = testing_dataset

        # input consequents
        consequents = f1

        # input membership functions
        mf = f2
        mfc = membership.membershipfunction.MemFuncs(mf)

        # activate module
        pred = prediction.PREDICTION(x_test, y_test, mfc, consequents)
        pred.prediction()
        # plot error
        # pred.plotError()
        return pred.fitted

    # predict neptune attack using trained parameters
    def neptune_attack_testing(testing_dataset, num):
        f1 = np.load(f'./parameters/neptune_consequents{num}.npy', allow_pickle=True)
        f2 = np.load(f'./parameters/neptune_mfs{num}.npy', allow_pickle=True)
        # input testing dataset
        x_test, y_test = testing_dataset

        # input consequents
        consequents = f1

        # input membership functions
        mf = f2
        mfc = membership.membershipfunction.MemFuncs(mf)

        # activate module
        pred = prediction.PREDICTION(x_test, y_test, mfc, consequents)
        pred.prediction()
        # plot error
        # pred.plotError()

        return pred.fitted

    if control == 0:
        # count training time
        start_time = datetime.now()
        # training for back attack
        print("Training back attack")
        back_attack_training()
        print("Training smurf attack")
        smurf_attack_training()
        print("Training neptune attack")
        neptune_attack_training()
        end_time = datetime.now()
        print('Duration: {}'.format(end_time - start_time))

    elif control == 1:
        # set up min-max normalization
        min_max_scaler = preprocessing.MinMaxScaler()

        # input training dataset
        testing_dataset = input_training_data("./data/kdd_test.csv", None, [24, 26, 37, 39], 42)

        # store results
        results = []
        back_result = []
        smurf_result = []
        neptune_result = []
        back_result.append(back_attack_testing(testing_dataset,2))
        smurf_result.append(smurf_attack_testing(testing_dataset,2))
        neptune_result.append(neptune_attack_testing(testing_dataset,2))

        for x in range(len(testing_dataset[0])):
            temp = []
            temp.append(back_result[0][x])
            temp.append(smurf_result[0][x])
            temp.append(neptune_result[0][x])
            results.append(temp)

        results_minmax = min_max_scaler.fit_transform(np.array(np.round_(results,2)))
        print(results_minmax)
control(1)