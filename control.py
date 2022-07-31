import anfis
import prediction
import membership.mfDerivs
import membership.membershipfunction
import numpy as np
import pandas as pd
import os
from datetime import datetime
from sklearn import preprocessing
import threading


# input training dataset
def input_training_data(file_name: str, data_size, features, target: int):
    path = os.path.dirname(os.path.abspath(__file__))
    df_back = pd.read_csv(os.path.join(path, file_name), low_memory=False)
    x = df_back.iloc[:data_size, features]
    y = df_back.iloc[:data_size, target]

    return x, y


def training_time_testing(mf, features):
    # count training time
    start_time = datetime.now()

    mf = mf
    mfc = membership.membershipfunction.MemFuncs(mf)
    x, y = input_training_data('./data/kddcup99_csv3.csv', 120, features, 42)
    anf = anfis.ANFIS(x, y, x, y, mfc)
    anf.trainHybridJangOffLine(epochs=10)

    end_time = datetime.now()
    print('Duration: {}'.format(end_time - start_time))

# training anfis model with initial data and parameters
def attack_training(attack):
    features = []

    # Customize input features for different attacks
    if attack == "back":
        features = [24, 26, 37, 39]
    elif attack == "smurf":
        features = [24, 26, 37, 39]
    elif attack == "neptune":
        features = [24, 26, 37, 39]

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

    x, y = input_training_data(f'./data/{attack}_attack.csv', 4000, features, 42)
    print(f"Current attack: {attack}")
    anf = anfis.ANFIS(x, y, x, y, mfc)
    anf.trainHybridJangOffLine(epochs=10)
    # record trained parameters
    np.save(f'./parameters/{attack}_consequents.npy', anf.consequents_final)
    np.save(f'./parameters/{attack}_mfs.npy', anf.memClass_final)


# predict back attack using trained parameters
def attack_prediction(dataset, attack):
    f1 = np.load(f'./parameters/{attack}_consequents2.npy', allow_pickle=True)
    f2 = np.load(f'./parameters/{attack}_mfs2.npy', allow_pickle=True)
    # input testing dataset
    x_test, y_test = dataset

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


# control either activate training or prediction function
def control(control):
    """
    Create a control panel to control either training or doing prediction
    0: activate training module
    1: activate prediction module
    """

    if control == 0:
        # count training time
        start_time = datetime.now()
        # training for back attack
        attacks = ["back", "smurf", "neptune"]
        for attack in attacks:
            threading.Thread(target=attack_training, args=[attack]).start()
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
        print("Back attack result:")
        back_result.append(attack_prediction(testing_dataset,'back'))
        print("Smurf attack result:")
        smurf_result.append(attack_prediction(testing_dataset,'smurf'))
        print("Neptune attack result:")
        neptune_result.append(attack_prediction(testing_dataset,'neptune'))

        for x in range(len(testing_dataset[0])):
            temp = []
            temp.append(back_result[0][x])
            temp.append(smurf_result[0][x])
            temp.append(neptune_result[0][x])
            results.append(temp)

        results_minmax = min_max_scaler.fit_transform(np.array(np.round_(results,2)))
        print("\nResult after min-max normalization:")
        print(results_minmax)
