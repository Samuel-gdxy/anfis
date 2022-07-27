import control
import anfis
import prediction
import membership.mfDerivs
import membership.membershipfunction
from multiprocessing import Pool, cpu_count


if __name__ == "__main__":
    # control.control(1)
    def call_mfs(num:int):
        mf = []
        features = []
        if num == 4:
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
            features = [6,7,8,9]
        elif num == 6:
            mf = [
                [['gaussmf', {'mean': 0., 'sigma': 1.}], ['gaussmf', {'mean': -1., 'sigma': 2.}],
                 ['gaussmf', {'mean': 2., 'sigma': 3.}]],
                [['gaussmf', {'mean': 0., 'sigma': 1.}], ['gaussmf', {'mean': -1., 'sigma': 2.}],
                 ['gaussmf', {'mean': 2., 'sigma': 3.}]],
                [['gaussmf', {'mean': 0., 'sigma': 1.}], ['gaussmf', {'mean': -1., 'sigma': 2.}],
                 ['gaussmf', {'mean': 2., 'sigma': 3.}]],
                [['gaussmf', {'mean': 0., 'sigma': 1.}], ['gaussmf', {'mean': -1., 'sigma': 2.}],
                 ['gaussmf', {'mean': 2., 'sigma': 3.}]],
                [['gaussmf', {'mean': 0., 'sigma': 1.}], ['gaussmf', {'mean': -1., 'sigma': 2.}],
                 ['gaussmf', {'mean': 2., 'sigma': 3.}]],
                [['gaussmf', {'mean': 0., 'sigma': 1.}], ['gaussmf', {'mean': -1., 'sigma': 2.}],
                 ['gaussmf', {'mean': 2., 'sigma': 3.}]]
            ]
            features = [6, 7, 8, 9, 10, 11]
        elif num == 8:
            mf = [
                [['gaussmf', {'mean': 0., 'sigma': 1.}], ['gaussmf', {'mean': -1., 'sigma': 2.}],
                 ['gaussmf', {'mean': 2., 'sigma': 3.}]],
                [['gaussmf', {'mean': 0., 'sigma': 1.}], ['gaussmf', {'mean': -1., 'sigma': 2.}],
                 ['gaussmf', {'mean': 2., 'sigma': 3.}]],
                [['gaussmf', {'mean': 0., 'sigma': 1.}], ['gaussmf', {'mean': -1., 'sigma': 2.}],
                 ['gaussmf', {'mean': 2., 'sigma': 3.}]],
                [['gaussmf', {'mean': 0., 'sigma': 1.}], ['gaussmf', {'mean': -1., 'sigma': 2.}],
                 ['gaussmf', {'mean': 2., 'sigma': 3.}]],
                [['gaussmf', {'mean': 0., 'sigma': 1.}], ['gaussmf', {'mean': -1., 'sigma': 2.}],
                 ['gaussmf', {'mean': 2., 'sigma': 3.}]],
                [['gaussmf', {'mean': 0., 'sigma': 1.}], ['gaussmf', {'mean': -1., 'sigma': 2.}],
                 ['gaussmf', {'mean': 2., 'sigma': 3.}]],
                [['gaussmf', {'mean': 0., 'sigma': 1.}], ['gaussmf', {'mean': -1., 'sigma': 2.}],
                 ['gaussmf', {'mean': 2., 'sigma': 3.}]],
                [['gaussmf', {'mean': 0., 'sigma': 1.}], ['gaussmf', {'mean': -1., 'sigma': 2.}],
                 ['gaussmf', {'mean': 2., 'sigma': 3.}]]
            ]
            features = [6, 7, 8, 9, 10, 11, 12, 13]
        elif num == 12:
            mf = [
                [['gaussmf', {'mean': 0., 'sigma': 1.}], ['gaussmf', {'mean': -1., 'sigma': 2.}],
                 ['gaussmf', {'mean': 2., 'sigma': 3.}]],
                [['gaussmf', {'mean': 0., 'sigma': 1.}], ['gaussmf', {'mean': -1., 'sigma': 2.}],
                 ['gaussmf', {'mean': 2., 'sigma': 3.}]],
                [['gaussmf', {'mean': 0., 'sigma': 1.}], ['gaussmf', {'mean': -1., 'sigma': 2.}],
                 ['gaussmf', {'mean': 2., 'sigma': 3.}]],
                [['gaussmf', {'mean': 0., 'sigma': 1.}], ['gaussmf', {'mean': -1., 'sigma': 2.}],
                 ['gaussmf', {'mean': 2., 'sigma': 3.}]],
                [['gaussmf', {'mean': 0., 'sigma': 1.}], ['gaussmf', {'mean': -1., 'sigma': 2.}],
                 ['gaussmf', {'mean': 2., 'sigma': 3.}]],
                [['gaussmf', {'mean': 0., 'sigma': 1.}], ['gaussmf', {'mean': -1., 'sigma': 2.}],
                 ['gaussmf', {'mean': 2., 'sigma': 3.}]],
                [['gaussmf', {'mean': 0., 'sigma': 1.}], ['gaussmf', {'mean': -1., 'sigma': 2.}],
                 ['gaussmf', {'mean': 2., 'sigma': 3.}]],
                [['gaussmf', {'mean': 0., 'sigma': 1.}], ['gaussmf', {'mean': -1., 'sigma': 2.}],
                 ['gaussmf', {'mean': 2., 'sigma': 3.}]],
                [['gaussmf', {'mean': 0., 'sigma': 1.}], ['gaussmf', {'mean': -1., 'sigma': 2.}],
                 ['gaussmf', {'mean': 2., 'sigma': 3.}]],
                [['gaussmf', {'mean': 0., 'sigma': 1.}], ['gaussmf', {'mean': -1., 'sigma': 2.}],
                 ['gaussmf', {'mean': 2., 'sigma': 3.}]],
                [['gaussmf', {'mean': 0., 'sigma': 1.}], ['gaussmf', {'mean': -1., 'sigma': 2.}],
                 ['gaussmf', {'mean': 2., 'sigma': 3.}]],
                [['gaussmf', {'mean': 0., 'sigma': 1.}], ['gaussmf', {'mean': -1., 'sigma': 2.}],
                 ['gaussmf', {'mean': 2., 'sigma': 3.}]]
            ]
            features = [6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17]
        return mf, features

    def multi(num):
        mfs, features = call_mfs(num)
        control.training_time_testing(mfs, features)

    multi(4)
    # with Pool(cpu_count()) as p:
    #     p.map(multi, [4, 6, 8, 12])
