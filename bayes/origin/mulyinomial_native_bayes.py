import numpy as np
from bayes.origin.native_bayes import NativeBayes


class MultinomialNB(NativeBayes):
    def feed_data(self, x, y, sample_weight=None):
        print('x:', x)
        if isinstance(x, list):
            features = map(list, zip(*x))
        else:
            features = x.T
        print('features:', list(features))
        features = [set(feat) for feat in features]
        feat_dicts = [{_l: i for i, _l in enumerate(feats)} for feats in features]
        label_dict = {_l: i for i, _l in enumerate(set(y))}

        # 利用转换字典更新训练集
        x = np.array([[feat_dicts[i][_l] for i, _l in enumerate(sample)] for sample in x])
        y = np.array([label_dict[yy] for yy in y])

        # 获得各个类别数据个数
        cat_counter = np.bincount(y)
        # 记录各个维度特征的取值个数
        n_possibilities = [len(feats) for feats in features]
        # 获得各个类别数据的下标
        labels = [y == value for value in range(len(cat_counter))]
        # 利用下表获得记录按类别分开后的输入数据的数组
        labelled_x = [x[ci].T for ci in labels]



    def feed_sample_weight(self, sample_weight=None):
        pass

    def _fit(self, lb):
        pass

    def _transfer_x(self, x):
        pass