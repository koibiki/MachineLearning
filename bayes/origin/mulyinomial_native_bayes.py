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
        print('features:', list(features))
        feat_dicts = [{_l: i for i, _l in enumerate(feats)} for feats in features]
        print('feat_dicts:', list(feat_dicts))
        label_dict = {_l: i for i, _l in enumerate(set(y))}
        print('label_dict:', label_dict)

        # 利用转换字典更新训练集
        x = np.array([[feat_dicts[i][_l] for i, _l in enumerate(sample)] for sample in x])
        print(x)
        y = np.array([label_dict[yy] for yy in y])
        print(y)

        # 获得各个类别数据个数
        cat_counter = np.bincount(y)
        # 记录各个维度特征的取值个数
        n_possibilities = [len(feats) for feats in features]
        # 获得各个类别数据的下标
        labels = [y == value for value in range(len(cat_counter))]
        # 利用下表获得记录按类别分开后的输入数据的数组
        labelled_x = [x[ci].T for ci in labels]
        # 更新模型的各个属性
        self._x, self._y = x, y
        self._labelled_x, self._label_zip = labelled_x, list(zip(labels, labelled_x))
        (self._cat_counter, self._feat_dicts, self._n_possibilities) = (cat_counter, feat_dicts, n_possibilities)
        self.label_dict = {i: _l for _l, i in label_dict.items()}
        # 调用权重函数 更新记录条件概率的数组
        self.feed_sample_weight(sample_weight)

    def feed_sample_weight(self, sample_weight=None):
        self._con_counter = []
        for dim, _p in enumerate(self._n_possibilities):
            if sample_weight is None:
                self._con_counter.append([np.bincount(xx[dim], minlength=_p) for xx in self._labelled_x])
            else:
                self._con_counter.append([np.bincount(xx[dim], weights=sample_weight[label] / sample_weight[label].mean(), minlength=_p) for label, xx in self._label_zip])

    def _fit(self, lb):
        pass

    def _transfer_x(self, x):
        pass