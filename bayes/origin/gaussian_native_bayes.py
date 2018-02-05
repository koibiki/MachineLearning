import numpy as np
from math import pi, exp

# 记录根号下2 pi
from bayes.origin.native_bayes import NativeBayes

sqrt_pi = (2 * pi) ** 0.5


class NBFunctions:
    # 定义正太分布的密度函数
    @staticmethod
    def gaussion(x, mu, sigma):
        return exp(-(x - mu) ** 2 / (2 * sigma ** 2)) / (sqrt_pi * sigma)

    # 定义进行极大似然估计的函数
    @staticmethod
    def gaussian_maximum_likelihood(labelled_x, n_category, dim):
        mu = [np.sum(labelled_x[c][dim]) / len(labelled_x[c][dim]) for c in range(n_category)]
        sigma = [np.sum((labelled_x[c][dim] - mu[c]) ** 2) / len(labelled_x[c][dim]) for c in range(n_category)]

        # 利用极大似然估计得到 mu 和 sigma, 定义生成计算条件概率密度的函数的函数func
        def func(_c):
            def sub(xx):
                return NBFunctions.gaussion(xx, mu[_c], sigma[_c])
            return sub
        # 利用func返回目标列表
        return [func(_c=c) for c in range(n_category)]


class GaussianNB(NativeBayes):
    def feed_data(self, x, y, sample_weight=None):
        x = np.array([list(map(lambda c: float(c), sample)) for sample in x])
        # 数值化类别向量
        labels = list(set(y))
        label_dict = {label: i for i, label in enumerate(labels)}
        y = np.array([label_dict[yy] for yy in y])
        cat_counter = np.bincount(y)
        labels = [y == value for value in range(len(cat_counter))]
        labelled_x = [x[label].T for label in labels]
        # 更新模型的各个属性
        self._x, self._y = x.T, y
        self._labelled_x, self._label_zip = labelled_x, labels
        self._cat_counter, self.label_dict = cat_counter, {i:_l for _l, i in label_dict.items()}
        self.feed_sample_weight(sample_weight)

    # 定义处理样本权重的函数
    def feed_sample_weight(self, sample_weight=None):
        if sample_weight is not None:
            local_weights = sample_weight * len(sample_weight)
            for i, label in enumerate(self._label_zip):
                self._labelled_x[i] *= local_weights[label]

    def _fit(self, lb):
        n_category = len(self._cat_counter)
        p_category = self.get_prior_probability(lb)
        # 利用极大似然估计获得计算条件概率的函数, 使用数组变量data进项cunc
        data = \
            [NBFunctions.gaussian_maximum_likelihood(self._labelled_x, n_category, dim) for dim in range(len(self._x))]
        self._data = data

        def func(input_x, tar_category):
            rs = 1
            for d, xx in enumerate(input_x):
                rs *= data[d][tar_category](xx)
            return rs * p_category[tar_category]
        return func

    def _transfer_x(self, x):
        return x