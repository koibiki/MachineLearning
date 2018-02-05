import numpy as np

from bayes.origin.native_bayes import NativeBayes
from bayes.origin.mulyinomial_native_bayes import MultinomialNB
from bayes.origin.gaussian_native_bayes import GaussianNB

class MergedNB(NativeBayes):
    """
        初始化结构
        self._whether_discrete: 记录各个唯独的变量是否离散型变量
        self._whether_continuous: 记录各个维度的变量是否连续型变量
        self._multinomial, self._gaussion: 离散型, 连续型朴素贝叶斯模型
    """
    def __init__(self, whether_continuous):
        super().__init__()
        self._multinomial, self._gaussion = MultinomialNB(), GaussianNB()
        if whether_continuous is None:
            self._whether_discrete = self._whether_continuous = None
        else:
            self._whether_continuous = np.array(whether_continuous)
            self._whether_discrete = ~self._whether_continuous

    def feed_data(self, x, y, sample_weight=None):
        if sample_weight is not None:
            sample_weight = np.array(sample_weight)
