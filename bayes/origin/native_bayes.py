import numpy as np
from abc import abstractmethod


class NativeBayes:
    """
    初始化结构
    self._x , self._y: 几率训练集的变量
    self._data: 核心数组, 存储实际使用的条件概率的相关信息
    self._func: 模型核心 决策函数,  根据输入的x , y 输出对应的后验概率
    self._n_possibilities: 记录哥哥唯独特征取值个数的数组: [S1, S2, ... , Sn]
    self._labelled_x: 记录按类别分开后的输入数据的数组
    self._label_zip: 记录类别相关信息的数组, 视具体算法, 定义会有所不同
    self._cat_counter: 核心数组, 记录第i类数据的个数 category
    self._con_counter: 核心数组, 用于记录数据条件概率的原始极大似然估计
                                         ^  (d)
            self._con_counter[d][c][p] = p(X  = p|y = c)
    self.label_dict: 核心字典, 用于记录数值化类别时的转换关系
    self._feat_dicts: 核心字典, 用于记录数值化各维度特征(feat)时的转换关系
    """
    def __init__(self):
        self._x = self._y = None
        self._data = self._func = None
        self._n_possibilities = None
        self._labelled_x = self._label_zip = None
        self._cat_counter = self._con_counter = None
        self.label_dict = self._feat_dicts = None

    # 重载 __getitem__ 运算符 避免定义大量的property
    def __getitem__(self, item):
        if isinstance(item, str):
            return getattr(self, '_' + item)

    # sample_weight 样本权重 用于体现各个样本的重要性
    @abstractmethod
    def feed_data(self, x, y, sample_weight=None):
        pass

    @abstractmethod
    def feed_sample_weight(self, sample_weight=None):
        pass

    # 定义计算先验概率的函数, lb为各个估计中的平滑项 λ
    # lb 默认值为 1, 也就是说采用默认的拉普拉斯平滑
    def get_prior_probability(self, lb=1):
        return [(_c_num + lb) / (len(self._y) + lb * len(self._cat_counter)) for _c_num in self._cat_counter]

    # 定义具有普遍性的训练函数
    def fit(self, x=None, y=None, sample_weight=None, lb=1):
        # 如果有传入x , y就用传入的x , y初始化模型
        if x is not None and y is not None:
            self.feed_data(x, y, sample_weight)
        # 调用核心算法得到决策函数
        self._func = self._fit(lb)

    # 核心算法由子类实现
    @abstractmethod
    def _fit(self, lb):
        pass

    @abstractmethod
    def _transfer_x(self, x):
        pass

    # 定义预测单一样本的函数
    # 参照get_raw_result 控制该函数是输出预测的类别还是输出相应的后验概率
    # get_raw_result=False 输出类别 , get_raw_result=True 输出概率
    def predict_one(self, x, get_raw_result=False):
        # 在进行预测之前, 要先把新的输入数据数值化
        # 如果输入的是Numpy数组, 要先将它转化为python数组  python数组操作更快
        if isinstance(x, np.ndarray):
            x = x.tolist()
        else:
            x = x[:]
        # 调用相关方法进行数值化, 该方法随具体模型改变
        x = self._transfer_x(x)
        m_arg, m_probability = 0, 0
        # 遍历各个类别 找到能让后验概率最大化的类别
        print(len(self._con_counter))
        for i in range(len(self._cat_counter)):
            print('predict:', i)
            p = self._func(x, i)
            if p > m_probability:
                m_arg, m_probability = i, p
        if get_raw_result:
            return m_probability
        else:
            return self.label_dict[m_arg]

    # 定义预测多样本的函数, 本质是不断调用predict_one函数
    def predict(self, x, get_raw_result=False):
        return np.array([self.predict_one(xx, get_raw_result) for xx in x])

    def evaluate(self, x, y):
        y_pred = self.predict(x)
        print("Acc: {:12.6} %".format(100 * np.sum(y_pred == y) / len(y)))
