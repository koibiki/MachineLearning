import numpy as np


class DataUtils:

    @staticmethod
    def get_data_set(path, train_num=None, tar_idx=None, shuffle=True, split=","):
        x= []
        with open(path, "r", encoding="utf8") as file:
            for sample in file:
                x.append(sample.strip().split(split))

        if shuffle:
            np.random.shuffle(x)

        tar_idx = -1 if tar_idx is None else tar_idx

        y = np.array([xx.pop(tar_idx) for xx in x])
        x = np.array(x)
        if train_num is None:
            return x, y
        else:
            return (x[:train_num], y[:train_num]), (x[train_num:], y[train_num:])
