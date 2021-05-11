import numpy as np


class Utilities:

    def scaling_features(self, data):

        means = []
        max_ = []
        features_no = data.shape[1]
        for i in range(features_no):
            mean_temp = np.mean(data[:, i])
            temp_max = np.max(data[:, i] - mean_temp)
            temp_min = np.min(data[:, i] - mean_temp)

            temp = (temp_max + temp_min) / 2
            means.append(mean_temp + temp)

            if abs(temp_max) > abs(temp_min):
                max_.append(abs(temp_max) - temp)
            else:
                max_.append(abs(temp_min - temp))
        data = (data - means) / max_
        return data

    def scaling_labels(self, data):

        labels = list(set(data))
        labels.sort()
        data[data == labels[0]] = -1
        data[data == labels[1]] = 1
        return data

    def tt_split(self, X, Y, train_p=0.8):
        n = np.size(Y)
        indeces = list(range(n))
        np.random.shuffle(indeces)
        train_p = int(n*train_p)
        return X[indeces[0:train_p]], X[indeces[train_p:]], Y[indeces[0:train_p]], Y[indeces[train_p:]]