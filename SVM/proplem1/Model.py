import numpy as np
import Utilities
from sklearn.svm import SVC


class Model:
    utilities = Utilities.Utilities()

    def __init__(self, data, C=100):
        self.C = C
        self.models = []
        self.data = data
        self.X_data_scaled = Model.utilities.scaling_features( np.array(data.iloc[:, 0:24]))
        self.Y_data_scaled = Model.utilities.scaling_labels(data= np.array(data.iloc[:, 24]))
        self.X_data = np.array(data.iloc[:, 0:24])
        self.Y_data = Model.utilities.scaling_labels(data= np.array(data.iloc[:, 24]))
        self.scaled_data = data.copy()

    def train(self, X, Y):

        svc = SVC(C=self.C, kernel="linear")  # ,class_weight={0:0.29875,1:0.70125 })
        svc.fit(X, Y)
        return svc

    def test(self, X, Y, svc):

        return svc.score(X, Y)

    def get_score_list(self, X, Y):
        score = []
        for i in range(10):
            X_train, X_test, Y_train, Y_test = Model.utilities.tt_split(X, Y)
            svc = self.train(X_train, Y_train)
            temp_scor = self.test(X_test, Y_test, svc)
            score.append(temp_scor)
        return score

    def get_result(self):
        score = self.get_score_list(self.X_data, self.Y_data)
        scaled_score = self.get_score_list(self.X_data_scaled, self.Y_data_scaled)

        return f"scaled model acc: {np.mean(scaled_score)} , original model: {np.mean(score)} \n" + f"scaled model acc: {scaled_score} \n original model: {score}"




