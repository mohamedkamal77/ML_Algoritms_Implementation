from SVMModel import SVM

svm = SVM()
svm.fit()
svm.test()
svm.draw(svm.X_train, svm.Y_train, "TRAIN")
svm.draw(svm.X_test, svm.Y_test, "TEST")