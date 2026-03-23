from sklearn.datasets import fetch_openml
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.neighbors import KNeighborsClassifier
import numpy as np
from scipy.ndimage import shift

# downloading the data and separating the training and test sets
mnist = fetch_openml('mnist_784', as_frame=False)
X, y = mnist.data, mnist.target
X_train, X_test, y_train, y_test = X[:60000], X[60000:], y[:60000], y[60000:]

def createShift(X_train, vertical, horizontal):
    X_train_shift = X_train.copy()
    for idx, image in enumerate(X_train):
        reshaped = image.reshape(28,28)
        shifted = shift(reshaped, (vertical, horizontal))
        X_train_shift[idx] = shifted.reshape(-1)
    return X_train_shift

X_train_down = createShift(X_train, 1, 0)
X_train_up = createShift(X_train, -1, 0)
X_train_left = createShift(X_train, 0, -1)
X_train_right = createShift(X_train, 0, 1)


X_fully_reshaped = np.vstack((X_train, X_train_down, X_train_up, X_train_left, X_train_right))
y_twice_big = np.concatenate((y_train, y_train))
y_quadruple = np.concatenate((y_twice_big, y_twice_big))
y_fully_reshaped = np.concatenate((y_quadruple, y_train))


knbr_clf = KNeighborsClassifier(n_neighbors=4, weights='distance')
knbr_clf.fit(X_fully_reshaped, y_fully_reshaped)

y_pred = knbr_clf.predict(X_test)

print("test accuracy:", accuracy_score(y_test, y_pred))
print(classification_report(y_test, y_pred))
print(confusion_matrix(y_test, y_pred))


# test accuracy: 0.9763
#               precision    recall  f1-score   support

#            0       0.98      0.99      0.99       980
#            1       0.97      1.00      0.98      1135
#            2       0.99      0.97      0.98      1032
#            3       0.98      0.98      0.98      1010
#            4       0.98      0.97      0.98       982
#            5       0.97      0.97      0.97       892
#            6       0.98      0.99      0.99       958
#            7       0.97      0.97      0.97      1028
#            8       0.99      0.95      0.97       974
#            9       0.96      0.97      0.96      1009

#     accuracy                           0.98     10000
#    macro avg       0.98      0.98      0.98     10000
# weighted avg       0.98      0.98      0.98     10000

# [[ 974    1    1    0    0    1    2    1    0    0]
#  [   0 1132    2    0    0    0    0    0    0    1]
#  [   6    2 1004    2    2    0    2   13    1    0]
#  [   0    2    2  988    1    7    0    5    3    2]
#  [   0    5    0    0  952    0    4    2    0   19]
#  [   2    2    0    7    0  869    5    1    3    3]
#  [   5    4    0    0    2    2  945    0    0    0]
#  [   0   19    5    0    2    0    0  993    0    9]
#  [   4    1    3   10    5   10    2    3  930    6]
#  [   2    5    1    5    6    5    0    9    0  976]]
