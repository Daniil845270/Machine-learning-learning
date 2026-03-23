from sklearn.datasets import fetch_openml
from sklearn.model_selection import cross_val_score
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import GridSearchCV

# downloading the data and separating the training and test sets
mnist = fetch_openml('mnist_784', as_frame=False)
X, y = mnist.data, mnist.target
X_train, X_test, y_train, y_test = X[:60000], X[60000:], y[:60000], y[60000:]

#############################################
# param_grid = {
#     "n_neighbors": list(range(1, 10)),
#     "weights": ['uniform', 'distance']
# }

# search = GridSearchCV(
#     estimator=KNeighborsClassifier(),
#     param_grid=param_grid,
#     scoring="accuracy",
#     cv=15,
#     n_jobs=-1,     # use all CPU cores
#     refit=False,   # don't retrain the best model, just evaluate
#     verbose=1
# )

# search.fit(X_train, y_train)

# for k, w, mean_score, std_score in zip(
#     search.cv_results_["param_n_neighbors"],
#     search.cv_results_["param_weights"],
#     search.cv_results_["mean_test_score"],
#     search.cv_results_["std_test_score"]
# ):
#     print(f"n_neighbors={k}: weights={w}: mean={mean_score:.5f}, std={std_score:.5f}")

# print("best:", search.best_params_, search.best_score_)

#######################################

# n_neighbors=1: weights=uniform: mean=0.97102, std=0.00418
# n_neighbors=1: weights=distance: mean=0.97102, std=0.00418
# n_neighbors=2: weights=uniform: mean=0.96473, std=0.00393
# n_neighbors=2: weights=distance: mean=0.97102, std=0.00418
# n_neighbors=3: weights=uniform: mean=0.97152, std=0.00350
# n_neighbors=3: weights=distance: mean=0.97258, std=0.00331
# n_neighbors=4: weights=uniform: mean=0.97000, std=0.00357
# n_neighbors=4: weights=distance: mean=0.97293, std=0.00353
# n_neighbors=5: weights=uniform: mean=0.97035, std=0.00360
# n_neighbors=5: weights=distance: mean=0.97157, std=0.00319
# n_neighbors=6: weights=uniform: mean=0.96940, std=0.00342
# n_neighbors=6: weights=distance: mean=0.97240, std=0.00352
# n_neighbors=7: weights=uniform: mean=0.96937, std=0.00392
# n_neighbors=7: weights=distance: mean=0.97037, std=0.00361
# n_neighbors=8: weights=uniform: mean=0.96828, std=0.00406
# n_neighbors=8: weights=distance: mean=0.97080, std=0.00367
# n_neighbors=9: weights=uniform: mean=0.96768, std=0.00423
# n_neighbors=9: weights=distance: mean=0.96867, std=0.00400
# best: {'n_neighbors': 4, 'weights': 'distance'} 0.9729333333333332

################################
knbr_clf = KNeighborsClassifier(n_neighbors=4, weights='distance')
knbr_clf.fit(X_train, y_train)

y_pred = knbr_clf.predict(X_test)

print("test accuracy:", accuracy_score(y_test, y_pred))
print(classification_report(y_test, y_pred))
print(confusion_matrix(y_test, y_pred))
##################################

# test accuracy: 0.9714
#               precision    recall  f1-score   support

#            0       0.97      0.99      0.98       980
#            1       0.97      1.00      0.98      1135
#            2       0.98      0.96      0.97      1032
#            3       0.97      0.96      0.97      1010
#            4       0.98      0.97      0.97       982
#            5       0.96      0.97      0.96       892
#            6       0.98      0.99      0.98       958
#            7       0.96      0.97      0.96      1028
#            8       0.99      0.94      0.97       974
#            9       0.96      0.96      0.96      1009

#     accuracy                           0.97     10000
#    macro avg       0.97      0.97      0.97     10000
# weighted avg       0.97      0.97      0.97     10000

# [[ 973    1    1    0    0    1    3    1    0    0]
#  [   0 1132    2    0    0    0    1    0    0    0]
#  [  10    5  995    2    1    0    0   16    3    0]
#  [   0    1    3  974    1   14    1    7    4    5]
#  [   1    5    0    0  950    0    4    3    0   19]
#  [   4    0    0    9    2  862    7    1    3    4]
#  [   4    2    0    0    3    3  946    0    0    0]
#  [   0   17    4    0    3    0    0  994    0   10]
#  [   5    2    4   14    5   11    4    4  920    5]
#  [   3    4    2    7    9    4    1   10    1  968]]