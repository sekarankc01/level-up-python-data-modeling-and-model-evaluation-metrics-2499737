from joblib import load
import os 
import numpy as np
import pandas as pd
from sklearn import svm
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
from sklearn.metrics import accuracy_score, roc_auc_score, balanced_accuracy_score, matthews_corrcoef
print(os.getcwd())
os.chdir(r'C:/Users/sekar/OneDrive/Documents/GitHub/level-up-python-data-modeling-and-model-evaluation-metrics-2499737/')
print(os.getcwd())
X_train_scaled, X_test_scaled, y_train, y_test = load(r'./data/model_data.joblib'
  )
print(" x train ", X_train_scaled, "y train ", y_train, " y test ", y_test)
svc_params = {
  'C': np.linspace(.1, 5, 3), 
  'kernel': ['linear', 'rbf']
  }

grid_search = GridSearchCV(
  estimator=svm.SVC(),
  param_grid=svc_params
  )
  
grid_search.fit(X_train_scaled, y_train)

pd.DataFrame(grid_search.cv_results_)

grid_predictions = grid_search.predict(X_test_scaled)

print("balanced_accuracy_score ",balanced_accuracy_score(y_test, grid_predictions))
print("matthews_corrcoef ",matthews_corrcoef(y_test, grid_predictions))
auc = roc_auc_score(y_test, grid_search.predict_proba(X_test_scaled))

random_search = RandomizedSearchCV(
  estimator=svm.SVC(),
  param_distributions=svc_params,
  n_iter=5,
  random_state=1001
  )

random_search.fit(X_train_scaled, y_train)

pd.DataFrame(random_search.cv_results_)

random_predictions = random_search.predict(X_test_scaled)

balanced_accuracy_score(y_test, random_predictions)
matthews_corrcoef(y_test, random_predictions)
auc = roc_auc_score(y_test, random_search.predict_proba(X_test_scaled))
print("auc  = ", auc)