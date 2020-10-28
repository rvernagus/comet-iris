import numpy as np
import pandas as pd
from comet_ml import Experiment
from sklearn import tree
from sklearn.model_selection import train_test_split
from sklearn.datasets import load_iris


experiment = Experiment(
    api_key=input('api key: '),
    project_name=input('project name: '),
    workspace=input('workspace: '),
)

X, y = load_iris(return_X_y=True)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=.2)
clf = tree.DecisionTreeClassifier()
clf = clf.fit(X_train, y_train)
experiment.log_metric('accuracy', clf.score(X_test, y_test))
