from sklearn.base import BaseEstimator
from sklearn.tree import DecisionTreeClassifier


class ModelSwitcher(BaseEstimator):
    def __init__(self, estimator=DecisionTreeClassifier()):
        """
        A Custom BaseEstimator that can switch between classifiers.
        param estimator: sklearn object - The classifier
        """
        self.estimator = estimator

    def fit(self, x, y=None, **kwargs):
        self.estimator.fit(x, y)
        return self

    def predict(self, x, y=None):
        return self.estimator.predict(x)

    def predict_proba(self, x):
        return self.estimator.predict_proba(x)

    def score(self, x, y):
        return self.estimator.score(x, y)
