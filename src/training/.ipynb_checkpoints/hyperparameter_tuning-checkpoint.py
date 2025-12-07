from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
from sklearn.metrics import make_scorer
import pandas as pd
import numpy as np

class HyperparameterTuner:
    def __init__(self, model, param_grid, scoring='accuracy', n_iter=10, cv=5, random_state=None):
        self.model = model
        self.param_grid = param_grid
        self.scoring = scoring
        self.n_iter = n_iter
        self.cv = cv
        self.random_state = random_state
        self.best_model = None
        self.best_params = None
        self.best_score = None

    def tune(self, X, y, method='grid'):
        if method == 'grid':
            search = GridSearchCV(estimator=self.model, param_grid=self.param_grid,
                                  scoring=self.scoring, cv=self.cv, n_jobs=-1)
        elif method == 'random':
            search = RandomizedSearchCV(estimator=self.model, param_distributions=self.param_grid,
                                         scoring=self.scoring, n_iter=self.n_iter, cv=self.cv,
                                         random_state=self.random_state, n_jobs=-1)
        else:
            raise ValueError("Method must be either 'grid' or 'random'.")

        search.fit(X, y)
        self.best_model = search.best_estimator_
        self.best_params = search.best_params_
        self.best_score = search.best_score_

        return self.best_model, self.best_params, self.best_score

    def get_best_model(self):
        return self.best_model

    def get_best_params(self):
        return self.best_params

    def get_best_score(self):
        return self.best_score
