from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
from sklearn.exceptions import FitFailedWarning
import warnings
warnings.filterwarnings("ignore", category=FitFailedWarning)
warnings.filterwarnings("ignore", category=UserWarning)

class HyperparameterTuner:
    def __init__(self, model, param_grid, scoring="accuracy", cv=3, n_iter=20, random_state=42):
        self.model = model
        self.param_grid = param_grid
        self.scoring = scoring
        self.cv = cv
        self.n_iter = n_iter
        self.random_state = random_state
        self.best_model = None
        self.best_params = None
        self.best_score = None

    def tune(self, X, y, method="random"):

        if method == "random":
            search = RandomizedSearchCV(
                estimator=self.model,
                param_distributions=self.param_grid,
                n_iter=self.n_iter,
                scoring=self.scoring,
                cv=self.cv,
                random_state=self.random_state,
                error_score="raise",  
                n_jobs=-1,
                verbose=1
            )
        elif method == "grid":
            search = GridSearchCV(
                estimator=self.model,
                param_grid=self.param_grid,
                scoring=self.scoring,
                cv=self.cv,
                error_score="raise",
                n_jobs=-1,
                verbose=1
            )
        else:
            raise ValueError("Method must be 'grid' or 'random'")

        print("\n🔍 Running hyperparameter search...")
        search.fit(X, y)

        self.best_model = search.best_estimator_
        self.best_params = search.best_params_
        self.best_score = search.best_score_

        return self.best_model, self.best_params, self.best_score