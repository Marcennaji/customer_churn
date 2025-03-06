from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV


class ModelTrainer:
    """Handles training and hyperparameter tuning of models."""

    def __init__(self, random_state=42):
        self.random_state = random_state
        self.rf_model = None
        self.lr_model = None

    def train_random_forest(self, X_train, y_train):
        """Trains a Random Forest model using Grid Search."""
        param_grid = {
            "n_estimators": [200, 500],
            "max_features": ["auto", "sqrt"],
            "max_depth": [4, 5, 100],
            "criterion": ["gini", "entropy"],
        }

        rfc = RandomForestClassifier(random_state=self.random_state)
        grid_search = GridSearchCV(estimator=rfc, param_grid=param_grid, cv=5)
        grid_search.fit(X_train, y_train)
        self.rf_model = grid_search.best_estimator_
        return self.rf_model

    def train_logistic_regression(self, X_train, y_train):
        """Trains a Logistic Regression model."""
        lrc = LogisticRegression(solver="lbfgs", max_iter=3000)
        lrc.fit(X_train, y_train)
        self.lr_model = lrc
        return self.lr_model
