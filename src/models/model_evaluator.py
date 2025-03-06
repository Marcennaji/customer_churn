from sklearn.metrics import classification_report


class ModelEvaluator:
    """Handles model evaluation and reporting."""

    def evaluate(self, model, X_train, X_test, y_train, y_test):
        """Evaluates a trained model on training and test sets."""
        y_train_preds = model.predict(X_train)
        y_test_preds = model.predict(X_test)

        print(f"Model: {model.__class__.__name__}")
        print("\nTest Results")
        print(classification_report(y_test, y_test_preds))
        print("\nTrain Results")
        print(classification_report(y_train, y_train_preds))
