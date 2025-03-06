from sklearn.model_selection import train_test_split


class DatasetSplitter:
    """Handles train-test data splitting."""

    def __init__(self, test_size=0.3, random_state=42):
        self.test_size = test_size
        self.random_state = random_state

    def split(self, X, y):
        """Splits the dataset into training and test sets."""
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=self.test_size, random_state=self.random_state
        )
        return X_train, X_test, y_train, y_test
