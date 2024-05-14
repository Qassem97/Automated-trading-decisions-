from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error

class ModelTrainer:
    def __init__(self, X_train, y_train):
        self.X_train = X_train
        self.y_train = y_train
        self.models = {}

    def train_models(self):
        # Linear Regression Model
        lr = LinearRegression()
        lr.fit(self.X_train, self.y_train)
        self.models['linear_regression'] = lr

        # Random Forest Model
        rf = RandomForestRegressor(n_estimators=100, random_state=42)
        rf.fit(self.X_train, self.y_train)
        self.models['random_forest'] = rf

        return self.models

    def predict_and_evaluate(self, X_test, y_test):
        results = {}
        for name, model in self.models.items():
            predictions = model.predict(X_test)
            mse = mean_squared_error(y_test, predictions)
            results[name] = mse
            print(f"{name} MSE: {mse}")
        return results
