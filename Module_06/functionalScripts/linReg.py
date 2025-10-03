import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score

class LinearRegression:
    def __init__ (self):
        self.params = np.zeros(int(np.random.random()), float)[:, np.newaxis]

    def fit (self, X_train, y_train):
        bias = np.ones(len(X_train))
        X_b = np.c_[bias, X_train]
        inner_part = np.transpose(X_b) @ X_b
        inverse = np.linalg.inv(inner_part)
        X_part = inverse @ np.transpose(X_b)
        lse = X_part @ y_train
        self.params = lse
        return self.params
    
    def predict (self, X):
        bias_test = np.ones(len(X))
        X_test = np.c_[bias_test , X]
        y_hat = X_test @ self.params
        return y_hat
        

if __name__ == '__main__':
    X = np.array([
        [1, 4],
        [2, 5],
        [3, 8],
        [4, 2]
    ])

    y = np.array([1, 6, 8, 12])

    model = LinearRegression()
    lse = model.fit (X, y)
    print (f'Least Square Estimate : {lse}')

    y_hat = model.predict ([[5, 3]])
    print (f'Predicted value of weekly sale is {y_hat}')

    lr = LinearRegression ()
    lr.fit (X, y)
    pred = lr.predict (X)
    print (f'Library implementation : {pred}')

    r2 = r2_score(y, pred)
    print (f'Goodness of Fit : {r2}')

