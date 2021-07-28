import numpy as np

from kklgb.model import KkLGBMRegressor


if __name__ == "__main__":
    n_data  = 1000
    x_train = np.random.rand(n_data, 100)
    x_valid = np.random.rand(n_data, 100)
    y_train = np.random.rand(n_data)
    y_valid = np.random.rand(n_data)
    model = KkLGBMRegressor(objective="mse")
    model.fit(
        x_train, y_train,
        eval_set=(x_valid, y_valid, ),
        early_stopping_rounds=100,
    )
    model.predict(x_valid)
    """
    >>> model.predict(x_valid)
    array([
        0.59325235, 0.47467501, 0.40927531, 0.59626465, 0.52191296,
        0.38809764, 0.47171268, 0.30436612, 0.50006505, 0.39067254,
        0.29527686, 0.34637758, 0.39129693, 0.38841459, 0.40546926,
        0.51336475, 0.53381281, 0.52195705, 0.5357639 , 0.63475759,
        ...
    """