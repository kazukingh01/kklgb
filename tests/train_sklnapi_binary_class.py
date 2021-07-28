import numpy as np

from kklgb.model import KkLGBMClassifier


if __name__ == "__main__":
    n_data  = 1000
    x_train = np.random.rand(n_data, 100)
    x_valid = np.random.rand(n_data, 100)
    y_train = np.random.randint(0, 2, n_data)
    y_valid = np.random.randint(0, 2, n_data)
    model = KkLGBMClassifier(objective="binary")
    model.fit(
        x_train, y_train,
        eval_set=(x_valid, y_valid, ),
        early_stopping_rounds=100,
    )
    model.predict_proba(x_valid)
    """
    >>> model.predict_proba(x_valid)
    array([[0.89584652, 0.10415348],
        [0.67034925, 0.32965075],
        [0.20407602, 0.79592398],
        ...,
        [0.64226316, 0.35773684],
        [0.7718203 , 0.2281797 ],
        [0.91286637, 0.08713363]])
    """