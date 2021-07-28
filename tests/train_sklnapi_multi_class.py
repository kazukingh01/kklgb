import numpy as np

from kklgb.model import train, KkLGBMClassifier


if __name__ == "__main__":
    n_data    = 1000
    n_classes = 5
    x_train = np.random.rand(n_data, 100)
    x_valid = np.random.rand(n_data, 100)
    y_train = np.random.randint(0, n_classes, n_data)
    y_valid = np.random.randint(0, n_classes, n_data)
    model = KkLGBMClassifier(objective="multiclass")
    model.fit(
        x_train, y_train,
        eval_set=(x_valid, y_valid, ),
        early_stopping_rounds=100,
    )
    model.predict_proba(x_valid)
    """
    >>> model.predict_proba(x_valid)
    array([[0.1444223 , 0.28401512, 0.33114644, 0.14477927, 0.09563687],
        [0.59541939, 0.04564735, 0.07159583, 0.06691825, 0.22041917],
        [0.06492371, 0.2274204 , 0.02255402, 0.66411463, 0.02098724],
        ...,
        [0.51846475, 0.10423766, 0.07336129, 0.12330448, 0.18063181],
        [0.52813132, 0.1512596 , 0.12766683, 0.06308485, 0.1298574 ],
        [0.13664064, 0.06131266, 0.56524724, 0.12767244, 0.10912702]])
    """