import numpy as np

from kklgb.model import KkLGBMClassifier


if __name__ == "__main__":
    n_data    = 1000
    n_classes = 120
    x_train = np.random.rand(n_data, 100)
    y_train = np.random.randint(0, n_classes, n_data)
    model = KkLGBMClassifier(objective="multiclass")
    model.fit(x_train, y_train)
    model.fit(x_train, y_train, refit=300)
