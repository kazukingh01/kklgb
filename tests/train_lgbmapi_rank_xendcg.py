import numpy as np

from kklgb.model import train
from kklgb.loss import CrossEntropyNDCGLoss

if __name__ == "__main__":
    n_data    = 1000
    n_classes = 10
    x_train = np.random.rand(n_data, 100)
    x_valid = np.random.rand(n_data, 100)
    y_train = np.random.rand(n_data, n_classes)
    y_valid = np.random.rand(n_data, n_classes)
    model = train(
        dict(learning_rate=0.1, num_tree=20, num_class=n_classes, n_jobs=1), x_train, y_train, 
        loss_func=CrossEntropyNDCGLoss(n_classes),
        x_valid=x_valid, y_valid=y_valid
    )
    """
    >>> np.argsort(np.argsort(model.predict(x_train), axis=1), axis=1)
    array([ [8, 5, 2, ..., 9, 6, 3],
            [0, 8, 9, ..., 6, 1, 2],
            [0, 8, 9, ..., 7, 6, 2],
            ...,
            [0, 4, 8, ..., 2, 7, 9],
            [0, 9, 6, ..., 8, 7, 4],
            [3, 7, 4, ..., 6, 1, 5]])
    >>> np.argsort(np.argsort(y_train, axis=1), axis=1)
    array([ [8, 4, 2, ..., 9, 6, 3],
            [0, 8, 9, ..., 6, 1, 2],
            [0, 8, 9, ..., 6, 7, 2],
            ...,
            [0, 4, 8, ..., 2, 7, 9],
            [0, 9, 5, ..., 8, 7, 4],
            [3, 7, 4, ..., 6, 1, 5]])
    """
