import numpy as np

from kklgb.model import train
from kklgb.loss import CrossEntropyLoss, MultiTaskLoss, MultiTaskEvalLoss, CrossEntropyNDCGLoss

if __name__ == "__main__":
    n_data    = 1000
    n_classes = 10
    x_train = np.random.rand(n_data, 100)
    x_valid = np.random.rand(n_data, 100)
    y_train = np.concatenate(
        [
              np.eye(3)[np.random.randint(0, 3, n_data)],
              np.argsort(np.random.rand(n_data, n_classes - 3)) + 1,
        ], axis=1
    ).astype(float)
    y_valid = np.concatenate(
        [
              np.eye(3)[np.random.randint(0, 3, n_data)],
              np.argsort(np.random.rand(n_data, n_classes - 3)) + 1,
        ], axis=1
    ).astype(float)
    model = train(
        dict(learning_rate=0.1, num_tree=50, num_class=n_classes, n_jobs=1), x_train, y_train, 
        loss_func=MultiTaskLoss([
            CrossEntropyLoss(n_classes=3),
            CrossEntropyNDCGLoss(n_classes=n_classes-3),
        ]),
        loss_func_eval=[
            MultiTaskEvalLoss(CrossEntropyLoss(n_classes=3), [i for i in range(0,3)]),
            MultiTaskEvalLoss(CrossEntropyNDCGLoss(n_classes=n_classes-3), [i for i in range(3,n_classes)]),
        ], x_valid=x_valid, y_valid=y_valid
    )
    """
    >>> model.predict(x_valid)
    array([[ 0.14862849,  0.11858507,  0.03087535, -0.44927477, -0.49183245],
        [ 0.51036759, -0.1404629 , -0.1775952 ,  0.01696814, -0.19565474],
        [ 1.04624275, -0.31175909,  0.07366724,  0.49365783, -0.06588246],
        ...,
        [-0.1083156 , -0.08062755, -0.4977626 , -0.09511865, -0.1078224 ],
        [-0.04068951, -0.59838371,  0.29662548,  0.13934497,  0.28176972],
        [ 0.27716315, -0.55457334, -0.71450534, -0.38727594,  0.37677172]])
    """
