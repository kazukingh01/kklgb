import numpy as np

from kklgb.model import train
from kklgb.util.functions import softmax
from kklgb.loss import CrossEntropyLoss, Accuracy, MultiTaskLoss, MultiTaskEvalLoss

if __name__ == "__main__":
    n_data    = 1000
    n_classes = 10
    x_train = np.random.rand(n_data, 100)
    x_valid = np.random.rand(n_data, 100)
    # CategoricalCrossEntropyLoss with smoothing
    y_train = np.eye(n_classes)[np.random.randint(0, 2, n_data)]
    y_valid = np.eye(n_classes)[np.random.randint(0, n_classes, n_data)]
    model = train(
        dict(learning_rate=0.1, num_tree=2 , num_class=n_classes, n_jobs=1), x_train, y_train, 
        loss_func=MultiTaskLoss([CrossEntropyLoss(n_classes=3), CrossEntropyLoss(n_classes=n_classes-3)]),
        loss_func_eval=[
            MultiTaskEvalLoss(Accuracy(top_k=1), [i for i in range(0,3)]), MultiTaskEvalLoss(Accuracy(top_k=1), [i for i in range(3,n_classes)]),
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
