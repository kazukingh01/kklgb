import numpy as np

from kklgb.model import KkBooster
from kklgb.util.functions import softmax
from kklgb.loss import CrossEntropyLoss, CategoricalCrossEntropyLoss, Accuracy

if __name__ == "__main__":
    n_data    = 1000
    n_classes = 5
    x_train = np.random.rand(n_data, 100)
    x_valid = np.random.rand(n_data, 100)
    # CategoricalCrossEntropyLoss with smoothing
    y_train = np.random.randint(0, n_classes, n_data)
    y_valid = np.random.randint(0, n_classes, n_data)
    model = KkBooster()
    model.fit(
        x_train, y_train, params=dict(learning_rate=0.1, num_tree=20 , num_class=n_classes), 
        loss_func=CategoricalCrossEntropyLoss(n_classes=n_classes, smoothing=0.1), loss_func_eval=[Accuracy(top_k=1), ],
        x_valid=x_valid, y_valid=y_valid, early_stopping_rounds=10, early_stopping_name=1
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
    >>> model.predict_proba(x_valid)
    array([[0.25404362, 0.24652479, 0.2258233 , 0.13971473, 0.13389355],
        [0.31965606, 0.16673657, 0.1606588 , 0.19516516, 0.15778341],
        [0.39376252, 0.10126554, 0.14888474, 0.22659472, 0.12949248],
        ...,
        [0.21189655, 0.21784553, 0.14354537, 0.21471147, 0.21200108],
        [0.18013197, 0.10313054, 0.25239738, 0.21566456, 0.24867554],
        [0.29192389, 0.12707235, 0.10829128, 0.15021296, 0.32249953]])
    """