import numpy as np

from kklgb.model import train
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
    model = train(
        dict(learning_rate=0.1, num_tree=10 , num_class=n_classes), x_train, y_train, 
        loss_func=CategoricalCrossEntropyLoss(n_classes=n_classes, smoothing=0.1), loss_func_eval=[Accuracy(top_k=1), ],
        x_valid=x_valid, y_valid=y_valid
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
    # CrossEntropyLoss
    y_train = softmax(np.random.rand(n_data, n_classes))
    y_valid = softmax(np.random.rand(n_data, n_classes))
    model = train(
        dict(learning_rate=0.1, num_tree=30 , num_class=n_classes), x_train, y_train, 
        loss_func=CrossEntropyLoss(n_classes=n_classes), loss_func_eval=[Accuracy(top_k=1), ],
        x_valid=x_valid, y_valid=y_valid, early_stopping_rounds=20, early_stopping_name="acc_top1"
    )
    """
    >>> model.predict(x_valid)
    array([[-0.03532138, -0.04090289,  0.01968107, -0.09898752,  0.06972061],
        [-0.02931828, -0.00734911, -0.07328964,  0.01730638, -0.08165091],
        [ 0.03476547, -0.06350686, -0.06448613,  0.08460536, -0.05620495],
        ...,
        [ 0.02694324,  0.05249052,  0.08360009, -0.08380778,  0.03392917],
        [-0.10413074, -0.00808495, -0.04076826, -0.04842817,  0.12692731],
        [ 0.01480251,  0.00765096,  0.01704932,  0.03700201,  0.01353969]])
    """
