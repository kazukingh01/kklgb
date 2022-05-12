from typing import Callable
import numpy as np
import lightgbm as lgb
from scipy.misc import derivative


__all__ = [
    "LGBCustomObjective",
    "LGBCustomEval",
    "calc_grad_hess",
]


class LGBCustomObjective:
    def __init__(self, func: Callable):
        self.func = func
    def __call__(self, y_pred: np.ndarray, data: lgb.Dataset):
        """
        customized objective base function
        Params::
            y_pred:
                Predicted value. In the case of multi class, the length is n_sample * n_class
                Value is ... array([0 data 0 label prediction, ..., N data 0 label prediction, 0 data 1 label prediction, ..., ])
            data:
                train_set
            func_loss:
                Take y_pred and y_true as input and return with the same shape as y_pred
                grad, hess = func_loss(y_pred, y_true)
        """
        y_pred, y_true = self.convert_lgb_input(y_pred, data)
        grad, hess = self.func(y_pred, y_true)
        return grad.T.reshape(-1), hess.T.reshape(-1)
    @classmethod
    def convert_lgb_input(cls, y_pred: np.ndarray, data: lgb.Dataset) -> (np.ndarray, np.ndarray):
        """
        Params::
            y_pred:
                Predicted value. In the case of multi class, the length is n_sample * n_class
                Value is ... array([0 data 0 label prediction, ..., N data 0 label prediction, 0 data 1 label prediction, ..., ])
            data:
                train_set
        """    
        if isinstance(data, lgb.Dataset):
            y_true = data.label
            if hasattr(data, "ndf_label"):
                y_true = data.get_culstom_label(y_true.astype(int))
        else:
            # If "data" is not lgb.dataset, the input will be reversed, so be careful.
            # "y_pred" -> label, "data" -> predicted value
            y_true = y_pred.copy()
            y_pred = data
        if y_pred.shape[0] != y_true.shape[0]:
            # multi class case
            y_pred = y_pred.reshape(-1 , y_true.shape[0]).T
        return y_pred, y_true


class LGBCustomEval(LGBCustomObjective):
    def __init__(self, func: Callable, name: str, is_higher_better: bool):
        assert isinstance(name, str)
        assert isinstance(is_higher_better, bool)
        super().__init__(func)
        self.name = name
        self.is_higher_better = is_higher_better
    def __call__(self, y_pred: np.ndarray, data: lgb.Dataset):
        """
        Params::
            y_pred:
                Predicted value. In the case of multi class, the length is n_sample * n_class
                Value is ... array([0 data 0 label prediction, ..., N data 0 label prediction, 0 data 1 label prediction, ..., ])
            data:
                train_set
            func_loss:
                Take y_pred and y_true as input and return with the same shape as y_pred
                value = func_loss(y_pred, y_true)
        """
        y_pred, y_true = self.convert_lgb_input(y_pred, data)
        value = self.func(y_pred, y_true)
        return self.name, value, self.is_higher_better


def calc_grad_hess(x: np.ndarray, t: np.ndarray, loss_func=None, dx=1e-6, **kwargs) -> (np.ndarray, np.ndarray, ):
    grad = derivative(lambda _x: loss_func(_x, t, **kwargs), x, n=1, dx=dx)
    hess = derivative(lambda _x: loss_func(_x, t, **kwargs), x, n=2, dx=dx)
    return grad, hess
