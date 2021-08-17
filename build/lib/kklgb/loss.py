import numpy as np
from typing import Callable

from kklgb.util.functions import sigmoid, softmax


__all__ = [
    "Loss",
    "BinaryCrossEntropyLoss",
    "CrossEntropyLoss",
    "CategoricalCrossEntropyLoss",
    "FocalLoss",
    "MSELoss",
    "MAELoss",
    "HuberLoss",
]


class Loss:
    def __init__(self, name: str, n_classes: int=1, target_dtype: np.dtype=np.int32, reduction: str="mean", is_higher_better: bool=False):
        assert isinstance(n_classes, int) and n_classes >= 0
        assert isinstance(target_dtype, np.dtype) or isinstance(target_dtype, type)
        assert isinstance(reduction, str) and reduction in ["mean", "sum"]
        assert isinstance(is_higher_better, bool)
        self.name         = name
        self.n_classes    = n_classes
        self.target_dtype = target_dtype
        self.conv_shape   = lambda x: x
        self.is_check     = True
        self.reduction    = (lambda x: np.mean(x)) if reduction == "mean" else (lambda x: np.sum(x))
        self.is_higher_better = is_higher_better
    def __str__(self):
        return f"{self.__class__.__name__}({self.name})"
    def convert(self, x: np.ndarray, t: np.ndarray):
        t = t.astype(self.target_dtype)
        if self.is_check:
            self.check(x, t)
            if self.n_classes == 1: self.conv_shape = lambda x: x.reshape(-1)
            self.is_check = False
        x = self.conv_shape(x)
        return x, t
    def check(self, x: np.ndarray, t: np.ndarray):
        print(f"input: {x}, \ninput shape{x.shape}\nlabel: {t}, \nlabel shape{t.shape}")
        assert isinstance(x, np.ndarray)
        assert isinstance(t, np.ndarray)
        assert x.shape[0] == t.shape[0]
        if self.n_classes > 0: assert x.shape[1] == self.n_classes
    def __call__(self, x: np.ndarray, t: np.ndarray):
        loss = self.loss(x, t)
        return self.reduction(loss)
    def loss(self, x: np.ndarray, t: np.ndarray):
        raise NotImplementedError
    def gradhess(self, x: np.ndarray, t: np.ndarray):
        raise NotImplementedError


class BinaryCrossEntropyLoss(Loss):
    def __init__(self, dx: float=1e-5):
        super().__init__("bce", n_classes=1, target_dtype=np.int32, is_higher_better=False)
        self.dx = dx
    def check(self, x: np.ndarray, t: np.ndarray):
        super().check(x, t)
        assert np.isin(np.unique(t), [0,1]).sum() == 2
    def loss(self, x: np.ndarray, t: np.ndarray):
        x, t = self.convert(x, t)
        x = sigmoid(x)
        x = np.clip(x, self.dx, 1 - self.dx)
        return -1 * (t * np.log(x) + (1 - t) * np.log(1 - x))
    def gradhess(self, x: np.ndarray, t: np.ndarray):
        x, t = self.convert(x, t)
        x = sigmoid(x)
        x = np.clip(x, self.dx, 1)
        grad = x - t
        hess = (1 - x) * x
        return grad, hess


class CrossEntropyLoss(Loss):
    def __init__(self, n_classes: int, dx: float=1e-5, target_dtype: np.dtype=np.float32):
        assert isinstance(n_classes, int) and n_classes > 1
        super().__init__("ce", n_classes=n_classes, target_dtype=target_dtype, is_higher_better=False)
        self.dx = dx
        self.conv_t_sum = lambda x: 1.0
    def check(self, x: np.ndarray, t: np.ndarray):
        super().check(x, t)
        if self.name == "ce":
            assert len(x.shape) == 2
            assert len(t.shape) == 2
            assert x.shape[1] == t.shape[1] == self.n_classes
            if ((t.sum(axis=1) / self.dx).round(0).astype(np.int32) == int(round(1 / self.dx, 0))).sum() != t.shape[0]:
                # If the sum of "t" is not equal to 1 (In other words, if "t" is not a probability)
                self.conv_t_sum = lambda x: x.sum(axis=1).reshape(-1, 1)
    def loss(self, x: np.ndarray, t: np.ndarray):
        x, t = self.convert(x, t)
        x = softmax(x)
        x = np.clip(x, self.dx, 1 - self.dx * (self.n_classes - 1))
        return (-1 * t * np.log(x)).sum(axis=1)
    def gradhess(self, x: np.ndarray, t: np.ndarray):
        """
        see: https://qiita.com/klis/items/4ad3032d02ff815e09e6
        see: https://www.ics.uci.edu/~pjsadows/notes.pdf
        """
        x, t = self.convert(x, t)
        x = softmax(x)
        x = np.clip(x, self.dx, 1 - self.dx * (self.n_classes - 1))
        t_sum = self.conv_t_sum(t)
        grad  = t_sum * x - t
        hess  = t_sum * x * (1 - x)
        return grad, hess


class CategoricalCrossEntropyLoss(CrossEntropyLoss):
    def __init__(self, n_classes: int, dx: float=1e-5, smoothing: float=False):
        assert (isinstance(smoothing, bool) and smoothing == False) or (isinstance(smoothing, float) and 0.0 < smoothing < 1.0)
        super().__init__(n_classes, dx=dx, target_dtype=np.int32)
        self.smoothing = smoothing
        self.name      = f"cce(smooth{round(self.smoothing, 2)})" if self.smoothing > 0.0 else "cce"
    def check(self, x: np.ndarray, t: np.ndarray):
        super().check(x, t)
        assert len(x.shape) == 2
        assert np.isin(np.unique(t), np.arange(self.n_classes)).sum() == self.n_classes
    def convert(self, x: np.ndarray, t: np.ndarray):
        x, t = super().convert(x, t)
        t = np.identity(x.shape[1])[t].astype(np.float32)
        if self.smoothing > 0.0:
            ndf = np.full(x.shape, self.smoothing / (self.n_classes - 1))
            ndf[t.astype(bool)] = 1 - self.smoothing
            t = ndf.astype(np.float32)
        return x, t


class FocalLoss(Loss):
    def __init__(self, n_classes: int, gamma: float=1.0, dx: float=1e-5):
        assert isinstance(n_classes, int) and n_classes > 1
        super().__init__("fl", n_classes=n_classes, target_dtype=np.int32, is_higher_better=False)
        self.gamma = gamma
        self.dx    = dx
    def check(self, x: np.ndarray, t: np.ndarray):
        super().check(x, t)
        assert len(x.shape) == 2
        assert np.isin(np.unique(t), np.arange(self.n_classes)).sum() == self.n_classes
    def loss(self, x: np.ndarray, t: np.ndarray):
        x, t = self.convert(x, t)
        x = softmax(x)
        x = np.clip(x, self.dx, 1 - self.dx * (self.n_classes - 1))
        t = np.identity(x.shape[1])[t].astype(bool)
        return -1 * ((1 - x[t]) ** self.gamma) * np.log(x[t])
    def gradhess(self, x: np.ndarray, t: np.ndarray):
        """
        see: https://hackmd.io/Kd14LHwETwOXLbzh_adpQQ
        """
        x, t = self.convert(x, t)
        x = softmax(x)
        x = np.clip(x, self.dx, 1 - self.dx * (self.n_classes - 1))
        t = np.identity(x.shape[1])[t].astype(bool)
        yk = x[t].reshape(-1, 1)
        grad    = x * ((1 - yk) ** (self.gamma - 1)) * (-self.gamma * yk * np.log(yk) + 1 - yk)
        grad[t] = (((1 - yk) ** self.gamma) * (yk + self.gamma * yk * np.log(yk) - 1)).reshape(-1)
        hess    = x * ((1 - yk) ** (self.gamma - 2)) * (
            (1 - x - yk + self.gamma * x * yk) * (1 - yk - self.gamma * yk * np.log(yk)) + \
            x * yk * (1 - yk) * (self.gamma * np.log(yk) + self.gamma + 1)
        )
        hess[t] = (yk * ((1 - yk) ** self.gamma) * (self.gamma * np.log(yk) * (1 - yk - self.gamma * yk) + 1 - yk - 2 * self.gamma * yk + 2 * self.gamma)).reshape(-1)
        return grad, hess


class MSELoss(Loss):
    def __init__(self, n_classes: int=1):
        assert isinstance(n_classes, int) and n_classes > 0
        super().__init__("mse", n_classes=n_classes, target_dtype=np.float32, is_higher_better=False)
    def loss(self, x: np.ndarray, t: np.ndarray):
        x, t = self.convert(x, t)
        return 0.5 * (x - t) ** 2
    def gradhess(self, x: np.ndarray, t: np.ndarray):
        x, t = self.convert(x, t)
        grad = x - t
        hess = np.ones(x.shape).astype(np.float32)
        return grad, hess


class MAELoss(Loss):
    def __init__(self, n_classes: int=1):
        assert isinstance(n_classes, int) and n_classes > 0
        super().__init__("mae", n_classes=n_classes, target_dtype=np.float32, is_higher_better=False)
    def loss(self, x: np.ndarray, t: np.ndarray):
        x, t = self.convert(x, t)
        return np.abs(x - t)
    def gradhess(self, x: np.ndarray, t: np.ndarray):
        x, t = self.convert(x, t)
        grad = np.ones(x.shape).astype(np.float32)
        grad[x < t] = -1
        hess = np.zeros(x.shape).astype(np.float32)
        return grad, hess


class HuberLoss(Loss):
    def __init__(self, n_classes: int=1, beta: float=1.0):
        assert isinstance(n_classes, int) and n_classes > 0
        super().__init__("huber", n_classes=n_classes, target_dtype=np.float32, is_higher_better=False)
        self.beta = beta
    def loss(self, x: np.ndarray, t: np.ndarray):
        x, t = self.convert(x, t)
        boolwk = (np.abs(x - t) < self.beta)
        loss   = np.abs(x - t) - 0.5 * self.beta
        loss[boolwk] = (0.5 * (x - t) ** 2)[boolwk] / self.beta
        return loss
    def gradhess(self, x: np.ndarray, t: np.ndarray):
        x, t = self.convert(x, t)
        boolwk = (np.abs(x - t) < self.beta)
        grad = np.ones(x.shape).astype(np.float32)
        grad[boolwk] = (x - t)[boolwk] / self.beta
        hess = np.zeros(x.shape).astype(np.float32)
        grad[boolwk] = 1 / self.beta
        return grad, hess


class Accuracy(Loss):
    def __init__(self, top_k: int=1):
        assert isinstance(top_k, int) and top_k >= 1
        self.top_k = top_k
        super().__init__(f"acc_top{self.top_k}", n_classes=0, target_dtype=np.float32, is_higher_better=True)
        self.conv_shape_t = lambda x: x
    def check(self, x: np.ndarray, t: np.ndarray):
        super().check(x, t)
        assert len(x.shape) == 2
        assert len(t.shape) in [1, 2]
        if len(t.shape) == 2:
            self.conv_shape_t = lambda x: np.argmax(x, axis=1)
    def convert(self, x: np.ndarray, t: np.ndarray):
        x, t = super().convert(x, t)
        t = self.conv_shape_t(t)
        return x, t
    def loss(self, x: np.ndarray, t: np.ndarray):
        x, t = self.convert(x, t)
        x = np.argsort(x, axis=1)[:, ::-1]
        x = x[:, :self.top_k]
        return (x == t.reshape(-1, 1)).sum(axis=1)
    def gradhess(self, x: np.ndarray, t: np.ndarray):
        raise Exception(f"class: {self.__class__.__name__} has not gradient and hessian.")