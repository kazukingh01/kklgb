import numpy as np
from typing import List
from kklgb.util.com import check_type_list
from kklgb.util.functions import sigmoid, softmax


__all__ = [
    "Loss",
    "BinaryCrossEntropyLoss",
    "CrossEntropyLoss",
    "CrossEntropyLossArgmax",
    "CategoricalCrossEntropyLoss",
    "FocalLoss",
    "MSELoss",
    "MAELoss",
    "HuberLoss",
    "Accuracy",
    "LogitMarginL1Loss",
    "MultiTaskLoss",
    "MultiTaskEvalLoss",
    "CrossEntropyNDCGLoss",
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
        if self.n_classes > 1: assert x.shape[1] == self.n_classes
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


class CrossEntropyLossArgmax(Loss):
    def __init__(self, n_classes: int, dx: float=1e-5, target_dtype: np.dtype=np.float32):
        assert isinstance(n_classes, int) and n_classes > 1
        super().__init__("cemax", n_classes=n_classes, target_dtype=target_dtype, is_higher_better=False)
        self.dx = dx
        self.conv_t_sum = lambda x: 1.0
        self.indexes    = None
    def check(self, x: np.ndarray, t: np.ndarray):
        super().check(x, t)
        if self.name == "cemax":
            assert len(x.shape) == 2
            assert len(t.shape) in [1, 2]
            if len(t.shape) == 2: assert x.shape[1] == t.shape[1] == self.n_classes
            else:                 assert x.shape[1] == self.n_classes
            self.indexes = np.arange(t.shape[0], dtype=int)
    def loss(self, x: np.ndarray, t: np.ndarray):
        x, t = self.convert(x, t)
        if len(t.shape) == 2: t = np.argmax(t, axis=1)
        else:                 t = t.astype(np.int32)
        x = softmax(x)
        x = np.clip(x, self.dx, 1 - self.dx * (self.n_classes - 1))
        x = x[np.arange(t.shape[0], dtype=int), t]
        return (-1 * np.log(x))
    def gradhess(self, x: np.ndarray, t: np.ndarray):
        raise Exception(f"class: {self.__class__.__name__} has not gradient and hessian.")


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


class LogitMarginL1Loss(Loss):
    def __init__(
        self, n_classes: int, alpha: float=0.1, margin: float=10.0, 
        dx: float=1e-5, target_dtype: np.dtype=np.float32
    ):
        """
        https://arxiv.org/pdf/2111.15430.pdf
        https://github.com/by-liu/MbLS/blob/167c0267b7a0ae29b255d44af0589a88af4d2410/calibrate/losses/logit_margin_l1.py
        """
        assert isinstance(n_classes, int) and n_classes > 1
        assert isinstance(alpha,  float) and alpha  > 0
        assert isinstance(margin, float) and margin > 0
        self.alpha  = alpha
        self.margin = margin
        super().__init__(f"ce_margin_{self.alpha}_{self.margin}", n_classes=n_classes, target_dtype=target_dtype, is_higher_better=False)
        self.dx = dx
        self.conv_t_sum = lambda x: 1.0
    def check(self, x: np.ndarray, t: np.ndarray):
        super().check(x, t)
        assert len(x.shape) == 2
        assert len(t.shape) == 2
        assert x.shape[1] == t.shape[1] == self.n_classes
        if ((t.sum(axis=1) / self.dx).round(0).astype(np.int32) == int(round(1 / self.dx, 0))).sum() != t.shape[0]:
            # If the sum of "t" is not equal to 1 (In other words, if "t" is not a probability)
            self.conv_t_sum = lambda x: x.sum(axis=1).reshape(-1, 1)
    def loss(self, x: np.ndarray, t: np.ndarray):
        x, t = self.convert(x, t)
        x_margin = np.clip(np.max(x, axis=1).reshape(-1, 1) - x - self.margin, 0.0, None)
        x = softmax(x)
        x = np.clip(x, self.dx, 1 - self.dx * (self.n_classes - 1))
        loss_ce     = (-1 * t * np.log(x)).sum(axis=1)
        loss_margin = self.alpha * np.sum(x_margin, axis=1)
        return loss_ce + loss_margin
    def gradhess(self, x: np.ndarray, t: np.ndarray):
        x, t = self.convert(x, t)
        x_margin = np.max(x, axis=1).reshape(-1, 1) - x - self.margin
        x_margin = (x_margin > 0).astype(x.dtype)
        x_margin[np.arange(x_margin.shape[0]), np.argmax(x, axis=1)] = 0
        x = softmax(x)
        x = np.clip(x, self.dx, 1 - self.dx * (self.n_classes - 1))
        t_sum = self.conv_t_sum(t)
        grad  = t_sum * x - t - (self.alpha * x_margin)
        hess  = t_sum * x * (1 - x)
        return grad, hess


class MultiTaskLoss(Loss):
    def __init__(self, losses: List[Loss], target_dtype: np.dtype=np.float32, weight: List[float]=None):
        assert check_type_list(losses, Loss)
        if weight is None: weight = [1.0] * len(losses)
        assert check_type_list(weight, float)
        indexes_loss = np.cumsum([x.n_classes for x in losses])
        super().__init__(f"multi_loss", n_classes=int(indexes_loss[-1]), target_dtype=target_dtype, is_higher_better=False)
        self.indexes_loss = indexes_loss
        self.losses       = losses
        self.weight       = weight
    def check(self, x: np.ndarray, t: np.ndarray):
        assert x.shape == t.shape
        super().check(x, t)
    def convert(self, x: np.ndarray, t: np.ndarray):
        t = t.astype(self.target_dtype)
        if self.is_check:
            self.check(x, t)
            self.is_check = False
        x = np.hsplit(x, self.indexes_loss[:-1])
        t = np.hsplit(t, self.indexes_loss[:-1])
        return x, t
    def loss(self, x: np.ndarray, t: np.ndarray):
        loss = np.zeros(x.shape[0], dtype=float)
        x, t = self.convert(x, t)
        for _x, _t, loss_func, _w in zip(x, t, self.losses, self.weight):
            loss += _w * loss_func.loss(_x, _t)
        return loss
    def gradhess(self, x: np.ndarray, t: np.ndarray):
        x, t = self.convert(x, t)
        list_grad, list_hess = [], []
        for _x, _t, loss_func, _w in zip(x, t, self.losses, self.weight):
            grad, hess = loss_func.gradhess(_x, _t)
            grad, hess = _w * grad, _w * hess
            list_grad.append(grad)
            list_hess.append(hess)
        return np.concatenate(list_grad, axis=1), np.concatenate(list_hess, axis=1)


class MultiTaskEvalLoss(Loss):
    def __init__(self, loss_func: Loss, indexes_loss: List[int]):
        assert isinstance(loss_func, Loss)
        assert check_type_list(indexes_loss, int)
        super().__init__(loss_func.name, n_classes=loss_func.n_classes, target_dtype=loss_func.target_dtype, is_higher_better=loss_func.is_higher_better)
        self.loss_func    = loss_func
        self.indexes_loss = indexes_loss
    def check(self, x: np.ndarray, t: np.ndarray):
        assert x.shape == t.shape
    def convert(self, x: np.ndarray, t: np.ndarray):
        if self.is_check:
            self.check(x, t)
            self.is_check = False
        x, t = x[:, self.indexes_loss], t[:, self.indexes_loss]
        return x, t
    def loss(self, x: np.ndarray, t: np.ndarray):
        x, t = self.convert(x, t)
        return self.loss_func.loss(x, t)
    def gradhess(self, x: np.ndarray, t: np.ndarray):
        raise Exception(f"class: {self.__class__.__name__} has not gradient and hessian.")


class CrossEntropyNDCGLoss(Loss):
    def __init__(self, n_classes: int, eta: List[float]=None):
        """
        https://arxiv.org/pdf/1911.09798.pdf
        """
        super().__init__("xendcg", n_classes=n_classes, target_dtype=np.float32, is_higher_better=True)
        if eta is not None:
            assert check_type_list(eta, [float, int])
            self.eta = np.array(eta).astype(float)
        else:
            self.eta = 0.0
    def check(self, x: np.ndarray, t: np.ndarray):
        assert x.shape == t.shape
        if isinstance(self.eta, np.ndarray):
            assert x.shape[-1] == self.eta.shape[-1]
        super().check(x, t)
    def loss(self, x: np.ndarray, t: np.ndarray):
        x, t = self.convert(x, t)
        return self.NDCG(x, t)
    def gradhess(self, x: np.ndarray, t: np.ndarray):
        x, t = self.convert(x, t)
        ro   = softmax(x)
        phi  = np.power(2, t) - self.eta
        phi  = phi / phi.sum(axis=1).reshape(-1, 1)
        grad = -phi + ro
        hess = ro * (1 - ro)
        return grad, hess
    @classmethod
    def DCG(cls, x: np.ndarray, t: np.ndarray):
        x_rank = (np.argsort(np.argsort(-x, axis=-1), axis=1) + 1)
        return ((np.power(2, t) - 1) / np.log2(1 + x_rank)).sum(axis=1)
    @classmethod
    def NDCG(cls, x: np.ndarray, t: np.ndarray):
        return __class__.DCG(x, t) / __class__.DCG(t, t)
