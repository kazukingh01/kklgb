import copy
from typing import List, Callable, Union
from functools import partial
import numpy as np
from sklearn.exceptions import NotFittedError

import lightgbm as lgb
from lightgbm import LGBMClassifier, LGBMRegressor, LGBMModel, Booster, Dataset
from lightgbm.callback import record_evaluation

# local package
from kklgb.loss import Loss
from kklgb.util.functions import softmax
from kklgb.util.lgbm import LGBCustomObjective, LGBCustomEval, calc_grad_hess
from kklgb.util.callbacks import callback_model_save, callback_stop_training, callback_best_iter, callback_lr_schedule, print_evaluation
from kklgb.util.com import check_type_list
from kklgb.util.logger import set_logger, MyLogger
logger = set_logger(__name__)


__all__ = [
    "KkLGBMModelBase",
    "KkLGBMClassifier",
    "KkLGBMRegressor",
    "_train",
    "train",
]


class KkLGBMModelBase(LGBMModel):
    """
    base class.
    Usege::
        model.fit(
            x_train, y_train,
            eval_set=(x_valid, y_valid, ),
            early_stopping_rounds=50
        )
    """
    def _fit(self, X, y, *argv, **kwargs):
        logger.info("START")
        self.dict_eval      = {}
        self.dict_eval_best = {}
        self.dict_eval_hist = []
        self.save_interval       = kwargs.get("save_interval")
        self.stopping_train_val  = kwargs.get("stopping_train_val")
        self.stopping_train_iter = kwargs.get("stopping_train_iter")
        if kwargs.get("verbose") is None:
            kwargs["verbose"] = int(self.n_estimators)
        if kwargs.get("eval_set") is None:
            kwargs["eval_set"] = [(X, y, ),]
        else:
            if   isinstance(kwargs.get("eval_set"), tuple):
                kwargs["eval_set"] = [(X, y, ), kwargs["eval_set"]]
            elif isinstance(kwargs.get("eval_set"), list):
                kwargs["eval_set"].insert(0, (X, y, ))
        if kwargs.get("eval_names") is None:
            kwargs["eval_names"] = []
            for i in range(len(kwargs["eval_set"])):
                if i == 0: kwargs["eval_names"].append("train")
                else:      kwargs["eval_names"].append(f"valid_{i}")
        else:
            if isinstance(kwargs.get("eval_names"), list):
                kwargs["eval_names"].insert(0, "train")
        kwargs = set_callbacks(
            kwargs, logger, self.dict_eval, 
            early_stopping_rounds=kwargs.get("early_stopping_rounds"), dict_eval_best=self.dict_eval_best,
            stopping_train_val=self.stopping_train_val, stopping_train_iter=self.stopping_train_iter,
            save_interval=self.save_interval
        )
        self.fit_common(X, y, *argv, **kwargs)
        self.dict_eval_hist.append(self.dict_eval.copy())
        try:
            if isinstance(self.best_iteration_, int) and self.save_interval is not None:
                base_step            = self.best_iteration_ - (self.best_iteration_ // self.save_interval)
                self.n_estimators    = base_step + 100
                kwargs["init_model"] = f"./model_{base_step}.txt"
                kwargs["callbacks"]  = [
                    record_evaluation(self.dict_eval), 
                    self.print_evaluation(logger),
                    self.callback_best_iter(self.dict_eval_best, kwargs.get("early_stopping_rounds"), logger),
                    self.callback_lr_schedule([base_step], lr_decay=0.2)
                ]
                logger.info(f'best model is {kwargs["init_model"]}. base_step: {base_step}, re-fitting: \n{self}')
                self.fit_common(X, y, *argv, **kwargs)
        except NotFittedError:
            # This error occurs when accessing "best_iteration_" if it is not fitting.
            pass
        logger.info("END")
    
    def fit_common(self, X, y, *argv, **kwargs):
        raise NotImplementedError
    
    def rm_objective(self):
        """
        Functions defined in lambda x: must be deleted before they can be pickleable
        """
        self.objective  = None
        self._objective = None
        self._fobj      = None


class KkLGBMClassifier(LGBMClassifier, KkLGBMModelBase):
    """
    Usage::
        >>> import numpy as np
        >>> from kklgb.model import KkLGBMClassifier
        >>> n_data  = 1000
        >>> x_train = np.random.rand(n_data, 100)
        >>> x_valid = np.random.rand(n_data, 100)
        >>> y_train = np.random.randint(0, 2, n_data)
        >>> y_valid = np.random.randint(0, 2, n_data)
        >>> model = KkLGBMClassifier(objective="binary")
        >>> model.fit(x_train, y_train, eval_set=(x_valid, y_valid, ), early_stopping_rounds=100)
        >>> model.predict_proba(x_valid)
        array([[0.61345901, 0.38654099],
            [0.21868721, 0.78131279],
            [0.69742278, 0.30257722],
            ...,
            [0.1238593 , 0.8761407 ],
            [0.74385392, 0.25614608],
            [0.83748049, 0.16251951]])
    """
    def fit(self, X, y, *argv, **kwargs):
        super()._fit(X, y, *argv, **kwargs)
    def fit_common(self, X, y, *argv, **kwargs):
        super().fit(X, y, *argv, **kwargs)


class KkLGBMRegressor(LGBMRegressor, KkLGBMModelBase):
    def fit(self, X, y, *argv, **kwargs):
        super()._fit(X, y, *argv, **kwargs)
    def fit_common(self, X, y, *argv, **kwargs):
        super().fit(X, y, *argv, **kwargs)


class KkBooster:
    def __init__(self):
        self.booster = None
    def fit(
        self, x_train: np.ndarray, y_train: np.ndarray, *args, params: dict=None,
        loss_func: Union[str, Callable]=None, loss_func_grad: Union[str, Callable]=None, 
        x_valid: np.ndarray=None, y_valid: np.ndarray=None, loss_func_eval: Union[str, Callable]=None, 
        **kwargs
    ):
        assert params is not None
        self.booster = train(
            params, x_train, y_train, *args,
            loss_func=loss_func, loss_func_grad=loss_func_grad, 
            x_valid=x_valid, y_valid=y_valid, loss_func_eval=loss_func_eval,
            **kwargs
        )
        self.set_parameter_after_training()
    def set_parameter_after_training(self):
        self.feature_importances_ = self.booster.feature_importance()
        if self.booster.params.get("num_class") is not None:
            self.classes_ = np.arange(self.booster.params.get("num_class"))
    def predict(self, data, *args, **kwargs):
        return self.booster.predict(data, *args, **kwargs)
    def predict_proba(self, data, *args, **kwargs):
        output = self.booster.predict(data, *args, **kwargs)
        return softmax(output)


class KkLgbDataset(Dataset):
    """
    Usage::
        >>> dataset = KkLgbDataset(x_train)
        >>> dataset.set_culstom_label(y_train)
    """
    def set_culstom_label(self, label: np.ndarray):
        self.label     = np.arange(label.shape[0]).astype(int)
        self.ndf_label = label
    def get_culstom_label(self, indexes: np.ndarray) -> np.ndarray:
        return self.ndf_label[indexes]


def set_callbacks(
    kwargs: dict, logger: MyLogger, dict_eval: dict, 
    early_stopping_rounds: int=None, eval_name: Union[int, str]=0, dict_eval_best: dict=None,
    stopping_train_val: float=None, stopping_train_iter: int=None,
    save_interval: int=None
):
    assert isinstance(kwargs,    dict)
    assert isinstance(dict_eval, dict)
    if kwargs.get("callbacks") is None:
        logger.info("set callbacks: record_evaluation, print_evaluation")
        kwargs["callbacks"] = [
            record_evaluation(dict_eval),
            print_evaluation(logger),
        ]
        if isinstance(early_stopping_rounds, int) and isinstance(dict_eval_best, dict):
            logger.info(f"set callbacks: callback_best_iter. params: early_stopping_rounds={early_stopping_rounds}, eval_name={eval_name}")
            kwargs["callbacks"].append(callback_best_iter(dict_eval_best, early_stopping_rounds, eval_name, logger))
        if isinstance(stopping_train_val, float) and isinstance(stopping_train_iter, dict):
            logger.info(f"set callbacks: callback_stop_training. params: stopping_train_val={stopping_train_val}, stopping_train_iter={stopping_train_iter}")
            kwargs["callbacks"].append(callback_stop_training(stopping_train_val, stopping_train_iter, logger))
        if isinstance(save_interval, float):
            logger.info("set callbacks: callback_model_save")
            kwargs["callbacks"].append(callback_model_save(save_interval))
    for x in ["save_interval", "stopping_train_val", "stopping_train_iter", "early_stopping_rounds", "early_stopping_name"]:
        if kwargs.get(x) is not None: del kwargs[x]
    return kwargs


def _train(
    params: dict, x_train: np.ndarray, y_train: np.ndarray, *args, 
    loss_func: Union[str, Callable]=None, loss_func_grad: Union[str, Callable]=None, 
    x_valid: np.ndarray=None, y_valid: np.ndarray=None, loss_func_eval: Union[str, Callable]=None, 
    func_train=lgb.train,
    **kwargs
):
    """
    Params::
        loss_func: custom loss or string
            string:
                see lightgbm: https://lightgbm.readthedocs.io/en/latest/Parameters.html#core-parameters
                binary, multiclass, regression_l1, huber, ...
            custom loss
        loss_func_grad:
            function return values with graddient and hessian
            If None and loss_func is Callable, calcurate grad and hess with scipy.
            If None and loss_func is String, use implemented in original.
        loss_func_eval:
            eval function or string.
            If Callable, use function "func_embed" with mean value
            If String, see lightgbm: https://lightgbm.readthedocs.io/en/latest/Parameters.html#metric-parameters
            If None:
                If loss_func is String, use loss_func.
                If loss_func is Callable, use "func_embed" embedded loss_func
    """
    logger.info("START")
    assert isinstance(params, dict)
    if params.get("verbosity")    is None: params["verbosity"]    = -1
    if kwargs.get("verbose_eval") is None: kwargs["verbose_eval"] = False
    # set training dataset
    assert len(y_train.shape) <= 2
    dataset = None
    is_custom_dataset = True if len(y_train.shape) == 2 else False
    if is_custom_dataset:
        logger.info(f"set custom dataset. x_train shape: {x_train.shape}, y_train shape: {y_train.shape}")
        dataset = KkLgbDataset(x_train)
        dataset.set_culstom_label(y_train)
    else:
        logger.info(f"set normal dataset. x_train shape: {x_train.shape}, y_train shape: {y_train.shape}")
        dataset = Dataset(x_train, label=y_train)
    # set validation dataset
    if not (isinstance(x_valid, list) or isinstance(x_valid, tuple)):
        x_valid = [] if x_valid is None else [x_valid]
        y_valid = [] if y_valid is None else [y_valid]
    list_dataset_valid = [dataset]
    for _x_valid, _y_valid in zip(x_valid, y_valid):
        if is_custom_dataset:
            list_dataset_valid.append(KkLgbDataset(_x_valid))
            list_dataset_valid[-1].set_culstom_label(_y_valid)
        else:
            list_dataset_valid.append(Dataset(_x_valid, label=_y_valid))
    # set objective
    ## set parameters: params["objective"], params["metric"], fobj, feval
    ## if public objective: set params["objective"] and params["metric"]
    ## if custom objective: set fobj and feval
    ## objective: str. https://lightgbm.readthedocs.io/en/latest/Parameters.html#objective
    ##    metric: str. https://lightgbm.readthedocs.io/en/latest/Parameters.html#metric
    fobj, feval, is_no_metric = None, [], False
    if params.get("metric") is None:
        params["metric"] = []
        is_no_metric     = True
    if params.get("objective") is None:
        if isinstance(loss_func, str):
            params["objective"] = loss_func
        elif isinstance(loss_func, Loss):
            assert loss_func_grad is None
            fobj =       LGBCustomObjective(loss_func.gradhess)
            feval.append(LGBCustomEval(loss_func, loss_func.name, is_higher_better=loss_func.is_higher_better))
        else:
            if loss_func_grad is None: loss_func_grad = partial(calc_grad_hess, loss_func=loss_func)
            fobj = LGBCustomObjective(loss_func_grad)
    if is_no_metric:
        if loss_func_eval is None:
            if params.get("objective") is not None:
                params["metric"] = None
        else:
            if isinstance(loss_func_eval, str):
                params["metric"].append(loss_func_eval)
            elif isinstance(loss_func_eval, Loss):
                feval.append(LGBCustomEval(loss_func_eval, loss_func_eval.name, is_higher_better=loss_func_eval.is_higher_better))
            elif check_type_list(loss_func_eval, [str, Loss]):
                for _loss in loss_func_eval:
                    if   isinstance(_loss, str):
                        params["metric"].append(_loss)
                    elif isinstance(_loss, Loss):
                        feval.append(LGBCustomEval(_loss, _loss.name, is_higher_better=_loss.is_higher_better))
            else:
                feval = loss_func_eval
    # set callbacks
    dict_eval, dict_eval_best = {}, {}
    kwargs = set_callbacks(
        kwargs, logger, dict_eval, 
        early_stopping_rounds=kwargs.get("early_stopping_rounds"), eval_name=kwargs.get("early_stopping_name"), 
        dict_eval_best=dict_eval_best,
        stopping_train_val=kwargs.get("stopping_train_val"), stopping_train_iter=kwargs.get("stopping_train_iter"),
        save_interval=kwargs.get("save_interval"),
    )
    evals_result = {} # metric history
    logger.info(f"params: {params}, dataset: {dataset}, fobj: {fobj}, feval: {feval}")
    obj = func_train(
        params, dataset, 
        valid_sets=list_dataset_valid, valid_names=["train"]+["valid"+str(i) for i in range(len(list_dataset_valid)-1)],
        fobj=fobj, feval=feval, evals_result=evals_result,
        **kwargs
    )
    logger.info("END")
    return obj


def train(
    params: dict, x_train: np.ndarray, y_train: np.ndarray, *args, 
    loss_func: Union[str, Callable]=None, loss_func_grad: Union[str, Callable]=None, 
    x_valid: np.ndarray=None, y_valid: np.ndarray=None, loss_func_eval: Union[str, Callable]=None, 
    **kwargs
):
    logger.info("START")
    obj = _train(
        params, x_train, y_train, *args,
        loss_func=loss_func, loss_func_grad=loss_func_grad,
        x_valid=x_valid, y_valid=y_valid, loss_func_eval=loss_func_eval,
        func_train=lgb.train,
        **kwargs
    )
    def predict_proba(data, *args, _model=None, **kwargs):
        output = _model.predict(data, *args, **kwargs)
        return softmax(output)
    obj.predict_proba = partial(predict_proba, _model=obj)
    logger.info("END")
    return obj