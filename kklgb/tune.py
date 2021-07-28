from typing import Callable
import numpy as np
from optuna.integration import lightgbm as lgbtune

from kklgb.model import _train


__all__ = [
    "autotuner",
    "tune",
]


def autotuner(
    params: dict, x_train: np.ndarray, y_train: np.ndarray, loss_func: str, *args, 
    x_valid: np.ndarray=None, y_valid: np.ndarray=None, 
    loss_func_grad: Callable[[float, float], float]=None, 
    loss_func_eval: Callable[[float, float], float]=None, 
    use_custom_dataset: bool=False,
    **kwargs
):
    logger.info("START")
    obj = _train(
        params, x_train, y_train, *args, 
        x_valid=x_valid, y_valid=y_valid,
        loss_func=loss_func, loss_func_grad=loss_func_grad,
        loss_func_eval=loss_func_eval,
        use_custom_dataset=use_custom_dataset,
        func_train=lgbtune.train,
        **kwargs
    )
    logger.info("END")
    return obj


def tune(
    x_train: np.ndarray, y_train: np.ndarray, 
    loss_func, n_trials: int, params_add: dict=None,
    x_valid: np.ndarray=None, y_valid: np.ndarray=None, 
    loss_func_grad: Callable[[float, float], float]=None, 
    loss_func_eval: Callable[[float, float], float]=None, 
    use_custom_dataset: bool=False, optuna_db: str=None, 
    func_params={}, **kwargs
):
    """
    Usage::
        df_optuna, dict_param_ret = tune(
            x_train, y_train, "multiclass", 500,
            x_valid=x_valid, y_valid=y_valid,
            params_add={"num_class": 3, "verbosity": -1, "class_weight":"balanced", "n_jobs": 1},
            early_stopping_rounds=50,
        )
    """
    logger.info("START")
    import optuna
    from kkutils.util.ml.hypara import create_optuna_params
    params = {
        "task"             : ["const", "train"],
        'verbosity'        : ["const", -1],
        'boosting'         : ["const", "gbdt"],
        "n_jobs"           : ["const", 1],
        "random_seed"      : ["const", 1],
        "learning_rate"    : ["const", 0.03],
        "max_depth"        : ["const", -1],
        "num_iterations"   : ["const", 1000],
        'bagging_freq'     : ["const", 1],
        'num_leaves'       : ["const", 100],
        'lambda_l1'        : ["log", 1e-2, 1e2],
        'lambda_l2'        : ["log", 1e-1, 1e3],
        "min_hessian"      : ["log", 1e-4, 1e2],
        'feature_fraction' : ["float", 0.01, 0.8],
        'bagging_fraction' : ["float", 0.01, 0.8],
        'min_child_samples': ["int", 1, 100],
    }
    if params_add is not None:
        for x, y in params_add.items():
            if isinstance(y, list) and y[0] in ["const", "log", "float", "int", "step", "category"]:
                params[x] = y
            else:
                params[x] = ["const", y]
    if len(y_train.shape) == 2:
        params["num_class"] = ["const", y_train.shape[-1]]
    def objective(
        trial, params=None, x_train=None, y_train=None,
        x_valid=None, y_valid=None, 
        loss_func=None, loss_func_grad=None, loss_func_eval=None, 
        use_custom_dataset=None, func_params={}, **kwargs
    ):
        _params = create_optuna_params(params, trial)
        gbm = _train(
            _params, x_train=x_train, y_train=y_train,
            x_valid=x_valid, y_valid=y_valid,
            loss_func=loss_func, loss_func_grad=loss_func_grad,
            loss_func_eval=loss_func_eval,
            use_custom_dataset=use_custom_dataset,
            func_train=lgb.train,
            **kwargs
        )
        func = func_embed(loss_func_eval, calc_type="mean", **func_params)
        val  = func(gbm.predict(x_valid), y_valid)
        return val
    _objective = partial(
        objective, 
        params=params,
        x_train=x_train, y_train=y_train,
        x_valid=x_valid, y_valid=y_valid,
        loss_func=loss_func, loss_func_grad=loss_func_grad,
        loss_func_eval=loss_func_eval if loss_func_eval is not None else loss_func,
        use_custom_dataset=use_custom_dataset, func_params=func_params,
        **kwargs
    )
    study = None
    if optuna_db is not None:
        study_name = os.path.basename(optuna_db).replace(".db", "")
        study = optuna.load_study(
            study_name=study_name, storage='sqlite:///' + optuna_db
        )
    else:
        study = optuna.create_study(
            study_name='optuna_'+datetime.datetime.now().strftime("%Y%m%d%H%M%S"),
            storage='sqlite:///optuna_'+datetime.datetime.now().strftime("%Y%m%d%H%M%S")+'.db',
        )
    # パラメータ探索
    study.optimize(_objective, n_trials=n_trials)
    # 結果を保存する
    df_optuna = pd.DataFrame()
    for i_trial in study.trials:
        sewk = pd.Series(i_trial.params)
        sewk["value"]  = i_trial.value
        df_optuna = df_optuna.append(sewk, ignore_index=True)
    dict_param_ret = {}
    for key, val in  params.items():
        if val[0] == "const": dict_param_ret[key] = val[-1]
    for key, val in  study.best_params.items():
        dict_param_ret[key] = val
    logger.info("END")
    return df_optuna, dict_param_ret
