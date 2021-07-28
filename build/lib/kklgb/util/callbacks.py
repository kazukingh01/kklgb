from typing import List
from lightgbm.callback import EarlyStopException, _format_eval_result
from kklgb.util.logger import MyLogger


__all__ = [
    "callback_model_save",
    "callback_stop_training",
    "callback_best_iter",
    "callback_lr_schedule",
    "print_evaluation"
]


def callback_model_save(save_interval: int):
    def _callback(env):
        if (env.iteration % save_interval) == 0:
            env.model.save_model(f'model_{env.iteration}.txt')
    _callback.order = 110
    return _callback

def callback_stop_training(stopping_val: float, stopping_rounds: int, logger: MyLogger):
    """
    If training loss does not reach the threshold, it will be terminated first.
    """
    def _callback(env):
        _, _, result, _ = env.evaluation_result_list[0]
        if isinstance(stopping_rounds, int) and env.iteration >= stopping_rounds and result > stopping_val:
            logger.info(f'stop training. iteration: {env.iteration}, score: {result}')
            raise EarlyStopException(env.iteration, env.evaluation_result_list)
    _callback.order = 150
    return _callback

def callback_best_iter(dict_eval: dict, stopping_rounds: int, logger: MyLogger):
    """
    Determine best iteration for valid_1
    """
    def _init(env):
        dict_eval["best_iter"]  = 0
        dict_eval["eval_name"]  = ""
        dict_eval["best_score"] = float("inf")
        dict_eval["best_result_list"] = []
    def _callback(env):
        if not dict_eval:
            _init(env)
        for data_name, eval_name, result, _ in env.evaluation_result_list:
            if data_name == "valid_1":
                if dict_eval["best_score"] > result:
                    dict_eval["best_score"] = result
                    dict_eval["eval_name"]  = eval_name
                    dict_eval["best_iter"]  = env.iteration
                    dict_eval["best_result_list"] = env.evaluation_result_list
                break
        if isinstance(stopping_rounds, int) and env.iteration - dict_eval["best_iter"] >= stopping_rounds:
            logger.info(f'early stopping. iteration: {dict_eval["best_iter"]}, score: {dict_eval["best_score"]}')
            raise EarlyStopException(dict_eval["best_iter"], dict_eval["best_result_list"])
    _callback.order = 200
    return _callback

def callback_lr_schedule(lr_steps: List[int], lr_decay: float=0.2):
    def _callback(env):
        if int(env.iteration - env.begin_iteration) in lr_steps:
            lr = env.params.get("learning_rate", None)
            dictwk = {"learning_rate": lr * lr_decay}
            env.model.reset_parameter(dictwk)
            env.params.update(dictwk)
    _callback.before_iteration = True
    _callback.order = 100
    return _callback

def print_evaluation(logger: MyLogger, period=1, show_stdv=True):
    def _callback(env):
        if period > 0 and env.evaluation_result_list and (env.iteration + 1) % period == 0:
            result = '\t'.join([_format_eval_result(x, show_stdv) for x in env.evaluation_result_list])
            logger.info('[%d]\t%s' % (env.iteration + 1, result))
    _callback.order = 10
    return _callback
