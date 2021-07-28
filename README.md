# kklgb
- This package is wrapper package for lightgbm( https://github.com/microsoft/LightGBM ).
- You can do training with some useful functions. 

## Installation
```
pip install git+https://github.com/kazukingh01/kklgb.git
```

## Training examples with using Model Zoo
code samples is in "https://github.com/kazukingh01/kklgb/tree/main/tests"

### Run test code
```
# change working directory.
mkdir work
cd ./work
git clone https://github.com/kazukingh01/kklgb.git
cd ./kklgb/tests/
python train_lgbmapi_multi_class_custom.py
```

## Simple Usage
```
python
>>> import numpy as np
>>> from kklgb.model import train
>>> from kklgb.loss import Accuracy
>>> n_data    = 1000
>>> n_classes = 5
>>> x_train = np.random.rand(n_data, 100)
>>> x_valid = np.random.rand(n_data, 100)
>>> y_train = np.random.randint(0, n_classes, n_data)
>>> y_valid = np.random.randint(0, n_classes, n_data)
>>> model = train(
        dict(learning_rate=0.1, num_tree=10 , num_class=n_classes), x_train, y_train, 
        loss_func="multiclass", loss_func_eval=Accuracy(top_k=2),
        x_valid=x_valid, y_valid=y_valid
    )
2021-07-28 16:03:31,791 - kklgb.model - train - INFO : START
2021-07-28 16:03:31,791 - kklgb.model - _train - INFO : START
2021-07-28 16:03:31,792 - kklgb.model - _train - INFO : set normal dataset. x_train shape: (1000, 100), y_train shape: (1000,)
2021-07-28 16:03:31,792 - kklgb.model - set_callbacks - INFO : set callbacks: record_evaluation, print_evaluation
2021-07-28 16:03:31,792 - kklgb.model - _train - INFO : params: {'learning_rate': 0.1, 'num_tree': 10, 'num_class': 5, 'verbosity': -1, 'metric': None, 'objective': 'multiclass'}, dataset: <lightgbm.basic.Dataset object at 0x7f2dedab19d0>, fobj: None, feval: []
/home/pasona/10.git/kklgb/venv/lib/python3.7/site-packages/lightgbm/engine.py:148: UserWarning: Found `num_tree` in params. Will use it instead of argument
  _log_warning("Found `{}` in params. Will use it instead of argument".format(alias))
2021-07-28 16:06:18,450 - kklgb.model - _callback - INFO : [1]	train's multi_logloss: 1.50009	train's acc_top2: 0.844	valid0's multi_logloss: 1.61466	valid0's acc_top2: 0.385
2021-07-28 16:06:18,464 - kklgb.model - _callback - INFO : [2]	train's multi_logloss: 1.39363	train's acc_top2: 0.943	valid0's multi_logloss: 1.61824	valid0's acc_top2: 0.41
2021-07-28 16:06:18,471 - kklgb.model - _callback - INFO : [3]	train's multi_logloss: 1.29949	train's acc_top2: 0.977	valid0's multi_logloss: 1.62828	valid0's acc_top2: 0.404
2021-07-28 16:06:18,481 - kklgb.model - _callback - INFO : [4]	train's multi_logloss: 1.20866	train's acc_top2: 0.991	valid0's multi_logloss: 1.63263	valid0's acc_top2: 0.405
2021-07-28 16:06:18,491 - kklgb.model - _callback - INFO : [5]	train's multi_logloss: 1.12912	train's acc_top2: 0.995	valid0's multi_logloss: 1.63772	valid0's acc_top2: 0.402
2021-07-28 16:06:18,497 - kklgb.model - _callback - INFO : [6]	train's multi_logloss: 1.05006	train's acc_top2: 0.996	valid0's multi_logloss: 1.64419	valid0's acc_top2: 0.409
2021-07-28 16:06:18,504 - kklgb.model - _callback - INFO : [7]	train's multi_logloss: 0.975304	train's acc_top2: 0.998	valid0's multi_logloss: 1.65043	valid0's acc_top2: 0.403
2021-07-28 16:06:18,514 - kklgb.model - _callback - INFO : [8]	train's multi_logloss: 0.908711	train's acc_top2: 1	valid0's multi_logloss: 1.65558	valid0's acc_top2: 0.381
2021-07-28 16:06:18,523 - kklgb.model - _callback - INFO : [9]	train's multi_logloss: 0.848215	train's acc_top2: 1	valid0's multi_logloss: 1.66483	valid0's acc_top2: 0.39
2021-07-28 16:06:18,529 - kklgb.model - _callback - INFO : [10]	train's multi_logloss: 0.789673	train's acc_top2: 1	valid0's multi_logloss: 1.66857	valid0's acc_top2: 0.389
2021-07-28 16:06:18,531 - kklgb.model - _train - INFO : END
2021-07-28 16:06:18,531 - kklgb.model - train - INFO : END
>>> model.predict(x_valid)
array([[0.12326449, 0.30881188, 0.27225069, 0.19363369, 0.10203924],
       [0.19689639, 0.34617741, 0.1030561 , 0.19454431, 0.15932579],
       [0.22655402, 0.17707429, 0.25159177, 0.16018043, 0.18459949],
       ...,
       [0.20212842, 0.28939344, 0.1582116 , 0.16743092, 0.18283563],
       [0.18701638, 0.15592378, 0.27234954, 0.23948643, 0.14522387],
       [0.11100745, 0.434116  , 0.17652652, 0.13723243, 0.1411176 ]])
>>> 
```