import numpy as np
import optuna
from optuna import Trial
from optuna.samplers import TPESampler
from sklearn.metrics import accuracy_score
from sklearn.model_selection import cross_val_score
from xgboost import XGBClassifier

from data.data_loader import load_combined_data
from features.features_manager import features_normalize
from features.get_feature_names import get_feature_names


def get_xy_4_xgboost(df):
    # 转成numpy.ndarray
    features_array = df.iloc[:, :].values
    # 前一日的特征
    x = features_array[:-1, :-1]
    # 后一天的标签
    y = features_array[1:, -1]
    return [x, y]


# 转换成训练数据
def transform_data_4_xgboost(raw_df):
    normalize_train_df, normalize_test_df = features_normalize(raw_df)
    x_train, y_train = get_xy_4_xgboost(normalize_train_df)
    x_test, y_test = get_xy_4_xgboost(normalize_test_df)
    return [x_train, y_train, x_test, y_test]


def fit_predict(params, train_X, train_y, test_X, test_y):
    model = XGBClassifier(**params)
    model.fit(train_X, train_y)
    pred_y = model.predict(test_X)
    accuracy = accuracy_score(test_y, pred_y)
    return accuracy


def objective(trial: Trial, train_X, train_y, test_X, test_y) -> float:
    params = {
        "n_estimators": trial.suggest_int('n_estimators', 0, 1000),
        'max_depth': trial.suggest_int('max_depth', 2, 25),
        'reg_alpha': trial.suggest_int('reg_alpha', 0, 10),
        'reg_lambda': trial.suggest_int('reg_lambda', 0, 10),
        'min_child_weight': trial.suggest_int('min_child_weight', 0, 20),
        'gamma': trial.suggest_int('gamma', 0, 5),
        'learning_rate': trial.suggest_loguniform('learning_rate', 0.0001, 0.5),
        'colsample_bytree': trial.suggest_discrete_uniform('colsample_bytree', 0.1, 1, 0.01),
        'nthread': -1,
        'scale_pos_weight': trial.suggest_int('scale_pos_weight', 1, 10),
        'random_state': trial.suggest_int('random_state', 1, 30),
        'subsample': trial.suggest_float('subsample', 0.5, 0.9)
    }
    model = XGBClassifier(**params)

    model.fit(train_X, train_y)

    return cross_val_score(model, test_X, test_y).mean()


if __name__ == '__main__':
    ts_code = "000651.SZ"
    start_time = "20100101"
    # 调用API获取原始数据
    raw_df = load_combined_data(ts_code, start_time)
    print(raw_df)
    train_X, train_y, test_X, test_y = transform_data_4_xgboost(raw_df)
    study = optuna.create_study(direction='maximize', sampler=TPESampler())
    study.optimize(lambda trial: objective(trial, train_X, train_y, test_X, test_y), n_trials=30)
    print(study.best_params)
    # study = optuna.create_study(direction='minimize')
    # study.optimize(lambda trial: objective_select_feature(trial, train_X, train_y, test_X, test_y), n_trials=100)
