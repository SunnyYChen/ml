import traceback
import data.data_loader as dl
import features.features_manager as fm
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score


def get_xy_4_xgboost(df):
    # 转成numpy.ndarray
    features_array = df.iloc[:, :].values
    # 前一日的特征
    x = features_array[:-1, :-1]
    # print("x")
    # print(x)
    # 后一天的标签
    y = features_array[1:, -1]
    # print("y")
    # print(y)
    return [x, y]


# 转换成训练数据
def transform_data_4_xgboost(raw_df):
    normalize_train_df, normalize_test_df = fm.features_normalize(raw_df)
    x_train, y_train = get_xy_4_xgboost(normalize_train_df)
    x_test, y_test = get_xy_4_xgboost(normalize_test_df)
    return [x_train, y_train, x_test, y_test]


def build_xg_model():
    model = XGBClassifier(
        n_estimators=193,  # 树的个数--1000棵树建立xgboost
        max_depth=14,  # 树的深度
        reg_alpha=1,
        reg_lambda=9,
        min_child_weight=6,  # 叶子节点最小权重
        gamma=2,  # 惩罚项中叶子结点个数前的参数
        learning_rate=0.019,
        colsample_btree=0.97,  # 随机选择80%特征建立决策树
        scale_pos_weight=7,  # 解决样本个数不平衡的问题
        random_state=12,  # 随机数
        subsample=0.69,  # 随机选择80%样本建立决策树
        objective='multi:softmax',  # 指定损失函数
        num_class=2)
    return model


def train_xg_model(x_train, y_train, x_test, y_test):
    model = build_xg_model()
    model.fit(x_train,
              y_train,
              eval_set=[(x_test, y_test)],
              eval_metric="mlogloss",
              early_stopping_rounds=60,
              verbose=True)
    return model


def prediction(model, x_test, y_test):
    y_pred = model.predict(x_test)
    # print("prediction:")
    # print(y_pred)
    # print("actual :")
    # print(y_test)
    accuracy = accuracy_score(y_test, y_pred)
    print("accuarcy: %.2f%%" % (accuracy * 100.0))
    return accuracy


def batch_train():
    # "600519.SH", "000002.SZ", "002594.SZ", "603259.SH", "600436.SH", "603027.SH", "002475.SZ",
    #  "000651.SZ", "600031.SH", "002142.SZ", "600030.SH", "600362.SH", "600547.SH", "601111.SH", "002352.SZ",
    #  "002079.SZ", "600739.SH", "300331.SZ", "300382.SZ", "300628.SZ", "600844.SH", "300761.SZ"
    ts_codes = ["000651.SZ"]
    for ts_code in ts_codes:
        try:
            start_time = "20100101"
            # 调用API获取原始数据
            raw_df = dl.load_combined_data(ts_code, start_time)
            # 归一化、转换成模型接受的格式
            x_train, y_train, x_test, y_test = transform_data_4_xgboost(raw_df)
            print(ts_code)
            # 训练模型
            model = train_xg_model(x_train, y_train, x_test, y_test)
            prediction(model, x_test, y_test)
        except Exception as ex:
            print("出现如下异常%s" % ex)
            # 打印异常堆栈
            traceback.print_exc()


if __name__ == '__main__':
    batch_train()
