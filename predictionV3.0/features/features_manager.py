from sklearn import preprocessing
import features.generate_features as gf
import features.get_feature_names as gfn


# 移除不要的特征
def remove_features(df):
    # 深度拷贝一份 不改变原来的df
    new_df = df.copy(deep=True)
    # 获取需要保留的字段名
    feature_names = gfn.get_feature_names()
    new_df_columns = new_df.columns
    for i in range(len(new_df_columns)):
        if new_df_columns[i] not in feature_names:
            new_df.drop(new_df_columns[i], 1, inplace=True)
    return new_df


# 归一化特征值
def normalize_data(target_df, base_df):
    # 深度拷贝一份 不改变原来的df
    normalize_df = target_df.copy(deep=True)
    min_max_scaler = preprocessing.MinMaxScaler()
    columns = normalize_df.columns
    for index in range(len(columns)):
        column_name = columns[index]
        min_max_scaler.fit(base_df[column_name].values.reshape(-1, 1))
        normalize_df[column_name] = min_max_scaler.transform(normalize_df[column_name].values.reshape(-1, 1))
    return normalize_df


def update_features(df):
    # 生成一些新特征
    new_df = gf.generate_features(df)
    # 移除不需要的特征
    clean_df = remove_features(new_df)
    return clean_df


def features_normalize(raw_df):
    # 生成新特征，去除无用特征
    clean_features_df = update_features(raw_df)
    # 拆分成一部分训练，一部分验证
    train_df, test_df = split_train_test(clean_features_df)
    # 归一化特征值
    normalize_train_df = normalize_data(train_df, train_df)
    # 归一化的基准是训练集
    normalize_test_df = normalize_data(test_df, train_df)
    return [normalize_train_df, normalize_test_df]


def split_train_test(clean_features_df):
    # 训练数据要与测试数据分开归一化
    test_range = 22 * 3
    train_df = clean_features_df.iloc[:-test_range, :]
    test_df = clean_features_df.iloc[-test_range:, :]
    return [train_df, test_df]