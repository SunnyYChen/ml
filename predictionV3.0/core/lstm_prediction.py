import traceback
import pandas as pd
import numpy as np
import features.features_manager as fm
from keras.callbacks import ModelCheckpoint
from keras.layers import Conv1D, MaxPooling1D
from keras.layers.core import Dense, Flatten, Dropout
from keras.layers.recurrent import LSTM
from keras.models import Sequential
import matplotlib.pyplot as plt
import backtrader.backtrader as bt


def show_line(model, x_test, y_test):
    p_test = model.predict(x_test)
    pt = pd.DataFrame(p_test)
    yt = pd.DataFrame(y_test)
    normalize_p = fm.normalize_data(pt, pt).values
    normalize_y = fm.normalize_data(yt, pt).values

    p = normalize_p.flatten()
    y = normalize_y.flatten()

    mat = [y, p]
    mat = np.transpose(mat)
    df = pd.DataFrame(mat)
    # print(df)
    data = df[[0, 1]]
    print('correlation:', data.corr().iat[0, 1])

    trade_date = np.arange(len(p))

    plt.plot(trade_date, y, c='green', label='real')
    plt.plot(trade_date, p, c='red', label='predict')
    plt.legend()
    plt.show()


def prediction(model, x_test, y_test):
    p = model.predict(x_test)
    # 变成一维数组
    y_test = y_test.flatten()
    p = p.flatten()
    print(p)
    print(y_test)
    # 涨跌趋势一致的数量
    count = 0
    countCorrectly = 0
    for index in range(1, len(p)):
        if p[index] - p[index - 1] > 0:
            count = count + 1
            if y_test[index] - y_test[index - 1] > 0:
                countCorrectly = countCorrectly + 1

    print("正确预测为涨的占比：", countCorrectly / count)
    return p


def get_xy_4_lstm(df, window):
    # 转成numpy
    array_df = df.values
    array_x = array_df[:, :-1]
    array_y = array_df[:, -1]
    x = []
    y = []
    for i in range(len(df) - window):
        window_data = array_x[i:(i + window)]
        after_window = array_y[i + window]
        x.append(window_data)
        y.append(after_window)
    return np.array(x), np.array(y)


# 转换成训练数据
def transform_data_4_lstm(raw_df, window):
    normalize_train_df, normalize_test_df = fm.features_normalize(raw_df)
    x_train, y_train = get_xy_4_lstm(normalize_train_df, window)
    x_test, y_test = get_xy_4_lstm(normalize_test_df, window)
    return [x_train, y_train, x_test, y_test]


def build_lstm_model(layers):
    d = 0.2
    model = Sequential()
    # layer[1]表示步长TIME_STEPS，layer[0]表示特征数量INPUT_DIM
    # model.add(Conv1D(256, kernel_size=5, input_shape=(layers[1], layers[0]), activation='relu'))
    # model.add(MaxPooling1D(pool_size=5))
    # model.add(Dropout(d))

    model.add(LSTM(512, input_shape=(layers[1], layers[0]), return_sequences=True))
    model.add(Dropout(d))

    model.add(LSTM(512, input_shape=(layers[1], layers[0]), return_sequences=False))
    model.add(Dropout(d))

    # 隐藏层,连接着带有64个神经元的全连接输出层
    model.add(Dense(128, kernel_initializer="uniform", activation='relu'))
    model.add(Dense(64, kernel_initializer="uniform", activation='relu'))
    model.add(Dense(1, kernel_initializer="uniform", activation='linear'))

    model.compile(loss='mse', optimizer='adam', metrics=['accuracy'])
    model.summary()
    print("-----------------------------------------")
    return model


# 训练模型
def train_lstm_model(x_train, y_train, window, epochs, batch_size, validation_split=0.1):
    model = build_lstm_model([x_train.shape[2], window, 1])
    # batch_size：整数，指定进行梯度下降时每个batch包含的样本数。训练时一个batch的样本会被计算一次梯度下降，使目标函数优化一步。
    # validation_split：0~1。之间的浮点数，用来指定训练集的一定比例数据作为验证集。验证集将不参与训练，并在每个epoch结束后测试的模型的指标，如损失函数、精确度等。注意，validation_split的划分在shuffle之前，因此如果你的数据本身是有序的，需要先手工打乱再指定validation_split，否则可能会出现验证集样本不均匀。
    model.fit(x_train, y_train, batch_size=batch_size, epochs=epochs, validation_split=validation_split, verbose=1)
    return model


def batch_train_lstm():
    ts_codes = ['000651.SZ']
    for ts_code in ts_codes:
        try:
            # 获取数据
            raw_df = pd.read_csv('../data/%s.csv' % ts_code)

            window = 5
            batch_size = 32
            epochs = 30

            # 归一化、转换成模型接受的格式
            x_train, y_train, x_test, y_test = transform_data_4_lstm(raw_df, window)

            # 训练模型
            model = train_lstm_model(x_train, y_train, window, epochs, batch_size)
            print(ts_code)

            # 预测测试
            pre = prediction(model, x_test, y_test)

            show_line(model, x_test, y_test)

            # bt.backtrader(pre)
        except Exception as ex:
            print("出现如下异常%s" % ex)
            # 打印异常堆栈
            traceback.print_exc()
