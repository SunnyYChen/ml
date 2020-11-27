# Wait, CNN? Are you crazy? it should be used for image!
# Yes, actually we would use image! I will show you
# Implementation of
#
# Sezer, Omer Berat, and Ahmet Murat Ozbayoglu. "Algorithmic financial trading with deep convolutional neural
# networks: Time series to image conversion approach." Applied Soft Computing 70 (2018): 525-538.
import traceback
import numpy as np
import pandas as pd
from keras.callbacks import ModelCheckpoint
from keras.layers import Conv2D, MaxPooling2D
from keras.layers.core import Dense, Flatten, Dropout
from keras.models import Sequential
import data.data_loader as dl
import features.features_manager as fm
import matplotlib.pyplot as plt


def get_xy_4_cnn(df, window):
    # 转成numpy
    array_df = df.values
    array_x = array_df[:, :-1]
    array_y = array_df[:, -1]
    x = []
    y = []
    for i in range(len(df) - window):
        window_data = array_x[i:(i + window)]
        after_window = array_y[i + window]
        window_data = [[x] for x in window_data]
        x.append(window_data)
        y.append(after_window)
    return np.array(x), np.array(y)


# 转换成训练数据
def transform_data_4_cnn(raw_df, window):
    normalize_train_df, normalize_test_df = fm.features_normalize(raw_df)
    x_train, y_train = get_xy_4_cnn(normalize_train_df, window)
    x_test, y_test = get_xy_4_cnn(normalize_test_df, window)

    x_train = np.reshape(x_train, (x_train.shape[0], x_train.shape[2], x_train.shape[1], x_train.shape[3]))
    x_test = np.reshape(x_test, (x_test.shape[0], x_test.shape[2], x_test.shape[1], x_test.shape[3]))
    return [x_train, y_train, x_test, y_test]


def train_cnn_model(x_train, y_train, x_test, y_test, epochs, batch_size):
    dropout = 0.3
    # CNN Model
    cnn = Sequential()
    cnn.add(Conv2D(8, kernel_size=(1, 2), strides=(1, 1), padding='valid',
                   activation='relu', input_shape=(x_train.shape[1], x_train.shape[2], x_train.shape[3])))
    cnn.add(MaxPooling2D(pool_size=(1, 2)))
    cnn.add(Flatten())
    cnn.add(Dense(64, activation="relu"))
    cnn.add(Dropout(dropout))
    cnn.add(Dense(1, activation="relu"))
    cnn.summary()
    cnn.compile(loss='mean_squared_error', optimizer='nadam')

    # monitor = EarlyStopping(monitor='val_loss', min_delta=1, patience=2, verbose=2, mode='auto')
    checkpointer = ModelCheckpoint(filepath="./CNN_Parameters.hdf5", verbose=0,
                                   save_best_only=True)  # save best model

    cnn.fit(x_train, y_train, validation_split=0.1, batch_size=batch_size, callbacks=[checkpointer], verbose=0,
            epochs=epochs)

    cnn.load_weights('./CNN_Parameters.hdf5')
    return cnn, x_test, y_test


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
    data = df[[0, 1]]
    print('correlation:', data.corr().iat[0, 1])
    trade_date = np.arange(len(p))

    plt.plot(trade_date, y, c='green', label='real')
    plt.plot(trade_date, p, c='red', label='predict')
    plt.show()


def prediction(model, x_test, y_test):
    p = model.predict(x_test)
    # 变成一维数组
    y_test = y_test.flatten()
    p = p.flatten()
    # 涨跌趋势一致的数量
    count = 0
    countCorrectly = 0
    for index in range(1, len(p)):
        if p[index] - p[index - 1] > 0:
            count = count + 1
            if y_test[index] - y_test[index - 1] > 0:
                countCorrectly = countCorrectly + 1

    print("正确预测为涨的占比：", countCorrectly / count)


def batch_train_cnn():
    # ts_codes = [ "600519.SH", "000002.SZ", "002594.SZ", "603259.SH", "600436.SH", "603027.SH", "002475.SZ",
    #            "000651.SZ", "600031.SH", "002142.SZ", "600030.SH", "600362.SH", "600547.SH", "601111.SH", "002352.SZ",
    #            "002079.SZ", "600739.SH", "300331.SZ", "300382.SZ", "300628.SZ", "600844.SH", "300761.SZ"]
    ts_codes = ["000651.SZ"]
    for ts_code in ts_codes:
        try:
            # 获取数据
            raw_df = pd.read_csv('../data/%s.csv' % ts_code)

            batch_size = 32
            epochs = 80
            window = 36

            # 归一化、转换成模型接受的格式
            x_train, y_train, x_test, y_test = transform_data_4_cnn(raw_df, window)
            # 训练模型
            cnn, x_test, y_test = train_cnn_model(x_train, y_train, x_test, y_test, epochs, batch_size)
            print(ts_code)
            prediction(cnn, x_test, y_test)
            show_line(cnn, x_test, y_test)
        except Exception as ex:
            print("出现如下异常%s" % ex)
            # 打印异常堆栈
            traceback.print_exc()
