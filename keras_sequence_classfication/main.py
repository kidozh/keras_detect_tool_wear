# -*- coding: UTF-8 -*-
# author = kidozh

from keras_sequence_classfication.models import build_stateful_lstm_model
from keras_sequence_classfication.data import dataSet
from keras.callbacks import TensorBoard

import numpy as np

data = dataSet(force_update=False,sample_skip_num=50)
MODEL_PATH = 'model_skip_%s.dat'%(data.sample_skip_num)
# currently no data get found
x = data.get_all_sample_data()
y = data.get_res_data_in_numpy

BATCH_SIZE = 1
TIME_STEP =None
INPUT_DIM = 7
OUTPUT_DIM = 3

def train():
    model = build_stateful_lstm_model(BATCH_SIZE, TIME_STEP, INPUT_DIM, OUTPUT_DIM, dropout=0.1)

    # model.fit(x_train,y_train,validation_data=(x_train[:10],y_train[:10]),epochs=5,callbacks=[TensorBoard()],batch_size=1)

    for index, y_dat in enumerate(y):
        print('Run test on %s' % (index))
        model.fit(np.array([x[index]]), y_dat.reshape(1, 3),
                  validation_data=(np.array([x[index]]), y_dat.reshape(1, 3)), epochs=10, callbacks=[TensorBoard()])
        model.save(MODEL_PATH)
        x_pred = model.predict(np.array([x[index]]))
        print(x_pred)
        print(y_dat)

    model.save(MODEL_PATH)

def show_result():
    from keras.models import load_model
    model = load_model(MODEL_PATH)
    # model.fit(x_train,y_train,validation_data=(x_train[:10],y_train[:10]),epochs=5,callbacks=[TensorBoard()],batch_size=1)

    SAMPLE_NUM = 315

    a = np.zeros(SAMPLE_NUM)
    b = np.zeros(SAMPLE_NUM)
    c = np.zeros(SAMPLE_NUM)

    real_a = np.zeros(SAMPLE_NUM)
    real_b = np.zeros(SAMPLE_NUM)
    real_c = np.zeros(SAMPLE_NUM)

    for index, y_dat in enumerate(y):
        print('Run prediction on %s' % (index))
        # model.fit(np.array([x[index]]), y_dat.reshape(1, 3),
        #           validation_data=(np.array([x[index]]), y_dat.reshape(1, 3)), epochs=10, callbacks=[TensorBoard()])
        x_pred = model.predict(np.array([x[index]]))
        print(x_pred,y_dat)
        print(x_pred.shape,y_dat.shape)
        real_a[index] = y_dat.reshape(1,3)[0][0]
        real_b[index] = y_dat.reshape(1,3)[0][1]
        real_c[index] = y_dat.reshape(1,3)[0][2]

        a[index] = x_pred[0][0]
        b[index] = x_pred[0][1]
        c[index] = x_pred[0][2]

    import matplotlib.pyplot as plt

    plt.plot(np.arange(SAMPLE_NUM), a, label='a')
    plt.plot(np.arange(SAMPLE_NUM), real_a, label='real_a')
    plt.title('A')
    plt.legend()
    plt.show()

    plt.plot(np.arange(SAMPLE_NUM), b, label='b')
    plt.plot(np.arange(SAMPLE_NUM), real_b, label='real_b')
    plt.title('B')
    plt.legend()
    plt.show()

    plt.plot(np.arange(SAMPLE_NUM), c, label='c')
    plt.plot(np.arange(SAMPLE_NUM), real_c, label='real_c')
    plt.title('C')
    plt.legend()
    plt.show()

if __name__ == '__main__':
    show_result()