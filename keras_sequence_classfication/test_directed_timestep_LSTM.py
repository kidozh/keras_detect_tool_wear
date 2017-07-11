# -*- coding: UTF-8 -*-
# author = kidozh

from keras_sequence_classfication.models import *
from keras_sequence_classfication.data import dataSet
from keras.callbacks import TensorBoard

import numpy as np

data = dataSet(force_update=False,sample_skip_num=10)
MODEL_PATH = 'model_LSTM_FIXED_STEP_skip_%s.dat'%(data.sample_skip_num)
# currently no data get found
x = data.get_all_sample_data()
y = data.get_res_data_in_numpy

BATCH_SIZE = 10
# only pick up first 2000 points
TIME_STEP = 2000
INPUT_DIM = 7
OUTPUT_DIM = 3

#model = build_stateful_lstm_model(BATCH_SIZE,TIME_STEP,INPUT_DIM,OUTPUT_DIM,dropout=0.1)

# model = build_multi_cnn_model(BATCH_SIZE,
#                               TIME_STEP,
#                               INPUT_DIM,
#                               OUTPUT_DIM,
#                               dropout=0.1,kernel_size=1)

# x_train = np.zeros()
SAMPLE_NUM,_,OUTPUT_DIM=y.shape

x_train = np.zeros(shape=(SAMPLE_NUM,TIME_STEP,INPUT_DIM))

for index,y_dat in enumerate(y):
    x_train[index] = x[index][:TIME_STEP]

y_train = y.reshape(SAMPLE_NUM,OUTPUT_DIM)

# model = build_multi_1d_cnn_model(BATCH_SIZE,
#                               TIME_STEP,
#                               INPUT_DIM,
#                               OUTPUT_DIM,
#                               dropout=0.1,
#                               kernel_size=5,
#                                 conv_dim=(128,64,32),
#                                  stack_loop_num=3)

def train():
    model = build_real_stateful_lstm_model_with_normalization(BATCH_SIZE, TIME_STEP, INPUT_DIM, OUTPUT_DIM)

    # deal with x,y



    # x_train = x


    model.fit(x_train[:SAMPLE_NUM//BATCH_SIZE*BATCH_SIZE],
              y_train[:SAMPLE_NUM//BATCH_SIZE*BATCH_SIZE],
              batch_size=BATCH_SIZE,
              validation_split=0,
              epochs=30, callbacks=[TensorBoard(log_dir='./stateful_lstm_fixed')])

    # for index,y_dat in enumerate(y):
    #     print('Run test on %s' %(index))
    #     # print(y_dat.reshape(3,1))
    #     model.fit(np.array([x[index]]),np.array([y_dat.reshape(1,3)]),validation_data=(np.array([x[index]]),np.array([y_dat.reshape(1,3)])),epochs=100,callbacks=[TensorBoard()])
    #     model.save(MODEL_PATH)
    #     x_pred = model.predict(np.array([x[index]]))
    #     print(x_pred,x_pred.shape)
    #     print(np.array([y_dat.reshape(1,3)]))

    import random

    randomIndex = random.randint(0, SAMPLE_NUM)

    print('Selecting %s as the sample' % (randomIndex))

    pred = model.predict(x_train[randomIndex:randomIndex + 1])

    print(pred)

    print(y_train[randomIndex])

    model.save(MODEL_PATH)

def test():
    from keras.models import load_model
    model = load_model(MODEL_PATH)

    while True:
        numb = input('Please input your data')
        if numb == 'c':
            break
        else:
            numb = int(numb)
            pass

        pred = model.predict(x_train[numb:numb + 1])

        print(pred)

        print(y_train[numb])

    import matplotlib.pyplot as plt
    all_pred = model.predict(x_train[:SAMPLE_NUM//BATCH_SIZE*BATCH_SIZE])

    a = np.zeros(SAMPLE_NUM//BATCH_SIZE*BATCH_SIZE)
    real_a = np.zeros(SAMPLE_NUM//BATCH_SIZE*BATCH_SIZE)

    b = np.zeros(SAMPLE_NUM//BATCH_SIZE*BATCH_SIZE)
    real_b = np.zeros(SAMPLE_NUM//BATCH_SIZE*BATCH_SIZE)

    c = np.zeros(SAMPLE_NUM//BATCH_SIZE*BATCH_SIZE)
    real_c = np.zeros(SAMPLE_NUM//BATCH_SIZE*BATCH_SIZE)

    for index,everyDat in enumerate(all_pred):

        a[index] = everyDat[0]
        b[index] = everyDat[1]
        c[index] = everyDat[2]
        real_a[index] = y_train[index][0]
        real_b[index] = y_train[index][1]
        real_c[index] = y_train[index][2]

    plt.plot(np.arange(SAMPLE_NUM//BATCH_SIZE*BATCH_SIZE),a,label='a')
    plt.plot(np.arange(SAMPLE_NUM//BATCH_SIZE*BATCH_SIZE), real_a, label='real_a')
    plt.title('A')
    plt.legend()
    plt.show()

    plt.plot(np.arange(SAMPLE_NUM//BATCH_SIZE*BATCH_SIZE), b, label='b')
    plt.plot(np.arange(SAMPLE_NUM//BATCH_SIZE*BATCH_SIZE), real_b, label='real_b')
    plt.title('B')
    plt.legend()
    plt.show()

    plt.plot(np.arange(SAMPLE_NUM//BATCH_SIZE*BATCH_SIZE), c, label='c')
    plt.plot(np.arange(SAMPLE_NUM//BATCH_SIZE*BATCH_SIZE), real_c, label='real_c')
    plt.title('C')
    plt.legend()
    plt.show()

if __name__ == '__main__':
    # train()
    test()