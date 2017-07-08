# -*- coding: UTF-8 -*-
# author = kidozh

from keras_sequence_classfication.models import build_stateful_lstm_model
from keras_sequence_classfication.data1 import dataSet
from keras.callbacks import TensorBoard

import numpy as np

data = dataSet(force_update=False)

# currently no data get found
x = data.get_all_sample_data()
y = data.get_res_data_in_numpy

BATCH_SIZE = 1
TIME_STEP =None
INPUT_DIM = 7
OUTPUT_DIM = 3

model = build_stateful_lstm_model(BATCH_SIZE,TIME_STEP,INPUT_DIM,OUTPUT_DIM,dropout=0.1)

x_train =x
y_train =y

for i in x_train:
    for j in i:
        print(j.shape)


model.fit(x_train,y_train,validation_data=(x_train[:10],y_train[:10]),epochs=5,callbacks=[TensorBoard()],batch_size=1)

# for index,y_dat in enumerate(y):
#     print('Run test on %s' %(index))
#     model.fit(np.array([x[index]]),y_dat.reshape(1,3),validation_data=(np.array([x[index]]),y_dat.reshape(1,3)),epochs=5,callbacks=[TensorBoard()])

MODEL_PATH = 'model.dat'
model.save(MODEL_PATH)