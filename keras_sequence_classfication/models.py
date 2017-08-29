# -*- coding: UTF-8 -*-
# author = kidozh

from keras.layers import Dense,LSTM,GRU,SimpleRNN,Dropout,TimeDistributed,Conv1D,Activation,MaxPool1D,Flatten,Conv2D,MaxPool2D
from keras.models import Input,Sequential
from keras.optimizers import RMSprop,Adam
from keras.layers.normalization import BatchNormalization

def build_stateful_lstm_model(batch_size,time_step,input_dim,output_dim,dropout=0.2,rnn_layer_num=2,hidden_dim=128,hidden_num=0,rnn_type='LSTM'):

    model = Sequential()
    # may use BN for accelerating speed
    # add first LSTM
    if rnn_type == 'LSTM':
        rnn_cell = LSTM
    elif rnn_type == 'GRU':
        rnn_cell = GRU
    elif rnn_type == 'SimpleRNN':
        rnn_cell = SimpleRNN
    else:
        raise ValueError('Option rnn_type could only be configured as LSTM, GRU or SimpleRNN')
    model.add(rnn_cell(hidden_dim,return_sequences=True,batch_input_shape=(batch_size,time_step,input_dim)))

    for _ in range(rnn_layer_num-2):
        model.add(rnn_cell(hidden_dim, return_sequence=True))
        # prevent over fitting
        model.add(Dropout(dropout))



    model.add(rnn_cell(hidden_dim,return_sequences=False))

    # add hidden layer

    for _ in range(hidden_num):
        model.add(Dense(hidden_dim))

    model.add(Dropout(dropout))

    model.add(Dense(output_dim))

    rmsprop = RMSprop(lr=0.01)
    adam = Adam(lr=0.01)


    model.compile(loss='mse',metrics=['acc'],optimizer=rmsprop)

    return model


def build_multi_cnn_model(batch_size,
                          time_step,
                          input_dim,
                          output_dim,
                          kernel_size = 2,
                          conv_dim=(64,32,16),
                          dropout=0.2,
                          stack_loop_num=15):
    model = Sequential()

    first_dim,second_dim,loop_dim = conv_dim

    # build first conventional NN
    # https://stanfordmlgroup.github.io/projects/ecg/

    # use valid
    model.add(Conv2D(first_dim,kernel_size,input_shape=(batch_size,time_step,input_dim)))
    # for reducing overfitting and accelerating speed
    model.add(BatchNormalization())
    model.add(Activation('relu'))

    # then next conv
    model.add(Conv2D(second_dim,kernel_size))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(Dropout(dropout))
    model.add(Conv2D(second_dim,kernel_size))
    model.add(MaxPool2D(kernel_size))

    # next 15 loop
    for _ in range(stack_loop_num):
        model.add(BatchNormalization())
        model.add(Activation('relu'))
        model.add(Conv2D(loop_dim,kernel_size))
        model.add(BatchNormalization())
        model.add(Activation('relu'))
        model.add(Dropout(dropout))
        model.add(Conv2D(loop_dim,kernel_size))
        model.add(MaxPool2D(kernel_size))

    # result
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(Dense(output_dim))

    # this is not classfication problem, so softmax is not needed
    # model.add(Activation('softmax'))
    model.compile(loss='mse',metrics=['mse'],optimizer='adam')

    return model

def build_stateful_lstm_model_with_normalization(batch_size,
                                                 time_step,
                                                 input_dim,
                                                 output_dim,
                                                 dropout=0.2,
                                                 rnn_layer_num=2,
                                                 hidden_dim=128,
                                                 hidden_num=0,

                                                 rnn_type='LSTM'):

    model = Sequential()
    # may use BN for accelerating speed
    # add first LSTM
    if rnn_type == 'LSTM':
        rnn_cell = LSTM
    elif rnn_type == 'GRU':
        rnn_cell = GRU
    elif rnn_type == 'SimpleRNN':
        rnn_cell = SimpleRNN
    else:
        raise ValueError('Option rnn_type could only be configured as LSTM, GRU or SimpleRNN')
    model.add(rnn_cell(hidden_dim,return_sequences=True,batch_input_shape=(batch_size,time_step,input_dim)))
    model.add(BatchNormalization())

    for _ in range(rnn_layer_num-2):
        model.add(rnn_cell(hidden_dim, return_sequence=True))
        # prevent over fitting
        model.add(Dropout(dropout))


    model.add(BatchNormalization())
    model.add(rnn_cell(hidden_dim,return_sequences=False))

    # add hidden layer

    for _ in range(hidden_num):
        model.add(Dense(hidden_dim))

    model.add(Dropout(dropout))

    model.add(Dense(output_dim))

    rmsprop = RMSprop(lr=0.01)
    adam = Adam(lr=0.01)


    model.compile(loss='mse',metrics=['acc'],optimizer=rmsprop)

    return model

def build_multi_1d_cnn_model(batch_size,
                          time_step,
                          input_dim,
                          output_dim,
                          kernel_size = 2,
                          pooling_size=2,
                          conv_dim=(64,32,16),
                          dropout=0.2,
                          stack_loop_num=15,
                          padding_method='valid'):
    model = Sequential()

    first_dim,second_dim,loop_dim = conv_dim

    # build first conventional NN
    # https://stanfordmlgroup.github.io/projects/ecg/

    # use valid
    model.add(Conv1D(first_dim,kernel_size,input_shape=(time_step,input_dim),padding='same'))
    # for reducing overfitting and accelerating speed
    model.add(BatchNormalization())
    model.add(Activation('relu'))

    # then next conv
    model.add(Conv1D(second_dim,kernel_size,padding=padding_method))
    model.add(MaxPool1D(pooling_size, padding=padding_method))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(Dropout(dropout))
    model.add(Conv1D(second_dim,kernel_size,padding=padding_method))
    model.add(MaxPool1D(pooling_size,padding=padding_method))

    # next 15 loop
    for _ in range(stack_loop_num):
        model.add(BatchNormalization())
        model.add(Activation('relu'))
        model.add(Conv1D(loop_dim,kernel_size,padding=padding_method))
        model.add(BatchNormalization())
        model.add(Activation('relu'))
        model.add(Dropout(dropout))
        model.add(Conv1D(loop_dim,kernel_size,padding=padding_method))
        model.add(MaxPool1D(pooling_size,padding=padding_method))

    # result
    model.add(BatchNormalization())
    model.add(Activation('relu'))

    model.add(Flatten())

    model.add(Dense(output_dim))

    # this is not classfication problem, so softmax is not needed
    # model.add(Activation('softmax'))

    adam = Adam(lr=0.001)

    model.compile(loss='mse',metrics=['mse'],optimizer=adam)

    return model


def build_real_stateful_lstm_model_with_normalization(batch_size,
                                                 time_step,
                                                 input_dim,
                                                 output_dim,
                                                 dropout=0.2,
                                                 rnn_layer_num=2,
                                                 hidden_dim=128,
                                                 hidden_num=0,

                                                 rnn_type='LSTM'):

    model = Sequential()
    # may use BN for accelerating speed
    # add first LSTM
    if rnn_type == 'LSTM':
        rnn_cell = LSTM
    elif rnn_type == 'GRU':
        rnn_cell = GRU
    elif rnn_type == 'SimpleRNN':
        rnn_cell = SimpleRNN
    else:
        raise ValueError('Option rnn_type could only be configured as LSTM, GRU or SimpleRNN')
    model.add(rnn_cell(hidden_dim,stateful=True,return_sequences=True,batch_input_shape=(batch_size,time_step,input_dim)))
    model.add(BatchNormalization())

    for _ in range(rnn_layer_num-2):
        model.add(rnn_cell(hidden_dim,stateful=True, return_sequence=True))
        # prevent over fitting
        model.add(Dropout(dropout))


    model.add(BatchNormalization())
    model.add(rnn_cell(hidden_dim,stateful=True,return_sequences=False))

    # add hidden layer

    for _ in range(hidden_num):
        model.add(Dense(hidden_dim))

    model.add(Dropout(dropout))

    model.add(Dense(output_dim))

    rmsprop = RMSprop(lr=0.01)
    adam = Adam(lr=0.01)


    model.compile(loss='mse',metrics=['acc'],optimizer=rmsprop)

    return model

if __name__ == '__name__':
    model = build_stateful_lstm_model(2,None,7,3)
    model = build_multi_cnn_model(2,None,7,3)
    print(model)
