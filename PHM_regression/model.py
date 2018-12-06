from keras.layers import Dense,Dropout
from keras.layers.recurrent import LSTM,GRU
from keras.models import Input,Sequential,Model
from keras.optimizers import Adam,Nadam

def build_simple_rnn_model(timestep,input_dim,output_dim,dropout=0.4,lr=0.001):
    input = Input((timestep,input_dim))
    # LSTM, Single
    output = LSTM(50,return_sequences=False)(input)
    # for _ in range(1):
    #     output = LSTM(32,return_sequences=True)(output)
    # output = LSTM(50,return_sequences=False)(output)
    output = Dropout(dropout)(output)
    output = Dense(output_dim)(output)

    model =  Model(inputs=input,outputs=output)

    optimizer = Adam(lr=lr)

    model.compile(loss='mae',optimizer=optimizer,metrics=['mse'])

    return model

from keras.layers import merge
from keras.layers.merge import add,concatenate
from keras.layers.convolutional import Conv2D,MaxPooling2D,ZeroPadding2D,AveragePooling2D,Conv1D,MaxPooling1D
from keras.layers.core import Dense,Activation,Flatten,Dropout,Masking
from keras.layers.normalization import BatchNormalization
from keras.models import Model
from keras.layers import Input,TimeDistributed
from keras.layers.recurrent import LSTM

# looking for stanfordmlgroup.github.io/projects/ecg/ for detail

def first_block(tensor_input,filters,kernel_size=3,pooling_size=1,dropout=0.5):
    k1,k2 = filters

    out = Conv1D(k1,1,padding='same')(tensor_input)
    out = BatchNormalization()(out)
    out = Activation('relu')(out)
    out = Dropout(dropout)(out)
    out = Conv1D(k2,kernel_size,strides=2,padding='same')(out)


    pooling = MaxPooling1D(pooling_size,strides=2,padding='same')(tensor_input)


    # out = merge([out,pooling],mode='sum')
    out = add([out,pooling])
    return out

def repeated_block(x,filters,kernel_size=3,pooling_size=1,dropout=0.5):

    k1,k2 = filters


    out = BatchNormalization()(x)
    out = Activation('relu')(out)
    out = Conv1D(k1,kernel_size,padding='same')(out)
    out = BatchNormalization()(out)
    out = Activation('relu')(out)
    out = Dropout(dropout)(out)
    out = Conv1D(k2,kernel_size,strides=2,padding='same')(out)


    pooling = MaxPooling1D(pooling_size,strides=2,padding='same')(x)

    out = add([out, pooling])

    #out = merge([out,pooling])
    return out

def first_rul_block(tensor_input,filters,kernel_size=3,pooling_size=1,dropout=0.5):
    k1,k2 = filters

    out = Conv1D(k1,1,padding='same')(tensor_input)
    out = BatchNormalization()(out)
    out = Activation('relu')(out)
    out = Dropout(dropout)(out)
    out = Conv1D(k2,kernel_size,strides=2,padding='same')(out)


    pooling = MaxPooling1D(pooling_size,strides=2,padding='same')(tensor_input)


    # out = merge([out,pooling],mode='sum')
    out = add([out,pooling])
    return out

def repeated_rul_block(x,filters,kernel_size=5,pooling_size=1,dropout=0.5):

    k1,k2 = filters


    out = BatchNormalization()(x)
    out = Activation('relu')(out)
    out = Conv1D(k1,kernel_size,padding='same')(out)
    out = BatchNormalization()(out)
    out = Activation('relu')(out)
    out = Dropout(dropout)(out)
    out = Conv1D(k2,kernel_size,strides=3,padding='same')(out)


    pooling = MaxPooling1D(pooling_size,strides=3,padding='same')(x)

    out = add([out, pooling])

    #out = merge([out,pooling])
    return out

def build_multi_input_main_residual_network(batch_size,
                                a2_time_step,
                                d2_time_step,
                                d1_time_step,
                                input_dim,
                                output_dim,
                                loop_depth=15,
                                dropout=0.5):
    '''
    a multiple residual network for wavelet transformation
    :param batch_size: as you might see
    :param a2_time_step: a2_size
    :param d2_time_step: d2_size
    :param d1_time_step: d1_size
    :param input_dim: input_dim
    :param output_dim: output_dim
    :param loop_depth: depth of residual network
    :param dropout: rate of dropout
    :return: 
    '''
    a2_inp = Input(shape=(a2_time_step,input_dim),name='a2')
    d2_inp = Input(shape=(d2_time_step,input_dim),name='d2')
    d1_inp = Input(shape=(d1_time_step,input_dim),name='a1')

    out = concatenate([a2_inp,d2_inp,d1_inp],axis=1)



    out = Conv1D(128,5)(out)
    out = BatchNormalization()(out)
    out = Activation('relu')(out)

    out = first_block(out,(64,128),dropout=dropout)

    for _ in range(loop_depth):
        out = repeated_block(out,(64,128),dropout=dropout)

    # add flatten
    out = Flatten()(out)

    out = BatchNormalization()(out)
    out = Activation('relu')(out)
    out = Dense(output_dim)(out)

    model = Model(inputs=[a2_inp,d2_inp,d1_inp],outputs=[out])

    model.compile(loss='mse',optimizer='adam',metrics=['mse','mae'])
    return model

def build_rul_multi_input_main_residual_network(batch_size,
                                a2_time_step,
                                d2_time_step,
                                d1_time_step,
                                input_dim,
                                output_dim,
                                loop_depth=15,
                                dropout=0.5):
    '''
    a multiple residual network for wavelet transformation
    :param batch_size: as you might see
    :param a2_time_step: a2_size
    :param d2_time_step: d2_size
    :param d1_time_step: d1_size
    :param input_dim: input_dim
    :param output_dim: output_dim
    :param loop_depth: depth of residual network
    :param dropout: rate of dropout
    :return:
    '''
    a2_inp = Input(shape=(a2_time_step,input_dim),name='a2')
    d2_inp = Input(shape=(d2_time_step,input_dim),name='d2')
    d1_inp = Input(shape=(d1_time_step,input_dim),name='a1')

    out = concatenate([a2_inp,d2_inp,d1_inp],axis=1)



    out = Conv1D(128,5)(out)
    out = BatchNormalization()(out)
    out = Activation('relu')(out)

    out = first_rul_block(out,(64,128),dropout=dropout)

    for _ in range(loop_depth):
        out = repeated_rul_block(out,(64,128),dropout=dropout)

    # add flatten
    out = Flatten()(out)

    out = BatchNormalization()(out)
    out = Activation('relu')(out)
    out = Dense(output_dim)(out)

    model = Model(inputs=[a2_inp,d2_inp,d1_inp],outputs=[out])

    model.compile(loss='mse',optimizer='adam',metrics=['mse','mae'])
    return model


def first_2d_block(tensor_input,filters,kernel_size=3,pooling_size=2,dropout=0.5):
    k1,k2 = filters

    out = Conv2D(k1,1,padding='same',data_format='channels_last')(tensor_input)
    out = BatchNormalization()(out)
    out = Activation('relu')(out)
    out = Dropout(dropout)(out)
    out = Conv2D(k2,kernel_size,2,padding='same',data_format='channels_last')(out)


    pooling = MaxPooling2D(pooling_size,padding='same',data_format='channels_last')(tensor_input)


    # out = merge([out,pooling],mode='sum')
    out = add([out,pooling])
    return out

def repeated_2d_block(x,filters,kernel_size=3,pooling_size=1,dropout=0.5):

    k1,k2 = filters


    out = BatchNormalization()(x)
    out = Activation('relu')(out)
    out = Conv2D(k1,kernel_size,2,padding='same',data_format='channels_last')(out)
    out = BatchNormalization()(out)
    out = Activation('relu')(out)
    out = Dropout(dropout)(out)
    out = Conv2D(k2,kernel_size,2,padding='same',data_format='channels_last')(out)


    pooling = MaxPooling2D(pooling_size,padding='same',data_format='channels_last')(x)

    out = add([out, pooling])

    #out = merge([out,pooling])
    return out

def build_2d_main_residual_network(batch_size,
                                width,
                                height,
                                channel_size,
                                output_dim,
                                loop_depth=15,
                                dropout=0.3):
    inp = Input(shape=(width,height,channel_size))

    # add mask for filter invalid data
    out = TimeDistributed(Masking(mask_value=0))(inp)


    out = Conv2D(128,5,data_format='channels_last')(out)
    out = BatchNormalization()(out)
    out = Activation('relu')(out)

    out = first_2d_block(out,(64,128),dropout=dropout)

    for _ in range(loop_depth):
        out = repeated_2d_block(out,(64,128),dropout=dropout)

    # add flatten
    out = Flatten()(out)

    out = BatchNormalization()(out)
    out = Activation('relu')(out)
    out = Dense(output_dim)(out)

    model = Model(inp,out)

    model.compile(loss='mse',optimizer='adam',metrics=['mse','mae'])
    return model


def build_main_residual_network_with_lstm(batch_size,
                                time_step,
                                input_dim,
                                output_dim,
                                loop_depth=15,
                                rnn_layer_num = 2,
                                dropout=0.3):


    inp = Input(shape=(time_step,input_dim))



    # add mask for filter invalid data
    out = TimeDistributed(Masking(mask_value=0))(inp)

    # add LSTM module
    for _ in range(rnn_layer_num):
        out = LSTM(128,return_sequences=True)(out)



    out = Conv1D(128,5)(out)
    out = BatchNormalization()(out)
    out = Activation('relu')(out)

    out = first_block(out,(64,128),dropout=dropout)

    for _ in range(loop_depth):
        out = repeated_block(out,(64,128),dropout=dropout)

    # add flatten
    out = Flatten()(out)

    out = BatchNormalization()(out)
    out = Activation('relu')(out)
    out = Dense(output_dim)(out)

    model = Model(inp,out)

    model.compile(loss='mse',optimizer='adam',metrics=['mse','mae'])
    return model