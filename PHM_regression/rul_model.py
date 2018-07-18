from keras.layers import Dense,Conv1D,Dropout,Activation,BatchNormalization,MaxPooling1D,Flatten,Masking,TimeDistributed
from keras.layers.recurrent import LSTM,GRU,SimpleRNN
from keras.models import Input,Sequential,Model
from keras.layers.merge import add
from keras.optimizers import Adam
from keras.losses import MSE,MSLE
from keras import backend as K
from keras.layers.merge import concatenate

from keras import backend as K
from keras.layers import Activation
from keras.utils.generic_utils import get_custom_objects
def swish(x):
    return (K.sigmoid(x) * x)
get_custom_objects().update({'swish': Activation(swish )})


def mean_squared_error(y_true, y_pred):
    # filter all the non-zero value
    non_zero_label = y_true[y_true!=0]
    y_true = y_true[non_zero_label]
    y_pred = y_pred[non_zero_label]
    return K.mean(K.square(y_pred - y_true), axis=-1)


def first_block(tensor_input,filters,kernel_size=3,pooling_size=1,dropout=0.5):
    k1,k2 = filters

    out = Conv1D(k1,1,padding='same')(tensor_input)
    out = BatchNormalization()(out)
    out = Activation('relu')(out)
    out = Dropout(dropout)(out)
    out = Conv1D(k2,kernel_size,padding='same')(out)


    pooling = MaxPooling1D(pooling_size,padding='same')(tensor_input)


    # out = merge([out,pooling],mode='sum')
    out = add([out,pooling])
    return out


def repeated_block(x,filters,kernel_size=3,pooling_size=3,dropout=0.5,is_first_layer_of_block=False):

    k1,k2 = filters
    # program control it





    out = BatchNormalization()(x)
    out = Activation('relu')(out)
    out = Conv1D(k1,kernel_size,strides=1,padding='same')(out)
    out = BatchNormalization()(out)
    out = Activation('relu')(out)
    out = Dropout(dropout)(out)
    out = Conv1D(k2,kernel_size,strides=2,padding='same')(out)

    if is_first_layer_of_block:
        # add conv here
        pooling = Conv1D(k2,kernel_size,strides=2,padding='same')(x)
        #pooling = MaxPooling1D(pooling_size, strides=2, padding='same')(x)
    else:
        pooling = MaxPooling1D(pooling_size, strides=2, padding='same')(x)
        pass




    out = add([out, pooling])

    #out = merge([out,pooling])
    return out

def build_model(timestep,input_dim,output_dim,dropout=0.5,recurrent_layers_num=4,cnn_layers_num=6,lr=0.001):
    inp = Input(shape=(timestep,input_dim))
    output = TimeDistributed(Masking(mask_value=0))(inp)
    #output = inp
    output = Conv1D(128, 2)(output)
    output = BatchNormalization()(output)
    output = Activation('relu')(output)

    output = first_block(output, (64, 128), dropout=dropout)


    output = Dropout(dropout)(output)
    base_filter = 16
    block_part_num = 5
    for cur_layer_num in range(cnn_layers_num):
        is_first_layer = False
        if cur_layer_num % block_part_num == 0:
            is_first_layer = True
        # determine kernel size
        # tranverse all the pyramid structure
        filter_times = cnn_layers_num//block_part_num - cur_layer_num // block_part_num
        filter = (base_filter*(2**filter_times),base_filter*(2**(filter_times+1)))
        print(cur_layer_num,":",filter)
        output = repeated_block(output, filter, dropout=dropout,is_first_layer_of_block=is_first_layer)

    output = Flatten()(output)
    #output = LSTM(128, return_sequences=False)(output)

    output = BatchNormalization()(output)
    output = Activation('relu')(output)
    output = Dense(output_dim)(output)


    model = Model(inp,output)

    optimizer = Adam(lr=lr)

    model.compile(optimizer,'mse',['mae'])
    return model


def build_wavelet_model(a2_time_step,
                        d2_time_step,
                        d1_time_step,input_dim,output_dim,dropout=0.5,recurrent_layers_num=4,cnn_layers_num=6,lr=0.001):

    a2_inp = Input(shape=(a2_time_step, input_dim), name='a2')
    d2_inp = Input(shape=(d2_time_step, input_dim), name='d2')
    d1_inp = Input(shape=(d1_time_step, input_dim), name='a1')

    out = concatenate([a2_inp, d2_inp, d1_inp], axis=1)
    #output = inp
    output = Conv1D(128, 2)(out)
    output = BatchNormalization()(output)
    output = Activation('relu')(output)

    output = first_block(output, (64, 128), dropout=dropout)


    output = Dropout(dropout)(output)
    for _ in range(cnn_layers_num):
        output = repeated_block(output, (64, 128), dropout=dropout)

    output = Flatten()(output)
    #output = LSTM(128, return_sequences=False)(output)

    output = BatchNormalization()(output)
    output = Activation('relu')(output)
    output = Dense(output_dim)(output)


    model = Model(inputs=[a2_inp,d2_inp,d1_inp],outputs=output)

    optimizer = Adam(lr=lr)

    model.compile(optimizer,'mse',['mae'])
    return model