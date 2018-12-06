from keras.layers import Dense,Conv1D,Dropout,Activation,BatchNormalization,MaxPooling1D,Flatten,Masking,TimeDistributed
from keras.layers.recurrent import LSTM,GRU,SimpleRNN
from keras.models import Input,Sequential,Model
from keras.layers.merge import add
from keras.optimizers import Adam
from keras.losses import MSE,MSLE
from keras import backend as K

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


def repeated_block(x,filters,kernel_size=3,pooling_size=1,dropout=0.5):

    k1,k2 = filters


    out = BatchNormalization()(x)
    out = Activation('relu')(out)
    out = Conv1D(k1,kernel_size,strides=2,padding='same')(out)
    out = BatchNormalization()(out)
    out = Activation('relu')(out)
    out = Dropout(dropout)(out)
    out = Conv1D(k2,kernel_size,strides=2,padding='same')(out)


    pooling = MaxPooling1D(pooling_size,strides=4,padding='same')(x)

    out = add([out, pooling])

    #out = merge([out,pooling])
    return out

def build_model(timestep,input_dim,output_dim,dropout=0.5,recurrent_layers_num=4,cnn_layers_num=6,lr=0.001):
    inp = Input(shape=(timestep,input_dim))
    output = TimeDistributed(Masking(mask_value=0))(inp)
    #output = inp
    output = Conv1D(128, 1)(output)
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


    model = Model(inp,output)

    optimizer = Adam(lr=lr)

    model.compile(optimizer,'mse',['mae'])
    return model