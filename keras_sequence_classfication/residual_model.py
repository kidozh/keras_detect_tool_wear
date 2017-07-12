# -*- coding: UTF-8 -*-
# author = kidozh

from keras.layers import merge
from keras.layers.merge import add
from keras.layers.convolutional import Conv2D,MaxPooling2D,ZeroPadding2D,AveragePooling2D,Conv1D,MaxPooling1D
from keras.layers.core import Dense,Activation,Flatten,Dropout,Masking
from keras.layers.normalization import BatchNormalization
from keras.models import Model
from keras.layers import Input,TimeDistributed

# looking for stanfordmlgroup.github.io/projects/ecg/ for detail

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
    out = Conv1D(k1,kernel_size,padding='same')(out)
    out = BatchNormalization()(out)
    out = Activation('relu')(out)
    out = Dropout(dropout)(out)
    out = Conv1D(k2,kernel_size,padding='same')(out)


    pooling = MaxPooling1D(pooling_size,padding='same')(x)

    out = add([out, pooling])

    #out = merge([out,pooling])
    return out

def build_main_residual_network(batch_size,
                                time_step,
                                input_dim,
                                output_dim,
                                loop_depth=15,
                                dropout=0.3):
    inp = Input(shape=(time_step,input_dim))

    # add mask for filter invalid data
    out = TimeDistributed(Masking(mask_value=0))(inp)


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


def first_2d_block(tensor_input,filters,kernel_size=3,pooling_size=1,dropout=0.5):
    k1,k2 = filters

    out = Conv2D(k1,1,padding='same',data_format='channels_last')(tensor_input)
    out = BatchNormalization()(out)
    out = Activation('relu')(out)
    out = Dropout(dropout)(out)
    out = Conv2D(k2,kernel_size,padding='same',data_format='channels_last')(out)


    pooling = MaxPooling2D(pooling_size,padding='same',data_format='channels_last')(tensor_input)


    # out = merge([out,pooling],mode='sum')
    out = add([out,pooling])
    return out

def repeated_2d_block(x,filters,kernel_size=3,pooling_size=1,dropout=0.5):

    k1,k2 = filters


    out = BatchNormalization()(x)
    out = Activation('relu')(out)
    out = Conv2D(k1,kernel_size,padding='same',data_format='channels_last')(out)
    out = BatchNormalization()(out)
    out = Activation('relu')(out)
    out = Dropout(dropout)(out)
    out = Conv2D(k2,kernel_size,padding='same',data_format='channels_last')(out)


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


if __name__ == '__main__':
    # build_main_residual_network(None,2000,7,4)
    build_2d_main_residual_network(None,4000,7,2,4)



