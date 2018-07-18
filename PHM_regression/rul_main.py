from PHM_regression.dataSet import auto_stride_dataset,wavelet_dataset,rul_data_source
from PHM_regression.model import build_simple_rnn_model,build_multi_input_main_residual_network,build_rul_multi_input_main_residual_network
from keras.callbacks import TensorBoard
from PHM_regression.rul_model import build_model,build_wavelet_model
import numpy as np
import pywt
#a = auto_stride_dataset()
#y = a.get_all_loc_y_sample_data()
#x = a.get_all_loc_x_sample_data()


def mean_absolute_percentage_error(y_true, y_pred):
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    return np.mean(np.abs((y_true - y_pred) / y_true)) * 100

a = rul_data_source()

x,y = a.get_data()

a2,d2,d1 = pywt.wavedec(x,'db4',mode='symmetric',level=2,axis=1)

# import matplotlib.pyplot as plt
# plt.plot(y)
# plt.show()


import random

index = [i for i in range(len(y))]
random.shuffle(index)
y = y[index]
a2, d2, d1 = a2[index],d2[index],d1[index]
x = x[index]


# y = data.get_rul_dat()

# reshape y

lr = 0.01

#for i in [20,16,12,8]:
for i in [15,5,25,35]:
    DEPTH = i

    log_dir = 'RUL_logs_alt_long/'

    train_name = 'RUL_TRANSVERSE_PYRAMID_DEPTH_%s_LR_%s' % (DEPTH,lr)

    #train_name = 'RUL_SWISS_conv_3_2_DEPTH_20_LR_0.01_Change_filter_dropout_0.1'

    model_name = '%s.kerasmodel' % (train_name)

    predict = False

    if not predict:
        tb_cb = TensorBoard(log_dir=log_dir + train_name,write_grads=True)

        # model = build_wavelet_model(a2.shape[1], d2.shape[1], d1.shape[1],7,1,dropout=0.6,cnn_layers_num=DEPTH,lr=lr)

        model = build_model(a.SAMPLE_TIMESTEP//20,7,1,recurrent_layers_num=0,cnn_layers_num=DEPTH,lr=lr,dropout=0.1)

        #from keras.utils import plot_model
        #model.summary()
        #plot_model(model, to_file='ResNet_mine.png')

        print('Model has been established.')

        model.fit(x, y, batch_size=16, epochs=10000, callbacks=[tb_cb], validation_split=0.5)

        # model.fit([a2,d2,d1],y,batch_size=16,epochs=1000,callbacks=[tb_cb],validation_split=0.4)

        model.save(model_name)

    else:

        PRED_PATH = 'RUL_Y_PRED'

        x, y = a.get_data()
        from keras.models import load_model

        model = load_model(model_name)
        y_pred = model.predict(x)
        print(model.evaluate(x,y))
        from sklearn.metrics import r2_score
        #np.save(PRED_PATH, y_pred)
        # y_pred.save(PRED_PATH)

        print(y_pred.shape)
        print('R2',r2_score(y,y_pred))
        print('mape',mean_absolute_percentage_error(y,y_pred))

        import matplotlib.pyplot as plt
        import matplotlib
        import matplotlib.mlab as mlab

        plt.plot(y,label='real')
        plt.plot(y_pred,label='pred')
        plt.title(train_name)
        plt.savefig('%s.pdf'%(train_name))
        plt.show()

        # loss survey
        bais = y_pred - y

        print(bais.shape)





