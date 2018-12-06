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

lr=0.01

for i in [20,16,12,8]:
#for i in [15,5,25,35]:
    DEPTH = i

    log_dir = 'RUL_logs_alt_long/'

    # train_name = 'RUL_TRANSVERSE_PYRAMID_DEPTH_%s_LR_%s' % (DEPTH,lr)

    train_name = 'residual_logs_adam_adjust_kernel_depth_20_shuffle_manually'

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
        from keras.models import Model
        import matplotlib.pyplot as plt

        model = load_model(model_name)

        print('CONV1D:',model.get_weights()[0].shape)
        # model analysis

        for INDEX in [50,100,1000,5000,20000,22500,25000,27500,30000,32500,35000,37500,40000]:
            break
            # data visualization started...
            for layer in range(47):

                conv_model = Model(inputs=model.input,
                                   outputs=model.get_layer('conv1d_%s' % (layer + 1)).output)

                conv1_output = conv_model.predict([a2,d2,d1])

                # INDEX = 10000
                print('#',INDEX, layer, x.shape)
                fig = plt.gcf()
                for i in range(7):
                    plt.subplot(7 + conv1_output.shape[-1], 1, i + 1)
                    plt.plot(x[INDEX, :, i])

                # plot first
                for i in range(conv1_output.shape[-1]):
                    plt.subplot(7 + conv1_output.shape[-1], 1, i + 8)
                    # plt.title(i)
                    # plt.plot(x[INDEX,:,0],label='REAL_ONE')
                    plt.plot(conv1_output[INDEX, :, i], label='FILTER_%s' % (i), color='red')

                # plt.legend()
                fig.set_size_inches(20, 100)
                plt.savefig('Tot_data@%sin%s.pdf' % (INDEX, layer))
                # plt.show()
                plt.close('all')

        conv_model = Model(inputs=model.input,
                           outputs=model.get_layer('conv1d_%s' % (46 + 1)).output)

        conv1_output = conv_model.predict([a2,d2,d1])
        print(conv1_output.shape)


        for i in range(128):
            plt.title(i)
            # plt.plot(x[INDEX,:,0],label='REAL_ONE')
            plt.plot(conv1_output[:,:,i],label='FILTER_%s'%(i))
            plt.legend()
            plt.show()

        y_pred = model.predict([a2,d2,d1])
        # print(model.evaluate(x,y))
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





