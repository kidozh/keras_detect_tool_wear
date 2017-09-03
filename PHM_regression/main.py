from PHM_regression.dataSet import auto_stride_dataset,wavelet_dataset
from PHM_regression.model import build_simple_rnn_model,build_multi_input_main_residual_network
from keras.callbacks import TensorBoard

a = wavelet_dataset()

a = auto_stride_dataset()
y = a.get_all_loc_y_sample_data()
x = a.get_all_loc_x_sample_data()

# import matplotlib.pyplot as plt
# import matplotlib
# import numpy as np

# see the data distribution
# for i in range(0):
#     plt.subplot(7,1,i+1)
#     print(x[:,4999,i].shape,np.arange(1,946).shape)
#     plt.plot(np.arange(1,946),x[:,4999,i])
#     plt.legend()
#
# plt.show()

data = wavelet_dataset()
a2,d2,d1 = data.gen_x_dat()
y = data.gen_y_dat()

# reshape y




log_dir = 'resiual_logs/'

train_name = 'residual_logs_adam_adjust_kernel_depth_20_shuffle_manually'

model_name = '%s.kerasmodel'%(train_name)

predict = True

if not predict:
    tb_cb = TensorBoard(log_dir=log_dir + train_name, histogram_freq=20, write_grads=True)

    model = build_multi_input_main_residual_network(32, a2.shape[1], d2.shape[1], d1.shape[1], 7, 3, loop_depth=2)

    # model = build_simple_rnn_model(5000,7,3)

    model.fit([a2, d2, d1], y, batch_size=32, epochs=50, callbacks=[tb_cb], validation_split=0.2)

    model.save(model_name)

else:


    PRED_PATH = 'Y_PRED'

    try:
        import numpy as np
        y_pred = np.load(PRED_PATH+'.npy')
    except Exception as e:
        print(e)
        from keras.models import load_model
        model = load_model(model_name)
        y_pred = model.predict([a2,d2,d1])
        np.save(PRED_PATH,y_pred)
        # y_pred.save(PRED_PATH)

    print(y_pred.shape)

    import matplotlib.pyplot as plt
    import matplotlib
    import matplotlib.mlab as mlab


    # loss survey
    bais = y_pred - y

    print(bais.shape)

    import seaborn as sns

    for i in range(3):
        sns.set(color_codes=True)
        sns.distplot(bais[:,i],kde=True)
        print(bais[:,i].shape)
        print('%'*20)
        plt.tight_layout()
        plt.show()

        # print(np.arange(1,946,1).shape,bais[:,i].shape)
        # n, bins, patches = plt.hist(bais[:,i],'auto',normed=1, facecolor='blue', alpha=0.5)
        #
        # # curve
        # #y = mlab.normpdf(bins, mu, sigma)
        # #plt.plot(bins, y, 'r--')
        # plt.xlabel('Error')
        # plt.ylabel('Frequency')
        # plt.legend()
        # plt.show()

    plt.rcParams['font.sans-serif'] = ['PingFang SC']  # 用来正常显示中文标签
    plt.rcParams['axes.unicode_minus'] = False  # 用来正常显示负号


    # for i in range(3):
    #     plt.title('No.%s tool wear' % (i + 1))
    #     for j in range(3):
    #         plt.subplot('31%s'%(j+1))
    #
    #         plt.plot(y[j*315:(j+1)*315,i],label='y_real_%s_tool'%(j+1))
    #         plt.plot(y_pred[j*315:(j+1)*315,i],label='y_pred_%s_tool'%(j+1))
    #         plt.ylabel('Wear')
    #         plt.xlabel('Time')
    #         plt.legend()
    #
    #
    #
    #     plt.show()
    #
    #     pass


