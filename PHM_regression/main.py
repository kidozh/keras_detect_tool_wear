from PHM_regression.dataSet import auto_stride_dataset,wavelet_dataset
from PHM_regression.model import build_simple_rnn_model,build_multi_input_main_residual_network
from keras.callbacks import TensorBoard



a = wavelet_dataset()

a = auto_stride_dataset()
y = a.get_all_loc_y_sample_data()
x = a.get_all_loc_x_sample_data()

import matplotlib.pyplot as plt
import matplotlib
import numpy as np
plt.rcParams['font.sans-serif'] = ['Hiragino Sans GB']  # 用来正常显示中文标签
# plt.rcParams['font.sans-serif'] = ['YaHei Consolas Hybrid']  # 用来正常显示中文标签
plt.rcParams['axes.unicode_minus'] = False  # 用来正常显示负号
matplotlib.rcParams['pdf.fonttype'] = 42
matplotlib.rcParams['ps.fonttype'] = 42

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
# y = data.gen_y_dat()

y = data.get_rul_dat()

import random

index = [i for i in range(len(y))]
random.shuffle(index)
y = y[index]

print(y.shape,d1.shape,d2.shape,a2.shape)

a2, d2, d1 = a2[index],d2[index],d1[index]

print(y.shape,d1.shape,d2.shape,a2.shape)


# y = data.get_rul_dat()

# reshape y


for i in [20,15,10,5,35]:
    DEPTH = i

    log_dir = 'resiual_RUL_logs/'

    train_name = 'residual_logs_adam_adjust_kernel_depth_%s_shuffle_manually_LOCAL' % (DEPTH)

    model_name = '%s.kerasmodel' % (train_name)

    predict = True

    if not predict:
        tb_cb = TensorBoard(log_dir=log_dir + train_name)

        model = build_multi_input_main_residual_network(32, a2.shape[1], d2.shape[1], d1.shape[1], 7, 1,
                                                        loop_depth=DEPTH)

        # model = build_simple_rnn_model(5000,7,3)

        print('Model has been established.')

        model.fit([a2, d2, d1], y, batch_size=16, epochs=1000, callbacks=[tb_cb], validation_split=0.2)

        model.save(model_name)

    else:

        PRED_PATH = 'Y_PRED'

        try:
            import numpy as np

            y_pred = np.load(PRED_PATH + '.npy')
        except Exception as e:
            print(e)
            from keras.models import load_model

            model = load_model(model_name)
            y_pred = model.predict([a2, d2, d1])
            np.save(PRED_PATH, y_pred)
            y_pred.save(PRED_PATH)

        print(y_pred.shape)

        import matplotlib.pyplot as plt
        import matplotlib as  mpl
        import matplotlib
        import matplotlib.mlab as mlab

        # plt.rc("font", family="Hiragino Sans GB W3")
        # plt.rc("font", family="PingFang SC")
        # plt.rcParams['font.sans-serif'] = ['Microsoft Yahei']  # 用来正常显示中文标签
        # #plt.rcParams['axes.unicode_minus']=False #用来正常显示负号
        # plt.rcParams['font.sans-serif'] = ['YaHei Consolas Hybrid']  # 用来正常显示中文标签
        # plt.rcParams['axes.unicode_minus'] = False  # 用来正常显示负号

        for i in range(3):
            # plt.title('No.%s tool wear' % (i + 1))
            for j in range(3):
                # plt.subplot('31%s'%(j+1))
                fig = plt.figure()
                print("%s tool @ %s teeth" %(i,j))

                plt.plot(y[j * 315:(j + 1) * 315, i], label=u'刀具磨损真实值')
                plt.plot(y_pred[j * 315:(j + 1) * 315, i], label=u'深度学习预测值')
                plt.ylabel('磨损量 ($\mu m$)')
                plt.xlabel('行程')
                # plt.show()

                # plt.plot(y[j*315:(j+1)*315,i],label='Real')
                # plt.plot(y_pred[j*315:(j+1)*315,i],label='Predicted')
                # plt.ylabel('Tool wear ($\mu m$)')
                # plt.xlabel('Run')
                plt.legend()
                plt.savefig("%s_tool_@_%s_teeth_ZH.svg"%(i,j))
                # plt.show()
        break
    break

    # if True:
    #
    #     bais = y_pred - y
    #
    #     print(bais.shape)
    #
    #     import seaborn as sns
    #
    #     for i in range(3):
    #         sns.set(color_codes=True)
    #         sns.distplot(bais[:, i], kde=True)
    #         print(bais[:, i].shape)
    #         print('%' * 20)
    #         plt.tight_layout()
    #         # plt.show()
    #
    #         # print(np.arange(1,946,1).shape,bais[:,i].shape)
    #         # n, bins, patches = plt.hist(bais[:,i],'auto',normed=1, facecolor='blue', alpha=0.5)
    #         #
    #         # # curve
    #         # #y = mlab.normpdf(bins, mu, sigma)
    #         # #plt.plot(bins, y, 'r--')
    #         # plt.xlabel('Error')
    #         # plt.ylabel('Frequency')
    #         # plt.legend()
    #         # plt.show()
    #
    #     plt.rcParams['font.sans-serif'] = ['PingFang SC']  # 用来正常显示中文标签
    #     plt.rcParams['axes.unicode_minus'] = False  # 用来正常显示负号






