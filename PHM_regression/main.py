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

train_name = 'residual_logs_adam'

model_name = '%s.kerasmodel'%(train_name)

tb_cb = TensorBoard(log_dir=log_dir+train_name,histogram_freq=20,write_grads=True)

model = build_multi_input_main_residual_network(32,a2.shape[1],d2.shape[1],d1.shape[1],7,3,loop_depth=2)

# model = build_simple_rnn_model(5000,7,3)

model.fit([a2,d2,d1],y,batch_size=32,epochs=50,callbacks=[tb_cb],validation_split=0.2)

model.save(model_name)
