from RUL_estimation.data import data_source,test_data_source
from RUL_estimation.model import build_model
from keras.callbacks import TensorBoard
import numpy as np
from keras.preprocessing.sequence import pad_sequences

DEPTH = 20

lr = 0.01

data = data_source()
x,y = data.gen_sample()

new_x = pad_sequences(x)
new_y = pad_sequences(y)

#new_x = new_x[:,:,np.newaxis,:]
# new_y = new_y[:,np.newaxis,:]
print(new_x.shape,new_y.shape)


def sample_generator():
    while True:
        for i,single_y in enumerate(y):
            sample_x,sample_y = x[i][:,np.newaxis,:],y[i][:,np.newaxis]
            yield (sample_x,sample_y)

TRAIN_NAME = 'MLP_PURE_RESIDUAL_CNN_%s_NON_GEN_lr_%s' %(DEPTH,lr)

#TRAIN_NAME = 'MLP_PURE_RESIDUAL_CNN_20_NON_GEN'

MODEL_NAME = '%s.kerasmodel'%(TRAIN_NAME)

TRAIN = 0
if TRAIN:
    # model
    model = build_model(357,24,357,recurrent_layers_num=0,cnn_layers_num=DEPTH,lr=lr)
    # callback
    tb_cb = TensorBoard('./RUL_logs/'+TRAIN_NAME)

    #model.fit_generator(sample_generator(),data.id_range,epochs=1000,callbacks=[tb_cb])

    model.fit(new_x,new_y,batch_size=16,epochs=5000,validation_split=0.2,callbacks=[tb_cb])
    model.save(MODEL_NAME)
else:
    from keras.models import load_model
    model = load_model(MODEL_NAME)
    #eva_mse,eva_mae = model.evaluate(new_x,new_y)

    import matplotlib.pyplot as plt
    #from matplotlib.pyplot import plt

    test_data = test_data_source()
    test_x = test_data.gen_test_dat()
    test_new_x = pad_sequences(test_x,maxlen=357)
    # print(test_new_x.shape,new_x.shape)
    # print(test_new_x)
    # print('!!!!\n',new_x)
    y_pred = model.predict(new_x)
    test_y_pred = model.predict(test_new_x)

    valid_dat = []

    for index,batch in enumerate(test_x):
        select_window = batch.shape[0]
        valid_dat.append(test_y_pred[index][-select_window:])


    # get rid of full zero
    zero_dim_list = []
    for batch_index,timesteps in enumerate(test_new_x):
        for timestep_index,multi_dim in enumerate(timesteps):
            all_zero = True
            for dim in multi_dim:
                if dim != 0:
                    all_zero = False
                    break

            if all_zero:
                zero_dim_list.append((batch_index,timestep_index))

    #print(zero_dim_list)
    # get ride of it to pred
    #print(test_y_pred)

    REMOVE_NUM = -100.222

    for batch_index,timestep_index in zero_dim_list:
        test_y_pred[batch_index][timestep_index] = REMOVE_NUM


    test_y_pred.reshape(-1,1)

    #print(test_y_pred[test_y_pred!=REMOVE_NUM])
    save_arr = []
    for i in valid_dat:
        for j in i :
            save_arr.append(j)

    #save_array.reshape((-1,1))
    np.savetxt('valid_res.csv',save_arr,delimiter=',')

    np.savetxt('res1.csv', test_y_pred[test_y_pred!=REMOVE_NUM], delimiter=',')



    # for index,value in enumerate(new_y):
    #     plt.plot(y_pred[index],label='y_pred')
    #     plt.plot(new_y[index],label='y')
    #     plt.legend()
    #     plt.show()

