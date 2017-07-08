# -*- coding: UTF-8 -*-
# author = kidozh

import pandas as pd
import numpy as np

class dataSet(object):
    res_data_path = 'c1_wear.csv'
    cache_dir_path = '.cache/'
    res_data_storage = 'res_dat'
    sample_data_storage = 'sample_dat'
    total_signal_num = 315

    def __init__(self,force_update=False,sample_skip_num=5):
        self.sample_skip_num = sample_skip_num
        self.force_update = force_update
        if self.force_update:
            print('Please make sure that your cache file have a corresponding sampling frequence as your directed %s'%(self.sample_skip_num))
        # reduce sampling frequence for reducing caculation time

        pass

    def get_sample_csv_path(self,num):
        return 'c1/c_1_%03d.csv' %(num)

    def get_signal_data_by_pandas(self,num):
        return pd.read_csv(self.get_sample_csv_path(num),header=None)

    @property
    def get_res_data_in_numpy(self):
        storage_path = self.cache_dir_path+self.res_data_storage
        if not self.force_update:
            # try to load numpy array first
            try:
                res_array = np.load(storage_path+'.npy')
                print('Successfully get data from cache...')
                return res_array
            except Exception as e:
                print('Ohhh, cache is not found or destroyed. Reason lists as following.')
                print('-'*20)
                print(e)
                print('-'*20)
                pass
        res_csv_data = self.get_res_data_by_pandas
        res_array = np.array([np.array(i).reshape(1,3) for i in res_csv_data.values])
        np.save(storage_path,res_array)
        return res_array

    @property
    def get_res_data_by_pandas(self):
        return pd.read_csv(self.res_data_path,index_col='cut')

    def gen_x_batch_by_num(self,num):
        pd_data = self.get_signal_data_by_pandas(num)
        print('Retreive data from %s'%(self.get_sample_csv_path(num)))
        # print(np.array(pd_data.values).shape)
        # reduce sample freq for accelerating speed
        return np.array(pd_data.values[::self.sample_skip_num])

    def get_all_sample_data(self):
        storage_path = self.cache_dir_path + self.sample_data_storage
        if not self.force_update:
            try:
                print('# Attention !')
                print('The saved data may have a error because its sampling frequence may differ from %s (current)'%(self.sample_skip_num))
                print('#'*20)
                res_dat = np.load(storage_path+'.npy')
                return res_dat
            except Exception as e:
                print('Ohhh, sample cache is not found or destroyed. Reason lists as following.')
                print('-' * 20)
                print(e)
                print('-' * 20)
        res_dat = []
        for i in range(1, self.total_signal_num + 1):
            res_dat.append(self.gen_x_batch_by_num(i))

        res_dat = np.array(res_dat)
        print('#'*20)
        print('Your computer may become very slow to run, please keep nothing util computer start to respond.')
        print('#'*20)
        np.save(storage_path,res_dat)
        return res_dat




if __name__ =="__main__":
    a = dataSet(force_update=False)
    # print(a.get_sample_csv_path(1))
    # print(a.get_res_data_by_pandas)
    # print(a.get_signal_data_by_pandas(1))
    #
    res_pd_data = a.get_res_data_in_numpy

    x = a.get_all_sample_data()
    y = a.get_res_data_in_numpy

    for index,y_dat in enumerate(res_pd_data):
        print(y_dat)

    # print(a.get_res_data_in_numpy)
    print(a.get_all_sample_data().shape)

