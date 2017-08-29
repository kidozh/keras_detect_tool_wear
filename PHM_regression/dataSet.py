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
        if not self.force_update:
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



class testDataSet(object):
    res_data_path = 'c4_wear.csv'
    cache_dir_path = '.test_cache/'
    res_data_storage = 'test_res_dat'
    sample_data_storage = 'test_sample_dat'
    total_signal_num = 315

    def __init__(self, force_update=False, sample_skip_num=50):
        self.sample_skip_num = sample_skip_num
        self.force_update = force_update
        if not self.force_update:
            print(
                'Please make sure that your cache file have a corresponding sampling frequence as your directed %s' % (
                self.sample_skip_num))
        # reduce sampling frequence for reducing caculation time

        pass
    def get_sample_csv_path(self,num):
        return 'c4/c_4_%03d.csv' %(num)

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
        print('# '*20)
        print('Your computer may become very slow to run, please keep nothing util computer start to respond.')
        print('# '*20)
        np.save(storage_path,res_dat)
        return res_dat



class allDataSet(object):
    # res_data_path = 'c1_wear.csv'
    cache_dir_path = '.cache/'
    res_data_storage = 'all_res_dat'
    sample_data_storage = 'all_sample_dat'
    total_signal_num = 315
    # only 1,4 and 6 is correct
    sample_loc = 1

    @property
    def res_data_path(self):
        return 'c%s_wear.csv' %(self.sample_loc)

    def __init__(self, force_update=False, sample_skip_num=50,sample_loc=1):
        self.sample_skip_num = sample_skip_num
        self.force_update = force_update
        pass

    def get_sample_csv_path(self, num):
        return 'c%s/c_%s_%03d.csv' % (self.sample_loc,self.sample_loc,num)

    def get_signal_data_by_pandas(self, num):
        return pd.read_csv(self.get_sample_csv_path(num), header=None)

    @property
    def get_res_data_in_numpy(self):
        # remove cache because it's not needed
        res_csv_data = self.get_res_data_by_pandas
        res_array = np.array([np.array(i).reshape(1, 3) for i in res_csv_data.values])
        # np.save(storage_path, res_array)
        return res_array

    @property
    def get_res_data_by_pandas(self):
        return pd.read_csv(self.res_data_path, index_col='cut')

    def gen_x_batch_by_num(self, num):
        pd_data = self.get_signal_data_by_pandas(num)
        print('Retreive data from %s' % (self.get_sample_csv_path(num)))
        # print(np.array(pd_data.values).shape)
        # reduce sample freq for accelerating speed
        return np.array(pd_data.values[::self.sample_skip_num])

    def get_all_sample_data(self):
        storage_path = self.cache_dir_path + self.sample_data_storage
        res_dat = []
        for i in range(1, self.total_signal_num + 1):
            res_dat.append(self.gen_x_batch_by_num(i))

        # res_dat = np.array(res_dat)
        # np.save(storage_path, res_dat)
        return res_dat

    def get_all_loc_y_sample_data(self):
        storage_path = self.cache_dir_path + self.res_data_storage
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


        self.sample_loc = 1
        y_dat = self.get_res_data_in_numpy
        for i in [4,6]:
            print('Fetch %s'%(i))
            self.sample_loc = i
            y_dat = np.append(self.get_res_data_in_numpy,y_dat,axis=0)

        print(y_dat.shape)

        print('-' * 40)
        print('Your computer may become very slow to run, please keep nothing util computer start to respond.')
        print('-' * 40)

        np.save(storage_path,y_dat)
        return y_dat

    def get_all_loc_x_sample_data(self):
        storage_path = self.cache_dir_path + self.sample_data_storage
        if not self.force_update:
            try:
                print('# Attention !')
                print('The saved data may have a error because its sampling frequence may differ from %s (current)' % (
                self.sample_skip_num))
                print('#' * 20)
                res_dat = np.load(storage_path + '.npy')
                return res_dat
            except Exception as e:
                print('Ohhh, sample cache is not found or destroyed. Reason lists as following.')
                print('-' * 20)
                print(e)
                print('-' * 20)

        self.sample_loc = 1
        x_dat = self.get_all_sample_data()
        for i in [4, 6]:
            self.sample_loc = i
            print('Fetch %s'%(i))
            x_dat = np.append(self.get_all_sample_data(), x_dat,axis=0)

        print('-' * 40)
        print('Your computer may become very slow to run, please keep nothing util computer start to respond.')
        print('-' * 40)

        print(x_dat.shape)

        np.save(storage_path,x_dat)
        return x_dat

class auto_stride_dataset(object):
    # extract all data
    cache_dir_path = '.cache/'
    total_signal_num = 315
    # only 1,4 and 6 is correct
    sample_label = [1,4,6]
    sample_loc = 1

    @property
    def sample_data_storage(self):
        return 'all_sample_dat_num_%s' % (self.sample_num)

    @property
    def res_data_storage(self):
        return 'all_res_dat_num_%s'%(self.sample_num)

    @property
    def res_data_path(self):
        return 'c%s_wear.csv' % (self.sample_loc)

    def __init__(self, force_update=False, sample_num = 5000):
        self.sample_num = sample_num
        self.force_update = force_update
        pass


    def get_sample_csv_path(self, num):
        return 'c%s/c_%s_%03d.csv' % (self.sample_loc,self.sample_loc,num)

    def get_signal_data_by_pandas(self, num):
        return pd.read_csv(self.get_sample_csv_path(num), header=None)

    @property
    def get_res_data_in_numpy(self):
        # remove cache because it's not needed
        res_csv_data = self.get_res_data_by_pandas
        res_array = np.array([np.array(i).reshape(3) for i in res_csv_data.values])
        # np.save(storage_path, res_array)
        return res_array

    @property
    def get_res_data_by_pandas(self):
        return pd.read_csv(self.res_data_path, index_col='cut')

    def gen_x_batch_by_num(self, num):
        pd_data = self.get_signal_data_by_pandas(num)
        print('Retreive data from %s' % (self.get_sample_csv_path(num)))
        # print(np.array(pd_data.values).shape)
        # reduce sample freq for accelerating speed

        interval = len(pd_data.values) // self.sample_num

        print('# Num %s, total %s, interval %s' %(num,len(pd_data.values),interval))

        return np.array(pd_data.values[::interval][:self.sample_num])

    def get_all_sample_data(self):
        storage_path = self.cache_dir_path + self.sample_data_storage
        res_dat = []
        for i in range(1, self.total_signal_num + 1):
            res_dat.append(self.gen_x_batch_by_num(i))

        # res_dat = np.array(res_dat)
        # np.save(storage_path, res_dat)
        return res_dat

    def get_all_loc_y_sample_data(self):
        storage_path = self.cache_dir_path + self.res_data_storage
        if not self.force_update:
            try:

                print('#'*20)
                res_dat = np.load(storage_path+'.npy')
                return res_dat
            except Exception as e:
                print('Ohhh, sample cache is not found or destroyed. Reason lists as following.')
                print('-' * 20)
                print(e)
                print('-' * 20)


        self.sample_loc = 1
        y_dat = self.get_res_data_in_numpy
        for i in [4,6]:
            print('Fetch %s'%(i))
            self.sample_loc = i
            y_dat = np.append(self.get_res_data_in_numpy,y_dat,axis=0)

        print(y_dat.shape)

        print('-' * 40)
        print('Your computer may become very slow to run, please keep nothing util computer start to respond.')
        print('-' * 40)

        np.save(storage_path,y_dat)
        return y_dat

    def get_all_loc_x_sample_data(self):
        storage_path = self.cache_dir_path + self.sample_data_storage
        if not self.force_update:
            try:
                print('# Attention !')
                print('#' * 20)
                res_dat = np.load(storage_path + '.npy')
                return res_dat
            except Exception as e:
                print('Ohhh, sample cache is not found or destroyed. Reason lists as following.')
                print('-' * 20)
                print(e)
                print('-' * 20)

        self.sample_loc = 1
        x_dat = self.get_all_sample_data()
        for i in [4, 6]:
            self.sample_loc = i
            print('Fetch %s'%(i))
            x_dat = np.append(self.get_all_sample_data(), x_dat,axis=0)

        print('-' * 40)
        print('Your computer may become very slow to run, please keep nothing util computer start to respond.')
        print('-' * 40)

        print(x_dat.shape)

        np.save(storage_path,x_dat)
        return x_dat

class wavelet_dataset(object):
    def __init__(self):
        a = auto_stride_dataset()
        self.sample_x = a.get_all_loc_x_sample_data()
        self.sample_y = a.get_all_loc_y_sample_data()


    def gen_y_dat(self):
        return self.sample_y

    def gen_x_dat(self):
        import pywt
        print('Wavelet is used, db4 and 2 level')
        A2,D2,D1 = pywt.wavedec(self.sample_x,'db4',mode='symmetric',level=2,axis=1)
        # print(self.sample_x[1,:,1].shape,A2.shape,D2.shape,D1.shape)
        # import matplotlib.pyplot as plt
        # plt.plot(D1)
        # plt.show()
        return A2,D2,D1


if __name__ =="__main__":
    wavelet_dat = wavelet_dataset()

    # a = allDataSet(force_update=True)
    # y = a.get_all_loc_y_sample_data()
    # x = a.get_all_loc_x_sample_data()


