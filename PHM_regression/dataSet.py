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
        return pd.read_csv(self.get_sample_csv_path(num), header=None, engine='c')

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

    def get_duration_data(self):
        rul_path = 'rul_data'
        storage_path = self.cache_dir_path + rul_path
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
        save_array = np.zeros([self.total_signal_num*3])
        cnt = -1

        for i in [1,4,6]:
            cnt += 1
            self.sample_loc = i
            # read data from dataset
            for j in range(1, self.total_signal_num + 1):
                pd_data = self.get_signal_data_by_pandas(j)

                period_time = 1/(50e3)

                duration = len(pd_data.values)

                print(i,j,duration)

                save_array[cnt*self.total_signal_num+j-1] = period_time*duration



        np.save(storage_path,save_array)
        return save_array


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

    def get_rul_dat(self):
        a = auto_stride_dataset()
        duration_dat =  a.get_duration_data()

        rul_array = np.zeros([315 * 3])

        for i in range(0,3):
            slice = duration_dat[i*315:(i+1)*315]
            remain_time = slice.sum()
            print('#',remain_time)
            for j in range(0,315):
                rul_array[i*315+j] = remain_time - duration_dat[i*315+j]
                remain_time -= duration_dat[i*315+j]

        import matplotlib.pyplot as plt

        #plt.plot(rul_array)
        # plt.show()
        return rul_array



class rul_data_source(object):
    DATA_FILE_NUMBER = [1,4,6]

    SCRAP_PERIOD = [270,265,219]

    MAX_CUT_NUM = 315

    frequency = 50e3

    CACHE_DIR = '.cache'

    X_SAVE_PATH = '%s/%s' %(CACHE_DIR,'RUL_X')

    Y_SAVE_PATH = '%s/%s' %(CACHE_DIR,'RUL_Y')

    SAMPLE_TIMESTEP = 5000

    SECOND_SAMPLE = 5

    def __init__(self):
        pass

    def get_sample_csv_path(self,tool,num):
        return 'c%s/c_%s_%03d.csv' % (tool,tool,num)

    def get_pandas_dat(self,tool,num):
        return pd.read_csv(self.get_sample_csv_path(tool,num),header=None)

    def get_data(self):
        try:
            X = np.load(self.X_SAVE_PATH+'.npy')
            Y = np.load(self.Y_SAVE_PATH+'.npy')
            return X,Y
        except Exception as e:
            print(e)

        X = []
        Y = []

        for cut_list_index,cut_index in enumerate(self.DATA_FILE_NUMBER):
            used_duration = 0
            section_time = []
            scapped_time = 0
            for j in range(1,self.MAX_CUT_NUM+1):
                print('>',cut_index,j)

                if j == self.SCRAP_PERIOD[cut_list_index]:
                    # record it
                    scapped_time = used_duration
                    print('> find scrapped time',j,used_duration)

                pd_data = self.get_pandas_dat(cut_index,j)
                pd_value = pd_data.values
                cut_section = pd_value.shape[0]
                # traverse it and extract data
                for timestep_index in range(cut_section//self.SAMPLE_TIMESTEP):
                    raw_dat = pd_value[(timestep_index) * self.SAMPLE_TIMESTEP:(timestep_index + 1) * self.SAMPLE_TIMESTEP, :]
                    X.append(raw_dat[::20])
                    section_time.append(used_duration+1/self.frequency*self.SAMPLE_TIMESTEP*timestep_index)

                #
                used_duration += cut_section * 1/self.frequency

            # - scrapped time
            section_time = [scapped_time-i for i in section_time]
            Y.extend(section_time)

        X = np.array(X)
        Y = np.array(Y)

        # store it
        np.save(self.X_SAVE_PATH,X)
        np.save(self.Y_SAVE_PATH,Y)
        return X,Y



if __name__ =="__main__":
    #wavelet_dat = wavelet_dataset()
    #rul_dat = wavelet_dat.get_rul_dat()
    #print(rul_dat)

    rul = rul_data_source()
    train_x,train_y = rul.get_data()
    print(train_x.shape,train_y.shape)

    # a = allDataSet(force_update=True)
    # y = a.get_all_loc_y_sample_data()
    # x = a.get_all_loc_x_sample_data()


