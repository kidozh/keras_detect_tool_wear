import pandas as pd
import numpy as np

CSV_PATH = 'PHM08.csv'

class data_source(object):

    id_range = 218

    def __init__(self):
        self.raw_data = pd.read_csv(CSV_PATH)
        #print(self.raw_data)
        pass

    def gen_sample(self):
        signal_list = []
        rul_list = []
        for i in range(1,self.id_range+1):
            extract = self.raw_data[self.raw_data.id == i]
            sensors_data = extract[extract.columns.difference(['id','cycle','RUL'])]
            rul_data = extract['RUL']
            signal_list.append(sensors_data.values)
            rul_list.append(rul_data.values)

        return signal_list,rul_list


class test_data_source(object):
    test_data_path = 'test.csv'
    id_range = 218
    def __init__(self):
        self.raw_data = pd.read_csv(self.test_data_path)
        pass

    def gen_test_dat(self):
        signal_list = []
        for i in range(1, self.id_range + 1):
            extract = self.raw_data[self.raw_data.id == i]
            sensors_data = extract[extract.columns.difference(['id', 'cycle'])]
            signal_list.append(sensors_data.values)

        return np.array(signal_list)

if __name__ == '__main__':
    a = test_data_source()
    print(a.gen_test_dat())


    # a = data_source()
    # x_dat,y_dat = a.gen_sample()
    # for i in range(a.id_range):
    #     print(x_dat[i].shape,y_dat[i].shape)