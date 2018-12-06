import numpy as np
import matplotlib.pyplot as plt
from PHM_regression.dataSet import rul_data_source
from sklearn.cluster import KMeans
import pandas as pd
from pandas.tools.plotting import *
from sklearn.manifold import TSNE
from matplotlib import animation

# a = rul_data_source()
# x, y = a.get_data()

try:
    conv1_output = np.load('conv1d.npy')
except Exception as e:
    print(e)

INDEX = 600

fig = plt.figure()
axes1 = fig.add_subplot(111)
line, = axes1.plot(np.random.rand(10))

def get_data_by_index(index):
    '''
    get directed data by index
    :param index:
    :return: (timestep,dimension) => (249,128)
    '''
    # need reshape for better extraction -> (128,249)
    dat_list = []
    for i in range(128):
        a = conv1_output[index, :, i]
        dat_list.append(a)
    np_dat = np.array(dat_list)
    print(np_dat.shape)
    return np_dat

def conduct_k_means(data,n_clusters=35):

    estimator = KMeans(n_clusters=n_clusters)  # 构造聚类器
    estimator.fit(data)  # 聚类
    label_pred = estimator.labels_  # 获取聚类标签
    centroids = estimator.cluster_centers_  # 获取聚类中心
    inertia = estimator.inertia_  # 获取聚类准则的总和
    return label_pred,centroids,inertia

def plot_TSNE(np_data,label,i):
    ax.cla()
    tsne = TSNE(n_components=2,init='pca', random_state=0)
    x = tsne.fit_transform(np_data)
    print('TSNE Shape',x.shape)
    ax.set_title("frame {}".format(i))
    for i in range(x.shape[0]):
        ax.scatter(x[i, 0], x[i, 1],label=str(label[i]))

    #plt.legend()
    plt.pause(0.1)

if __name__ == '__main__':
    fig, ax = plt.subplots()

    for i in range(40000):
        if i<298:
            continue
        dat = get_data_by_index(i)
        label,centroids,inertia = conduct_k_means(dat,n_clusters=10)
        plot_TSNE(dat,label,i)
