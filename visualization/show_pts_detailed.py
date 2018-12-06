from visualization.data import get_model
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

from sklearn.preprocessing import StandardScaler

model = get_model()
print(model.summary())

FINAL_LAYER = 42

for i in range(133,175):
    layer = model.get_layer("conv1d_%s"%(i))
    weight,bais = layer.get_weights()
    print(weight.shape)
    kernel_size, input_dim, output_dim = weight.shape
    squeezed_weights = weight.reshape((-1, output_dim))
    # focus on output shape
    weight_transponse = squeezed_weights.transpose()
    data = weight_transponse
    weight_embedded = TSNE(n_components=2,init="pca").fit_transform(data)
    weight_embedded = StandardScaler().fit_transform(weight_embedded)
    # print(weight_embedded.shape)
    fig = plt.figure()
    #ax = Axes3D(fig)
    plt.scatter(weight_embedded[:,0],weight_embedded[:,1])
    plt.xlabel("x")
    plt.ylabel("y")
    fig.savefig("plots/direct_pts_conv1d_%s.pdf" %(i))
    # plt.savefig("plots/direct_pts_conv1d_%s.svg" % (i))
    # plt.savefig("plots/direct_pts_conv1d_%s.emf" % (i))
    # plt.show()