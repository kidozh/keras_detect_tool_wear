import numpy as np
from mogutda import SimplicialComplex
from visualization.data import get_model,KERAS_MODEL_NAME
import matplotlib
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from mapper import em
from mapper import ff
from mapper import slc
from mapper.em_help import *
from tests.em_3d_help import *
import kmapper as km
import sklearn
import tests.params
from hdbscan import HDBSCAN
from scipy.misc import imsave, toimage
import io
import sys
import base64

from PIL import Image
import PIL
import random
import numpy.random

TOPO_DIR = "RUL_PHM_vis/"

pl_jet=[[0.0, 'rgb(0, 0, 127)'],
 [0.03, 'rgb(0, 0, 163)'],
 [0.07, 'rgb(0, 0, 204)'],
 [0.1, 'rgb(0, 0, 241)'],
 [0.13, 'rgb(0, 8, 255)'],
 [0.17, 'rgb(0, 40, 255)'],
 [0.2, 'rgb(0, 76, 255)'],
 [0.23, 'rgb(0, 108, 255)'],
 [0.27, 'rgb(0, 144, 255)'],
 [0.3, 'rgb(0, 176, 255)'],
 [0.33, 'rgb(0, 212, 255)'],
 [0.37, 'rgb(12, 244, 234)'],
 [0.4, 'rgb(41, 255, 205)'],
 [0.43, 'rgb(66, 255, 179)'],
 [0.47, 'rgb(95, 255, 150)'],
 [0.5, 'rgb(124, 255, 121)'],
 [0.53, 'rgb(150, 255, 95)'],
 [0.57, 'rgb(179, 255, 66)'],
 [0.6, 'rgb(205, 255, 41)'],
 [0.63, 'rgb(234, 255, 12)'],
 [0.67, 'rgb(255, 229, 0)'],
 [0.7, 'rgb(255, 196, 0)'],
 [0.73, 'rgb(255, 166, 0)'],
 [0.77, 'rgb(255, 133, 0)'],
 [0.8, 'rgb(255, 103, 0)'],
 [0.83, 'rgb(255, 70, 0)'],
 [0.87, 'rgb(255, 40, 0)'],
 [0.9, 'rgb(241, 7, 0)'],
 [0.93, 'rgb(204, 0, 0)'],
 [0.97, 'rgb(163, 0, 0)'],
 [1.0, 'rgb(127, 0, 0)']]


pl_brewer=[[0.0, '#a50026'],
           [0.1, '#d73027'],
           [0.2, '#f46d43'],
           [0.3, '#fdae61'],
           [0.4, '#fee08b'],
           [0.5, '#ffffbf'],
           [0.6, '#d9ef8b'],
           [0.7, '#a6d96a'],
           [0.8, '#66bd63'],
           [0.9, '#1a9850'],
           [1.0, '#006837']]

SEED_NUM = 123
random.seed(SEED_NUM)
numpy.random.seed(SEED_NUM)

mkdir_p(tests.params.PLOT_PATH)

def get_gradient_color(startColor:str,endColor:str,colorNum:int):
    from colour import Color
    startColorObj = Color(startColor)
    endColorObj = Color(endColor)

    return list(startColorObj.range_to(endColorObj,colorNum))

def plot_tda(data):
    data = data.reshape((-1,3))
    plot_3d(data, fname='3d.png')

    foo = em.ExploreMapper(data, slc.SingleLinkageClustering, ff.EccentricityP)

def zoom_out_image(img:Image,zoom_times:int):
    dst_image = Image.new(img.mode,(img.size[0]*zoom_times,img.size[1]*zoom_times))
    for col in range(img.size[0]):
        for row in range(img.size[1]):
            dst_image[col:col*zoom_times,row:row*zoom_times] = img[col,row]

    return dst_image

if __name__ == "__main__":
    model = get_model()
    # weight -> 0, bias -> 1
    print(model.summary())


    mkdir_p(TOPO_DIR)

    for i in range(47):
        if i < 0:
            continue
        layer = "conv1d_%s"%(i+1)
        print(layer,":",model.get_layer(layer).get_weights()[0].shape)
        weight = model.get_layer(layer).get_weights()[0]
        kernel_size, input_dim, output_dim = weight.shape
        # generate gray picture
        tooltip_s = []
        for j in range(output_dim):
            output = io.BytesIO()
            print(weight[:,:,j].shape)
            img = toimage(weight[:,:,j])
            ZOOM_TIMES = 10
            img = img.resize((img.size[0]*ZOOM_TIMES,img.size[1]*ZOOM_TIMES))
            # img = zoom_out_image(img,20)
            img.save(output, format="PNG")
            contents = output.getvalue()
            img_encoded = base64.b64encode(contents)
            img_tag = """<img src="data:image/png;base64,{}"> """.format(img_encoded.decode('utf-8'))
            tooltip_s.append(img_tag)
            output.close()
        tooltip_s = np.array(tooltip_s)

        squeezed_weights = weight.reshape((-1,output_dim))
        # focus on output shape
        weight_transponse = squeezed_weights.transpose()
        data = weight_transponse
        labels = np.linspace(1,output_dim,output_dim)
        print("transpose shape :",weight_transponse.shape)
        mapper = km.KeplerMapper(verbose=2)
        # projected_data = data
        projected_data = mapper.fit_transform(data,projection=sklearn.manifold.TSNE())
        print("TSNE result",projected_data.shape)

        fig = plt.figure()

        hdb_unweighted = HDBSCAN(min_cluster_size=3, gen_min_span_tree=True, allow_single_cluster=True)
        clusterer = hdb_unweighted.fit(projected_data)

        hdb_unweighted.single_linkage_tree_.plot()
        fig.savefig(TOPO_DIR+"single_linkage_tree_%s_%s.pdf"%(KERAS_MODEL_NAME.split(".")[0],layer))
        fig = plt.figure()
        import seaborn as sns
        color_palette = sns.color_palette('deep', 8)
        try:
            cluster_colors = [color_palette[x] if x >= 0
                          else (0.5, 0.5, 0.5)
                          for x in clusterer.labels_]
            cluster_member_colors = [sns.desaturate(x, p) for x, p in
                                 zip(cluster_colors, clusterer.probabilities_)]
            plt.scatter(*projected_data.T,
                    s=50,
                    linewidth=0,
                    c=cluster_member_colors,
                    alpha=0.25
                    )
            fig.savefig(TOPO_DIR+"DATA_DIST%s_%s.pdf"%(KERAS_MODEL_NAME.split(".")[0],layer))
        except Exception as e:
            print(e)
        fig = None
        fig = plt.figure()
        cd = hdb_unweighted.condensed_tree_.plot()

        # fig.suptitle('Unweighted HDBSCAN condensed tree plot')
        fig.savefig(TOPO_DIR+"HDBSCAN_%s_%s.pdf"%(KERAS_MODEL_NAME.split(".")[0],layer))
        # plt.show()


        # Create the graph (we cluster on the projected data and suffer projection loss)
        graph = mapper.map(projected_data,
                           clusterer=sklearn.cluster.DBSCAN(eps=0.8, min_samples=3),
                           # clusterer=sklearn.cluster.DBSCAN(eps=5),
                           #clusterer=HDBSCAN(min_cluster_size=5, gen_min_span_tree=True, allow_single_cluster=True),
                           # coverer=km.Cover(35, 0.9)
                           coverer=km.Cover(nr_cubes=10, overlap_perc=0.2),
                           )
        print(layer,"map successfully")
        simplicial_complex = graph


        print(labels)
        print(tooltip_s)
        try:
            # mapper.visualize(graph,
            #              path_html="label_layer_%s_keplermapper_weights_visualization.html"%(layer),
            #             custom_tooltips = labels
            #                  )

            mapper.visualize(graph,path_html=TOPO_DIR + "picture_overlap_layer_%s_%s.html" % (KERAS_MODEL_NAME,layer),custom_tooltips=tooltip_s)
            # from visualization import plotlyviz as pl
            # from plotly.offline import download_plotlyjs, init_notebook_mode, iplot
            # # PLot the graph with Kamada Kawai layout
            # plotly_graph_data = pl.plotly_graph(graph,
            #                                     # tooltips=labels,
            #                                     graph_layout='kk',
            #                                     colorscale=pl_jet,
            #                                     factor_size=3,
            #                                     edge_linewidth=0.5)  # here colorscale could be 'jet'; in this case the above definition
            # # of pl_jet is not necessary anymore
            # layout = pl.plot_layout(title='Mapper graph of digits dataset', width=800, height=800,
            #                         # annotation_text=meta,
            #                         bgcolor='rgba(0,0,0, 0.95)')
            #
            # fig = dict(data=plotly_graph_data, layout=layout)
            # iplot(fig)
        except Exception as e:
            # raise e
            print(e)

    print(model.summary())




    # weight = model.get_layer("conv1d_32").get_weights()[0]
    #
    # # kernel size, input shape, output shape
    # fig = plt.figure()
    # color_list = get_gradient_color("#b92b27","#1565C0",weight.shape[2])
    # ax = plt.subplot(111, projection='3d')
    # for i in range(weight.shape[2]):
    #
    #     print("$",i)
    #
    #     kernel_weight = weight[:,:,i]
    #     x,y,z = kernel_weight[0,:],kernel_weight[1,:],kernel_weight[2,:]
    #     plot_tda(kernel_weight)
    #     ax.scatter(x,y,z,c=color_list[i].get_hex_l())
    #     # plt.legend()
    #     plt.show()
    #
    # print(model.summary())
    # print(weight.shape)
