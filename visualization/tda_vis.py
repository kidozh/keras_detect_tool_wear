import numpy as np
from mogutda import SimplicialComplex
from visualization.data import get_model
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

    for i in range(43):
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
        hdb_unweighted.fit(projected_data)

        cd = hdb_unweighted.condensed_tree_.plot()

        fig.suptitle('Unweighted HDBSCAN condensed tree plot')
        fig.savefig("HDBSCAN_%s"%(layer))
        # plt.show()


        # Create the graph (we cluster on the projected data and suffer projection loss)
        graph = mapper.map(projected_data,
                           clusterer=sklearn.cluster.DBSCAN(eps=0.1, min_samples=3),
                           # clusterer=sklearn.cluster.DBSCAN(eps=5),
                           #clusterer=HDBSCAN(min_cluster_size=5, gen_min_span_tree=True, allow_single_cluster=True),
                           # coverer=km.Cover(35, 0.9)
                           )
        print(layer,"map successfully")
        simplicial_complex = graph
        print(simplicial_complex["nodes"])
        print(simplicial_complex["links"])
        for i, (node_id, member_ids) in enumerate(graph["nodes"].items()):
            print(i,node_id,member_ids)
        # # print(simplicial_complex["meta"])


        print(labels)
        print(tooltip_s)
        try:
            # mapper.visualize(graph,
            #              path_html="label_layer_%s_keplermapper_weights_visualization.html"%(layer),
            #             custom_tooltips = labels
            #                  )

            mapper.visualize(graph,path_html="picture_layer_%s_keplermapper_weights_visualization.html" % (layer),
                             custom_tooltips=tooltip_s
                             )
        except Exception as e:
            print(e)
            # raise (e)
            print(projected_data.shape)

            # input("Error Happened, Press Enter to Continue")




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
