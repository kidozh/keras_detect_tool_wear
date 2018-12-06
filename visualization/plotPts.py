import matplotlib as mpl
mpl.use('cairo')
from visualization.data import *
from visualization.tda_vis import TOPO_DIR
import json
import igraph as ig
import re

HTML_FILE = "picture_overlap_layer_RUL_TRANSVERSE_PYRAMID_DEPTH_20_LR_0.01_conv1d_43.html"

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

PTS_COLOR_PATTERNS = ["#FF0000","#FF1400","#FF2800","#FF3c00","#FF5000","#FF6400",
                      "#FF7800","#FF8c00","#FFa000","#FFb400","#FFc800","#FFdc00",
                      "#FFf000","#fdff00","#b0ff00","#65ff00","#17ff00","#00ff36",
                      "#00ff83","#00ffd0","#00e4ff","#00c4ff","#00a4ff","#00a4ff",
                      "#0084ff","#0064ff","#0044ff","#0022ff","#0002ff","#0100ff",
                      "#0300ff","#0500ff"]

from bs4 import BeautifulSoup

import matplotlib.pyplot as plt

from matplotlib.artist import Artist
from igraph import BoundingBox, Graph, palettes
plt.rc("font",family="Noto Sans CJK SC")




from igraph import *
from PIL import Image

colors_type = ["yellow", "red", "green", "coral", "alice blue", "cyan", "pink",
               "gray", "blue", "green yellow", "orange", "light blue", "hot pink", "light green", "gold"]

def autolabel(rects):
    for rect in rects:
        height = rect.get_height()
        plt.text(rect.get_x()+rect.get_width()/6., 1.01*height, '%s' % float(height))


def PlotNetworks(graph, real_label="Unknown Type"):
    ## read files
    network = graph
    Graph.to_undirected(network)
    if (real_label != "Unknown Type"):
        f2 = open(real_label)
        line = f2.readline()
        line = line.strip()
        str_line = line.split('\t')
        rlabel = [int(ele) for ele in str_line]
        network.vs["rlabel"] = rlabel

    # plot networks
    nnodes = len(network.vs)
    network.vs["name"] = [str(i + 1) for i in range(nnodes)]

    layout = network.layout("drl")
    visual_style = {}
    if (nnodes < 100):
        visual_style["vertex_size"] = 22
    else:
        visual_style["vertex_size"] = 18
    visual_style["vertex_label"] = network.vs["name"]
    visual_style["layout"] = layout
    visual_style["bbox"] = (500, 500)
    visual_style["margin"] = 20
    visual_style["edge_curved"] = 0.3
    # visual_style["vertex_color"] = [colors_type[i - 1] for i in network.vs["dlabel"]]
    plot(network, "social_network1.png", **visual_style)
    figure1 = Image.open("social_network1.png")
    figure1.save("social_network1.bmp")
    if (real_label != "Unknown Type"):
        visual_style["vertex_color"] = [colors_type[i - 1] for i in network.vs["rlabel"]]
        plot(network, "social_network2.png", **visual_style)
        figure2 = Image.open("social_network2.png")
        figure2.save("social_network2.bmp")


class GraphArtist(Artist):
    def __init__(self, graph, bbox, palette=None, *args, **kwds):
        Artist.__init__(self)

        if not isinstance(graph, Graph):
            raise TypeError("expected igraph.Graph, got %r" % type(graph))

        self.graph = graph
        self.palette = palette or palettes["gray"]
        self.bbox = BoundingBox(bbox)
        self.args = args
        self.kwds = kwds

    def draw(self, renderer):
        from matplotlib.backends.backend_cairo import RendererCairo
        if not isinstance(renderer, RendererCairo):
            raise TypeError("graph plotting is supported only on Cairo backends")
        self.graph.__plot__(renderer.gc.ctx, self.bbox, self.palette, *self.args, **self.kwds)


class visulizer(object):

    def __init__(self):
        self.html_path = "%s%s"%(TOPO_DIR,HTML_FILE)
        with open(self.html_path,"r") as html_file:
            self.html_content = html_file.read()
        self.graph = self.get_json_data(self.html_content)

    def get_json_data(self,html_content:str):
        soup = BeautifulSoup(html_content, "lxml")
        data_graph_div = soup.find_all("div")[0]
        data_graph_data = data_graph_div.get("data-graph")
        return json.loads(data_graph_data)

    def get_all_attributes(self):
        return [i for i in self.graph.keys()]

    def get_all_node(self):
        return [node for node in self.graph["nodes"]]

    def plot_igraph(self,kmgraph):
        n_nodes = len(kmgraph['nodes'])
        if n_nodes == 0:
            raise ValueError('Your graph has 0 nodes')
        G = ig.Graph(n=n_nodes)

        # print(kmgraph["links"])
        links = [(e['source'], e['target']) for e in kmgraph['links']]
        G.add_edges(links)
        # color
        for i in range(n_nodes):
            pts_color_number = self.graph["nodes"][i]["color"]
            # print(pts_color_number)
            pts_color = PTS_COLOR_PATTERNS[int(pts_color_number)]
            G.vs[i]["color"] = pts_color

        return G

    def plot_graph(self):
        from visualization import plotlyviz as pl
        from plotly.offline import download_plotlyjs, init_notebook_mode, iplot
        # PLot the graph with Kamada Kawai layout
        fig = plt.figure()
        axes = fig.add_subplot(211)
        graph = self.graph

        # fig = plt.figure()
        #plt.subplot(211)
        #ig.plot(graph,target="graph_%s.pdf"%(KERAS_MODEL_NAME))
        network = self.plot_igraph(graph)

        # PlotNetworks(network)
        plot(network,target="plot_graph_%s.pdf"%(KERAS_MODEL_NAME))
        graph = network
        graph_artist = GraphArtist(graph,
                                   bbox=(0, 0, 450, 500),
                                   margin=(20, 20, 20, 20)
                                   )
        graph_artist.set_zorder(float('inf'))
        axes.artists.append(graph_artist)
        plt.axis("off")
        # plt.subplot(212)
        axes = fig.add_subplot(212)
        histogram_info = self.get_histogram()
        raw_percent_val_list = [float(i[0]) for i in histogram_info]
        percent_val_list = []
        for index,value in enumerate(raw_percent_val_list):
            label = "%s"%(value)
            for repeat_num in range(index):
                if raw_percent_val_list[repeat_num] == value:
                    label += " "
            percent_val_list.append(label)


        percent_color_list = [i[1] for i in histogram_info]
        percent_height_list = [i[2] for i in histogram_info]
        label_num = [i+1 for i in range(len(histogram_info))]
        # for bin_val,bin_color,bin_height in histogram_info:
        #
        #     axes.bar([bin_val],[bin_height],color=bin_color)
        rect = axes.bar(label_num,raw_percent_val_list,color=percent_color_list,align="center")
        axes.set_xticks(label_num)
        axes.set_ylabel("占总数比 (%)")
        axes.set_xlabel("分块编号")
        autolabel(rect)





        # cmap = mpl.cm.jet
        # print(cmap)
        # norm = mpl.colors.Normalize(vmin=0, vmax=50)  # 自己设置
        # sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
        # sm._A = []
        # cbar = fig.colorbar(sm)
        # fig.show()
        fig.set_size_inches(7, 14)
        fig.savefig("Graph_%s.pdf"%(HTML_FILE.split(".")[:-1]))

    def get_histogram(self):
        soup = BeautifulSoup(self.html_content, "lxml")
        data_graph_div = soup.find_all(id="histogram")[0]
        # print(data_graph_div)
        histogram_info = []
        for histo_bin_obj in data_graph_div.find_all(name='div',attrs={"class":"bin"}):
            bin_val = re.findall(r"<div>(.+)%</div>",str(histo_bin_obj))[0]
            # print(bin_val)
            style_str = histo_bin_obj.get("style")
            bin_color = re.findall(r"background:(.+)",style_str)[0]
            # print(style_str,bin_color)
            bin_height = re.findall(r"height:(.+)px", style_str)[0]
            histogram_info.append((str(bin_val),bin_color,float(bin_height)))

        return histogram_info



if __name__ == "__main__":
    vis = visulizer()
    vis.get_histogram()
    # import matplotlib.pyplot as plt
    #
    # fig = plt.figure()
    # axes = fig.add_subplot(111)
    vis.plot_graph()
