import numpy as np
import collections
import matplotlib.pyplot as plt
from matplotlib.pyplot import plot, show


def plot_fig(zx, labels_list, categories, zphi):
    collection_center = {}
    markers = np.copy(labels_list)
    marker_list = ["."]*len(categories)
    collection_znk_label = labels_list
    collection_znk_label = np.array(collection_znk_label)
    collection_znk_label = np.array([str(e) for e in collection_znk_label])

    for key in categories:
        np.place(collection_znk_label, collection_znk_label==key, categories[key])

    fig, ax = plt.subplots( figsize=(20, 20))
    ax.scatter(zx[:,0], zx[:,1], alpha=0.8, c=collection_znk_label, facecolors='none', s=8 )

    for indx, topic in enumerate(zphi):
        ax.scatter(zphi[:, 0], zphi[:, 1], alpha=1.0,  edgecolors='black', facecolors='none', s=30)

def get_topwords(beta, id_vocab):
    topic_indx = 0
    topwords_topic = []
    for i in range(len(beta)):
        topwords_topic.append( str(topic_indx)+": "+ " ".join([id_vocab[j] for j in beta[i].argsort()[:-10 - 1:-1]]))
        topic_indx+=1
    return topwords_topic