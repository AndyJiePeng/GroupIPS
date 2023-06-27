# Clustering visulisation on groups
# 1. Random sample X with size of N
# 2. Ordered them based on the Y vector 
# 3. Color of each node depends on the clusters of our method

# Try T-SNE

# Goal:
# 1. Check whether group action is close to each other in terms of contribution
# 2. Lipschitz loss to see whether it helps to group the action.

import numpy as np 
from syntheticData import Synthetic_data
import matplotlib.pyplot as plt
import openTSNE
from visual_utils import plot as visual_plot
import seaborn as sns
import pandas as pd
import matplotlib.patches as patches
from itertools import chain

def groupVisulisation(model, Lipschitz_loss, cluster, heatmap, group_k, t_d, treatmentEffect=0.8):
    """
        cluster: 2d array [[0,1,2,3], [4,5]].... 
        each array serve as the treatments in the cluster.
    """
    # Random sample the X feature
    print(cluster)
    rng = np.random.RandomState(618)
    # Size and number of feature
    size = 500
    x_d = 10
    X = rng.normal(0, 1, size=[size, x_d])
    synthetic_data = Synthetic_data(size=size,
                                    x_dimension=x_d,
                                    t_d=t_d,
                                    seed=618)

    # Flatten the cluster:
    
    

    # The reward of users varying the treatment
    Y = synthetic_data.get_reward(X, t_d, cluster, treatmentEffect)
    # order = np.random.permutation(group_k)
    # Y = Y_copy.copy()
    # print(Y[0])
    # count = 0
    # for o in order:
    #     print(o)
    #     Y[:,count * 5:(count + 1) * 5] = Y_copy[:,5 * o: 5 * o + 5]
    #     count += 1

    dataFrame = pd.DataFrame(Y)

    fig, ax = plt.subplots(figsize=(12,9), tight_layout=True)
    def histogram_intersection(a, b):
        dist = np.linalg.norm(a-b, ord=2)
        return 1/(1 + dist)
    print(dataFrame.corr(method=histogram_intersection))
    print(Y)
    if len(dataFrame.columns) > 14:
        heatmap = sns.heatmap(heatmap,xticklabels=False, yticklabels=False,
         vmin=0, vmax=1,center=0.8, annot=False, cmap="Blues")
    else:
        heatmap = sns.heatmap(dataFrame.corr(method=histogram_intersection), vmin=0, vmax=1, annot=False,  cmap="BrBG")
    # heatmap.set_title("Heatmap", fontdict={'fontsize':30}, pad=10)

    # ax.add_patch(
    #  patches.Rectangle(
    #      (0, 0),
    #      5.0,
    #      5.0,
    #     #  hatch='|',
    #      edgecolor='orange',
    #      linestyle='--',
    #      fill=False,
    #      lw=10,
    #  ))
    for i in range(6):
        ax.add_patch(
        patches.Rectangle(
            (5 * i, 5 * i),
            5.0,
            5.0,
            #  hatch='|',
            edgecolor='orange',
            linestyle='--',
            fill=False,
            lw=10,
        ))
    # ax.add_patch(
    # patches.Rectangle(
    #      (12, 12),
    #      3.0,
    #      3.0,
    #     #  hatch='|',
    #      edgecolor='orange',
    #      linestyle='--',
    #      fill=False,
    #      lw=10,
    #  ))
    plt.show()
    plt.savefig("./clusterVisualisation/heatMap.png", dpi=300)
    return 1
    # Define the T-SNE to visualise the effect of Y
    aff50 = openTSNE.affinity.PerplexityBasedNN(
    Y,
    perplexity=100,
    n_jobs=32,
    random_state=0,
    )
    init = openTSNE.initialization.rescale(Y[:, :2])
    embedding_standard = openTSNE.TSNE(
        exaggeration=2.38,
        n_jobs=5,
        verbose=True,
    ).fit(affinities=aff50, initialization=init)
    print(embedding_standard.shape)
    y =np.zeros(shape=(t_d))
    for i in range(0, t_d, 5):
        y[i:i+5] = int(i//5)



    plot(embedding_standard, y=y, s=40)




def plot(x, y, **kwargs):
    fig, ax = plt.subplots(ncols=1, figsize=(4, 4))
    # alpha = kwargs.pop("alpha", 0.1)
    alpha=1
    visual_plot(
        x,
        y,
        ax=ax,
        colors=MOUSE_10X_COLORS,
        alpha=alpha,
        draw_legend=False,
        **kwargs,
    )
    
    # visual_plot(
    #     x,
    #     y,
    #     ax=ax[1],
    #     colors=MOUSE_10X_COLORS,
    #     alpha=alpha,
    #     draw_legend=False,
    #     **kwargs,
    # )
    plt.show()
    plt.savefig("./clusterVisualisation/clusterVisual.png", dpi=300)

def main():

    MOUSE_10X_COLORS = {
        0: "#FFFF00",
        1: "#1CE6FF",
        2: "#FF34FF",
        3: "#FF4A46",
        4: "#008941",
        5: "#006FA6",
        6: "#A30059",
        7: "#FFDBE5",
        8: "#7A4900",
        9: "#0000A6",
        10: "#63FFAC",
        11: "#B79762",
        12: "#004D43",
        13: "#8FB0FF",
        14: "#997D87",
        15: "#5A0007",
        16: "#809693",
        17: "#FEFFE6",
        18: "#1B4400",
        19: "#4FC601",
        20: "#3B5DFF",
        21: "#4A3B53",
        22: "#FF2F80",
        23: "#61615A",
        24: "#BA0900",
        25: "#6B7900",
        26: "#00C2A0",
        27: "#FFAA92",
        28: "#FF90C9",
        29: "#B903AA",
        30: "#D16100",
        31: "#DDEFFF",
        32: "#000035",
        33: "#7B4F4B",
        34: "#A1C299",
        35: "#300018",
        36: "#0AA6D8",
        37: "#013349",
        38: "#00846F",
    }
    cluster_num = 6
    num_k = 30

    cluster = [4, 4, 4, 4, 4, 0, 0, 0, 0, 0, 3, 3, 3, 3, 3, 1, 1, 1, 1, 3, 5, 1, 5, 5, 5, 2, 2, 2, 2, 2]
    cluster_index = {i:[] for i in range(cluster_num)}
    for i in range(len(cluster)):
        cluster_index[int(cluster[i])].append(i)
    print(cluster_index)

    from itertools import permutations 
    heatmap = np.zeros(shape=(num_k, num_k))
    for i in range(num_k):
        heatmap[i][i] = 1
    for i in range(cluster_num):
        perm = permutations(cluster_index[i], 2)
        for k, p in list(perm):
            heatmap[k][p] = 1

    print(heatmap)
    groupVisulisation(None, None, cluster_index, heatmap, 6, 30)
    
if __name__ == '__main__':
    main()