
import matplotlib.pyplot as plt
from matplotlib.cm import get_cmap
from matplotlib.colors import Normalize
import random
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

from frechet_set_computation import *
from frechet_set_estimation import *
from topology_functions import *
from read_tree_data import *

# Function to plot a binary tree
def plot_binary_tree(tree, ax, x, y, x_spacing, y_spacing):
    """
    Input: tree - List representing a binary tree: contains the lengths of its inner edges,
           ax - Matplotlib axis to plot the tree,
           x, y - Coordinates for the starting point of the tree,
           x_spacing, y_spacing - Spacing between nodes in the x and y directions.
    Output: Plots the binary tree on the given axis.
    """
    if not tree:
        return
    # Draw the current node
    #ax.add_patch(plt.Circle((x, y), radius=0.2, facecolor='lightblue'))
    if len(tree)==1: #leaf
        ax.text(x, y-0.1, str(tree[0]), ha='center', va='center', color='black', fontsize=12)
    else:
        nb_left_nodes = len([x for x in nodes_in_topology(tree[0])])
        nb_right_nodes = len([x for x in nodes_in_topology(tree[1])])
        total_nb_nodes = nb_left_nodes + nb_right_nodes

        # Draw left subtree
        x_left = x - x_spacing*(nb_right_nodes/2)
        y_left = y - y_spacing*(nb_right_nodes/2)
        ax.plot([x, x_left], [y, y_left], color='black')
        plot_binary_tree(tree[0], ax, x_left, y_left, x_spacing, y_spacing)

        x_right = x + x_spacing*(nb_left_nodes/2)
        y_right = y - y_spacing*(nb_left_nodes/2)
        ax.plot([x, x_right], [y, y_right], color='black')
        plot_binary_tree(tree[1], ax, x_right, y_right, x_spacing, y_spacing)
        ax.axis('off')



# Function to plot the relaxed Frechet set intersected with a given tree topology
def plot_relaxed_frechet_set_on_topology(N, sample_list, topology, relaxation, ax, col):
    """
    Input: N - Number of leaves in {4,5},
           sample_list - List of vectors v_1, ..., v_m each of dimension N*(N-1)/2 representing distance matrices,
           topology - List representing the topology of a tree,
           relaxation - Relaxation to apply if necessary,
           ax - Matplotlib axis to plot the Frechet set,
           col - Color for plotting.
    Output: Plots the relaxed Frechet set on the given topology which is in a (N-2)-dimensional space.
           For N=4: returns the corresponding 2D plot
           For N=5: returns the corresponding 3D plot
    """
    vertices = intersection_with_topology(N, sample_list, topology, relaxation)
    vertices = list(set(map(tuple,[vertex for vertex in vertices])))
    vertices = np.array(vertices)

    if N == 4 and len(vertices)>0:
        if len(vertices) <= 2:
            ax.scatter(vertices[:,0], vertices[:,1], color=col)
            ax.plot(vertices[:,0], vertices[:,1])
        else:
            hull = ConvexHull(vertices)
            set_1 = set([simplex[0] for simplex in hull.simplices])
            set_2 = set([simplex[1] for simplex in hull.simplices])
            list_hull = list(set_1.union(set_2))

            ax.scatter(vertices[list_hull,0], vertices[list_hull,1], color=col)
            for i,simplex in enumerate(hull.simplices) :
                if i==0: plt.plot(vertices[simplex, 0], vertices[simplex, 1], 'k-')
                else: plt.plot(vertices[simplex, 0], vertices[simplex, 1], 'k-')
        return True


    elif N == 5 and len(vertices)>0:
        if len(vertices) == 1: ax.scatter(vertices[0], vertices[1], vertices[2])
        if len(vertices) <= 2: ax.scatter(vertices[:,0], vertices[:,1], vertices[:,2])
        else:
            hull = ConvexHull(vertices)
            set_1 = set([simplex[0] for simplex in hull.simplices])
            set_2 = set([simplex[1] for simplex in hull.simplices])
            set_3 = set([simplex[2] for simplex in hull.simplices])
            list_hull = list(set_1.union(set_2).union(set_3))

            ax.scatter(vertices[list_hull,0], vertices[list_hull,1], vertices[list_hull,2], color='m', alpha = 0)
            for i,simplex in enumerate(hull.simplices):
                new_simplex = [vertices[simplex[i],:] for i in range(len(simplex))]
                ax.add_collection3d(Poly3DCollection([new_simplex], alpha=.25, linewidths=1, edgecolors=col, facecolors=col))
        return True

    return False


# Function to plot both relaxed and unrelaxed Frechet sets on a given topology
def plot_relaxed_and_unrelaxed_frechet_set_on_topology(N, sample_list, topology, relaxation, eps):
    """
    Input: N - Number of leaves in {4,5},
           sample_list - List of vectors v_1, ..., v_m each of dimension N*(N-1)/2 representing distance matrices,
           topology - List representing the topology of a tree,
           relaxation - Relaxation to apply if necessary,
           eps - Tolerance used for numerical reasons to compute the unrelaxed Frechet set,
    Output: Plots both relaxed and unrelaxed Frechet sets on the given topology,
            For N=4: returns the corresponding 2D plot
            For N=5: returns the corresponding 3D plot
    """
    fig = plt.figure()
    if N==4: ax = fig.add_subplot(121)
    elif N==5: ax = fig.add_subplot(121, projection='3d')
    else: return
    need_plot_1 = plot_relaxed_frechet_set_on_topology(N, sample_list, topology, relaxation, ax, 'r')
    need_plot_2 = plot_relaxed_frechet_set_on_topology(N, sample_list, topology, eps, ax, 'b')

    if need_plot_1 or need_plot_2:
        ax = fig.add_subplot(122)
        plot_binary_tree(topology, ax, x=0, y=4, x_spacing=1/2, y_spacing=1/2)
        plt.show()

    plt.close()
    return (need_plot_1, need_plot_2)


# Function to compare relaxed and unrelaxed Frechet sets on all represented topologies
def compare_relaxed_and_unrelaxed_frechet_sets(seed, Newick_file, N, nb_samples):
    """
    Input: seed - Seed for random number generation,
           Newick_file - Path to the Newick file containing the tree data,
           N - Number of leaves,
           nb_samples - Number of samples to consider.
    Output: Compares relaxed and unrelaxed Frechet sets on the sampled trees on the space of phylogenetic trees.
            This set is separated along certain topologies represented,
            the function plots the relaxed and unrelaxed Frechet mean set interesected all represented topologies.
    """
    random.seed(seed)
    np.random.seed(seed)
    decs = 2

    v_list = read_tree_data(Newick_file, 1, 1000, N, equidistant = True)
    indices = np.random.choice(np.arange(1000), size = 200, replace = True)
    sample_list = np.round(v_list[indices,:], decs)
    sample_list = sample_list[:nb_samples,:]

    obj = opt_frechet_value(sample_list)

    relaxation = (obj/nb_samples)*np.sqrt(nb_samples*math.log(math.log(nb_samples)))
    eps = 0.001
    for topology in all_possible_trees([x for x in range(N)]):
        (need_plot_relax, need_plot_no_relax) = plot_relaxed_and_unrelaxed_frechet_set_on_topology(N, sample_list, topology, relaxation, eps)


seed=1
Newick_file = 'N_NYh3n2_HA_20000_5_1993.txt'
compare_relaxed_and_unrelaxed_frechet_sets(seed, Newick_file, N=5, nb_samples=5)
