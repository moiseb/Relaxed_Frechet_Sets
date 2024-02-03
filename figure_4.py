
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

plt.rcParams.update({
    "text.usetex": True,
    "font.family": "times"
})


def get_num_leaves(tree):
    if len(tree) <= 1:
        return len(tree)
    else:
        return get_num_leaves(tree[0]) + get_num_leaves(tree[1])

def get_rightmost_depth(tree):
    if len(tree) <= 1:
        return len(tree)-1
    else:
        return get_rightmost_depth(tree[1])+1

def get_leftmost_depth(tree):
    if len(tree) <= 1:
        return len(tree)-1
    else:
        return get_leftmost_depth(tree[0])+1


def plot_tree(ax, topology, trop_dist_mat, root_coords, height):
    root_x = root_coords[0]
    root_y = root_coords[1]
    left_topology = topology[0]
    right_topology = topology[1]

    # plot the left subtree
    if len(left_topology) == 1:
        ax.plot([root_x,root_x-1], [root_y, root_y], color='black')
        ax.plot([root_x-1,root_x-1], [root_y, root_y-height], color='black')
        ax.text(root_x-1-0.15, root_y-height-10, str(left_topology[0]+1), fontsize=16)
    else:
        l_rightmost_depth = get_rightmost_depth(left_topology)
        l_size = get_num_leaves(left_topology)
        l_height_decr = trop_dist_mat[0]
        l_root_x = root_x - 2*l_rightmost_depth
        l_root_y = root_y - l_height_decr
        l_trop_dist_mat = trop_dist_mat[1:l_size-1] # TODO: is this the right size?

        ax.plot([root_x,l_root_x], [root_y, root_y], color='black')
        ax.plot([l_root_x,l_root_x], [root_y, l_root_y], color='black')
        plot_tree(ax, left_topology, l_trop_dist_mat, (l_root_x,l_root_y), height-l_height_decr)

    # plot the right subtree
    if len(right_topology) == 1:
        ax.plot([root_x,root_x+1], [root_y, root_y], color='black')
        ax.plot([root_x+1,root_x+1], [root_y, root_y-height], color='black')
        ax.text(root_x+1-0.15, root_y-height-10, str(right_topology[0]+1), fontsize=16)
    else:
        # TODO: editing here
        r_leftmost_depth = get_leftmost_depth(right_topology)
        r_size = get_num_leaves(right_topology)
        r_height_decr = trop_dist_mat[len(trop_dist_mat)-r_size+1]
        r_root_x = root_x + 2*r_leftmost_depth
        r_root_y = root_y - r_height_decr
        r_trop_dist_mat = trop_dist_mat[len(trop_dist_mat)-r_size+1:]

        ax.plot([root_x,r_root_x], [root_y, root_y], color='black')
        ax.plot([r_root_x,r_root_x], [root_y, r_root_y], color='black')
        plot_tree(ax, right_topology, r_trop_dist_mat, (r_root_x,r_root_y), height-r_height_decr)
    ax.axis('off')

def plot_figure_4(seed, Newick_file, N, nb_samples, height=50):
    random.seed(seed)
    np.random.seed(seed)
    decs=2

    v_list = read_tree_data(Newick_file, 1, 1000, N, equidistant = True)
    indices = np.random.choice(np.arange(1000), size = 200, replace = True)
    sample_list = np.round(v_list[indices,:], decs)
    sample_list = sample_list[:nb_samples,:]

    obj = opt_frechet_value(sample_list)

    relaxation = (obj/nb_samples)*np.sqrt(nb_samples*math.log(math.log(nb_samples)))
    eps = 0.001
    topologies_relaxation, topologies_no_relax = [], []

    topos_unrelaxed = []
    vertex_samples_unrelaxed = []
    topos_relaxed = []
    vertex_samples_relaxed = []
    for topology in all_possible_trees([x for x in range(N)]):
        vertices_unrelaxed = intersection_with_topology(N, sample_list, topology, eps)
        vertices_unrelaxed = np.array(list(set(map(tuple,[vertex for vertex in vertices_unrelaxed]))))
        topology_present_unrelaxed = (len(vertices_unrelaxed)>= N-2)

        if topology_present_unrelaxed:
            topos_unrelaxed += [topology]
            vertices_unrelaxed = keep_extreme_points(vertices_unrelaxed)
            print('unrelaxed frechet mean set -- nb of vertices: ', len(vertices_unrelaxed), ', topology: ', topology)
            random_vertex_no_relax = vertices_unrelaxed[np.random.randint(len(vertices_unrelaxed)),:]
            vertex_samples_unrelaxed += [random_vertex_no_relax]
            print(random_vertex_no_relax)

        vertices_relaxed = intersection_with_topology(N, sample_list, topology, relaxation)
        vertices_relaxed = np.array(list(set(map(tuple,[vertex for vertex in vertices_relaxed]))))
        topology_present_relaxed = (len(vertices_relaxed)>= N-2)

        if topology_present_relaxed:
            topos_relaxed += [topology]
            vertices_relaxed = keep_extreme_points(vertices_relaxed)
            print('relaxed frechet mean set -- nb of vertices: ', len(vertices_relaxed), ', topology: ', topology)
            random_vertex_relax = vertices_relaxed[np.random.randint(len(vertices_relaxed)),:]
            vertex_samples_relaxed += [random_vertex_relax]
            print(random_vertex_relax)

    num_topos = len(topos_relaxed)
    fig, axs = plt.subplots(1,num_topos)
    for i in range(num_topos):
        topology = topos_unrelaxed[i]
        plot_tree(axs[i], topology, vertex_samples_unrelaxed[i], (0.,0.), height)
    plt.show()


    plt.clf()
    fig, ax = plt.subplots(figsize=(2.2, 4.8))
    plot_tree(ax, topos_unrelaxed[0], vertex_samples_unrelaxed[0], (0.,0.), height)
    plt.savefig('Figures/figure_4a.pdf')

    plt.clf()
    fig, axs = plt.subplots(1, num_topos)
    for i in range(num_topos):
        topology = topos_relaxed[i]
        plot_tree(axs[i], topology, vertex_samples_relaxed[i], (0.,0.), height)
    plt.savefig('Figures/figure_4b.pdf')


seed=1997
Newick_file = 'N_NYh3n2_HA_20000_5_1993.txt'
plot_figure_4(seed, Newick_file, N=5, nb_samples=12)
