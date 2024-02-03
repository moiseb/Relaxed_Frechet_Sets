from scipy.spatial import ConvexHull
import itertools
from itertools import chain, combinations

from frechet_set_computation import *

# Function to generate the powerset of an iterable
def powerset(iterable):
    """
    Input: iterable - Any iterable object.
    Output: Returns the powerset of the input iterable.
    """
    #powerset([1,2,3]) --> () (1,) (2,) (3,) (1,2) (1,3) (2,3) without the complete set (1,2,3)
    s = list(iterable)
    return chain.from_iterable(combinations(s, r) for r in range(len(s)))


# Function to generate all possible topological binary trees with a given set of indices
def all_possible_trees(set_list):
    """
    Input: set_list - A list of indices.
    Output: Returns all topological binary trees with leaves having these indices.
        for set_list = [1,2,...,n] there are (2n-3)!! such topological binary trees
    """
    if len(set_list) == 1:
        yield [set_list[0]]
    for split in powerset(set_list[1:]):
        left_list = [set_list[0]] + list(split)
        right_list =  [x for x in set_list[1:] if x not in split]
        gen_left = all_possible_trees(left_list)
        gen_right = all_possible_trees(right_list)
        for left, right in itertools.product(gen_left, gen_right):
            yield [left, right]


# Function to yield nodes in a given topology
def nodes_in_topology(topology):
    """
    Input: topology - List representing a tree topology as formatted by all_possible_trees.
    Output: Yields the set of leaves in order of appearance in the topology.
    """
    if len(topology) == 1:
        yield topology[0]
    else:
        left_tree, right_tree = topology[0], topology[1]
        nodes = nodes_in_topology(left_tree)
        for left in nodes: yield left
        nodes = nodes_in_topology(right_tree)
        for right in nodes: yield right


# Function to convert a tree topology to a 3-dimensional tensor representing the map between
# the representation of the tree as lengths of inner edges
# and the representation as distance matrices between leaves
def from_tree_to_tropical_space(n,topology, distances, index):
    """
    Input: n - Number of leaves,
           topology - List representing a tree topology,
           distances - List representing distance matrices (for recursion purposes),
           index - Current index in the distance matrix list.
    Output: Returns a 3-dimensional tensor which represents the linear conversion
           from a vector with (n-2) degrees of liberty representing the lengths of inner edges of the tree
           to the corresponding distance matrix between leaves,
           for the given tree topology.
    """

    left_tree = topology[0]
    right_tree = topology[1]
    nb_left_nodes = len(left_tree)
    nb_right_nodes = len(right_tree)


    if nb_left_nodes > 1:
        for left_1, left_2 in itertools.product(nodes_in_topology(left_tree), nodes_in_topology(left_tree)):
            if left_1 > left_2: distances[index][left_1][left_2] = -1
            if left_1 < left_2: distances[index][left_2][left_1] = -1
        distances, index = from_tree_to_tropical_space(n,left_tree,distances, index + 1)


    if nb_right_nodes > 1:
        for right_1, right_2 in itertools.product(nodes_in_topology(right_tree), nodes_in_topology(right_tree)):
            if right_1 > right_2: distances[index][right_1][right_2] = -1
            if right_1 < right_2: distances[index][right_2][right_1] = -1
        distances, index = from_tree_to_tropical_space(n,right_tree,distances, index + 1)

    return (distances, index)


# Function to format a list of distance matrices into a single vector
def formatting_from_matrix_to_list(distances):
    """
    Input: distances - List of distance matrices.
    Output: Concatenates all distances matrices into a single vectors of length n(n-1)/2, then stacks the resulting vectors
            For n leaves, the result is a matrix of size (n-2)*(n(n-1)/2)
    """
    return np.stack([[item for sublist in distances[i] for item in sublist] for i in range(len(distances))])


# Function to get a matrix from a given tree topology, which represents the linear map between
# the representation of the tree as lengths of inner edges
# and the representation as distance matrices between leaves concatenated as vectors
def get_matrix_from_topology(n, topology):
    """
    Input: n - Number of leaves,
           topology - List representing a tree topology as formatted by all_possible_trees..
    Output: Returns a matrix A such that {Ax, x>=0} represents all valid trees for the given topology.
    """
    distances = [[[0 for j in range(i)] for i in range(n)] for x in range(n-2)]
    distances, _ = from_tree_to_tropical_space(n, topology, distances, 0)
    distances = formatting_from_matrix_to_list(distances)
    dim = len(distances[0])
    return distances - np.dot( distances[:,0].reshape(len(distances[:,0]),1),np.ones((1,dim)) )


# Function to find the intersection of a relaxed Frechet mean set polyhedron with a given tree topology
def intersection_with_topology(N, v_list, topology, relaxation = 0):
    """
    Input: N - Number of leaves,
           v_list - List of vectors v_1, ..., v_m of dimension N*(N-1)/2 representing tree distances,
           topology - Topology of a tree as formatted by all_possible_trees,
           relaxation - Relaxation to apply if necessary.
    Output: Returns the list of extreme points of the Frechet means set intersected with the given topology space.
    """
    dim = N*(N-1)//2
    m = len(v_list)

    constraints, right_hand_side = polyhedron_constraints(v_list)
    constraints = constraints[:,1:]

    opt_value = np.round(opt_frechet_value(v_list),2)

    value = opt_value + relaxation
    obj = np.concatenate((np.array([0 for x in range(dim-1)]),np.array([1 for x in range(m)])))
    constraints = np.vstack((constraints, obj))
    right_hand_side = np.vstack((right_hand_side,np.array([value])))

    #until this point, {x:constraints * (u,c) <= right_hand_side} is precisely the complete relaxed Frechet means set
    #now, we need to restrict to the topology
    subspace_matrix = get_matrix_from_topology(N, topology)[:,1:]
    constraints_subspace = np.dot(constraints[:,:dim-1], subspace_matrix.T)
    constraints_subspace = np.concatenate([constraints_subspace, constraints[:,dim-1:]], axis = 1)

    #additional_constraints saying the the lengths of arcs from trees must be non-negative
    additional_constraints = -np.eye(N-2,N-2+m)
    complete_constraints = np.vstack([additional_constraints, constraints_subspace])

    #updating the right hand side term
    complete_rhs = np.vstack([np.zeros((N-2,1)),right_hand_side])
    ineq = (complete_constraints, complete_rhs) # complete_constraints * (x,c) <= complete_rhs

    # Projection onto the first N-2 coordinates
    E = np.eye(N-2, N-2+m)
    f = np.zeros(N-2)
    proj = (E, f.T)  # proj(x) = E * x + f

    return project_polytope(proj, ineq, method='cdd')
