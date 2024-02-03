from scipy.spatial import ConvexHull
import math

from frechet_set_computation import *


# Function to keep only extreme points from a given set of vertices
def keep_extreme_points(vertices):
    """
    Input: vertices - List of vertices.
    Output: Returns the extreme points by computing the convex hull of the vertices.
    """
    hull = ConvexHull(vertices)
    set_hull = set()
    for simplex in hull.simplices:
        set_hull = set_hull.union(set(simplex))
    list_hull = list(set_hull)
    return vertices[list_hull,:]


# Function to compute the tropical distance between two points
def tropical_distance(u,v):
    """
    Input: u, v - Two-dimensional vectors.
    Output: Returns the tropical distance between the vectors.
    """
    return np.max([np.abs(u[0]-v[0]), np.abs(u[1]-v[1]), np.abs(u[0]-u[1]-(v[0]-v[1])) ])


# Function to compute the maximum variance among tropical distances for a set of vertices and a list of points
# The max variance is needed to compute the relaxation parameter for the Frechet mean set estimators
def max_variance(vertices,v_list,previous_variance=False):
    """
    Input: vertices - List of vertices,
           v_list - List of points,
           previous_variance - Optional argument representing the previous variance: when the current variance is larger than the previous variance the computation stops (no need for further relaxations).
    Output: Returns the maximum variance among tropical distances for a set of vertices and a list of points.
    """
    nb_vertices = len(vertices)
    max_var = 0

    if previous_variance == False:
        for i in range(nb_vertices):
            for j in range(i):
                temp = 1/len(v_list)*np.sum([(tropical_distance(vertices[i],vertex[1:]) - tropical_distance(vertices[j],vertex[1:]))**2 for vertex in v_list])
                max_var = np.max([temp, max_var])
    else:
        for i in range(nb_vertices):
            for j in range(i):
                temp = 1/len(v_list)*np.sum([(tropical_distance(vertices[i],vertex[1:]) - tropical_distance(vertices[j],vertex[1:]))**2 for vertex in v_list])
                max_var = np.max([temp, max_var])
                if max_var >= previous_variance: return False
    return max_var


# Function to compute the maximum variance for a set of duplicated points
def max_variance_duplicate_version(vertices,support, nb_duplicates):
    """
    Input: vertices - List of vertices,
           support - List of support points,
           nb_duplicates - Sequence of integers representing the number of duplicates for each point.
    Output: Returns the maximum variance among tropical distances for duplicated points.
    """
    nb_vertices = len(vertices)
    max_var = 0
    for i in range(nb_vertices):
        #print(vertices[i], v_list)
        for j in range(i):
            temp = 1/np.sum(nb_duplicates)*np.sum([nb_duplicates[l]*(tropical_distance(vertices[i],support[l][1:]) - tropical_distance(vertices[j],support[l][1:]))**2 for l in range(len(support))])
            max_var = np.max([temp, max_var])
    return max_var



# Function to compute the Fermat-Weber set estimator with adaptive relaxation
def adaptive_relaxation_fermat_weber_set(v_list, tolerance=0.1):
    """
    Input: v_list - List of points,
           tolerance - Tolerance value for convergence.
    Output: Returns the extreme points of the polyhedron from the estimated Fermat-Weber set of the points in v_list.
    """
    nb_samples = len(v_list)
    obj = opt_frechet_value(v_list)
    prefactor = (obj/n)* math.log(math.log(n))
    relaxation = prefactor*(nb_samples**0.5)

    vertices = relaxed_frechet_mean_set(v_list, relaxation)
    vertices = keep_extreme_points(np.array(vertices)[:,1:])

    max_var = max_variance(vertices, v_list)
    new_prefactor = np.sqrt(2*max_var*math.log(math.log(nb_samples)))


    while new_prefactor<prefactor-tolerance:
        prefactor = new_prefactor
        relaxation = prefactor*(nb_samples**0.5)

        vertices = relaxed_frechet_mean_set(v_list, relaxation)
        vertices = keep_extreme_points(np.array(vertices)[:,1:])

        max_var = max_variance(vertices, sample_list)
        new_prefactor = np.sqrt(2*max_var*math.log(math.log(nb_samples)))

    return vertices
