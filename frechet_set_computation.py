import numpy as np
import cvxpy as cp
from pypoman import plot_polygon, project_polytope, compute_polytope_vertices


#By convention for the tropical projective space -- the first coordinate is by default set to 0
#The next function deletes the first coordinate as a result (it equivalently assumes it is 0)


# Function to generate the polyhedron constraints of the Frechet mean set
def polyhedron_constraints(v_list):
    """
    Input: v_list - List of vectors v_1, ..., v_m each of dimension n.
    Output: Returns matrices A, b such that Ax <= b represents the polyhedron
            on which we minimize to get the optimal Frechet functional value.
    """
    n = len(v_list[0])
    m = len(v_list)


    matrix_u,matrix_c, input_contribution = [], [], []
    for i in range(m):
        z = [0 for x in range(m)]
        z[i] = 1
        for j in range(n):
            for k in range(j):
                y = [0 for x in range(n)]
                y[j] = 1
                y[k] = -1
                matrix_u.append(y)
                matrix_c.append(z)
                input_contribution.append([v_list[i][j] - v_list[i][k]])

    matrix_u = np.array(matrix_u)
    matrix_c = np.array(matrix_c)
    input_contribution = np.array(input_contribution)

    final_u = np.vstack((matrix_u,-matrix_u))
    final_c = np.vstack((-matrix_c,-matrix_c))

    constraint = np.concatenate((final_u,final_c), axis = 1)
    right_hand = np.vstack((input_contribution,-input_contribution))

    return(constraint, right_hand)


# Function to compute the minimum value of the Frechet functional
def opt_frechet_value(v_list):
    """
    Input: v_list - List of vectors v_1, ..., v_m each of dimension n.
    Output: Returns the minimum value of the Frechet functional.
    """
    n = len(v_list[0])

    u = cp.Variable((1,n))
    objvt = cp.Minimize(cp.sum(cp.max(v_list-u, axis=1)-cp.min(v_list-u, axis=1)))
    cnstr = []
    prob = cp.Problem(objvt, cnstr)
    prob.solve()
    return prob.value


# Function to find the minimum value of the Frechet functional of a set with duplicated points
def opt_frechet_value_duplicate_version(v_list, nb_duplicates):
    """
    Input: v_list - List of vectors v_1, ..., v_m each of dimension n,
           nb_duplicates - Sequence of integers representing the number of duplicates for each point.
    Output: Returns the minimum value of the Frechet functional for the set of points
            where v_i is duplicated nb_duplicates[i] times.
    """
    n = len(v_list[0])
    m = len(v_list)

    x = cp.Variable(shape=(n+m-1,1), name="x")
    constraints, right_hand_side = polyhedron_constraints(v_list)


    constraints = constraints[:,1:]
    constraints = [cp.matmul(constraints, x) <= right_hand_side]

    obj = np.concatenate((np.array([0 for x in range(n-1)]),np.array(nb_duplicates)))
    objective = cp.Minimize(cp.matmul(obj, x))

    problem = cp.Problem(objective, constraints)

    solution = problem.solve()
    return(solution)


# Function to post-process vertices by ensuring the first coordinate is 0
def postprocess_vertices(vertices):
    """
    Input: List of vertices written as an array.
    Output: Returns the same vertices making sure that the first coordinate is 0.
    """
    output = []
    for vertex in vertices: output.append(np.concatenate([np.zeros(1),vertex]))
    return output



####### strategy: projecting the polyhedron on (u,c) onto simply u to obtain the Frechet mean set


# Function to compute the extreme points of the relaxed Frechet mean set
def relaxed_frechet_mean_set(v_list, relaxation=0, ):
    """
    Input: v_list - List of vectors v_1, ..., v_m each of dimension n,
           relaxation - Relaxation parameter for the Frechet mean set.
    Output: Returns the extreme points of the relaxed Frechet means set.
    """
    n = len(v_list[0])
    m = len(v_list)

    constraints, right_hand_side = polyhedron_constraints(v_list)
    constraints = constraints[:,1:]

    value = opt_frechet_value(v_list) + relaxation

    obj = np.concatenate((np.array([0 for x in range(n-1)]),np.array([1 for x in range(m)])))

    constraints = np.vstack((constraints, obj))
    right_hand_side = np.vstack((right_hand_side,np.array([value])))
    ineq = (constraints, right_hand_side)  # constraints * x <= right_hand_side

    # Projection onto the first n coordinates
    E = np.eye(n-1, n+m-1)
    f = np.zeros(n-1)
    proj = (E, f.T)  # proj(x) = E * x + f

    vertices = project_polytope(proj, ineq, method='cdd')
    return postprocess_vertices(vertices)


# Function to compute the extreme points of the relaxed Frechet mean set for a set of duplicated points
def relaxed_frechet_mean_set_duplicate_version(v_list, nb_duplicates, relaxation=0):
    """
    Input: v_list - List of vectors v_1, ..., v_m each of dimension n,
           nb_duplicates - Sequence of integers representing the number of duplicates for each point,
           relaxation - Relaxation parameter for the Frechet mean set.
    Output: Returns the extreme points of the relaxed Frechet means set.
    """
    n = len(v_list[0])
    m = len(v_list)

    constraints, right_hand_side = polyhedron_constraints(v_list)
    constraints = constraints[:,1:]

    value = opt_frechet_value_duplicate_version(v_list, nb_duplicates) + relaxation
    obj = np.concatenate((np.array([0 for x in range(n-1)]),np.array(nb_duplicates)))

    constraints = np.vstack((constraints, obj))
    right_hand_side = np.vstack((right_hand_side,np.array([value])))
    ineq = (constraints, right_hand_side)  # constraints * x <= right_hand_side

    # Projection onto the first n coordinates
    E = np.eye(n-1, n+m-1)
    f = np.zeros(n-1)
    proj = (E, f.T)  # proj(x) = E * x + f

    vertices = project_polytope(proj, ineq, method='cdd')
    return postprocess_vertices(vertices)


# Function to clean vertices by rounding and removing duplicates
def clean_vertices(vertices, decs):
    """
    Input: vertices - List of vertices
            decs - integer for the number of decimals desired.
    Output: Cleans vertices by rounding and removing duplicates.
    """
    return list(set(map(tuple,[np.round(vertex, decs) for vertex in vertices])))
