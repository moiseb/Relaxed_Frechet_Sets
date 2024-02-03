import random
import time
import matplotlib.pyplot as plt
from matplotlib.cm import get_cmap
from matplotlib.colors import Normalize

from frechet_set_computation import *
from frechet_set_estimation import *
from read_tree_data import *

plt.rcParams.update({
    "text.usetex": True,
    "font.family": "times"
})

###### Figure 3 presents projections of the Frechet means set in 2D, because these are too high-dimentional
###### In these high dimensions, computing the Frechet relaxed sets is quite computationally expensive, as implemented in frechet_set_computation.py
###### Instead of computing the exact Frechet relaxed set, here we discretize all 2D directions and compute where the relaxed Frechet set lies on these discretized directions, which speeds up the computation


#Compute the extreme points of the relaxed Frechet means set polyhedron projected onto a 2-dimensional plane
def projected_relaxed_frechet_mean(v_list,opt_objective,relaxation, proj, nb_discretization):
    """
    Args:
    - v_list: List of vectors v_1, ..., v_m to compute the relaxed Frechet mean set of
    - opt_objective: Optimal Frechet objective value for the given samples
    - relaxation: Relaxation parameter for computing the relaxed Frechet set
    - proj: Projection matrix for 2D projection
    - nb_discretization: Number of discretization points used to compute the projection

    What it does:
      For a discretization of all angles possibles with nb_discretization rays,
      for each discretized dircetion, compute the extreme point of the Frechet mean set along this direction,
      which can be efficiently computed as the solution of a linear program
    Returns:
    - values_output: List of objective values of the linear program at each discretized direction.
    - extreme_points: List of extreme points in the original n-dimensional space.
    - projected_points: List of extreme points in the projected 2-dimensional space.
    """
    #input: v_list should be formatted as a list of vectors v_1, ... , v_m each of dimension n.
    #output: returns the extreme points of the Frechet means set
    n = len(v_list[0])
    m = len(v_list)
    projection_dim = len(proj)

    constraints, right_hand_side = polyhedron_constraints(v_list)
    constraints = constraints[:,1:]

    obj = np.concatenate((np.array([0 for x in range(n-1)]),np.array([1 for x in range(m)])))

    constraints = np.vstack((constraints, obj))
    right_hand_side = np.vstack((right_hand_side,np.array([opt_objective + relaxation])))

    values_output = []
    extreme_points = []
    projected_points = []
    for i in range(nb_discretization):
        angle = 2*np.pi*i/nb_discretization
        u = np.dot(proj,np.array([np.cos(angle),np.sin(angle)]))

        x = cp.Variable(shape=(n+m-1,1), name="x")
        constraint_list = [cp.matmul(constraints, x) <= right_hand_side]
        objective = cp.Maximize(cp.matmul(u, x[:n-1]))

        problem = cp.Problem(objective, constraint_list)

        solution = problem.solve()
        optimal_x = [z[0] for z in x.value]
        values_output.append(solution)
        extreme_points.append(np.array(optimal_x[:n-1]))
        projected_points.append(np.dot(proj.T,np.array(optimal_x[:n-1])))

    return (values_output,extreme_points,projected_points)


##### In the previous procedure, the convex hull of the vertices in projected_points is in fact a lower bound on the projection of the exact relaxed Frechet set (contained by it)
##### The following procedure gives an upper bound (the convex hull of the vertices returned contains the projection of the exact relaxed Frechet set)
##### In practice, we observe that taking nb_discretization = 100 already gives indistinguishable lower bound (contained) and upper bound (containing) sets.


#Compute an upper bound of (a set containing) the relaxed Frechet means set projected on a 2D plane.
def upper_bound_projected_relaxed_frechet_mean(values,nb_discretization):
    """
    Args:
    - values: List of objective values for the linear program computed at each discretized direction by projected_relaxed_frechet_mean.
    - nb_discretization: Number of discretization points.

    Returns:
    - vertices: Vertices of the upper bound convex hull in the projected space,
                computed as the intersection of the hyperplanes {x: d^T x <= value}
                where d is any discretized direction, and value the corresponding value for the linear program computed in projected_relaxed_frechet_mean.
    """
    constraints = []
    for i in range(nb_discretization):
        angle = 2*np.pi*i/nb_discretization
        u = np.array([np.cos(angle),np.sin(angle)])

        constraints.append(u)
    constraints = np.array(constraints)
    values = np.array(values)

    vertices = compute_polygon_hull(constraints,values)
    return vertices


#Compute both unrelaxed and relaxed Frechet means sets projected onto various projections.
def projected_relaxed_and_unrelaxed_frechet_mean(v_list,sample_list,projs,nb_discretization = 100):
    """
    Args:
    - v_list: List of vectors to compute the Frechet mean sets of.
    - sample_list: List of sample sizes for each iteration.
    - projs: List of projection matrices for 2D projection.
    - nb_discretization: Number of discretizations for 2D directions.

    Returns:
    - vertices_unrelaxed: List of unrelaxed Frechet mean vertices for each projection.
    - vertices_relaxed: List of relaxed Frechet mean vertices for each projection.
    """
    nb_projs = len(projs)
    vertices_unrelaxed, vertices_relaxed = [[] for it in range(nb_projs)], [[] for it in range(nb_projs)]

    for i,nb_samples in enumerate(sample_list):
        sample_list = v_list[:nb_samples]
        obj = opt_frechet_value(sample_list)
        extreme_points = []
        ########### unrelaxed Frechet mean ################
        for it in range(nb_projs):
            start_time = time.time()
            values_unrelaxed,_, projected_points_unrelaxed = projected_relaxed_frechet_mean(sample_list,obj,0.0001,projs[it],nb_discretization)
            end_time = time.time()
            print('no relaxation       , nb of samples: ',nb_samples, ', elapsed time: ',end_time-start_time)

            vertices_unrelaxed[it].append(np.array(projected_points_unrelaxed))

            ########### naive relaxed Frechet mean ################
            prefactor = (obj/nb_samples)* math.log(math.log(nb_samples))
            relaxation = prefactor*(nb_samples**0.5)

            start_time = time.time()
            values_relaxed, extreme_points_relaxed, projected_points_relaxed = projected_relaxed_frechet_mean(sample_list,obj,relaxation,projs[it],nb_discretization)
            end_time = time.time()
            print('naive relaxation    , nb of samples: ',nb_samples,', elapsed time: ',end_time-start_time)

            vertices_relaxed[it].append(np.array(projected_points_relaxed))
            extreme_points.append(extreme_points_relaxed)

        ########### multi-step relaxation ################
        previous_variance = prefactor**2 / (2*math.log(math.log(nb_samples)))

        extreme_points = np.vstack(extreme_points)
        start_time = time.time()
        max_var = max_variance(postprocess_vertices(extreme_points), sample_list, previous_variance)
        end_time = time.time()
        print('compute_variance elapsed time: ',end_time-start_time)

        prefactor_new = np.sqrt(2*max_var*math.log(math.log(nb_samples)))
        while max_var!=False and prefactor_new<prefactor-0.1:
            prefactor = prefactor_new
            relaxation = prefactor*(nb_samples**0.5)
            extreme_points = []

            for it in range(nb_projs):
                start_time = time.time()
                values_relaxed, extreme_points_relaxed, projected_points_relaxed = projected_relaxed_frechet_mean(sample_list,obj,relaxation,projs[it],nb_discretization)
                end_time = time.time()

                print('multistep relaxation, nb of samples: ',nb_samples,', elapsed time: ',end_time-start_time)
                extreme_points.append(extreme_points_relaxed)
                vertices_relaxed[it][i] = np.array(projected_points_relaxed)

            extreme_points = np.vstack(extreme_points)
            start_time = time.time()
            max_var = max_variance(postprocess_vertices(extreme_points), sample_list, max_var)
            end_time = time.time()
            print('compute_variance elapsed time: ',end_time-start_time)
            prefactor_new = np.sqrt(2*max_var*math.log(math.log(nb_samples)))

    return (vertices_unrelaxed, vertices_relaxed)


#Generate a list of random projection matrices.
def random_projections(dim, nb_proj):
    """
    Args:
    - dim: Dimensionality of the original space.
    - nb_proj: Number of projection matrices to generate.

    Returns:
    - projs: List of random projection matrices.
    """
    projs = []
    for it in range(nb_proj):
        random_matrix = np.random.rand(dim,dim)
        q, _ = np.linalg.qr(random_matrix)
        proj = q[:,:2]
        projs.append(proj)

    return projs


#Plot Figure 3 with unrelaxed and relaxed Frechet means for different projections.
def plot_figure_3(seed=1,nb_discretization=100, nb_proj=3):
    """
    Args:
    - seed: Random seed for reproducibility.
    - nb_discretization: Number of discretization points for each direction.
    - nb_proj: Number of random projections to generate.
    """
    np.random.seed(seed)
    random.seed(seed)

    sample_list = [5,10,20,30,40,50]
    nb_samples_max, decs = 10000, 2

    Newick_file = 'NYh3n2_HA_20000_4_1995.txt'
    v_list = read_tree_data(Newick_file, 1, nb_samples_max,N=4)
    # sample the data
    indices = [random.randint(0, nb_samples_max-1) for _ in range(sample_list[-1])]
    v_list = np.round(v_list[indices,:], decs)

    #construct the random projections
    dim = 5
    projs = random_projections(dim, nb_proj)
    #compute the unrelaxed frechet means and the output of the relaxation procedure
    (vertices_unrelaxed, vertices_relaxed) = projected_relaxed_and_unrelaxed_frechet_mean(v_list,sample_list,projs,nb_discretization)

    # set up the plotting
    cmap = get_cmap('viridis')
    norm = Normalize(vmin=0, vmax=len(sample_list) - 1)
    fig, axs = plt.subplots(2, nb_proj)
    for it in range(nb_proj):
        axs[1, it].get_shared_x_axes().join(axs[0, it], axs[1, it])
        axs[1, it].get_shared_y_axes().join(axs[0, it], axs[1, it])

    fig.set_figheight(8)
    fig.set_figwidth(4*nb_proj)
    style = '-'

    #start plotting
    for it in range(nb_proj):
        for i in range(len(sample_list)):
            color_plot = cmap(norm(i))
            nb_samples = sample_list[i]

            for a,vertices_list in enumerate([vertices_unrelaxed[it][i], vertices_relaxed[it][i]]):
                if len(vertices_list)>2:
                    hull = ConvexHull(vertices_list)
                    lab = True
                    for simplex in hull.simplices:
                        if lab and a==1 and it==0: #only plot the labels once for each number of samples (given by i)
                            axs[a,it].plot(vertices_list[simplex, 0], vertices_list[simplex, 1], style, color = color_plot, label='%d' % nb_samples)
                        else:
                            axs[a,it].plot(vertices_list[simplex, 0], vertices_list[simplex, 1], style, color = color_plot)
                        lab = False
                elif len(vertices_list) == 2: axs[a,it].plot(vertices_list[:, 0], vertices_list[:, 1], style, color = color_plot)
                elif len(vertices_list)==1: axs[a,it].scatter(vertices_list[:, 0], vertices_list[:, 1], color = color_plot)

        axs[0,it].set_title('No Relaxation')
        axs[1,it].set_title('$\\textsc{AdaptRelaxFermatWeberSet}$')

    plt.figlegend(loc='center right', title='Number\nof Samples')
    plt.savefig('Figures/figure_3.pdf')



plot_figure_3(seed=1,nb_discretization=100, nb_proj=3)
