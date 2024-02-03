import random
import time
import matplotlib.pyplot as plt
from matplotlib.cm import get_cmap
from matplotlib.colors import Normalize

from frechet_set_computation import *
from frechet_set_estimation import *

plt.rcParams.update({
    "text.usetex": True,
    "font.family": "times"
})



def plot_figure_2(seed):
    # parameters of the problem
    np.random.seed(seed)
    random.seed(seed)
    nb_samples_max, decs = 2000, 2
    sample_list = [50, 100, 200, 500,1000,2000]
    support = np.array([[0,0,0], [0,3,1], [0,2,5]])

    # sample the data
    indices = [random.randint(0, len(support)-1) for _ in range(nb_samples_max)]
    v_list = np.round([support[index,:] for index in indices], decs)

    # set up the plotting
    cmap = get_cmap('viridis')
    norm = Normalize(vmin=0, vmax=len(sample_list) - 1)
    fig, axs = plt.subplots(1, 3, sharex=True, sharey=True)
    fig.set_figheight(4)
    fig.set_figwidth(13)

    for p in range(3):
        # plot the population information
        axs[p].scatter(np.array([0., 3., 2.]), np.array([0., 1., 5.]), color='black', alpha=0.5)
        axs[p].fill(np.array([1., 2., 2., 1.]), np.array([1., 1., 2., 1.]), color='black', alpha=0.5)

    for i,nb_samples in enumerate(sample_list):
        nb_duplicates = [np.sum(np.array(indices[:nb_samples])==j) for j in range(len(support))]

        ########### optimal relaxation Frechet mean ################
        prefactor =  np.sqrt(2*2/3*math.log(math.log(nb_samples)))
        relaxation = prefactor*(nb_samples**0.5)
        vertices = relaxed_frechet_mean_set_duplicate_version(support, nb_duplicates, relaxation)
        vertices_opt = keep_extreme_points(np.array(vertices)[:,1:])

        ########### unrelaxed Frechet mean ################
        vertices = relaxed_frechet_mean_set_duplicate_version(support, nb_duplicates, 0.0001)
        vertices = clean_vertices(vertices, 2)
        vertices_unrelaxed = np.array(vertices)[:,1:]

        ########### naive relaxed Frechet mean ################
        obj = opt_frechet_value_duplicate_version(support, nb_duplicates)
        prefactor_naive = (obj/nb_samples)* math.log(math.log(nb_samples))
        relaxation = prefactor_naive*(nb_samples**0.5)
        vertices = relaxed_frechet_mean_set_duplicate_version(support, nb_duplicates, relaxation)
        vertices = np.array(vertices)[:,1:]

        ########### multi-step relaxed Frechet mean ################
        prefactor = prefactor_naive
        nb_points = len(vertices)
        max_var = max_variance_duplicate_version(vertices,support, nb_duplicates)
        prefactor_new = np.sqrt(2*max_var*math.log(math.log(nb_samples)))

        while prefactor_new<prefactor-0.1:
            prefactor = prefactor_new
            relaxation = prefactor*(nb_samples**0.5)

            vertices = relaxed_frechet_mean_set_duplicate_version(support, nb_duplicates, relaxation)
            vertices = np.array(vertices)[:,1:]
            nb_points = len(vertices)

            max_var = max_variance_duplicate_version(vertices,support, nb_duplicates)
            prefactor_new = np.sqrt(2*max_var*math.log(math.log(nb_samples)))

        vertices = keep_extreme_points(vertices)

        for p,vertices_list in enumerate([vertices_unrelaxed, vertices_opt, vertices]):

            color_here = cmap(norm(i))
            if len(vertices_list)>2:
                hull = ConvexHull(vertices_list)
                lab = True
                for simplex in hull.simplices:
                    if lab and p==1:
                        axs[p].plot(vertices_list[simplex, 0], vertices_list[simplex, 1], '-', color = color_here, label='%d' % nb_samples)
                    else:
                        axs[p].plot(vertices_list[simplex, 0], vertices_list[simplex, 1], '-', color = color_here)
                    lab = False
            else:
                axs[p].scatter(vertices_list[:, 0], vertices_list[:, 1], color=color_here)

    axs[0].set_title('No Relaxation')
    axs[1].set_title('Optimal Relaxation')
    axs[2].set_title('$\\textsc{AdaptRelaxFermatWeberSet}$')

    plt.figlegend(loc='center right', title='Number\nof Samples')

    plt.savefig('Figures/figure_2.pdf')



plot_figure_2(seed=1)
