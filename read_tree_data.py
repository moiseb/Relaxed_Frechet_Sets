import numpy as np
from Bio import Phylo

#Newick_file = 'N_NYh3n2_HA_20000_5_1993.txt'

# Function to read tree data from a Newick file and compute distance matrices
def read_tree_data(Newick_file, start, end, N=5, equidistant = False):
    """
    Input: Newick_file - Path to the Newick file containing the tree data,
           start, end - Range of tree indices to consider,
           N - Number of leaves in the trees,
           equidistant - Flag indicating whether to force the phylogenetic trees to be equidistant or not.
    Output: Returns the distance matrices based on the tree data within the specified range.
    """
    dim = N*(N-1)//2

    # Extract the data
    trees = Phylo.parse('Data/'+Newick_file, 'newick')
    distmats_list = []
    tree_inds = [i for i in range(start, end+1)]
    ind = 0
    for tree in trees:
        if ind in tree_inds:
            d = np.zeros(dim)
            if equidistant:
                max_depth = 0
                for m in range(N):
                    depth_node = tree.distance('Sample%d' % (m+1))
                    if depth_node > max_depth:
                        max_depth = depth_node

                it = 0
                for m in range(N):
                    for n in range(m):
                        d[it] = tree.distance('Sample%d' % (m+1), 'Sample%d' % (n+1)) + (max_depth - tree.distance('Sample%d' % (m+1))) + (max_depth - tree.distance('Sample%d' % (n+1)))
                        it += 1

            else:
                d = np.zeros(dim)
                it = 0
                for m in range(N):
                    for n in range(m):
                        d[it] = tree.distance('Sample%d' % (m+1), 'Sample%d' % (n+1))
                        it += 1

            distmats_list.append(d)
        ind += 1

    ntrees, n = len(distmats_list), len(distmats_list[0])

    distmats = np.stack(distmats_list, axis=0)
    return(distmats - np.matmul(distmats[:,0].reshape(ntrees,1),np.ones(n).reshape(1,n)))
