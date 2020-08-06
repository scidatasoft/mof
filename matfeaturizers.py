import numpy as np
from pymatgen.analysis.local_env import VoronoiNN

from matminer.featurizers.base import BaseFeaturizer
from matminer.featurizers.utils.stats import PropertyStats


class GaussianSymmFuncModified(BaseFeaturizer):
    """
    Gaussian symmetry function features suggested by Behler et al.
    The function is based on pair distances and angles, to approximate the functional
    dependence of local energies, originally used in the fitting of
    machine-learning potentials.
    The symmetry functions can be divided to a set of radial functions
    (g2 function), and a set of angular functions (g4 function).
    The number of symmetry functions returned are based on parameters
    of etas_g2, etas_g4, zetas_g4 and gammas_g4.
    See the original papers for more details:
    “Atom-centered symmetry functions for constructing high-dimensional
    neural network potentials”, J Behler, J Chem Phys 134, 074106 (2011).
    The cutoff function is taken as the polynomial form (cosine_cutoff)
    to give a smoothed truncation.
    A Fortran and a different Python version can be found in the code
    Amp: Atomistic Machine-learning Package
    (https://bitbucket.org/andrewpeterson/amp).
    Args:
        etas_g2 (list of floats): etas used in radial functions.
                                  (default: [0.05, 4., 20., 80.])
        etas_g4 (list of floats): etas used in angular functions.
                                  (default: [0.005])
        zetas_g4 (list of floats): zetas used in angular functions.
                                   (default: [1., 4.])
        gammas_g4 (list of floats): gammas used in angular functions.
                                    (default: [+1., -1.])
        cutoff (float): cutoff distance. (default: 6.5)
    """

    def __init__(self, etas_g2=None, etas_g4=None, zetas_g4=None,
                 gammas_g4=None, cutoff=6.5):
        self.etas_g2 = etas_g2 if etas_g2 else [0.05, 4., 20., 80.]
        self.etas_g4 = etas_g4 if etas_g4 else [0.005]
        self.zetas_g4 = zetas_g4 if zetas_g4 else [1., 4.]
        self.gammas_g4 = gammas_g4 if gammas_g4 else [+1., -1.]
        self.cutoff = cutoff

    @staticmethod
    def cosine_cutoff(rs, cutoff):
        """
        Polynomial cutoff function to give a smoothed truncation of the Gaussian
        symmetry functions.
        Args:
            rs (ndarray): distances to elements
            cutoff (float): cutoff distance.
        Returns:
            (ndarray) cutoff function.
        """
        cutoff_fun = 0.5 * (np.cos(np.pi * rs / cutoff) + 1.)
        cutoff_fun[rs > cutoff] = 0
        return cutoff_fun

    @staticmethod
    def g2(eta, rs, cutoff):
        """
        Gaussian radial symmetry function of the center atom,
        given an eta parameter.
        Args:
            eta: radial function parameter.
            rs: distances from the central atom to each neighbor
            cutoff (float): cutoff distance.
        Returns:
            (float) Gaussian radial symmetry function.
        """
        ridge = (np.exp(-eta * (rs ** 2.) / (cutoff ** 2.)) *
                 GaussianSymmFuncModified.cosine_cutoff(rs, cutoff))
        return ridge.sum()

    @staticmethod
    def g4(etas, zetas, gammas, neigh_dist, neigh_coords, cutoff):
        """
        Gaussian angular symmetry function of the center atom,
        given a set of eta, zeta and gamma parameters.
        Args:
            eta ([float]): angular function parameters.
            zeta ([float]): angular function parameters.
            gamma ([float]): angular function parameters.
            neigh_coords (list of [floats]): coordinates of neighboring atoms, with respect
                to the central atom
            cutoff (float): cutoff parameter.
        Returns:
            (float) Gaussian angular symmetry function for all combinations of eta, zeta, gamma
        """

        output = np.zeros((len(etas)*len(zetas)*len(gammas),))

        # Loop over each neighbor j
        for j, neigh_j in enumerate(neigh_coords):

            # Compute the distance of each neighbor (k) to r
            r_ij = neigh_dist[j]
            d_jk = neigh_coords[(j+1):] - neigh_coords[j]
            r_jk = np.linalg.norm(d_jk, 2, axis=1)
            r_ik = neigh_dist[(j+1):]

            # Compute the cosine term
            cos_theta = np.dot(neigh_coords[(j + 1):], neigh_coords[j]) / r_ij / r_ik

            # Compute the cutoff function (independent of eta/zeta/gamma)
            cutoff_fun = GaussianSymmFuncModified.cosine_cutoff(np.array([r_ij]), cutoff) * \
                         GaussianSymmFuncModified.cosine_cutoff(r_ik, cutoff) * \
                         GaussianSymmFuncModified.cosine_cutoff(r_jk, cutoff)

            # Compute the g4 for each combination of eta/gamma/zeta
            ind = 0
            for eta in etas:
                # Compute the eta term
                eta_term = np.exp(-eta * (r_ij ** 2. + r_ik ** 2. + r_jk ** 2.) /
                                  (cutoff ** 2.)) * cutoff_fun
                for zeta in zetas:
                    for gamma in gammas:
                        term = (1. + gamma * cos_theta) ** zeta * eta_term
                        output[ind] += term.sum() * 2. ** (1. - zeta)
                        ind += 1
        return output

    def featurize(self, site, neighbors):
        """
        Get Gaussian symmetry function features of site with given index
        in input structure.
        Args:
            struct (Structure): Pymatgen Structure object.
            idx (int): index of target site in structure.
        Returns:
            (list of floats): Gaussian symmetry function features.
        """
        gaussian_funcs = []

        # Get coordinates of the neighbors, relative to the central atom
        neigh_coords = np.subtract([neigh[0].coords for neigh in neighbors], site.coords)

        # Get the distances for later use
        neigh_dists = np.array([neigh[1] for neigh in neighbors])

        # Compute all G2
        for eta_g2 in self.etas_g2:
            gaussian_funcs.append(self.g2(eta_g2, neigh_dists, self.cutoff))

        # Compute all G4s
        gaussian_funcs.extend(GaussianSymmFuncModified.g4(self.etas_g4, self.zetas_g4, self.gammas_g4,
                                                  neigh_dists, neigh_coords, self.cutoff))
        return gaussian_funcs

    def feature_labels(self):
        return ['G2_{}'.format(eta_g2) for eta_g2 in self.etas_g2] + \
               ['G4_{}_{}_{}'.format(eta_g4, zeta_g4, gamma_g4)
                for eta_g4 in self.etas_g4
                for zeta_g4 in self.zetas_g4
                for gamma_g4 in self.gammas_g4]
        
    def featurize_structure(self, struct):
        all_neighbors = struct.get_all_neighbors(self.cutoff)
        features = [self.featurize(site, neighbors) for site, neighbors in zip(struct, all_neighbors)]
        return features
    
    
class VoronoiFingerprintModified(BaseFeaturizer):
    """
    Voronoi tessellation-based features around target site.
    Calculate the following sets of features based on Voronoi tessellation
    analysis around the target site:
    Voronoi indices
        n_i denotes the number of i-edged facets, and i is in the range of 3-10.
        e.g.
        for bcc lattice, the Voronoi indices are [0,6,0,8,...];
        for fcc/hcp lattice, the Voronoi indices are [0,12,0,0,...];
        for icosahedra, the Voronoi indices are [0,0,12,0,...];
    i-fold symmetry indices
        computed as n_i/sum(n_i), and i is in the range of 3-10.
        reflect the strength of i-fold symmetry in local sites.
        e.g.
        for bcc lattice, the i-fold symmetry indices are [0,6/14,0,8/14,...]
            indicating both 4-fold and a stronger 6-fold symmetries are present;
        for fcc/hcp lattice, the i-fold symmetry factors are [0,1,0,0,...],
            indicating only 4-fold symmetry is present;
        for icosahedra, the Voronoi indices are [0,0,1,0,...],
            indicating only 5-fold symmetry is present;
    Weighted i-fold symmetry indices
        if use_weights = True
    Voronoi volume
        total volume of the Voronoi polyhedron around the target site
    Voronoi volume statistics of sub_polyhedra formed by each facet + center
        stats_vol = ['mean', 'std_dev', 'minimum', 'maximum']
    Voronoi area
        total area of the Voronoi polyhedron around the target site
    Voronoi area statistics of the facets
        stats_area = ['mean', 'std_dev', 'minimum', 'maximum']
    Voronoi nearest-neighboring distance statistics
        stats_dist = ['mean', 'std_dev', 'minimum', 'maximum']
    Args:
        cutoff (float): cutoff distance in determining the potential
                        neighbors for Voronoi tessellation analysis.
                        (default: 6.5)
        use_symm_weights(bool): whether to use weights to derive weighted
                                i-fold symmetry indices.
        symm_weights(str): weights to be used in weighted i-fold symmetry
                           indices.
                           Supported options: 'solid_angle', 'area', 'volume',
                           'face_dist'. (default: 'solid_angle')
        stats_vol (list of str): volume statistics types.
        stats_area (list of str): area statistics types.
        stats_dist (list of str): neighboring distance statistics types.
    """

    def __init__(self, cutoff=6.5,
                 use_symm_weights=False, symm_weights='solid_angle',
                 stats_vol=None, stats_area=None, stats_dist=None):
        self.cutoff = cutoff
        self.use_symm_weights = use_symm_weights
        self.symm_weights = symm_weights
        self.stats_vol = ['mean', 'std_dev', 'minimum', 'maximum'] \
            if stats_vol is None else copy.deepcopy(stats_vol)
        self.stats_area = ['mean', 'std_dev', 'minimum', 'maximum'] \
            if stats_area is None else copy.deepcopy(stats_area)
        self.stats_dist = ['mean', 'std_dev', 'minimum', 'maximum'] \
            if stats_dist is None else copy.deepcopy(stats_dist)

    def featurize(self, n_w):
        """
        Get Voronoi fingerprints of site with given index in input structure.
        Args:
            struct (Structure): Pymatgen Structure object.
            idx (int): index of target site in structure.
        Returns:
            (list of floats): Voronoi fingerprints.
                -Voronoi indices
                -i-fold symmetry indices
                -weighted i-fold symmetry indices (if use_symm_weights = True)
                -Voronoi volume
                -Voronoi volume statistics
                -Voronoi area
                -Voronoi area statistics
                -Voronoi dist statistics
        """
        # Prepare storage for the Voronoi indices
        voro_idx_list = np.zeros(8, int)
        voro_idx_weights = np.zeros(8)
        vol_list = []
        area_list = []
        dist_list = []

        # Get statistics
        for nn in n_w:
            if nn['poly_info']['n_verts'] <= 10:
                # If a facet has more than 10 edges, it's skipped here.
                voro_idx_list[nn['poly_info']['n_verts'] - 3] += 1
                vol_list.append(nn['poly_info']['volume'])
                area_list.append(nn['poly_info']['area'])
                dist_list.append(nn['poly_info']['face_dist'] * 2)
                if self.use_symm_weights:
                    voro_idx_weights[nn['poly_info']['n_verts'] - 3] += \
                        nn['poly_info'][self.symm_weights]

        symm_idx_list = voro_idx_list / sum(voro_idx_list)
        if self.use_symm_weights:
            symm_wt_list = voro_idx_weights / sum(voro_idx_weights)
            voro_fps = list(np.concatenate((voro_idx_list, symm_idx_list,
                                           symm_wt_list), axis=0))
        else:
            voro_fps = list(np.concatenate((voro_idx_list,
                                           symm_idx_list), axis=0))

        voro_fps.append(sum(vol_list))
        voro_fps.append(sum(area_list))
        voro_fps += [PropertyStats().calc_stat(vol_list, stat_vol)
                     for stat_vol in self.stats_vol]
        voro_fps += [PropertyStats().calc_stat(area_list, stat_area)
                     for stat_area in self.stats_area]
        voro_fps += [PropertyStats().calc_stat(dist_list, stat_dist)
                     for stat_dist in self.stats_dist]
        return voro_fps

    
    def featurize_structure(self, struct):
        n_w = VoronoiNN(cutoff=self.cutoff, allow_pathological=True).get_all_nn_info(struct)
        features = [self.featurize(n_w[i]) for i in range(len(struct.sites))]
        return features
    

    def feature_labels(self):
        labels = ['Voro_index_%d' % i for i in range(3, 11)]
        labels += ['Symmetry_index_%d' % i for i in range(3, 11)]
        if self.use_symm_weights:
            labels += ['Symmetry_weighted_index_%d' % i for i in range(3, 11)]
        labels.append('Voro_vol_sum')
        labels.append('Voro_area_sum')
        labels += ['Voro_vol_%s' % stat_vol for stat_vol in self.stats_vol]
        labels += ['Voro_area_%s' % stat_area for stat_area in self.stats_area]
        labels += ['Voro_dist_%s' % stat_dist for stat_dist in self.stats_dist]
        return labels