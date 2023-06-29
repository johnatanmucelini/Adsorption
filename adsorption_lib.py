"""Tools employed in the absorption and comparison scripts."""

import itertools
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.spatial.distance import cdist
from scipy.cluster.hierarchy import single, fcluster
from scipy.spatial.distance import pdist
from scipy.cluster.vq import kmeans, vq
from sklearn.manifold import TSNE


NOTE_LESS_THAN_REQUESTED = """
+------------------------------------------------------------------------------+
| WARNING: The algorithm found {:>4d} ({:>3.0%}) structure, but {:>4d} were requested.  |
| WARNING: Thus, the final clustering is impracticable and will be skipped.    |
| To increase the number of structures, you can:                               |
| - increase surf_ks or n_rot -> increasing sampling quality                   |
| - decrease ovlp_threshold   -> increasing the size of the phase-space        |
| - decrease sim_threshold    -> similarity filter will accept more structures |
+------------------------------------------------------------------------------+
"""

NOTE_EQUAL_TO_REQUESTED = """
+------------------------------------------------------------------------------+
| WARNING: The algorithm found exactly {:>4d} structure, as requested.          |
| WARNING: Thus, the final clustering is impracticable and will be skipped.    |
| If you wish to increase the number of structures, you can:                   |
| - increase surf_ks or n_rot -> increasing sampling quality                   |
| - decrease ovlp_threshold   -> increasing the size of the phase-space        |
| - decrease sim_threshold    -> similarity filter will accept more structures |
+------------------------------------------------------------------------------+
"""

NOTE_SINGLE_ATOM_NKM_NOTE_ONE = """+------------------------------------------------------------------------------+
| WARNING: The following molecule is a single atom, and thus, the number of    |
|          chemical envirowments will be redefined to one: {:<20s}|
+------------------------------------------------------------------------------+"""


class Mol:
    """The mol object carry all the information of the molecule."""

    def __init__(self, path=None, positions=None, cheme=None, verbose=True):
        """Initiate from a path or from the postions and chemical elements."""

        # if provided, it initiate from the positions + chemical elements
        self.path = path
        self.positions = positions
        self.cheme = cheme

        # other variables
        self.radii = None
        self.surf_dots = None
        self.surf_dots_atom = None
        self.surf_atoms_index = None
        self.surf_cheme = None
        self.surf_atoms_positions = None
        self.surf_atoms_raddii = None
        self.surf_dots_km_index = None
        self.surf_dots_km_rep_kmindex = None
        self.surf_dots_km_rep_idx = None
        self.surf_dots_km_rep = None
        self.surf_dots_features = None
        self.ipositions = None
        self.isurf_dots = None
        self.isurf_dots_km_rep = None

        # if niether of the positions were privided exit
        if self.path is None and (self.positions is None and self.cheme is
                                  None):
            err = 'Mol_path or positions + cheme must be provided to create ' \
                  'a Mol.'
            raise AttributeError(err)

        # initiate from the path
        if (self.positions is None and self.cheme is None) and self.path is not None:
            if verbose:
                print('Reading mol from {}'.format(self.path))
            with open(self.path) as mol_file:
                lines_as_list = mol_file.readlines()
            self.n = int(lines_as_list[0].split()[0])
            _positions = np.array([line.split()[0:]
                                   for line in lines_as_list[2:self.n+2]])
            _cheme = np.array([line.split()[0]
                               for line in lines_as_list[2:self.n+2]])
            self.cheme = np.array(_cheme, dtype=str)
            self.positions = np.array(_positions[:, 1:4], dtype=float)

    def get_radii(self, atoms_radii_preferences=[2]):
        """Get the raddii of the present atoms based in a list of its
        van-der-walls radius"""

        msg_ref = "    Atom {}, vdw radii {}, ref {}."
        msg_not_ref = "    Atom {}, vdw radii {}, missing reference!\n" \
            "        WARNING: missing reference!\n" \
            "        To add vdw radius search for VDW RADII in adsorption.py."

        cheme_radii_dict = {}
        for u_cheme in np.unique(self.cheme):
            found = False
            for obj in atoms_radii_preferences:
                if not found:
                    if isinstance(obj, dict):
                        if u_cheme in obj.keys():
                            cheme_radii_dict[u_cheme] = obj[u_cheme]
                            found = True
                            if 'ref' in obj.keys():
                                print(msg_ref.format(
                                    u_cheme, obj[u_cheme], obj['ref']))
                            else:
                                print(msg_not_ref.format(
                                    u_cheme, obj[u_cheme]))
                    if isinstance(obj, int):
                        cheme_radii_dict[u_cheme] = obj[u_cheme]
                        found = True
                        print(msg_not_ref.format(
                            u_cheme, obj[u_cheme], obj.ref))
        self.radii = np.array([cheme_radii_dict[ele] for ele in self.cheme])

    def build_surface(self, atoms_surface_density=10, name=''):
        """This algorithm finds a surface of dots around the surface of the
        molecule, considering the atoms as ridge spheres of given radii. It
        also measures the exposed area per atom and the total area of the
        surface."""

        print("Mapping surface of {}".format(name))

        # building the surface
        dots, dots_atom, area = build_surfac_func(
            self.cheme, self.positions, self.radii,
            atoms_surface_density=atoms_surface_density)

        # handling variables
        self.surf_dots = dots
        self.surf_dots_atom = dots_atom
        self.surf_atoms_index = np.unique(dots_atom)
        self.surf_cheme = self.cheme[self.surf_atoms_index]
        self.surf_atoms_positions = self.positions[self.surf_atoms_index]
        self.surf_atoms_raddii = self.radii[self.surf_atoms_index]

        # area:
        print('    N surface points       {:7d}'.format(len(dots)))
        print('    Surface area            {:10.3f} AA'.format(area))
        print(
            '    Points density          {:10.3f} AA^-1'.format(len(dots)/area))

    def featurization_surface_dots(self):
        """Calculate the features vectors for a molecule as follows:
        1) calculate the distance between references and positions,
        2) calculate a slow-decay function of these distance values,
        3) sort these results, for each reference and chemical element.
        The slow-decay function is exp(-dist**(2/3)).
        If no referece were privided it employed the molecules geometric
        centroid.
        """

        print("Featurization of the surface dots.")
        self.surf_dots_features = get_feature(self, self.surf_dots)

    def clustering_surface_dots(self, name, n_cluster, n_repeat=5):
        """Calculate the cluster of the surface dots: indexes and centroid
        nearest"""

        print("Clustering of the surface dots.")
        data = self.surf_dots_features
        top_score = 1e20

        # kmeans
        for seed in range(n_repeat):
            # clustering, returning centroids and a score
            centroids, score = kmeans(data, n_cluster, seed=seed)
            if score < top_score:
                # keeping only the best model
                top_score = score
                top_centroids = centroids
        # getting the representative dots: the closest dots to each centroids
        idx, _ = vq(data, top_centroids)
        dists = cdist(top_centroids, data)
        centroids_nearst_idx = np.argmin(dists, axis=1)

        # handling variables
        self.surf_dots_km_index = idx
        self.surf_dots_km_rep_kmindex = idx[centroids_nearst_idx]
        self.surf_dots_km_rep_idx = centroids_nearst_idx
        self.surf_dots_km_rep = self.surf_dots[centroids_nearst_idx]

        # ploting the results
        plot_kmeans_tsne(name + '_km_tsne', data, idx, centroids_nearst_idx)

    def translate_by(self, vector, image=False):
        """Translate structure positions by a vector, real or image data."""
        if image:
            self.ipositions = self.positions + vector
            self.isurf_dots = self.surf_dots + vector
            self.isurf_dots_km_rep = self.surf_dots_km_rep + vector
        else:
            self.positions = self.positions + vector

    def centralize(self):
        """Centralize in itself."""
        self.positions = self.positions - self.positions.mean(axis=0)

    def rotate(self, rot_matrix, image=False):
        """applay a rotation over the structure, real + surf image data."""
        if image:
            self.ipositions = np.dot(self.ipositions, rot_matrix)
            self.isurf_dots = np.dot(self.isurf_dots, rot_matrix)
            self.isurf_dots_km_rep = np.dot(self.isurf_dots_km_rep, rot_matrix)
        else:
            self.positions = np.dot(self.positions, rot_matrix)

    def to_xyz(self, file_name, surf_dots=False, surf_dots_color_by=None,
               special_surf_dots=None, verbose=True):
        """Write mol object to a xyz files with and without surf dots."""

        if verbose:
            print('Writing mol to: {}'.format(file_name))

        # creating the dummy atoms to represent surf and representative dots
        letras = ['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k', 'l',
                  'm', 'n', 'o', 'p', 'q', 'r', 's', 't', 'u', 'v', 'w', 'x',
                  'y', 'z']
        color_base_surf = []
        color_base_rep = []
        for c1, c2 in zip(['@', '&', '$', '?', '%'], ['#', '!', '+', '-', '*']):
            for le in letras:
                color_base_surf.append(c1 + le)
                color_base_rep.append(c2 + le)
                if False:
                    # printing data for vesta atoms colors configuration file
                    color = np.random.rand(3)
                    print('  1  {}  0.25  0.25  0.25     {:5.3f}    {:5.3f}    {:5.3f}'.format(
                        c1+le, *color))
                    print('  1  {}  0.40  0.40  0.40     {:5.3f}    {:5.3f}    {:5.3f}'.format(
                        c2+le, *color))
        color_base_surf = np.array(color_base_surf)
        color_base_rep = np.array(color_base_rep)
        surf_index_to_cheme_dict = {}
        rep_index_to_cheme_dict = {}
        for ith in range(len(color_base_surf)):
            surf_index_to_cheme_dict[ith] = color_base_surf[ith]
            rep_index_to_cheme_dict[ith] = color_base_rep[ith]

        # adding xyz information
        atoms_positions = self.positions
        atoms_cheme = self.cheme
        if len(atoms_positions) != len(atoms_cheme):
            err = 'Tamanho dos vetores positions cheme é diferente b: ' \
                '{} {}'.format(len(atoms_positions), len(atoms_cheme))
            raise AttributeError(err)

        # adding surf dots
        if surf_dots:
            atoms_positions = np.append(
                atoms_positions, self.surf_dots, axis=0)
            if surf_dots_color_by == 'kmeans':
                # surf dots colored by kmeans cluster
                new_chemes = []
                for dot_km_index in self.surf_dots_km_index:
                    new_chemes.append(surf_index_to_cheme_dict[dot_km_index])
            elif surf_dots_color_by == 'atoms':
                # surf dots colored by origin atom
                new_chemes = []
                for origin_atom_index in self.surf_dots_atom:
                    new_chemes.append(
                        surf_index_to_cheme_dict[origin_atom_index])
            else:
                new_chemes = np.array(['@A']*len(self.surf_dots))
            atoms_cheme = np.append(atoms_cheme, np.array(new_chemes))

        if len(atoms_positions) != len(atoms_cheme):
            print('Tamanho dos vetores positions cheme é diferente s: {} {}'.format(
                len(atoms_positions), len(atoms_cheme)))

        # adding km representative dots result
        if special_surf_dots:
            # print('Special surf dots:', len(self.surf_dots_km_rep))
            atoms_positions = np.append(atoms_positions, self.surf_dots_km_rep,
                                        axis=0)
            new_chemes = []
            n_clusters = max(self.surf_dots_km_rep_kmindex) + 1
            # print('Max km index:', max(self.surf_dots_km_rep_kmindex)
            #       + 1, self.surf_dots_km_rep_kmindex)
            for i in range(n_clusters):
                if special_surf_dots == 'kmeans':
                    new_chemes.append(rep_index_to_cheme_dict[i])
                else:
                    new_chemes.append(['XX'])
            # print('new_chemes', len(new_chemes))
            atoms_cheme = np.append(atoms_cheme, new_chemes)

        # writting
        if len(atoms_positions) != len(atoms_cheme):
            err = 'Tamanho dos vetores positions cheme é diferente: '\
                '{} {}'.format(len(atoms_positions), len(atoms_cheme))
            raise AttributeError(err)
        with open(file_name, mode='w') as xyz_file:
            size = len(atoms_positions)
            xyz_file.write(str(size) + '\n\n')
            for ith, (element, position) in enumerate(zip(atoms_cheme, atoms_positions)):
                if ith < size - 1:
                    xyz_file.write('{} {} {} {}\n'.format(element, *position))
                elif ith == size - 1:
                    xyz_file.write('{} {} {} {}'.format(element, *position))

    def image_to_real_with_surf(self):
        """Copy the image vars of ipositions, isurf_dots and isurf_dots_km_rep
        to they analog real vars positions, surf_dots and surf_dots_km_rep."""
        if 'ipositions' in dir(self):
            self.positions = self.ipositions * 1.
            self.surf_dots = self.isurf_dots * 1.
            self.surf_dots_km_rep = self.isurf_dots_km_rep * 1.


def build_s2_grid(radius, n_sample_init, coords_system='cart'):
    """Generate an "almost" regular grid on the surface of sphere of given
    radius.
    More deatiail of the implementation in the article "How to generate
    equidistributed points on the surface of a sphere" by Markus Deserno:
    https://www.cmu.edu/biolphys/deserno/pdf/sphere_equi.pdf"""

    cart_coordinates = []
    spherical_coordinates = []
    r_val = 1
    a_val = 4. * np.pi * r_val**2 / n_sample_init
    d_val = np.sqrt(a_val)
    m_theta = int(round(np.pi/d_val))
    dtheta = np.pi / m_theta
    dphi = a_val / dtheta
    for mth in range(0, m_theta):
        theta = np.pi * (mth + 0.5) / m_theta
        m_phi = int(round(2 * np.pi * np.sin(theta) / dphi))
        for nth in range(0, m_phi):
            phi = 2 * np.pi * nth / m_phi
            y_val = radius * np.sin(theta) * np.cos(phi)
            x_val = radius * np.sin(theta) * np.sin(phi)
            z_val = radius * np.cos(theta)
            cart_coordinates.append([x_val, y_val, z_val])
            spherical_coordinates.append([theta, phi])
    cart_coordinates = np.array(cart_coordinates)
    spherical_coordinates = np.array(spherical_coordinates)

    if coords_system == 'cart':
        result = cart_coordinates
    elif coords_system == 'spher':
        result = spherical_coordinates

    return result


def build_s1_grid(radisus, n_samples, coords_system='circ'):
    """Generate a grid in a circle in circular or cartezian coordinates."""
    dphi = 2 * np.pi / n_samples
    circular_coords = []
    cartezian_coords = []
    for ith in range(n_samples):
        phi = dphi*ith
        circular_coords.append(phi)
        x_val = radisus * np.cos(phi)
        y_val = radisus * np.sin(phi)
        cartezian_coords.append([x_val, y_val])
    circular_coords = np.array(circular_coords)
    cartezian_coords = np.array(cartezian_coords)
    if coords_system == 'circ':
        result = circular_coords
    elif coords_system == 'cart':
        result = cartezian_coords
    return result


def build_s3_from_s1s2(s1_coords, s2_coords):
    """Build the quarternion rotation matrix from s1 and s2 grids.
    Considering, s1_coords (psi) in [0, 2pi) and s2_coords (theta, phi) in
    [0, pi) and [0, 2pi)."""

    quarternions = []
    for psi in s1_coords:
        for theta, phi in s2_coords:
            qr = np.cos(theta/2) * np.cos(psi/2)
            qi = np.cos(theta/2) * np.sin(psi/2)
            qj = np.sin(theta/2) * np.cos(phi + psi/2)
            qk = np.sin(theta/2) * np.sin(phi + psi/2)
            quarternions.append([qr, qi, qj, qk])
    quarternions = np.array(quarternions)

    rot_matrix = []
    for qr, qi, qj, qk in quarternions:
        # s = 1/sum(quarternions[nth]**2)
        # print(s)
        rot_matrix.append(np.array([
            [1-2*(qj**2+qk**2),   2*(qi*qj-qk*qr),   2*(qi*qk+qj*qr)],
            [2*(qi*qj+qk*qr),   1-2*(qi**2+qk**2),   2*(qj*qk-qi*qr)],
            [2*(qi*qk-qj*qr),     2*(qj*qk+qi*qr), 1-2*(qi**2+qj**2)]
        ]))
    return rot_matrix


def build_surfac_func(cheme, positions, radii, atoms_surface_density=10):
    """This algorithm finds a surface of dots around the surface of the
    molecule, considering the atoms as ridge spheres of given radii. It also
    measures the exposed area per atom and the total area of the surface."""

    # calculation dots positions considering each atom indivualy
    trial_n_dots_per_atom = atoms_surface_density * 4 * np.pi * radii**2
    n_atoms = len(positions)
    dots_atom = np.empty(0, int)
    n_dots_per_atom = []
    dots = np.empty((0, 3), float)
    cheme_n_dots_per_atom_dict = {}
    for ith, ele in enumerate(cheme):
        dots_default = build_s2_grid(1, trial_n_dots_per_atom[ith])
        dots = np.append(
            dots, positions[ith] + dots_default * (radii[ith] + 1e-4), axis=0)
        n_dots_per_atom.append(len(dots_default))
        dots_atom = np.append(dots_atom, np.array([ith] * len(dots_default)))
        if ele not in cheme_n_dots_per_atom_dict.keys():
            cheme_n_dots_per_atom_dict[ele] = len(dots_default)
    n_dots_per_atom = np.array(n_dots_per_atom)
    n_dots = sum(n_dots_per_atom)
    for key in cheme_n_dots_per_atom_dict.keys():
        print('    Initial N of dots per {:<2s} {:>5d}'.format(
            key, cheme_n_dots_per_atom_dict[key]))

    # remove dots inside other atoms sphere
    dots_atoms_distance = cdist(positions, dots)
    radii_by_dot = np.array([radii]*n_dots).reshape(n_dots, n_atoms).T
    overleped = np.any(dots_atoms_distance < radii_by_dot, axis=0)
    non_overleped = np.invert(overleped)
    dots = dots[non_overleped]
    dots_atom = dots_atom[non_overleped]

    # Search if there are more than one surfaces in dots (internal surfaces)
    dots_large_atom = dots_default*max(radii)
    dotdot_distances = cdist(dots_large_atom, dots_large_atom)
    dotdot_distances += np.eye(len(dotdot_distances))*10
    min_dotdot_distance = np.min(dotdot_distances)
    eps = 2.1 * min_dotdot_distance
    condensed_distances_matrix = pdist(dots)
    zzz = single(condensed_distances_matrix)
    labels_per_dot = fcluster(zzz, eps, criterion='distance')
    labels, quanity = np.unique(labels_per_dot, return_counts=True)
    if len(labels) > 1:
        print('Warning: {} surfaces were found (sizes: {})'.format(
              labels, str(quanity).replace('[', '').replace(']', '')))
        print('         the biggest surface were selected as the external one!'
              )

    # taking the larger surface
    larges_cluster_dots = labels_per_dot == labels[np.argmax(quanity)]
    dots = dots[larges_cluster_dots]
    dots_atom = dots_atom[larges_cluster_dots]

    # final measure
    area_per_atom_dot = (4 * np.pi*radii ** 2) / n_dots_per_atom
    area = sum(area_per_atom_dot[dots_atom])

    return dots, dots_atom, area


def get_distance(features_1, features_2):
    """Distance is norm"""
    return np.linalg.norm(features_1 - features_2)


def get_feature(mol, reference=None):
    """Calculate the features vectors for a molecule as follows:
    1) calculate the distance between references and positions positions
    2) calculate a slow-decay function of these distance values,
    3) sort these results, for each reference and chemical element.
    The slow-decay function is exp(-dist**(2/3)).
    If no referece were privided it employed the molecules geometric centroid.
    """

    # handling features
    if reference is None:
        reference_positions = mol.positions.mean(1)
    else:
        reference_positions = reference

    ucheme = sorted(np.unique(mol.cheme))

    # calculating features
    features = np.empty((len(reference_positions), 0))
    for ucheme_e in ucheme:
        ucheme_e_index = mol.cheme == ucheme_e
        ucheme_e_features = -np.sort(
            -np.exp(-cdist(reference_positions,
                           mol.positions[ucheme_e_index])**(2/3)), axis=1)
        features = np.append(features, ucheme_e_features, axis=1)

    if reference is None:
        result = features[0]
    else:
        result = features

    return result


def plot_kmeans_tsne(name, data, idx, rep_idx):
    """It save a plot of the kmenas result after a t-SNE transformation"""

    palette = itertools.cycle(sns.color_palette())
    sns.set()

    fig, ax = plt.subplots(1, 1, figsize=(8, 6), dpi=160)

    print('    t-SNE feature reduction')
    features_2d = TSNE(n_components=2, learning_rate='auto',
                       init='random', random_state=42, n_iter=3000,
                       n_iter_without_progress=500).fit_transform(data)

    for all_cluster_index, rep_index in zip(range(max(idx)+1), rep_idx):
        data_indexes = idx == all_cluster_index
        data_indexes[rep_index] = False
        color = next(palette)
        x_rep = features_2d[rep_index, 0]
        y_rep = features_2d[rep_index, 1]

        if np.any(data_indexes):
            x_vals = features_2d[data_indexes, 0]
            y_vals = features_2d[data_indexes, 1]
            ax.scatter(x=x_vals, y=y_vals, alpha=0.4, marker='o', s=35,
                       color=color, zorder=2)

            for single_data_idx in np.argwhere(data_indexes):
                xi = features_2d[single_data_idx, 0][0]
                yi = features_2d[single_data_idx, 1][0]
                xs = np.array([xi, x_rep], float)
                ys = np.array([yi, y_rep], float)
                ax.plot(xs, ys, alpha=0.15, color=color, zorder=1)

        ax.scatter(x=x_rep, y=y_rep, alpha=1., marker='+', s=45, color=color,
                   zorder=4)
        ax.set_title('t-SNE vizualization of Kmeans')
        ax.set_xlabel('t-SNE coord 1')
        ax.set_ylabel('t-SNE coord 2')
        fig.set_tight_layout(True)

    print('    Writing results to: {}'.format(name + '.png'))
    fig.savefig(name + '.png')


def add_mols(mol_a, mol_b, image=False, add_surf_info=False):
    """Combine two molecules, based in their real or image postion, optionally,
    also adding their surface info"""

    # combaning basic information
    if image:
        mol_f_positions = np.append(mol_a.ipositions, mol_b.ipositions, axis=0)
        mol_f_cheme = np.append(mol_a.surf_cheme, mol_b.surf_cheme)
    else:
        mol_f_positions = np.append(mol_a.positions, mol_b.positions, axis=0)
        mol_f_cheme = np.append(mol_a.cheme, mol_b.cheme)

    # defining the Mol object
    mol_f = Mol(positions=mol_f_positions, cheme=mol_f_cheme)

    # combining surface information
    if add_surf_info:
        #print('surf_dots_km_index:', mol_a.surf_dots_km_index)
        #print('surf_dots_atom:', mol_a.surf_dots_atom)
        #print('surf_dots_km_rep:', mol_a.surf_dots_km_rep)
        #print('surf_dots_km_rep_kmindex:', mol_a.surf_dots_km_rep_kmindex)
        # surf_dots
        mol_f.surf_dots = np.append(mol_a.surf_dots, mol_b.surf_dots, axis=0)
        # surf_dots_km_index
        mol_f.surf_dots_km_index = np.append(
            mol_a.surf_dots_km_index, mol_b.surf_dots_km_index
            + max(mol_a.surf_dots_km_index) + 1, axis=0)
        # surf_dots_atom
        mol_f.surf_dots_atom = np.append(
            mol_a.surf_dots_atom, mol_b.surf_dots_atom + mol_a.n, axis=0)
        # surf_dots_km_rep
        mol_f.surf_dots_km_rep = np.append(
            mol_a.surf_dots_km_rep, mol_b.surf_dots_km_rep, axis=0)
        # surf_dots_km_rep_kmindex
        mol_f.surf_dots_km_rep_kmindex = np.append(
            mol_a.surf_dots_km_rep_kmindex, mol_b.surf_dots_km_rep_kmindex
            + max(mol_a.surf_dots_km_rep_kmindex) + 1, axis=0)
        # combining image surface position
        if image:
            mol_f.ipositions = np.append(
                mol_a.ipositions, mol_b.ipositions, axis=0)
            mol_f.isurf_dots = np.append(
                mol_a.isurf_dots, mol_b.isurf_dots, axis=0)
            mol_f.isurf_dots_km_rep = np.append(
                mol_a.isurf_dots_km_rep, mol_b.isurf_dots_km_rep, axis=0)
    mol_f.n = len(mol_f.positions)
    return mol_f


def overlap(mol_a, mol_b, ovlp_threshold=0.85, image=False):
    """Check if two atomic structures overlap, real or just surface atoms
    images."""
    if image:
        distances = cdist(mol_a.ipositions, mol_b.ipositions)
        radii_sum = mol_a.radii.reshape(-1, 1) + mol_b.radii.reshape(1, -1)
        if np.any((distances/radii_sum) < ovlp_threshold):
            result = True
        else:
            result = False
    return result


def status(c_all, n_config, c_repeated, c_overlapped, c_accepted, refused_ds):
    """Print the partial results of the algorithm"""

    print('-'*80)
    print("N structures analyzed   {:10d}  {:>8.2%}".format(
        c_all, c_all/n_config))
    print("N structures overlapped {:10d}  {:>8.2%}".format(
        c_overlapped, c_overlapped/c_all))
    print("N structures repeated   {:10d}  {:>8.2%}".format(
        c_repeated, c_repeated/c_all))
    print("N structures accepted   {:10d}  {:>8.2%}".format(
        c_accepted, c_accepted/c_all))

    if len(np.array(refused_ds)) > 1:
        print("Refused distances quantiles: {:1.2e}, {:1.2e}, {:1.2e}, {:1.2e}, {:1.2e}".format(
            *np.quantile(np.array(refused_ds), [0, 0.25, 0.5, 0.75, 1])))
