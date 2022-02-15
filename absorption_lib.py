import shutil
import os
import sys
import numpy as np
from scipy.stats import describe
from scipy.spatial.distance import cdist
from scipy.cluster.hierarchy import single, fcluster
from scipy.spatial.distance import pdist
from scipy.cluster.vq import kmeans, vq


def build_s2_grid(radius, n_sample_init, coords_system='cart'):
    """Generate a (almost) regular grid on the surface of sphere of given radius.
    More deatiail of the implementation in the article "How to generate
    equidistributed points on the surface of a sphere" by Markus Deserno:
    https://www.cmu.edu/biolphys/deserno/pdf/sphere_equi.pdf"""

    cart_coordinates = []
    spherical_coordinates = []
    r = 1
    Ncount = 0
    a = 4. * np.pi * r**2 / n_sample_init
    d = np.sqrt(a)
    Mtheta = int(round(np.pi/d))
    dtheta = np.pi / Mtheta
    dphi = a / dtheta
    for m in range(0, Mtheta):
        theta = np.pi * (m + 0.5) / Mtheta
        Mphi = int(round(2 * np.pi * np.sin(theta) / dphi))
        for n in range(0, Mphi):
            phi = 2 * np.pi * n / Mphi
            Ncount += 1
            y = radius * np.sin(theta) * np.cos(phi)
            x = radius * np.sin(theta) * np.sin(phi)
            z = radius * np.cos(theta)
            cart_coordinates.append([x, y, z])
            spherical_coordinates.append([theta, phi])
    cart_coordinates = np.array(cart_coordinates)
    spherical_coordinates = np.array(spherical_coordinates)

    if coords_system == 'cart':
        result = cart_coordinates
    elif coords_system == 'spher':
        result = spherical_coordinates

    return result


def build_s1_grid(radisus, n_samples, coords_system='circ'):
    """Circle grid, result in circular and cartezian coordinates"""
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


def build_SO3_from_S1S2(s1_coords, s2_coords):
    """Build the quarternion rotation matrix from s1 and s2 grids:
    s1_coords (psi) in [0, 2pi)
    s2_coords (theta, phi) in [0, pi) and [0, 2pi), respectively."""
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
    for nth, (qr, qi, qj, qk) in enumerate(quarternions):
        # s = 1/sum(quarternions[nth]**2)
        # print(s)
        rot_matrix.append(np.array([
            [1-2*(qj**2+qk**2),   2*(qi*qj-qk*qr),   2*(qi*qk+qj*qr)],
            [2*(qi*qj+qk*qr),   1-2*(qi**2+qk**2),   2*(qj*qk-qi*qr)],
            [2*(qi*qk-qj*qr),     2*(qj*qk+qi*qr), 1-2*(qi**2+qj**2)]
        ]))
    return rot_matrix


def build_surface(positions, radii, atoms_surface_density=5):
    """This algorithm classify atoms in surface and core atoms employing the
    concept of atoms as ridge spheres. Then the surface atoms are the ones that
    could be touched by an fictitious adatom that approach the cluster, while
    the core atoms are the remaining atoms.
    See more of my algorithms im GitHub page Johnatan.mucelini.
    Articles which employed thi analysis: Mendes P. XXX
    .
    Parameters
    ----------
    positions: numpy array of floats (n,3) shaped.
               Cartezian positions of the atoms, in angstroms.
    radii: numpy array of floats (n,) shaped.
           Radius of the atoms, in the same order which they appear in
           positions, in angstroms.
    ssampling: intiger (optional, default=1000).
               Quantity of samplings over the touched sphere surface of each
               atom.

    Return
    ------
    dots: numpy array of floats (n,3).
          The positions of the surface dots, where n is the number of dots.
    dots_atom: numpy array of floats (n,3).
               The index of the atom that originate the dot, where n is the
               number of dots.
    """

    trial_n_dots_per_atom = atoms_surface_density * 4 * np.pi * radii**2
    # calculation dots positions in surfaces arround each atom
    n_atoms = len(positions)
    #
    #n_dots_per_atom = len(dots_default)

    dots_atom = np.empty(0, int)
    n_dots_per_atom = []
    dots = np.empty((0, 3), float)
    for ith in range(n_atoms):
        dots_default = build_s2_grid(1, trial_n_dots_per_atom[ith])
        dots = np.append(
            dots, positions[ith] + dots_default * (radii[ith] + 1e-4), axis=0)
        n_dots_per_atom.append(len(dots_default))
        dots_atom = np.append(dots_atom, np.array([ith] * len(dots_default)))
    n_dots_per_atom = np.array(n_dots_per_atom)
    n_dots = sum(n_dots_per_atom)
    print('    Initial number of dots per atom: {}'.format(n_dots_per_atom))

    # remove dots inside other atoms touch sphere
    dots_atoms_distance = cdist(positions, dots)
    radii_by_dot = np.array([radii]*n_dots).reshape(n_dots, n_atoms).T
    overleped = np.any(dots_atoms_distance < radii_by_dot, axis=0)
    non_overleped = np.invert(overleped)
    dots = dots[non_overleped]
    dots_atom = dots_atom[non_overleped]

    # Seach if there are more than one surfaces in dots (internal surfaces)
    dots_large_atom = dots_default*max(radii)
    dotdot_distances = cdist(dots_large_atom, dots_large_atom)
    dotdot_distances += np.eye(len(dotdot_distances))*10
    min_dotdot_distance = np.min(dotdot_distances)
    eps = 2.1 * min_dotdot_distance
    condensed_distances_matrix = pdist(dots)
    Z = single(condensed_distances_matrix)
    labels_per_dot = fcluster(Z, eps, criterion='distance')
    labels, quanity = np.unique(labels_per_dot, return_counts=True)
    if len(labels) > 1:
        print('Warning: {} surfaces were found (sizes: {})'.format(
              labels, str(quanity).replace('[', '').replace(']', '')))
        print('         the biggest surface were selected as the external one!')
    larges_cluster_dots = labels_per_dot == labels[np.argmax(quanity)]
    dots = dots[larges_cluster_dots]
    dots_atom = dots_atom[larges_cluster_dots]

    area_per_atom_dot = (4 * np.pi*radii ** 2) / n_dots_per_atom
    #print(area_per_atom_dot, dots_atom, area_per_atom_dot[dots_atom])
    area = sum(area_per_atom_dot[dots_atom])

    return dots, dots_atom, area

# TODO: Tem um erro aqui, as distâncias de cada vetor de feature precisam ser
#       sorteadas pra que fiquem na ordem certa


class Matric_euclidian:
    """Euclidian metrics tools"""

    def get_distance(self, features_1, features_2):
        """Distance metric between two samples with features_1 and features_2"""
        dividendo = np.sum((features_1 - features_2)**2, axis=0)
        divisor = np.sum(features_1**2 + features_2**2, axis=0)
        return (dividendo/divisor) * 1E6

    def get_feature(self, mol, reference=None):
        """calculates the euclidian distances features for the molecules or for a
        reference point in space"""
        features = []
        if reference is None:
            reference_positions = mol.positions.mean(1)
        else:
            reference_positions = reference
        dists = np.sort(cdist(reference_positions, mol.positions))
        if reference is None:
            result = dists[0]
        else:
            result = dists
        return result


class Matric_euclidian_mod:
    """Euclidian metrics tools"""

    def get_distance(self, features_1, features_2):
        """Distance metric between two samples with features_1 and features_2"""
        return np.linalg.norm(features_1 - features_2)

    def get_feature(self, mol, reference=None):
        """calculates the euclidian distances features for the molecules or for a
        reference point in space"""
        features = []
        if reference is None:
            reference_positions = mol.positions.mean(1)
        else:
            reference_positions = reference
        dists = []
        ucheme = sorted(np.unique(mol.cheme))
        n_cheme = len(ucheme)
        e_index = mol.cheme == ucheme[0]
        dists = np.sort(cdist(reference_positions, mol.positions[e_index]))
        #print('dist1:', dists.shape)
        for ith in range(1, n_cheme):
            e_index = mol.cheme == ucheme[ith]
            #dist = cdist(reference_positions, mol.positions[e_index])
            #sorted_dist = np.sort(dist, axis=1)
            #print('sorting:', dist[1000], '-->', sorted_dist[1000])
            #dists = np.append(dists, sorted_dist, axis=1)
            dists = np.append(dists, np.sort(
                cdist(reference_positions, mol.positions[e_index]), axis=1), axis=1)
            #print('dist:', dist.shape, 'sorted_dist',
            #      sorted_dist.shape, 'dists', dists.shape)
        if reference is None:
            result = dists[0]
        else:
            result = dists
        return result


def add_mols(mol_a, mol_b, image=False, add_surf_info=False):
    """Add the image of the molecules in a new molecule"""
    # basic information
    if image:
        mol_f_positions = np.append(mol_a.ipositions, mol_b.ipositions, axis=0)
        mol_f_cheme = np.append(mol_a.surf_cheme, mol_b.surf_cheme)
    else:
        mol_f_positions = np.append(mol_a.positions, mol_b.positions, axis=0)
        mol_f_cheme = np.append(mol_a.cheme, mol_b.cheme)
    # defining Mol
    mol_f = Mol(positions=mol_f_positions, cheme=mol_f_cheme)
    # adding surface information
    if add_surf_info:
        # surf_dots
        mol_f.surf_dots = np.append(mol_a.surf_dots, mol_b.surf_dots, axis=0)
        # surf_dots_km_index
        #print('surf_dots_km_index:', mol_a.surf_dots_km_index)
        mol_f.surf_dots_km_index = np.append(
            mol_a.surf_dots_km_index, mol_b.surf_dots_km_index + max(mol_a.surf_dots_km_index) + 1, axis=0)
        # surf_dots_atom
        #print('surf_dots_atom:', mol_a.surf_dots_atom)
        mol_f.surf_dots_atom = np.append(
            mol_a.surf_dots_atom, mol_b.surf_dots_atom + mol_a.n, axis=0)
        # surf_dots_km_rep
        #print('surf_dots_km_rep:', mol_a.surf_dots_km_rep)
        mol_f.surf_dots_km_rep = np.append(
            mol_a.surf_dots_km_rep, mol_b.surf_dots_km_rep, axis=0)
        # surf_dots_km_rep_kmindex
        #print('surf_dots_km_rep_kmindex:', mol_a.surf_dots_km_rep_kmindex)
        mol_f.surf_dots_km_rep_kmindex = np.append(
            mol_a.surf_dots_km_rep_kmindex, mol_b.surf_dots_km_rep_kmindex + max(mol_a.surf_dots_km_rep_kmindex) + 1, axis=0)
    if image and add_surf_info:
        mol_f.ipositions = np.append(
            mol_a.ipositions, mol_b.ipositions, axis=0)
        mol_f.isurf_dots = np.append(
            mol_a.isurf_dots, mol_b.isurf_dots, axis=0)
        mol_f.isurf_dots_km_rep = np.append(
            mol_a.isurf_dots_km_rep, mol_b.isurf_dots_km_rep, axis=0)
    mol_f.n = len(mol_f.positions)
    return mol_f


DEFAULT_ATOMS_RADII = {'Pd': 2.1, 'O': 2.2, 'C': 2}


class Mol:
    def __init__(self, path=None, positions=None, cheme=None):
        self.path = path
        self.positions = positions
        self.cheme = cheme

        # if niether of the positions were privided
        if self.path is None and (self.positions is None and self.cheme is None):
            print(
                'Mol_path or positions + cheme must be provided to create a Mol.')
            sys.exit(1)

        # from the path
        if (self.positions is None and self.cheme is None) and self.path is not None:
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
        msg_ref = "  Atom {}, vdw radii {}, ref {}"
        msg_not_ref = "  WARNING: missing reference for {} vdw radius, employing {}."
        radii = []
        for ith, cheme in enumerate(self.cheme):
            found = False
            for obj in atoms_radii_preferences:
                if not found:
                    if isinstance(obj, dict):
                        if cheme in obj.keys():
                            radii.append(obj[cheme])
                            found = True
                            if 'ref' in obj.keys():
                                print(msg_ref.format(
                                    cheme, obj[cheme], obj['ref']))
                            else:
                                print(msg_not_ref.format(cheme, obj[cheme]))
                    if isinstance(obj, int):
                        radii.append(obj)
                        found = True
                        print(msg_not_ref.format(cheme, obj[cheme], obj.ref))
        self.radii = np.array(radii)

    def build_surface(self, atoms_surface_density=4):
        """Map the surface dots positions and its atoms."""
        print("Mapping surface dots arround the atomic structure.")
        dots, dots_atom, area = build_surface(
            self.positions, self.radii, atoms_surface_density=atoms_surface_density)
        self.surf_dots = dots
        self.surf_dots_atom = dots_atom
        self.surf_atoms_index, counts = np.unique(
            dots_atom, return_counts=True)
        self.surf_cheme = self.cheme[self.surf_atoms_index]
        self.surf_atoms_positions = self.positions[self.surf_atoms_index]
        self.surf_atoms_raddii = self.radii[self.surf_atoms_index]
        # area:
        print('    N surface points {:7d}'.format(len(dots)))
        print('    Surface area      {:10.3f} AA'.format(area))
        print('    Points density    {:10.3f} AA^-1'.format(len(dots)/area))

    def featurization_surface_dots(self, metric):
        """Calculate the features for each dot."""
        print("Featurization of the surface dots.")
        self.surf_dots_features = metric.get_feature(self, self.surf_dots)

    def clustering_surface_dots(self, n_cluster, n_repeat=5):
        """Calculate the cluster of the surface dots: indexes and centroid nearest"""
        print("Clustering of the surface dots.")
        data = self.surf_dots_features
        top_score = 1e20
        for seed in range(n_repeat):
            # clustering: find centroids and a score*
            # *the mean euclidian_distance_to_the centroids
            centroids, score = kmeans(data, n_cluster, seed=seed)
            if score < top_score:
                top_score = score
                top_centroids = centroids
        idx, _ = vq(data, top_centroids)
        dists = cdist(top_centroids, data)
        centroids_nearst_idx = np.argmin(dists, axis=1)
        self.surf_dots_km_index = idx
        self.surf_dots_km_rep_kmindex = idx[centroids_nearst_idx]
        self.surf_dots_km_rep_idx = centroids_nearst_idx
        self.surf_dots_km_rep = self.surf_dots[centroids_nearst_idx]

    def translate_by(self, vector, image=False):
        """Translate atoms and isurf_dots (optional) on any given position."""
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
        """Random sequence of SO(3) rotations, based in the Hopf coordinates
        parametrizations for SO(3)."""
        if image:
            self.ipositions = np.dot(self.ipositions, rot_matrix)
            self.isurf_dots = np.dot(self.isurf_dots, rot_matrix)
            self.isurf_dots_km_rep = np.dot(self.isurf_dots_km_rep, rot_matrix)
        else:
            self.positions = np.dot(self.positions, rot_matrix)

    def to_xyz(self, file_name, surf_dots=False, surf_dots_color_by=None,
               special_surf_dots=None, verbose=True):
        """Write positions, a list or array of R3 points, in a xyz file file_named.
        file_name: a string with the path of the xyz document which will be writed.
        positions: a list or numpy array with the atoms positions."""

        if verbose:
            print('Writing mol to: {}'.format(file_name))

        # modiffied vesta atoms to colorize
        letras = ['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k', 'l', 'm', 'n',
                  'o', 'p', 'q', 'r', 's', 't', 'u', 'v', 'w', 'x', 'y', 'z']
        color_base_surf = []
        color_base_rep = []
        for c1, c2 in zip(['@', '&'], ['#', '!']):
            for l in letras:
                color_base_surf.append(c1 + l)
                color_base_rep.append(c2 + l)
        color_base_surf = np.array(color_base_surf)
        color_base_rep = np.array(color_base_rep)
        surf_index_to_cheme_dict = {}
        rep_index_to_cheme_dict = {}
        for ith in range(len(color_base_surf)):
            surf_index_to_cheme_dict[ith] = color_base_surf[ith]
            rep_index_to_cheme_dict[ith] = color_base_rep[ith]

        if False:
            print('Add this lines in the file: elements.ini (inside VESTA folder)')
            for s_val, r_val in zip(surf_index_to_cheme_dict.values(), rep_index_to_cheme_dict.values()):
                color = np.random.rand(3)
                str_s = '  1  {}  0.25  0.25  0.25     {:0.5f}    {:0.5f}    {:0.5f}'
                str_r = '  1  {}  0.45  0.45  0.45     {:0.5f}    {:0.5f}    {:0.5f}'
                print(str_s.format(s_val, *color))
                print(str_r.format(r_val, *color))

        # adding xyz information
        # atoms:
        atoms_positions = self.positions
        atoms_cheme = self.cheme
        if len(atoms_positions) != len(atoms_cheme):
            print('Tamanho dos vetores positions cheme é diferente b: {} {}'.format(
                len(atoms_positions), len(atoms_cheme)))

        # surf dots:
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
            print('Tamanho dos vetores positions cheme é diferente w: {} {}'.format(
                len(atoms_positions), len(atoms_cheme)))
        with open(file_name, mode='w') as xyz_file:
            size = len(atoms_positions)
            xyz_file.write(str(size) + '\n\n')
            for ith, (element, position) in enumerate(zip(atoms_cheme, atoms_positions)):
                #print(ith, element, position)
                if ith < size - 1:
                    xyz_file.write('{} {} {} {}\n'.format(element, *position))
                elif ith == size - 1:
                    #print(element, position)
                    xyz_file.write('{} {} {} {}'.format(element, *position))
            # print('size:', size, self.n, len(
            #     self.surf_dots), len(self.surf_dots_km_rep), atoms_cheme)
            # print('last chemes:', atoms_cheme[-5:])
            # print('last positions:', atoms_positions[-5:])

    def add_atoms(self, elements, positions):
        """Add atoms to the system"""
        self.positions = np.append(self.positions, positions, axis=0)
        self.cheme = np.append(self.cheme, elements)
        self.n = len(self.positions)
        if 'ipositions' in dir(self):
            self.ipositions = np.append(self.ipositions, positions, axis=0)

    def surf_to_real(self):
        if 'ipositions' in dir(self):
            # print('to_real', len(self.positions), len(self.ipositions))
            self.positions = self.ipositions * 1.
            self.surf_dots = self.isurf_dots * 1.
            self.surf_dots_km_rep = self.isurf_dots_km_rep * 1.


def overlap(mol_a, mol_b, flexibility=0.85, image=False):
    """Verify if two atomic structures overlap, all structure or just surface
    atoms images."""
    if image:
        distances = cdist(mol_a.ipositions, mol_b.ipositions)
        radii_sum = mol_a.radii.reshape(-1, 1) + mol_b.radii.reshape(1, -1)
        #print(distances)
        #print(radii_sum)
        if np.any((distances/radii_sum) < flexibility):
            result = True
        else:
            result = False
    return result
    # atoms_raddii_by_dot = np.array(
    #     [atoms_raddii_ref]*n_dots).reshape(n_dots, n_atoms).T
    # print(np.sum(distances < atoms_raddii_by_dot))
    # if np.sum(distances < atoms_raddii_by_dot):
    #     result = True
    # else:
    #     result = False
    # return result


def status(c_all, n_config, c_repeated, c_overlapped, c_accepted, refused_ds):
    print('-'*80)
    print("N structures analyzed   {:10d}  {:>8.2%}".format(
        c_all, c_all/n_config))
    print("N structures accepted   {:10d}  {:>8.2%}".format(
        c_accepted, c_accepted/c_all))
    print("N structures repeated   {:10d}  {:>8.2%}".format(
        c_repeated, c_repeated/c_all))
    print("N structures overlapped {:10d}  {:>8.2%}".format(
        c_overlapped, c_overlapped/c_all))
    print("Refused distances quantiles: {:0.3e}, {:0.3f}, {:0.3f}, {:0.3f}, {:0.3f}".format(
        *np.quantile(np.array(refused_ds), [0, 0.25, 0.5, 0.75, 1])))
