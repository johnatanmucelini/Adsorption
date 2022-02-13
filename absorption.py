import numpy as np
from scipy.stats import describe
from scipy.spatial.distance import cdist
from scipy.cluster.hierarchy import single, fcluster
from scipy.spatial.distance import pdist
from scipy.cluster.vq import kmeans, vq

# parser = argparse.ArgumentParser(description='This script calculate the atoms exposition to the vacuum.')
# parser._action_groups.pop()
# required = parser.add_argument_group('required arguments')
# required.add_argument('--mol',       nargs='+',      metavar=('mol1','mol2'),  required=True, help='One or more molecular files (xyz, geometry.in, etc) to analyze.')
# optional = parser.add_argument_group('optional arguments')
# optional.add_argument('--r_adatom',  action='store', metavar='val', default=1.1,              help='The radius of the adatom. (Default=1.1)')
# optional.add_argument('--ssamples',  action='store', metavar='val', default=1000,             help='The (approximately) number of points to distribute over each atom. (Default=1000)')
# optional.add_argument('--r_atoms',   nargs='*',      metavar=('val1', 'val2'),                help='This flag controls the radii of the atoms in molecular files: If not defined, the atomic radii will defined as half of the average bond distance. If a single float value was priveded, it will be the radius for every atom on the molecular files. If N float values were provided, were N is the number of atoms in the molecular files, they will be the radius for each atom following the sequence of atoms in the molecular file. (Default=dav/2)')
# optional.add_argument('--save_surf', action='store', metavar='file.xyz',                      help='If defined, the position of the surface points found are writen in this xyz file as H atoms.')
# optional.add_argument('--save_json', action='store', metavar='file.json',                     help='If defined, all the collected data are writen in this json file.')
# args = parser.parse_args()
#
# mols_names = args.mol
# adatom_radius = float(args.r_adatom)
# ssamples = int(args.ssamples)
# sp_file = args.save_surf
# json_file = args.save_json

DEFAULT_ATOMS_RADII = {'Pd': 2.2, 'O': 2.2}


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


def build_surface(positions, radii, trial_n_dots_per_atom=1000):
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
    atomic_radii: numpy array of floats (n,) shaped.
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

    # calculation dots positions in surfaces arround each atom
    n_atoms = len(positions)
    dots_default = build_s2_grid(1, trial_n_dots_per_atom)
    n_dots_per_atom = len(dots_default)
    n_dots = n_dots_per_atom * n_atoms
    dots = []
    dots_atom = []
    for ith in range(n_atoms):
        dots.append(positions[ith] + dots_default * (radii[ith] + 1e-4))
        dots_atom.append([ith] * n_dots_per_atom)
    dots = np.array(dots, dtype=float).reshape(-1, 3)
    dots_atom = np.array(dots_atom).flatten()

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
    dotdot_distances += np.eye(n_dots_per_atom)*10
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

    return dots, dots_atom

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


class Mol:
    def __init__(self, path=None, positions=None, cheme=None):
        self.path = path
        self.positions = positions
        self.cheme = cheme

        # if niether of the positions were privided
        if not self.path and (not self.positions and not self.cheme):
            print(
                'Mol_path or positions + cheme must be provided to create a Mol.')

        # from the path
        if (not self.positions and not self.cheme) and self.path:
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

    def get_radii(self, atoms_radii_list=DEFAULT_ATOMS_RADII):
        """Get the raddii of the present atoms based in a list of its
        van-der-walls radius"""
        radii = []
        for ith, cheme in enumerate(self.cheme):
            radii.append(atoms_radii_list[cheme])
        self.radii = np.array(radii)

    def build_surface(self, n_dots_per_atom):
        """Map the surface dots positions and its atoms."""
        print("Mapping surface dots arround the atomic structure.")
        dots, dots_atom = build_surface(
            self.positions, self.radii, n_dots_per_atom)
        self.surf_dots = dots
        self.surf_dots_atom = dots_atom
        self.surf_atoms = np.unique(dots_atom)
        self.surf_atoms_positions = self.positions[self.surf_atoms]
        self.surf_atoms_raddii = self.radii[self.surf_atoms]

    def featurization_surface_dots(self, metric):
        """Calculate the features for each dot."""
        print("Featurization of the surface dots.")
        self.surf_dots_features = metric.get_feature(self, self.surf_dots)

    def clusterization_surface_dots(self, n_cluster, n_repeat=5):
        """Calculate the cluster of the surface dots: indexes and centroid nearest"""
        print("Clusterization of the surface dots.")
        data = self.surf_dots_features
        top_score = 1e20
        for seed in range(n_repeat):
            # clusterizaiton: find centroids and a score*
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

    def center_on(self, position, surface_image=False):
        """Center positions or surface atoms images positions on given
        position."""
        if surface_image:
            self.ipositions = self.surf_atoms_positions - position
        else:
            self.positions = self.positions - position

    def rotate(self, rot_matrix, surface_image=False):
        """Random sequence of SO(3) rotations, based in the Hopf coordinates
        parametrizations for SO(3)."""
        if surface_image:
            self.ipositions = np.dot(self.ipositions, rot_matrix)
        else:
            self.positions = np.dot(self.positions, rot_matrix)

    def to_xyz(self, file_name, surf_dots=False, surf_dots_color_by=None,
               special_surf_dots=None):
        """Write positions, a list or array of R3 points, in a xyz file file_named.
        file_name: a string with the path of the xyz document which will be writed.
        positions: a list or numpy array with the atoms positions."""

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

        print('Add this lines in the file: elements.ini (inside VESTA folder)')
        for s_val, r_val in zip(surf_index_to_cheme_dict.values(), rep_index_to_cheme_dict.values()):
            color = np.random.rand(3)
            print('  1  {}  0.25  0.25  0.25     {:0.5f}    {:0.5f}    {:0.5f}'.format(s_val,
                                                                                       *color
                                                                                       ))
            print('  1  {}  0.45  0.45  0.45     {:0.5f}    {:0.5f}    {:0.5f}'.format(r_val,
                                                                                       *color
                                                                                       ))

        # adding xyz information
        # atoms:
        atoms_positions = self.positions
        atoms_cheme = self.cheme
        if surf_dots:
            # surf dots:
            atoms_positions = np.append(
                atoms_positions, self.surf_dots, axis=0)
            if surf_dots_color_by is None:
                new_chemes = np.array(['@A']*len(self.surf_dots))
            elif surf_dots_color_by == 'kmeans':
                # surf dots colored by kmeans cluster
                new_chemes = []
                for dot_km_index in self.surf_dots_km_index:
                    new_chemes.append(surf_index_to_cheme_dict[dot_km_index])
            else:
                # surf dots colored by origin atom
                new_chemes = []
                for origin_atom_index in self.surf_dots_atom:
                    new_chemes.append(
                        surf_index_to_cheme_dict[origin_atom_index])
            atoms_cheme = np.append(atoms_cheme, np.array(new_chemes))
        if special_surf_dots:
            # adding km representative dots result
            atoms_positions = np.append(atoms_positions, self.surf_dots_km_rep,
                                        axis=0)
            new_chemes = []
            n_clusters = max(self.surf_dots_km_rep_kmindex) + 1
            for i in range(n_clusters):
                if special_surf_dots == 'kmeans':
                    new_chemes.append(rep_index_to_cheme_dict[i])
                else:
                    new_chemes.append('XX')
            atoms_cheme = np.append(atoms_cheme, new_chemes)

        with open(file_name, mode='w') as xyz_file:
            size = len(atoms_positions)
            xyz_file.write(str(size) + '\n\n')
            # print('size:', size, self.n, len(
            #     self.surf_dots), len(self.surf_dots_km_rep), atoms_cheme)
            for ith, (element, position) in enumerate(zip(atoms_cheme, atoms_positions)):
                #print(ith, element, position)
                if ith < size - 1:
                    xyz_file.write('{} {} {} {}\n'.format(element, *position))
                elif ith == size - 1:
                    #print(element, position)
                    xyz_file.write('{} {} {} {}'.format(element, *position))
            print('size:', size, self.n, len(
                self.surf_dots), len(self.surf_dots_km_rep), atoms_cheme)
            # print('last chemes:', atoms_cheme[-5:])
            # print('last positions:', atoms_positions[-5:])


def overlap(mol_a, mol_b, surface_image=False):
    """Verify if two atomic structures overlap, all structure or just surface
    atoms images."""
    if surface_image:
        ab_distance_matrix_size = len(mol_a.surf_dots) * mol_b.n
        ba_distance_matrix_size = len(mol_b.surf_dots) * mol_a.n
        if ab_distance_matrix_size > ba_distance_matrix_size:
            distances = cdist(mol_a.surf_atoms_positions, mol_b.surf_dots)
            atoms_raddii_ref = mol_a.surf_atoms_raddii
            n_dots = len(mol_b.surf_dots)
            n_atoms = len(mol_a.surf_atoms_positions)
        else:
            distances = cdist(mol_b.surf_atoms_positions, mol_a.surf_dots)
            atoms_raddii_ref = mol_b.surf_atoms_raddii
            n_dots = len(mol_a.surf_dots)
            n_atoms = len(mol_b.surf_atoms_positions)
        atoms_raddii_by_dot = np.array(
            [atoms_raddii_ref]*n_dots).reshape(n_dots, n_atoms).T
        print(np.sum(distances < atoms_raddii_by_dot))
        if np.sum(distances < atoms_raddii_by_dot):
            result = True
        else:
            result = False
        return result


def cluster_adsorption(mol_a_path, mol_b_path, n_surf_dots=2000, n_struc=1e4):
    """It build adsorbed structures between two molecules, mol_a and mol_b.
    Both molecules surface are maped based in a """

    # reading input structures
    mol_a = Mol(path=mol_a_path)
    mol_b = Mol(path=mol_b_path)

    # getting raddii
    # TODO: interpretation of the atoms: C sp3 , C sp2, =O, -O, H, N
    mol_a.get_radii()  #
    mol_b.get_radii()

    # mapping mol_a and mol_b surfaces
    mol_a.build_surface(n_dots_per_atom=n_surf_dots)
    mol_b.build_surface(n_dots_per_atom=n_surf_dots)

    metric = Matric_euclidian_mod()
    mol_a.featurization_surface_dots(metric)
    mol_b.featurization_surface_dots(metric)
    #print(mol_a.surf_dots_features.shape)
    mol_a.clusterization_surface_dots(n_cluster=25, n_repeat=5)
    mol_b.clusterization_surface_dots(n_cluster=12, n_repeat=5)
    mol_a.to_xyz('cluster_km.xyz', surf_dots=True,
                 surf_dots_color_by='kmeans', special_surf_dots='kmeans')
    mol_b.to_xyz('mol_km.xyz', surf_dots=True,
                 surf_dots_color_by='kmeans', special_surf_dots='kmeans')

    # calculation all rotations around each contact between mols a and b
    mesh_size = 0.8
    us1 = 2*np.pi*1
    us2 = 4*np.pi*1
    n_samples_spher = int(round(us1/mesh_size))
    n_samples_circ = int(round(us2/mesh_size**2))
    s2_coords = build_s2_grid(1, n_samples_spher, coords_system='spher')
    s1_coords = build_s1_grid(1, n_samples_circ, coords_system='circ')
    print("Number of rotations:", len(s2_coords)*len(s1_coords))
    rots = build_SO3_from_S1S2(s1_coords, s2_coords)

    for ith_a, centroid_a in enumerate(mol_a.surf_dots_km_rep):
        for jth_b, centroid_b in enumerate(mol_b.surf_dots_km_rep):
            for kth, rot in enumerate(rots):
                #print(ith_a, jth_b, centroid_a, centroid_b)
                mol_a.center_on(centroid_a, surface_image=True)
                mol_b.center_on(centroid_b, surface_image=True)
                mol_b.rotate(rot, surface_image=True)

                print(overlap(mol_a, mol_b, surface_image=True))

                break
            break
        break

    #     - pontos de maior importancia química OU pontos randomicos(distancia em função dos ráios dos átomos):
    #         - adicionar mol_A a mol_B
    #         - rotações na molécula adsorvida
    #         - verificação de overlap, caso passe, estrutura final --> poll de estruturas válidas
    #     - kmeans
    #     # output
    #     - set de opçoes de output: {pasta_output name: 'folder_xyz_files',
    #                                 moléculals_name: '1.xyz', '2.xyz', '3.xyz', ...}
    #     - exmplos de estruturas: mol_A_xyz (atomo x) + mol_B_xyz (ataomo y)
    #                             + ligação + indicação das distâncias
    #                             (atomo_x -- superficie_A -- supercie_B -- atomo_y)
    #     return folder de todos os xyz validas,  folder de xyz kmeans
    # def comparacao_novo_velho(
    #         # inputs
    #         folder de todos os xyz encontrados novo
    #         folder de todos os xyz encontrados velho):
    #     - pega os resultados dos metodos
    #     - t-SNE map pra reduzir a dimensionalidade
    #     - comparação das distribuições
    #     - para todas as moléculas que forem encontradas
    #     - para todas as moléculas selecionadas


#path = 'arquivos_ref/Cluster_AD_Pd4O8/cluster.xyz'
path_cluster = 'C:\\Users\\User\\Documents\\GitHub\\lucas_script\\arquivos_ref\\Cluster_AD_Pd4O8\\cluster.xyz'
path_mol = 'C:\\Users\\User\\Documents\\GitHub\\lucas_script\\arquivos_ref\\Cluster_AD_Pd4O8\\molecule.xyz'
cluster_adsorption(path_cluster, path_mol)
