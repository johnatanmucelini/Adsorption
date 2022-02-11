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

DEFAULT_ATOMS_RADII = {'Pd': 2., 'O': 1.}


def RegRDS_set(sampling_distance, N):
    """Return a set of N R3 dots (almost) regular  distributed in the surface of
    a sphere of radius 'sampling_distance'.
    More deatiail of the implementation in the article "How to generate
    equidistributed points on the surface of a sphere" by Markus Deserno:
    https://www.cmu.edu/biolphys/deserno/pdf/sphere_equi.pdf
    samplind_distance: a float or a int grater than zero.
    N: intiger grater than zero."""

    cart_coordinates = []
    r = 1
    Ncount = 0
    a = 4. * np.pi * r**2 / N
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
            y = sampling_distance * np.sin(theta) * np.cos(phi)
            x = sampling_distance * np.sin(theta) * np.sin(phi)
            z = sampling_distance * np.cos(theta)
            cart_coordinates.append([x, y, z])
    cart_coordinates = np.array(cart_coordinates)

    return cart_coordinates


def writing_points_xyz(file_name, positions):
    """Write positions, a list or array of R3 points, in a xyz file file_named.
    Several softwares open xyz files, such as Avogadro and VESTA
    In the xyz file all the atoms are H.
    file_name: a string with the path of the xyz document which will be writed.
    positions: a list or numpy array with the atoms positions."""

    if type(file_name) != str:
        sys.exit("file_name must be a string")

    for index, element in enumerate(positions):
        if len(element) != 3:
            sys.exit("Element " + str(index)
                     + " of positions does not present three elements.")

    if type(positions) != list:
        positions = np.array(positions)

    ase.io.write(file_name, ase.Atoms(
        'H'+str(len(positions)), list(map(tuple, positions))))


def map_surface(positions, radii, trial_n_dots_per_atom=1000):
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

    # Centralizing atoms positions:
    mol_av_position = np.average(positions, axis=0)
    positions = positions - mol_av_position
    n_atoms = len(positions)

    # calculation dots positions in surfaces arround each atom
    dots_default = RegRDS_set(1, trial_n_dots_per_atom)
    n_dots_per_atom = len(dots_default)
    n_dots = n_dots_per_atom * n_atoms
    # print('    Number of dots per atom: {}'.format(n_dots_per_atom))
    # print('    Number of investigated dots: {}'.format(n_dots))
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
        dists = cdist(reference_positions, mol.positions)
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

    def map_surface(self, n_dots_per_atom):
        """Map the surface dots positions and its atoms."""
        print("Mapping surface dots arround the atomic structure.")
        dots, dots_atom = map_surface(
            self.positions, self.radii, n_dots_per_atom)
        self.surf_dots = dots
        self.surf_dors_atom = dots_atom

    def featurization_surface_dots(self, metric):
        """Calculate the features for each dot."""
        print("Featurization of the surface dots.")
        self.surface_dots_features = metric.get_feature(self, self.surf_dots)

    def clusterization_surface_dots(self, n_cluster, n_repeat=5):
        """Calculate the cluster of the surface dots: indexes and centroid nearest"""
        print("Clusterization of the surface dots.")
        data = self.surface_dots_features
        clusterization_info = []
        for seed in range(n_repeat):
            # clusterizaiton: find centroids and a score*
            # *the mean euclidian_distance_to_the centroids
            centroids, score = kmeans(data, n_cluster, seed=seed)
            clusterization_info.append([score, centroids])
        top_centroids = sorted(clusterization_info)[0][1]
        idx, _ = vq(data, top_centroids)
        dists = cdist(top_centroids, data)
        centroids_nearst_idx = np.argmin(dists, axis=1)
        self.surface_dots_km_index = idx
        self.surface_dots_km_rep_idx = centroids_nearst_idx
        self.surface_dots_km_rep = data[centroids_nearst_idx]


def cluster_adsorption(mol_a_path, mol_b_path, n_surf_dots=100, n_struc=1e4):
    """It build adsorbed structures between two molecules, mol_a and mol_b.
    Both molecules surface are maped based in a """

    # reading input structures
    mol_a = Mol(path=mol_a_path)
    mol_b = Mol(path=mol_b_path)
    mol_a.get_radii()  # distância ráios dos átomos padrão,  # C sp3 , C sp2, =O, -O, H, N
    mol_b.get_radii()
    mol_a.map_surface(n_dots_per_atom=200)
    mol_b.map_surface(n_dots_per_atom=200)

    metric = Matric_euclidian()
    mol_a.featurization_surface_dots(metric)
    mol_b.featurization_surface_dots(metric)
    mol_a.clusterization_surface_dots(n_cluster=12, n_repeat=3)
    mol_b.clusterization_surface_dots(n_cluster=11, n_repeat=3)

    for ith_a, centroid_a in enumerate(mol_a.surface_dots_km_rep):
        for jth_b, centroid_b in enumerate(mol_b.surface_dots_km_rep):
            #print(ith_a, jth_b, centroid_a, centroid_b)
            pass

    #     - pontos de maior importancia química OU pontos randomicos(distancia em função dos ráios dos átomos):
    #         - adicionar mol_A a mol_B
    #         - rotações na molécula adsorvida
    #         - verificação de overlap, caso passe, estrutura final --> poll de estruturas válidas
    #
    #     - kmeans
    #
    #     # output
    #     - set de opçoes de output: {pasta_output name: 'folder_xyz_files',
    #                                 moléculals_name: '1.xyz', '2.xyz', '3.xyz', ...}
    #
    #     - exmplos de estruturas: mol_A_xyz (atomo x) + mol_B_xyz (ataomo y)
    #                             + ligação + indicação das distâncias
    #                             (atomo_x -- superficie_A -- supercie_B -- atomo_y)
    #
    #     return folder de todos os xyz validas,  folder de xyz kmeans
    #
    #
    # def kmeans(
    #         # inputs
    #         pasta de estruturas,
    #         metrica,
    # ):
    #     # arquivos ref/Cluster_AD_Pd4O8/script1_joh.py  1 (só a estrutura)
    #     return
    #
    # def comparacao_novo_velho(
    #         # inputs
    #         folder de todos os xyz encontrados novo
    #         folder de todos os xyz encontrados velho):
    #
    #     - pega os resultados dos metodos
    #
    #     - t-SNE map pra reduzir a dimensionalidade
    #     - comparação das distribuições
    #     - para todas as moléculas que forem encontradas
    #     - para todas as moléculas selecionadas


#path = 'arquivos_ref/Cluster_AD_Pd4O8/cluster.xyz'
path = 'C:\\Users\\User\\Documents\\GitHub\\lucas_script\\arquivos_ref\\Cluster_AD_Pd4O8\\cluster.xyz'
cluster_adsorption(path, path)
