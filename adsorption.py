import shutil
import os
import os.path
import argparse
from scipy.spatial.distance import cdist
from scipy.cluster.vq import kmeans, vq
import numpy as np
import adsorption_lib as lib

help_text = """MOLECULAR ADSORPTION BY SURFACE MAPPING ALGORITHM

For a given pair of atomic structures, the present algorithm generates sets of
adsorbed configurations, considering ridge structures and atoms as spheres of
VDW radius (which could overlap regulated):

(1) Read the mol and associate VDW radii for each atom:
    There are VDW radii for some atoms and their reference, but others can be
    added manually, search for "VDW RADII AND ITS REF" in this document.

(2) Both molecule surfaces are mapped to find representative points:
    The surface of a molecule is an outside surface built with the union of
    ridge spheres of VDW radii around each atom. The files mol_a_surf.xyz and
    mol_b_surf.xyz present this data [a].

    Points in these spheres (SO2) are obtained with the algorithm described by
    Deserno (see the article "How to generate equidistributed points on the
    surface of a sphere" [b]). Then, for each point of the surface, features
    are extracted. The features vector contains the sorted distances from the
    point to each atom of the molecule, separated by the chemical element of
    the atoms. Based on a K-means clustering, the surface dots are
    clusters/groups, and the point nearest to the centroid of its clusters is
    selected as its representative point in the molecular surface. The files
    mol_a_km.xyz and mol_b_km.xyz present this data [a].

    [a] The structure with the surface dots can be seen in the VESTA code.
    To correct read their data, you must replace the VESTA configuration file
    elements.ini with the elements.ini file added in the present project. These
    files present the types of atoms, colors, and other properties to
    automatically add colors to the representation. In the surf.xyz files
    present the surface dots with a color for the points associated with each
    atom. The km.xyz files present the surface dots with a color for the points
    associated with each cluster of surface dots.
    [b] Article: https://www.cmu.edu/biolphys/deserno/pdf/sphere_equi.pdf.

(3) Adsorption is performed by combining both molecules by each pair of the
    representative point of its surfaces. Also, for each pair of representative
    points, many rotations are performed to guarantee good matches between the
    molecules. These rotations are performed with a grid of rotations of SO3,
    obtained with a method called Successive Orthogonal Images on SOn. The
    method was first presented by Mitchell (DOI:10.1137/030601879), but for the
    present implementation, I followed the paper by Yershova
    (DOI:10.1177/0278364909352700).
        Thus, the number of adsorbed molecules configurations to analyze is
    deterministic and is the product of the number of surface clusters for each
    molecule and the number of rotations in SO3. The configurations are added
    to a pull of structures when:
        (i) The molecule did not overlap in the adsorption, which is considered
        to have happened when a pair of atoms of different molecules were
        distant by less than the sum of their VDW radii multiplied by the
        parameter ovlp_threshold;
        (ii) The molecule is not similar to any other in the pull of structure,
        which is verified with a simple filter. Calculate the features vectors
        for a molecule as follows: a) three references are calculated, the
        geometrical center of the molecules and their touch contact.
        b) calculate the distance between references and positions,
        c) calculate a slow-decay function of these distance values, d) sort
        these results, for each reference and chemical element. The slow-decay
        function is exp(-dist**(2/3)). If the euclidian distance between the
        present configuration and all other structures in the pull were smaller
        than sim_threshold parameter.

(4) Representative structures are sampled from the pool with another K-means,
    and the structures are written in folder_xyz_files (adsorbed structures)
    and folder_xyz_files_withsurfs (adsorbed structures with surface
    information).

Usage example:
$ python adsorption.py --mols cluster.xyz molecule.xyz \\
                       --surf_ks         5 9           \\
                       --n_final         100           \\
                       --surf_d           10           \\
                       --n_repeat_km      20           \\
                       --n_rot            60           \\
                       --ovlp_threshold 0.95           \\
                       --sim_threshold  0.04           \\
                       --out_sufix        _2
"""

# Parsing objects
parser = argparse.ArgumentParser(
    description=help_text, formatter_class=argparse.RawTextHelpFormatter)
parser._action_groups.pop()
req = parser.add_argument_group('required arguments')
opt = parser.add_argument_group('optional arguments')
req.add_argument('--mols', nargs=2, action='store', metavar=('a.xyz', 'b.xyz'),
                 required=True, help='The two molecules to adsorb.')
req.add_argument('--surf_ks', nargs=2, action='store', metavar=('K_a', 'K_b'),
                 required=True,
                 help='Numbers k (K-means) for each surface dots clustering')
req.add_argument('--n_final', nargs=None, action='store',
                 metavar='N_final', required=True,
                 help=('Number of final structures, in the representative set.'
                       ))
opt.add_argument('--surf_d', nargs=None, action='store', metavar='val',
                 default=10,
                 help='Density of points over the atoms. (default: 10, '
                 + 'AA^(-2))')
opt.add_argument('--n_repeat_km', nargs=None, action='store', metavar='val',
                 default=20,
                 help='Number of times to repeat each clustering. (default: '
                      + '20)')
opt.add_argument('--n_rots', nargs=None, action='store', metavar='val',
                 default=60,
                 help='approximatated number of rotations (default: 60)')
opt.add_argument('--ovlp_threshold', nargs=None, action='store', metavar='val',
                 default=0.9,
                 help='Structures overlap threshold (default: 0.9)')
opt.add_argument('--sim_threshold', nargs=None, action='store', metavar='val',
                 default=0.04,
                 help='Structures similarity threshold (default: 0.04)')
opt.add_argument('--out_sufix', nargs=None, action='store', metavar='sufix',
                 default='',
                 help='Sufix of the output folders:  folder_xyz_files+surfix '
                      + 'and folder_xyz_files_withsurfs+surfix (default: None)'
                 )
args = parser.parse_args()

# Example or argument
# argument = '--mols molecule.xyz cluster.xyz --surf_ks 10 40 --n_final 505 ' +
#            '--surf_d 10 --n_repeat_km 10 --n_rots 60 ' +
#            '--ovlp_threshold 0.90 --sim_threshold 0.04 --out_sufix _3'
# args = parser.parse_args(argument.split())
# args = parser.parse_args(['--help'])


def cluster_adsorption(mol_a_path, mol_a_surf_km_k, mol_b_path,
                       mol_b_surf_km_k, final_n_structures, n_km_repeat,
                       surface_density, n_rot_r, sim_threshold,
                       ovlp_threshold, out_sufix=''):
    """It build sets of adsorbed structures between two atomic structures,
    mol_a and mol_b. See the package description."""

    # algorithm parameters:
    surface_km_mol_a_cluster = mol_a_surf_km_k
    surface_km_mol_b_cluster = mol_b_surf_km_k
    n_repeat_final_km = n_km_repeat
    surface_km_mol_a_n_repeat = n_km_repeat
    surface_km_mol_b_n_repeat = n_km_repeat

    # Calculating all rotations matrix
    n_rots_s1_r = int(round((np.pi*n_rot_r)**(1/3)))
    n_rots_s2_r = int(round((n_rot_r**2/np.pi)**(1/3)))
    s2_coords = lib.build_s2_grid(1, n_rots_s1_r, coords_system='spher')
    s1_coords = lib.build_s1_grid(1, n_rots_s2_r, coords_system='circ')
    rots = lib.build_s3_from_s1s2(s1_coords, s2_coords)
    n_rots_s1 = len(s1_coords)
    n_rots_s2 = len(s2_coords)
    n_rots = n_rots_s1 * n_rots_s2
    n_config = n_rots * surface_km_mol_a_cluster * surface_km_mol_b_cluster

    # Testing rotations uniformity:
    # coords_cart = lib.build_s2_grid(1, 100, coords_system='cart')
    # keep = []
    # for cart in coords_cart:
    #     x, y, z = cart
    #     if x > 0 and y > 0 and z > 0:  # removing any symmetry
    #         keep.append(cart)
    # keep = np.array(keep)
    # rotate_images = np.empty((0, 3), float)
    # for rot in rots:
    #     rotate_images = np.append(rotate_images, np.dot(keep, rot), axis=0)
    # rotate_images = np.array(rotate_images)
    # mean_xyz_value = np.abs(np.sum(rotate_images, axis=0))
    # if np.any(mean_xyz_value > 1e-8):
    #     print('{:<{}s} {:1.2e} {:1.2e} {:1.2e}'.format(
    #         'Uniformity deviation', left, *mean_xyz_value))
    #     print('    WARNING: rotations may not be uniform enought.\n'
    #           + '    Please, consider change the number of ratations. If it '
    #           + '    persist, please, contact me: johnatan.mucelini@gmail.com')

    # Printing Header: parameters and important numbers
    left = 25
    print('+'+'-'*78+'+')
    print(' {:<78s} '.format(
        'MOLECULAR ADSORPTION BY SURFACE MAPPING ALGORITHM'))
    print('+'+'-'*78+'+')
    print('{:<{}s} '.format('PARAMETERS:', left))
    print('{:<{}s} {}'.format('Mol A', left, mol_a_path))
    print('{:<{}s} {}'.format('Mol B', left, mol_b_path))

    # Surface mapping
    print('{:<{}s} {}'.format('N cluster surface A', left, mol_a_surf_km_k))
    print('{:<{}s} {}'.format('N cluster surface B', left, mol_b_surf_km_k))
    print('{:<{}s} {} {}'.format('Surface mapping density',
                                 left, surface_density, 'AA^-2'))
    # Rotations
    print('{:<{}s} {}'.format('N rotations final', left, n_rots))
    print('{:<{}s} {}'.format('N rotations requested', left, n_rot_r))
    print('{:<{}s} {}'.format('N rotations S2', left, n_rots_s2))
    print('{:<{}s} {}'.format('N rotations S1', left, n_rots_s1))

    # Thresholds
    print('{:<{}s} {}'.format('Simility threshold', left, sim_threshold))
    print('{:<{}s} {}'.format('Overlatp threshold', left, ovlp_threshold))

    # Others
    print('{:<{}s} {}'.format('N configurations', left, n_config))
    print('{:<{}s} {}'.format('N final structure', left, final_n_structures))
    print('{:<{}s} {}'.format('N kmeans repetition', left, n_km_repeat))

    # VDW RADII:
    # Add or edit vdw radii here, see the example bellow:
    # ref: The Journal of Physical Chemistry, 68, 3, 1964
    vdw_atomic_radius_bondi = {'ref': 'J. Physical Chemistry, 68, 3, 1964',
                               'H': 1.20, 'He': 1.40, 'C': 1.70, 'N': 1.55,
                               'O': 1.52, 'F': 1.47, 'Ne': 1.54, 'Si': 2.10,
                               'P': 1.80, 'S': 1.80, 'Cl': 1.75, 'Ar': 1.88,
                               'As': 1.85, 'Se': 1.90, 'Br': 1.85, 'Kr': 2.02,
                               'Te': 2.06, 'I': 1.98, 'Xe': 2.16}
    # Data with no ref (UNRILIABLE), if employed, a warning will be rised.
    wdw_atomic_radius_no_ref = {'H': 1.20, 'Tl': 1.96, 'He': 1.40, 'Pb': 2.02,
                                'Li': 1.82, 'C': 1.70, 'Pd': 1.63, 'N': 1.55,
                                'Ag': 1.72, 'O': 1.52, 'Cd': 1.58, 'F': 1.47,
                                'In': 1.93, 'Ne': 1.54, 'Sn': 2.17, 'Na': 2.27,
                                'Mn': 1.73, 'Tu': 2.06, 'Ur': 1.86, 'I': 1.98,
                                'Si': 2.10, 'Xe': 2.16, 'P': 1.80, 'S': 1.80,
                                'Cl': 1.75, 'K': 2.75, 'Ni': 1.63, 'Co': 1.40,
                                'Zi': 1.39, 'Ga': 1.87, 'Ar': 1.85, 'Se': 1.90,
                                'Br': 1.85, 'Kr': 2.02, 'Pt': 1.75, 'Mg': 1.55}
    preference_order = [vdw_atomic_radius_bondi,
                        wdw_atomic_radius_no_ref,
                        2.0]

    # Reading input structures
    print('+'+'-'*78+'+')
    print('READING MOLECULES:')
    mol_a = lib.Mol(path=mol_a_path)
    mol_a_name = os.path.basename(mol_a_path)
    mol_a.centralize()
    mol_a.get_radii(preference_order)
    mol_b = lib.Mol(path=mol_b_path)
    mol_b_name = os.path.basename(mol_b_path)
    mol_b.centralize()
    mol_b.get_radii(preference_order)

    # MAPPING SURFACES
    print('+'+'-'*78+'+')
    print('SURFACE MAPPING:')
    mol_a.build_surface(atoms_surface_density=surface_density)
    mol_b.build_surface(atoms_surface_density=surface_density)
    mol_a.to_xyz(mol_a_name + '_surf.xyz', surf_dots=True,
                 surf_dots_color_by='atoms')
    mol_b.to_xyz(mol_b_name + '_surf.xyz', surf_dots=True,
                 surf_dots_color_by='atoms')

    # getting features for the surface dots
    mol_a.featurization_surface_dots()
    mol_b.featurization_surface_dots()

    # surface dots clustering
    mol_a.clustering_surface_dots(name=mol_a_name,
                                  n_cluster=surface_km_mol_a_cluster,
                                  n_repeat=surface_km_mol_a_n_repeat)
    mol_b.clustering_surface_dots(name=mol_b_name,
                                  n_cluster=surface_km_mol_b_cluster,
                                  n_repeat=surface_km_mol_b_n_repeat)
    # surface dots clustering results:
    mol_a.to_xyz(mol_a_name + '_km.xyz', surf_dots=True,
                 surf_dots_color_by='kmeans', special_surf_dots='kmeans')
    mol_b.to_xyz(mol_b_name + '_km.xyz', surf_dots=True,
                 surf_dots_color_by='kmeans', special_surf_dots='kmeans')

    # ADSORPTION
    print('+'+'-'*78+'+')
    print('ADSORPTION:')
    print('Number of configuration: {}'.format(n_config))

    # Defining counters and variables
    c_all = 0
    c_repeated = 0
    c_overlapped = 0
    c_accepted = 0
    filtered_pool = []
    refused_ds = []

    # Iterating over A surface centroids
    for centroid_a in mol_a.surf_dots_km_rep:

        # Iterating over B surface centroids
        for centroid_b in mol_b.surf_dots_km_rep:

            # Iterating over rotations
            c_rot = 0
            for rot in rots:

                # Translating molecules image, centroids -> origin
                mol_a.translate_by(-centroid_a, image=True)
                mol_b.translate_by(-centroid_b, image=True)
                # Rotation image of B
                mol_b.rotate(rot, image=True)

                # Checking overlap, it proced if they do not overlap
                if not lib.overlap(mol_a, mol_b, ovlp_threshold=ovlp_threshold,
                                   image=True):

                    # Creating AB mol object from A and B mol image structures
                    mol_ab = lib.add_mols(mol_a, mol_b, image=True)

                    # Taking references points to compute the features
                    refs = np.array([mol_a.ipositions.mean(axis=0),
                                     mol_b.ipositions.mean(axis=0),
                                     np.zeros(3)])

                    # Computing the features
                    # TODO: isso poderia ser criado uma única vez?
                    mol_ab.features = lib.get_feature(mol_ab,
                                                      reference=refs).flatten()

                    # Checking repetition on the pool of structures
                    repeated = False
                    # Interating over the pool of strutures
                    for a_mol_ab in filtered_pool:
                        # calculating the distance between the current and
                        # already selected structures
                        dist = lib.get_distance(mol_ab.features,
                                                a_mol_ab.features)
                        # Checking if the distance < similarity threshold
                        if dist < sim_threshold:
                            repeated = True
                            break

                    # if the molecule is unprecedented in the filtered_pool
                    if not repeated:
                        # build the molecule AB, with surface info
                        mol_ab_acc = lib.add_mols(
                            mol_a, mol_b, image=True, add_surf_info=True)
                        # TODO: can i just repeat from previous mol_ab?
                        # calculate the features again
                        # mol_ab.features = metric.get_feature(
                        # mol_ab, reference=refs).flatten()
                        mol_ab_acc.features = mol_ab.features * 1.
                        mol_ab_acc.image_to_real_with_surf()
                        filtered_pool.append(mol_ab_acc)

                        c_accepted += 1
                        c_rot += 1
                    else:
                        c_repeated += 1
                        refused_ds.append(dist)
                else:
                    c_overlapped += 1
                c_all += 1

                if (c_all % 10000) == 0:
                    lib.status(c_all, n_config, c_repeated, c_overlapped,
                               c_accepted, refused_ds)
    lib.status(c_all, n_config, c_repeated,
               c_overlapped, c_accepted, refused_ds)

    # Comparing the filtered_pool size with the requested number of structures:
    if len(filtered_pool) > final_n_structures:     # Larger than requested
        perform_final_clustering = True
    elif len(filtered_pool) == final_n_structures:  # Equal to requested
        print(lib.NOTE_EQUAL_TO_REQUESTED.format(len(filtered_pool)))
        perform_final_clustering = False
    elif len(filtered_pool) < final_n_structures:   # Less than requested
        print(lib.NOTE_LESS_THAN_REQUESTED.format(len(filtered_pool), len(
            filtered_pool)/final_n_structures, final_n_structures))
        perform_final_clustering = False

    # Final clustering
    if perform_final_clustering:
        print('-'*80)
        print('Selecting representative structures')
        # organizing data
        # mols_ab = np.array(filtered_pool)
        # features = []
        # for s_mol_ab in mols_ab:
        #     features.append(s_mol_ab.features)
        # features = np.array(features)
        features = np.array([mol_ab.features for mol_ab in filtered_pool])
        # Kmeans analysis, keeping the best model from n_repeat_final_km
        top_score = np.inf
        for seed in range(n_repeat_final_km):
            # creating and evaluating a model
            centroids, score = kmeans(features, final_n_structures, seed=seed)
            # saving the best model
            if score < top_score:
                top_score = score
                top_centroids = centroids

        # taking the representative molecule: the closest molecule to each
        # centroid and the cluster indexes of each molecule
        idx, _ = vq(features, top_centroids)
        dists = cdist(top_centroids, features)
        representative_structures_index = np.argmin(dists, axis=1)
        representative_mols = [filtered_pool[i]
                               for i in representative_structures_index]
        # plotting a tsne dimensionality reduction to shows the clustering
        lib.plot_kmeans_tsne('clustering_representatives' + out_sufix,
                             features, idx, representative_structures_index)

        mol_list = filtered_pool
    else:
        mol_list = representative_mols

    # Saving structures
    print('+'+'-'*78+'+')
    print('SAVING INFORMATION ')

    # Creating folders to save the xyz files, with and without surf data
    path_final_pool_w = 'folder_xyz_files{}_withsurfs/'.format(out_sufix)
    path_final_pool = 'folder_xyz_files{}/'.format(out_sufix)
    for path in [path_final_pool, path_final_pool_w]:
        if not os.path.isdir(path):
            print('creating folder: {}'.format(path))
            os.makedirs(path)
        else:
            # if the folders exist, remove then first
            print('removing folder: {}'.format(path))
            shutil.rmtree(path)
            print('creating folder: {}'.format(path))
            os.makedirs(path)

    # saving molecules to xyz files
    for ith, mol_ab in enumerate(mol_list):
        mol_ab.to_xyz(path_final_pool_w + '/{}.xyz'.format(ith), surf_dots=True,
                      surf_dots_color_by='kmeans', special_surf_dots='kmeans',
                      verbose=False)
        mol_ab.to_xyz(path_final_pool + '/{}.xyz'.format(ith), verbose=False)

    print('+'+'-'*78+'+')


if __name__ == '__main__':
    cluster_adsorption(args.mols[0],
                       int(args.surf_ks[0]),
                       args.mols[1],
                       int(args.surf_ks[1]),
                       final_n_structures=int(args.n_final),
                       n_km_repeat=int(args.n_repeat_km),
                       surface_density=float(args.surf_d),
                       n_rot_r=int(args.n_rots),
                       sim_threshold=float(args.sim_threshold),
                       ovlp_threshold=float(args.ovlp_threshold),
                       out_sufix=args.out_sufix)
