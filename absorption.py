import shutil
import os
import argparse
from scipy.spatial.distance import cdist
from scipy.cluster.vq import kmeans
import numpy as np
import absorption_lib as lib

help_text = """    MOLECULAR ADSORPTION BY SURFACE MAPPING ALGORITHM

    This script find representative structures for the adsorption between two
molecules:
    (1) Read the mol and associate a vdw radii for each atom:
        There are vdw radii for some atoms and its reference, but other can be
        added manualy, search for "VDW RADII AND ITS REF" in this document.
    (2) Both molecules surfaces are mapped to find representative point:
        The surface of a molecule is a outside surface built with the union of
        ridge spheres of vdw radii around each atom. The files mol_a_surf.xyz
        and mol_b_surf.xyz present this data [a].
        Points in these spheres (SO2) are obtained with the algorithm described
        by Deserno (see the article "How to generate equidistributed points on
        the surface of a sphere" [b]).
        Then, for each point of the surface, features are extracted. The
        features vector contain the sorted distaces from the point to each atom
        of the molecule, saparataded by the atoms chemical element.
        Based in a K-means clustering, the surface dots are clusters/groups,
        and the point nearest to the centroind of its clusters is selected as
        its representative point in the molecular surface. The files
        mol_a_km.xyz and mol_b_km.xyz present this data [a].
        [a] The structure with the surface dots can be see in VESTA code. To
            correct read they data, you must replace the VESTA configuration
            file elements.ini by the elements.ini file added in the project.
            These file present the types of atoms, colors, and other properties
            to automatically add colors to the representation. In the surf.xyz
            files present the surface dots colorized for each atoms associated
            to the surface point. The km.xyz files present the surface dots
            colorized for each clusters of surface dots.
        [b] https://www.cmu.edu/biolphys/deserno/pdf/sphere_equi.pdf
    code. You can also read it with the surface dots in colors:
    (3) Adsorption are performed combining both molecules by each pair of
        representative point of its surfaces. Also, for each pair or
        representative points, many rotations are performed to garantee that
        good matches between the molecules.
        These rotations are performed with a grid of rotations of SO3, obtained
        with a method called Successive Orthogonal Images on SOn. The method
        was first presented by Mitchell (DOI:10.1137/030601879), but in my
        implementation I followed the paper by Yershova
        (DOI:10.1177/0278364909352700).
        Thus, the number of adsorbed molecules configurations to analyze is
        deterministic, and is product of the number of clusters in the surface
        of each molecule and the number of rotations in SO3.
        The configurations are added to a pull of structures, if: (i) the
        molecules did not overlapped in the adsorption, which is considered
        when a pair of atoms of different molecules were distant by less than
        the sum of vdw radii of the atoms multiplied by the parameter
        ovlp_threshold; (ii) the present structures is considerably differnt
        from the other structures in the pull. To check it, we build a feature
        vector with the sorted distaces between three key points and each atom,
        separated by atom type and key point. The key points are the
        geometrical center of each molecule and the position of the
        representative surface dots that were employed to create the present
        configuration. If the euclidian distance between the present
        configuration and all other structures in the pull were smaller than
        sim_threshold parameter.
    (4) Representative structures are sampled from the poll with another
        K-means, and the structures are writed in folder_xyz_files (adsorbed
        sturctures) and folder_xyz_files_withsurfs (adsorbed structures with
        surface information).

    Usage example:
    $ python absorption.py --mols arquivos_ref/Cluster_AD_Pd4O8/cluster.xyz
                                  arquivos_ref/Cluster_AD_Pd4O8/molecule.xyz
                           --surf_ks       15 5
                           --n_final        100
                           --surf_d           5
                           --n_repeat_km     20
                           --rot_mesh_size  0.8
                           --ovlp_threshold 0.8
                           --sim_threshold  1.0
                           --out_sufix       _1
"""

# TODO: Metrica tem 2 problemas: 1) tem que usar o inverso da distância, não a
# distância. 2) tem que definir um máximo de 1. pra caso algum átomo esteja
# muito perto.
# TODO: testar o comando do help


parser = argparse.ArgumentParser(
    description=help_text, formatter_class=argparse.RawTextHelpFormatter)
parser._action_groups.pop()
required = parser.add_argument_group('required arguments')
optional = parser.add_argument_group('optional arguments')
required.add_argument('--mols', nargs=2, action='store',
                      metavar=('a.xyz', 'b.xyz'), required=True,
                      help='The two molecules to adsorb.')
required.add_argument('--surf_ks', nargs=2, action='store',
                      metavar=('K_a', 'K_b'), required=True,
                      help='Numbers k (kmeans) for each surface dots '
                      + 'clustering')
required.add_argument('--n_final', nargs=None, action='store',
                      metavar='N_final', required=True,
                      help='Number of final structures, representative set.')
optional.add_argument('--surf_d', nargs=None, action='store', metavar='val',
                      default=5,
                      help='Density of points over the atoms. (default: 5, '
                      + 'unity: 4.pi AA^(-2))')
optional.add_argument('--n_repeat_km', nargs=None, action='store',
                      metavar='val', default=20,
                      help='Number of times to repeat each clustering. '
                      + '(default: 20)')
optional.add_argument('--rot_mesh_size', nargs=None, action='store',
                      metavar='val', default=0.8,
                      help='Size of the rotations mesh (default: 0.8, change '
                      + 'it slowly)')
optional.add_argument('--ovlp_threshold', nargs=None, action='store',
                      metavar='val', default=0.8,
                      help='Structures overlap threshold (default: 0.8)')
optional.add_argument('--sim_threshold', nargs=None, action='store',
                      metavar='val', default=1.,
                      help='Structures similarity threshold (default: 1)')
optional.add_argument('--out_sufix', nargs=None, action='store',
                      metavar='sufix', default='',
                      help='Sufix of the output folders: '
                      + 'folder_xyz_files+surfix and '
                      + 'folder_xyz_files_withsurfs+surfix (default: None)')
# args = parser.parse_args(('--mols arquivos_ref/Cluster_AD_Pd4O8/cluster.xyz '
#                           + 'arquivos_ref/Cluster_AD_Pd4O8/molecule.xyz '
#                           + '--surf_ks 15 5 --n_final 100 --surf_d 5 '
#                           + '--n_repeat_km 20 --rot_mesh_size 0.8 '
#                           + '--ovlp_threshold 0.8 --sim_threshold 1.0 '
#                           + '--out_sufix _1'
#                           ).split())
args = parser.parse_args(['--help'])

# sim_threshold=float(args.sim_threshold),
# ovlp_threshold=float(args.ovlp_threshold),


def cluster_adsorption(mol_a_path, mol_a_surf_km_k, mol_b_path,
                       mol_b_surf_km_k, final_n_structures=100, n_km_repeat=20,
                       surface_density=4, rot_mesh_size=0.8, sim_threshold=1,
                       ovlp_threshold=0.85, out_sufix=''):
    """It build adsorbed structures between two molecules, mol_a and mol_b.
    Both molecules surface are maped based in a """

    # parameters:
    surface_km_mol_a_cluster = mol_a_surf_km_k
    surface_km_mol_b_cluster = mol_b_surf_km_k
    n_repeat_final_km = n_km_repeat
    surface_km_mol_a_n_repeat = n_km_repeat
    surface_km_mol_b_n_repeat = n_km_repeat

    # preprocessing all rotations matri
    n_samples_spher = int(round(2*np.pi*1/rot_mesh_size))
    n_samples_circ = int(round(4*np.pi*1/rot_mesh_size**2))
    s2_coords = lib.build_s2_grid(1, n_samples_spher, coords_system='spher')
    s1_coords = lib.build_s1_grid(1, n_samples_circ, coords_system='circ')
    rots = lib.build_SO3_from_S1S2(s1_coords, s2_coords)
    n_rots_s1 = len(s1_coords)
    n_rots_s2 = len(s2_coords)
    n_rots = n_rots_s1 * n_rots_s2
    n_config = n_rots * surface_km_mol_a_cluster * surface_km_mol_b_cluster

    print('+'+'-'*78+'+')
    print(' {:^76s} '.format(
        'MOLECULAR ADSORPTION BY SURFACE MAPPING ALGORITHM'))
    print('+'+'-'*78+'+')
    left = 25
    print('{:<{}s} '.format('PARAMETERS AND INFO:', left))
    print('{:<{}s} {}'.format('Mol A', left, mol_a_path))
    print('{:<{}s} {}'.format('Mol B', left, mol_b_path))
    print('{:<{}s} {}'.format('N kmeans repetition', left, n_km_repeat))
    print('{:<{}s} {} {}'.format('Surface mapping density',
                                 left, surface_density, '[4piAA^{-2}]'))
    print('{:<{}s} {}'.format('N cluster surface A', left, mol_a_surf_km_k))
    print('{:<{}s} {}'.format('N cluster surface B', left, mol_b_surf_km_k))
    print('{:<{}s} {}'.format('Rotational mesh size', left, rot_mesh_size))
    print('{:<{}s} {}'.format('Simility threshold', left, sim_threshold))
    print('{:<{}s} {}'.format('Overlatp threshold', left, ovlp_threshold))
    print('{:<{}s} {}'.format('N rotations S1', left, n_rots_s1))
    print('{:<{}s} {}'.format('N rotations S2', left, n_rots_s2))
    print('{:<{}s} {}'.format('N rotations total', left, n_rots))
    print('{:<{}s} {}'.format('N configurations', left, n_config))
    print('{:<{}s} {}'.format('N final structure', left, final_n_structures))

    # VDW RADII AND ITS REF:
    # Add or edit vdw radii here, see the example bellow:
    # ref: THE JOURNAL OF PHYSICAL CHEMISTRY, 68, 3, 1964
    # In agrement with J. Phys. Chem., Vol. 100, No. 18, 1996
    vdw_atomic_radius_bondi = {'H': 1.20, 'He': 1.40, 'C': 1.70, 'N': 1.55,
                               'O': 1.52, 'F': 1.47, 'Ne': 1.54, 'Si': 2.10,
                               'P': 1.80, 'S': 1.80, 'Cl': 1.75, 'Ar': 1.88,
                               'As': 1.85, 'Se': 1.90, 'Br': 1.85, 'Kr': 2.02,
                               'Te': 2.06, 'I': 1.98, 'Xe': 2.16,
                               'ref': 'J. Physical Chemistry, 68, 3, 1964'}
    # Data with no ref (do not trust it), if employed an warning will raise.
    wdw_atomic_radius_net = {'H': 1.20, 'Tl': 1.96, 'He': 1.40, 'Pb': 2.02,
                             'Li': 1.82, 'C': 1.70, 'Pd': 1.63, 'N': 1.55,
                             'Ag': 1.72, 'O': 1.52, 'Cd': 1.58, 'F': 1.47,
                             'In': 1.93, 'Ne': 1.54, 'Sn': 2.17, 'Na': 2.27,
                             'Mn': 1.73, 'Tu': 2.06, 'Ur': 1.86, 'I': 1.98,
                             'Si': 2.10, 'Xe': 2.16, 'P': 1.80, 'S': 1.80,
                             'Cl': 1.75, 'K': 2.75, 'Ni': 1.63, 'Co': 1.40,
                             'Zi': 1.39, 'Ga': 1.87, 'Ar': 1.85, 'Se': 1.90,
                             'Br': 1.85, 'Kr': 2.02, 'Pt': 1.75, 'Mg': 1.55}
    preference_order = [vdw_atomic_radius_bondi,
                        wdw_atomic_radius_net,
                        2]
    # reading input structures
    print('+'+'-'*78+'+')
    print('READING MOLECULES:')
    mol_a = lib.Mol(path=mol_a_path)
    mol_a.centralize()
    mol_a.get_radii(preference_order)
    mol_b = lib.Mol(path=mol_b_path)
    mol_b.centralize()
    mol_b.get_radii(preference_order)

    # mapping mol_a and mol_b surfaces
    print('+'+'-'*78+'+')
    print('SURFACE MAPPING:')
    mol_a.build_surface(atoms_surface_density=surface_density)
    mol_b.build_surface(atoms_surface_density=surface_density)
    mol_a.to_xyz('mol_a_surf.xyz', surf_dots=True,
                 surf_dots_color_by='atoms')
    mol_b.to_xyz('mol_b_surf.xyz', surf_dots=True,
                 surf_dots_color_by='atoms')

    # getting features for the surface dots
    metric = lib.Matric_euclidian_mod()
    mol_a.featurization_surface_dots(metric)
    mol_b.featurization_surface_dots(metric)

    # surface dots Kmeans
    mol_a.clustering_surface_dots(
        n_cluster=surface_km_mol_a_cluster, n_repeat=surface_km_mol_a_n_repeat)
    mol_b.clustering_surface_dots(
        n_cluster=surface_km_mol_b_cluster, n_repeat=surface_km_mol_b_n_repeat)
    print('Surface clustering results:')
    mol_a.to_xyz('mol_a_km.xyz', surf_dots=True,
                 surf_dots_color_by='kmeans', special_surf_dots='kmeans')
    mol_b.to_xyz('mol_a_km.xyz', surf_dots=True,
                 surf_dots_color_by='kmeans', special_surf_dots='kmeans')

    # ADSORPTION
    print('+'+'-'*78+'+')
    print('ADSORPTION:')

    c_all = 0
    c_repeated = 0
    c_overlapped = 0
    c_accepted = 0
    selected_mols_ab = []
    refused_ds = []

    print('Number of configuration: {}'.format(n_config))

    for centroid_a in mol_a.surf_dots_km_rep:
        for centroid_b in mol_b.surf_dots_km_rep:
            for rot in rots:
                mol_a.translate_by(-centroid_a, image=True)
                mol_b.translate_by(-centroid_b, image=True)
                mol_b.rotate(rot, image=True)

                if not lib.overlap(mol_a, mol_b, ovlp_threshold=ovlp_threshold,
                                   image=True):
                    mol_ab = lib.add_mols(mol_a, mol_b, image=True)
                    refs = np.array([mol_a.ipositions.mean(
                        axis=0), mol_b.ipositions.mean(axis=0), np.zeros(3)])
                    metric = lib.Matric_euclidian_mod()
                    mol_ab.features = metric.get_feature(
                        mol_ab, reference=refs).flatten()
                    repeated = False
                    for s_mol_ab in selected_mols_ab:
                        dist = metric.get_distance(
                            mol_ab.features, s_mol_ab.features)
                        if dist < sim_threshold:
                            repeated = True
                            break
                    if not repeated:
                        mol_ab = lib.add_mols(
                            mol_a, mol_b, image=True, add_surf_info=True)
                        mol_ab.features = metric.get_feature(
                            mol_ab, reference=refs).flatten()
                        mol_ab.surf_to_real()
                        selected_mols_ab.append(mol_ab)
                        c_accepted += 1

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

    # final clustering
    print('-'*80)
    print('Selecting representative structures')
    mols_ab = np.array(selected_mols_ab)
    features = []
    for s_mol_ab in mols_ab:
        features.append(s_mol_ab.features)
    features = np.array(features)
    top_score = 1e20
    for seed in range(n_repeat_final_km):
        centroids, score = kmeans(features, final_n_structures, seed=seed)
        if score < top_score:
            top_score = score
            top_centroids = centroids
    # idx, _ = vq(features, top_centroids)
    dists = cdist(top_centroids, features)
    representative_structures_index = np.argmin(dists, axis=1)
    representative_mols = mols_ab[representative_structures_index]

    print('+'+'-'*78+'+')
    print('SAVING INFORMATION ')
    # saving structures
    path_poll_w = 'folder_xyz_files{}_withsurfs/'.format(out_sufix)
    path_poll = 'folder_xyz_files{}/'.format(out_sufix)
    for path in [path_poll, path_poll_w]:
        if not os.path.isdir(path):
            print('creating folder: {}'.format(path))
            os.makedirs(path)
        else:
            print('removing folder: {}'.format(path))
            shutil.rmtree(path)
            print('creating folder: {}'.format(path))
            os.makedirs(path)
    for ith, mol_ab in enumerate(representative_mols):
        mol_ab.to_xyz(path_poll + '/{}.xyz'.format(ith), surf_dots=True,
                      surf_dots_color_by='kmeans', special_surf_dots='kmeans',
                      verbose=False)
        mol_ab.to_xyz(path_poll_w + '/{}.xyz'.format(ith), verbose=False)

    # END
    print('+'+'-'*78+'+')


# os.chdir('/home/acer/lucas_script/')
os.chdir('C:\\Users\\User\\Documents\\GitHub\\lucas_script\\')

cluster_adsorption(args.mols[0],
                   int(args.surf_ks[0]),
                   args.mols[1],
                   int(args.surf_ks[1]),
                   final_n_structures=int(args.n_final),
                   n_km_repeat=int(args.n_repeat_km),
                   surface_density=float(args.surf_d),
                   rot_mesh_size=float(args.rot_mesh_size),
                   sim_threshold=float(args.sim_threshold),
                   ovlp_threshold=float(args.ovlp_threshold),
                   out_sufix=args.out_sufix)

# def comparacao_novo_velho(
#         # inputs
#         folder de todos os xyz encontrados novo
#         folder de todos os xyz encontrados velho):
#     - pega os resultados dos metodos
#     - t-SNE map pra reduzir a dimensionalidade
#     - comparação das distribuições
#     - para todas as moléculas que forem encontradas
#     - para todas as moléculas selecionadas
#     - para todas as moléculas selecionadas
