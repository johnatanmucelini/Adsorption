import shutil
import os
import numpy as np
from scipy.spatial.distance import cdist
from scipy.cluster.vq import kmeans, vq
from absorption_lib import *

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


def cluster_adsorption(mol_a_path, mol_a_surf_km_k, mol_b_path,
                       mol_b_surf_km_k, final_n_structures=100, n_repeat_km=20, surface_density=4, rot_mesh_size=0.8):
    """It build adsorbed structures between two molecules, mol_a and mol_b.
    Both molecules surface are maped based in a """

    # parameters:
    surface_km_mol_a_cluster = mol_a_surf_km_k
    surface_km_mol_b_cluster = mol_b_surf_km_k
    n_repeat_final_km = n_repeat_km
    surface_km_mol_a_n_repeat = n_repeat_km
    surface_km_mol_b_n_repeat = n_repeat_km

    # reading input structures
    mol_a = Mol(path=mol_a_path)
    mol_a.centralize()
    mol_b = Mol(path=mol_b_path)
    mol_b.centralize()

    # getting raddii
    # TODO: interpretation of the atoms: C sp3 , C sp2, =O, -O, H, N
    mol_a.get_radii()  #
    mol_b.get_radii()

    # mapping mol_a and mol_b surfaces
    mol_a.build_surface(atoms_surface_density=surface_density)
    mol_b.build_surface(atoms_surface_density=surface_density)
    mol_a.to_xyz('mol_a_surf.xyz', surf_dots=True,
                 surf_dots_color_by='atoms')
    mol_b.to_xyz('mol_b_surf.xyz', surf_dots=True,
                 surf_dots_color_by='atoms')

    # getting features for the surface dots
    metric = Matric_euclidian_mod()
    mol_a.featurization_surface_dots(metric)
    mol_b.featurization_surface_dots(metric)

    # Kmeans
    mol_a.clustering_surface_dots(
        n_cluster=surface_km_mol_a_cluster, n_repeat=surface_km_mol_a_n_repeat)
    mol_b.clustering_surface_dots(
        n_cluster=surface_km_mol_b_cluster, n_repeat=surface_km_mol_b_n_repeat)
    mol_a.to_xyz('mol_a_km.xyz', surf_dots=True,
                 surf_dots_color_by='kmeans', special_surf_dots='kmeans')
    mol_b.to_xyz('mol_a_km.xyz', surf_dots=True,
                 surf_dots_color_by='kmeans', special_surf_dots='kmeans')

    # calculation all rotations around each contact between mols a and b
    us1 = 2*np.pi*1
    us2 = 4*np.pi*1
    n_samples_spher = int(round(us1/rot_mesh_size))
    n_samples_circ = int(round(us2/rot_mesh_size**2))
    s2_coords = build_s2_grid(1, n_samples_spher, coords_system='spher')
    s1_coords = build_s1_grid(1, n_samples_circ, coords_system='circ')
    rots = build_SO3_from_S1S2(s1_coords, s2_coords)
    n_rots = len(s2_coords)*len(s1_coords)
    print("Number of rotations: {} ({}, {})".format(
        n_rots, len(s2_coords), len(s1_coords)))

    c_all = 0
    c_repeated = 0
    c_overlapped = 0
    c_accepted = 0
    selected_mols_ab = []
    refused_ds = []
    threshold = 1

    print('Total number of trial configuration: {}'.format(
        n_rots*len(mol_a.surf_dots_km_rep)*len(mol_b.surf_dots_km_rep)))

    for ith_a, centroid_a in enumerate(mol_a.surf_dots_km_rep):
        for jth_b, centroid_b in enumerate(mol_b.surf_dots_km_rep):
            for kth, rot in enumerate(rots):
                #print(ith_a, jth_b, centroid_a, centroid_b)
                mol_a.translate_by(-centroid_a, image=True)
                mol_b.translate_by(-centroid_b, image=True)
                mol_b.rotate(rot, image=True)

                if not overlap(mol_a, mol_b, image=True):
                    mol_ab = add_mols(mol_a, mol_b, image=True)
                    refs = np.array([mol_a.ipositions.mean(
                        axis=0), mol_b.ipositions.mean(axis=0), np.zeros(3)])
                    metric = Matric_euclidian_mod()
                    mol_ab.features = metric.get_feature(
                        mol_ab, reference=refs).flatten()
                    repeated = False
                    for nth, s_mol_ab in enumerate(selected_mols_ab):
                        d = metric.get_distance(
                            mol_ab.features, s_mol_ab.features)
                        if d < threshold:
                            repeated = True
                            break
                    if not repeated:
                        mol_ab = add_mols(
                            mol_a, mol_b, image=True, add_surf_info=True)
                        mol_ab.features = metric.get_feature(
                            mol_ab, reference=refs).flatten()
                        mol_ab.surf_to_real()
                        selected_mols_ab.append(mol_ab)
                        c_accepted += 1

                    else:
                        c_repeated += 1
                        refused_ds.append(d)
                else:
                    c_overlapped += 1

                c_all += 1
                if (c_all % 10000) == 0:
                    status(c_all, c_repeated, c_overlapped,
                           c_accepted, refused_ds)
        #     break
        # break
    status(c_all, c_repeated, c_overlapped, c_accepted, refused_ds)

    # final clustering
    print('Clustering to selec representative structures.')
    mols_ab = np.array(selected_mols_ab)
    features = []
    for nth, s_mol_ab in enumerate(mols_ab):
        features.append(s_mol_ab.features)
    features = np.array(features)
    top_score = 1e20
    for seed in range(n_repeat_final_km):
        centroids, score = kmeans(features, final_n_structures, seed=seed)
        if score < top_score:
            top_score = score
            top_centroids = centroids
    idx, _ = vq(features, top_centroids)
    dists = cdist(top_centroids, features)
    representative_structures_index = np.argmin(dists, axis=1)
    representative_mols = mols_ab[representative_structures_index]

    # saving structures
    sufix = ''
    path_poll_w = 'C:\\Users\\User\\Documents\\GitHub\\lucas_script\\folder_xyz_files_withsurfs{}'.format(
        sufix)
    path_poll = 'C:\\Users\\User\\Documents\\GitHub\\lucas_script\\folder_xyz_files{}'.format(
        sufix)
    # path_poll_w = '/home/acer/lucas_script/poll_withsurfs/'
    # path_poll = '/home/acer/lucas_script/poll/'
    paths = [path_poll, path_poll_w]
    for path in paths:
        if not os.path.isdir(path):
            print('creating folder: {}'.format(path))
            os.makedirs(path)
        else:
            print('removing folder: {}'.format(path))
            shutil.rmtree(path)
            print('creating folder: {}'.format(path))
            os.makedirs(path)
    for ith, mol_ab in enumerate(representative_mols):
        mol_ab.to_xyz(path_poll + '\\{}.xyz'.format(ith), surf_dots=True, surf_dots_color_by='kmeans',
                      special_surf_dots='kmeans', verbose=False)
        mol_ab.to_xyz(path_poll_w + '\\{}.xyz'.format(ith), verbose=False)


path_cluster = 'arquivos_ref/Cluster_AD_Pd4O8/cluster.xyz'
path_mol = 'arquivos_ref/Cluster_AD_Pd4O8/molecule.xyz'
#path_cluster = 'C:\\Users\\User\\Documents\\GitHub\\lucas_script\\arquivos_ref\\Cluster_AD_Pd4O8\\cluster.xyz'
#path_mol = 'C:\\Users\\User\\Documents\\GitHub\\lucas_script\\arquivos_ref\\Cluster_AD_Pd4O8\\molecule.xyz'

cluster_adsorption(path_cluster, 15, path_mol, 5, 100)

# def comparacao_novo_velho(
#         # inputs
#         folder de todos os xyz encontrados novo
#         folder de todos os xyz encontrados velho):
#     - pega os resultados dos metodos
#     - t-SNE map pra reduzir a dimensionalidade
#     - comparação das distribuições
#     - para todas as moléculas que forem encontradas
#     - para todas as moléculas selecionadas
