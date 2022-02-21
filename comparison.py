import numpy as np
import scipy as sp
import adsorption_lib as lib
import pandas as pd
import os
import argparse
from scipy.spatial.distance import cdist, pdist
import seaborn as sns
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans

help_text = """    Compare representative sets"""

parser = argparse.ArgumentParser(
    description=help_text, formatter_class=argparse.RawTextHelpFormatter)
parser._action_groups.pop()
required = parser.add_argument_group('required arguments')
optional = parser.add_argument_group('optional arguments')
required.add_argument('--folders', nargs='+', action='store',
                      metavar=('folder_xyz_files_1',
                               'folder_xyz_files_2', '...'),
                      required=True,
                      help='Folders with molecules in xyz format.')
required.add_argument('--subs_ns', nargs='*', action='store',
                      metavar=('n_sub_a', 'n_sub_b', '...'), required=False,
                      help='Split the structure into multiple substructures.'
                      + 'The number of atoms in the molecules must be the sum '
                      + 'of the argurments of subs_ns.')
args = parser.parse_args(('--folders folder_xyz_files_2 '
                          + 'arquivos_ref/Cluster_AD_Pd4O8/folder_xyz_files '
                          + '--subs_ns 3 9'
                          ).split())
# args = parser.parse_args(['--help'])


def sets_comparison(folders, subs_ns):
    """It compare two representative sets"""

    if subs_ns is not None:
        subs_ns = np.array(subs_ns, int)

    print('+'+'-'*78+'+')
    print(' {:^76s} '.format(
        'MOLECULAR SETS COMPARISION ALGORITHM'))
    print('+'+'-'*78+'+')
    left = 25
    print('{:<{}s} '.format('PARAMETERS:', left))
    for ith, folder in enumerate(folders):
        print('{:<{}s} {}'.format('Folder {}'.format(ith), left, folder))
    if subs_ns is not None:
        print(('{:<{}s} ' + '{} '*len(subs_ns)).format(
            'Substructures sizes', left, *subs_ns))

    # reading input structures
    print('+'+'-'*78+'+')
    print('Reading molecules')
    # when subs_ns is employed we must test mol size

    if subs_ns is not None:
        size_accrd_subs = np.sum(subs_ns)
    n_folders = len(folders)
    list_folderes_mols = []
    for ith, folder in enumerate(folders):
        folder_mols = []
        for mol_path in os.listdir(folder):
            mol = lib.Mol(path=folder+'/'+mol_path, verbose=0)
            # comparing molecules sizes with subs_ns
            if subs_ns is not None:
                mol_size = mol.n
                if size_accrd_subs != mol_size:
                    print(("ERROR, substructures indicat {} atoms, but {} were "
                           + "found for {}").format(size_accrd_subs, mol_size,
                                                    folder+'/'+mol_path))
            # adding molecule to the current folder list
            folder_mols.append(mol)
        # adding folder list to the all folder list
        list_folderes_mols.append(folder_mols)

    # getting features for the molecules
    print('Getting features')
    metric = lib.Matric_euclidian_mod()
    all_features = []
    all_foder_indexes = []
    list_folder_features = []
    for folder_index, folder_mols in enumerate(list_folderes_mols):
        folder_features = []
        for mol in folder_mols:
            references = mol.positions.mean(axis=0).reshape(-1, 3)
            if subs_ns is not None:
                subs_n_sum = 0
                for n in subs_ns:
                    new_ref = mol.positions[subs_n_sum:subs_n_sum
                                            + n, :].mean(axis=0).reshape(-1, 3)
                    references = np.append(references, new_ref, axis=0)
                    subs_n_sum += n
            features = metric.get_feature(mol, reference=references).flatten()
            mol.features = features
            all_features.append(features)
            all_foder_indexes.append(folder_index)
            folder_features.append(features)
        list_folder_features.append(np.array(folder_features))
    all_foder_indexes = np.array(all_foder_indexes, int)
    folder_features = np.array(folder_features, float)

    fig, axes = plt.subplots(1, 4)
    data = pd.DataFrame()
    data['features'] = all_features
    data['folder'] = all_foder_indexes

    # t-SNE
    print('t-SNE analysis')
    X = np.vstack(data.features.values)
    features_2d = TSNE(n_components=2, learning_rate='auto',
                       init='random', random_state=2).fit_transform(X)

    for i in range(n_folders):
        i_data_index = data['folder'].values == i
        x = features_2d[i_data_index, 0]
        y = features_2d[i_data_index, 1]
        axes[1].scatter(x=x, y=y, alpha=0.4)

    # distances histogram
    print('Distance histogram')
    pdists = []
    for i in range(n_folders):
        i_data_index = data['folder'].values == i
        i_data = data['features'].values[i_data_index]
        pdists.append(pdist(np.vstack(i_data)))
    pdists = np.array(pdists)

    for pdist_row in pdists:
        axes[0].hist(pdist_row, density=True, bins=20,
                     range=(np.min(pdists), np.max(pdists)), alpha=0.7)

    # getting scores
    print('Clustering sequence')
    scores = []
    step = 10
    ks = np.arange(5, len(i_data)+1, step)
    n_ks = len(ks)
    counter = 0
    for i in range(n_folders):
        i_data_index = data['folder'].values == i
        i_data = np.vstack(data['features'].values[i_data_index])
        scores.append([])
        for kth, k in enumerate(ks):
            km = KMeans(n_clusters=k, n_init=10).fit(i_data)
            scores[-1].append(km.inertia_)
            counter += 1
            if (counter % 10) == 0:
                print('{:2.0%}'.format(counter/(n_ks*n_folders)))
    scores = np.array(scores)
    axes[2].plot(ks, scores[0])
    axes[2].plot(ks, scores[1])
    axes[3].plot(ks, (scores[0]-scores[1])/scores[1])
    axes[3].plot(ks, scores[0]-scores[0])

    plt.show()

    # END
    print('+'+'-'*78+'+')


sets_comparison(args.folders, args.subs_ns)
