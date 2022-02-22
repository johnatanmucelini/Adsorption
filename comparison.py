import os
import argparse
import numpy as np
from scipy.spatial.distance import pdist
from scipy.cluster.vq import kmeans
import pandas as pd
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import adsorption_lib as lib

help_text = """    Compare representative sets"""

parser = argparse.ArgumentParser(
    description=help_text, formatter_class=argparse.RawTextHelpFormatter)
parser._action_groups.pop()
required = parser.add_argument_group('required arguments')
optional = parser.add_argument_group('optional arguments')
required.add_argument('--folders', nargs='*', action='store',
                      metavar=('folderA', 'folderB'),
                      required=True,
                      help='Folders with molecules in xyz format.')
optional.add_argument('--subs_ns', nargs='*', action='store',
                      metavar=('n_sub_a', 'n_sub_b'), required=False,
                      help='Split the structure into multiple substructures.\n'
                      + 'The number of atoms in the molecules must be the \n'
                      + 'sum of the argurments of subs_ns.')
args = parser.parse_args(('--folders folder_xyz_files_1 folder_xyz_files_2 '
                          + 'folder_xyz_files_3 '
                          + '--subs_ns 3 9'
                          ).split())
# args = parser.parse_args(['--help'])


def sets_comparison(folders, subs_ns):
    """It compare two representative sets"""

    if subs_ns is not None:
        subs_ns = np.array(subs_ns, int)

    print('+'+'-'*78+'+')
    print(' {:^78s} '.format(
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
    n_mim_mol = 1e10
    for ith, folder in enumerate(folders):
        # print('folder {}'.format(ith))
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
        if len(folder_mols) < n_mim_mol:
            n_mim_mol = len(folder_mols)
        # adding folder list to the all folder list
        list_folderes_mols.append(folder_mols)

    # getting features for the molecules
    print('Getting features')
    metric = lib.Matric_euclidian_mod()
    all_features = []
    all_foder_indexes = []
    list_folder_features = []
    for folder_index, folder_mols in enumerate(list_folderes_mols):
        # print('folder {}'.format(folder_index))
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

    print('+'+'-'*78+'+')
    print('Data analysis:')
    fig, axes = plt.subplots(2, 2, figsize=(8, 6), dpi=240)
    axes = axes.flatten()
    data = pd.DataFrame()
    data['features'] = all_features
    data['folder'] = all_foder_indexes

    # Distance histogram
    print('    Distance histogram')
    pdists_list = []
    for i in range(n_folders):
        # print('folder {}'.format(i))
        i_data_index = data['folder'].values == i
        i_data = np.vstack(data['features'].iloc[i_data_index])
        pdists_list.append(pdist(i_data))
    min_val = np.min([np.min(p) for p in pdists_list])
    max_val = np.max([np.max(p) for p in pdists_list])
    for pdist_row, name in zip(pdists_list, folders):
        # print(pdist_row.shape, pdist_row)
        axes[0].hist(pdist_row, density=True, bins=20,
                     range=(min_val, max_val), alpha=0.4, label=name,
                     zorder=i+2)
        axes[0].legend()
    axes[0].set_xlabel('distances')
    axes[0].set_ylabel('distribution density')

    # t-SNE
    print('    t-SNE dimensionality reduction')
    X = np.vstack(data.features.values)
    features_2d = TSNE(n_components=2, learning_rate='auto',
                       init='random', random_state=2).fit_transform(X)
    for i in range(n_folders):
        # print('folder {}'.format(i))
        i_data_index = data['folder'].values == i
        x = features_2d[i_data_index, 0]
        y = features_2d[i_data_index, 1]
        axes[1].scatter(x=x, y=y, alpha=0.2, zorder=i+2)
    axes[1].set_xlabel('t-SNE coord 1')
    axes[1].set_ylabel('t-SNE coord 2')

    # Clustering sequence
    print('    Clustering sequence')
    scores = []
    counter = 0
    n_repeat = 2
    nstep = min((20, n_mim_mol))
    n_ks_sum = nstep*n_folders
    ks_fraction = np.arange(1/nstep, 1+1/nstep, 1/nstep)
    for i in range(n_folders):
        # print('folder {}'.format(i))
        i_data_index = data['folder'].values == i
        i_data = np.vstack(data['features'].iloc[i_data_index])
        scores.append([])
        scale = len(i_data)
        ks = np.array(np.round(ks_fraction * scale), float)
        for k in ks:
            top_score = 1e20
            for seed in range(n_repeat):
                _, score = kmeans(i_data, k, seed=seed)
                if score < top_score:
                    top_score = score
                    # top_centroids = centroids
            # idx, _ = vq(i_data, top_centroids)
            # dists = cdist(top_centroids, i_data)
            # centroids_nearst_idx = np.argmin(dists, axis=1)

            scores[-1].append(top_score)
            counter += 1
            if (counter % 10) == 0:
                print('        {:2.0%}'.format(counter/n_ks_sum))
    scores = np.array(scores)
    for i in range(n_folders):
        # print('ploting {}'.format(i))
        axes[2].plot(ks_fraction, scores[i], zorder=i+2)
        axes[3].plot(ks_fraction, (scores[i]-scores[-1])
                     / scores[-1], zorder=i+2)
    axes[2].set_xlabel('Nsamples/K')
    axes[2].set_ylabel('Kmeans score')
    axes[3].set_xlabel('Nsamples/K')
    axes[3].set_ylabel('Kmeans score - {}'.format(name))

    # savin info
    print('+'+'-'*78+'+')
    fig.set_tight_layout(True)
    figure_name = 'result_comparison.png'
    print('Writing results to: {}'.format(figure_name))
    fig.savefig(figure_name)

    # END
    print('+'+'-'*78+'+')


sets_comparison(args.folders, args.subs_ns)
