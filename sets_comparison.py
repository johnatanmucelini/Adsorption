import numpy as np
import scipy as sp
import absorption_lib as lib
import pandas as pd
import os
import argparse
from scipy.spatial.distance import cdist, pdist

help_text = """    Compare representative sets"""

parser = argparse.ArgumentParser(
    description=help_text, formatter_class=argparse.RawTextHelpFormatter)
parser._action_groups.pop()
required = parser.add_argument_group('required arguments')
optional = parser.add_argument_group('optional arguments')
required.add_argument('--folders', nargs='+', action='store',
                      metavar=('folder_xyz_files_n',
                               'folder_xyz_files_2', '...'),
                      required=True,
                      help='Folders with molecules in xyz format.')
required.add_argument('--subs_ns', nargs='*', action='store',
                      metavar=('n_sub_a', 'n_sub_b', '...'), required=False,
                      help='Split the structure into multiple substructures.'
                      + 'The number of atoms in the molecules must be the sum '
                      + 'of the argurments of subs_ns.')
args = parser.parse_args(('--folders folder_xyz_files_1 '
                          + 'arquivos_ref/Cluster_AD_Pd4O8/folder_xyz_files '
                          + '--subs_ns 3 9'
                          ).split())
# args = parser.parse_args(['--help'])


def sets_comparison(folders, subs_ns):
    """It compare two representative sets"""

    print('+'+'-'*78+'+')
    print(' {:^76s} '.format(
        'MOLECULAR SETS COMPARISION ALGORITHM'))
    print('+'+'-'*78+'+')
    left = 25
    print('{:<{}s} '.format('PARAMETERS:', left))
    for ith, folder in enumerate(folders):
        print('{:<{}s} {}'.format('Folder {}'.format(ith), left, folder))
    if subs_ns:
        print(('{:<{}s} ' + '{} '*len(subs_ns)).format(
            'Substructures sizes', left, *subs_ns))

    # reading input structures
    print('+'+'-'*78+'+')
    print('Reading molecules')
    # when subs_ns is employed we must test mol size

    if subs_ns:
        size_by_subs = np.sum(np.array(subs_ns, int))
    n_folders = len(folders)
    all_mols = []
    for ith, folder in enumerate(folders):
        folder_mols = []
        for mol_path in os.listdir(folder):
            mol = lib.Mol(path=folder+'/'+mol_path, verbose=0)
            if subs_ns:
                mol_size = mol.n
                if size_by_subs != mol_size:
                    print(("ERROR, substructures indicat {} atoms, but {} were "
                           + "found for {}").format(size_by_subs, mol_size,
                                                    folder+'/'+mol_path))
            folder_mols.append(mol)

        all_mols.append(folder_mols)

    # getting features for the molecules
    print('Getting features')
    metric = lib.Matric_euclidian_mod()
    features_data = []
    folder_index_data = []
    for folder_index, folder_mols in enumerate(all_mols):
        folder_features = []
        for mol in folder_mols:
            references = mol.positions.mean(axis=0).reshape(-1, 3)
            if subs_ns:
                ns_sum = 0
                for n in np.array(subs_ns, int):
                    new_ref = mol.positions[ns_sum:ns_sum
                                            + n, :].mean(axis=0).reshape(-1, 3)
                    references = np.append(references, new_ref, axis=0)
                    ns_sum += n
            features = metric.get_feature(mol, reference=references).flatten()
            mol.features = features
            features_data.append(features)
            folder_index_data.append(folder_index)

    # comparação variança das distribuições
    data = pd.DataFrame()
    data['features'] = features_data
    data['folder'] = folder_index_data

    # t-SNE
    from sklearn.manifold import TSNE
    X = np.vstack(data.features.values)
    features_2d = TSNE(n_components=2, learning_rate='auto',
                       init='random', random_state=2).fit_transform(X)

    # analysis
    import seaborn as sns
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 2)

    pdists = []
    for i in range(n_folders):
        i_data_index = data['folder'].values == i
        i_data = data['features'].values[i_data_index]
        pdists.append(pdist(np.vstack(i_data)))
    pdists = np.array(pdists)

    for pdist_row in pdists:
        axes[0].hist(pdist_row, density=True, bins=20,
                     range=(np.min(pdists), np.max(pdists)), alpha=0.7)
        print(np.mean(pdist_row))

    for i in range(n_folders):
        i_data_index = data['folder'].values == i
        x = features_2d[i_data_index, 0]
        y = features_2d[i_data_index, 1]
        axes[1].scatter(x=x, y=y, alpha=0.4)
    plt.show()

    # END
    print('+'+'-'*78+'+')


sets_comparison(args.folders, args.subs_ns)
