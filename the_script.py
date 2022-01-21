
from sys import argv as sys_argv
from os import walk as os_walk
from os import path as os_path
from os import rename as os_rename
from re import search as re_search


def read_inp(file_path, verbose=0):
    # reading all lines of *.inp file
    with open(file_path) as f:
        lines = f.readlines()

    # spliting file into settings_lines and structure_lines componenets
    settings_lines = []
    structure_lines = []
    n_stars = 0
    for line in reversed(lines):  # reading *.inp lines from bottom to top
        line_str_infos = line.split()

        if line_str_infos and line_str_infos[0] == '*':
            n_stars += 1

        if n_stars == 1 and len(line_str_infos) == 4:
            structure_lines.append(line)

        if n_stars == 2 and line_str_infos:
            settings_lines.append(line)

    # rereversing
    structure_lines = list(reversed(structure_lines))
    settings_lines = list(reversed(settings_lines))

    return settings_lines, structure_lines


def read_xyz(path, verbose=0):
    # reading all lines of *.xyz file
    with open(path) as f:
        lines = f.readlines()

    # splitting xyz file into n_atoms, comment_line and structure_liens
    # componenets
    n_atoms = int(lines[0])
    comment_line = lines[1]
    structure_lines = [line for line in lines[2:]]

    return n_atoms, comment_line, structure_lines


def write_inp(file_path, settings_lines, structure_lines, verbose=0):
    # reading all lines of *.inp file
    with open(file_path, mode='w') as f:
        f.writelines(settings_lines)
        f.writelines(structure_lines)
        f.writelines(['*'])


def read_slurm(file_path, verbose=0):
    # reading all lines of slurm-*.out file
    with open(file_path) as f:
        lines = f.readlines()

    # information
    info = {}

    # verifying the convergence
    info['geometry_converged'] = False
    for line in lines:
        if re_search('     THE OPTIMIZATION HAS CONVERGED   ', line):
            info['geometry_converged'] = True
            break

    return info


def analyse_folders(folder_path, verbose=1):
    if not os_path.isdir(folder_path):
        if verbose > 0:
            print("{} is not a folder, exiting...".format(
                folder_path))
        exit()

    for folder_path, folder_folders, folder_files in os_walk(folder_path):
        # if the folder was set to ok status it is not analyzed...
        if folder_path[-3:] == '_ok':
            if verbose > 0:
                print(">> {} folder was already setted to ok status".format(
                    folder_path))
            continue

        # analysing files, if there are no inp and slurms filename
        inp, slurms = None, []
        for file in folder_files:
            # inp
            if re_search('^\w+\.inp$', file):
                inp = re_search('^\w+\.inp$', file).group()
            # xyz
            if re_search('^\w+\.xyz$', file):
                xyz = re_search('^\w+\.xyz$', file).group()
            # slurms
            if re_search('^slurm-\w+', file):
                name = re_search('^slurm-\d+.out', file).group()
                number = int(re_search('slurm-(\d+?).out', file).group(1))
                slurms.append((number, name))

        # if basic files are not present it can not be analyzed...
        if not inp or not xyz or not slurms:
            if verbose > 0:
                print(
                    ">> {} folder miss *.inp, *.xyz, or slurm-*.out files".format(folder_path))
            continue

        # print(folder_path, xyz, inp, slurms)
        # the files here are probably fine and can be analyzed
        if verbose > 0:
            print(">> {} folder is probably fine and will be analyzed".format(
                folder_path))

        # copping structure in xyz to inp
        inp_settings, inp_structure = read_inp(
            folder_path + '/' + inp, verbose=verbose)
        n_atoms, comment, xyz_structure = read_xyz(
            folder_path + '/' + xyz, verbose=verbose)
        write_inp(folder_path + '/' + inp + '_', inp_settings, xyz_structure)
        if verbose > 0:
            print("xyz structure -> inp file")

        # setting calculation folder to ok status
        last_slurm_file = sorted(slurms)[-1][1]
        name = folder_path + '/' + last_slurm_file
        info = read_slurm(name, verbose=verbose)
        if info["geometry_converged"] == True:
            old_path = folder_path.replace('/','')
            new_path = folder_path.replace('/','') + '_ok'
            if verbose > 0:
                print("geometry optimization converged")
            os_rename(old_path, new_path )
            if verbose > 0:
                print("{} -> {}".format(old_path, new_path))


if __name__ == '__main__':
    analyse_folders(sys_argv[1])
