# MOLECULAR ADSORPTION BY SURFACE MAPPING ALGORITHM

The present algorithm generates sets of atomic structures of adsorbed molecules, considering ridge structures and atoms as spheres of VDW radius (or a fraction of it).

| A.xyz ![](.figures/cluster.png)  | **+** | B.xyz ![](.figures/molecule.png) | **=** | AB_1.xyz ![](.figures/99.png)  AB_2.xyz ![](.figures/97.png) ... |
|----------------------------------|-------|----------------------------------|-----|------------------------------------------------------------------|

---

## Methodology

Our methods mimic the ideia that two melecules could interact based on different chemical environments on the surface of the molecules. First, it get a representative sets of chemical environments of each one of the molecules. Second, we find structures combining the molecules through each possible pair of chemical environments in the molecules. Finally, this poll of structures is sampled to find a representative set of thepossible ways of interaction.

#### Surface Mapping

The objective of this step is to get a set of K points on the surface of each molecule, these points must represent the diversity of different chemical environments around the molecule:

- Read the mol and associate VDW radii for each atom:
    There are VDW radii for some atoms and their reference, but others can be
    added manually, search for "VDW RADII AND ITS REF" in this document.

- Both molecule surfaces are mapped with dots:
    The surface of a molecule is an outside surface built with the union of  ridge spheres of VDW radii around each atom. The files mol_a_surf.xyz and mol_b_surf.xyz present this data [a]. Points in these spheres (SO2) are obtained with the algorithm describedby Deserno (see the article "How to generate equidistributed points on the surface of a sphere", https://www.cmu.edu/biolphys/deserno/pdf/sphere_equi.pdf).

- Featurize, clusterize and find representative dots among the surface dots.
    For each point of the surface, features are extracted. The features vector contains the sorted distances from the point to each atom of the molecule, separated by the chemical element of the atoms. Based on a K-means clustering, the surface dots are clusters/groups, and the point nearest to the centroid of its clusters is selected as its representative point in the molecular surface. The files mol_a_km.xyz and mol_b_km.xyz present this data [a].

| Input structures (*.xyz)   | ![](.figures/cluster.png)      | ![](.figures/molecule.png)
|---------------------------|-------------------------------|---------------------------
| **Surface dots (*_surf.xyz)**         | ![](.figures/cluster_surf.png) | ![](.figures/molecule_surf.png)
| **Surface dots clustering (*_km.xyz)** | ![](.figures/cluster_km.png)   | ![](.figures/molecule_km.png)
| **Clustering in t-SNE reduced features** | ![](.figures/cluster_km_tsne.png) | ![](.figures/molecule_km_tsne.png)

The \*_surf.xyz files present the surface dots with a color for the points associated with each atom. The \*_km.xyz files present the surface dots with a color for the points associated with each cluster of surface dots. Similar colors of different figure have no relation with each other.

The structure with the surface dots can be seen in the [VESTA](https://jp-minerals.org/vesta/en/download.html) code. To correct read their data, you must replace the VESTA configuration file *elements.ini* with the *elements.ini* file added in the present project. These files present the types of atoms, colors, and other properties to automatically add colors to the representation.

#### Adsorption

The objective of this step is to obtain a pull with many and diverse adsorbed structures. Adsorption is performed by combining both molecules by each pair of the representative point of its surfaces. Moreover, for each pair of representative points, many rotations are performed to guarantee good matches between the molecules. These rotations are performed with a grid of rotations of SO3, obtained with a method called Successive Orthogonal Images on SOn. The method was first presented by Mitchell ([DOI:10.1137/030601879](https://doi.org/10.1137/030601879)), but for the present implementation I followed the paper by Yershova ([DOI:10.1177/0278364909352700](https://doi.org/10.1177/0278364909352700)). Note that, the number of adsorbed molecules configurations to analyze is deterministic and is the product of the number of surface clusters for each molecule and the number of rotations in SO3.

A configurations is added to a pull when:
 - The molecules did not overlap in the adsorption, which is considered to have happened when a pair of atoms of different molecules were distant by less than the sum of their VDW radii multiplied by the parameter ovlp_threshold;

 - The present structures is not similar to any one in the in the pull of structure, which is verify with a simple filtering. The adsorbed configurations are featurized with a method similar to the surface points. First, the distances between three key points and each atom are calculated and sorted, keeping separations by each atom type and key point. The key points are the geometrical center of each molecule and the position of the representative surface dots that were employed to create the present configuration. If the euclidian distance between the present configuration and all other structures in the pull were smaller than sim_threshold parameter.

Example structures:
---
| ![](.figures/97.png) | ![](.figures/97_surf_km.png)
|---------------------|-----------------------------|
| ![](.figures/99.png) | ![](.figures/99_surf_km.png)
| ![](.figures/98.png) | ![](.figures/98_surf_km.png)

#### Representative set extraction

Finaly, the structures in poll are clusterized with K-means yielding a representative set. The structures are written in folder_xyz_files (adsorbed structures) and folder_xyz_files_withsurfs (adsorbed structures with surface information).
A vizualization of the clustering process is indicated in the file clustering_representatives*.png.

![](.figures/clustering_representatives_2.png)

### Run code example

Example with required arguments:
```bash
$ cd example
$ python ../adsorption.py --mols cluster.xyz molecule.xyz --surf_ks 20 10 --n_final 100
```

Example with all arguments:
```bash
$ cd example
$ python ../adsorption.py --mols cluster.xyz molecule.xyz --surf_ks 30 10 --n_final 100 --surf_d 10 --n_repeat_km 20 --n_rot 100 --ovlp_threshold 0.90 --sim_threshold  0.04 --out_sufix _2
```

## Comparison of representative sets

The script comparison.py allow to compare multiple sets of adsorbed molecules, for instance, generated with different parameters. It read the molecules in the xyz format, in saparated folders, and get features for they. The features are the same of the employed in the adsorption step described above, if the size of the adsorbed molecules presented in the order of the xyz file. Otherwise, it employs a single key point, the geometric center of the system, which should decrease the quality of the description of the features.

Then, few analysis/plots are performed with this data:
- A histogram of distances among the samples;
- A t-SNE dimensionality reduction;
- A sequence of K-means clustering, increasing K/data size de approximatated 0 a 1.

The results are saved to a file named result_comparison.png:

![](.figures/result_comparison.png)

### Run code example

```bash
$ cd example
$ python ../comparison.py --folders set_1 set_2 set_3 set_4 set_5 --subs_ns 9 3
```
