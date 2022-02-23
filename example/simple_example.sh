python ../adsorption.py --mols cluster.xyz molecule.xyz --surf_ks 20 10 --n_final 100 --surf_d 6 --n_repeat_km 20 --n_rot 40 --ovlp_threshold 0.50 --sim_threshold  0.04 --out_sufix _1
python ../adsorption.py --mols cluster.xyz molecule.xyz --surf_ks 20 10 --n_final 100 --surf_d 6 --n_repeat_km 20 --n_rot 40 --ovlp_threshold 0.60 --sim_threshold  0.04 --out_sufix _2
python ../adsorption.py --mols cluster.xyz molecule.xyz --surf_ks 20 10 --n_final 100 --surf_d 6 --n_repeat_km 20 --n_rot 40 --ovlp_threshold 0.70 --sim_threshold  0.04 --out_sufix _3
python ../adsorption.py --mols cluster.xyz molecule.xyz --surf_ks 20 10 --n_final 100 --surf_d 6 --n_repeat_km 20 --n_rot 40 --ovlp_threshold 0.80 --sim_threshold  0.04 --out_sufix _4
python ../adsorption.py --mols cluster.xyz molecule.xyz --surf_ks 20 10 --n_final 100 --surf_d 6 --n_repeat_km 20 --n_rot 40 --ovlp_threshold 0.90 --sim_threshold  0.04 --out_sufix _5
mv folder_xyz_files_1 set_1
mv folder_xyz_files_2 set_2
mv folder_xyz_files_3 set_3
mv folder_xyz_files_4 set_4
mv folder_xyz_files_5 set_5
