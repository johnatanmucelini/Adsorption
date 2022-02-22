#echo 'Example 1:'
#python ../adsorption.py --mols cluster.xyz molecule.xyz --surf_ks 20 10 --n_final 100

echo 'Example 2:'
python ../adsorption.py --mols cluster.xyz molecule.xyz --surf_ks 30 10 --n_final 100 --surf_d 10 --n_repeat_km 20 --n_rot 100 --ovlp_threshold 0.90 --sim_threshold  0.04 --out_sufix _2
