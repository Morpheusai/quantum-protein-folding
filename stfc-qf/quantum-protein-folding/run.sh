export PROJECT_ROOT="/home/ubuntu/workspace/stfc/quantum-protein-folding"

python scripts/create_energy_files.py -p 2residue.pdb --num_rot 3 -i 0
 
python scripts/calculate_exact_energies.py --num_res 2 --num_rot 3
 
python scripts/run_qaoa.py -p 2 --num_res 2 --num_rot 3
