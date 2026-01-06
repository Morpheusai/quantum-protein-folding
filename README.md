# Install
```
conda create -n duneq python=3.12
pip3 install -r requirements.txt
conda activate duneq
```

# Run
```
# use default backend
python run_protein_folding.py 

# use aws sv1 or pc
python run_protein_folding.py --backend="aws"

# use ibm
python run_protein_folding.py --backend="ibm"


#get top result (default=1)
#results will be stored in the directory `result/{timestamp}_{energy value}_{environment}`.
python run_protein_folding.py --max_results=5

#Compare results to obtain the optimal value
#Step 1: python run_protein_folding.py  # Obtain results; multiple runs are possible
#Step 2: Execute comparison program: python compare_results.py
#rmark:  Default, all output results are compared.
python compare_results.py

#Compare the results of the specified directory
python compare_results.py  --result_dirs submenu1 submenu2 ....

#Return the top N best comparison results (default=3)
python compare_results.py --top_n=5
```
