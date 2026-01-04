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
```
