# Install
```
conda create -n duneq python=3.12
pip3 install -r requirements.txt
conda activate duneq
```

# Run
```
# use default backend
python run_sampler.py 

# use aws sv1 or pc
python run_estimator.py --backend="aws"

# use ibm
python run_sampler.py --backend="ibm"


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

# Generate image from JSON result file (output: result.png in the same directory)
python structure2pic.py --input results/20260107_145139_local/result_xxx.json

# Generate image from XYZ structure file (output: structure.png in the same directory)
python structure2pic.py --input results/example.xyz

```
# 版本代码说明
```
1. run_opt
   run_opt_sample.py
   我们基于qiskit-community/quantum-protein-folding修改的两个文件
2. qthesis-pf
   原始link: https://github.com/QFold-Thesis/quantum-protein-folding
   修改文件：src/enums.py
   class BackendType(Enum):
    """Enum representing quantum backend types for VQE execution."""
        
    LOCAL_STATEVECTOR = "local_statevector"
    IBM_QUANTUM = "ibm_quantum"
    AWS_SIM_QUANTUM = "aws_sim_quantum"
    AWS_QC_QUANTUM = "aws_qc_quantum"  

   修改文件：src/constants.py
   #BACKEND_TYPE: BackendType = BackendType.LOCAL_STATEVECTOR
   #BACKEND_TYPE: BackendType = BackendType.IBM_QUANTUM
   #BACKEND_TYPE: BackendType = BackendType.AWS_SIM_QUANTUM #AWS 模拟器
    BACKEND_TYPE: BackendType = BackendType.AWS_QC_QUANTUM #AWS QC真机

   修改文件：src/backend/backend_factory.py
    添加：_get_aws_sim_quantum_sampler
         _get_aws_qc_quantum_sampler
    添加逻辑参考：_get_ibm_quantum_sampler

    运行环境：同1
    实际aws机器运行错误：模拟器 -> 不支持program sets方法, 这个确实在AWS模拟器使用说明
                      真机 -> 电路错误、不支持: Cannot measure previously measured qubit {qubit_index}
                              这个说明_get_ibm_quantum_sampler中的TranspilingSampler对AWS真机不适用
3. stfc-qf
   原始link: https://github.com/stfc/quantum-protein-folding.git
   create_energy_files.py
   calculate_exact_energies.py
   这两步可以按照官方执行 run.sh:
        export PROJECT_ROOT="/home/ubuntu/workspace/stfc/quantum-protein-folding"
        python scripts/create_energy_files.py -p 2residue.pdb --num_rot 3 -i 0
        python scripts/calculate_exact_energies.py --num_res 2 --num_rot 3
        python scripts/run_qaoa.py -p 2 --num_res 2 --num_rot 3
    实际第3步的run_qaoa.py中，最新版本的qiskit和repo本身对应的qiskit版本都出错，还未进行系统debug

4. QuPepFold
   原始link: 
   修改文件：QuPepFold/qupepfold/qupepfold.py
   添加AWS真机代码：read_cli_inputs -> backend_mode
   运行环境：同1，当前可以成功运行
   
##########################################
## 1. quantum-protein-folding##
## from github:https://github.com/qiskit-community/quantum-protein-folding
## from aritcle:https://www.nature.com/articles/s41534-021-00368-4
## run_opt.py，run_estimator.py   -- based on estimator(Difference: The former is runnable code, while the latter places some class libraries in the lib.)
## run_opt_sampler.py,run_samper.py -- based on sampler(Difference: The former is runnable code, while the latter places some class libraries in the lib.)
python run_sampler.py --max_result=5
python run_opt_sampler --backend=aws_sv1
#############################################################################

## 2.qthesis-pf ##
## from github:https://github.com/QFold-Thesis/quantum-protein-folding
## run_qthesis.py
python run_qthesis.py --main_chain APRLRFY --backend aws_sv1

#############################################################################
## 3.stfc-qf ##
## from github:https://github.com/stfc/quantum-protein-folding.git
## from article: Quantum Algorithm for Protein Side-Chain Optimisation: Comparing Quantum to Classical Methods
## run_stfc_qf.py
# 精确计算
    python run_stfc_qf.py -res 2 -rot 2 -m exact
    
    # 模拟退火
    python run_stfc_qf.py -res 2 -rot 2 -m sa
    
    # QAOA（量子）- 本地模拟器
    python run_stfc_qf.py -res 2 -rot 2 -m qaoa -p 1 -s 100
    
    # QAOA（量子）- AWS 后端
    python run_stfc_qf.py -res 2 -rot 2 -m qaoa -p 1 -s 100 --backend aws_sv1
#############################################################################
## 4.QuPepFold ##
## from github:https://github.com/qiskit-community/qupepfold
## run_qupepfold.py
python run_qupepfold.py --backend aws_sv1  --seq APRLRFY

``` 