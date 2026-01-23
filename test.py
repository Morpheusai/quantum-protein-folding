import sys
import os
sys.path.insert(0, os.path.join(os.getcwd(), 'src'))

# 测试导入必要的库
try:
    from qiskit.primitives import StatevectorEstimator
    from qiskit_algorithms import VQE
    from qiskit_algorithms.optimizers import COBYLA
    from qiskit.circuit.library import RealAmplitudes
    print('✓ 量子计算库导入成功')
except ImportError as e:
    print(f'⚠ 部分库未安装: {e}')

# 测试基本语法
try:
    code = '''
import argparse
import numpy as np
import copy

# 模拟参数
class Args:
    shots = 100
    backend = 'local'
    max_results = 1
    main_chain = 'AP'

args = Args()

# 测试收敛数据结构
convergence = {'counts': [], 'values': [], 'cumulative_shots': []}
actual_shots_list = []

def record_shots_from_metadata(metadata):
    actual_shots = 0
    if \"shots_per_circuit\" in metadata:
        actual_shots = sum(metadata[\"shots_per_circuit\"])
    elif \"execution\" in metadata and \"circuits\" in metadata[\"execution\"]:
        actual_shots = sum(c[\"shots\"] for c in metadata[\"execution\"][\"circuits\"])
    elif \"shots\" in metadata:
        shots_per_circuit = metadata[\"shots\"]
        num_circuits = metadata.get(\"num_circuits\", 1)
        actual_shots = shots_per_circuit * num_circuits
    return actual_shots

# 测试回调函数逻辑
def callback(eval_count, parameters, mean, std):
    convergence['counts'].append(eval_count)
    convergence['values'].append(mean)
    
    current_step_shots = args.shots
    actual_shots_list.append(current_step_shots)
    
    cumulative_shots = sum(actual_shots_list)
    convergence['cumulative_shots'].append(cumulative_shots)

# 模拟几次回调调用
for i in range(3):
    callback(i+1, None, -1.0-i*0.1, 0.01)

print(f'测试收敛数据: {convergence}')
print('✓ 语法测试通过')
'''
    exec(code)
except Exception as e:
    print(f'✗ 语法测试失败: {e}')