#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
蛋白质折叠量子算法 - 采样器模式 (Sampler Mode)

功能特点：
1. 支持本地模拟器和AWS Braket后端
2. 使用CVaR (Conditional Value at Risk) 优化策略
3. 基于采样的能量计算方法
4. 支持多轮独立实验及结果汇总分析
"""

import argparse
import os
import sys
import warnings
import datetime
import json
import numpy as np
import copy
from qiskit import QuantumCircuit, transpile
from qiskit.circuit.library import RealAmplitudes
from qiskit_braket_provider import BraketProvider
from scipy.optimize import minimize
import traceback
from job_metadata_logger import JobMetadataLogger

# 创建元数据记录器实例
metadata_logger = JobMetadataLogger("protein_folding_jobs.csv")

# 环境配置
current_dir = os.path.dirname(os.path.abspath(__file__))
if os.name == 'nt':
    os.environ['PYTHONUTF8'] = '1'
    os.environ['PYTHONIOENCODING'] = 'utf-8'
sys.path.insert(0, os.path.join(current_dir, 'src'))

import matplotlib
matplotlib.use('Agg') 
import matplotlib.pyplot as plt
warnings.filterwarnings('ignore')

# ====================
# 参数配置
# ====================
parser = argparse.ArgumentParser(description='蛋白质折叠量子算法 - 采样器模式')
parser.add_argument('--backend', default='local', help='量子后端选择: local (本地模拟器), aws_sv1 (AWS模拟器), aws_garnet (AWS量子芯片), aws_ionq (AWS IonQ量子设备), aws_forte (AWS IonQ Forte量子设备), ibm (IBM量子设备), ibm_simulator (IBM模拟器)')
parser.add_argument('--random_seed', type=int, default=23, help='随机种子，用于确保结果可重现')
parser.add_argument('--max_optimization_iterations', type=int, default=10, help='最大优化迭代次数')
parser.add_argument('--ansatz_reps', type=int, default=1, help='变分量子线路的重复层数')
parser.add_argument('--main_chain', default='APRLRFY', help='蛋白质主链氨基酸序列')
parser.add_argument('--penalty_back', type=float, default=10, help='几何约束惩罚系数')
parser.add_argument('--penalty_chiral', type=float, default=10, help='手性约束惩罚系数')
parser.add_argument('--penalty_local_overlap', type=float, default=10, help='局部重叠惩罚系数')
parser.add_argument('--alpha', type=float, default=0.1, help='CVaR参数: 选择最低能量的alpha比例样本')
parser.add_argument('--shots', type=int, default=100, help='量子测量采样次数')
parser.add_argument('--max_results', type=int, default=1, help='计算结果的最大数量')
parser.add_argument('--aws_region', default=None, help='AWS区域设置（可选）')
args = parser.parse_args()

TIMESTAMP = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
RESULT_DIR = os.path.join("results", f"{TIMESTAMP}_{args.backend}_sampler")
os.makedirs(RESULT_DIR, exist_ok=True)

# ====================
# 量子后端配置与运行逻辑
# ====================

def setup_sampler_backend(backend_name, aws_region=None, shots=1000):
    """配置支持采样器模式的量子后端，兼容多种量子云服务"""
    try:
        if backend_name.lower() == 'aws_sv1':
            from qiskit_braket_provider import BraketProvider
            if aws_region: os.environ['AWS_DEFAULT_REGION'] = aws_region
            provider = BraketProvider()
            backend = provider.get_backend('SV1')
            print(f"✓ AWS SV1 已连接")
            return backend
        elif backend_name.lower() == 'aws_garnet':
            from qiskit_braket_provider import BraketProvider
            if aws_region: os.environ['AWS_DEFAULT_REGION'] = aws_region
            provider = BraketProvider()
            backend = provider.get_backend('Garnet')
            print(f"✓ AWS Garnet 已连接")
            return backend
        elif backend_name.lower() == 'aws_ionq':
            from qiskit_braket_provider import BraketProvider
            if aws_region: os.environ['AWS_DEFAULT_REGION'] = aws_region
            provider = BraketProvider()
            # 获取 IonQ Harmony (11 qubits)
            backend = provider.get_backend('IonQ Device')
            print(f"✓ AWS IonQ Harmony 已连接")
            return backend
        elif backend_name.lower() == 'aws_forte':
            from qiskit_braket_provider import BraketProvider
            if aws_region: os.environ['AWS_DEFAULT_REGION'] = aws_region
            provider = BraketProvider()
            # 获取 IonQ Forte-1 (30+ qubits)
            backend = provider.get_backend('Forte 1')
            print(f"✓ AWS IonQ Forte-1 已连接")
            return backend
        elif backend_name.lower() == 'ibm':
            try:
                from qiskit_ibm_runtime import QiskitRuntimeService
                service = QiskitRuntimeService()
                # 使用性能最好的可用设备
                backend = service.least_busy(operational=True, simulator=False)
                print(f"✓ IBM 量子后端已连接: {backend.name}")
                return backend
            except Exception as e:
                print(f"✗ IBM 真实硬件连接失败: {e}")
                print("  提示: 请确保已通过 'qiskit-ibm-runtime' 配置 IBM Quantum 访问凭据")
                # 回退到IBM模拟器
                try:
                    from qiskit_ibm_runtime import QiskitRuntimeService
                    service = QiskitRuntimeService()
                    # 使用IBM模拟器
                    backend = service.backend("ibmq_qasm_simulator")
                    print(f"✓ IBM 模拟器已连接: {backend.name}")
                    return backend
                except Exception as sim_e:
                    print(f"✗ IBM 模拟器连接也失败: {sim_e}")
                    # 最终回退到本地模拟器
                    from qiskit.providers.basic_provider import BasicSimulator
                    backend = BasicSimulator()
                    print("  ✓ 已回退到本地 BasicSimulator")
                    return backend
        elif backend_name.lower() == 'ibm_simulator':
            from qiskit_ibm_runtime import QiskitRuntimeService
            service = QiskitRuntimeService()
            # 使用IBM模拟器
            backend = service.backend("ibmq_qasm_simulator")
            print(f"✓ IBM 模拟器已连接: {backend.name}")
            return backend
        else:
            from qiskit.providers.basic_provider import BasicSimulator
            backend = BasicSimulator()
            print("✓ 本地 BasicSimulator 已就绪")
            return backend
    except Exception as e:
        print(f"✗ 后端设置失败: {e}")
        traceback.print_exc()  # 输出详细的错误追踪
        from qiskit.providers.basic_provider import BasicSimulator
        return BasicSimulator()

# ====================
# 能量计算逻辑
# ====================

def estimate_energy_from_bitstring(bitstring, qubit_op):
    """
    计算单个采样比特串对应的哈密顿量能量值
    
    Args:
        bitstring (str): 量子测量得到的比特串，如 '010110'
        qubit_op: 量子比特哈密顿量算子
        
    Returns:
        float: 对应的能量值
    """
    energy = 0.0
    # 将比特串反转，使其与哈密顿量的索引顺序对应
    bit_list = [int(b) for b in reversed(bitstring)]
    
    # 遍历哈密顿量的每一项 (Pauli算子及其系数)
    for pauli_str, coeff in qubit_op.to_list():
        val = 1.0
        # 对于每个Pauli项，计算其在当前比特态下的期望值
        for i, char in enumerate(reversed(pauli_str)):
            if char == 'Z' and bit_list[i] == 1:
                # Z算子在|1>态下贡献-1，在|0>态下贡献+1
                val *= -1.0
            elif char == 'X' or char == 'Y':
                # X,Y算子在计算能量时会引入非对角项，此处简化处理为0
                val = 0.0 
                break
        energy += coeff.real * val
    return energy

def calculate_cvar_energy(counts, qubit_op, alpha):
    """
    实现CVaR (Conditional Value at Risk) 优化策略
    CVaR是一种风险度量方法，只考虑能量最低的一部分样本
    
    Args:
        counts (dict): 量子测量结果，键为比特串，值为出现次数
        qubit_op: 量子比特哈密顿量算子
        alpha (float): CVaR参数，取值[0,1]，表示使用最低能量样本的比例
        
    Returns:
        float: CVaR能量值（最低alpha比例样本的平均能量）
    """
    energies = []
    total_shots = sum(counts.values())
    
    # 计算每个测量结果对应的能量
    for bitstring, count in counts.items():
        e = estimate_energy_from_bitstring(bitstring, qubit_op)
        # 按照测量次数复制能量值
        energies.extend([e] * count)
    
    # 按能量值升序排列
    energies.sort()
    
    # 计算需要保留的样本数量（最低能量的alpha比例）
    num_keep = max(1, int(total_shots * alpha))
    
    # 返回最低能量样本的平均值
    return np.mean(energies[:num_keep])

# ====================
# 主计算流程
# ====================
@metadata_logger
def main():
    print(f"🚀 启动蛋白质折叠计算任务 (采样器模式) | 序列: {args.main_chain}")
    
    # 导入蛋白质折叠相关模块
    from protein_folding.interactions.miyazawa_jernigan_interaction import MiyazawaJerniganInteraction
    from protein_folding.peptide.peptide import Peptide
    from protein_folding.protein_folding_problem import ProteinFoldingProblem
    from protein_folding.penalty_parameters import PenaltyParameters
    from src.protein_folding.utils.detailed_pdb_generator import convert_xyz_to_detailed_pdb

    # 1. 构建蛋白质折叠问题模型
    # 创建肽对象：包含主链序列和侧链序列（这里设为空字符串）
    peptide = Peptide(args.main_chain, [""] * len(args.main_chain))
    
    # 定义约束项的惩罚系数
    penalty_terms = PenaltyParameters(args.penalty_chiral, args.penalty_back, args.penalty_local_overlap)
    
    # 创建蛋白质折叠问题实例
    problem = ProteinFoldingProblem(peptide, MiyazawaJerniganInteraction(), penalty_terms)
    
    # 生成对应的量子比特哈密顿量
    qubit_op = problem.qubit_op()
    num_qubits = qubit_op.num_qubits  # 获取所需量子比特数

    # 2. 根据参数选择合适的量子计算后端
    backend = setup_sampler_backend(args.backend, args.aws_region, args.shots)
    print(f"✓ 量子后端已就绪 | 量子比特数: {num_qubits}")

    # 3. 构建参数化量子电路
    # 使用RealAmplitudes作为变分波函数 ansatz
    ansatz = RealAmplitudes(num_qubits=num_qubits, reps=args.ansatz_reps)
    
    # 将ansatz分解为基本门操作，便于后续处理
    decomposed = ansatz.decompose()
    
    # 创建清洁的量子电路（移除barrier等不必要的元素）
    clean_circuit = QuantumCircuit(num_qubits)
    for inst in decomposed.data:
        # 移除barrier指令，因为它不影响量子计算结果但可能影响转译
        if inst.operation.name != 'barrier':
            clean_circuit.append(inst.operation, inst.qubits, inst.clbits)
    
    # 添加测量门，以便获取量子比特的状态
    clean_circuit.measure_all()
    
    # 根据后端类型决定是否转译电路
    # 本地模拟器通常不需要转译，而云量子设备需要针对硬件特性进行转译
    if args.backend.lower() == 'local':
        # 对于本地模拟器，直接使用原始电路
        transpiled_circuit = clean_circuit
    else:
        # 对于AWS后端，需要转译电路以适配目标设备的拓扑和门集
        transpiled_circuit = transpile(clean_circuit,backend=backend,
                                        initial_layout=list(range(num_qubits)) if args.backend != 'local' else None,
                                        optimization_level=3
                                      )
    print(f"   逻辑比特数 (算法需求): {clean_circuit.num_qubits}")
    print(f"   转译后物理比特数 (硬件占用): {transpiled_circuit.num_qubits}")
    all_conv_data = []
    for i in range(args.max_results):
        print(f"\n--- 实验 {i+1}/{args.max_results} ---")
        curr_seed = args.random_seed + i
        np.random.seed(curr_seed)
        
        convergence_history = []

        def objective_function(params):
            """
            优化目标函数
            该函数接受参数，执行量子电路，计算CVaR能量，并返回用于优化的值
            
            Args:
                params: 变分参数数组
                
            Returns:
                float: CVaR能量值（优化目标）
            """
            # 将参数绑定到量子电路
            bound_circ = transpiled_circuit.assign_parameters(params)
            
            # 在选定的后端上执行量子电路
            job = backend.run(bound_circ, shots=args.shots)
            
            # 获取量子测量结果
            counts = job.result().get_counts()
            
            # 使用CVaR策略计算能量
            energy = calculate_cvar_energy(counts, qubit_op, args.alpha)
            
            # 记录收敛历史，用于分析优化过程
            convergence_history.append(energy)
            
            # 每隔一次迭代打印一次进度（避免过多输出）
            if len(convergence_history) % 2 == 0:
                print(f"    迭代 {len(convergence_history)}: CVaR能量 = {energy:.4f}")
            return energy

        # 开始优化 (使用 COBYLA)
        initial_params = np.random.uniform(-np.pi, np.pi, ansatz.num_parameters)
        res = minimize(objective_function, initial_params, method='COBYLA', 
                       options={'maxiter': args.max_optimization_iterations})
        
        all_conv_data.append({'counts': list(range(len(convergence_history))), 'values': convergence_history})

        # --- 结果解析与保存 ---
        print(f"    - 正在分析最优结果...")
        
        # 使用最优参数生成最终量子电路并执行测量
        final_bound_circuit = transpiled_circuit.assign_parameters(res.x)
        final_job = backend.run(final_bound_circuit, shots=args.shots)
        final_counts = final_job.result().get_counts()
        total_shots = sum(final_counts.values())
        
        # 提取出现频率最高的比特串作为最优解
        # 注意：这里将比特串反转以符合蛋白质折叠问题的编码约定
        best_raw_bit = max(final_counts, key=final_counts.get)
        best_bitstring = best_raw_bit[::-1]  # 反转比特串顺序
        
        # 创建模拟结果对象，以兼容蛋白质折叠问题的解释接口
        class MockResult:
            def __init__(self, bs, val, counts, total):
                # 设置最高概率比特串的振幅为1（理想情况）
                self.eigenstate = {bs: 1.0}
                # 设置优化得到的最小能量值
                self.eigenvalue = val
                # 提供完整的概率分布用于后续分析
                self.probabilities = {k[::-1]: v/total for k, v in counts.items()}

        # 创建模拟结果并解析为蛋白质结构
        raw_res = MockResult(best_bitstring, res.fun, final_counts, total_shots)
        result = problem.interpret(raw_res)
        
        # 提取最终能量值
        energy = float(res.fun)
        print(f"    - 最优转向序列: {result.turn_sequence}")

        # 1. 保存JSON格式的结果参数
        xyz_data = result.protein_shape_file_gen.get_xyz_data()
        result_data = {
            "result_index": i + 1,
            "energy": energy,
            "turn_sequence": result.turn_sequence,
            "main_chain_sequence": args.main_chain,
            "xyz_coordinates": [list(row) for row in xyz_data] if xyz_data is not None else []
        }
        json_path = os.path.join(RESULT_DIR, f'result_{i+1}_energy_{energy:.4f}.json')
        with open(json_path, 'w') as f:
            json.dump(result_data, f, indent=2)

        # 2. 保存详细 PDB 文件
        if xyz_data is not None:
            pdb_path = os.path.join(RESULT_DIR, f'structure_{i+1}_energy_{energy:.4f}.pdb')
            convert_xyz_to_detailed_pdb(xyz_data, pdb_path, f"Structure {i+1}")
            print(f"    ✓ PDB文件已保存")

        # 3. 保存 3D 结构图
        try:
            fig_struct = result.get_figure(title=f"Result {i+1} (E={energy:.4f})")
            png_path = os.path.join(RESULT_DIR, f"structure_{i+1}_energy_{energy:.4f}.png")
            fig_struct.savefig(png_path)
            plt.close(fig_struct)
        except Exception as e:
            print(f"    ⚠ 绘图失败: {e}")

    # 生成优化过程的收敛图
    plt.figure(figsize=(10, 6))
    for idx, data in enumerate(all_conv_data):
        plt.plot(data['counts'], data['values'], label=f'Run {idx+1}')
    plt.xlabel("Evaluation Counts")
    plt.ylabel("Energy")
    plt.title(f"VQE Sampler Convergence ({args.main_chain})")
    plt.legend()
    plt.grid(True)
    plt.savefig(os.path.join(RESULT_DIR, "vqe_optimization_summary.png"))
    
    print(f"\n🎉 蛋白质折叠计算任务完成！所有结果保存在: {RESULT_DIR}")

if __name__ == "__main__":
    main()