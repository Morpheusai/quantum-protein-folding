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
# 环境配置
current_dir = os.path.dirname(os.path.abspath(__file__))
if os.name == 'nt':
    os.environ['PYTHONUTF8'] = '1'
    os.environ['PYTHONIOENCODING'] = 'utf-8'
sys.path.insert(0, os.path.join(current_dir, 'src'))

from qiskit import QuantumCircuit, transpile
from qiskit.circuit.library import RealAmplitudes
from qiskit_braket_provider import BraketProvider
from scipy.optimize import minimize
from qiskit_algorithms.optimizers import SPSA
import traceback
from lib.job_metadata_logger import JobMetadataLogger

# 创建元数据记录器实例
metadata_logger = JobMetadataLogger("protein_folding_jobs_detailed.csv")

# 引入 Job 记录器
from lib.quantum_optimizer import JobRecorder



import matplotlib
matplotlib.use('Agg') 
import matplotlib.pyplot as plt
warnings.filterwarnings('ignore')

class MockResult:
    """模拟量子结果类，用于解析蛋白质结构"""
    def __init__(self, eigenstate, eigenvalue, counts, total_shots):
        self.eigenstate = eigenstate  # 字典 {bitstring: 1.0}
        self.eigenvalue = eigenvalue
        self.counts = counts
        self.total_shots = total_shots
        self.probabilities = {k[::-1]: v/total_shots for k, v in counts.items()}
    
    def get_eigenstate(self): return self.eigenstate
    def get_eigenvalue(self): return self.eigenvalue
    def get_counts(self): return self.counts
    def get_total_shots(self): return self.total_shots
    def get_most_probable_state(self):
        if not self.counts: return self.eigenstate
        max_c = -1
        best_s = None
        for s, c in self.counts.items():
            if c > max_c:
                max_c = c
                best_s = s
        return best_s

# ====================
# 参数配置
# ====================
parser = argparse.ArgumentParser(description='蛋白质折叠量子算法 - 采样器模式')
parser.add_argument('--backend', default='local', help='量子后端选择: local (本地模拟器，优先使用AerSimulator), local_aer (强制使用AerSimulator), aws_sv1 (AWS模拟器), aws_garnet (AWS量子芯片), aws_ionq (AWS IonQ量子设备), aws_forte (AWS IonQ Forte量子设备), ibm (IBM量子设备), ibm_simulator (IBM模拟器)')
parser.add_argument('--random_seed', type=int, default=42, help='随机种子，用于确保结果可重现')
parser.add_argument('--max_optimization_iterations', type=int, default=10, help='最大优化迭代次数')
parser.add_argument('--ansatz_reps', type=int, default=1, help='变分量子线路的重复层数')
parser.add_argument('--main_chain', default='APRLRFY', help='蛋白质主链氨基酸序列')
parser.add_argument('--penalty_back', type=float, default=10, help='几何约束惩罚系数')
parser.add_argument('--penalty_chiral', type=float, default=10, help='手性约束惩罚系数')
parser.add_argument('--penalty_local_overlap', type=float, default=10, help='局部重叠惩罚系数')
parser.add_argument('--alpha', type=float, default=0.25, help='CVaR参数: 选择最低能量的alpha比例样本')
parser.add_argument('--shots', type=int, default=100, help='量子测量采样次数')
parser.add_argument('--max_results', type=int, default=1, help='每次迭代的结果数量')
parser.add_argument('--aws_region', default=None, help='AWS区域设置（可选）')

# 优化增强参数
parser.add_argument('--optimizer', default='COBYLA', help='优化器选择: COBYLA (默认), SPSA, SLSQP')

# 自适应Shots参数
parser.add_argument('--adaptive_shots', action='store_true', help='开启自适应Shots策略')
parser.add_argument('--min_shots', type=int, default=100, help='自适应Shots的最小采样数 (默认为100)')
parser.add_argument('--max_shots', type=int, default=2000, help='自适应Shots的最大采样数 (默认为2000)')
parser.add_argument('--unique_structures', action='store_true', help='开启结构去重：仅返回折叠结构不同的最优结果')
parser.add_argument('--restarts', type=int, default=1, help='独立实验运行次数 (Multi-Restart)，用于避免局部最优，默认为1')
parser.add_argument('--dry_run', action='store_true', help='干跑模式：仅在真实提交前进行本地预检')
parser.add_argument('--resume', type=str, default=None, help='断点续传：指定结果目录以恢复历史任务')
parser.add_argument('--initial_params', type=str, default=None, help='Warm-Start：从 JSON 文件加载初始参数向量')
args = parser.parse_args()

TIMESTAMP = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
RESULT_DIR = os.path.join("results", f"{TIMESTAMP}_{args.backend}_sampler")
os.makedirs(RESULT_DIR, exist_ok=True)

# ====================
# 量子后端配置与运行逻辑
# ====================

def setup_sampler_backend(backend_name, aws_region=None, shots=1000):
    """配置支持采样器模式的量子后端，兼容多种量子云服务，并支持 SamplerV2"""
    backend_info = {'backend': None, 'sampler_v2': None}
    try:
        if backend_name.lower() == 'aws_sv1':
            from qiskit_braket_provider import BraketProvider
            if aws_region: os.environ['AWS_DEFAULT_REGION'] = aws_region
            provider = BraketProvider()
            backend = provider.get_backend('SV1')
            print(f"✓ AWS SV1 已连接")
            backend_info['backend'] = backend
        elif backend_name.lower() == 'aws_garnet':
            from qiskit_braket_provider import BraketProvider
            if aws_region: os.environ['AWS_DEFAULT_REGION'] = aws_region
            provider = BraketProvider()
            backend = provider.get_backend('Garnet')
            print(f"✓ AWS Garnet 已连接")
            backend_info['backend'] = backend
        elif backend_name.lower() == 'aws_ionq':
            from qiskit_braket_provider import BraketProvider
            if aws_region: os.environ['AWS_DEFAULT_REGION'] = aws_region
            provider = BraketProvider()
            backend = provider.get_backend('IonQ Device')
            print(f"✓ AWS IonQ Harmony 已连接")
            backend_info['backend'] = backend
        elif backend_name.lower() == 'aws_forte':
            from qiskit_braket_provider import BraketProvider
            if aws_region: os.environ['AWS_DEFAULT_REGION'] = aws_region
            provider = BraketProvider()
            backend = provider.get_backend('Forte 1')
            print(f"✓ AWS IonQ Forte-1 已连接")
            backend_info['backend'] = backend
        elif backend_name.lower() == 'ibm':
            from qiskit_ibm_runtime import QiskitRuntimeService
            service = QiskitRuntimeService()
            backend = service.least_busy(operational=True, simulator=False)
            print(f"✓ IBM 量子后端已连接: {backend.name}")
            backend_info['backend'] = backend
            # 初始化 IBM SamplerV2
            try:
                from qiskit_ibm_runtime import SamplerV2 as IBMSampler
                backend_info['sampler_v2'] = IBMSampler(mode=backend)
                print(f"✓ 已配置 IBM SamplerV2 接口")
            except: pass
        elif backend_name.lower() == 'ibm_simulator':
            from qiskit_ibm_runtime import QiskitRuntimeService
            service = QiskitRuntimeService()
            backend = service.backend("ibmq_qasm_simulator")
            print(f"✓ IBM 模拟器已连接: {backend.name}")
            backend_info['backend'] = backend
        elif backend_name.lower() in ('local_aer', 'local'):
            try:
                from qiskit_aer import AerSimulator
                backend = AerSimulator()
                print("✓ 本地 AerSimulator 已就绪")
                # 初始化本地 SamplerV2
                from qiskit.primitives import StatevectorSampler
                backend_info['sampler_v2'] = StatevectorSampler()
                print("✓ 已配置本地 SamplerV2 (StatevectorSampler)")
            except ImportError:
                from qiskit.providers.basic_provider import BasicSimulator
                backend = BasicSimulator()
                print("✓ 本地 BasicSimulator 已就绪")
            backend_info['backend'] = backend
        else:
            from qiskit.providers.basic_provider import BasicSimulator
            backend = BasicSimulator()
            backend_info['backend'] = backend
            print("✓ 本地 BasicSimulator 已就绪")
            
        return backend_info
    except Exception as e:
        print(f"✗ 后端设置失败: {e}")
        from qiskit.providers.basic_provider import BasicSimulator
        backend_info['backend'] = BasicSimulator()
        return backend_info

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
                # X,Y算符在计算基态下的期望值为0，这是正确的处理
                # 因为计算基态是Z算符的本征态，X/Y算符的期望值确实为0
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
    
    # 计算每个比特串的能量，并根据出现次数复制
    for bitstring, count in counts.items():
        e = estimate_energy_from_bitstring(bitstring, qubit_op)
        # 根据出现次数复制能量值，保持正确的权重
        energies.extend([e] * count)
    
    # 按能量值升序排列
    energies.sort()
    
    # 计算需要保留的样本数量（基于总shots数）
    total_shots = sum(counts.values())
    num_keep = max(1, int(total_shots * alpha))
    
    # 返回最低能量样本的平均值
    return np.mean(energies[:num_keep])

def extract_top_results(counts, qubit_op, top_n):
    """
    从量子测量结果中提取能量最低的top_n个结果
    
    Args:
        counts (dict): 量子测量结果，键为比特串，值为出现次数
        qubit_op: 量子比特哈密顿量算子
        top_n (int): 需要提取的最优结果数量
        
    Returns:
        list: 包含top_n个最优结果的列表，每个元素为(bitstring, energy, count)元组
    """
    # 计算每个比特串的能量
    bitstring_energies = []
    for bitstring, count in counts.items():
        energy = estimate_energy_from_bitstring(bitstring, qubit_op)
        bitstring_energies.append((bitstring, energy, count))
    
    # 按能量值升序排列
    bitstring_energies.sort(key=lambda x: x[1])
    
    # 提取能量最低的top_n个结果
    top_results = bitstring_energies[:top_n]
    
    return top_results

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
    backend_info = setup_sampler_backend(args.backend, args.aws_region, args.shots)
    backend = backend_info['backend']
    sampler_v2 = backend_info['sampler_v2']
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
    if args.backend.lower() in ['local', 'local_aer', 'local_qiskit']:
        # 对于本地模拟器，直接使用原始电路
        transpiled_circuit = clean_circuit
    else:
        # 对于其他后端，特别是云端设备，需要转译电路以适配目标设备的拓扑和门集
        transpiled_circuit = transpile(clean_circuit, backend=backend,
                                        initial_layout=list(range(clean_circuit.num_qubits)),
                                        optimization_level=3
                                      )
    print(f"   逻辑比特数 (算法需求): {clean_circuit.num_qubits}")
    print(f"   转译后物理比特数 (硬件占用): {transpiled_circuit.num_qubits}")
    
    # 干跑模式 (Dry-Run)
    if args.dry_run:
        print("\n    [Dry-Run] 正在执行 Sampler 预检...")
        try:
            test_params = np.random.uniform(-np.pi, np.pi, ansatz.num_parameters)
            if sampler_v2:
                job = sampler_v2.run([(transpiled_circuit, test_params)], shots=10)
            else:
                bound_c = transpiled_circuit.assign_parameters(test_params)
                job = backend.run(bound_c, shots=10)
            JobRecorder.record_job(job, RESULT_DIR, label="dry_run_sampler")
            job.result()
            print("    ✓ Sampler 预检通过")
        except Exception as e:
            print(f"    ❌ Sampler 预检失败: {e}")
            raise e

    all_conv_data = []
    global_candidates = [] # 初始化全球候选池
    best_energy = float('inf')
    best_results_tuple = None # 将在第一次实验后初始化
    best_trial_idx = 1
    
    # 开始 Multi-Restart 循环
    for trial_idx in range(args.restarts):
        print(f"\n" + "="*60)
        print(f"--- 实验 {trial_idx + 1}/{args.restarts} (Multi-Restart) ---")
        print(f"随机种子: {args.random_seed + trial_idx}")
        np.random.seed(args.random_seed + trial_idx)
        
        # 加载 Warm-Start 初始参数 (如果指定)
        if args.initial_params:
            try:
                with open(args.initial_params, 'r') as f:
                    params_data = json.load(f)
                    args.initial_point = np.array(params_data['initial_point'])
                    print(f"✓ 已加载 Warm-Start 参数向量 (维度: {len(args.initial_point)})")
            except Exception as e:
                print(f"⚠ 加载初始参数失败: {e}，将使用随机初始化")
                args.initial_point = None
        else:
            args.initial_point = None
        
        trial_dir = os.path.join(RESULT_DIR, f"trial_{trial_idx + 1}")
        os.makedirs(trial_dir, exist_ok=True)
        
        convergence_history = []
        std_history = [] 
        cumulative_shots_history = []  
        iteration_shots_history = []  
        iteration_results = []  
        all_top_energies = []
        
        iteration_result_dir = os.path.join(trial_dir, "iter_all_results")
        os.makedirs(iteration_result_dir, exist_ok=True)

        def objective_function(params):
            """
            优化目标函数
            该函数接受参数，执行量子电路，使用CVaR策略计算能量，并返回用于优化的值
            """
            # 执行量子电路
            current_shots = args.shots
            if args.adaptive_shots:
                max_iter = args.max_optimization_iterations
                current_iter = len(convergence_history)
                if max_iter > 1:
                    ratio = min(current_iter / (max_iter - 1), 1.0)
                    current_shots = int(args.min_shots + (args.max_shots - args.min_shots) * ratio)

            if sampler_v2:
                job = sampler_v2.run([(transpiled_circuit, params)], shots=current_shots)
            else:
                bound_circ = transpiled_circuit.assign_parameters(params)
                job = backend.run(bound_circ, shots=current_shots)
            
            # 记录 Job ID
            JobRecorder.record_job(job, trial_dir, label=f"sampler_step_{len(convergence_history)+1}")
            
            if sampler_v2:
                result = job.result()[0]
                data_name = 'meas' if 'meas' in result.data else next(iter(result.data))
                counts = result.data[data_name].get_counts()
            else:
                counts = job.result().get_counts()
            
            actual_shots = sum(counts.values())
            energy = calculate_cvar_energy(counts, qubit_op, args.alpha)
            
            # 计算标准差
            energies_std = [estimate_energy_from_bitstring(bs, qubit_op) for bs in counts.keys()]
            weights_std = [counts[bs] for bs in counts.keys()]
            mean_std = np.average(energies_std, weights=weights_std) if energies_std else 0
            std_e = np.sqrt(np.average((np.array(energies_std) - mean_std)**2, weights=weights_std)) if energies_std else 0
            std_history.append(std_e)
            
            top_results = extract_top_results(counts, qubit_op, max(args.max_results * 5, 20) if args.unique_structures else args.max_results)
            top_energies = [result[1] for result in top_results]
            
            convergence_history.append(energy)
            iteration_shots_history.append(actual_shots)
            cumulative_shots_history.append((cumulative_shots_history[-1] if cumulative_shots_history else 0) + actual_shots)
            all_top_energies.append(top_energies)
            
            if len(convergence_history) % 2 == 0:
                print(f"    迭代 {len(convergence_history)}: CVaR能量 = {energy:.4f}, 实际shots = {actual_shots}")
            
            iteration_data = {
                "iteration": len(convergence_history),
                "cvar_energy": energy,
                "top_energies": top_energies,
                "top_results": [(result[0], float(result[1]), result[2]) for result in top_results],
                "total_counts": len(counts),
                "actual_shots": actual_shots,
                "counts": counts, # 完整计数以供后续分析
                "protein_structure": {"best_bitstring": top_results[0][0] if top_results else ""}
            }
            iteration_results.append(iteration_data)
            
            # 保存单轮迭代
            with open(os.path.join(iteration_result_dir, f'iteration_{len(convergence_history)}_result.json'), 'w') as f:
                json.dump({"iteration": iteration_data["iteration"], "cvar_energy": energy, "top_results": [{"bitstring": r[0], "energy": r[1]} for r in iteration_data["top_results"]]}, f, indent=2)
            
            return energy

        # 5. 执行优化计算
        print(f"   初始参数已生成，维度: {ansatz.num_parameters}")
        initial_params = np.random.uniform(-np.pi, np.pi, ansatz.num_parameters)
        
        if args.optimizer.upper() == 'SPSA':
            optimizer = SPSA(maxiter=args.max_optimization_iterations)
            result = optimizer.minimize(objective_function, initial_params)
            res = type('obj', (object,), {'x': result.x, 'fun': result.fun})
        else:
            method = 'COBYLA' if args.optimizer.upper() == 'COBYLA' else 'SLSQP'
            res = minimize(objective_function, initial_params, method=method, options={'maxiter': args.max_optimization_iterations})
        
        print(f"  [优化结束] 最低能量: {res.fun:.4f}")
        
        for iter_data in iteration_results:
            for cand in iter_data.get("top_results", []): global_candidates.append(cand)

        if best_results_tuple is None or res.fun < best_energy:
            best_energy = res.fun
            best_trial_idx = trial_idx + 1
            best_results_tuple = (res, convergence_history, std_history, iteration_results, all_top_energies, cumulative_shots_history, iteration_shots_history)
            print(f"  ★ 发现新最佳实验: {best_trial_idx} (能量: {best_energy:.4f})")
            
        print(f"  [实验 {trial_idx + 1} 结束] 当前最佳能量: {best_energy:.4f}")


    # ====================
    # 全局分析与去重 (Global Analysis)
    # ====================
    print(f"\n" + "="*60)
    print(f"所有实验结束。最佳实验: {best_trial_idx} (能量: {best_energy:.4f})")
    print(f"正在进行跨运行结果汇总与去重...")
    
    # 解包最佳结果供可视化
    res, convergence_history, std_history, iteration_results, all_top_energies, cumulative_shots_history, iteration_shots_history = best_results_tuple
    
    # 汇总去重逻辑
    unique_candidates = {}
    for bs, en, count in global_candidates:
        if bs not in unique_candidates or en < unique_candidates[bs][0]:
            unique_candidates[bs] = (en, count)
    
    sorted_candidates = sorted(unique_candidates.items(), key=lambda x: x[1][0])
    
    if args.unique_structures:
        print(f"    - 正在执行跨运行结构去重 (目标: {args.max_results} 个不同结构)...")
        final_top_results = []
        seen_structures = set()
        
        for bitstring, (energy, count) in sorted_candidates:
            try:
                # 临时解析结构用于去重
                # 注意：此时 total_shots 并不重要，只为了解析结构
                temp_mock_result = MockResult({bitstring: 1.0}, energy, {bitstring: count}, 100)
                temp_result = problem.interpret(temp_mock_result)
                structure_sig = str(temp_result.turn_sequence)
                
                if structure_sig not in seen_structures:
                    seen_structures.add(structure_sig)
                    final_top_results.append((bitstring, energy, count))
                    
                if len(final_top_results) >= args.max_results:
                    break
            except Exception as e:
                print(f"    ⚠ 解析比特串 {bitstring} 失败: {e}")
                continue
    else:
        final_top_results = [(bs, en, count) for bs, (en, count) in sorted_candidates[:args.max_results]]

    # 准备可视化数据
    all_conv_data = [{
        'counts': list(range(len(convergence_history))), 
        'values': convergence_history, 
        'stds': std_history,
        'cumulative_shots': cumulative_shots_history, 
        'iteration_shots': iteration_shots_history,
        'label': f'Best Trial ({best_trial_idx})'
    }]
    
    # 原有的 all_top_energies 可视化 (来自最佳 Trial)
    if all_top_energies:
        for i in range(min(args.max_results, len(all_top_energies[0]))):
            top_i_energies = [energies[i] if i < len(energies) else None for energies in all_top_energies]
            valid_counts = []
            valid_energies = []
            for j, energy in enumerate(top_i_energies):
                if energy is not None:
                    valid_counts.append(j)
                    valid_energies.append(energy)
            all_conv_data.append({
                'counts': valid_counts, 
                'values': valid_energies, 
                'label': f'Trial {best_trial_idx} Top {i+1} Energy'
            })

    # --- 结果解析与保存 ---
    print(f"    - 正在保存最终结果到 {RESULT_DIR}...")
    
    # 查找最佳实验的最终计数，用于 MockResult (虽然主要用于展示分布)
    final_counts = {}
    if iteration_results:
        # 获取最佳实验最后一次迭代的 counts
        final_counts = iteration_results[-1].get("counts", {})
    total_shots = sum(final_counts.values()) if final_counts else args.shots
    
    # 将迭代结果转换为可序列化格式
    processed_iteration_results = []
    for iter_data in iteration_results:
        processed_top_results = [
            {"bitstring": r[0], "energy": float(r[1]), "count": r[2]}
            for r in iter_data.get("top_results", [])
        ]
        processed_iteration_results.append({
            "iteration": iter_data["iteration"],
            "cvar_energy": iter_data["cvar_energy"],
            "top_energies": [float(e) for e in iter_data.get("top_energies", [])],
            "top_results": processed_top_results,
            "total_counts": iter_data.get("total_counts", 0),
            "actual_shots": iter_data.get("actual_shots", 0)
        })
    
    # 为每个最优结果生成蛋白质结构和文件
    for idx, (bitstring, energy, count) in enumerate(final_top_results):
        print(f"\n    - 结果 {idx+1}/{args.max_results}: 能量 = {energy:.4f}, 出现次数 = {count}")
        
        # 使用全局定义的 MockResult 级解析蛋白质结构
        raw_res = MockResult({bitstring: 1.0}, energy, final_counts, total_shots)
        result = problem.interpret(raw_res)
        
        print(f"    - 转向序列: {result.turn_sequence}")
        
        # 1. 保存JSON格式的结果参数
        xyz_data = result.protein_shape_file_gen.get_xyz_data()
        result_data = {
            "result_index": idx + 1,
            "energy": energy,
            "turn_sequence": result.turn_sequence,
            "main_chain_sequence": args.main_chain,
            "shots_requested": args.shots,
            "bitstring": bitstring,
            "count": count,
            "max_results": args.max_results,
            "optimization_convergence": {
                "evaluation_counts": list(range(len(convergence_history))),
                "cvar_energy_values": convergence_history,
                "cumulative_shots": cumulative_shots_history,
                "iteration_shots": iteration_shots_history
            },
            "iteration_results": processed_iteration_results,
            "xyz_coordinates": [list(row) for row in xyz_data] if xyz_data is not None else []
        }
        json_path = os.path.join(RESULT_DIR, f'result_{idx+1}_energy_{energy:.4f}.json')
        with open(json_path, 'w') as f:
            json.dump(result_data, f, indent=2)
        print(f"    ✓ JSON文件已保存: {json_path}")
        
        # 2. 保存详细 PDB 文件
        if xyz_data is not None:
            pdb_path = os.path.join(RESULT_DIR, f'structure_{idx+1}_energy_{energy:.4f}.pdb')
            convert_xyz_to_detailed_pdb(xyz_data, pdb_path, f"Structure {idx+1}")
            print(f"    ✓ PDB文件已保存: {pdb_path}")
        
        # 3. 保存 3D 结构图
        try:
            fig_struct = result.get_figure(title=f"Result {idx+1} (E={energy:.4f})")
            png_path = os.path.join(RESULT_DIR, f"structure_{idx+1}_energy_{energy:.4f}.png")
            fig_struct.savefig(png_path)
            plt.close(fig_struct)
            print(f"    ✓ 结构图已保存: {png_path}")
        except Exception as e:
            print(f"    ⚠ 绘图失败: {e}")

    # 生成优化过程的收敛图
    plt.figure(figsize=(12, 8))
    for idx, data in enumerate(all_conv_data):
        label = data.get('label', f'Run {idx+1}')
        # CVaR Energy 使用虚线，其他使用实线
        if 'CVaR' in label:
            plt.plot(data['counts'], data['values'], marker='o', label=label, linewidth=2, linestyle='--')
        else:
            plt.plot(data['counts'], data['values'], marker='o', label=label, linewidth=2)
    plt.xlabel("Evaluation Counts")
    plt.ylabel("Energy")
    plt.title(f"VQE Sampler Convergence ({args.main_chain}) - CVaR (alpha={args.alpha})")
    plt.legend(loc='upper right')
    plt.grid(True, alpha=0.3)
    plt.savefig(os.path.join(RESULT_DIR, "vqe_optimization_summary.png"))
    plt.close()
    
    # 双y轴图：展示能量和每次迭代的shots数
    if iteration_shots_history and len(iteration_shots_history) > 0:
        fig, ax1 = plt.subplots(figsize=(12, 8))
        
        # 定义颜色映射，为每个运行结果使用相同颜色的不同样式
        colors = plt.cm.tab10(np.linspace(0, 1, len(all_conv_data)))
        
        # 绘制能量曲线
        energy_lines = []
        labels = []
        for idx, data in enumerate(all_conv_data):
            color = colors[idx]
            label = data.get('label', f'Run {idx+1}')
            # CVaR Energy 使用虚线，其他使用实线
            if 'CVaR' in label:
                values = np.array(data['values'])
                line, = ax1.plot(data['counts'], values, marker='o', label=label, 
                         linewidth=3, color=color, linestyle='--')
                # 绘制误差带
                stds = np.array(data.get('stds', []))
                if len(stds) == len(values):
                     ax1.fill_between(data['counts'], values - stds, values + stds, 
                                      color=color, alpha=0.2, label=f'{label} Std Range')
            else:
                line, = ax1.plot(data['counts'], data['values'], marker='o', label=label, 
                         linewidth=3, color=color)
            energy_lines.append(line)
            labels.append(label)
        ax1.set_xlabel('Evaluation Counts')
        ax1.set_ylabel('Energy', color='black')
        ax1.tick_params(axis='y', labelcolor='black')
        ax1.grid(True, alpha=0.3)
        
        # 创建第二个y轴用于shots
        shots_bars = []
        ax2 = ax1.twinx()
        # 使用每次迭代的实际shots数，而不是累计shots
        bars = ax2.bar(range(len(iteration_shots_history)), iteration_shots_history, alpha=0.3, width=0.5, 
               color='gray', edgecolor='gray', linewidth=0.5, 
               label='Shots per Iteration')
        ax2.set_ylabel('Shots per Iteration', color='black')
        ax2.tick_params(axis='y', labelcolor='black')
        
        # 只使用能量曲线的图例，避免重复
        ax1.legend(energy_lines, labels, loc='upper right')
        
        plt.title(f"VQE Sampler Convergence with Shots per Iteration ({args.main_chain}) - CVaR (alpha={args.alpha})")
        fig.tight_layout()
        plt.savefig(os.path.join(RESULT_DIR, "vqe_sampler_optimization_with_shots.png"))
        plt.close()
        print(f"✓ 带shots信息的VQE采样器优化图已保存为 {os.path.join(RESULT_DIR, 'vqe_sampler_optimization_with_shots.png')}")
    
    print(f"\n✓ 蛋白质折叠计算完成！")
    
    # 导出最优参数向量（供 Warm-Start 使用）
    try:
        print(f"\n正在导出最优参数向量...")
        # 从最佳实验的优化结果中获取最优参数
        if res is not None and hasattr(res, 'x'):
            best_params = res.x
            best_params_file = os.path.join(RESULT_DIR, "best_params.json")
            with open(best_params_file, 'w') as f:
                json.dump({
                    "initial_point": best_params.tolist(),
                    "energy": best_energy,
                    "num_parameters": len(best_params),
                    "best_trial": best_trial_idx
                }, f, indent=2)
            print(f"✓ 最优参数已保存到: {best_params_file}")
            print(f"  - 参数维度: {len(best_params)}")
            print(f"  - 最优能量: {best_energy:.4f}")
            print(f"  - 最佳实验: Trial {best_trial_idx}")
            print(f"  提示: 可使用 --initial_params {best_params_file} 进行 Warm-Start")
        else:
            print(f"⚠ 无法找到最优参数，跳过导出")
    except Exception as e:
        print(f"⚠ 导出最优参数时出错: {e}")
    
    print(f"\n🎉 蛋白质折叠计算任务完成！所有结果保存在: {RESULT_DIR}")
    try:
        metrics_path = os.path.join(RESULT_DIR, "metrics.json")
        total_shots = 0
        total_iters = 0
        for d in all_conv_data:
            if isinstance(d, dict) and 'iteration_shots' in d and isinstance(d['iteration_shots'], list):
                try:
                    total_shots += sum(int(x) for x in d['iteration_shots'])
                    total_iters += len(d['iteration_shots'])
                except:
                    pass
        try:
            ops = transpiled_circuit.count_ops()
            twoq = int(ops.get('cx', 0)) + int(ops.get('cz', 0)) + int(ops.get('swap', 0))
            transpile_metrics = {
                "logical_qubits": int(ansatz.num_qubits),
                "physical_qubits": int(transpiled_circuit.num_qubits),
                "depth": int(transpiled_circuit.depth() or 0),
                "two_qubit_gates": twoq,
                "ops": {k: int(v) for k, v in ops.items()}
            }
        except Exception:
            transpile_metrics = {}
        try:
            per_run = []
            for d in all_conv_data:
                values = d.get('values', [])
                counts = d.get('counts', [])
                iters = len(values)
                start = float(values[0]) if iters > 0 else None
                end = float(values[-1]) if iters > 0 else None
                min_e = float(min(values)) if iters > 0 else None
                avg_drop = float((values[0] - values[-1]) / max(1, iters - 1)) if iters > 1 else 0.0
                per_run.append({
                    "iterations": iters,
                    "energy_start": start,
                    "energy_end": end,
                    "min_energy": min_e,
                    "avg_decrease_per_iter": avg_drop
                })
            convergence_metrics = {
                "iterations_total": int(sum(len(d.get('values', [])) for d in all_conv_data)),
                "best_energy": float(best_energy),
                "per_run": per_run
            }
        except Exception:
            convergence_metrics = {}
        metrics = {
            "backend": args.backend,
            "shots_requested": int(args.shots),
            "shots_actual_total": int(total_shots),
            "iteration_count": int(total_iters),
            "outcome_summary": f"min_energy={float(best_energy):.6f}",
            "qubits_used": int(ansatz.num_qubits),
            "transpile_metrics": transpile_metrics,
            "convergence_metrics": convergence_metrics
        }
        with open(metrics_path, "w", encoding="utf-8") as f:
            json.dump(metrics, f, indent=2)
        print(f"✓ 指标摘要已保存到: {metrics_path}")
    except Exception:
        pass

if __name__ == "__main__":
    main()
